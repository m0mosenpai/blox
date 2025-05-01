import os
import sys
import time
import uuid
import grpc
import argparse
import logging
import heapq # For priority queue
import threading
import copy # For deep copying state if needed by policy
from collections import deque
from concurrent import futures
from typing import Dict, List, Set, Tuple, Any, Optional

from .cluster_state import ClusterState
from .job_state import JobState # Adapt this if needed, or use simple dicts
import deployment.grpc_server_rm as rm_serve # Use updated server start
import deployment.grpc_client_nm as nm_client # Use updated client

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from schedulers import make_policy # Scheduler factory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BloxManager(object):
    """
    Central scheduler managing nodes, jobs, and dispatching using a pluggable policy.
    """
    def __init__(self, args):
        self.args = args
        self.scheduler_port = args.scheduler_port # Port this RMServer listens on
        self.node_manager_port = args.node_manager_port # Default port NMs listen on

        self.cluster_state = ClusterState(args) # Tracks nodes, resources
        self.job_state = JobState(args)       # Tracks active/pending jobs

        # Communication Client (For talking to Node Managers)
        self.comm_node_manager = nm_client.NodeManagerComm()

        # Inference Request Queue (Priority Queue)
        self.pending_inference_requests: List[Tuple[int, float, Dict[str, Any]]] = []
        self.request_lock = threading.Lock() # Protect queue and worker_info access

        # Worker Info Store
        # Key: Node IP Addr (str), Value: Dict with registration info & status
        self.worker_info: Dict[str, Dict[str, Any]] = {}

        # Scheduling Policy
        # Dynamically load policy based on args.policy
        policy_args = {k: getattr(args, k) for k in ["scheduling_interval"] if hasattr(args, k)} # Add policy-specific args
        try:
             self.scheduler_policy = make_policy(
                  args.policy,
                  cluster_state=self.cluster_state, # Policy might need access
                  job_state=self.job_state,         # Policy might need access
                  **policy_args
             )
             logger.info(f"Initialized scheduler policy: {args.policy}")
        except KeyError:
             logger.error(f"Unknown scheduler policy: {args.policy}. Exiting.")
             raise ValueError(f"Unknown scheduler policy: {args.policy}")


        # 6. Control flags for the main loop
        self.scheduling_interval_sec = args.scheduling_interval
        self.scheduler_run_signal = threading.Event() # Used to signal shutdown
        self.current_simulation_time = 0 # Used by scheduler policy?

    # --- Methods called by RMServer ---

    def register_worker(self, ipaddr, interface, num_gpus, nm_port, available_kv_memory, supported_models):
        """Handles worker registration, updating cluster state and worker info."""
        logger.info(f"Registering worker {ipaddr}:{nm_port} (GPUs: {num_gpus}, KV Mem: {available_kv_memory}GB)")
        with self.request_lock:
             self.worker_info[ipaddr] = {
                 "ipaddr": ipaddr,
                 "port": nm_port,
                 "interface": interface,
                 "num_gpus": num_gpus,
                 "kv_memory_gb": available_kv_memory,
                 "models": list(supported_models),
                 "last_seen": time.time(),
                 "status": "active", # e.g., active, inactive, migrating_from, migrating_to
                 # Add other state if needed (e.g., local queue depth if reported)
             }
        # Update ClusterState representation as well
        self.cluster_state.add_node(ipaddr, num_gpus) # Adapt ClusterState API
        return True

    def add_inference_request(self, job_details: dict) -> Tuple[bool, Optional[str], str]:
        """
        Adds a new inference request from frontend to the pending queue.
        Assigns a Blox Job ID.
        Returns: (accepted: bool, job_id: Optional[str], message: str)
        """
        priority_str = job_details.get("priority", "normal")
        priority_map = {"high": 0, "normal": 1, "low": 2}
        priority_val = priority_map.get(priority_str.lower(), 1)
        submit_time = job_details.get("submit_time", time.time())

        # Assign a unique Blox Job ID (use string representation)
        # Using UUID ensures uniqueness even if manager restarts (if state isn't persisted)
        blox_job_id = str(uuid.uuid4())
        job_details["job_id"] = blox_job_id # Add to the dict

        # TODO: Check against max queue length if desired
        max_queue_len = getattr(self.args, "max_global_queue_size", 1000)
        with self.request_lock:
            if len(self.pending_inference_requests) >= max_queue_len:
                 logger.warning(f"Global inference queue full ({max_queue_len}). Rejecting request {job_details['request_id']}.")
                 return False, None, "Rejected: Global queue full"

            # Use a tuple for heapq: (priority, submission_time, job_details)
            request_entry = (priority_val, submit_time, job_details)
            heapq.heappush(self.pending_inference_requests, request_entry)
            logger.info(f"Queued inference request {job_details['request_id']} as JobID: {blox_job_id} with priority {priority_val}.")

            # Update JobState as well
            self.job_state.add_job(blox_job_id, job_details) # Adapt JobState API

        return True, blox_job_id, "Request queued"


    def handle_job_completion(self, job_id_str: str, success: bool):
        """Updates job status upon completion notification from Node Manager."""
        logger.debug(f"Handling completion for job {job_id_str}, Success: {success}")
        status = "finished" if success else "failed"
        # Update JobState
        self.job_state.update_job_status(job_id_str, status) # Adapt JobState API
        # Update ClusterState (release resources associated with the job)
        node_ip = self.job_state.get_job_location(job_id_str) # Need method in JobState
        if node_ip:
             # Need details on which GPU(s) were used if tracking that level
             self.cluster_state.release_resources(node_ip, job_id_str) # Adapt ClusterState API
             logger.info(f"Job {job_id_str} completed on node {node_ip}. Status: {status}.")
        else:
             logger.warning(f"Could not find node location for completed job {job_id_str}.")


    def handle_migration_completion(self, job_id: str, from_node: str, to_node: str):
        """Updates job location after successful migration notification."""
        logger.info(f"Handling migration completion for job {job_id}: {from_node} -> {to_node}")
        # Update JobState with the new location
        self.job_state.update_job_location(job_id, to_node) # Adapt JobState API
        # Update ClusterState if needed (e.g., mark nodes as non-migrating)
        self.cluster_state.update_node_status(from_node, "active") # Adapt API
        self.cluster_state.update_node_status(to_node, "active")   # Adapt API
        with self.request_lock:
             if from_node in self.worker_info: self.worker_info[from_node]["status"] = "active"
             if to_node in self.worker_info: self.worker_info[to_node]["status"] = "active"


    # --- Main Scheduling Loop ---

    def _run_schedule_and_dispatch(self):
        """Performs one round of scheduling and dispatches decisions."""
        self.current_simulation_time += self.scheduling_interval_sec # Advance time if simulating

        # 1. Get current state (make copies if policy modifies them)
        # These need to be accessible/copyable representations of the state
        active_jobs_snapshot = self.job_state.get_active_jobs() # Adapt API
        pending_jobs_snapshot = self.get_pending_requests_snapshot() # Get from queue
        cluster_snapshot = self.cluster_state.get_cluster_view() # Adapt API
        worker_snapshot = copy.deepcopy(self.worker_info) # Need deep copy?

        # 2. Call the scheduling policy
        # Policy should return dispatch decisions and migration decisions
        # dispatch_decisions: Dict[job_id, node_ip]
        # migration_decisions: List[Tuple[job_id, from_node_ip, to_node_ip]]
        dispatch_decisions, migration_decisions = {}, []
        try:
             # Policy needs access to pending jobs too
             dispatch_decisions, migration_decisions = self.scheduler_policy.schedule(
                  active_jobs=active_jobs_snapshot,
                  pending_jobs=pending_jobs_snapshot, # Pass pending queue snapshot
                  nodes=worker_snapshot, # Pass worker info/status
                  cluster_view=cluster_snapshot, # Pass resource view
                  current_time=self.current_simulation_time # Pass time context
             )
        except Exception as e:
             logger.error(f"Error during scheduling policy execution: {e}", exc_info=True)
             return # Skip dispatch if policy fails

        if dispatch_decisions: logger.debug(f"Dispatch decisions: {dispatch_decisions}")
        if migration_decisions: logger.debug(f"Migration decisions: {migration_decisions}")

        # 3. Execute Migration Decisions
        if migration_decisions:
             self._execute_migrations(migration_decisions, worker_snapshot)

        # 4. Execute Dispatch Decisions (Launch new jobs)
        if dispatch_decisions:
             self._execute_dispatch(dispatch_decisions, worker_snapshot)


    def get_pending_requests_snapshot(self) -> List[Dict[str, Any]]:
         """Returns a list of pending job details for the scheduler."""
         with self.request_lock:
              # Return only the job_details dict part from the priority queue entries
              # Sorting by priority/time might be useful for some policies
              sorted_requests = sorted(self.pending_inference_requests, key=lambda x: (x[0], x[1]))
              return [details for _, _, details in sorted_requests]


    def _execute_migrations(self, migrations: List[Tuple[str, str, str]], workers: Dict[str, Any]):
         """Initiates migration RPC calls based on policy decisions."""
         with self.request_lock: # Protect worker_info status changes
              for job_id, from_node_ip, to_node_ip in migrations:
                   if from_node_ip not in workers or to_node_ip not in workers:
                        logger.warning(f"Cannot migrate job {job_id}: Node(s) {from_node_ip} or {to_node_ip} not found.")
                        continue

                   # Check if nodes are in a state suitable for migration
                   from_node_status = workers[from_node_ip].get("status", "active")
                   to_node_status = workers[to_node_ip].get("status", "active")
                   if from_node_status != "active" or to_node_status != "active":
                        logger.warning(f"Skipping migration for job {job_id}: Node status conflict ({from_node_ip}:{from_node_status}, {to_node_ip}:{to_node_status})")
                        continue

                   # Get NMServer addresses
                   from_node_port = workers[from_node_ip].get("port", self.node_manager_port)
                   to_node_port = workers[to_node_ip].get("port", self.node_manager_port)
                   from_nm_address = f"{from_node_ip}:{from_node_port}"
                   to_nm_address = f"{to_node_ip}:{to_node_port}"

                   logger.info(f"Initiating migration of job {job_id} from {from_nm_address} to {to_nm_address}")

                   # Mark nodes as migrating in our state immediately
                   self.worker_info[from_node_ip]["status"] = "migrating_from"
                   self.worker_info[to_node_ip]["status"] = "migrating_to"
                   self.cluster_state.update_node_status(from_node_ip, "migrating") # Adapt API
                   self.cluster_state.update_node_status(to_node_ip, "migrating") # Adapt API

                   # Send the InitiateMigration RPC to the source node
                   success = self.comm_node_manager.initiate_migration(
                        target_nm_address=from_nm_address, # Send to the source node
                        job_id=job_id,
                        target_node_ip=to_node_ip, # Tell source where to send state
                        target_node_nm_address=to_nm_address
                   )

                   if not success:
                        logger.error(f"Failed to send InitiateMigration RPC for job {job_id} to {from_nm_address}. Reverting status.")
                        # Revert status if RPC call itself failed
                        self.worker_info[from_node_ip]["status"] = "active"
                        self.worker_info[to_node_ip]["status"] = "active"
                        self.cluster_state.update_node_status(from_node_ip, "active")
                        self.cluster_state.update_node_status(to_node_ip, "active")
                   # Else: Migration is initiated, wait for MigrationComplete callback


    def _execute_dispatch(self, dispatches: Dict[str, str], workers: Dict[str, Any]):
         """Launches newly scheduled jobs via RPC calls."""
         jobs_launched_this_round = set()
         with self.request_lock: # Protect access to pending queue
              # Use a temporary list to hold requests we attempt to dispatch
              requests_to_remove = []
              original_pending_count = len(self.pending_inference_requests)

              # Iterate through pending requests (highest priority first)
              temp_pending = sorted(self.pending_inference_requests, key=lambda x: (x[0], x[1]))
              processed_indices = set()

              for job_id, target_node_ip in dispatches.items():
                   # Find the corresponding job details in the pending queue
                   found = False
                   for i, (prio, ts, details) in enumerate(temp_pending):
                        if i in processed_indices: continue # Skip if already processed
                        if details["job_id"] == job_id:
                             found = True
                             job_details = details
                             queue_entry = (prio, ts, details)
                             queue_index_in_original = -1
                             # Find index in original heapq list (less efficient but needed for removal)
                             try:
                                  queue_index_in_original = self.pending_inference_requests.index(queue_entry)
                             except ValueError:
                                  logger.error(f"Consistency error: Job {job_id} from dispatch decision not found in live pending queue.")
                                  continue # Skip this dispatch

                             processed_indices.add(i) # Mark as processed in temp list

                             if target_node_ip not in workers:
                                  logger.warning(f"Cannot dispatch job {job_id}: Target node {target_node_ip} not found.")
                                  continue # Try next dispatch

                             node_port = workers[target_node_ip].get("port", self.node_manager_port)
                             target_nm_address = f"{target_node_ip}:{node_port}"

                             # Check node status before dispatching
                             node_status = workers[target_node_ip].get("status", "active")
                             if node_status != "active":
                                 logger.warning(f"Skipping dispatch of job {job_id} to node {target_node_ip}: Status is '{node_status}'")
                                 continue # Skip dispatch to non-active node


                             logger.info(f"Dispatching job {job_id} (ReqID: {job_details['request_id']}) to {target_nm_address}")

                             # Send LaunchJob RPC
                             accepted = self.comm_node_manager.launch_job(
                                  target_nm_address=target_nm_address,
                                  job_id=job_id,
                                  request_id=job_details["request_id"],
                                  prompt=job_details["prompt"],
                                  sampling_params=job_details["sampling_params"],
                                  priority=job_details.get("priority", "normal")
                             )

                             if accepted:
                                  logger.info(f"Node {target_node_ip} ACKed job {job_id}. Removing from pending queue.")
                                  # Mark for removal from the actual queue
                                  requests_to_remove.append(queue_entry)
                                  jobs_launched_this_round.add(job_id)
                                  # Update JobState and ClusterState
                                  self.job_state.update_job_status(job_id, "running", location=target_node_ip) # Adapt API
                                  self.cluster_state.allocate_resources(target_node_ip, job_id) # Adapt API
                             else:
                                  logger.warning(f"Node {target_node_ip} NAKed job {job_id}. Leaving in pending queue.")
                                  # Job stays in the pending queue for the next round

                             break # Move to the next dispatch decision
                   if not found:
                        logger.warning(f"Job {job_id} in dispatch decisions not found in pending list snapshot.")


              # Remove successfully dispatched jobs from the actual pending queue
              if requests_to_remove:
                   # Efficient removal from list/heapq is tricky. Rebuild if needed.
                   current_pending_set = set(self.pending_inference_requests)
                   removed_set = set(requests_to_remove)
                   self.pending_inference_requests = list(current_pending_set - removed_set)
                   heapq.heapify(self.pending_inference_requests) # Restore heap property
                   logger.debug(f"Removed {len(requests_to_remove)} jobs from pending queue. New size: {len(self.pending_inference_requests)}.")
                   assert len(self.pending_inference_requests) == original_pending_count - len(requests_to_remove)



    def _update_cluster_health(self):
        """Checks for dead nodes based on last_seen time."""
        cutoff_time = time.time() - getattr(self.args, "node_timeout_sec", 60) # Example timeout
        dead_nodes = []
        with self.request_lock: # Protect worker_info access
             for ip, info in self.worker_info.items():
                  # Only consider active nodes for timeout check
                  if info.get("status", "active") == "active" and info.get("last_seen", 0) < cutoff_time:
                       dead_nodes.append(ip)
                       info["status"] = "inactive" # Mark as inactive

        if dead_nodes:
            logger.warning(f"Detected inactive/timed-out nodes: {dead_nodes}")
            # Update cluster_state accordingly
            for node_ip in dead_nodes:
                 self.cluster_state.update_node_status(node_ip, "inactive") # Adapt API
                 # TODO: Handle jobs that were running on the dead node (mark as failed?)
                 failed_jobs = self.job_state.get_jobs_on_node(node_ip) # Adapt API
                 for jid in failed_jobs:
                      logger.warning(f"Marking job {jid} as failed due to node {node_ip} inactivity.")
                      self.handle_job_completion(jid, False) # Mark as failed

    def scheduler_loop(self):
        """The main periodic scheduling loop."""
        logger.info("BloxManager scheduler loop started.")
        while not self.scheduler_run_signal.is_set():
            start_time = time.time()
            logger.debug("--- Scheduler Tick ---")
            try:
                # 1. Update cluster health (check for dead nodes)
                self._update_cluster_health()

                # 2. Run scheduling policy and dispatch decisions
                self._run_schedule_and_dispatch()

            except Exception as e:
                logger.error(f"Error in scheduler loop iteration: {e}", exc_info=True)

            # 3. Wait until the next interval
            elapsed_time = time.time() - start_time
            wait_time = max(0.01, self.scheduling_interval_sec - elapsed_time) # Ensure minimum wait
            if elapsed_time > self.scheduling_interval_sec:
                 logger.warning(f"Scheduler loop iteration took longer ({elapsed_time:.3f}s) than interval ({self.scheduling_interval_sec}s).")

            # Wait for the calculated time or until shutdown signal
            self.scheduler_run_signal.wait(timeout=wait_time)

        logger.info("BloxManager scheduler loop stopped.")

    def _run_schedule_and_dispatch(self):
        """Performs one round of scheduling and dispatches decisions."""
        self.current_simulation_time += self.scheduling_interval_sec

        # 1. Get current state snapshots
        active_jobs_snapshot = self.job_state.get_active_jobs()
        pending_jobs_snapshot = self.get_pending_requests_snapshot()
        cluster_snapshot = self.cluster_state.get_cluster_view()
        # Use a read lock or be careful if iterating worker_info while RMServer might modify it
        with self.request_lock:
            worker_snapshot = copy.deepcopy(self.worker_info)

        # 2. Call the scheduling policy
        dispatch_decisions, migration_decisions, jobs_to_cancel = {}, [], set() # Initialize cancellation set
        try:
            dispatch_decisions, migration_decisions, jobs_to_cancel = self.scheduler_policy.schedule(
                active_jobs=active_jobs_snapshot,
                pending_jobs=pending_jobs_snapshot,
                nodes=worker_snapshot,
                cluster_view=cluster_snapshot,
                current_time=self.current_simulation_time
            )
        except Exception as e:
            logger.error(f"Error during scheduling policy execution: {e}", exc_info=True)
            return # Skip dispatch/migration/cancellation if policy fails

        # --- Log Decisions ---
        if dispatch_decisions: logger.debug(f"Dispatch decisions: {dispatch_decisions}")
        if migration_decisions: logger.debug(f"Migration decisions: {migration_decisions}")
        if jobs_to_cancel: logger.debug(f"Cancellation decisions: {jobs_to_cancel}")


        # 3. Execute Cancellation Decisions FIRST (to potentially free resources)
        if jobs_to_cancel:
             self._execute_cancellations(jobs_to_cancel, active_jobs_snapshot, worker_snapshot)


        # 4. Execute Migration Decisions
        if migration_decisions:
            # Filter out migrations for jobs that were just cancelled
            valid_migrations = [(jid, fip, tip) for jid, fip, tip in migration_decisions if jid not in jobs_to_cancel]
            if valid_migrations:
                 self._execute_migrations(valid_migrations, worker_snapshot)


        # 5. Execute Dispatch Decisions (Launch new jobs)
        if dispatch_decisions:
            # Filter out dispatches for jobs whose destination node might now be involved in migration
            # (A simpler approach is to let the NM NAK the job if it's busy migrating)
            self._execute_dispatch(dispatch_decisions, worker_snapshot)


    def _execute_cancellations(self, jobs_to_cancel: Set[str], active_jobs: Dict[str, Any], workers: Dict[str, Any]):
         """Sends termination requests for jobs identified by the scheduler."""
         logger.info(f"Executing cancellations for {len(jobs_to_cancel)} jobs: {jobs_to_cancel}")
         for job_id in jobs_to_cancel:
              if job_id not in active_jobs:
                   logger.warning(f"Cannot cancel job {job_id}: Not found in active jobs snapshot.")
                   continue

              node_ip = active_jobs[job_id].get("node_ip")
              if not node_ip or node_ip not in workers:
                   logger.warning(f"Cannot cancel job {job_id}: Node IP '{node_ip}' not found or invalid.")
                   continue

              node_port = workers[node_ip].get("port", self.node_manager_port)
              target_nm_address = f"{node_ip}:{node_port}"

              logger.info(f"Sending NotifyTerminate for job {job_id} to {target_nm_address}")
              success = self.comm_node_manager.notify_terminate(
                   target_nm_address=target_nm_address,
                   job_id=job_id
              )
              if success:
                   logger.info(f"NotifyTerminate for job {job_id} acknowledged by {target_nm_address}.")
                   # Optionally update job state immediately to 'cancelling',
                   # but rely on JobCompleted(success=False) callback for final state.
                   # self.job_state.update_job_status(job_id, "cancelling")
              else:
                   logger.warning(f"NotifyTerminate for job {job_id} failed or was not acknowledged by {target_nm_address}.")



    def run(self):
        """Starts the RMServer and the main scheduler loop."""
        # Start the RMServer gRPC service
        self.grpc_server = rm_serve.start_server(self, self.scheduler_port)

        # Start the scheduler loop in a separate thread
        self.scheduler_thread = threading.Thread(target=self.scheduler_loop, daemon=True)
        self.scheduler_thread.start()

        logger.info("BloxManager running. Press Ctrl+C to exit.")
        # Keep the main thread alive (e.g., wait for server termination)
        try:
            # self.grpc_server.wait_for_termination() # This blocks until server stops
            # Or simply sleep while threads run
            while True:
                 time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("Ctrl+C received, initiating BloxManager shutdown...")
        finally:
            # Graceful shutdown sequence
            logger.info("Signaling scheduler loop to stop...")
            self.scheduler_run_signal.set()
            self.scheduler_thread.join(timeout=max(1, self.scheduling_interval_sec * 2)) # Wait for loop exit
            if self.scheduler_thread.is_alive():
                 logger.warning("Scheduler thread did not exit gracefully.")

            logger.info("Stopping RMServer gRPC service...")
            self.grpc_server.stop(grace=1).wait() # Wait for server stop
            logger.info("RMServer stopped.")
            logger.info("BloxManager shutdown complete.")


# --- Argument Parsing for BloxManager ---
def parse_blox_manager_args(parser: argparse.ArgumentParser):
    parser.add_argument("--scheduler-port", type=int, default=50051, help="Port for the BloxManager RMServer")
    parser.add_argument("--node-manager-port", type=int, default=50051, help="Default port Node Managers listen on")
    parser.add_argument("--scheduling-interval", type=float, default=0.1, help="Scheduling loop interval in seconds")
    parser.add_argument("--node-timeout-sec", type=int, default=60, help="Seconds of inactivity before marking a node dead")
    parser.add_argument("--max-global-queue-size", type=int, default=1000, help="Maximum pending inference requests globally")

    # Add policy choice (example based on "another output")
    parser.add_argument("--policy", type=str, default="fifo", help="Global scheduling policy (e.g., fifo, llumnix)") # Add choices as policies are implemented
    parser.add_argument("--scheduler-name",   type=str, default="fifo",       help="Name of the scheduler (for logs/metrics)")
    parser.add_argument("--placement-name",  type=str, default="Place",      help="Placement policy name (unused for inference)")
    parser.add_argument("--acceptance-policy",type=str, default="AcceptAll",help="Acceptance policy name (unused for inference)")
    parser.add_argument("--exp-prefix",      type=str, default="exp",        help="Prefix for output logs and metrics")
    parser.add_argument("--load",            type=int, default=10,           help="Load factor (jobs per hour) for simulation")
    parser.add_argument("--round-duration",  type=int, default=300,         help="Round duration in seconds for metrics collection")
    parser.add_argument("--start-id-track",  type=int, default=0,           help="First job ID to track for completion metrics")
    parser.add_argument("--stop-id-track",   type=int, default=0,           help="Last  job ID to track for completion metrics")

    args = parser.parse_args()
    return args

# --- Main Execution ---
if __name__ == "__main__":
    args = parse_blox_manager_args(
        argparse.ArgumentParser(description="Blox Central Scheduler for Inference")
    )
    try:
        manager = BloxManager(args)
        manager.run()
    except ValueError as e:
         logger.error(f"Initialization failed: {e}")
    except Exception as e:
         logger.error(f"BloxManager encountered critical error: {e}", exc_info=True)