# schedulers/fifo_scheduler.py

import logging
import random # For potential tie-breaking
from typing import Dict, List, Tuple, Any, Set
from .scheduler_policy import BaseSchedulerPolicy

logger = logging.getLogger(__name__)

class FIFOScheduler(BaseSchedulerPolicy):
    """
    Basic FIFO scheduler. Assigns pending jobs (sorted by submit time)
    to the currently least loaded available node, considering assignments
    made within the current scheduling round.
    Does not migrate or cancel jobs.
    """

    def _get_node_capacity(self, node_info: Dict[str, Any]) -> int:
        """Estimate capacity based on scaled batch size."""
        # Assume base batch size was passed via args and scales with node GPUs
        # Need access to the base batch size arg here. Hacky: Assume it's on self if passed via kwargs.
        # Ensure vllm_max_batch_size is passed to make_policy!
        base_batch_size = getattr(self, 'vllm_max_batch_size', 8)
        num_gpus = node_info.get("num_gpus", 1)
        scaled_capacity = base_batch_size * num_gpus
        return max(1, scaled_capacity) # Ensure capacity is at least 1

    def schedule(self,
                 active_jobs: Dict[str, Any],
                 pending_jobs: List[Dict[str, Any]],
                 nodes: Dict[str, Any],
                 cluster_view: Any, # Not used by basic FIFO
                 current_time: float
                ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]], Set[str]]:

        dispatch_decisions = {}
        migration_decisions = [] # FIFO doesn't migrate
        jobs_to_cancel = set()   # FIFO doesn't cancel

        # Filter nodes that are ready to accept jobs
        active_nodes_map = {ip: info for ip, info in nodes.items() if info.get("status") == "active"}
        if not active_nodes_map:
            logger.debug("FIFO: No active nodes available for scheduling.")
            return {}, [], set()

        # Calculate current actual load based on active jobs snapshot
        initial_node_load = {ip: 0 for ip in active_nodes_map}
        for job_detail in active_jobs.values():
             node_ip = job_detail.get("node_ip")
             if node_ip in initial_node_load:
                  initial_node_load[node_ip] += 1

        # <<< FIX: Track assignments made *within this round* >>>
        assignments_this_round = {ip: 0 for ip in active_nodes_map}

        # Assign pending jobs (in order received)
        pending_jobs_sorted = sorted(pending_jobs, key=lambda j: j.get("submit_time", 0))

        dispatched_count = 0
        for job_details in pending_jobs_sorted:
            job_id = job_details["job_id"]

            # Find the node with the lowest *combined* load (initial + this round's assignments)
            best_target_node = None
            min_combined_load = float('inf')

            # Iterate through available nodes to find the best target
            # Use sorted list for deterministic tie-breaking if needed, or shuffle for random
            node_candidates = sorted(list(active_nodes_map.keys()))
            # random.shuffle(node_candidates) # Optional: Randomize tie-breaking

            for node_ip in node_candidates:
                 current_assigned_load = initial_node_load.get(node_ip, 0) + assignments_this_round.get(node_ip, 0)

                 # Check capacity before considering it best
                 node_capacity = self._get_node_capacity(active_nodes_map[node_ip])
                 if current_assigned_load < node_capacity:
                      # This node has capacity. Is it the least loaded so far?
                      if current_assigned_load < min_combined_load:
                           min_combined_load = current_assigned_load
                           best_target_node = node_ip
                           # Continue checking other nodes in case one has the same minimum load

                 # If multiple nodes have the same minimum load, the first one encountered in the
                 # (potentially sorted) list `node_candidates` will be chosen.

            # If a suitable node was found
            if best_target_node:
                 node_capacity = self._get_node_capacity(active_nodes_map[best_target_node])
                 current_assigned_load = initial_node_load.get(best_target_node, 0) + assignments_this_round.get(best_target_node, 0)

                 # Final check: ensure capacity not exceeded *before* assigning
                 if current_assigned_load < node_capacity:
                      dispatch_decisions[job_id] = best_target_node
                      assignments_this_round[best_target_node] += 1 # Increment planned load *after* assigning
                      logger.debug(f"FIFO: Assigning pending job {job_id} to node {best_target_node} (Load: {current_assigned_load+1}/{node_capacity})")
                      dispatched_count += 1
                 else:
                      # This case should ideally not be hit if min_combined_load logic is correct, but as safety break
                      logger.warning(f"FIFO: Best node {best_target_node} became full ({current_assigned_load}/{node_capacity}) during dispatch round. Stopping dispatch.")
                      break
            else:
                 # No node found with capacity for this job
                 logger.debug(f"FIFO: No node found with capacity for job {job_id}. Stopping dispatch.")
                 break # Stop trying to dispatch further jobs in this round

        # logger.info(f"FIFO: Dispatched {dispatched_count} jobs this round.") # Less verbose logging
        return dispatch_decisions, migration_decisions, jobs_to_cancel