# inference_simulator.py

import argparse
import heapq
import json
import logging
import math
import random
import threading
import time
import uuid
from collections import deque
from typing import Dict, List, Tuple, Any, Optional, Set

from schedulers import make_policy, _POLICY_REGISTRY
from inference_metrics import summarize_inference_metrics
from schedulers.llumnix_scheduler import NORMAL_PRIORITY, PRIORITY_MAP

# --- Simulation Configuration ---
DEFAULT_SCHEDULING_INTERVAL = 0.1 # seconds, how often the global scheduler runs
DEFAULT_NODE_UPDATE_INTERVAL = 0.01 # seconds, how often node state/vLLM updates

# --- Logging Setup ---
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("InferenceSimulator")

# =============================================================================
# 1. Workload Generation
# =============================================================================

class InferenceWorkloadGenerator:
    """Generates inference request arrivals based on trace or synthetic model."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workload_type = config.get("type", "synthetic") # "synthetic" or "trace"
        self.request_queue: List[Dict[str, Any]] = []
        self.last_arrival_time = 0.0
        self.request_counter = 0

        if self.workload_type == "trace":
            self._load_trace(config["trace_file"])
        elif self.workload_type == "synthetic":
            self.arrival_rate = config.get("arrival_rate", 10.0) # reqs/sec
            self.avg_prompt_len = config.get("avg_prompt_len", 512)
            self.avg_output_len = config.get("avg_output_len", 256)
            self.priority_distribution = config.get("priority_distribution", {"high": 0.1, "normal": 0.8, "low": 0.1})
            # Simple exponential distribution for lengths for now
        else:
            raise ValueError(f"Unknown workload type: {self.workload_type}")

        logger.info(f"Workload generator initialized (Type: {self.workload_type})")

    def _load_trace(self, trace_file: str):
        """Loads requests from a trace file (e.g., CSV or JSONL)."""
        logger.info(f"Loading workload trace from: {trace_file}")
        try:
            with open(trace_file, 'r') as f:
                # Example: Assuming JSONL format with arrival_time, prompt_len, output_len, priority
                for line in f:
                    try:
                        data = json.loads(line)
                        request = {
                            "request_id": data.get("request_id", f"trace_{self.request_counter}"),
                            "arrival_time": float(data["arrival_time"]), # Must be absolute time
                            "prompt_len": int(data.get("prompt_len", self.config.get("avg_prompt_len", 128))),
                            "output_len": int(data.get("output_len", self.config.get("avg_output_len", 128))),
                            "priority": data.get("priority", "normal"),
                            # Add other fields from trace if needed
                        }
                        self.request_queue.append(request)
                        self.request_counter += 1
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        logger.warning(f"Skipping invalid trace line: {line.strip()} - Error: {e}")
            # Sort by arrival time REQUIRED
            self.request_queue.sort(key=lambda r: r["arrival_time"])
            logger.info(f"Loaded {len(self.request_queue)} requests from trace.")
        except FileNotFoundError:
            logger.error(f"Trace file not found: {trace_file}")
            raise
        except Exception as e:
            logger.error(f"Error loading trace file {trace_file}: {e}")
            raise

    def _generate_synthetic_request(self) -> Dict[str, Any]:
        """Generates a single synthetic request."""
        # Exponential inter-arrival time
        inter_arrival_time = random.expovariate(self.arrival_rate)
        arrival_time = self.last_arrival_time + inter_arrival_time
        self.last_arrival_time = arrival_time

        prompt_len = max(1, int(random.expovariate(1.0 / self.avg_prompt_len)))
        output_len = max(1, int(random.expovariate(1.0 / self.avg_output_len)))

        # Choose priority based on distribution
        priority_rand = random.random()
        cumulative_prob = 0.0
        priority = "low" # Default if something goes wrong
        for prio, prob in self.priority_distribution.items():
             cumulative_prob += prob
             if priority_rand <= cumulative_prob:
                  priority = prio
                  break

        self.request_counter += 1
        return {
            "request_id": f"synth_{self.request_counter}",
            "arrival_time": arrival_time,
            "prompt_len": prompt_len,
            "output_len": output_len,
            "priority": priority,
        }

    def get_arrivals(self, current_time: float) -> List[Dict[str, Any]]:
        """Returns requests that have arrived by current_time."""
        arrivals = []
        if self.workload_type == "trace":
            # Pop requests from the pre-loaded queue
            while self.request_queue and self.request_queue[0]["arrival_time"] <= current_time:
                arrivals.append(self.request_queue.pop(0))
        elif self.workload_type == "synthetic":
            # Generate requests until their arrival time exceeds current_time
            # Generate first request if none exist
            if self.last_arrival_time == 0:
                 next_req = self._generate_synthetic_request()
                 self.request_queue.append(next_req) # Use queue to hold the next one

            # Generate more while the next one is within the current time
            while self.request_queue and self.request_queue[0]["arrival_time"] <= current_time:
                 arrivals.append(self.request_queue.pop(0))
                 # Generate the *next* request and add it to the queue
                 next_req = self._generate_synthetic_request()
                 self.request_queue.append(next_req) # Queue holds the one after the current time

        return arrivals

# =============================================================================
# 2. Simulated vLLM Engine
# =============================================================================

class SimulatedVllmEngine:
    """Simulates key aspects of a vLLM engine: batching, KV cache, processing time."""

    def __init__(self, node_id: str, gpu_id: int, config: Dict[str, Any]):
        self.node_id = node_id
        self.gpu_id = gpu_id
        self.config = config
        self.ttft = config.get("ttft", 0.05) # Time To First Token overhead (seconds)
        self.tpot = config.get("tpot", 0.005) # Time Per Output Token (seconds/token)
        self.max_batch_size = config.get("max_batch_size", 8)
        # KV Cache Simulation (Simplified: based on total tokens)
        self.kv_bytes_per_token = config.get("kv_bytes_per_token", 2 * 2 * 2) # Example: 2 layers * 2 bytes/dtype * 2 (key+value) - Needs real model info!
        self.max_kv_tokens = int(config.get("kv_cache_size_gb", 4) * (1024**3) / self.kv_bytes_per_token)

        self.queued_requests: deque[Dict[str, Any]] = deque()
        self.active_batch: List[Dict[str, Any]] = []
        self.batch_finish_time: Optional[float] = None
        self.current_kv_tokens: int = 0

        logger.debug(f"Node {node_id} GPU {gpu_id}: Engine init (Batch: {self.max_batch_size}, KV Tok: {self.max_kv_tokens}, TTFT: {self.ttft:.3f}, TPOT: {self.tpot:.4f})") # <<< DEBUG >>>

    def _can_fit_kv(self, request: Dict[str, Any]) -> bool:
        """Checks if a request's estimated KV cache fits."""
        # Simplified: assumes prompt_len + output_len occupy cache for duration
        estimated_tokens = request["prompt_len"] + request["output_len"]
        return (self.current_kv_tokens + estimated_tokens) <= self.max_kv_tokens

    def add_request(self, request: Dict[str, Any]) -> bool:
        """Add request to internal queue if initial KV check passes."""
        if self._can_fit_kv(request): # Check if it *could* potentially fit eventually
            self.queued_requests.append(request)
            logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Queued job {request['job_id']} (Qsize: {len(self.queued_requests)})") # <<< DEBUG >>>
            return True
        else:
            kv_needed = request["prompt_len"] + request["output_len"]
            logger.warning(f"Node {self.node_id} GPU {self.gpu_id}: Rejecting job {request['job_id']} - KV limit (Need: {kv_needed}, Have: {self.max_kv_tokens - self.current_kv_tokens}, Max: {self.max_kv_tokens})") # <<< DEBUG >>>
            return False

    def _form_batch(self) -> List[Dict[str, Any]]:
        """Attempts to form a batch from the queue based on limits."""
        batch = []
        potential_kv_load = self.current_kv_tokens
        indices_to_remove = []

        # Simple greedy batching: Take from front of queue if fits
        for i, req in enumerate(self.queued_requests):
            if len(batch) >= self.max_batch_size:
                break

            req_kv_estimate = req["prompt_len"] + req["output_len"] # Simplified estimate
            if (potential_kv_load + req_kv_estimate) <= self.max_kv_tokens:
                batch.append(req)
                potential_kv_load += req_kv_estimate
                indices_to_remove.append(i)
            # else: logger.debug(f"Req {req['job_id']} kv {req_kv_estimate} exceeds limit {self.max_kv_tokens - potential_kv_load}")

        # Remove selected requests from queue (in reverse index order to avoid shifting issues)
        for i in sorted(indices_to_remove, reverse=True):
            del self.queued_requests[i]

        return batch

    def update(self, current_time: float) -> List[Tuple[str, bool]]:
        """Advance simulation: check completions, form new batch."""
        completed_jobs = []

        # 1. Check for completion of active batch
        if self.active_batch and self.batch_finish_time is not None: # and current_time >= self.batch_finish_time:
            active_ids = [j['job_id'] for j in self.active_batch]
            logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Update Check - Time={current_time:.4f}, FinishTime={self.batch_finish_time:.4f}, Batch={active_ids}")
            if current_time >= self.batch_finish_time:
              batch_job_ids = [req["job_id"] for req in self.active_batch]
              logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Batch COMPLETED at {current_time:.3f} (Finish time was: {self.batch_finish_time:.3f}). Jobs: {batch_job_ids}") # <<< DEBUG: Changed log level
              batch_kv_freeing = 0
              for req in self.active_batch:
                # Instrument end time
                req["streaming_end_time"] = current_time
                # Mark completion
                completed_jobs.append((req["job_id"], True))
                # Free KV cache associated with this request
                req_kv_estimate = req["prompt_len"] + req["output_len"]
                self.current_kv_tokens -= req_kv_estimate
              self.active_batch = []
              self.batch_finish_time = None
              self.current_kv_tokens = max(0, self.current_kv_tokens) # Floor at 0
              logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Freed {batch_kv_freeing} KV tokens. Current KV: {self.current_kv_tokens}") # <<< DEBUG >>>


        # 2. Try to form and start a new batch if idle
        if not self.active_batch and self.queued_requests:
            new_batch = self._form_batch()
            if new_batch:
                self.active_batch = new_batch
                # Assign KV cache usage
                batch_kv = sum(r["prompt_len"] + r["output_len"] for r in new_batch)
                self.current_kv_tokens += batch_kv
                max_output = 0

                # Instrument per-phase start/end for each request
                for req in self.active_batch:
                    req["sim_start_time"] = current_time
                    req.setdefault("prefill_start_time", current_time)
                    req.setdefault("prefill_end_time", current_time + self.ttft)
                    req.setdefault("streaming_start_time", current_time + self.ttft)
                    max_output = max(max_output, req["output_len"])
                processing_time = self.ttft + (self.tpot * max_output)
                if processing_time <= 0:
                    logger.warning(f"Node {self.node_id} GPU {self.gpu_id}: Calculated zero or negative processing time ({processing_time:.4f}) for batch. Using small default.")
                    processing_time = 0.001
                self.batch_finish_time = current_time + processing_time       

                for req in self.active_batch:
                    req.setdefault("streaming_end_time", self.batch_finish_time)
                job_ids = [j['job_id'] for j in self.active_batch]
                # <<< DEBUG: Changed log level >>>
                logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Starting batch (size {len(self.active_batch)}, KV {batch_kv}/{self.max_kv_tokens}) at {current_time:.3f}. Finish est: {self.batch_finish_time:.3f}. Jobs: {job_ids}")
                logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Est Proc Time: {processing_time:.4f} (TTFT: {self.ttft:.4f}, MaxOutput: {max_output}, TPOT: {self.tpot:.4f})") # <<< DEBUG >>>
                logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Current KV: {self.current_kv_tokens}") # <<< DEBUG >>>

        return completed_jobs        

    def cancel_request(self, job_id: str):
        """Remove request from queue or active batch."""
        # Check queue first
        initial_len = len(self.queued_requests)
        self.queued_requests = deque(req for req in self.queued_requests if req["job_id"] != job_id)
        if len(self.queued_requests) < initial_len:
             logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Cancelled queued request {job_id}")
             return

        # Check active batch
        req_to_remove = None
        for req in self.active_batch:
            if req["job_id"] == job_id:
                req_to_remove = req
                break
        if req_to_remove:
            logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Cancelled active request {job_id}")
            self.active_batch.remove(req_to_remove)
            # Free KV cache (simplified)
            req_kv_estimate = req_to_remove["prompt_len"] + req_to_remove["output_len"]
            self.current_kv_tokens -= req_kv_estimate
            self.current_kv_tokens = max(0, self.current_kv_tokens)
            logger.debug(f"Node {self.node_id} GPU {self.gpu_id}: Freed {req_kv_estimate} KV tokens from cancelled job. Current KV: {self.current_kv_tokens}") # <<< DEBUG >>>
            # If batch becomes empty, reset finish time
            if not self.active_batch:
                 self.batch_finish_time = None
        # else: logger.debug(f"Job {job_id} not found for cancellation.")


    def get_status(self) -> Dict[str, Any]:
        """Return current status for scheduler."""
        return {
            "queued": len(self.queued_requests),
            "running": len(self.active_batch),
            "kv_tokens_used": self.current_kv_tokens,
            "kv_tokens_max": self.max_kv_tokens,
            "load_factor": (len(self.active_batch) + len(self.queued_requests)) / (self.max_batch_size + 1e-6) # Example load metric
        }

    # --- Migration Simulation (Simplified) ---
    def get_state_sim(self, job_id: str) -> Optional[bytes]:
         # Find job in active batch
         active_req = next((req for req in self.active_batch if req["job_id"] == job_id), None)
         if active_req:
              logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Simulating get_state for {job_id}")
              # Simulate state size based on current progress (very crude)
              # elapsed = current_time - active_req["sim_start_time"] ? Need current_time
              # progress = elapsed / (self.batch_finish_time - active_req["sim_start_time"]) ?
              # state_size = active_req['prompt_len'] + active_req['output_len'] * progress
              state_size_bytes = (active_req['prompt_len'] + active_req['output_len']) * self.kv_bytes_per_token
              # Simulate some overhead
              return b'sim_state_' * (state_size_bytes // 10 + 1)
         else:
              logger.warning(f"Node {self.node_id} GPU {self.gpu_id}: Cannot get state for {job_id}, not in active batch.")
              return None

    def resume_from_state_sim(self, job_info: Dict[str, Any]) -> bool:
         # Simulate adding job directly to queue (or maybe active batch if logic allows)
         # Assume KV cache check happens before calling this
         logger.info(f"Node {self.node_id} GPU {self.gpu_id}: Simulating resume for {job_info['job_id']}")
         # Simplification: Just add it to the queue like a new request
         self.queued_requests.appendleft(job_info) # Add to front maybe?
         return True

# =============================================================================
# 3. Simulated Local Scheduler
# =============================================================================
class SimulatedLocalScheduler:
     """Simulates basic local acceptance/queueing (less complex than real one)."""
     def __init__(self, node_id: str, engine: SimulatedVllmEngine, max_local_queue: int):
          self.node_id = node_id
          self.engine = engine # Direct access to the single engine per node (simplification)
          self.max_local_queue = max_local_queue

     def submit_request(self, job_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
          """Try to add job to the engine's queue."""
          current_qsize = len(self.engine.queued_requests) + len(self.engine.active_batch)
          # Check local queue limit *before* engine's KV check
          if current_qsize >= self.max_local_queue:
               logger.warning(f"Node {self.node_id}: Local NAK job {job_info['job_id']} - local queue full ({current_qsize}/{self.max_local_queue})")
               return False, "queue_full"

          accepted_by_engine = self.engine.add_request(job_info)
          if accepted_by_engine:
               # logger.debug(f"Node {self.node_id}: Local ACK job {job_info['job_id']}")
               return True, None
          else:
               # Engine rejected (e.g., impossible KV demands)
               logger.warning(f"Node {self.node_id}: Local NAK job {job_info['job_id']} - rejected by engine (KV limit?)")
               return False, "kv_limit"

     def get_status(self) -> Dict[str, Any]:
          return self.engine.get_status() # Delegate status to engine


# =============================================================================
# 4. Simulated Node Manager
# =============================================================================
class SimulatedNodeManager:
    """Represents a node with GPUs and a vLLM engine simulation."""
    def __init__(self, node_id: str, config: Dict[str, Any]):
        # Simplification: Assume one engine manages all GPUs via tensor parallel
        self.node_id = node_id
        self.num_gpus = config.get("num_gpus", 1)
        self.config = config # Store original node config

        # <<< FIX: Scale Engine Config based on num_gpus >>>
        engine_config = config.copy() # Start with node config
        # Assume base config values are per-GPU or need scaling
        base_batch_size = config.get("vllm_max_batch_size", 8)
        base_kv_gb = config.get("vllm_kv_cache_gb", 4.0)
        # Scale capacity linearly with #GPUs (adjust this scaling if needed)
        engine_config["max_batch_size"] = base_batch_size * self.num_gpus
        engine_config["kv_cache_size_gb"] = base_kv_gb * self.num_gpus
        logger.info(f"Node {node_id}: Scaled Engine Config - Max Batch: {engine_config['max_batch_size']}, KV Cache GB: {engine_config['kv_cache_size_gb']}")
        # <<< END FIX >>>

        # Pass the SCALED config to the engine
        self.engine = SimulatedVllmEngine(node_id, 0, engine_config)
        # Pass the ORIGINAL node config to local scheduler if it needs max_local_queue_size etc.
        self.local_scheduler = SimulatedLocalScheduler(node_id, self.engine, config.get("max_local_queue_size", 32))



        # State for simulation control
        self.status = "active" # 'active', 'migrating_from', 'migrating_to', 'inactive'
        self.migrating_job_id: Optional[str] = None
        self.migration_target_node: Optional[str] = None
        self.migration_state_blob: Optional[bytes] = None
        self.migration_finish_time: Optional[float] = None

        logger.info(f"Node {node_id}: Initialized with {self.num_gpus} GPUs.")

    def update(self, current_time: float) -> List[Tuple[str, bool]]:
        """Update engine state, check for migration completion."""
        batch_size_before_update = len(self.engine.active_batch) if self.engine.active_batch else 0
        was_busy = bool(self.engine.active_batch or self.engine.batch_finish_time)

        completed_jobs = self.engine.update(current_time) # Engine updates itself

        # Check if the batch *just* finished in this update step
        if was_busy and not self.engine.active_batch:
            # Use the size captured *before* the update
            batch_size = batch_size_before_update # Size of the batch that JUST finished

            # <<< ADD CRITICAL LOG HERE >>>
            logger.critical(f"STATS_DEBUG Node {self.node_id}: Recording completed batch. Size = {batch_size}. Prior Sum = {self.sim_state.node_batch_size_sum.get(self.node_id, 'Not Found')}")

            if batch_size > 0: # Only record stats if a non-empty batch completed
                if self.sim_state:
                    # Use .get for safety, though initialization should prevent KeyErrors now
                    current_count = self.sim_state.node_batch_counts.get(self.node_id, 0)
                    current_sum = self.sim_state.node_batch_size_sum.get(self.node_id, 0)

                    self.sim_state.node_batch_counts[self.node_id] = current_count + 1
                    self.sim_state.node_batch_size_sum[self.node_id] = current_sum + batch_size

                    # <<< ADD CRITICAL LOG HERE >>>
                    logger.critical(f"STATS_DEBUG Node {self.node_id}: Updated batch stats. New Count={current_count + 1}, New Sum={current_sum + batch_size}")
                else:
                    logger.error(f"Node {self.node_id}: sim_state reference missing, cannot record batch stats.")
            else:
                 logger.warning(f"Node {self.node_id}: Detected batch completion but captured size was 0. Not recording stats for this event.")

        # Check if migration state transfer finished
        if self.status == "migrating_from" and self.migrating_job_id and self.migration_finish_time and current_time >= self.migration_finish_time:
             logger.info(f"Node {self.node_id}: Finished simulated state transfer for job {self.migrating_job_id} to {self.migration_target_node}")
             # Simulate sending state to target (in main loop) and then receiving MigrationComplete callback
             # For now, just mark self as active again, main loop handles notification
             blob_to_send = self.migration_state_blob
             target = self.migration_target_node
             job_id = self.migrating_job_id
             self.status = "active"
             self.migrating_job_id = None
             self.migration_target_node = None
             self.migration_state_blob = None
             self.migration_finish_time = None
             # Return state blob to main loop to send to target node?
             # Needs refinement based on how migration is orchestrated. Let main loop handle it.

        return completed_jobs

    def receive_job_sim(self, job_info: Dict[str, Any]) -> bool:
        """Simulates receiving a job dispatch from BloxManager."""
        if self.status != "active":
             logger.warning(f"Node {self.node_id}: Rejecting job {job_info['job_id']} due to status '{self.status}'")
             return False, None # NAK if not active
        return self.local_scheduler.submit_request(job_info)

    def initiate_migration_sim(self, job_id: str, target_node_ip: str, current_time: float) -> bool:
        """Simulates starting a migration *away* from this node."""
        if self.status != "active":
             logger.warning(f"Node {self.node_id}: Cannot initiate migration for {job_id}, status is {self.status}")
             return False

        # Simulate getting state (adds delay)
        state_blob = self.engine.get_state_sim(job_id)
        if state_blob is None:
            logger.error(f"Node {self.node_id}: Failed to get simulated state for migrating job {job_id}")
            return False # Failed to start migration

        logger.info(f"Node {self.node_id}: Initiating migration of {job_id} to {target_node_ip}. Simulating state transfer.")
        self.status = "migrating_from"
        self.migrating_job_id = job_id
        self.migration_target_node = target_node_ip
        self.migration_state_blob = state_blob # Store blob to be 'sent' later

        # Simulate state transfer time (e.g., based on size)
        transfer_time = len(state_blob) / (100 * 1024 * 1024) + 0.1 # Example: 0.1s + 100MB/s
        self.migration_finish_time = current_time + transfer_time

        # Cancel the job locally in the engine *after* state is retrieved
        self.engine.cancel_request(job_id)

        return True # Acknowledged initiation

    def receive_migration_sim(self, job_info: Dict[str, Any], state_blob: bytes) -> bool:
        """Simulates receiving a migration *to* this node."""
        if self.status != "migrating_to":
             logger.error(f"Node {self.node_id}: Received unexpected migration state for {job_info['job_id']} while status is {self.status}")
             # Should ideally rollback status in BloxManager Sim
             return False

        logger.info(f"Node {self.node_id}: Receiving migration state for {job_info['job_id']}")
        # Simulate resuming in engine
        success = self.engine.resume_from_state_sim(job_info) # Pass job_info too

        if success:
             logger.info(f"Node {self.node_id}: Successfully resumed migrated job {job_info['job_id']}. Setting status to active.")
             self.status = "active"
             # Clear migration state variables if stored locally
        else:
             logger.error(f"Node {self.node_id}: Failed to resume migrated job {job_info['job_id']}. Reverting status to active (problematic state).")
             # This node failed, the job is lost. Revert status.
             self.status = "active"

        return success


    def cancel_job_sim(self, job_id: str):
        """Simulates receiving a cancellation request."""
        logger.info(f"Node {self.node_id}: Received cancellation request for job {job_id}")
        self.engine.cancel_request(job_id)


    def get_status_for_scheduler(self) -> Dict[str, Any]:
        """Returns node status relevant to the global scheduler."""
        engine_status = self.engine.get_status()
        return {
            "ipaddr": self.node_id, # Use node_id as ipaddr identifier
            "port": 50051, # Dummy port
            "num_gpus": self.num_gpus,
            "status": self.status,
            "models": [self.config.get("model", "default_model")], # Report supported model
            # Add engine status details
            "queued_local": engine_status["queued"],
            "running_local": engine_status["running"],
            "kv_tokens_used": engine_status["kv_tokens_used"],
            "kv_tokens_max": engine_status["kv_tokens_max"],
            "load_factor": engine_status["load_factor"],
            # Add last_seen timestamp externally in main loop
        }

# =============================================================================
# 5. Simulation State
# =============================================================================

class SimulationState:
    """Holds the overall state of the simulation."""
    def __init__(self):
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.global_pending_requests: List[Tuple[int, float, Dict[str, Any]]] = []
        self.pending_lock = threading.Lock() # Keep if needed
        self.gpu_busy_time: Dict[str, float] = {}
        self.local_queue_naks: Dict[str, int] = {}
        self.kv_reject_naks: Dict[str, int] = {}
        self.other_naks: Dict[str, int] = {}
        # <<< RESTORE STATS DICTIONARIES >>>
        self.node_batch_counts: Dict[str,int] = {}    # how many batches ran
        self.node_batch_size_sum: Dict[str,int] = {}    # sum of all batch sizes



    def add_job(self, job_info: Dict[str, Any]):
        """Add a newly arrived job."""
        job_id = job_info["job_id"]
        if job_id in self.jobs:
             logger.warning(f"Job {job_id} already exists in state.")
             return
        self.jobs[job_id] = {**job_info, "status": "pending"}
        # logger.debug(f"State: Added job {job_id} (status: pending)")

    def update_job_status(self, job_id: str, status: str, location: Optional[str] = None, end_time: Optional[float] = None):
        """Update the status and optionally location/end_time of a job."""
        if job_id in self.jobs:
            old_status = self.jobs[job_id].get("status")
            self.jobs[job_id]["status"] = status
            if location:
                self.jobs[job_id]["node_ip"] = location
            if status in ["finished", "failed", "cancelled"] and "end_time" not in self.jobs[job_id]:
                 self.jobs[job_id]["end_time"] = end_time if end_time else time.time() # Use sim time
            if old_status == "dispatching" and status == "running" and "launch_time" not in self.jobs[job_id]:
                 self.jobs[job_id]["launch_time"] = end_time # Assume running starts now
            elif status == "running" and "launch_time" not in self.jobs[job_id]:
                 # If directly set to running (e.g. migration resume), use current time
                 self.jobs[job_id]["launch_time"] = end_time
        else:
            logger.warning(f"State: Cannot update status for unknown job {job_id}")

    def get_active_jobs(self) -> Dict[str, Any]:
        """Return snapshot of jobs currently considered active (running/migrating)."""
        return {jid: info for jid, info in self.jobs.items() if info.get("status") in ["running", "migrating"]}

    def get_all_jobs_history(self) -> List[Dict[str, Any]]:
         """Return list of all job records for final metrics."""
         return list(self.jobs.values())

# =============================================================================
# 6. Main Simulation Loop
# =============================================================================

def run_simulation(args):
    """Orchestrates the inference simulation."""
    logger.info("Starting Inference Simulation...")
    logger.info(f"Args: {vars(args)}")

    start_sim_wall_time = time.time()
    current_sim_time = 0.0
    simulation_duration = args.simulation_duration

    # --- Initialization ---
    sim_state = SimulationState()
    workload_gen = InferenceWorkloadGenerator(vars(args)) # Pass args as config

    # Initialize Nodes
    nodes: Dict[str, SimulatedNodeManager] = {}
    for i in range(args.num_nodes):
        node_id = f"node_{i+1}"
        node_config = { "num_gpus": args.gpus_per_node, "model": args.model,"ttft": args.vllm_ttft, "tpot": args.vllm_tpot,"max_batch_size": args.vllm_max_batch_size,"kv_cache_size_gb": args.vllm_kv_cache_gb,"max_local_queue_size": args.max_local_queue_size,"kv_bytes_per_token": 8 }
        node = SimulatedNodeManager(node_id, node_config)
        node.sim_state = sim_state
        nodes[node_id] = node
        sim_state.gpu_busy_time[node_id] = 0.0
        sim_state.node_batch_counts[node_id] = 0
        sim_state.node_batch_size_sum[node_id] = 0

    policy_args = {
         "job_timeout_sec": args.job_timeout_sec,
         "enable_migration": args.enable_migration,
         "vllm_max_batch_size": args.vllm_max_batch_size, # <<< ADDED for capacity calc
         # Add others...
    }
    policy_args = {k:v for k,v in policy_args.items() if hasattr(args, k)}
    scheduler_policy = make_policy(args.policy, **policy_args)
    last_scheduler_run_time = -args.scheduling_interval
    last_node_update_time = 0.0
    processed_arrivals_count = 0
    migration_state_transfers = {}

    while current_sim_time < simulation_duration:

        # --- A. Workload Arrivals ---
        new_arrivals = workload_gen.get_arrivals(current_sim_time)
        if new_arrivals:
            logger.info(f"Time {current_sim_time:.3f}: {len(new_arrivals)} new requests arrived.")
            with sim_state.pending_lock:
                 for req in new_arrivals:
                      # Assign a unique Blox Job ID
                      job_id = f"blox_{uuid.uuid4()}"
                      req["job_id"] = job_id
                      req["submit_time"] = req["arrival_time"] # Use arrival as submit time
                      # Add to global state and pending queue
                      sim_state.add_job(req)
                      prio_val = PRIORITY_MAP.get(req.get("priority", "normal").lower(), NORMAL_PRIORITY)
                      heapq.heappush(sim_state.global_pending_requests, (prio_val, req["submit_time"], req))
                 processed_arrivals_count += len(new_arrivals)

        # --- B. Node Updates (More Frequent) ---
        node_update_time_elapsed = current_sim_time - last_node_update_time
        if node_update_time_elapsed >= DEFAULT_NODE_UPDATE_INTERVAL or last_node_update_time < 0:
            completions_from_nodes = []
            for node in nodes.values():
                 completed = node.update(current_sim_time)
                 completions_from_nodes.extend(completed)
            # Track GPU busy time for each node
            for nid, node in nodes.items():
                status = node.engine.get_status()
                # If there's an active batch, that GPU is busy
                if status["running"] > 0:
                    sim_state.gpu_busy_time[nid] += DEFAULT_NODE_UPDATE_INTERVAL

            # Process completions
            if completions_from_nodes:
              logger.debug(f"Time {current_sim_time:.3f}: Processing {len(completions_from_nodes)} completions from node updates.") # <<< DEBUG >>>
              for job_id, success in completions_from_nodes:
                  status = "finished" if success else "failed"
                  sim_state.update_job_status(job_id, status, end_time=current_sim_time)
            last_node_update_time = current_sim_time

        # --- C. Global Scheduler Run (Less Frequent) ---
        scheduler_time_elapsed = current_sim_time - last_scheduler_run_time
        if scheduler_time_elapsed >= args.scheduling_interval or last_scheduler_run_time < 0:
            logger.debug(f"--- Time {current_sim_time:.3f}: Running Global Scheduler ({args.policy}) ---")
            last_scheduler_run_time = current_sim_time

            # 1. Get State Snapshots for Scheduler
            active_jobs_snapshot = sim_state.get_active_jobs()
            with sim_state.pending_lock:
                 # Create sorted list of pending job details
                 pending_jobs_snapshot = [details for _, _, details in sorted(sim_state.global_pending_requests)]
            # Get current status from each node
            node_status_snapshot = {nid: node.get_status_for_scheduler() for nid, node in nodes.items()}
            # Add last_seen timestamp (using current sim time)
            for info in node_status_snapshot.values(): info["last_seen"] = current_sim_time
            # Cluster view (simplified: just the node status dict for now)
            cluster_view_snapshot = node_status_snapshot

            # 2. Run Scheduler Policy
            dispatch_decisions, migration_decisions, jobs_to_cancel = scheduler_policy.schedule(
                 active_jobs=active_jobs_snapshot,
                 pending_jobs=pending_jobs_snapshot,
                 nodes=node_status_snapshot,
                 cluster_view=cluster_view_snapshot,
                 current_time=current_sim_time
            )

            # 3. Execute Cancellations
            if jobs_to_cancel:
                logger.info(f"Time {current_sim_time:.3f}: Cancelling jobs: {jobs_to_cancel}")
                for job_id in jobs_to_cancel:
                     if job_id in active_jobs_snapshot: # Check if still active
                          node_ip = active_jobs_snapshot[job_id].get("node_ip")
                          if node_ip and node_ip in nodes:
                               nodes[node_ip].cancel_job_sim(job_id)
                               # Update state - will be marked failed/cancelled upon next completion check
                               sim_state.update_job_status(job_id, "cancelling", end_time=current_sim_time)
                          else: logger.warning(f"Cannot cancel job {job_id}, node {node_ip} invalid.")
                     else: logger.debug(f"Job {job_id} already inactive, skipping cancellation.")


            # 4. Execute Migrations
            if migration_decisions:
                 logger.info(f"Time {current_sim_time:.3f}: Initiating migrations: {migration_decisions}")
                 for job_id, from_node_id, to_node_id in migration_decisions:
                      if job_id not in active_jobs_snapshot: continue # Job finished/cancelled already
                      if from_node_id not in nodes or to_node_id not in nodes: continue # Nodes invalid

                      # Check current node statuses before initiating
                      if nodes[from_node_id].status == "active" and nodes[to_node_id].status == "active":
                            # Initiate migration on source node
                            mig_ack = nodes[from_node_id].initiate_migration_sim(job_id, to_node_id, current_sim_time)
                            if mig_ack:
                                 logger.info(f"Migration initiated for {job_id} from {from_node_id} -> {to_node_id}")
                                 # Mark nodes as migrating in simulation state
                                 nodes[from_node_id].status = "migrating_from"
                                 nodes[to_node_id].status = "migrating_to"
                                 sim_state.update_job_status(job_id, "migrating", location=from_node_id) # Location is still source for now
                            else:
                                 logger.error(f"Source node {from_node_id} failed to initiate migration for {job_id}")
                      else:
                           logger.warning(f"Skipping migration {job_id} -> {to_node_id}: Node status conflict ({nodes[from_node_id].status}, {nodes[to_node_id].status})")

            # 5. Execute Dispatches
            if dispatch_decisions:
                 logger.info(f"Time {current_sim_time:.3f}: Dispatching jobs: {dispatch_decisions}")
                 dispatched_this_round = set()
                 with sim_state.pending_lock:
                      # Find corresponding job details and attempt dispatch
                      temp_pending = list(sim_state.global_pending_requests) # Work on a copy
                      indices_to_remove_from_heap = []

                      for i, (prio, ts, details) in enumerate(temp_pending):
                           job_id = details["job_id"]
                           if job_id in dispatch_decisions:
                                target_node_id = dispatch_decisions[job_id]
                                if target_node_id in nodes:
                                     # Send job to simulated node
                                     node_ack , node_reason = nodes[target_node_id].receive_job_sim(details)
                                     if node_ack:
                                          logger.debug(f"Node {target_node_id} ACKed job {job_id}. Updating state.")
                                          sim_state.update_job_status(job_id, "dispatching", location=target_node_id) # Intermediate state
                                          # Mark for removal from heap (using index is fragile if heap modified, find by content)
                                          entry_to_remove = (prio, ts, details)
                                          try:
                                              # Find exact entry in original heap and mark for removal
                                              original_index = sim_state.global_pending_requests.index(entry_to_remove)
                                              indices_to_remove_from_heap.append(original_index)
                                              dispatched_this_round.add(job_id)
                                              # NOTE: Actual launch happens inside node update based on engine
                                          except ValueError:
                                               logger.error(f"Consistency Error: Cannot find pending job {job_id} in heap for removal!")
                                     else:
                                          if node_reason == "queue_full":
                                            sim_state.local_queue_naks[target_node_id] = (
                                             sim_state.local_queue_naks.get(target_node_id, 0) + 1
                                            )
                                          elif node_reason == "kv_limit":
                                            sim_state.kv_reject_naks[target_node_id] = (
                                              sim_state.kv_reject_naks.get(target_node_id, 0) + 1
                                            )
                                          else:
                                            # fallback if reason is missing
                                            sim_state.local_queue_naks[target_node_id] = (
                                              sim_state.local_queue_naks.get(target_node_id, 0) + 1
                                            )
                                else:
                                     logger.warning(f"Target node {target_node_id} for job {job_id} not found.")

                      # Remove dispatched jobs from actual heap efficiently (rebuild)
                      if indices_to_remove_from_heap:
                            new_heap = []
                            removed_job_ids = {
                              sim_state.global_pending_requests[i][2]['job_id']
                              for i in indices_to_remove_from_heap
                            }

                            # rebuild the pending list, dropping anything whose job_id is in that set
                            sim_state.global_pending_requests = [
                              entry for entry in sim_state.global_pending_requests
                              if entry[2]['job_id'] not in removed_job_ids
                            ]
                            
                            heapq.heapify(sim_state.global_pending_requests) # Maintain heap property

            # --- Handle Migration State Transfers ---
            # Check nodes that finished simulated transfer
            nodes_finished_transfer = []
            for node_id, node in nodes.items():
                 if node.status == "migrating_from" and node.migration_finish_time and current_sim_time >= node.migration_finish_time:
                      nodes_finished_transfer.append(node)

            for source_node in nodes_finished_transfer:
                 job_id = source_node.migrating_job_id
                 target_node_id = source_node.migration_target_node
                 state_blob = source_node.migration_state_blob
                 logger.info(f"Time {current_sim_time:.3f}: Simulating state transfer complete for {job_id}. Sending state to {target_node_id}.")

                 # Reset source node state (done in its update method, but ensure here)
                 source_node.status = "active"
                 source_node.migrating_job_id = None
                 source_node.migration_target_node = None
                 source_node.migration_state_blob = None
                 source_node.migration_finish_time = None

                 # Send state to target node
                 if target_node_id in nodes and state_blob is not None:
                      job_info = sim_state.jobs.get(job_id) # Get original job details
                      if job_info:
                           # Target node must be in 'migrating_to' state
                           if nodes[target_node_id].status == "migrating_to":
                                success = nodes[target_node_id].receive_migration_sim(job_info, state_blob)
                                if success:
                                     logger.info(f"Target node {target_node_id} successfully received migration for {job_id}")
                                     sim_state.update_job_status(job_id, "running", location=target_node_id) # Now running on target
                                else:
                                     logger.error(f"Target node {target_node_id} failed to receive migration for {job_id}. Job failed.")
                                     sim_state.update_job_status(job_id, "failed", end_time=current_sim_time)
                           else:
                                logger.error(f"Migration Error: Target node {target_node_id} not in migrating_to state for job {job_id}. Job failed.")
                                sim_state.update_job_status(job_id, "failed", end_time=current_sim_time)
                      else:
                           logger.error(f"Migration Error: Cannot find original job details for {job_id}. Job failed.")
                           sim_state.update_job_status(job_id, "failed", end_time=current_sim_time) # Mark as failed if details lost

                 else:
                      logger.error(f"Migration Error: Target node {target_node_id} not found or state blob missing for job {job_id}. Job failed.")
                      if job_id in sim_state.jobs:
                           sim_state.update_job_status(job_id, "failed", end_time=current_sim_time)


        # --- D. Advance Simulation Time ---
        # Simple fixed time step for now
        current_sim_time += DEFAULT_NODE_UPDATE_INTERVAL # Advance by smaller interval
        # Or use event-driven: next_event_time = min(next_arrival, next_completion, next_scheduler_run)

    # --- Simulation End ---
    logger.info(f"Simulation finished at time {current_sim_time:.3f}")
    end_sim_wall_time = time.time()
    logger.info(f"Total simulation wall clock time: {end_sim_wall_time - start_sim_wall_time:.3f} seconds")
    # --- NAK Summary ---
    logger.info("=== NAK Stats ===")
    for node_id in nodes:
        lq = sim_state.local_queue_naks.get(node_id, 0)
        kv = sim_state.kv_reject_naks.get(node_id, 0)
        logger.info(f"   {node_id}: queue-full_NAKs={lq}, kv-reject_NAKs={kv}")

        # <<< RESTORED/CORRECTED: Print Per-Node Batch/Util Stats >>>
    logger.info("=== Batch / Utilization Stats (End of Sim) ===")
    total_time = simulation_duration # Use the actual duration
    for node_id in nodes:
        busy = sim_state.gpu_busy_time.get(node_id, 0.0) # Use the correct busy time
        batches = sim_state.node_batch_counts.get(node_id, 0)
        szsum = sim_state.node_batch_size_sum.get(node_id, 0) # Get the sum

        # CALCULATION: Is 'batches' non-zero here? Is 'szsum' non-zero here?
        avg_bs = (szsum / batches) if batches else 0.0

        util = 0.0
        if total_time > 0:
            util = 100.0 * busy / total_time
        # PRINTING: Does avg_bs get formatted to 0.00?
        logger.info(f"   {node_id}: batches={batches}, avg_batch_size={avg_bs:.2f}, util={util:.1f}% (BusyTime={busy:.3f}s)")

    # --- Calculate and Print Metrics ---
    metrics = summarize_inference_metrics(
        job_state=sim_state,
        cluster_state=None, # Pass simulated cluster state if metrics need it
        start_time=0.0,
        end_time=current_sim_time
    )
    # Compute actual per-GPU utilization
    total_busy = sum(sim_state.gpu_busy_time.values())
    total_gpus = args.num_nodes * args.gpus_per_node if args.num_nodes and args.gpus_per_node else 1
    utilization_pct = (total_busy / (total_gpus * simulation_duration)) * 100
    metrics["avg_gpu_utilization"] = f"{utilization_pct:.2f}%"
    print("\n--- Simulation Metrics ---")
    print(json.dumps(metrics, indent=2))

    # Optional: Save final job state
    if args.output_job_log:
         try:
              with open(args.output_job_log, 'w') as f:
                   json.dump(sim_state.jobs, f, indent=2)
              logger.info(f"Saved final job state to {args.output_job_log}")
         except Exception as e:
              logger.error(f"Failed to save job log: {e}")


# =============================================================================
# 7. Argument Parsing and Main Execution
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Blox Inference Simulator")
    parser.add_argument("--simulation-duration", type=float, default=3600.0, help="Total simulation time in seconds")
    parser.add_argument("--scheduling-interval", type=float, default=DEFAULT_SCHEDULING_INTERVAL, help="Global scheduler run interval (seconds)")
    parser.add_argument("--output-job-log", type=str, default="inference_sim_job_log.json", help="File to save final job states")
    parser.add_argument("--num-nodes", type=int, default=16, help="Number of simulated nodes")
    parser.add_argument("--gpus-per-node", type=int, default=1, help="Number of simulated GPUs per node")
    parser.add_argument("--model", type=str, default="sim_model", help="Simulated model identifier (used for node capabilities)")
    parser.add_argument("--vllm-ttft", type=float, default=0.05, help="Simulated vLLM TTFT overhead (seconds)")
    parser.add_argument("--vllm-tpot", type=float, default=0.005, help="Simulated vLLM TPOT (seconds/token)")
    parser.add_argument("--vllm-max-batch-size", type=int, default=8, help="Simulated vLLM max batch size")
    parser.add_argument("--vllm-kv-cache-gb", type=float, default=4.0, help="Simulated available KV cache per node (GB)")
    parser.add_argument("--max-local-queue-size", type=int, default=16, help="Maximum jobs queued locally on a node before NAK")
    parser.add_argument("--workload-type", choices=["synthetic", "trace"], default="synthetic", help="Type of workload generator")
    parser.add_argument("--workload-trace-file", type=str, default=None, help="Path to trace file (JSONL format) if workload-type is trace")
    parser.add_argument("--workload-arrival-rate", type=float, default=10.0, help="Average requests per second (for synthetic)") # <<< Reduced default rate
    parser.add_argument("--workload-avg-prompt-len", type=int, default=128, help="Average prompt length (for synthetic)")
    parser.add_argument("--workload-avg-output-len", type=int, default=64, help="Average output length (for synthetic)")
    parser.add_argument("--policy", type=str, default="fifo", choices=list(_POLICY_REGISTRY.keys()), help="Global scheduling policy")
    parser.add_argument("--enable-migration", action="store_true", help="Enable migration for supported policies (e.g., Llumnix)")
    parser.add_argument("--job-timeout-sec", type=int, default=300, help="Job timeout in seconds (for FIFOx policy)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="Set the logging level")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    log_level_map = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logging.getLogger().setLevel(log_level_map.get(args.log_level, logging.INFO))

    # Pass args directly, workload generator init expects a dict
    workload_args = {
         "type": args.workload_type,
         "trace_file": args.workload_trace_file,
         "arrival_rate": args.workload_arrival_rate,
         "avg_prompt_len": args.workload_avg_prompt_len,
         "avg_output_len": args.workload_avg_output_len,
         # Add priority distribution if needed
    }
    # Add workload config to main args namespace for simplicity
    vars(args).update({"workload_config": workload_args})

    run_simulation(args)