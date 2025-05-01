# schedulers/llumnix_scheduler.py

import logging
import time
import random # For random node selection among equals
from typing import Dict, List, Tuple, Any, Set, Optional
from .scheduler_policy import BaseSchedulerPolicy

logger = logging.getLogger(__name__)

# --- Constants and Configuration (move to args/config later) ---
HIGH_PRIORITY = 0
NORMAL_PRIORITY = 1
LOW_PRIORITY = 2
PRIORITY_MAP = {"high": HIGH_PRIORITY, "normal": NORMAL_PRIORITY, "low": LOW_PRIORITY}

# --- Tunable Parameters (Consider moving to args) ---
MIGRATION_KV_PRESSURE_THRESHOLD_HIGH = 0.85 # Consider node overloaded if KV usage > X%
MIGRATION_KV_PRESSURE_THRESHOLD_LOW = 0.50  # Consider node underloaded if KV usage < Y%
# Minimum priority difference required to consider migrating a running job for a pending one
MIGRATION_PRIORITY_DIFF_THRESHOLD = 1 # e.g., Normal (1) can displace Low (2)

class LlumnixScheduler(BaseSchedulerPolicy):
    """
    Implementation of Llumnix scheduling policy concepts within the simulator.
    Features two-phase dispatch and migration logic based on KV pressure and priority.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure enable_migration is set correctly from args or default
        self.enable_migration = bool(getattr(self, 'enable_migration', True))
        self.vllm_max_batch_size = int(getattr(self, 'vllm_max_batch_size', 8)) # Get from args

        logger.info(f"LlumnixScheduler initialized (Migration: {self.enable_migration}, BaseBatchSize: {self.vllm_max_batch_size}).")

    # === Helper Functions ===

    def _get_node_capacity(self, node_info: Dict[str, Any]) -> int:
        """Capacity = max concurrent requests (simplified to scaled batch size)."""
        num_gpus = node_info.get("num_gpus", 1)
        # Use the base batch size passed during init
        scaled_capacity = self.vllm_max_batch_size * num_gpus
        return max(1, scaled_capacity)

    def _estimate_job_kv(self, job_details: Dict[str, Any]) -> int:
        """Estimate KV tokens needed for a job."""
        # Simple estimate, could be refined
        return job_details.get("prompt_len", 0) + job_details.get("output_len", 0)

    def _get_node_kv_pressure(self, node_info: Dict[str, Any]) -> float:
        """Calculate current KV cache pressure (0.0 to 1.0)."""
        used = node_info.get("kv_tokens_used", 0)
        max_kv = node_info.get("kv_tokens_max", 0)
        if max_kv <= 0:
            return 1.0 # Treat node with no KV capacity as full
        return used / max_kv

    def _can_node_run_job(self, node_info: Dict[str, Any], job_details: Dict[str, Any], assignments_this_round: Dict[str, int], node_capacity: int) -> bool:
        """Check if node is active, has capacity (slots), and enough KV cache."""
        if node_info.get("status") != "active":
            return False

        # Check model compatibility (if implemented)
        target_model = job_details.get("target_model")
        if target_model and target_model not in node_info.get("models", []):
            # logger.debug(f"Node {node_info['ipaddr']} incompatible model for job {job_details['job_id']}")
            return False

        # Check slot capacity (running + assigned this round)
        running_local = node_info.get("running_local", 0) # Get from node status snapshot
        assigned_count = assignments_this_round.get(node_info['ipaddr'], 0)
        if (running_local + assigned_count) >= node_capacity:
             # logger.debug(f"Node {node_info['ipaddr']} full on slots for job {job_details['job_id']} (Running:{running_local}, Assigned:{assigned_count}, Cap:{node_capacity})")
             return False

        # Check KV cache capacity
        kv_needed = self._estimate_job_kv(job_details)
        kv_used = node_info.get("kv_tokens_used", 0)
        kv_max = node_info.get("kv_tokens_max", 0)
        # Account for KV potentially freed by jobs completing *before* this one starts? Hard to predict.
        # For simplicity, check against current usage.
        if (kv_used + kv_needed) > kv_max:
            # logger.debug(f"Node {node_info['ipaddr']} insufficient KV for job {job_details['job_id']} (Need:{kv_needed}, Used:{kv_used}, Max:{kv_max})")
            return False

        return True

    def _find_best_candidate_node(self, job_details: Dict[str, Any], candidate_nodes: List[str], nodes_state: Dict[str, Any]) -> Optional[str]:
        """Finds the best node among candidates based on lowest KV pressure."""
        best_node = None
        min_pressure = float('inf')

        for node_ip in candidate_nodes:
            node_info = nodes_state.get(node_ip)
            if not node_info: continue # Should not happen if candidate_nodes is from nodes_state keys

            pressure = self._get_node_kv_pressure(node_info)
            if pressure < min_pressure:
                min_pressure = pressure
                best_node = node_ip
            elif pressure == min_pressure:
                 # Tie-breaking: random choice among equals? Or stick with first?
                 # Random might be better for spreading load.
                 if random.choice([True, False]):
                      best_node = node_ip

        # if best_node: logger.debug(f"Job {job_details['job_id']}: Best candidate node {best_node} (KV Pressure: {min_pressure:.2f})")
        return best_node

    # === Main Scheduling Logic ===

    def schedule(self,
                 active_jobs: Dict[str, Any], # Snapshot {job_id: {..., 'node_ip', 'priority', 'launch_time'}}
                 pending_jobs: List[Dict[str, Any]], # Snapshot [{..., 'job_id', 'priority', 'submit_time'}]
                 nodes: Dict[str, Any], # Snapshot {node_ip: {..., 'status', 'kv_tokens_used', 'kv_tokens_max', 'running_local', 'num_gpus'}}
                 cluster_view: Any, # Currently unused, could hold aggregated stats
                 current_time: float
                ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]], Set[str]]:

        dispatch_decisions = {}
        migration_decisions = []
        jobs_to_cancel = set() # Cancellation logic not part of base Llumnix

        # --- Preparation ---
        active_nodes_map = {ip: info for ip, info in nodes.items() if info.get("status") == "active"}
        if not active_nodes_map:
            # logger.debug("Llumnix: No active nodes available.")
            return {}, [], set()

        # Get node capacities
        node_capacities = {ip: self._get_node_capacity(info) for ip, info in active_nodes_map.items()}

        # Categorize and sort pending jobs
        pending_by_priority = {prio: [] for prio in [HIGH_PRIORITY, NORMAL_PRIORITY, LOW_PRIORITY]}
        for job in pending_jobs:
            prio_val = PRIORITY_MAP.get(job.get("priority", "normal").lower(), NORMAL_PRIORITY)
            pending_by_priority[prio_val].append(job)
        for prio in pending_by_priority:
            pending_by_priority[prio].sort(key=lambda j: j.get("submit_time", 0))

        # Track assignments and resource usage *within this round*
        assignments_this_round = {ip: 0 for ip in active_nodes_map}
        # Track tentative KV usage changes this round (positive for dispatch, negative for migration source)
        kv_delta_this_round = {ip: 0 for ip in active_nodes_map}


        # --- Phase 1: Non-preemptive Dispatch (High Priority) ---
        logger.debug("Llumnix: Starting Phase 1 Dispatch (High Priority)")
        processed_job_ids_p1 = set()
        candidate_nodes_p1 = list(active_nodes_map.keys()) # All active nodes are candidates

        for job in pending_by_priority[HIGH_PRIORITY]:
            job_id = job["job_id"]
            job_kv_estimate = self._estimate_job_kv(job)
            eligible_nodes = []

            # Find nodes that *can* run the job (basic checks)
            for node_ip in candidate_nodes_p1:
                 node_info = active_nodes_map[node_ip]
                 node_cap = node_capacities[node_ip]
                 current_kv_used = node_info.get("kv_tokens_used", 0) + kv_delta_this_round[node_ip]
                 current_kv_max = node_info.get("kv_tokens_max", 0)
                 current_slot_load = node_info.get("running_local", 0) + assignments_this_round[node_ip]

                 # Check basic fit: slots and KV
                 if current_slot_load < node_cap and (current_kv_used + job_kv_estimate) <= current_kv_max:
                      eligible_nodes.append(node_ip)

            if not eligible_nodes:
                 # logger.debug(f"Llumnix P1: No nodes found with basic capacity for high-prio job {job_id}")
                 continue # Cannot dispatch this job now

            # Find the best among eligible nodes (e.g., lowest KV pressure)
            target_node = self._find_best_candidate_node(job, eligible_nodes, nodes)

            if target_node:
                dispatch_decisions[job_id] = target_node
                processed_job_ids_p1.add(job_id)
                assignments_this_round[target_node] += 1      # Increment assignments count
                kv_delta_this_round[target_node] += job_kv_estimate # Add KV load
                logger.info(f"Llumnix P1: Dispatching high-prio job {job_id} to {target_node}")
            # else: # Should not happen if eligible_nodes was populated
            #     logger.warning(f"Llumnix P1: Eligible nodes found but best node selection failed for {job_id}")


        # --- Migration Decisions ---
        logger.debug("Llumnix: Evaluating Migration Decisions")
        migrated_jobs_this_round = set()
        nodes_receiving_migration = set()
        nodes_sending_migration = set()

        if self.enable_migration:
            overloaded_nodes = []
            underloaded_nodes = []
            candidate_migration_targets = list(active_nodes_map.keys()) # Start with all active

            for node_ip, info in active_nodes_map.items():
                 # Exclude nodes already involved in assignment/migration this round? Maybe not for identifying overload.
                 pressure = self._get_node_kv_pressure(info)
                 if pressure > MIGRATION_KV_PRESSURE_THRESHOLD_HIGH:
                      overloaded_nodes.append((pressure, node_ip))
                 # Node is underloaded if pressure is low AND it has capacity *after* P1 assignments
                 elif pressure < MIGRATION_KV_PRESSURE_THRESHOLD_LOW:
                      current_slot_load = info.get("running_local", 0) + assignments_this_round.get(node_ip, 0)
                      if current_slot_load < node_capacities[node_ip]:
                           underloaded_nodes.append((pressure, node_ip))

            overloaded_nodes.sort(reverse=True) # Highest pressure first
            underloaded_nodes.sort() # Lowest pressure first

            logger.debug(f"Migration check: Overloaded={overloaded_nodes}, Underloaded={underloaded_nodes}")

            for _, source_node_ip in overloaded_nodes:
                 if not underloaded_nodes: break # No potential targets left
                 if source_node_ip in nodes_sending_migration or source_node_ip in nodes_receiving_migration: continue

                 # Find lowest priority RUNNING job on the source node
                 victim_job_id: Optional[str] = None
                 victim_priority = -1 # Higher number is lower priority

                 for job_id, details in active_jobs.items():
                      if details.get("node_ip") == source_node_ip and details.get("status") == "running":
                           prio = PRIORITY_MAP.get(details.get("priority", "normal").lower(), NORMAL_PRIORITY)
                           if prio > victim_priority: # Find lowest priority (highest value)
                                victim_priority = prio
                                victim_job_id = job_id

                 if victim_job_id and victim_priority > HIGH_PRIORITY: # Found a non-high-prio victim
                      victim_kv = self._estimate_job_kv(active_jobs[victim_job_id])
                      target_node_ip: Optional[str] = None

                      # Find best underloaded target NOT already involved in migration
                      potential_targets = []
                      for _, ip in underloaded_nodes:
                           if ip == source_node_ip: continue # Don't migrate to self
                           if ip in nodes_receiving_migration or ip in nodes_sending_migration: continue

                           target_info = active_nodes_map[ip]
                           target_cap = node_capacities[ip]
                           target_slot_load = target_info.get("running_local", 0) + assignments_this_round.get(ip, 0)
                           target_kv_used = target_info.get("kv_tokens_used", 0) + kv_delta_this_round.get(ip, 0)
                           target_kv_max = target_info.get("kv_tokens_max", 0)

                           # Check if target has slot and KV capacity
                           if target_slot_load < target_cap and (target_kv_used + victim_kv) <= target_kv_max:
                                potential_targets.append(ip)

                      if potential_targets:
                            # Choose best among potential targets (e.g., lowest KV pressure)
                           target_node_ip = self._find_best_candidate_node(active_jobs[victim_job_id], potential_targets, nodes)

                      if target_node_ip:
                            logger.info(f"Llumnix Migration: Planning migration {victim_job_id} (Prio {victim_priority}) from {source_node_ip} -> {target_node_ip}")
                            migration_decisions.append((victim_job_id, source_node_ip, target_node_ip))
                            migrated_jobs_this_round.add(victim_job_id)
                            nodes_receiving_migration.add(target_node_ip)
                            nodes_sending_migration.add(source_node_ip)

                            # Update tentative resource changes for Phase 2
                            kv_delta_this_round[source_node_ip] -= victim_kv
                            kv_delta_this_round[target_node_ip] += victim_kv
                            # assignments_this_round doesn't change yet, handled by main loop status update

                            # Remove target from underloaded list for this round? Yes.
                            underloaded_nodes = [(p,ip) for p,ip in underloaded_nodes if ip != target_node_ip]

                      # else: logger.debug(f"No suitable target found for victim {victim_job_id}")
                 # else: logger.debug(f"No suitable victim found on {source_node_ip}")


        # --- Phase 2: Opportunistic Dispatch (Normal/Low Priority) ---
        logger.debug("Llumnix: Starting Phase 2 Dispatch (Normal/Low Priority)")
        remaining_pending = [
             job for job in pending_by_priority[NORMAL_PRIORITY] if job["job_id"] not in processed_job_ids_p1
        ] + [
             job for job in pending_by_priority[LOW_PRIORITY] if job["job_id"] not in processed_job_ids_p1
        ]
        remaining_pending.sort(key=lambda j: (PRIORITY_MAP.get(j.get("priority", "normal").lower(), NORMAL_PRIORITY), j.get("submit_time", 0)))

        # Nodes available for P2 dispatch are active ones not involved in migration this round
        candidate_nodes_p2 = [
             ip for ip, info in active_nodes_map.items()
             if ip not in nodes_sending_migration and ip not in nodes_receiving_migration
        ]

        for job in remaining_pending:
            job_id = job["job_id"]
            if job_id in dispatch_decisions: continue # Should already be handled if logic correct

            job_kv_estimate = self._estimate_job_kv(job)
            eligible_nodes_p2 = []

            # Check basic fit on candidate nodes
            for node_ip in candidate_nodes_p2:
                 node_info = active_nodes_map[node_ip]
                 node_cap = node_capacities[node_ip]
                 # Check resources considering P1 assignments and migrations
                 current_kv_used = node_info.get("kv_tokens_used", 0) + kv_delta_this_round.get(node_ip, 0)
                 current_kv_max = node_info.get("kv_tokens_max", 0)
                 current_slot_load = node_info.get("running_local", 0) + assignments_this_round.get(node_ip, 0)

                 if current_slot_load < node_cap and (current_kv_used + job_kv_estimate) <= current_kv_max:
                      eligible_nodes_p2.append(node_ip)

            if not eligible_nodes_p2:
                 # logger.debug(f"Llumnix P2: No nodes found with basic capacity for job {job_id}")
                 continue

            # Find best among eligible nodes
            target_node = self._find_best_candidate_node(job, eligible_nodes_p2, nodes)

            if target_node:
                dispatch_decisions[job_id] = target_node
                assignments_this_round[target_node] += 1
                kv_delta_this_round[target_node] += job_kv_estimate
                logger.info(f"Llumnix P2: Dispatching job {job_id} to {target_node}")
            # else: logger.warning(f"Llumnix P2: Eligible nodes found but best node selection failed for {job_id}")
            # Note: This P2 implementation doesn't consider preempting/migrating *for* P2 jobs


        logger.info(f"Llumnix: Final Decisions - Dispatch={len(dispatch_decisions)}, Migrate={len(migration_decisions)}")
        return dispatch_decisions, migration_decisions, jobs_to_cancel