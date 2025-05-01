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

# Thresholds for migration (example values)
MIGRATION_LOAD_THRESHOLD = 0.8 # Consider node overloaded if > 80% 'full'
MIGRATION_IDLE_THRESHOLD = 0.2 # Consider node underloaded if < 20% 'full'
MIGRATION_MIN_BENEFIT = 1 # Arbitrary unit: Migrate only if benefit outweighs cost estimate

# Simplified cost/benefit for migration (example)
MIGRATION_COST = 0.5 # Overhead cost estimate

class LlumnixScheduler(BaseSchedulerPolicy):
    """
    Implementation of the Llumnix scheduling policy based on arXiv:2406.03243.
    Includes two-phase dispatch and migration logic.

    Simplifications:
    - KV cache reuse based on simple heuristics (node load/idle status).
    - Resource usage estimated by job count per node (needs enhancement).
    - Migration cost/benefit analysis is simplified.
    - Assumes nodes report basic status ('active', 'migrating_from', 'migrating_to').
    """
    def __init__(self, enable_migration: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.enable_migration = getattr(self, 'enable_migration', enable_migration)
        logger.info(f"LlumnixScheduler initialized (Migration enabled: {self.enable_migration}).")
        # TODO: Initialize any Llumnix-specific state tracking if needed

  
    def _get_node_capacity(self, node_info: Dict[str, Any]) -> int:
        """Estimate capacity based on scaled batch size."""
        # Assume base batch size was passed via args and scales with node GPUs
        # Need access to the base batch size arg here. Hacky: Assume it's on self if passed via kwargs.
        base_batch_size = getattr(self, 'vllm_max_batch_size', 8)
        num_gpus = node_info.get("num_gpus", 1)
        scaled_capacity = base_batch_size * num_gpus
        # logger.debug(f"Node {node_info.get('ipaddr','?')}: Calculated capacity {scaled_capacity} (BaseBS: {base_batch_size}, GPUs: {num_gpus})")
        return scaled_capacity

    def _get_node_current_load_factor(self, node_ip: str, active_jobs: Dict[str, Any], nodes: Dict[str, Any], assignments_this_round: Dict[str, int]) -> float:
        """Estimate current load + planned load this round as fraction of capacity."""
        if node_ip not in nodes: return 1.0 # Treat unknown node as full
        capacity = self._get_node_capacity(nodes[node_ip])
        if capacity <= 0: return 1.0 # Avoid division by zero or nonsensical load

        # Count jobs currently running according to the snapshot
        current_jobs_on_node = sum(1 for job in active_jobs.values() if job.get("node_ip") == node_ip)
        # Add jobs planned for assignment this round
        planned_assignments = assignments_this_round.get(node_ip, 0)

        total_load = current_jobs_on_node + planned_assignments
        return total_load / capacity

    def _find_best_node_for_job(
        self,
        job_details: Dict[str, Any],
        nodes: Dict[str, Any], # Active nodes snapshot
        active_jobs: Dict[str, Any],
        current_available_slots: Dict[str, int] # Slots available *now*
        ) -> Optional[str]:

        target_model = job_details.get("target_model")
        candidate_nodes = []
        dummy_assignments = {} # For load factor calc based on current state only

        for node_ip, slots_available in current_available_slots.items():
            if slots_available <= 0: continue
            node_info = nodes.get(node_ip)
            if not node_info or node_info.get("status") != "active": continue
            if target_model and target_model not in node_info.get("models", []): continue

            # Calculate score based on current load factor (lower is better)
            # Use dummy_assignments because we only care about *current* load here
            load_factor = self._get_node_current_load_factor(node_ip, active_jobs, nodes, dummy_assignments)
            score = load_factor
            candidate_nodes.append((score, node_ip))

        if not candidate_nodes:
            return None

        candidate_nodes.sort()
        best_score = candidate_nodes[0][0]

        # <<< FIX: Random Tie-Breaking >>>
        # Get all nodes with the best score
        best_nodes = [ip for score, ip in candidate_nodes if score == best_score]

        if not best_nodes: # Should not happen if candidate_nodes is not empty
             return None

        # Choose randomly among the best nodes
        chosen_node_ip = random.choice(best_nodes)
        # <<< END FIX >>>

        logger.debug(f"Job {job_details['job_id']}: Best node candidate {chosen_node_ip} (Score: {best_score:.2f} among {len(best_nodes)} choices)")
        return chosen_node_ip


    def schedule(self,
                 active_jobs: Dict[str, Any],
                 pending_jobs: List[Dict[str, Any]],
                 nodes: Dict[str, Any],
                 cluster_view: Any,
                 current_time: float
                ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]], Set[str]]:

        dispatch_decisions = {}
        migration_decisions = []
        jobs_to_cancel = set()

        # --- Preparation ---
        # Filter only nodes reported as 'active' for scheduling decisions
        active_nodes_map = {ip: info for ip, info in nodes.items() if info.get("status") == "active"}
        if not active_nodes_map:
            logger.debug("Llumnix: No active nodes available.")
            return {}, [], set()

        # Calculate current available slots based on active jobs snapshot
        # This represents slots free *before* this scheduling round starts.
        initial_available_slots = {}
        node_active_job_count = {ip: 0 for ip in active_nodes_map}
        for job in active_jobs.values():
             node_ip = job.get("node_ip")
             if node_ip in node_active_job_count:
                  node_active_job_count[node_ip] += 1

        for ip, info in active_nodes_map.items():
             capacity = self._get_node_capacity(info)
             running_on_node = node_active_job_count.get(ip, 0)
             initial_available_slots[ip] = max(0, capacity - running_on_node)
             logger.debug(f"Node {ip}: Capacity={capacity}, Running={running_on_node}, InitialSlots={initial_available_slots[ip]}") # <<< DEBUG >>>

        # Categorize and sort pending jobs
        pending_by_priority = {prio: [] for prio in [HIGH_PRIORITY, NORMAL_PRIORITY, LOW_PRIORITY]}
        for job in pending_jobs:
            prio_val = PRIORITY_MAP.get(job.get("priority", "normal").lower(), NORMAL_PRIORITY)
            pending_by_priority[prio_val].append(job)
        for prio in pending_by_priority:
            pending_by_priority[prio].sort(key=lambda j: j.get("submit_time", 0))

        # <<< FIX: Track assignments made *within this round* >>>
        assignments_this_round = {ip: 0 for ip in active_nodes_map}


        # --- Phase 1: Non-preemptive Dispatch (High Priority) ---
        logger.debug("Llumnix: Starting Phase 1 Dispatch (High Priority)")
        processed_job_ids = set() # Track jobs handled in P1

        # Create a mutable copy of available slots to decrement during this round
        current_round_available_slots = initial_available_slots.copy()

        for job in pending_by_priority[HIGH_PRIORITY]:
            job_id = job["job_id"]
            # Find best node considering current round's assignments
            target_node = self._find_best_node_for_job(job, active_nodes_map, active_jobs, current_round_available_slots)

            if target_node:
                dispatch_decisions[job_id] = target_node
                processed_job_ids.add(job_id)
                current_round_available_slots[target_node] -= 1 # Consume a slot for this round
                assignments_this_round[target_node] += 1      # Increment assignments count
                logger.info(f"Llumnix P1: Dispatching high-prio job {job_id} to {target_node} (Slots left: {current_round_available_slots[target_node]})")
            else:
                logger.debug(f"Llumnix P1: No suitable idle node found for high-prio job {job_id}")


        # --- Migration Decisions ---
        logger.debug("Llumnix: Evaluating Migration Decisions")
        if self.enable_migration:
            # <<< DEBUG: Use assignments_this_round in load calculation for migration decisions >>>
            overloaded_nodes = []
            underloaded_nodes = []
            for node_ip, info in active_nodes_map.items():
                 if info.get("status") != "active": continue # Should already be filtered, but double check
                 # Calculate load including tentative assignments this round
                 load = self._get_node_current_load_factor(node_ip, active_jobs, nodes, assignments_this_round)
                 # Calculate available slots *after* tentative assignments
                 current_slots = initial_available_slots.get(node_ip, 0) - assignments_this_round.get(node_ip, 0)

                 if load > MIGRATION_LOAD_THRESHOLD:
                      overloaded_nodes.append((load, node_ip))
                 # Check if node is underloaded AND has actual slots free *after* this round's P1 dispatches
                 elif load < MIGRATION_IDLE_THRESHOLD and current_slots > 0:
                      underloaded_nodes.append((load, node_ip))

            overloaded_nodes.sort(reverse=True)
            underloaded_nodes.sort()

            migrated_jobs_this_round = set()
            # Temporary list of targets to avoid assigning multiple migrations to the same node in one round
            targets_receiving_migration = set()

            for _, source_node_ip in overloaded_nodes:
                 if not underloaded_nodes: break

                 candidate_job_id: Optional[str] = None
                 candidate_prio = float('inf')
                 # Find lowest priority job on source node THAT IS NOT ALREADY MIGRATING
                 for job_id, details in active_jobs.items():
                      if details.get("node_ip") == source_node_ip and details.get("status") == "running": # Only migrate running jobs
                           if job_id in migrated_jobs_this_round: continue
                           prio = PRIORITY_MAP.get(details.get("priority", "normal").lower(), NORMAL_PRIORITY)
                           # Prefer migrating lower priority jobs
                           if prio > candidate_prio: # Higher numeric value = lower priority
                                candidate_prio = prio
                                candidate_job_id = job_id
                           elif candidate_job_id is None: # Take first one if no others found yet
                                candidate_prio = prio
                                candidate_job_id = job_id


                 if candidate_job_id and candidate_prio > HIGH_PRIORITY: # Avoid migrating high prio if possible
                      target_node_ip: Optional[str] = None
                      # Find best available underloaded node NOT already getting a migration
                      for load, ip in underloaded_nodes:
                           if ip not in targets_receiving_migration:
                                # TODO: Add model compatibility check here if needed
                                target_node_ip = ip
                                break # Take the first suitable one

                      if target_node_ip:
                           benefit = 1 # Simplified benefit
                           if benefit > MIGRATION_COST:
                                logger.info(f"Llumnix Migration: Planning to migrate job {candidate_job_id} (Prio {candidate_prio}) from {source_node_ip} to {target_node_ip}")
                                migration_decisions.append((candidate_job_id, source_node_ip, target_node_ip))
                                migrated_jobs_this_round.add(candidate_job_id)
                                targets_receiving_migration.add(target_node_ip) # Mark target as busy for migration this round
                                # Decrement target's available slots this round because it will receive a job
                                if target_node_ip in current_round_available_slots:
                                     current_round_available_slots[target_node_ip] -= 1
                           # else: logger.debug(...)
                      # else: logger.debug(...)
                 # else: logger.debug(...)


        # --- Phase 2: Preemptive/Migration-assisted Dispatch ---
        logger.debug("Llumnix: Starting Phase 2 Dispatch (Normal/Low Priority)")
        # Combine remaining Normal and Low priority jobs
        remaining_pending = [
             job for job in pending_by_priority[NORMAL_PRIORITY] if job["job_id"] not in processed_job_ids
        ] + [
             job for job in pending_by_priority[LOW_PRIORITY] if job["job_id"] not in processed_job_ids
        ]
        # Sort combined list by priority value then submit time
        remaining_pending.sort(key=lambda j: (PRIORITY_MAP.get(j.get("priority", "normal").lower(), NORMAL_PRIORITY), j.get("submit_time", 0)))

        for job in remaining_pending:
            job_id = job["job_id"]
            if job_id in dispatch_decisions: continue # Should not happen if processed_job_ids is used correctly

            # Find best node considering slots available after P1 dispatches and migrations
            target_node = self._find_best_node_for_job(job, active_nodes_map, active_jobs, current_round_available_slots)

            if target_node:
                dispatch_decisions[job_id] = target_node
                current_round_available_slots[target_node] -= 1 # Consume slot
                assignments_this_round[target_node] += 1
                logger.info(f"Llumnix P2: Dispatching job {job_id} to idle slot on {target_node} (Slots left: {current_round_available_slots[target_node]})")
            # else: logger.debug(f"Llumnix P2: No suitable node found for job {job_id}")
            # (Still skipping complex migration-for-dispatch logic)


        logger.info(f"Llumnix: Final Decisions - Dispatch={len(dispatch_decisions)}, Migrate={len(migration_decisions)}")
        return dispatch_decisions, migration_decisions, jobs_to_cancel

