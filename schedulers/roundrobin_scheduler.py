# schedulers/roundrobin_scheduler.py
import logging
from typing import Dict, List, Tuple, Any, Set
from .scheduler_policy import BaseSchedulerPolicy

logger = logging.getLogger(__name__)

class RoundRobinScheduler(BaseSchedulerPolicy):
    """Assigns jobs to active nodes in simple round-robin order."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.node_index = 0
        logger.info("RoundRobinScheduler initialized.")

    def schedule(self, active_jobs, pending_jobs, nodes, cluster_view, current_time):
        dispatch_decisions = {}
        migration_decisions = []
        jobs_to_cancel = set()

        active_node_ips = sorted([ip for ip, info in nodes.items() if info.get("status") == "active"])

        if not active_node_ips:
            return {}, [], set()

        num_active_nodes = len(active_node_ips)
        pending_jobs_sorted = sorted(pending_jobs, key=lambda j: j.get("submit_time", 0))

        # Assign jobs round-robin without checking load
        for job in pending_jobs_sorted:
            target_node_ip = active_node_ips[self.node_index % num_active_nodes]
            dispatch_decisions[job["job_id"]] = target_node_ip
            logger.debug(f"RoundRobin: Assigning job {job['job_id']} to node {target_node_ip} (Index {self.node_index})")
            self.node_index += 1
            # Limit dispatches per round? Maybe dispatch only N jobs?
            # For simplicity, try dispatching all pending. Node will NAK if full.

        return dispatch_decisions, migration_decisions, jobs_to_cancel