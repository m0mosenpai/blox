# schedulers/fifox_scheduler.py

import logging
from typing import Dict, List, Tuple, Any, Set
from .fifo_scheduler import FIFOScheduler # Inherit basic FIFO dispatch

logger = logging.getLogger(__name__)

DEFAULT_JOB_TIMEOUT_SEC = 300 # 5 minutes

class FIFOxScheduler(FIFOScheduler):
    """
    FIFO scheduler with job timeout/expiration.
    Identifies jobs running longer than the configured timeout.
    """
    def __init__(self, job_timeout_sec: int = DEFAULT_JOB_TIMEOUT_SEC, **kwargs):
        super().__init__(**kwargs)
        # Ensure timeout is available, falling back to default
        self.job_timeout_sec = getattr(self, 'job_timeout_sec', job_timeout_sec)
        if not isinstance(self.job_timeout_sec, (int, float)) or self.job_timeout_sec <= 0:
             logger.warning(f"Invalid job_timeout_sec ({self.job_timeout_sec}), using default {DEFAULT_JOB_TIMEOUT_SEC}s.")
             self.job_timeout_sec = DEFAULT_JOB_TIMEOUT_SEC
        else:
             logger.info(f"FIFOxScheduler initialized with timeout {self.job_timeout_sec}s.")


    def schedule(self,
                 active_jobs: Dict[str, Any],
                 pending_jobs: List[Dict[str, Any]],
                 nodes: Dict[str, Any],
                 cluster_view: Any,
                 current_time: float
                ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]], Set[str]]:

        # 1. Call base FIFO scheduler to get dispatch decisions
        dispatch_decisions, migration_decisions, _ = super().schedule(
            active_jobs, pending_jobs, nodes, cluster_view, current_time
        )
        # FIFOx also doesn't migrate

        # 2. Identify timed-out jobs
        jobs_to_cancel = set()
        for job_id, details in active_jobs.items():
            launch_time = details.get("launch_time")
            if launch_time is None:
                # Cannot check timeout if launch time is missing
                # logger.warning(f"FIFOx: Cannot check timeout for job {job_id}: missing 'launch_time'.")
                continue

            run_duration = current_time - launch_time
            if run_duration > self.job_timeout_sec:
                logger.warning(f"FIFOx: Job {job_id} running for {run_duration:.2f}s exceeds timeout ({self.job_timeout_sec}s). Marking for cancellation.")
                jobs_to_cancel.add(job_id)

        if jobs_to_cancel:
            logger.info(f"FIFOx: Identified {len(jobs_to_cancel)} jobs to cancel due to timeout.")

        # Return dispatch decisions and the set of jobs to cancel
        return dispatch_decisions, migration_decisions, jobs_to_cancel