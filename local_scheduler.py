import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Deque, Callable, Tuple
from collections import deque

logger = logging.getLogger(__name__)

class LocalScheduler(ABC):
    """Abstract base for node-local scheduling policies."""
    def __init__(self, node_id: str, num_gpus: int):
        self.node_id = node_id
        self.num_gpus = num_gpus
        self.free_gpus = set(range(num_gpus))
        # job_id -> gpu_id assigned (represents running or submitted to vLLM)
        self.running: Dict[str, int] = {}
        # FIFO queue of job_info dicts waiting for a GPU
        self.queued: Deque[Dict[str, Any]] = deque()
        # Callback to launch job (set by NodeManagerMain) -> takes (job_id, job_info, gpu_id)
        self._launch_cb: Optional[Callable[[str, Dict[str, Any], int], None]] = None

    def set_launch_callback(self, launch_cb: Callable[[str, Dict[str, Any], int], None]):
        self._launch_cb = launch_cb

    @abstractmethod
    def submit_request(self, job_info: Dict[str, Any]) -> bool:
        """
        Called by NodeManager when a new inference request arrives via LaunchJob.
        Adds the job to the local system (queued or running immediately).
        Return True if accepted locally, False if rejected (e.g., queue full).
        """
        pass

    @abstractmethod
    def schedule_next(self) -> None:
        """
        Internal method to try starting queued jobs on free GPUs.
        Should be called after submit_request and on_job_complete.
        """
        pass

    def on_job_complete(self, job_id: str) -> None:
        """
        Called by NodeManager when a job finishes (successfully or not).
        Frees the GPU and triggers scheduling of the next job(s).
        """
        if job_id in self.running:
            gpu_id = self.running.pop(job_id)
            self.free_gpus.add(gpu_id)
            logger.info(f"LocalScheduler: Freed GPU {gpu_id} from completed job {job_id}")
            # Try to schedule now that a GPU is free
            self.schedule_next()
        else:
            logger.warning(f"LocalScheduler: Job {job_id} not found in running state during completion.")

    @abstractmethod
    def choose_migration_candidate(self) -> Optional[Tuple[str, int]]:
        """
        Policy decision: Pick a running job_id and its gpu_id to migrate, or None.
        """
        pass

    def get_queue_depth(self) -> int:
        """Returns the current number of queued jobs."""
        return len(self.queued)

    def get_running_count(self) -> int:
        """Returns the current number of running jobs."""
        return len(self.running)


class FIFOLocalScheduler(LocalScheduler):
    """Simple FIFO local scheduler."""
    def __init__(self, node_id: str, num_gpus: int, max_queue_size: int = 100):
        super().__init__(node_id, num_gpus)
        self.max_queue_size = max_queue_size

    def submit_request(self, job_info: Dict[str, Any]) -> bool:
        job_id = job_info["job_id"]
        if len(self.queued) >= self.max_queue_size:
            logger.warning(f"LocalScheduler: Rejecting job {job_id} - queue full ({self.max_queue_size}).")
            return False # NAK - Queue is full

        logger.info(f"LocalScheduler: Accepted job {job_id} locally.")
        self.queued.append(job_info)
        self.schedule_next() # Try to schedule immediately if possible
        return True # ACK - Accepted into queue

    def schedule_next(self) -> None:
        """Try to start any queued jobs on free GPUs."""
        while self.free_gpus and self.queued:
            # Get next job from queue
            job_info = self.queued.popleft()
            job_id = job_info["job_id"]

            # Allocate a GPU
            gpu_id = self.free_gpus.pop()
            self.running[job_id] = gpu_id
            logger.info(f"LocalScheduler: Assigning job {job_id} to GPU {gpu_id}")

            # Hand off to NodeManagerMain._launch_inference via callback
            if self._launch_cb:
                try:
                    self._launch_cb(job_id, job_info, gpu_id)
                except Exception as e:
                    logger.error(f"LocalScheduler: Error during launch callback for job {job_id}: {e}", exc_info=True)
                    # Put GPU back and potentially re-queue or fail the job
                    self.free_gpus.add(gpu_id)
                    del self.running[job_id]
                    # Re-queue for simplicity, though could also mark as failed
                    self.queued.appendleft(job_info)
                    break # Stop scheduling for now
            else:
                logger.error(f"LocalScheduler: Launch callback not set! Cannot launch job {job_id}.")
                # Put GPU back and fail the job?
                self.free_gpus.add(gpu_id)
                del self.running[job_id]
                # Don't re-queue if callback is missing

    def choose_migration_candidate(self) -> Optional[Tuple[str, int]]:
        """Pick the running job with the numerically smallest job_id (or oldest)."""
        if not self.running:
            return None
        # Find the job_id that corresponds to the minimum key when keys are treated numerically if possible
        # Simple approach: just get the first one added (dict iteration order ~= insertion order in Python 3.7+)
        # Or sort keys if job_ids are meaningful timestamps/integers
        try:
           # If job IDs are like UUIDs, sorting isn't meaningful. Pick oldest inserted.
           candidate_job_id = next(iter(self.running))
           candidate_gpu_id = self.running[candidate_job_id]
           logger.info(f"LocalScheduler: Choosing job {candidate_job_id} on GPU {candidate_gpu_id} as migration candidate.")
           return candidate_job_id, candidate_gpu_id
        except StopIteration:
            return None


# Factory function (can add more types later)
def get_local_scheduler(mode: str, node_id: str, num_gpus: int, max_queue_size: int) -> LocalScheduler:
    mode_lower = mode.lower()
    if mode_lower == "fifo":
        return FIFOLocalScheduler(node_id, num_gpus, max_queue_size)
    # Add other modes like 'strict', 'priority' here
    # elif mode_lower == "strict":
    #    return StrictLocalScheduler(...)
    else:
        logger.warning(f"Unknown local scheduler mode '{mode}'. Defaulting to FIFO.")
        return FIFOLocalScheduler(node_id, num_gpus, max_queue_size)
