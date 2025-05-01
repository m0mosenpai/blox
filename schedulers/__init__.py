import logging
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger(__name__)

# Import your scheduler implementations here
from .fifo_scheduler import FIFOScheduler # Simple baseline
from .fifox_scheduler import FIFOxScheduler
from .llumnix_scheduler import LlumnixScheduler # Your target scheduler
from .roundrobin_scheduler import RoundRobinScheduler
from .scheduler_policy import BaseSchedulerPolicy


# --- Registry and Factory ---

_POLICY_REGISTRY = {
    "fifo": FIFOScheduler,
    # Register your actual policies here once implemented:
    "llumnix": LlumnixScheduler,
    "fifox": FIFOxScheduler, # From "another output" - implement if needed
    "roundrobin": RoundRobinScheduler, 
}

def make_policy(name: str, **kwargs) -> BaseSchedulerPolicy:
    """Factory function to create a scheduler policy instance."""
    policy_class = _POLICY_REGISTRY.get(name.lower())
    if policy_class:
        logger.info(f"Creating scheduler policy '{name}'")
        return policy_class(**kwargs)
    else:
        logger.error(f"Unknown scheduler policy name: '{name}'. Available: {list(_POLICY_REGISTRY.keys())}")
        raise KeyError(f"Unknown scheduler policy name: '{name}'")