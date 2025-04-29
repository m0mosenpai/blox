from .scheduler_policy import SchedulingPolicy
import pandas as pd
from operator import getitem
from fractions import Fraction as frac
import numpy as np
import math
from collections import deque

from typing import Optional

class Pcs(SchedulingPolicy):
    """
    Implements PCS-WFQ Scheduler
    """

    def __init__(self, args):
        """
        Use this to hold any extra state the scheduler wants to hold
        """
        self.metric_to_track = ["per_iter_time", "attained_service"]
        self.default_metric_value = [0, 0]
        self.covariance_threshold = getattr(args, "pcs_t")
        self.weight_decay = getattr(args, "pcs_w")
        # TO-DO
        # self.Z = getattr(args, "pcs_z")

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        # TO-DO: can be optionally provided by the user
        # TO-DO: demand can be handled inside JobState itself
        # get all jobs -> demand mapping (demand: n resources -> T execution time)
        jobs = []
        for job_id, job_info in job_dict.items():
            demand_map = job_info.get("demand_map")
            if demand_map:
                job_demand = demand_map.get(job_info.get("min_alloc", 1), demand_map[max(demand_map)])
            else:
                job_demand = job_info["tracked_metrics"].get("remaining_time", 0)
            jobs.append((job_id, job_demand))

        # sort jobs in ascending order of their demand
        jobs.sort(key=lambda x: x[1])

        n = 0
        queue = [jobs[0]]
        demand_sum = 0
        demand_squared_sum = 0
        demand_mean = 0
        demand_variance = 0

        num_queues = 0
        buckets = []
        c_squared_history = []

        for i in range(len(jobs)):
            # include job in the queue
            n += 1

            # calculate running demand sum, mean and variance
            demand = jobs[i][1]
            demand_sum += demand
            demand_squared_sum += (demand * demand)
            demand_mean = ((n - 1) * demand_mean + demand) / n
            demand_variance = (1 / n) * (demand_squared_sum + (n * (demand_mean * demand_mean)) - (demand_mean * 2.0 * demand_sum))

            # calculate C^2
            c_squared = demand_variance / (demand_mean * demand_mean)
            c_squared_history.append(c_squared)

            # split queue if C^2 exceeds T
            if (c_squared > self.covariance_threshold):
                buckets.append(queue[:-1])
                # threshold_queues.append(int(jobs[i - 1][1]))

                # reset states and start a new queue
                n = 0
                queue = [jobs[i]]
                demand_sum = 0
                demand_squared_sum = 0
                demand_mean = 0
                demand_variance = 0
                num_queues += 1

        # calculate weights for each queue
        weights = [np.exp(-1.0 * self.weight_decay * i) for i in range(num_queues)]
        weights = list(map(lambda w: w / sum(weights), weights))
        weights = list(map(lambda w: frac(round(w, 3)).limit_denominator(10000), weights))
        weights = list(map(lambda w: frac(w, sum(weights)), weights))
        # ensure all weights sum to exactly 1
        weights[-1] = frac(1, 1) - sum(weights[:-1])

        # sanity checks
        assert all(list(map(lambda w: w > 0 and w <= 1.0, weights)))
        assert np.isclose(float(sum(weights)), 1.0)

        # get gpu usage statistics
        num_gpus = len(gpu_df)
        used_gpus = 0
        for gpu in gpu_df:
            if gpu["IN_USE"]:
                used_gpus += 1

        # initial GPU allocations per-queue (floor to round the weights)
        ideal_allocs = [w * num_gpus for w in weights]
        allocs = [math.floor(x) for x in ideal_allocs]
        # redistribute leftover to gpu with largest fractional diff
        fractions = [((ideal_allocs[i] - allocs[i]), i) for i in range(len(allocs))]
        fractions.sort(key=lambda x: x[0], reverse=True)
        allocs[fractions[0][1]] += 1

        # sanity checks
        assert num_gpus == sum(allocs)

        # convert buckets to deques for efficient dispatching
        buckets_d = [deque(b) for b in buckets]
        remaining_allocs = allocs[:]

        # dispath RR over FIFO queues
        schedule_order = []
        while any(remaining_allocs):
            for qid, b in enumerate(buckets_d):
                if remaining_allocs[qid] > 0 and b:
                    schedule_order.append(b.popleft())
                    remaining_allocs[qid] -= 1

        schedule_info = {
            "job_order": schedule_order,
            "run_all_jobs": True
        }
        return schedule_info
