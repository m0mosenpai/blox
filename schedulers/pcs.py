from .scheduler_policy import SchedulingPolicy
import pandas as pd
from operator import getitem
from fractions import Fraction as frac
import numpy as np
import math
from collections import deque
import time

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
        self.covariance_threshold = args.pcs_t
        self.weight_decay = args.pcs_w
        self.demand_cap = args.pcs_z

    @SchedulingPolicy.copy_arguments
    def schedule(
        self,
        job_dict: dict,
        node_info: dict,
        gpu_df: pd.DataFrame,
        global_placement_policy: Optional[str] = None,
    ) -> dict:
        for job in job_dict:
            if not job["demand_fn"]:
                job_dict[job]["demand_fn"] = lambda n: job_dict[job]["job_duration"] / n

        # sort jobs in ascending order of their demand(n) = T
        sorted_jobs = sorted(job_dict.items(), key=lambda x: x[1]["demand_fn"](job_dict[job]["job_gpu_demand"]))

        n = 0
        queue = [sorted_jobs[0]]
        demand_sum = 0
        demand_squared_sum = 0
        demand_mean = 0
        demand_variance = 0

        num_queues = 0
        buckets = []
        c_squared_history = []

        for i in range(len(sorted_jobs)):
            # include job in the queue
            n += 1

            # calculate running demand sum, mean and variance
            demand = sorted_jobs[i][1]
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
                # reset states and start a new queue
                n = 0
                queue = [sorted_jobs[i]]
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

        # get number of gpus to hand out
        free_gpus = self._get_free_gpus(gpu_df)

        # initial GPU allocations per-queue (floor to round the weights)
        ideal_allocs = [w * free_gpus for w in weights]
        allocs = [math.floor(x) for x in ideal_allocs]
        leftovers = free_gpus - sum(allocs)
        # distribute leftover jobs to gpus in descending order of fractional diff
        fractions = [((ideal_allocs[i] - allocs[i]), i) for i in range(len(allocs))]
        fractions.sort(key=lambda x: x[0], reverse=True)
        for i in range(leftovers):
            allocs[fractions[i][1]] += 1
        # sanity checks
        assert free_gpus == sum(allocs)

        # convert buckets to deques for efficient dispatching
        buckets_d = [deque(b) for b in buckets]
        remaining_allocs = allocs[:]

        # schedule jobs from FIFO queues in RR
        schedule_order = []
        while any(remaining_allocs):
            for qid, b in enumerate(buckets_d):
                if remaining_allocs[qid] > 0 and b:
                    job = b.popleft()
                    gpus = allocs[qid]

                    # cap each job to a pre-defined max
                    max_gpus = 1
                    for n in range(1, gpus + 1):
                        t_n = job[1]["demand_fn"](n)
                        z = job[1]["demand_fn"](1) / (n * t_n)
                        if z >= self.demand_cap:
                            max_gpus = n
                    # update job state to reflect new gpu demands and predicted JCT
                    job[1]["job_gpu_demand"] = max_gpus
                    job[1]["predicted_JCT"] = time.time() + job[1]["demand_fn"](max_gpus)
                    schedule_order.append(job)
                    remaining_allocs[qid] -= 1

        schedule_info = {
            "job_order": schedule_order,
            "run_all_jobs": True
        }
        return schedule_info

    def _get_free_gpus(self, gpu_df: pd.DataFrame):
        free_gpus = (
            gpu_df.loc[gpu_df["IN_USE"] == False]
            .groupby("Node_ID")["GPU_ID"]
            .apply(list)
            .to_dict()
        )
        number_free_gpus = sum([len(free_gpus[x]) for x in free_gpus])
        return number_free_gpus
