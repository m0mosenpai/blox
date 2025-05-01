import copy
from typing import Any, Dict, List, Tuple
import pandas as pd


# Placeholder base class (or define a formal interface)
class BaseSchedulerPolicy:
    def __init__(self, cluster_state=None, job_state=None, **kwargs):
        self.cluster_state = cluster_state
        self.job_state = job_state
        # Store any other args passed via kwargs

    def schedule(self,
                 active_jobs: Dict[str, Any], # Current running jobs {job_id: details}
                 pending_jobs: List[Dict[str, Any]], # Pending jobs [details] sorted by priority?
                 nodes: Dict[str, Any], # Available nodes {node_ip: status/caps}
                 cluster_view: Any, # Overall cluster resource view from ClusterState
                 current_time: float
                ) -> Tuple[Dict[str, str], List[Tuple[str, str, str]]]:
        """
        The main scheduling logic method.

        Args:
            active_jobs: Snapshot of currently running jobs.
            pending_jobs: Snapshot of jobs waiting in the global queue.
            nodes: Snapshot of worker node status and capabilities.
            cluster_view: Snapshot of overall cluster resources.
            current_time: Current scheduler time (for time-sensitive policies).

        Returns:
            A tuple containing:
            - dispatch_decisions: Dict[job_id, node_ip] - Which pending job to send to which node.
            - migration_decisions: List[Tuple[job_id, from_node_ip, to_node_ip]] - Which running job to migrate.
        """
        raise NotImplementedError



class SchedulingPolicy(object):
    def __init__(self, args):
        pass

    @staticmethod
    def copy_arguments(function):
        def function_wrapper(self, job_state, cluster_state):
            return function(
                self,
                job_state.active_jobs,
                cluster_state.server_map,
                cluster_state.gpu_df,
                # **copy.deepcopy(kwargs)
            )

        return function_wrapper

    @staticmethod
    def copy_arguments_old(function):
        def function_wrapper(self, job_state, cluster_state):
            return function(
                self,
                copy.deepcopy(job_state.active_jobs),
                copy.deepcopy(cluster_state.server_map),
                copy.deepcopy(cluster_state.gpu_df),
                # **copy.deepcopy(kwargs)
            )

        return function_wrapper

    # the __func__ thing one needs to do for the same class
    # if you inherit it, it is not going to be the same.
    @copy_arguments.__func__
    def schedule(
        self, job_dict: dict, node_info: dict, gpu_df: pd.DataFrame, **kwargs
    ) -> dict:
        """
        Implement the scheduling mechanism

        Schedules job based on input.
        Args:
            job_dict : Original Job dict which we have sort of maintained
            node_info: Same dict as received from the node register.
            gpu_df: Contains GPU dataframe.

        Returns:
                "order_job" : Mandatory key, list of dicts of jobs in the
                                 in the order they are supposed to run.
                "run_all_jobs": Some scheduler will only output the jobs to
                                    run which will fit on the GPU or expecting
                                    to perform over subscription. While some
                                    will just sort the value in the order and
                                    return the whole job sorted back. (I am not
                                    sure if we need this)
                Per Job key optional keys:
                placement_locations : If the scheduler is making placement decisions too.
                Like Gandiva does, we expect them to add a additional key in
                case of the dictionary. with probable places where we should
                place the jobs.

                placement_preference: Further each job could have a placement preference. Like in
                case of tiresias. Therefore if the scheduler wants to provide a
                placement they can insert a key "placement_preference" for each
                job. If we do not find this key. We perform placement by our
                case. There are two kinds of placement we support

                additional_keys:
                In case the user writes a custom placement policy. In that case
                they can update custom metrics and pass them to the placement
                policy. From the scheduler
                And read it as they like. More on this case.
        """
        raise NotImplementedError()
