# ---------------------------------------------------------------------------------------------
# Blox Abstractions for PCS:
# ---------------------------------------------------------------------------------------------
#
# 1. JOB ADMISSION [NO CHANGES]:
#    -> Accept-All in FIFO initially (in fixed number of WFQ Queues)
#    -> Map each new job to one of the limited queues
#
# 2. JOB SCHEDULING [TO-DO]:
#    -> Weighted Fair Queuing (WFQ) - hybrid of Fair-Share and Priority Scheduling
#    -> Incoming jobs mapped to one of the queues based on expected demand (each queue is FIFO)
#    -> GPU-time to each queue is proportional to assigned weight of each queue
#    -> Jobs are only reordered against others in the same queue
#    -> Tuning num of queues vs weights yields in different optimizations
#
# 3. JOB PLACEMENT [NO CHANGES]:
#    -> Consolidated Placement
#
# 4. JOB PREEMPTION [NO CHANGES]:
#    -> Iteration-aware / Checkpointed
#    -> Pre-emption naturally constrained by scheduler design (limited queues)
#
# 5. JOB LAUNCH [NO CHANGES]:
#    -> Standard (invoking via CLI in Blox)
#
# 6. CLUSTER MANAGEMENT [NO CHANGES]:
#    -> PCS uses Ray (existing cluster manager)
#    -> No PCS-specific management. Can probably use whatever Blox has? or use Ray with Blox?
#
# 7. METRIC COLLECTION [TO-DO]:
#    -> Needs each job's demand function (feedback from jobs to estimate completion time)
#    -> Tracks job start, end times with per-iteration times
#    -> WFQ relies on metrics for prediction. Possible changes to enable this to happen?
#
# ---------------------------------------------------------------------------------------------

import os
import warnings
import sys
import argparse

warnings.simplefilter(action="ignore", category=FutureWarning)

import schedulers
from placement import placement
import admission_control
from blox import ClusterState, JobState, BloxManager
import blox.utils as utils


def parse_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser with arguments
    """
    parser.add_argument(
        "--scheduler", default="Pcs", type=str, help="Name of the scheduling strategy"
    )

    parser.add_argument(
        "--node-manager-port", default=50052, type=int, help="Node Manager RPC port"
    )
    parser.add_argument(
        "--central-scheduler-port",
        default=50051,
        type=int,
        help="Central Scheduler RPC Port",
    )

    parser.add_argument(
        "--simulator-rpc-port",
        default=50050,
        type=int,
        help="Simulator RPC port to fetch ",
    )

    parser.add_argument(
        "--scheduler-name",
        default="Pcs",
        type=str,
        help="Name of the scheduling strategy",
    )

    parser.add_argument(
        "--placement-name",
        default="Pcs",
        type=str,
        help="Name of the scheduling strategy",
    )

    parser.add_argument(
        "--acceptance-policy",
        default="accept_all",
        type=str,
        help="Name of acceptance policy",
    )

    parser.add_argument(
        "--plot", action="store_true", default=False, help="Plot metrics"
    )
    parser.add_argument(
        "--exp-prefix", type=str, help="Unique name for prefix over log files"
    )

    parser.add_argument("--load", type=int, help="Number of jobs per hour")

    parser.add_argument("--simulate", action="store_true", help="Enable Simulation")

    parser.add_argument(
        "--round-duration", type=int, default=300, help="Round duration in seconds"
    )
    parser.add_argument(
        "--start-id-track", type=int, default=3000, help="Starting ID to track"
    )
    parser.add_argument(
        "--stop-id-track", type=int, default=4000, help="Stop ID to track"
    )

    args = parser.parse_args()
    return args


def main(args):
    placement_policy = placement.JobPlacement(args)
    scheduling_policy = schedulers.Pcs(args)
    admission_policy = admission_control.acceptAll(args)
    if args.simulate:
        # for simulation we get the config from the simulator
        # The config helps in providing file names and intialize
        blox_instance = BloxManager(args)
        new_config = blox_instance.rmserver.get_new_sim_config()
        print(f"New config {new_config}")
        if args.scheduler_name == "":
            # terminate the blox instance before exiting
            # if no scheduler provided break
            blox_instance.terminate_server()
            print("No Config Sent")
            sys.exit()

        blox_instance.scheduler_name = new_config["scheduler"]
        blox_instance.load = new_config["load"]

        args.scheduler_name = new_config["scheduler"]
        args.load = new_config["load"]
        args.start_id_track = new_config["start_id_track"]
        args.stop_id_track = new_config["stop_id_track"]
        print(
            f"Running Scheduler {args.scheduler_name}\nLoad {args.load} \n Placement Policy {args.placement_name} \nAcceptance Policy {args.acceptance_policy} \nTracking jobs from {args.start_id_track} to {args.stop_id_track}"
        )
        blox_instance.reset(args)
        cluster_state = ClusterState(args)
        job_state = JobState(args)
        os.environ["sched_policy"] = args.scheduler_name
        os.environ["sched_load"] = str(args.load)
        simulator_time = 0
        while True:
            # get new nodes for the cluster
            if blox_instance.terminate:
                blox_instance.terminate_server()
                print("Terminate current config {}".format(args))
                break
            blox_instance.update_cluster(cluster_state)
            blox_instance.update_metrics(cluster_state, job_state)
            new_jobs = blox_instance.pop_wait_queue(args.simulate)
            # get simulator jobs
            accepted_jobs = admission_policy.accept(new_jobs, cluster_state, job_state)
            job_state.add_new_jobs(accepted_jobs)
            new_job_schedule = scheduling_policy.schedule(job_state, cluster_state)
            # prune jobs - get rid of finished jobs
            utils.prune_jobs(job_state, cluster_state, blox_instance)
            # perform scheduling
            new_job_schedule = scheduling_policy.schedule(job_state, cluster_state)
            # get placement
            to_suspend, to_launch = placement_policy.place(
                job_state, cluster_state, new_job_schedule
            )

            utils.collect_custom_metrics(
                job_state, cluster_state, {"num_preemptions": len(to_suspend)}
            )

            utils.collect_cluster_job_metrics(job_state, cluster_state)
            # check if we have finished every job to track
            utils.track_finished_jobs(job_state, cluster_state, blox_instance)
            # execute jobs
            blox_instance.exec_jobs(to_launch, to_suspend, cluster_state, job_state)
            # update time
            simulator_time += args.round_duration
            job_state.time += args.round_duration
            cluster_state.time += args.round_duration
            blox_instance.time += args.round_duration


if __name__ == "__main__":
    args = parse_args(
        argparse.ArgumentParser(description="Arguments for Starting the scheduler")
    )

    main(args)
