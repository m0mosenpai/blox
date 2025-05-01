import numpy as np
import time
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

def summarize_inference_metrics(job_state: Any, cluster_state: Any, start_time: float, end_time: float) -> Dict[str, Any]:
    """
    Calculates standard inference performance metrics from JobState/ClusterState.

    Args:
        job_state: The JobState object (or similar dict structure) containing job history.
                   Needs methods like get_all_jobs() or similar.
                   Jobs should have 'submit_time', 'launch_time'?, 'end_time', 'status'.
        cluster_state: The ClusterState object (or similar). Needs methods to get
                       historical utilization data if available.
        start_time: The simulation/run start time for throughput calculation.
        end_time: The simulation/run end time.

    Returns:
        A dictionary containing calculated metrics.
    """
    logger.info("Calculating inference summary metrics...")
    metrics = {}
    run_duration = end_time - start_time
    if run_duration <= 0:
        logger.warning("Run duration is zero or negative, cannot calculate throughput.")
        return {"error": "Invalid run duration"}

    # Adapt this based on your actual JobState implementation
    try:
        all_jobs = job_state.get_all_jobs_history() # Assumes method exists
        if not all_jobs:
             logger.warning("No job history found in JobState.")
             return {"message": "No jobs processed."}
    except AttributeError:
        logger.error("JobState object does not have expected method 'get_all_jobs_history'. Cannot calculate metrics.")
        return {"error": "JobState API mismatch"}


    completed_jobs = [j for j in all_jobs if j.get("status") == "finished" and j.get("end_time") and j.get("submit_time")]
    failed_jobs = [j for j in all_jobs if j.get("status") == "failed"]
    # Consider pending/rejected jobs too if tracked

    # --- Latency ---
    if completed_jobs:
        # End-to-End Latency (from submission to end)
        e2e_latencies = [(j["end_time"] - j["submit_time"]) * 1000 for j in completed_jobs] # in ms
        metrics["avg_e2e_latency_ms"] = np.mean(e2e_latencies)
        metrics["p50_e2e_latency_ms"] = np.percentile(e2e_latencies, 50)
        metrics["p90_e2e_latency_ms"] = np.percentile(e2e_latencies, 90)
        metrics["p99_e2e_latency_ms"] = np.percentile(e2e_latencies, 99)
        metrics["max_e2e_latency_ms"] = np.max(e2e_latencies)

        # Optional: Service Latency (from launch to end) if launch_time is tracked
        # service_latencies = [(j["end_time"] - j["launch_time"]) * 1000 for j in completed_jobs if j.get("launch_time")]
        # if service_latencies:
        #     metrics["avg_service_latency_ms"] = np.mean(service_latencies)
        #     metrics["p99_service_latency_ms"] = np.percentile(service_latencies, 99)
    else:
        logger.warning("No completed jobs found to calculate latency metrics.")
        metrics["avg_e2e_latency_ms"] = None # Or 0?

    # --- Throughput ---
    num_completed = len(completed_jobs)
    metrics["completed_requests"] = num_completed
    metrics["throughput_req_per_sec"] = num_completed / run_duration

    # --- Success/Failure Rate ---
    total_processed = len(completed_jobs) + len(failed_jobs)
    metrics["failed_requests"] = len(failed_jobs)
    if total_processed > 0:
         metrics["success_rate"] = num_completed / total_processed
    else:
         metrics["success_rate"] = None # Or 1?

    # --- GPU Utilization (Placeholder) ---
    # This requires ClusterState to track historical GPU usage data.
    try:
        # avg_gpu_util = cluster_state.get_average_utilization(start_time, end_time) # Assumes method exists
        metrics["avg_gpu_utilization"] = "N/A (Requires ClusterState tracking)"
    except AttributeError:
         metrics["avg_gpu_utilization"] = "N/A (ClusterState API mismatch)"


    logger.info(f"Metrics Summary: {metrics}")
    return metrics