import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Configuration ---
# Adjust lists based on experiments run and files generated
FIFO_LOGS = {
    "FIFO (Load 15)": "fifo_base_log.json",
    #"FIFO (Load 30)": "fifo_burst_log.json",
    #"FIFO (Prio)": "fifo_prio_log.json",
}
LLUMNIX_LOGS = {
    "Llumnix (Load 15)": "llumnix_stub_log.json",
    #"Llumnix (Load 30)": "llumnix_burst_log.json",
    #"Llumnix (Prio)": "llumnix_prio_log.json",
    #"Llumnix (NoMig L20)": "exp2_llumnix_no_mig_log.json",
    #"Llumnix (WithMig L20)": "exp2_llumnix_with_mig_log.json",
}
ROUNDROBIN_LOGS = {
    #"RR (Load 20)": "exp2_rr_log.json",
}
PAGING_LOGS = {
    #"Strict KV": "exp4_strict_kv_log.json",
    #"Paged KV": "exp4_paged_kv_log.json",
}


ALL_LOG_GROUPS = {
    "Baseline": FIFO_LOGS,
    "Llumnix": LLUMNIX_LOGS,
    "RoundRobin": ROUNDROBIN_LOGS,
    "Paging": PAGING_LOGS,
}

OUTPUT_DIR = "plots"
PLOT_LATENCY_CDF = True
PLOT_PRIORITY_LATENCY = False # Set to True if Exp 3 logs exist
PLOT_UTIL_BAR = True

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Data Loading & Processing ---
all_results = {} # Store metrics per run
all_latency_data = {} # Store latency lists per run
per_priority_latency = {} # Store {run_label: {prio: [latencies]}}

def calculate_metrics(label, filename):
    """Loads json log, calculates metrics, returns metrics dict and latency list"""
    try:
        with open(filename, 'r') as f:
            job_dict = json.load(f)
        df = pd.DataFrame.from_dict(job_dict, orient='index')
        print(f"Loaded {len(df)} records from {filename} for {label}")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None, []

    metrics = {}
    completed = df[
        (df['status'] == 'finished') &
        df['end_time'].notna() &
        df['submit_time'].notna()
    ].copy()

    if completed.empty:
        print(f"Warning: No completed jobs found for {label}")
        return {"throughput": 0, "avg_latency_ms": None, "p99_latency_ms": None, "completed": 0}, []

    completed['latency_ms'] = (completed['end_time'] - completed['submit_time']) * 1000
    latencies = completed['latency_ms'].tolist()

    # Basic Metrics
    metrics['completed'] = len(completed)
    if completed['submit_time'].max() > completed['submit_time'].min():
         # Use time range of completed jobs for throughput calc
         duration = completed['end_time'].max() - completed['submit_time'].min()
         metrics['throughput'] = metrics['completed'] / duration if duration > 0 else 0
    else:
         metrics['throughput'] = 0

    metrics['avg_latency_ms'] = np.mean(latencies)
    metrics['p50_latency_ms'] = np.percentile(latencies, 50)
    metrics['p90_latency_ms'] = np.percentile(latencies, 90)
    metrics['p99_latency_ms'] = np.percentile(latencies, 99)

    # Per-priority metrics
    if 'priority' in completed.columns:
         per_priority_latency[label] = {}
         for prio in completed['priority'].unique():
              prio_lats = completed[completed['priority'] == prio]['latency_ms'].tolist()
              if prio_lats:
                   per_priority_latency[label][prio] = prio_lats
                   # Add to main metrics dict too if desired
                   metrics[f'avg_lat_{prio}_ms'] = np.mean(prio_lats)
                   metrics[f'p99_lat_{prio}_ms'] = np.percentile(prio_lats, 99)

    return metrics, latencies


# Load data for all specified logs
for group_name, logs in ALL_LOG_GROUPS.items():
    for label, filename in logs.items():
        metrics, latencies = calculate_metrics(label, filename)
        if metrics:
            all_results[label] = metrics
        if latencies:
            all_latency_data[label] = latencies

# --- Plotting Functions ---

def plot_latency_cdf(data_dict, title_suffix="", filename="latency_cdf.pdf"):
    """Plots latency CDFs for multiple runs."""
    if not data_dict: return
    plt.figure(figsize=(8, 5))
    for label, latencies in data_dict.items():
        if not latencies: continue
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        plt.plot(sorted_latencies, cdf, label=f"{label} (Avg: {np.mean(latencies):.0f}ms)", linewidth=2)

    plt.xlabel("End-to-End Latency (ms)")
    plt.ylabel("CDF")
    plt.title(f"Latency Distribution {title_suffix}")
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xscale("log")
    min_lat = min((min(l) for l in data_dict.values() if l), default=1)
    plt.xlim(left=max(1, min_lat * 0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, filename)}")

def plot_priority_latency_cdfs(priority_data_dict, filename_prefix="latency_cdf_priority"):
    """Plots separate CDFs comparing priorities within each run."""
    if not priority_data_dict: return
    for run_label, prio_dict in priority_data_dict.items():
         plt.figure(figsize=(8, 5))
         has_data = False
         priorities = sorted(prio_dict.keys(), key=lambda p: PRIORITY_MAP.get(p, 99)) # Sort high->low
         for priority_level in priorities:
              latencies = prio_dict.get(priority_level)
              if not latencies: continue
              has_data = True
              sorted_latencies = np.sort(latencies)
              cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
              plt.plot(sorted_latencies, cdf, label=f"{priority_level} (Avg: {np.mean(latencies):.0f}ms)", linewidth=2)

         if has_data:
              plt.xlabel("End-to-End Latency (ms)")
              plt.ylabel("CDF")
              plt.title(f"Latency by Priority ({run_label})")
              plt.legend()
              plt.grid(True, which="both", ls="--", alpha=0.6)
              plt.xscale("log")
              plt.xlim(left=1)
              plt.tight_layout()
              filename = f"{filename_prefix}_{run_label.replace(' ','_').lower()}.pdf"
              plt.savefig(os.path.join(OUTPUT_DIR, filename))
              plt.close()
              print(f"Saved plot: {os.path.join(OUTPUT_DIR, filename)}")

def plot_metric_bars(results_dict, metric_key, title, ylabel, filename="metric_bar.pdf", lower_is_better=True):
    """Plots a bar chart comparing a specific metric across runs."""
    if not results_dict: return
    labels = list(results_dict.keys())
    values = [results_dict[label].get(metric_key) for label in labels]

    # Filter out None values
    valid_data = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not valid_data: return
    labels, values = zip(*valid_data)

    plt.figure(figsize=(max(6, len(labels) * 1.5), 5)) # Dynamic width
    bars = plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15, ha='right') # Rotate labels if long

    # Add values on bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom' if yval >= 0 else 'top', ha='center')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, filename)}")


# --- Generate Plots ---
if PLOT_LATENCY_CDF:
    # Compare baseline vs Llumnix stub
    plot_latency_cdf(
        {k: v for k, v in all_latency_data.items() if k in ["FIFO (Load 15)", "Llumnix (Load 15)"]},
        title_suffix="(Load 15)",
        filename="latency_cdf_base_vs_llumnix.pdf"
    )
    # Compare burst performance (if logs exist)
    # plot_latency_cdf(
    #     {k: v for k, v in all_latency_data.items() if k in ["FIFO (Burst)", "Llumnix (Burst)"]},
    #     title_suffix="(Load 30)",
    #     filename="latency_cdf_burst.pdf"
    # )
     # Compare migration effect (if logs exist)
    # plot_latency_cdf(
    #     {k: v for k, v in all_latency_data.items() if k in ["Llumnix (NoMig L20)", "Llumnix (WithMig L20)"]},
    #     title_suffix="(Load 20)",
    #     filename="latency_cdf_migration.pdf"
    # )

if PLOT_PRIORITY_LATENCY:
     plot_priority_latency_cdfs(per_priority_latency)

# Plot bar charts for key metrics
plot_metric_bars(all_results, 'avg_latency_ms', 'Average End-to-End Latency', 'Latency (ms)', 'avg_latency_bars.pdf')
plot_metric_bars(all_results, 'p99_latency_ms', 'P99 End-to-End Latency', 'Latency (ms)', 'p99_latency_bars.pdf')
plot_metric_bars(all_results, 'throughput', 'Throughput', 'Requests/sec', 'throughput_bars.pdf', lower_is_better=False)
# plot_metric_bars(all_results, 'avg_gpu_utilization_pct', 'Average GPU Utilization', 'Utilization (%)', 'utilization_bars.pdf', lower_is_better=False) # Need to add this metric first


print("Plotting script finished.")