#!/usr/bin/env python3
"""Analyze and compare results from 3-config Spec-RL experiments.

Extracts metrics from training logs, computes improvements, and generates
comparison tables + visualizations.

Usage:
    python analyze_results.py --project_dir /path/to/project
    python analyze_results.py --log_vanilla vanilla.log --log_spec_fixed fixed.log --log_spec_adaptive adaptive.log
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

# Try importing matplotlib; if not available, skip plotting
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------

# Regex patterns for extracting metrics from veRL log output
PATTERNS = {
    # Training metrics
    "step": re.compile(r"training/global_step[^0-9]*(\d+)"),
    "epoch": re.compile(r"training/epoch[^0-9]*(\d+)"),

    # Step timing
    "step_time": re.compile(r"timing/step[^0-9]*([\d.]+)"),
    "gen_time": re.compile(r"timing/gen[^0-9]*([\d.]+)"),
    "reward_time": re.compile(r"timing/reward[^0-9]*([\d.]+)"),
    "old_log_prob_time": re.compile(r"timing/old_log_prob[^0-9]*([\d.]+)"),
    "update_actor_time": re.compile(r"timing/update_actor[^0-9]*([\d.]+)"),

    # Speculative decoding metrics
    "spec_window": re.compile(r"spec/window[^0-9]*([\d.]+)"),
    "spec_next_window": re.compile(r"spec/next_window[^0-9]*([\d.]+)"),
    "spec_action": re.compile(r"spec/awb_action\s*[:=]\s*(\w+)"),
    "spec_enabled": re.compile(r"spec/awb_enabled[^0-9]*([\d.]+)"),
    "spec_skip_ratio": re.compile(r"spec/skip_ratio[^0-9]*([\d.]+)"),
    "spec_cont_ratio": re.compile(r"spec/cont_ratio[^0-9]*([\d.]+)"),
    "spec_avg_cut_idx": re.compile(r"spec/avg_cut_idx[^0-9]*([\d.]+)"),
    "spec_avg_saved_tokens": re.compile(r"spec/avg_saved_tokens[^0-9]*([\d.]+)"),

    # Validation / critic metrics
    "val_reward": re.compile(r"val-core/[^/]+/reward/mean[^0-9]*([\d.]+)"),
    "val_acc": re.compile(r"val-core/[^/]+/acc/mean[^0-9]*([\d.]+)"),
    "val_aux_reward": re.compile(r"val-aux/[^/]+/reward/mean[^0-9]*([\d.]+)"),

    # Training reward
    "train_reward": re.compile(r"data_metrics/response_reward/mean[^0-9]*([\d.]+)"),
    "train_reward_max": re.compile(r"data_metrics/response_reward/max[^0-9]*([\d.]+)"),

    # Throughput
    "throughput_tokens": re.compile(r"throughout/total_token_per_second[^0-9]*([\d.]+)"),
    "throughput_samples": re.compile(r"throughout/sample_per_second[^0-9]*([\d.]+)"),
    "mfu_actor": re.compile(r"perf/mfu/actor[^0-9]*([\d.]+)"),
    "mfu_actor_infer": re.compile(r"perf/mfu/actor_infer[^0-9]*([\d.]+)"),
}


def parse_log_file(log_path):
    """Parse a training log file and extract per-step metrics.

    Returns a dict mapping metric_name -> list of (step, value) tuples.
    """
    path = Path(log_path)
    if not path.exists():
        print(f"[WARNING] Log file not found: {path}")
        return {}

    text = path.read_text()
    results = {}
    for metric_name, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        values = []
        for m in matches:
            try:
                v = float(m)
                values.append(v)
            except (ValueError, TypeError):
                continue
        if values:
            results[metric_name] = values

    return results


def align_data_by_step(parsed_data, steady_ratio=0.3):
    """Align metrics by step length and keep only steady-state tail."""
    if not parsed_data:
        return {}

    step_len = len(parsed_data.get("step", []))
    if step_len <= 0:
        return {k: v for k, v in parsed_data.items() if isinstance(v, list) and v}

    start_idx = int(step_len * (1.0 - steady_ratio))
    start_idx = max(0, min(start_idx, step_len - 1))

    aligned = {}
    for metric, values in parsed_data.items():
        if not isinstance(values, list) or not values:
            continue
        n = min(len(values), step_len)
        if n <= 0:
            continue
        aligned[metric] = values[start_idx:n]
    return aligned


def align_multi_configs(all_data, cfg_order=("vanilla", "spec_fixed", "spec_adaptive"), steady_ratio=0.3):
    """Trim configs to same steady-state step count for fair comparison."""
    prepared = {cfg: align_data_by_step(data, steady_ratio=steady_ratio) for cfg, data in all_data.items()}
    present = [cfg for cfg in cfg_order if cfg in prepared and prepared[cfg]]
    if len(present) < 2:
        return prepared

    step_lens = [len(prepared[cfg].get("step", [])) for cfg in present if prepared[cfg].get("step")]
    if not step_lens:
        return prepared

    common_len = min(step_lens)
    if common_len <= 0:
        return prepared

    for cfg in present:
        for metric, values in list(prepared[cfg].items()):
            if isinstance(values, list) and values:
                prepared[cfg][metric] = values[-common_len:]
    return prepared


def compute_statistics(values):
    """Compute mean, std, min, max, median for a list of values."""
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "count": len(arr),
    }


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compute_improvement(baseline_mean, target_mean):
    """Compute percentage improvement over baseline.

    For metrics where lower is better (time), positive = faster.
    For metrics where higher is better (throughput, score), positive = better.
    """
    if baseline_mean == 0:
        return 0.0
    return ((baseline_mean - target_mean) / baseline_mean) * 100.0


# Metrics where higher = better
HIGHER_IS_BETTER = {"train_reward", "val_reward", "val_acc", "throughput_tokens",
                     "throughput_samples", "mfu_actor", "mfu_actor_infer",
                     "spec_skip_ratio", "spec_avg_saved_tokens"}

# Metrics where lower = better
LOWER_IS_BETTER = {"step_time", "gen_time", "reward_time", "old_log_prob_time",
                    "update_actor_time"}


def compare_configs(baseline_data, target_data, baseline_name="vanilla", target_name="target"):
    """Compare target config against baseline and compute improvements."""
    # Only compare metrics present in both
    common_metrics = set(baseline_data.keys()) & set(target_data.keys())

    # Focus on key metrics
    key_metrics = ["step_time", "gen_time", "throughput_tokens", "throughput_samples",
                   "train_reward", "val_reward", "val_acc", "mfu_actor"]

    results = {}
    for metric in key_metrics:
        if metric not in common_metrics:
            continue
        base_stats = compute_statistics(baseline_data[metric])
        tgt_stats = compute_statistics(target_data[metric])

        # Improvement direction
        if metric in LOWER_IS_BETTER:
            # Lower is better: reduction %
            improvement = compute_improvement(base_stats["mean"], tgt_stats["mean"])
            improvement_abs = base_stats["mean"] - tgt_stats["mean"]
        else:
            # Higher is better: increase %
            improvement = compute_improvement(-base_stats["mean"], -tgt_stats["mean"])
            improvement_abs = tgt_stats["mean"] - base_stats["mean"]

        results[metric] = {
            "baseline_mean": base_stats["mean"],
            "baseline_std": base_stats["std"],
            "target_mean": tgt_stats["mean"],
            "target_std": tgt_stats["std"],
            "improvement_pct": improvement,
            "improvement_abs": improvement_abs,
        }

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_comparison_table(all_results):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  EXPERIMENT COMPARISON RESULTS")
    print("=" * 90)

    # Header
    print(f"\n  {'Metric':<28} {'Baseline':>14} {'Spec-Fixed':>14} {'Spec-Adaptive':>14} {'F-v-B':>8} {'A-v-B':>8}")
    print(f"  {'-'*28} {'-'*14} {'-'*14} {'-'*14} {'-'*8} {'-'*8}")

    metric_labels = {
        "step_time": "Step Time (s)",
        "gen_time": "Gen Time (s)",
        "throughput_tokens": "Token Throughput",
        "throughput_samples": "Sample Throughput",
        "train_reward": "Train Reward",
        "val_reward": "Val Reward",
        "val_acc": "Val Accuracy",
        "mfu_actor": "Actor MFU",
    }

    for metric, label in metric_labels.items():
        vals = []
        imps = []
        for cfg_name in ["vanilla", "spec_fixed", "spec_adaptive"]:
            if cfg_name not in all_results or metric not in all_results[cfg_name]:
                vals.append(None)
                imps.append(None)
            else:
                vals.append(all_results[cfg_name][metric]["mean"])

        if all(v is not None for v in vals):
            f_imp = ((vals[0] - vals[1]) / vals[0] * 100) if vals[0] != 0 else 0
            a_imp = ((vals[0] - vals[2]) / vals[0] * 100) if vals[0] != 0 else 0
            # For throughput/reward: invert direction
            if metric in HIGHER_IS_BETTER:
                f_imp = -f_imp
                a_imp = -a_imp

            def fmt(v):
                if v is None:
                    return "N/A"
                if isinstance(v, float):
                    return f"{v:.4f}"
                return str(v)

            print(f"  {label:<28} {fmt(vals[0]):>14} {fmt(vals[1]):>14} {fmt(vals[2]):>14} {f_imp:>+7.1f}% {a_imp:>+7.1f}%")


def print_spec_stats(spec_data, name):
    """Print speculative decoding specific stats."""
    print(f"\n  [{name}] Speculative Decoding Stats:")
    print(f"    {'Metric':<30} {'Mean':>10} {'Std':>10}")
    print(f"    {'-'*30} {'-'*10} {'-'*10}")

    for metric in ["spec_window", "spec_next_window", "spec_skip_ratio",
                    "spec_cont_ratio", "spec_avg_cut_idx", "spec_avg_saved_tokens"]:
        if metric in spec_data:
            stats = compute_statistics(spec_data[metric])
            print(f"    {metric:<30} {stats['mean']:>10.2f} {stats['std']:>10.2f}")


def generate_plots(all_data, output_dir):
    """Generate comparison plots."""
    if not HAS_PLOT:
        print("\n[WARNING] matplotlib not available, skipping plots")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = ["vanilla", "spec_fixed", "spec_adaptive"]
    colors = {"vanilla": "#e74c3c", "spec_fixed": "#3498db", "spec_adaptive": "#2ecc71"}
    labels = {"vanilla": "Vanilla", "spec_fixed": "Spec-Fixed", "spec_adaptive": "Spec-Adaptive"}

    # ---- Plot 1: Step Time over steps ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Spec-RL 3-Config Comparison", fontsize=14, fontweight="bold")

    plot_configs = [
        ("step_time", "Step Time (s)", 0, 0),
        ("gen_time", "Generation Time (s)", 0, 1),
        ("throughput_tokens", "Token Throughput (tok/s)", 0, 2),
        ("train_reward", "Train Reward", 1, 0),
        ("val_reward", "Val Reward", 1, 1),
        ("mfu_actor", "Actor MFU", 1, 2),
    ]

    for metric, title, row, col in plot_configs:
        ax = axes[row][col]
        for cfg in configs:
            data = all_data.get(cfg, {})
            if metric in data:
                vals = data[metric]
                steps = list(range(1, len(vals) + 1))
                ax.plot(steps, vals, label=labels[cfg], color=colors[cfg], alpha=0.7, linewidth=1)
        ax.set_xlabel("Step")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "comparison_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {plot_path}")

    # ---- Plot 2: Improvement bar chart ----
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute improvements
    baseline_data = all_data.get("vanilla", {})
    improvements = {}
    for cfg in ["spec_fixed", "spec_adaptive"]:
        cfg_data = all_data.get(cfg, {})
        improvements[cfg] = {}
        for metric in ["step_time", "gen_time", "throughput_tokens", "throughput_samples"]:
            if metric in baseline_data and metric in cfg_data:
                b = np.mean(baseline_data[metric])
                t = np.mean(cfg_data[metric])
                if metric in LOWER_IS_BETTER:
                    improvements[cfg][metric] = (b - t) / b * 100
                else:
                    improvements[cfg][metric] = (t - b) / b * 100

    metrics_plot = ["step_time", "gen_time", "throughput_tokens", "throughput_samples"]
    x = np.arange(len(metrics_plot))
    width = 0.35

    f_vals = [improvements["spec_fixed"].get(m, 0) for m in metrics_plot]
    a_vals = [improvements["spec_adaptive"].get(m, 0) for m in metrics_plot]

    ax.bar(x - width/2, f_vals, width, label="Spec-Fixed vs Vanilla", color=colors["spec_fixed"])
    ax.bar(x + width/2, a_vals, width, label="Spec-Adaptive vs Vanilla", color=colors["spec_adaptive"])

    ax.set_ylabel("Improvement (%)")
    ax.set_title("Relative Improvement over Vanilla")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics_plot], fontsize=9)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plot_path = output_dir / "improvement_bars.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {plot_path}")

    # ---- Plot 3: Spec window evolution (for adaptive) ----
    if "spec_window" in all_data.get("spec_adaptive", {}):
        fig, ax = plt.subplots(figsize=(10, 5))
        vals = all_data["spec_adaptive"]["spec_window"]
        steps = list(range(1, len(vals) + 1))
        ax.plot(steps, vals, color=colors["spec_adaptive"], linewidth=1.5, marker="o", markersize=3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Window Size (tokens)")
        ax.set_title("Adaptive Window Size Evolution")
        ax.grid(True, alpha=0.3)
        plot_path = output_dir / "adaptive_window.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {plot_path}")


def save_json_report(all_results, all_data, output_dir):
    """Save structured JSON report."""
    report = {
        "timestamp": str(Path().stat().st_mtime),
        "configs": {},
        "comparisons": {},
    }

    for cfg_name, data in all_data.items():
        stats = {k: compute_statistics(v) for k, v in data.items()}
        report["configs"][cfg_name] = stats

    # Spec-fixed vs vanilla
    if "vanilla" in all_data and "spec_fixed" in all_data:
        report["comparisons"]["spec_fixed_vs_vanilla"] = compare_configs(
            all_data["vanilla"], all_data["spec_fixed"], "vanilla", "spec_fixed"
        )
    # Spec-adaptive vs vanilla
    if "vanilla" in all_data and "spec_adaptive" in all_data:
        report["comparisons"]["spec_adaptive_vs_vanilla"] = compare_configs(
            all_data["vanilla"], all_data["spec_adaptive"], "vanilla", "spec_adaptive"
        )
    # Spec-adaptive vs spec-fixed
    if "spec_fixed" in all_data and "spec_adaptive" in all_data:
        report["comparisons"]["spec_adaptive_vs_spec_fixed"] = compare_configs(
            all_data["spec_fixed"], all_data["spec_adaptive"], "spec_fixed", "spec_adaptive"
        )

    output_path = Path(output_dir) / "report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  JSON report saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyze Spec-RL experiment results")
    parser.add_argument("--project_dir", type=str, required=True)
    parser.add_argument("--log_vanilla", type=str, default=None)
    parser.add_argument("--log_spec_fixed", type=str, default=None)
    parser.add_argument("--log_spec_adaptive", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    log_dir = project_dir / "logs"
    output_dir = Path(args.output_dir) if args.output_dir else (project_dir / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find log files
    logs = {}
    for key in ["vanilla", "spec_fixed", "spec_adaptive"]:
        explicit = getattr(args, f"log_{key}")
        if explicit and Path(explicit).exists():
            logs[key] = explicit
        else:
            # Auto-find latest
            if log_dir.exists():
                pattern = f"{key}_*.log"
                matching = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
                if matching:
                    logs[key] = str(matching[0])

    if not logs:
        print("[ERROR] No log files found")
        return

    print(f"\nFound logs: {logs}")

    # Parse all logs
    all_data = {}
    for cfg_name, log_path in logs.items():
        print(f"\n  Parsing: {cfg_name} → {log_path}")
        data = parse_log_file(log_path)
        all_data[cfg_name] = data
        print(f"    Metrics found: {list(data.keys())}")

    # Step-align data and keep steady-state tail for fair config comparison.
    all_data = align_multi_configs(all_data, steady_ratio=0.3)

    # Print spec-specific stats
    for cfg_name in ["spec_fixed", "spec_adaptive"]:
        if cfg_name in all_data:
            print_spec_stats(all_data[cfg_name], cfg_name)

    # Compute comparison stats
    all_results = {}
    for cfg_name in ["vanilla", "spec_fixed", "spec_adaptive"]:
        if cfg_name in all_data:
            all_results[cfg_name] = {k: compute_statistics(v) for k, v in all_data[cfg_name].items()}

    # Print table
    print_comparison_table(all_results)

    # Generate plots
    generate_plots(all_data, output_dir)

    # Save JSON
    save_json_report(all_results, all_data, output_dir)

    # Save summary text
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write("SPEC-RL EXPERIMENT COMPARISON\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Logs analyzed:\n")
        for k, v in logs.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        # Key improvements
        if "vanilla" in all_results:
            baseline = all_results["vanilla"]
            for cfg_name in ["spec_fixed", "spec_adaptive"]:
                if cfg_name in all_results:
                    f.write(f"\n[{cfg_name} vs vanilla]\n")
                    for metric in ["step_time", "gen_time", "throughput_tokens", "throughput_samples"]:
                        if metric in baseline and metric in all_results[cfg_name]:
                            b = baseline[metric]["mean"]
                            t = all_results[cfg_name][metric]["mean"]
                            if metric in LOWER_IS_BETTER:
                                imp = (b - t) / b * 100
                            else:
                                imp = (t - b) / b * 100
                            f.write(f"  {metric}: {imp:+.1f}% improvement\n")

    print(f"\n  Summary saved: {summary_path}")
    print(f"\n{'='*70}")
    print("  Analysis complete!")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
