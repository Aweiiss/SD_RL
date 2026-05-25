#!/usr/bin/env python3
"""Quick manual comparison tool for Spec-RL experiments.

Use this when log files are not available - manually input key metrics
from terminal output.
"""

import json
from pathlib import Path


def input_metrics(config_name):
    """Interactively input metrics from terminal output."""
    print(f"\n{'='*60}")
    print(f"  Input metrics for: {config_name}")
    print(f"{'='*60}")
    print("  (Copy values from your terminal output)")
    print()

    metrics = {}

    # Step time
    val = input(f"  timing_s/step (e.g., 71.79, 55.76): ").strip()
    metrics["step_time"] = float(val) if val else 0.0

    # Gen time
    val = input(f"  timing_s/gen (e.g., 37.97, 25.87): ").strip()
    metrics["gen_time"] = float(val) if val else 0.0

    # Throughput
    val = input(f"  perf/throughput (e.g., 1602.35, 2036.57): ").strip()
    metrics["throughput_tokens"] = float(val) if val else 0.0

    # Score/reward
    val = input(f"  critic/score/mean (e.g., -1.0, -0.996): ").strip()
    metrics["critic_score"] = float(val) if val else 0.0

    # Actor MFU
    val = input(f"  perf/mfu/actor (e.g., 0.1256, 0.1486): ").strip()
    metrics["mfu_actor"] = float(val) if val else 0.0

    return metrics


def compute_improvement(baseline, target, lower_is_better=False):
    """Compute percentage improvement."""
    if baseline == 0:
        return 0.0
    if lower_is_better:
        return (baseline - target) / baseline * 100
    return (target - baseline) / baseline * 100


def print_comparison(results):
    """Print formatted comparison table."""
    print("\n" + "="*70)
    print("  SPEC-RL EXPERIMENT COMPARISON")
    print("="*70)

    configs = list(results.keys())
    if len(configs) < 2:
        print("  Need at least 2 configs to compare!")
        return

    baseline_name = configs[0]
    baseline = results[baseline_name]

    # Header
    header = f"{'Metric':<25} {baseline_name:>12}"
    for cfg in configs[1:]:
        header += f" {cfg:>12} {cfg[:4]}_improve"
    print(header)
    print("-"*70)

    # Metrics to compare
    metrics = [
        ("Step Time (s)", "step_time", True),
        ("Gen Time (s)", "gen_time", True),
        ("Throughput (tok/s)", "throughput_tokens", False),
        ("Critic Score", "critic_score", False),
        ("Actor MFU", "mfu_actor", False),
    ]

    for label, key, lower_is_better in metrics:
        if key not in baseline:
            continue

        line = f"  {label:<25} {baseline.get(key, 0):>12.2f}"
        for cfg in configs[1:]:
            tgt_val = results[cfg].get(key, 0)
            imp = compute_improvement(baseline.get(key, 0), tgt_val, lower_is_better)
            line += f" {tgt_val:>12.2f} {imp:>+10.1f}%"
        print(line)

    print("="*70)
    print("  Note: improvement = % change vs baseline (vanilla)")
    print("        Positive = better (lower time or higher throughput/score)")
    print("="*70)


def save_results(results, output_dir):
    """Save results to JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "manual_comparison.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


def main():
    print("\n" + "="*70)
    print("  Spec-RL Quick Comparison Tool")
    print("  Manually input metrics from terminal output")
    print("="*70)

    results = {}

    # Baseline (vanilla)
    print("\n>>> First, input BASELINE (vanilla) metrics:")
    results["vanilla"] = input_metrics("vanilla")

    # Spec-fixed
    print("\n>>> Next, input Spec-Fixed metrics:")
    results["spec_fixed"] = input_metrics("spec_fixed")

    # Spec-adaptive
    print("\n>>> Finally, input Spec-Adaptive metrics:")
    results["spec_adaptive"] = input_metrics("spec_adaptive")

    # Print comparison
    print_comparison(results)

    # Save
    save_results(results, "/home/lingquh1xx/L2598/Temp/verl/analysis")


if __name__ == "__main__":
    main()
