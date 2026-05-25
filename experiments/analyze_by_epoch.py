#!/usr/bin/env python3
"""Analyze Spec-RL results grouped by epoch.

Usage:
    python analyze_by_epoch.py <log_file>

Extracts timing metrics per step and groups by epoch to show
acceleration improvement across epochs.
"""

import re
import sys
from collections import defaultdict


def parse_log(log_path):
    """Parse veRL log and extract per-step metrics."""
    with open(log_path) as f:
        text = f.read()

    # Pattern: step:N - ... training/global_step:N - training/epoch:M
    step_pattern = re.compile(
        r'step:(\d+).*?'
        r'timing_s/gen:([\d.]+).*?'
        r'perf/throughput:([\d.]+).*?'
        r'training/global_step:(\d+).*?'
        r'training/epoch:(\d+)',
        re.DOTALL
    )

    steps = []
    for match in step_pattern.finditer(text):
        step_display, gen_time, throughput, global_step, epoch = match.groups()
        steps.append({
            'step_display': int(step_display),
            'gen_time': float(gen_time),
            'throughput': float(throughput),
            'global_step': int(global_step),
            'epoch': int(epoch),
        })

    return steps


def group_by_epoch(steps, steps_per_epoch=48):
    """Group steps by epoch."""
    epochs = defaultdict(list)
    for s in steps:
        epoch_num = s['global_step'] // steps_per_epoch
        epochs[epoch_num].append(s)
    return epochs


def analyze_epochs(epochs, baseline_gen_time=None):
    """Print per-epoch statistics."""
    print("\n" + "=" * 70)
    print("  SPEC-RL ANALYSIS BY EPOCH")
    print("=" * 70)

    print(f"\n  {'Epoch':<8} {'Steps':<8} {'Avg Gen(s)':<12} {'Avg Thruput':<14} {'Gen Save%':<10}")
    print("  " + "-" * 54)

    prev_avg_gen = None
    for epoch_num in sorted(epochs.keys()):
        steps = epochs[epoch_num]
        if not steps:
            continue

        avg_gen = sum(s['gen_time'] for s in steps) / len(steps)
        avg_throughput = sum(s['throughput'] for s in steps) / len(steps)
        step_range = f"{steps[0]['global_step']}-{steps[-1]['global_step']}"

        # Calculate improvement vs previous epoch
        if prev_avg_gen is not None:
            gen_save_pct = (prev_avg_gen - avg_gen) / prev_avg_gen * 100
        else:
            gen_save_pct = 0.0

        print(f"  {epoch_num:<8} {step_range:<8} {avg_gen:<12.2f} {avg_throughput:<14.1f} {gen_save_pct:>+9.1f}%")

        prev_avg_gen = avg_gen

    # Summary
    if baseline_gen_time and len(epochs) > 1:
        print(f"\n  {'='*70}")
        print(f"  ACCELERATION SUMMARY (vs Epoch 0 as baseline)")
        print(f"  {'='*70}")
        epoch0_avg = sum(s['gen_time'] for s in epochs[0]) / len(epochs[0])
        for epoch_num in sorted(epochs.keys())[1:]:
            steps = epochs[epoch_num]
            avg_gen = sum(s['gen_time'] for s in steps) / len(steps)
            improvement = (epoch0_avg - avg_gen) / epoch0_avg * 100
            print(f"  Epoch {epoch_num}: gen time {avg_gen:.2f}s ({improvement:+.1f}% vs baseline)")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_by_epoch.py <log_file> [steps_per_epoch]")
        sys.exit(1)

    log_path = sys.argv[1]
    steps_per_epoch = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    steps = parse_log(log_path)
    if not steps:
        print("No step metrics found in log file!")
        sys.exit(1)

    print(f"Found {len(steps)} steps in log file")

    epochs = group_by_epoch(steps, steps_per_epoch)
    analyze_epochs(epochs)


if __name__ == "__main__":
    main()
