#!/usr/bin/env python3
"""Experiment Manager: Launch and manage the 3-config comparison pipeline.

Usage:
    # 1. Run all 3 experiments sequentially
    python experiment_manager.py --project_dir /path/to/project --model_path /path/to/model --data_path /path/to/data --num_gpu 2

    # 2. Run only one config
    python experiment_manager.py --config vanilla --project_dir /path/to/project ...

    # 3. Skip running, only analyze existing logs
    python experiment_manager.py --skip_run --project_dir /path/to/project
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

EXPERIMENTS = {
    "vanilla": {
        "script": "01_run_vanilla.sh",
        "name": "Vanilla (No Speculation)",
        "short": "vanilla",
        "spec_decoding": False,
    },
    "spec_fixed": {
        "script": "02_run_spec_fixed.sh",
        "name": "Spec-RL (Fixed Window)",
        "short": "spec_fixed",
        "spec_decoding": True,
    },
    "spec_adaptive": {
        "script": "03_run_spec_adaptive.sh",
        "name": "Spec-RL (Adaptive Buckets)",
        "short": "spec_adaptive",
        "spec_decoding": True,
    },
}


def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def run_experiment(script_path, env_vars):
    """Run a single experiment script and return the log file path."""
    print_header(f"Running: {script_path.name}")
    log_dir = script_path.parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"{script_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    env = {**os.environ, **env_vars}
    cmd = ["bash", str(script_path)]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_file}")
    print(f"  Start: {datetime.now()}")
    print("-" * 50)

    try:
        with open(log_file, "w") as lf:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(script_path.parent.parent),
            )
            for line in iter(process.stdout.readline, b""):
                decoded = line.decode("utf-8", errors="replace")
                print(decoded, end="")
                lf.write(decoded)
                lf.flush()
            process.wait()

        if process.returncode != 0:
            print(f"\n[ERROR] Experiment failed with return code {process.returncode}")
            return None

        print(f"\n  End: {datetime.now()}")
        print(f"  Log saved: {log_file}")
        return str(log_file)
    except KeyboardInterrupt:
        print("\n[INTERRUPT] Experiment interrupted by user")
        return None
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return None


def find_latest_logs(project_dir):
    """Find the most recent log files for each experiment."""
    log_dir = Path(project_dir) / "logs"
    if not log_dir.exists():
        return {}

    logs = {}
    for key in EXPERIMENTS:
        pattern = f"{key}_*.log"
        matching = sorted(log_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if matching:
            logs[key] = str(matching[0])
    return logs


def main():
    parser = argparse.ArgumentParser(description="Spec-RL 3-Config Experiment Manager")
    parser.add_argument("--project_dir", type=str, required=True, help="Project root directory")
    parser.add_argument("--model_path", type=str, default=None, help="Path to base model")
    parser.add_argument("--data_path", type=str, default=None, help="Path to data directory")
    parser.add_argument("--num_gpu", type=int, default=2, help="Number of GPUs")
    parser.add_argument("--config", type=str, choices=list(EXPERIMENTS.keys()), default=None,
                        help="Run only one config (vanilla/spec_fixed/spec_adaptive)")
    parser.add_argument("--skip_run", action="store_true", help="Skip running experiments, only analyze")
    parser.add_argument("--analyze_only", action="store_true", help="Alias for --skip_run")
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    exp_dir = Path(__file__).parent

    print_header("Spec-RL Experiment Manager")
    print(f"  Project: {project_dir}")
    print(f"  Experiments dir: {exp_dir}")
    print(f"  Model: {args.model_path or '(use default)'}")
    print(f"  Data: {args.data_path or '(use default)'}")
    print(f"  GPUs: {args.num_gpu}")

    # Determine which configs to run
    if args.config:
        configs = [args.config]
    else:
        configs = list(EXPERIMENTS.keys())

    # ------------------------------------------------------------------
    # Phase 1: Run experiments
    # ------------------------------------------------------------------
    log_files = {}
    if not args.skip_run and not args.analyze_only:
        env_vars = {
            "NUM_GPU": str(args.num_gpu),
        }
        if args.model_path:
            env_vars["MODEL_PATH"] = args.model_path
        if args.data_path:
            env_vars["DATA_PATH"] = args.data_path

        for key in configs:
            exp = EXPERIMENTS[key]
            script_path = exp_dir / exp["script"]
            if not script_path.exists():
                print(f"[ERROR] Script not found: {script_path}")
                continue
            log_file = run_experiment(script_path, env_vars)
            if log_file:
                log_files[key] = log_file

        if not log_files:
            print("[FATAL] No experiments completed successfully")
            sys.exit(1)
    else:
        print("\n[SKIP] Running experiments skipped (--skip_run)")
        log_files = find_latest_logs(project_dir)
        print(f"  Found existing logs: {list(log_files.keys())}")

    # ------------------------------------------------------------------
    # Phase 2: Analyze
    # ------------------------------------------------------------------
    print_header("Analysis Phase")

    # Build available log map
    available_logs = find_latest_logs(project_dir)
    for key in configs:
        if key in log_files:
            available_logs[key] = log_files[key]

    if not available_logs:
        print("[ERROR] No log files found for analysis")
        sys.exit(1)

    # Run the analysis script
    analyze_script = exp_dir / "analyze_results.py"
    if analyze_script.exists():
        cmd = [
            sys.executable,
            str(analyze_script),
            "--project_dir", str(project_dir),
        ]
        for key, log_path in available_logs.items():
            cmd.extend([f"--log_{key}", log_path])
        print(f"  Running: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"[WARNING] Analysis script not found: {analyze_script}")
        print("  Run it manually after experiments complete.")

    print_header("Done")


if __name__ == "__main__":
    main()