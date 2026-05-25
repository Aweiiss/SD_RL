#!/bin/bash
# =============================================================================
# Run all 3 configs sequentially with proper log capture
# =============================================================================
set -e

PROJECT_DIR="/home/lingquh1xx/L2598/Temp/verl"

echo "=========================================="
echo "  Running all 3 Spec-RL configs"
echo "=========================================="

for config in vanilla spec_fixed spec_adaptive; do
    echo ""
    echo ">>> Starting: $config"
    bash run_with_log.sh "$config"
    echo ">>> Finished: $config"
    echo ""
    
    # Optional: wait between runs for GPU cooldown
    # sleep 30
done

echo "=========================================="
echo "  All configs complete!"
echo "  Analyzing results..."
echo "=========================================="

python3 analyze_results.py --project_dir "${PROJECT_DIR}"
