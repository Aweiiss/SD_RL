#!/bin/bash
# =============================================================================
# Wrapper script that properly captures logs for analyze_results.py
# Usage: bash run_with_log.sh <config_name>
#   config_name: vanilla | spec_fixed | spec_adaptive
# =============================================================================
set -e

CONFIG=${1:-"vanilla"}
PROJECT_DIR="/home/lingquh1xx/L2598/Temp/verl"
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# Map config name to script filename
declare -A SCRIPT_MAP
SCRIPT_MAP["vanilla"]="01_run_vanilla.sh"
SCRIPT_MAP["spec_fixed"]="02_run_spec_fixed.sh"
SCRIPT_MAP["spec_adaptive"]="03_run_spec_adaptive.sh"

SCRIPT_NAME=${SCRIPT_MAP[$CONFIG]:-""}
if [ -z "$SCRIPT_NAME" ]; then
    echo "ERROR: Unknown config '$CONFIG'"
    echo "Valid configs: vanilla, spec_fixed, spec_adaptive"
    exit 1
fi

SCRIPT_PATH="${PROJECT_DIR}/experiments/${SCRIPT_NAME}"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script not found: $SCRIPT_PATH"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/${CONFIG}_${TIMESTAMP}.log"

echo "Running config: ${CONFIG}"
echo "Script: ${SCRIPT_PATH}"
echo "Log will be saved to: ${LOG_FILE}"
echo "=================================================="

# Use script command to capture ALL output (including Ray distributed logs)
script -q -c "bash ${SCRIPT_PATH}" "${LOG_FILE}"

echo "=================================================="
echo "Log saved to: ${LOG_FILE}"
echo "To analyze: python3 analyze_results.py --project_dir ${PROJECT_DIR}"