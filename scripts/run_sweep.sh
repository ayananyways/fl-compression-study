#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_FILE="$ROOT_DIR/sweep_log.txt"
NPROC=4

CONFIGS=(
    "configs/exp_baseline.yaml"
    "configs/exp_quant_sweep.yaml"
    "configs/exp_sz_sweep.yaml"
    "configs/exp_hybrid.yaml"
    "configs/exp_scale.yaml"
)

echo "Sweep started at $(date)" | tee -a "$LOG_FILE"

for config in "${CONFIGS[@]}"; do
    full_config="$ROOT_DIR/$config"
    echo "---" | tee -a "$LOG_FILE"
    echo "Starting: $config at $(date)" | tee -a "$LOG_FILE"

    torchrun \
        --nproc_per_node="$NPROC" \
        "$SCRIPT_DIR/train.py" \
        --config "$full_config" \
        2>&1 | tee -a "$LOG_FILE"

    echo "Finished: $config at $(date)" | tee -a "$LOG_FILE"
done

echo "---" | tee -a "$LOG_FILE"
echo "Sweep completed at $(date)" | tee -a "$LOG_FILE"
