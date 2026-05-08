#!/usr/bin/env bash
# Main FL compression sweep — ResNet-20, CIFAR-10, 3 seeds, 200 rounds.
# Runs: fp32_baseline, quant_8bit, quant_4bit, sz_eb0.001, sz_eb0.01
# Each compressor × 3 seeds = 15 runs. quant_4bit uses 1 seed (always collapses).
#
# Usage:
#   bash scripts/sweep_main.sh 2>&1 | tee sweep_main_log.txt

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

if command -v caffeinate &>/dev/null; then
    caffeinate -i -w $$ &
fi

ROUNDS=200
CLIENTS=10
ALPHA=0.5
LOCAL_EPOCHS=2
OUTPUT="results/resnet20_cifar10_main.csv"

echo "========================================"
echo " Main sweep: ResNet-20, CIFAR-10"
echo " Rounds: $ROUNDS | Clients: $CLIENTS | Alpha: $ALPHA | Seeds: 0,1,2"
echo " Output: $OUTPUT"
echo " Started: $(date)"
echo "========================================"

run_exp() {
    echo ""
    echo ">>> $* | $(date)"
    python fl-flower/run.py "$@" \
        --rounds "$ROUNDS" \
        --num-clients "$CLIENTS" \
        --alpha "$ALPHA" \
        --local-epochs "$LOCAL_EPOCHS" \
        --dataset cifar10 \
        --output "$OUTPUT"
    echo "<<< done at $(date)"
}

# Run each compressor across all seeds before moving on.
# This ensures at least one complete run per compressor if we crash midway.

for SEED in 0 1 2; do
    run_exp --compressor none              --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --compressor quantization --bits 8  --seed $SEED
done

# quant_4bit always collapses — one seed is enough to document it
run_exp --compressor quantization --bits 4 --seed 0

for SEED in 0 1 2; do
    run_exp --compressor sz --error-bound 0.001  --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --compressor sz --error-bound 0.01   --seed $SEED
done

echo ""
echo "========================================"
echo " Main sweep complete: $(date)"
echo " Results: $OUTPUT"
echo "========================================"
