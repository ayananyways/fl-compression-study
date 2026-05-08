#!/usr/bin/env bash
# Step 2: LR decay controls — ResNet-20, CIFAR-10, 3 seeds, 200 rounds.
# Purpose: isolate whether accuracy oscillation is from fixed LR or compression noise.
#
# Usage:
#   bash scripts/sweep_lr_decay.sh 2>&1 | tee sweep_lr_decay_log.txt

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
OUTPUT="results/resnet20_cifar10_lr_decay.csv"

echo "========================================"
echo " Step 2: LR Decay Controls"
echo " Rounds: $ROUNDS | Seeds: 0,1,2"
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
        --output "$OUTPUT" \
        --lr-decay
    echo "<<< done at $(date)"
}

for SEED in 0 1 2; do
    run_exp --compressor none              --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --compressor quantization --bits 8  --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --compressor sz --error-bound 0.001  --seed $SEED
done

echo ""
echo "========================================"
echo " LR decay sweep complete: $(date)"
echo " Results: $OUTPUT"
echo "========================================"
