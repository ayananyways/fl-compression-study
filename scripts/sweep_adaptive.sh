#!/usr/bin/env bash
# Step 3: Adaptive SZ scheduling — ResNet-20, CIFAR-10, 3 seeds, 200 rounds.
# Schedules A/B are deterministic. Schedule C is plateau-triggered (novel contribution).
#
# Usage:
#   bash scripts/sweep_adaptive.sh 2>&1 | tee sweep_adaptive_log.txt

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
OUTPUT="results/resnet20_cifar10_adaptive.csv"

echo "========================================"
echo " Step 3: Adaptive SZ Scheduling"
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
        --output "$OUTPUT"
    echo "<<< done at $(date)"
}

for SEED in 0 1 2; do
    run_exp --schedule A          --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --schedule B          --seed $SEED
done

for SEED in 0 1 2; do
    run_exp --schedule C          --seed $SEED
done

# Combined best-case: Schedule A + cosine LR
for SEED in 0 1 2; do
    run_exp --schedule A --lr-decay  --seed $SEED
done

echo ""
echo "========================================"
echo " Adaptive sweep complete: $(date)"
echo " Results: $OUTPUT"
echo "========================================"
