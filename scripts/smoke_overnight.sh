#!/usr/bin/env bash
# Overnight smoke tests — 50 rounds, seed 0, untried configs
# Order: quant8_cosine (~1.7 hrs) → sz_schedule_c (~6.7 hrs) → sz_schedule_b (~6.7 hrs)
# Total: ~15 hrs. quant8 + schedule_c should finish within a sleep session.
# Usage: bash scripts/smoke_overnight.sh 2>&1 | tee sweep_overnight_log.txt

set -euo pipefail
cd "$(dirname "$0")/.."
source venv/bin/activate

OUTPUT="results/resnet20_cifar10_smoke50.csv"
ROUNDS=50
SEED=0

BASE=(
    python fl-flower/run.py
    --dataset cifar10
    --rounds "$ROUNDS"
    --num-clients 10
    --alpha 0.5
    --local-epochs 2
    --seed "$SEED"
    --output "$OUTPUT"
)

log() { echo; echo "========================================"; echo " $*"; echo " $(date)"; echo "========================================"; }

log "Overnight smoke test started"
log "Configs: quant8_cosine → sz_schedule_c → sz_schedule_b"
log "Rounds: $ROUNDS | Seed: $SEED | Output: $OUTPUT"

# ── 1. Quant 8-bit + cosine LR decay ─────────────────────────────────────────
log "[1/3] quant8_cosine"
"${BASE[@]}" --compressor quantization --bits 8 --lr-decay
log "[1/3] quant8_cosine DONE"

# ── 2. SZ Schedule C (plateau-triggered — novel) ─────────────────────────────
log "[2/3] sz_schedule_c"
"${BASE[@]}" --compressor sz --schedule C
log "[2/3] sz_schedule_c DONE"

# ── 3. SZ Schedule B (3-stage step) ──────────────────────────────────────────
log "[3/3] sz_schedule_b"
"${BASE[@]}" --compressor sz --schedule B
log "[3/3] sz_schedule_b DONE"

log "All smoke tests complete"

# ── Quick summary ─────────────────────────────────────────────────────────────
python3 - <<'PYEOF'
import pandas as pd, numpy as np

df = pd.read_csv("results/resnet20_cifar10_smoke50.csv")

print("\n=== Overnight smoke test summary ===")
rows = []
for comp, grp in df.groupby("compressor"):
    peak   = grp["val_accuracy"].max()
    last5  = grp.nlargest(5, "round")["val_accuracy"].mean()
    max_r  = int(grp["round"].max())
    ratio  = grp[grp["round"] > 0]["compression_ratio"].mean()
    mb     = grp[grp["round"] > 0]["bytes_sent"].mean() / 10 / 1e6
    rows.append(dict(config=comp, rounds=max_r, peak_acc=round(peak,2),
                     last5_mean=round(last5,2), ratio=round(ratio,2),
                     MB_per_round=round(mb,3)))

out = pd.DataFrame(rows).sort_values("peak_acc", ascending=False)
print(out.to_string(index=False))
PYEOF
