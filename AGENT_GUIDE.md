# Agent Guide — FL Compression Study

This document is written for any agent (or human) picking up this project mid-stream.
Read it fully before touching anything.

---

## 1. What This Project Is

A federated learning compression study for a master's/research thesis.
The goal is to measure how different weight compression schemes affect model accuracy
and communication cost in a simulated FL environment, and to propose a novel
**plateau-triggered adaptive SZ scheduling strategy (Schedule C)** as a contribution.

**Research questions:**
1. Does lossy compression (quantization, SZ) hurt FL accuracy vs FP32 baseline?
2. Does cosine LR decay reduce accuracy oscillation at plateau under compression?
3. Does adaptive SZ error-bound scheduling (especially Schedule C) improve the
   accuracy–communication tradeoff vs fixed error bounds?

**Model:** ResNet-20 (272,474 params, ~1.09MB FP32)
**Dataset:** CIFAR-10, Dirichlet(α=0.5) non-IID partition across 10 clients
**Framework:** Flower (flwr) with Ray simulation — NOT torchrun/DDP
**Seeds:** 0, 1, 2 (for statistical validity)
**Rounds:** 200 per run

---

## 2. Repository Structure

```
fl-compression-study/
├── fl-flower/              ← ACTIVE codebase (use this, not src/)
│   ├── run.py              ← main entrypoint, all CLI flags
│   ├── client.py           ← FlowerClient: local training + compression
│   ├── strategy.py         ← LoggingFedAvg: CSV logging + checkpoints
│   ├── adaptive_strategy.py← AdaptiveSZStrategy: schedules A/B/C
│   ├── models.py           ← ResNet-20, get_parameters, set_parameters
│   └── data.py             ← Dirichlet partition, CIFAR-10/100 loaders
│
├── src/compressors/        ← Compressor implementations (used by fl-flower)
│   ├── no_compression.py   ← FP32 baseline (identity)
│   ├── quantization.py     ← Min-max uniform quantization (1/2/4/8/16-bit)
│   └── sz.py               ← SZ3 error-bounded compression (wraps pysz)
│
├── scripts/
│   ├── sweep_main.sh       ← Step 1: main compressor sweep
│   ├── sweep_lr_decay.sh   ← Step 2: LR decay controls
│   ├── sweep_adaptive.sh   ← Step 3: adaptive SZ scheduling
│   └── analyze_results.py  ← post-processing (run after all sweeps done)
│
├── results/
│   └── resnet20_cifar10_main.csv     ← ACTIVE results file (Step 1)
│   └── resnet20_cifar10_lr_decay.csv ← Step 2 results (written when ready)
│   └── resnet20_cifar10_adaptive.csv ← Step 3 results (written when ready)
│
├── checkpoints/
│   └── flower_cifar10/     ← per-run checkpoints (auto-managed, keep 2 latest)
│
└── plots/                  ← generated figures
```

**Important:** `src/` contains an older DDP/torchrun training stack. It is NOT used
for the main experiments. All active work goes through `fl-flower/` + `scripts/`.

---

## 3. Sweep Plan and Current Status

### Step 1 — Main Sweep (`sweep_main.sh`)
Output: `results/resnet20_cifar10_main.csv`

| Run | Seeds | Rounds | Status |
|-----|-------|--------|--------|
| fp32_baseline | 0,1,2 | 200 | ✅ COMPLETE |
| quant_8bit | 0,1,2 | 200 | 🔄 seed=0 in progress (~round 54) |
| quant_4bit | 0 only | 200 | ⏳ pending (always collapses, 1 seed enough) |
| sz_eb0.001 | 0,1,2 | 200 | ⏳ pending |
| sz_eb0.01 | 0,1,2 | 200 | ⏳ pending |

### Step 2 — LR Decay (`sweep_lr_decay.sh`)
Output: `results/resnet20_cifar10_lr_decay.csv`
Runs: fp32+cosine, quant8+cosine, sz0.001+cosine × 3 seeds × 200 rounds
Status: ⏳ pending (auto-starts after Step 1 via chain process)

### Step 3 — Adaptive SZ (`sweep_adaptive.sh`)
Output: `results/resnet20_cifar10_adaptive.csv`
Runs: schedule A, B, C, A+cosine × 3 seeds × 200 rounds
Status: ⏳ pending (auto-starts after Step 2 via chain process)

---

## 4. How to Run / Resume

### Check if sweep is running
```bash
pgrep -f "sweep_main.sh" && echo "running" || echo "dead"
tail -5 sweep_main_log.txt
wc -l results/resnet20_cifar10_main.csv
```

### Check what's completed
```python
import pandas as pd
df = pd.read_csv('results/resnet20_cifar10_main.csv')
print(df.groupby(['compressor','seed'])['round'].agg(['count','max']))
```

### Start / resume everything from scratch
```bash
source venv/bin/activate

# Start main sweep (auto-skips completed runs, auto-resumes from checkpoint)
nohup bash scripts/sweep_main.sh > sweep_main_log.txt 2>&1 &
MAIN_PID=$!
echo "Main PID: $MAIN_PID"

# Chain Steps 2 and 3 to start automatically after main
nohup bash -c "
while kill -0 $MAIN_PID 2>/dev/null; do sleep 30; done
echo '[chain] main done, starting lr_decay'
source venv/bin/activate
bash scripts/sweep_lr_decay.sh 2>&1 | tee sweep_lr_decay_log.txt
echo '[chain] lr_decay done, starting adaptive'
bash scripts/sweep_adaptive.sh 2>&1 | tee sweep_adaptive_log.txt
echo '[chain] ALL DONE'
" > sweep_chain_log.txt 2>&1 &
echo "Chain PID: $!"
```

### Run a single experiment manually
```bash
source venv/bin/activate
python fl-flower/run.py \
    --dataset cifar10 --compressor none --seed 0 \
    --rounds 200 --num-clients 10 --alpha 0.5 --local-epochs 2 \
    --output results/resnet20_cifar10_main.csv
```

### Stop everything safely
```bash
pkill -f "sweep_main.sh"; pkill -f "fl-flower/run.py"; pkill -9 -f "sweep_chain"
# Checkpoints are saved after every round — no work is lost
```

---

## 5. Critical Bugs Fixed (DO NOT REINTRODUCE)

### Bug 1 — BatchNorm running stats not aggregated (FIXED)
**Symptom:** val_loss explodes to 30+ after round 5, accuracy stuck at 10%.
**Cause:** `get_parameters()` used `model.parameters()` which excludes BN buffers
(`running_mean`, `running_var`). Server evaluated in `eval()` mode with stale
running stats (zeros/ones from init) → massive loss.
**Fix:** `get_parameters` now uses `model.state_dict()` (full state, float32 cast).
`set_parameters` restores original dtypes. See `fl-flower/models.py`.

### Bug 2 — Quantization range blown by num_batches_tracked (FIXED)
**Symptom:** quant_8bit diverges — accuracy 10%, NaN loss from round 1.
**Cause:** Compression flattened the ENTIRE state_dict, including `num_batches_tracked`
(integer value ~500+). This expanded the quantization range from [-0.5, 0.5] to
[-0.5, 500], so all actual weights mapped to the same bin → training failed.
**Fix:** Compression in `client.py fit()` uses ONLY `model.parameters()` (trainable
weights, not buffers). BN buffers returned uncompressed in the state_dict.
See `fl-flower/client.py`.

**Summary of the correct design:**
- `get_parameters` / `set_parameters`: use full `state_dict` (includes BN stats)
- Compression in `client.fit()`: applies to `model.parameters()` ONLY
- Returned `params_out`: full state_dict (compressed params + uncompressed buffers)

### Bug 3 — Smoke test checkpoints offset main sweep (FIXED)
**Symptom:** First run starts logging from round 3 instead of round 0.
**Cause:** A 3-round smoke test saved checkpoints; `find_checkpoint()` found them and
set `round_offset=3`.
**Fix:** Always delete checkpoint files before starting a fresh experiment.

---

## 6. Key Design Decisions

### Why ResNet-20 (not SimpleCNN)?
ResNet-20 is the standard FL benchmark for CIFAR-10 (He et al. 2016, ~272K params).
SimpleCNN was replaced because it was too small and not standard in the literature.
Results: fp32 plateau at 87.1% ± 0.8% across 3 seeds, consistent with published work.

### Why state_dict for FedAvg but parameters-only for compression?
FedAvg must aggregate BN running stats so the server model evaluates correctly.
Compression should only touch trainable parameters (what we're studying).
`num_batches_tracked` is an int64 that breaks quantization if included.

### Why 3 seeds?
Statistical validity. Cross-seed variation for fp32_baseline is ±0.8%, well within
a reportable range. Single-seed results would be anecdotal.

### Why 200 rounds?
Model converges at ~round 80 for all seeds. Rounds 80–200 give stable plateau
statistics (std ±0.3–0.9% within seed). 200 rounds is sufficient and standard.

### Why quant_4bit only 1 seed?
It always collapses to ~10% accuracy (catastrophic quantization error at 4-bit for
FL with non-IID data). One seed is enough to document the failure mode.

### Compression ratio formula
`compression_ratio = sum(original_bytes across clients) / sum(compressed_bytes across clients)`
- fp32_baseline: 1.0x (no compression)
- quant_8bit: ~4.0x (32/8 = 4x, exact)
- sz_eb0.001: variable (~6–8x expected)
- sz_eb0.01: variable (~8–12x expected)

### Uplink vs downlink
Only uplink is compressed (client→server). Downlink (server→client) is always FP32.
`bytes_sent` in the CSV = total uplink bytes across all clients per round.

---

## 7. Expected Results (Based on Literature + Early Data)

| Compressor | Expected plateau acc | Compression ratio | Key finding |
|---|---|---|---|
| fp32_baseline | 87.1% ± 0.8% | 1.0x | ✅ Confirmed |
| quant_8bit | ~86–87% | 4.0x | Near-lossless |
| quant_4bit | ~10% | 8.0x | Catastrophic collapse |
| sz_eb0.001 | ~86–87% | ~6–8x | Near-lossless, better ratio than quant_8bit |
| sz_eb0.01 | ~85–87% | ~8–12x | Small accuracy cost, high compression |
| sz_schedule_a | similar to sz_eb0.01 | variable | Step schedule |
| sz_schedule_b | similar | variable | Finer step schedule |
| sz_schedule_c | best tradeoff | variable | **Novel: plateau-triggered** |

---

## 8. Adaptive SZ Scheduling (The Novel Contribution)

Three schedules in `fl-flower/adaptive_strategy.py`:

- **Schedule A:** 2-stage step — eb=0.001 for rounds 1–50, eb=0.01 for rounds 51–200
- **Schedule B:** 3-stage step — eb=0.001→0.005→0.01 at rounds 34, 67
- **Schedule C (novel):** Plateau-triggered — starts at eb=0.001; if 5-round accuracy
  gain < 0.5%, step up to next eb level. Adapts to the model's actual convergence.

The hypothesis: high compression (large eb) is wasteful early when the model is
learning fast. Low compression (small eb) is wasteful late when the model has
plateaued. Schedule C automatically transitions at the right time.

---

## 9. Generating Plots and Analysis

Once all sweeps are done:
```bash
source venv/bin/activate
python scripts/analyze_results.py
```

For interim plots (only fp32 + whatever compressors have completed):
```bash
python3 -c "
import pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv('results/resnet20_cifar10_main.csv')
for name, g in df.groupby('compressor'):
    pivot = g.pivot_table(index='round', columns='seed', values='val_accuracy')
    m = pivot.mean(axis=1)
    plt.plot(m.index, m.values, label=name)
plt.legend(); plt.xlabel('Round'); plt.ylabel('Accuracy (%)')
plt.savefig('plots/interim.png', dpi=150)
print('saved plots/interim.png')
"
```

Early plots already generated:
- `plots/fig1_convergence.png` — fp32 learning curves + early quant_8bit comparison
- `plots/fig2_convergence_speed.png` — rounds to reach accuracy thresholds
- `plots/fig3_plateau_oscillation.png` — plateau oscillation (motivates Step 2+3)

---

## 10. Observations So Far (as of Step 1, ~20% complete)

1. **fp32_baseline plateau: 87.1% ± 0.8%** — matches published FedAvg + ResNet-20
   results for CIFAR-10 non-IID (α=0.5). Setup is validated.

2. **Convergence happens at ~round 80** for all seeds. Rounds 80–200 are plateau
   with ±1% oscillation from fixed LR — this is the motivation for Step 2 (LR decay).

3. **quant_8bit early trajectory matches fp32** — at round 37, quant_8bit is only
   ~3% behind fp32 at the same stage, with compression ratio exactly 4.0x.
   Strong early signal that 8-bit quantization is near-lossless for FL.

4. **Cross-seed reproducibility is good** — ±0.8% across seeds at plateau.
   This is tight enough for publishable statistical comparisons.

---

## 11. Environment

```bash
# Activate venv
source venv/bin/activate

# Key packages
# flwr (Flower FL framework)
# torch >= 2.0
# pysz (SZ3 Python wrapper — needed for sz compressor only)
# pandas, matplotlib, seaborn, numpy

# Python 3.11
# macOS (Apple Silicon) — no GPU, CPU only for local runs
# Estimated time: ~3 min/round on CPU
```

For GPU runs (Kaggle T4 or Colab):
- Clone fresh from GitHub (all fixes are committed)
- Mount Drive for persistent results/checkpoints
- Symlink results/ and checkpoints/ to Drive
- GPU fraction is auto-set: `1/num_clients` per client actor
- SZ compressor (pysz) may need special install — test before relying on it
- Timing metrics (compress_time_s) will differ from CPU — do NOT mix with local times
- Accuracy and compression_ratio results ARE mergeable across machines

---

## 12. Contact / Context

- Researcher: Ayan Ismayilova
- Project: 2-month research study on FL compression
- Main novel contribution: Schedule C (plateau-triggered adaptive SZ)
- Priority: correctness and reproducibility over speed
