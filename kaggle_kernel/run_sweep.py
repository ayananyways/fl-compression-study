"""
Kaggle T4 GPU sweep — FL Compression Study
Runs: quant_8bit x3 seeds, quant_4bit x1 seed, Step 2 (lr_decay) x9 runs
Results saved to /kaggle/working/results/

To download after completion:
    kaggle kernels output KAGGLE_USERNAME/fl-compression-quant -p kaggle_results/
"""

import os, subprocess, sys

REPO = "https://github.com/ayananyways/fl-compression-study.git"
WORK = "/kaggle/working"
REPO_DIR = f"{WORK}/fl-compression-study"
RESULTS = f"{REPO_DIR}/results"

# ── Install dependencies ──────────────────────────────────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "flwr[simulation]==1.8.0", "pysz", "pandas", "seaborn"], check=True)

# ── Clone repo ────────────────────────────────────────────────────────────────
if not os.path.exists(REPO_DIR):
    subprocess.run(["git", "clone", REPO, REPO_DIR], check=True)

os.chdir(REPO_DIR)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs("checkpoints/flower_cifar10", exist_ok=True)
sys.path.insert(0, REPO_DIR)

# ── Helper ────────────────────────────────────────────────────────────────────
def run(cmd):
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"WARNING: command exited with code {result.returncode} — continuing")

BASE = [sys.executable, "fl-flower/run.py",
        "--dataset", "cifar10",
        "--rounds", "200",
        "--num-clients", "10",
        "--alpha", "0.5",
        "--local-epochs", "2"]

MAIN_OUT    = f"{RESULTS}/resnet20_cifar10_main.csv"
LRDECAY_OUT = f"{RESULTS}/resnet20_cifar10_lr_decay.csv"

# ── Step 1 (partial): quant_8bit x3 seeds, quant_4bit x1 ────────────────────
print("\n" + "="*60)
print("  STEP 1: quant_8bit x3 seeds + quant_4bit x1 seed")
print("="*60)

for seed in [0, 1, 2]:
    run(BASE + ["--compressor", "quantization", "--bits", "8",
                "--seed", str(seed), "--output", MAIN_OUT])

run(BASE + ["--compressor", "quantization", "--bits", "4",
            "--seed", "0", "--output", MAIN_OUT])

# ── Step 2: LR decay controls ─────────────────────────────────────────────────
print("\n" + "="*60)
print("  STEP 2: LR decay controls (fp32+cosine, quant8+cosine)")
print("="*60)

for seed in [0, 1, 2]:
    run(BASE + ["--compressor", "none", "--lr-decay",
                "--seed", str(seed), "--output", LRDECAY_OUT])

for seed in [0, 1, 2]:
    run(BASE + ["--compressor", "quantization", "--bits", "8", "--lr-decay",
                "--seed", str(seed), "--output", LRDECAY_OUT])

print("\n" + "="*60)
print(f"  ALL DONE. Results in {RESULTS}")
print("="*60)

import pandas as pd
for f in [MAIN_OUT, LRDECAY_OUT]:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f"\n{f}:")
        print(df.groupby(["compressor","seed"])["round"].agg(["count","max"]).to_string())
