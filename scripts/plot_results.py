"""
Report figures for fl-compression-study.
Run from repo root: python scripts/plot_results.py
Outputs 5 figures to results/report_fig*.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":    "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":        150,
})

PALETTE = {
    "fp32_baseline":     "#2c7bb6",
    "quant_8bit":        "#d7191c",
    "quant_4bit":        "#fdae61",
    "sz_schedule_a":     "#1a9641",
    "sz_usnr_rms_a0.2":  "#7b2d8b",
}
LABELS = {
    "fp32_baseline":     "FP32 baseline",
    "quant_8bit":        "Quant 8-bit",
    "quant_4bit":        "Quant 4-bit (collapsed)",
    "sz_schedule_a":     "SZ Schedule A (partial, 1 seed)",
    "sz_usnr_rms_a0.2":  "USNR-RMS α=0.2 (partial, 1 seed)",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_mixed(path):
    """Load CSV that may have rows with different column counts.
    Uses the header from line 0 as the canonical format; rows whose
    column count matches the 21-col USNR header are mapped to that instead.
    All other mismatched rows are skipped gracefully."""
    with open(path) as f:
        lines = f.readlines()
    old_hdr = lines[0].strip().split(",")
    new_hdr = (
        "timestamp,round,compressor,seed,num_clients,alpha,"
        "val_accuracy,val_loss,bytes_sent,compression_ratio,"
        "compress_time_s,decompress_time_s,"
        "usnr_alpha,usnr_eb_mean,usnr_eb_min,usnr_eb_max,"
        "usnr_rms_mean,usnr_rms_min,usnr_rms_max,"
        "compressed_tensors_count,uncompressed_tensors_count"
    ).split(",")
    rows = []
    for ln in lines[1:]:
        ln = ln.strip()
        if not ln:
            continue
        parts = ln.split(",")
        if len(parts) == len(old_hdr):
            rows.append(dict(zip(old_hdr, parts)))
        elif len(parts) == len(new_hdr):
            rows.append(dict(zip(new_hdr, parts)))
        # rows with any other count are silently skipped
    df = pd.DataFrame(rows)
    for c in ["round", "seed", "val_accuracy", "val_loss",
               "bytes_sent", "compression_ratio",
               "compress_time_s", "decompress_time_s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def smooth(series, w=5):
    return series.rolling(w, min_periods=1).mean()

def seed_mean(df, comp, col="val_accuracy", tail_rounds=20):
    sub = df[df["compressor"] == comp]
    vals = []
    for _, sg in sub.groupby("seed"):
        vals.append(sg.nlargest(tail_rounds, "round")[col].mean())
    return np.mean(vals), np.std(vals)

# ── load ──────────────────────────────────────────────────────────────────────
main  = load_mixed("results/resnet20_cifar10_main.csv")
usnr  = pd.read_csv("results/resnet20_cifar10_usnr.csv")
adapt = load_mixed("results/resnet20_cifar10_adaptive.csv")

fp32_ref = main[(main["compressor"] == "fp32_baseline") & (main["round"] > 0)]["bytes_sent"].mean()

# ══════════════════════════════════════════════════════════════════════════════
# Fig 1 — Convergence curves (FP32 + Quant8 all seeds; others 1 seed partial)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

for comp in ["fp32_baseline", "quant_8bit"]:
    color = PALETTE[comp]
    sub   = main[main["compressor"] == comp]
    max_r = int(sub["round"].max())
    grid  = np.arange(0, max_r + 1)
    interp_curves = []
    for seed, sg in sub.groupby("seed"):
        sg  = sg.sort_values("round")
        rm  = smooth(sg["val_accuracy"])
        ax.plot(sg["round"], rm, color=color, alpha=0.25, lw=1.0)
        interp_curves.append(np.interp(grid, sg["round"].values, rm.values))
    mean_curve = np.mean(interp_curves, axis=0)
    ax.plot(grid, mean_curve, color=color, lw=2.5, label=LABELS[comp])

# quant 4bit (1 seed, collapsed)
q4 = main[main["compressor"] == "quant_4bit"].sort_values("round")
ax.plot(q4["round"], smooth(q4["val_accuracy"]),
        color=PALETTE["quant_4bit"], lw=1.8, ls="--",
        label=LABELS["quant_4bit"])

# Schedule A partial
sa = adapt[adapt["compressor"] == "sz_schedule_a"].sort_values("round")
ax.plot(sa["round"], smooth(sa["val_accuracy"]),
        color=PALETTE["sz_schedule_a"], lw=2, ls="-.",
        label=LABELS["sz_schedule_a"])

# USNR partial
un = usnr.sort_values("round")
ax.plot(un["round"], smooth(un["val_accuracy"]),
        color=PALETTE["sz_usnr_rms_a0.2"], lw=2, ls=":",
        label=LABELS["sz_usnr_rms_a0.2"])

ax.axhline(80, color="grey", lw=0.7, ls="--", alpha=0.5)
ax.text(201, 80.4, "80 %", fontsize=8, color="grey", va="bottom")
ax.set_xlim(0, 200); ax.set_ylim(0, 95)
ax.set_xlabel("Communication round", fontsize=12)
ax.set_ylabel("Validation accuracy (%)", fontsize=12)
ax.set_title(
    "Figure 1 — Convergence curves  |  ResNet-20, CIFAR-10, Dirichlet α=0.5, 10 clients\n"
    "Solid bold = mean of 3 seeds   ·   faint lines = individual seeds   ·   "
    "dashed/dotted = partial single-seed runs",
    fontsize=10)
ax.legend(fontsize=10, loc="upper left")
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("results/report_fig1_convergence.png", dpi=150)
plt.close()
print("Saved report_fig1_convergence.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 2 — Accuracy vs Compression Ratio scatter
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))

# FP32
fp32_mu, fp32_sd = seed_mean(main, "fp32_baseline")
ax.errorbar(1.0, fp32_mu, yerr=fp32_sd, fmt="o", ms=12,
            color=PALETTE["fp32_baseline"], capsize=6, lw=2,
            label="FP32 baseline (3 seeds)", zorder=5)
ax.annotate("FP32 baseline", (1.0, fp32_mu), xytext=(1.1, fp32_mu - 2),
            fontsize=9, color=PALETTE["fp32_baseline"])

# Quant 8
q8_mu, q8_sd = seed_mean(main, "quant_8bit")
q8_ratio = main[(main["compressor"] == "quant_8bit") & (main["round"] > 0)]["compression_ratio"].mean()
ax.errorbar(q8_ratio, q8_mu, yerr=q8_sd, fmt="s", ms=12,
            color=PALETTE["quant_8bit"], capsize=6, lw=2,
            label="Quant 8-bit (3 seeds)", zorder=5)
ax.annotate("Quant 8-bit", (q8_ratio, q8_mu), xytext=(q8_ratio + 0.1, q8_mu - 2.5),
            fontsize=9, color=PALETTE["quant_8bit"])

# Quant 4 (1 seed, collapsed)
q4 = main[main["compressor"] == "quant_4bit"]
q4_ratio = q4[q4["round"] > 0]["compression_ratio"].mean()
q4_acc   = q4["val_accuracy"].max()
ax.scatter(q4_ratio, q4_acc, s=140, marker="^",
           color=PALETTE["quant_4bit"], zorder=5,
           label="Quant 4-bit — 1 seed (collapsed)", alpha=0.8)
ax.annotate("Quant 4-bit\n(collapsed)", (q4_ratio, q4_acc),
            xytext=(q4_ratio + 0.1, q4_acc + 2), fontsize=9,
            color=PALETTE["quant_4bit"])

# Schedule A partial
sa_acc   = sa.tail(10)["val_accuracy"].mean()
sa_ratio = sa[sa["round"] > 0]["compression_ratio"].mean()
ax.scatter(sa_ratio, sa_acc, s=140, marker="D",
           color=PALETTE["sz_schedule_a"], zorder=5,
           label="SZ Sched. A — 1 seed, ~100 rounds (partial)")
ax.annotate("SZ Sched. A\n(partial)", (sa_ratio, sa_acc),
            xytext=(sa_ratio + 0.1, sa_acc - 3), fontsize=9,
            color=PALETTE["sz_schedule_a"])

# USNR partial
usnr_bytes = usnr[usnr["round"] > 0]["bytes_sent"].mean()
usnr_ratio = fp32_ref / usnr_bytes
usnr_acc   = usnr.tail(10)["val_accuracy"].mean()
ax.scatter(usnr_ratio, usnr_acc, s=140, marker="P",
           color=PALETTE["sz_usnr_rms_a0.2"], zorder=5,
           label="USNR-RMS α=0.2 — 1 seed, ~60 rounds (partial)")
ax.annotate("USNR-RMS\n(partial)", (usnr_ratio, usnr_acc),
            xytext=(usnr_ratio + 0.1, usnr_acc - 3), fontsize=9,
            color=PALETTE["sz_usnr_rms_a0.2"])

ax.axhline(fp32_mu, color=PALETTE["fp32_baseline"], lw=1, ls="--", alpha=0.4)
ax.set_xlabel("Compression ratio  (uncompressed ÷ compressed bytes, uplink)", fontsize=12)
ax.set_ylabel("Plateau validation accuracy — mean of last 20 rounds (%)", fontsize=12)
ax.set_title(
    "Figure 2 — Accuracy vs. Compression Ratio\n"
    "Error bars = std across 3 seeds  ·  hollow markers = partial / single-seed",
    fontsize=10)
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim(0.3, 10)
ax.set_ylim(10, 93)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("results/report_fig2_accuracy_vs_ratio.png", dpi=150)
plt.close()
print("Saved report_fig2_accuracy_vs_ratio.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 3 — Uplink bytes per round (bar chart, per-client)
# ══════════════════════════════════════════════════════════════════════════════
NUM_CLIENTS = 10
entries = [
    ("FP32 baseline",          fp32_ref / NUM_CLIENTS / 1e6,                                    PALETTE["fp32_baseline"],    True),
    ("Quant 8-bit",            main[(main["compressor"]=="quant_8bit")&(main["round"]>0)]["bytes_sent"].mean() / NUM_CLIENTS / 1e6, PALETTE["quant_8bit"], True),
    ("Quant 4-bit",            main[(main["compressor"]=="quant_4bit")&(main["round"]>0)]["bytes_sent"].mean() / NUM_CLIENTS / 1e6, PALETTE["quant_4bit"], False),
    ("SZ Sched. A\n(partial)", sa[sa["round"]>0]["bytes_sent"].mean()   / NUM_CLIENTS / 1e6,   PALETTE["sz_schedule_a"],    False),
    ("USNR-RMS α=0.2\n(partial)", usnr_bytes                            / NUM_CLIENTS / 1e6,   PALETTE["sz_usnr_rms_a0.2"], False),
]

fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(len(entries))
bars = []
for i, (label, val, color, complete) in enumerate(entries):
    b = ax.bar(i, val, color=color,
               alpha=1.0 if complete else 0.65,
               edgecolor="white", width=0.55)
    bars.append(b[0])

for bar, (label, val, color, complete) in zip(bars, entries):
    suffix = "" if complete else "*"
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
            f"{val:.2f} MB{suffix}", ha="center", va="bottom",
            fontsize=10, color=color, fontweight="bold")

# compression ratio annotation
for bar, (label, val, color, _) in zip(bars, entries):
    ratio = (fp32_ref / NUM_CLIENTS / 1e6) / val if val > 0 else 1.0
    if ratio > 1.01:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() / 2,
                f"{ratio:.1f}×", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")

ax.set_xticks(xs)
ax.set_xticklabels([e[0] for e in entries], fontsize=10)
ax.set_ylabel("Average uplink bytes per round\nper client (MB)", fontsize=11)
ax.set_title(
    "Figure 3 — Communication cost per client per round\n"
    "White numbers = compression ratio vs FP32  ·  * = partial run",
    fontsize=10)
ax.set_ylim(0, 1.5)
ax.axhline(fp32_ref / NUM_CLIENTS / 1e6, color=PALETTE["fp32_baseline"],
           ls="--", lw=1, alpha=0.4)
ax.grid(True, axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig("results/report_fig3_bytes_per_client.png", dpi=150)
plt.close()
print("Saved report_fig3_bytes_per_client.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 4 — Plateau oscillation: FP32 vs Quant8 (rounds 150-200)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 5))

for comp in ["fp32_baseline", "quant_8bit"]:
    color = PALETTE[comp]
    sub   = main[main["compressor"] == comp]
    for seed, sg in sub.groupby("seed"):
        sg_p = sg[sg["round"] >= 150].sort_values("round")
        lbl  = LABELS[comp] if seed == 0 else "_"
        ax.plot(sg_p["round"], sg_p["val_accuracy"],
                color=color, lw=1.4, alpha=0.75, label=lbl)

fp32_p = mpatches.Patch(color=PALETTE["fp32_baseline"], label="FP32 baseline (3 seeds)")
q8_p   = mpatches.Patch(color=PALETTE["quant_8bit"],    label="Quant 8-bit (3 seeds)")
ax.legend(handles=[fp32_p, q8_p], fontsize=11)
ax.set_xlim(150, 200); ax.set_ylim(82, 92)
ax.set_xlabel("Communication round", fontsize=12)
ax.set_ylabel("Validation accuracy (%)", fontsize=12)
ax.set_title(
    "Figure 4 — Plateau oscillation (rounds 150–200)\n"
    "FP32 baseline vs 8-bit quantization, 3 seeds each",
    fontsize=10)
ax.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig("results/report_fig4_plateau_oscillation.png", dpi=150)
plt.close()
print("Saved report_fig4_plateau_oscillation.png")

# ══════════════════════════════════════════════════════════════════════════════
# Fig 5 — USNR-RMS: accuracy track + adaptive error bounds over rounds
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                gridspec_kw={"height_ratios": [2, 1]})

un_valid = usnr[usnr["round"] > 0].sort_values("round")
rounds   = un_valid["round"].values

# top panel: accuracy comparison
ax1.plot(rounds, smooth(un_valid["val_accuracy"]),
         color=PALETTE["sz_usnr_rms_a0.2"], lw=2.5, label="USNR-RMS α=0.2")
for comp, color, lbl in [
    ("fp32_baseline", PALETTE["fp32_baseline"], "FP32 s0"),
    ("quant_8bit",    PALETTE["quant_8bit"],    "Quant 8-bit s0"),
]:
    ref = main[(main["compressor"]==comp)&(main["seed"]==0)].sort_values("round")
    ref = ref[ref["round"] <= rounds[-1]]
    ax1.plot(ref["round"], smooth(ref["val_accuracy"]),
             color=color, lw=1.5, ls="--", alpha=0.6, label=lbl)

ax1.set_ylabel("Validation accuracy (%)", fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.25)
ax1.set_title(
    "Figure 5 — USNR-RMS: per-tensor adaptive error bounds\n"
    "Top: accuracy vs baselines (seed 0, partial)  ·  "
    "Bottom: SZ error bound range across tensor population per round",
    fontsize=10)

# bottom panel: adaptive error bound range
eb_mean = un_valid["usnr_eb_mean"].values
eb_min  = un_valid["usnr_eb_min"].values
eb_max  = un_valid["usnr_eb_max"].values
ax2.fill_between(rounds, eb_min, eb_max,
                 color=PALETTE["sz_usnr_rms_a0.2"], alpha=0.2,
                 label="eb range (min–max per round)")
ax2.plot(rounds, eb_mean,
         color=PALETTE["sz_usnr_rms_a0.2"], lw=2, label="eb mean")
ax2.set_ylabel("SZ error bound\n(absolute)", fontsize=10)
ax2.set_xlabel("Communication round", fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.25)

fig.tight_layout()
fig.savefig("results/report_fig5_usnr_adaptive_eb.png", dpi=150)
plt.close()
print("Saved report_fig5_usnr_adaptive_eb.png")

print("\nAll 5 report figures written to results/report_fig*.png")
