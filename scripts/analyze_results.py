"""
Step 4: Post-processing analysis of FL compression sweep results.

Reads all result CSVs and computes:
  - 5-round rolling mean/std of accuracy (per run)
  - Cumulative uplink bytes (client → server, where compression is applied)
  - Cumulative downlink bytes (server → client, always FP32 — no compression)
  - Round and cumulative MB at which accuracy first crosses 30%, 35%, 40%
  - Best accuracy per GB uplink transmitted (headline efficiency metric)

Communication cost note:
  Compression in this study is applied UPLINK only (client → server).
  Downlink (server → client) is always FP32: 98 MB per round per client.
  The paper's main communication-efficiency plot uses UPLINK bytes,
  since that is where the compression is applied. Both are reported here
  for completeness.

Usage:
  python scripts/analyze_results.py
  python scripts/analyze_results.py --csvs results/flower_cifar10_sweep.csv results/flower_cifar10_lr_decay.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np

_DEFAULT_CSVS = [
    "results/flower_cifar10_sweep.csv",
    "results/flower_cifar10_lr_decay.csv",
    "results/flower_cifar10_adaptive.csv",
]

_THRESHOLDS = [30.0, 35.0, 40.0]
_FP32_BYTES_PER_CLIENT = 98_980_240  # ~98 MB (SimpleCNN FP32 parameters)


def load_all(csv_paths):
    frames = []
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            df["source_file"] = os.path.basename(path)
            frames.append(df)
        else:
            print(f"[warn] not found: {path}")
    if not frames:
        sys.exit("No CSV files found.")
    return pd.concat(frames, ignore_index=True)


def analyze(df: pd.DataFrame, num_clients: int = 10) -> pd.DataFrame:
    rows = []
    for run_name, group in df.groupby("compressor"):
        g = group.sort_values("round").reset_index(drop=True)
        acc = g["val_accuracy"]

        # Rolling stats (min_periods=1 so early rounds still have a value)
        rolling_mean = acc.rolling(5, min_periods=1).mean()
        rolling_std  = acc.rolling(5, min_periods=1).std().fillna(0.0)

        # Cumulative uplink bytes (compressed: what clients actually sent)
        cum_uplink_mb   = g["bytes_sent"].cumsum() / 1e6

        # Cumulative downlink bytes (server always broadcasts FP32 to all clients)
        fp32_per_round  = _FP32_BYTES_PER_CLIENT * num_clients
        cum_downlink_mb = (pd.Series(range(1, len(g) + 1)) * fp32_per_round) / 1e6

        cum_total_mb    = cum_uplink_mb + cum_downlink_mb

        # Thresholds: round and cumulative uplink MB at first crossing
        threshold_rows = {}
        for t in _THRESHOLDS:
            idx = acc[acc >= t].index
            if len(idx) > 0:
                first = idx[0]
                threshold_rows[f"round_first_{int(t)}pct"]    = int(g.loc[first, "round"])
                threshold_rows[f"uplink_mb_at_{int(t)}pct"]   = round(float(cum_uplink_mb.iloc[first]), 1)
            else:
                threshold_rows[f"round_first_{int(t)}pct"]    = None
                threshold_rows[f"uplink_mb_at_{int(t)}pct"]   = None

        # Headline efficiency: peak accuracy per GB of uplink traffic
        total_uplink_gb = cum_uplink_mb.iloc[-1] / 1000
        peak_acc        = acc.max()
        eff             = peak_acc / total_uplink_gb if total_uplink_gb > 0 else 0.0

        row = {
            "run":                run_name,
            "rounds_completed":   len(g),
            "peak_accuracy":      round(peak_acc, 2),
            "final_acc_last5":    round(rolling_mean.iloc[-5:].mean(), 2),
            "final_std_last5":    round(rolling_std.iloc[-5:].mean(), 3),
            "mean_compression":   round(g["compression_ratio"].mean(), 2),
            "total_uplink_mb":    round(cum_uplink_mb.iloc[-1], 1),
            "total_downlink_mb":  round(cum_downlink_mb.iloc[-1], 1),
            "total_bytes_mb":     round(cum_total_mb.iloc[-1], 1),
            "acc_per_gb_uplink":  round(eff, 2),
            **threshold_rows,
        }
        rows.append(row)

        # Save per-run round-level detail
        g = g.copy()
        g["rolling_mean_acc"]  = rolling_mean.values
        g["rolling_std_acc"]   = rolling_std.values
        g["cum_uplink_mb"]     = cum_uplink_mb.values
        g["cum_downlink_mb"]   = cum_downlink_mb.values
        g["cum_total_mb"]      = cum_total_mb.values

        out_path = f"results/detail_{run_name}.csv"
        g.to_csv(out_path, index=False)

    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csvs", nargs="+", default=_DEFAULT_CSVS)
    p.add_argument("--num-clients", type=int, default=10)
    args = p.parse_args()

    df = load_all(args.csvs)
    print(f"Loaded {len(df)} rows across {df['compressor'].nunique()} runs.\n")

    summary = analyze(df, num_clients=args.num_clients)
    summary_path = "results/analysis_summary.csv"
    summary.to_csv(summary_path, index=False)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(summary.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print("Per-run detail CSVs saved to results/detail_<run>.csv")


if __name__ == "__main__":
    main()
