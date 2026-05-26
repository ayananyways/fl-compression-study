"""
Adaptive SZ error-bound scheduling strategies.

Schedule A: two-stage step  — eb=0.001 rounds 1-50, eb=0.01 rounds 51-100
Schedule B: three-stage step — eb=0.001 → 0.005 → 0.01 at rounds 34, 67
Schedule C: plateau-triggered — start eb=0.001; if 5-round accuracy gain < 0.5%,
            step up to next level (cap at 0.01). This is the novel contribution.

All schedules can be combined with cosine LR decay (lr_decay=True).
Results include a `current_eb` column so the schedule can be reconstructed.
"""

import csv
import os
from typing import List

import flwr as fl
from flwr.common import FitIns

from strategy import LoggingFedAvg, _CSV_COLUMNS, find_checkpoint

_CSV_COLUMNS_ADAPTIVE = _CSV_COLUMNS + ["current_eb"]  # _CSV_COLUMNS already includes "seed"

_EB_LEVELS = [0.001, 0.005, 0.01]


class AdaptiveSZStrategy(LoggingFedAvg):
    """
    Extends LoggingFedAvg with per-round SZ error-bound adaptation.
    The client reads `eb` from the fit config and overrides its compressor.
    """

    _csv_columns = _CSV_COLUMNS_ADAPTIVE

    def __init__(self, schedule: str, **kwargs) -> None:
        assert schedule in ("A", "B", "C"), f"Unknown schedule: {schedule}"
        super().__init__(**kwargs)
        self.schedule = schedule
        self.current_eb: float = _EB_LEVELS[0]
        self.recent_accuracies: List[float] = []
        self._last_step_round: int = -1  # round when eb last stepped up

    # ── Config injection (called every round before clients train) ────────────

    def _fit_config(self, server_round: int) -> dict:
        self._maybe_step_eb(server_round)
        config = super()._fit_config(server_round)
        config["eb"] = self.current_eb
        return config

    def _maybe_step_eb(self, server_round: int) -> None:
        if self.schedule == "A":
            self.current_eb = 0.001 if server_round <= 50 else 0.01

        elif self.schedule == "B":
            if server_round <= 33:
                self.current_eb = 0.001
            elif server_round <= 66:
                self.current_eb = 0.005
            else:
                self.current_eb = 0.01

        elif self.schedule == "C":
            # 10-round window, 1.0% threshold, 15-round cooldown between steps.
            # - 10-round window: active convergence gains are typically 2-10%,
            #   so noise that looks like a plateau over 5 rounds won't fool this.
            # - 15-round cooldown: gives the model time to adapt to the new
            #   error bound before we evaluate the next plateau, preventing
            #   cascading triggers.
            # None of these numbers depend on knowing when the plateau occurs.
            rounds_since_step = server_round - self._last_step_round
            if len(self.recent_accuracies) >= 10 and rounds_since_step >= 15:
                gain = self.recent_accuracies[-1] - self.recent_accuracies[-10]
                if gain < 1.0 and self.current_eb < _EB_LEVELS[-1]:
                    idx = _EB_LEVELS.index(self.current_eb)
                    old_eb = self.current_eb
                    self.current_eb = _EB_LEVELS[idx + 1]
                    self._last_step_round = server_round
                    print(
                        f"  [AdaptiveSZ-C] round {server_round}: plateau "
                        f"(10-round gain={gain:.2f}%) → eb {old_eb} → {self.current_eb}"
                    )

    # ── Track accuracy for Schedule C plateau detection ───────────────────────

    def log_round(self, server_round: int, val_accuracy: float, val_loss: float) -> None:
        self.recent_accuracies.append(val_accuracy)

        actual_round = server_round + self.round_offset
        m = self.round_fit_metrics.get(server_round, {})

        from datetime import datetime, timezone
        row = {
            "timestamp":         datetime.now(timezone.utc).isoformat(),
            "round":             actual_round,
            "compressor":        self.compressor_name,
            "seed":              self.seed,
            "num_clients":       self.num_clients,
            "alpha":             self.alpha,
            "val_accuracy":      val_accuracy,
            "val_loss":          val_loss,
            "bytes_sent":        m.get("bytes_sent", 0),
            "compression_ratio": m.get("compression_ratio", 1.0),
            "compress_time_s":   m.get("compress_time_s", 0.0),
            "decompress_time_s": m.get("decompress_time_s", 0.0),
            "current_eb":        self.current_eb,
        }
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_columns, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
