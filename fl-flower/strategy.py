import csv
import glob
import os
import pickle
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import flwr as fl
from flwr.common import FitIns, FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

_CSV_COLUMNS = [
    "timestamp", "round", "compressor", "seed", "num_clients", "alpha",
    "val_accuracy", "val_loss",
    "bytes_sent", "compression_ratio",
    "compress_time_s", "decompress_time_s",
]


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """
    FedAvg strategy that:
    - Injects server_round (and optionally lr_decay/eb) into each client's fit config.
    - Collects per-round compression metrics from client fit() responses.
    - Writes one CSV row per round (flushed + fsynced for crash safety).
    - Saves a model checkpoint after every round so crashes can resume.
    """

    _csv_columns = _CSV_COLUMNS  # subclasses can extend

    def __init__(
        self,
        output_path: str,
        compressor_name: str,
        num_clients: int,
        alpha: float,
        checkpoint_dir: str = "checkpoints",
        round_offset: int = 0,
        lr_decay: bool = False,
        seed: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.output_path = output_path
        self.compressor_name = compressor_name
        self.num_clients = num_clients
        self.alpha = alpha
        self.checkpoint_dir = checkpoint_dir
        self.round_offset = round_offset
        self.lr_decay = lr_decay
        self.seed = seed

        self.round_fit_metrics: Dict[int, Dict] = {}

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not os.path.exists(output_path):
            with open(output_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_columns).writeheader()

    # ── Config injection ──────────────────────────────────────────────────────

    def _fit_config(self, server_round: int) -> dict:
        """Override in subclasses to add extra per-round config fields."""
        return {"server_round": server_round, "lr_decay": self.lr_decay}

    def configure_fit(self, server_round, parameters, client_manager):
        client_instructions = super().configure_fit(server_round, parameters, client_manager)
        extra = self._fit_config(server_round)
        return [
            (proxy, FitIns(fi.parameters, {**fi.config, **extra}))
            for proxy, fi in client_instructions
        ]

    # ── Aggregation & logging ─────────────────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if results:
            total_bytes    = sum(r.metrics.get("bytes_sent", 0)     for _, r in results)
            orig_bytes     = sum(r.metrics.get("original_bytes", 0)  for _, r in results)
            avg_compress   = float(np.mean([r.metrics.get("compress_time", 0)    for _, r in results]))
            avg_decompress = float(np.mean([r.metrics.get("decompress_time", 0)  for _, r in results]))
            ratio = orig_bytes / total_bytes if total_bytes > 0 else 1.0

            self.round_fit_metrics[server_round] = {
                "bytes_sent":        total_bytes,
                "compression_ratio": ratio,
                "compress_time_s":   avg_compress,
                "decompress_time_s": avg_decompress,
            }

        if aggregated is not None:
            params, _ = aggregated
            if params is not None:
                actual_round = server_round + self.round_offset
                ckpt_path = os.path.join(
                    self.checkpoint_dir,
                    f"{self.compressor_name}_s{self.seed}_round{actual_round:04d}.pkl",
                )
                with open(ckpt_path, "wb") as f:
                    pickle.dump(parameters_to_ndarrays(params), f)
                self._prune_checkpoints()

        return aggregated

    def log_round(self, server_round: int, val_accuracy: float, val_loss: float) -> None:
        actual_round = server_round + self.round_offset
        m = self.round_fit_metrics.get(server_round, {})
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
        }
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_columns, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def _prune_checkpoints(self) -> None:
        pattern = os.path.join(self.checkpoint_dir, f"{self.compressor_name}_s{self.seed}_round*.pkl")
        files = sorted(glob.glob(pattern))
        for old in files[:-2]:
            os.remove(old)


def find_checkpoint(checkpoint_dir: str, compressor_name: str, seed: int = 0):
    """
    Find the latest checkpoint for a (compressor, seed) pair.
    Returns (ndarrays, round_number) or (None, 0) if none found.
    """
    pattern = os.path.join(checkpoint_dir, f"{compressor_name}_s{seed}_round*.pkl")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, 0
    latest = files[-1]
    round_num = int(os.path.basename(latest).split("_round")[1].replace(".pkl", ""))
    with open(latest, "rb") as f:
        arrays = pickle.load(f)
    return arrays, round_num
