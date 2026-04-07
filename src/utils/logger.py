import csv
import os
from datetime import datetime

from src.training.metrics import MetricsTracker

_CSV_COLUMNS = [
    "timestamp",
    "experiment_id",
    "round",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "compress_time_s",
    "decompress_time_s",
    "bytes_sent",
    "compression_ratio",
    "wall_clock_s",
]


class ResultsLogger:

    def __init__(self, output_path: str, experiment_id: str) -> None:
        self.output_path = output_path
        self.experiment_id = experiment_id
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if not os.path.exists(output_path):
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
                writer.writeheader()

    def log(self, metrics: MetricsTracker, extra: dict | None = None) -> None:
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment_id": self.experiment_id,
        }
        row.update(metrics.to_dict())
        if extra:
            row.update(extra)
        with open(self.output_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
