from dataclasses import dataclass, field

import torch


@dataclass
class MetricsTracker:
    round_num: int = 0
    train_loss: float = 0.0
    val_accuracy: float = 0.0
    val_loss: float = 0.0
    compress_time_s: float = 0.0
    decompress_time_s: float = 0.0
    bytes_sent: int = 0
    wall_clock_s: float = 0.0
    compression_ratio: float = 1.0

    def to_dict(self) -> dict:
        return {
            "round": self.round_num,
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "compress_time_s": self.compress_time_s,
            "decompress_time_s": self.decompress_time_s,
            "bytes_sent": self.bytes_sent,
            "wall_clock_s": self.wall_clock_s,
            "compression_ratio": self.compression_ratio,
        }


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()
