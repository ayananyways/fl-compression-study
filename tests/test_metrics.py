import torch

from src.training.metrics import MetricsTracker, compute_accuracy


class TestComputeAccuracy:

    def test_all_correct(self) -> None:
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        acc = compute_accuracy(logits, targets)
        assert abs(acc - 1.0) < 1e-6

    def test_none_correct(self) -> None:
        logits = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        targets = torch.tensor([1, 0])
        acc = compute_accuracy(logits, targets)
        assert abs(acc - 0.0) < 1e-6

    def test_half_correct(self) -> None:
        logits = torch.tensor([[0.9, 0.1], [0.2, 0.8]])
        targets = torch.tensor([0, 0])
        acc = compute_accuracy(logits, targets)
        assert abs(acc - 0.5) < 1e-6


class TestMetricsTracker:

    def test_to_dict_keys(self) -> None:
        tracker = MetricsTracker(
            round_num=3,
            train_loss=0.5,
            val_accuracy=0.8,
            val_loss=0.4,
            compress_time_s=0.01,
            decompress_time_s=0.005,
            bytes_sent=1024,
            wall_clock_s=2.3,
            compression_ratio=4.0,
        )
        d = tracker.to_dict()
        expected_keys = {
            "round",
            "train_loss",
            "val_loss",
            "val_accuracy",
            "compress_time_s",
            "decompress_time_s",
            "bytes_sent",
            "wall_clock_s",
            "compression_ratio",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self) -> None:
        tracker = MetricsTracker(round_num=7, train_loss=1.23)
        d = tracker.to_dict()
        assert d["round"] == 7
        assert abs(d["train_loss"] - 1.23) < 1e-6
