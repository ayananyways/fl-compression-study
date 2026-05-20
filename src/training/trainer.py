import logging
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from mpi4py import MPI
from torch.nn.parallel import DistributedDataParallel as DDP

from src.compressors.base import Compressor
from src.data.etth1 import get_etth1_loaders
from src.data.tiny_imagenet import get_tiny_imagenet_loaders
from src.models.lstm import TimeSeriesLSTM
from src.models.resnet import get_resnet18
from src.training.comm_hook import HookState, build_compression_hook
from src.training.metrics import MetricsTracker, compute_accuracy
from src.utils.dist_utils import cleanup_distributed, get_device, setup_distributed
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


class Trainer:

    def __init__(
        self,
        config: dict,
        rank: int,
        world_size: int,
        compressor: Compressor,
    ) -> None:
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.compressor = compressor
        self.device: torch.device | None = None
        self.model: DDP | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.train_loader = None
        self.val_loader = None
        self.hook_state: HookState | None = None
        self._criterion: nn.Module | None = None
        self.start_round: int = 0
        self._checkpoint_path: str = self._build_checkpoint_path()

    def _build_checkpoint_path(self) -> str:
        checkpoint_dir = self.config.get("checkpoint_dir", "./checkpoints")
        experiment_id = self.config.get("experiment_id", "experiment")
        os.makedirs(checkpoint_dir, exist_ok=True)
        return os.path.join(checkpoint_dir, f"{experiment_id}.pt")

    def setup(self) -> None:
        set_seed(self.config["seed"] + self.rank)
        setup_distributed(self.rank, self.world_size, self.config.get("backend", "gloo"))

        self.device = get_device(self.rank)
        dataset = self.config.get("dataset", "tiny_imagenet")

        if dataset == "tiny_imagenet":
            raw_model = get_resnet18(num_classes=200)
            self.train_loader, self.val_loader = get_tiny_imagenet_loaders(
                data_dir=self.config["data_dir"],
                rank=self.rank,
                world_size=self.world_size,
                batch_size=self.config.get("batch_size", 64),
                num_workers=self.config.get("num_workers", 2),
            )
            self._criterion = nn.CrossEntropyLoss()
        elif dataset == "etth1":
            raw_model = TimeSeriesLSTM(
                input_size=7,
                hidden_size=64,
                num_layers=2,
                output_size=1,
            )
            self.train_loader, self.val_loader = get_etth1_loaders(
                path=self.config["data_dir"],
                rank=self.rank,
                world_size=self.world_size,
                batch_size=self.config.get("batch_size", 64),
            )
            self._criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        raw_model = raw_model.to(self.device)
        self.model = DDP(raw_model, device_ids=None)

        hook_fn, self.hook_state = build_compression_hook(self.compressor, self.world_size)
        self.model.register_comm_hook(state=None, hook=hook_fn)

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 0.01),
            momentum=self.config.get("momentum", 0.9),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get("total_rounds", 200),
        )

        self._load_checkpoint()

    def save_checkpoint(self, round_num: int) -> None:
        comm = MPI.COMM_WORLD
        if self.rank == 0:
            torch.save(
                {
                    "round": round_num,
                    "model": self.model.module.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                },
                self._checkpoint_path,
            )
            logger.debug("Checkpoint saved at round %d → %s", round_num, self._checkpoint_path)
        comm.Barrier()

    def _load_checkpoint(self) -> None:
        comm = MPI.COMM_WORLD
        exists = os.path.isfile(self._checkpoint_path) if self.rank == 0 else False
        exists = comm.bcast(exists, root=0)
        if not exists:
            return

        checkpoint = torch.load(self._checkpoint_path, map_location=self.device)
        self.model.module.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.start_round = checkpoint["round"] + 1
        if self.rank == 0:
            logger.info(
                "Resumed from checkpoint: %s (resuming at round %d)",
                self._checkpoint_path,
                self.start_round,
            )

    def train_one_round(self, round_num: int) -> MetricsTracker:
        self.model.train()
        self.train_loader.sampler.set_epoch(round_num)

        total_loss = 0.0
        steps = 0
        local_steps = self.config.get("local_steps", 5)
        compress_time = 0.0
        decompress_time = 0.0
        bytes_sent = 0
        wall_start = time.perf_counter()

        train_iter = iter(self.train_loader)
        for _ in range(local_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            if isinstance(self._criterion, nn.MSELoss):
                targets = targets.squeeze(-1) if targets.dim() == 3 else targets
                loss = self._criterion(outputs, targets)
            else:
                loss = self._criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            steps += 1

            if self.hook_state is not None:
                compress_time += self.hook_state.compress_time
                decompress_time += self.hook_state.decompress_time
                bytes_sent += self.hook_state.bytes_sent

        self.scheduler.step()

        val_loss, val_acc = self.evaluate()
        wall_clock = time.perf_counter() - wall_start

        ratio = 1.0
        if bytes_sent > 0:
            try:
                sample_tensor = next(iter(self.model.parameters())).detach().cpu().flatten()
                compressed_sample = self.compressor.compress(sample_tensor)
                ratio = self.compressor.compression_ratio(sample_tensor, compressed_sample)
            except Exception:
                ratio = 1.0

        return MetricsTracker(
            round_num=round_num,
            train_loss=total_loss / max(steps, 1),
            val_loss=val_loss,
            val_accuracy=val_acc,
            compress_time_s=compress_time,
            decompress_time_s=decompress_time,
            bytes_sent=bytes_sent,
            wall_clock_s=wall_clock,
            compression_ratio=ratio,
        )

    def evaluate(self) -> tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0
        is_classification = isinstance(self._criterion, nn.CrossEntropyLoss)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)

                if is_classification:
                    loss = self._criterion(outputs, targets)
                    acc = compute_accuracy(outputs, targets)
                else:
                    targets = targets.squeeze(-1) if targets.dim() == 3 else targets
                    loss = self._criterion(outputs, targets)
                    acc = 0.0

                total_loss += loss.item()
                total_acc += acc
                n_batches += 1

        n = max(n_batches, 1)
        return total_loss / n, total_acc / n

    def cleanup(self) -> None:
        cleanup_distributed()
