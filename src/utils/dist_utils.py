import os

import torch
import torch.distributed as dist


def is_main_process(rank: int) -> bool:
    return rank == 0


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "gloo",
) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(rank: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")
