import os
import socket

import torch
import torch.distributed as dist
from mpi4py import MPI


def is_main_process(rank: int) -> bool:
    return rank == 0


def get_mpi_comm() -> MPI.Comm:
    return MPI.COMM_WORLD


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "gloo",
) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    if "MASTER_PORT" not in os.environ:
        comm = MPI.COMM_WORLD
        port = _find_free_port() if rank == 0 else 0
        port = comm.bcast(port, root=0)
        os.environ["MASTER_PORT"] = str(port)
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
