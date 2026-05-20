from dataclasses import dataclass

import numpy as np
import torch
from mpi4py import MPI
from torch.distributed import GradBucket

from src.compressors.base import Compressor
from src.utils.timer import Timer


@dataclass
class HookState:
    compress_time: float = 0.0
    decompress_time: float = 0.0
    bytes_sent: int = 0


def build_compression_hook(
    compressor: Compressor,
    world_size: int,
) -> tuple[callable, HookState]:
    state = HookState()
    comm = MPI.COMM_WORLD

    def compression_hook(
        hook_state: object,
        bucket: GradBucket,
    ) -> torch.futures.Future[torch.Tensor]:
        buf = bucket.buffer()
        shape = tuple(buf.shape)
        dtype = buf.dtype

        with Timer() as compress_timer:
            compressed = compressor.compress(buf.cpu())
        state.compress_time = compress_timer.elapsed
        state.bytes_sent = len(compressed)

        # Exchange compressed sizes so every rank knows how much to receive
        my_size = np.array([len(compressed)], dtype=np.int64)
        all_sizes = np.zeros(world_size, dtype=np.int64)
        comm.Allgather(my_size, all_sizes)

        # Allgatherv: each rank sends its compressed bytes, receives everyone else's
        send_buf = np.frombuffer(compressed, dtype=np.uint8).copy()
        displacements = np.concatenate([[0], np.cumsum(all_sizes[:-1])]).astype(np.int64)
        recv_buf = np.zeros(int(all_sizes.sum()), dtype=np.uint8)
        comm.Allgatherv(
            send_buf,
            [recv_buf, all_sizes.tolist(), displacements.tolist(), MPI.BYTE],
        )

        # Decompress every rank's gradients and average them locally
        with Timer() as decompress_timer:
            total = torch.zeros(shape, dtype=torch.float32)
            offset = 0
            for size in all_sizes:
                chunk = bytes(recv_buf[offset : offset + int(size)])
                total += compressor.decompress(chunk, shape, dtype).float()
                offset += int(size)
        state.decompress_time = decompress_timer.elapsed

        averaged = (total / world_size).to(dtype=dtype, device=buf.device)
        buf.copy_(averaged)

        fut: torch.futures.Future[torch.Tensor] = torch.futures.Future()
        fut.set_result(buf)
        return fut

    return compression_hook, state
