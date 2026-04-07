import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
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
) -> tuple[Callable, HookState]:
    state = HookState()

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

        with Timer() as decompress_timer:
            decompressed = compressor.decompress(compressed, shape, dtype)
        state.decompress_time = decompress_timer.elapsed

        decompressed = decompressed.to(buf.device)

        fut = dist.all_reduce(
            decompressed, op=dist.ReduceOp.SUM, async_op=True
        ).get_future()

        def average(fut: torch.futures.Future) -> torch.Tensor:
            result = fut.value()[0]
            averaged = result / world_size
            buf.copy_(averaged)
            return buf

        return fut.then(average)

    return compression_hook, state
