import io
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

        tensor_bytes = torch.frombuffer(
            bytearray(compressed), dtype=torch.uint8
        )
        padded_size = torch.tensor([tensor_bytes.numel()], dtype=torch.long)
        max_size = torch.zeros(1, dtype=torch.long)
        dist.all_reduce(padded_size, op=dist.ReduceOp.MAX)
        max_size[0] = padded_size[0]

        padded = torch.zeros(max_size[0].item(), dtype=torch.uint8)
        padded[: tensor_bytes.numel()] = tensor_bytes

        fut = dist.all_reduce(padded, op=dist.ReduceOp.SUM, async_op=True).get_future()

        def decompress_and_average(fut: torch.futures.Future) -> torch.Tensor:
            result_bytes = bytes(fut.value()[0][: state.bytes_sent].tolist())
            with Timer() as decompress_timer:
                decompressed = compressor.decompress(result_bytes, shape, dtype)
            state.decompress_time = decompress_timer.elapsed
            averaged = decompressed.to(buf.device) / world_size
            buf.copy_(averaged)
            return buf

        return fut.then(decompress_and_average)

    return compression_hook, state
