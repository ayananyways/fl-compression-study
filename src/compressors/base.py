import abc

import torch


class Compressor(abc.ABC):

    @abc.abstractmethod
    def compress(self, tensor: torch.Tensor) -> bytes:
        ...

    @abc.abstractmethod
    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        ...

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    def compression_ratio(self, original: torch.Tensor, compressed: bytes) -> float:
        return len(original.numpy().tobytes()) / len(compressed)
