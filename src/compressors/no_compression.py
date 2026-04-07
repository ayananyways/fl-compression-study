import numpy as np
import torch

from .base import Compressor


class NoCompression(Compressor):

    @property
    def name(self) -> str:
        return "fp32_baseline"

    def compress(self, tensor: torch.Tensor) -> bytes:
        return tensor.numpy().tobytes()

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        np_dtype = torch.zeros(1, dtype=dtype).numpy().dtype
        arr = np.frombuffer(data, dtype=np_dtype).reshape(shape)
        return torch.from_numpy(arr.copy())
