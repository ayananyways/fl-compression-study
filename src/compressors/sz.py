import numpy as np
import torch

from .base import Compressor


class SZCompressor(Compressor):

    def __init__(self, error_bound: float) -> None:
        self.error_bound = error_bound
        try:
            import sz as _sz  # noqa: F401
        except ImportError:
            raise ImportError(
                "pyszz is required for SZCompressor. "
                "Install it with: pip install pyszz"
            )

    @property
    def name(self) -> str:
        return f"sz_eb{self.error_bound}"

    def compress(self, tensor: torch.Tensor) -> bytes:
        import sz
        arr = tensor.float().numpy()
        compressed = sz.compress(arr, abs_err_bound=self.error_bound)
        return compressed

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        import sz
        arr = sz.decompress(data, np.float32, shape)
        return torch.from_numpy(arr).to(dtype)
