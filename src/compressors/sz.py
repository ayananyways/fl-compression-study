import numpy as np
import torch

from .base import Compressor


class SZCompressor(Compressor):

    def __init__(self, error_bound: float) -> None:
        self.error_bound = error_bound
        try:
            import pysz as _pysz  # noqa: F401
        except ImportError:
            raise ImportError("pysz is required for SZCompressor. Install with: pip install pysz")

    @property
    def name(self) -> str:
        return f"sz_eb{self.error_bound}"

    def compress(self, tensor: torch.Tensor) -> bytes:
        import pysz
        arr = np.ascontiguousarray(tensor.float().numpy())
        cfg = pysz.szConfig()
        cfg.errorBoundMode = pysz.szErrorBoundMode.ABS
        cfg.absErrorBound = self.error_bound
        compressed, _ = pysz.sz.compress(arr, cfg)
        return compressed.tobytes()

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        import pysz
        compressed = np.frombuffer(data, dtype=np.uint8)
        decompressed, _ = pysz.sz.decompress(compressed, np.float32, shape)
        return torch.from_numpy(decompressed.copy()).to(dtype)


class SZRelCompressor(Compressor):
    """SZ3 with relative error bound: eb × (max - min) of the tensor."""

    def __init__(self, error_bound: float) -> None:
        self.error_bound = error_bound
        try:
            import pysz as _pysz  # noqa: F401
        except ImportError:
            raise ImportError("pysz is required for SZRelCompressor. Install with: pip install pysz")

    @property
    def name(self) -> str:
        return f"sz_rel_eb{self.error_bound}"

    def compress(self, tensor: torch.Tensor) -> bytes:
        import pysz
        arr = np.ascontiguousarray(tensor.float().numpy())
        cfg = pysz.szConfig()
        cfg.errorBoundMode = pysz.szErrorBoundMode.REL
        cfg.relErrorBound = self.error_bound
        compressed, _ = pysz.sz.compress(arr, cfg)
        return compressed.tobytes()

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        import pysz
        compressed = np.frombuffer(data, dtype=np.uint8)
        decompressed, _ = pysz.sz.decompress(compressed, np.float32, shape)
        return torch.from_numpy(decompressed.copy()).to(dtype)
