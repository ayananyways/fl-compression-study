import torch

from .base import Compressor
from .quantization import QuantizationCompressor
from .sz import SZCompressor


class HybridCompressor(Compressor):
    """
    SZ3 for multi-dimensional tensors, quantization for 1-D tensors.

    ResNet-20 has 19 multi-dimensional tensors (conv weights, linear weights)
    and 46 one-dimensional tensors (BN scale/bias/running stats, conv biases).
    SZ3's Lorenzo predictor exploits spatial structure in 2-D+ arrays; it adds
    non-trivial per-tensor metadata overhead on small 1-D arrays where it
    rarely beats quantization. This compressor applies each method where it has
    a practical advantage.

    Wire format: one flag byte (0x00 = quant, 0x01 = SZ3) prepended to the
    compressed payload. The decompressor reads the flag to dispatch.
    """

    _FLAG_QUANT = b'\x00'
    _FLAG_SZ    = b'\x01'

    def __init__(self, sz_error_bound: float = 0.001, quant_bits: int = 8) -> None:
        self.sz_error_bound = sz_error_bound
        self.quant_bits = quant_bits
        self._sz    = SZCompressor(error_bound=sz_error_bound)
        self._quant = QuantizationCompressor(bits=quant_bits)

    @property
    def name(self) -> str:
        return f"hybrid_sz{self.sz_error_bound}_q{self.quant_bits}bit"

    def compress(self, tensor: torch.Tensor) -> bytes:
        if tensor.ndim >= 2:
            try:
                return self._FLAG_SZ + self._sz.compress(tensor)
            except Exception:
                pass  # tensor too small for SZ3 (e.g. conv1, shortcut, fc) — fall through
        return self._FLAG_QUANT + self._quant.compress(tensor)

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        flag, payload = data[:1], data[1:]
        if flag == self._FLAG_SZ:
            return self._sz.decompress(payload, shape, dtype)
        return self._quant.decompress(payload, shape, dtype)
