import struct

import torch

from .base import Compressor
from .quantization import QuantizationCompressor
from .sz import SZCompressor


class HybridCompressor(Compressor):

    def __init__(self, bits: int, error_bound: float) -> None:
        self.bits = bits
        self.error_bound = error_bound
        self._quant = QuantizationCompressor(bits)
        self._sz = SZCompressor(error_bound)

    @property
    def name(self) -> str:
        return f"hybrid_quant{self.bits}bit_sz{self.error_bound}"

    def compress(self, tensor: torch.Tensor) -> bytes:
        original_shape = tensor.shape
        original_dtype = tensor.dtype
        quant_bytes = self._quant.compress(tensor)
        quant_tensor = self._quant.decompress(quant_bytes, original_shape, original_dtype)
        sz_bytes = self._sz.compress(quant_tensor)
        header = struct.pack("!I", len(quant_bytes))
        return header + sz_bytes

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        header_size = struct.calcsize("!I")
        _quant_len = struct.unpack("!I", data[:header_size])[0]
        sz_bytes = data[header_size:]
        return self._sz.decompress(sz_bytes, shape, dtype)
