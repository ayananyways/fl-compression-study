import struct

import numpy as np
import torch

from .base import Compressor

_VALID_BITS = {1, 2, 4, 8, 16}


class QuantizationCompressor(Compressor):

    def __init__(self, bits: int) -> None:
        if bits not in _VALID_BITS:
            raise ValueError(f"bits must be one of {_VALID_BITS}, got {bits}")
        self.bits = bits

    @property
    def name(self) -> str:
        return f"quant_{self.bits}bit"

    def compress(self, tensor: torch.Tensor) -> bytes:
        flat = tensor.float().flatten()
        t_min = flat.min().item()
        t_max = flat.max().item()
        quantized = self._quantize(flat, t_min, t_max)
        header = struct.pack("!ff", t_min, t_max)
        packed = self._pack_bits(quantized)
        return header + packed

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        header_size = struct.calcsize("!ff")
        t_min, t_max = struct.unpack("!ff", data[:header_size])
        n_elements = 1
        for s in shape:
            n_elements *= s
        quantized = self._unpack_bits(data[header_size:], n_elements)
        flat = self._dequantize(quantized, t_min, t_max)
        return flat.reshape(shape).to(dtype)

    def _quantize(self, flat: torch.Tensor, t_min: float, t_max: float) -> np.ndarray:
        levels = (1 << self.bits) - 1
        if t_max == t_min:
            return np.zeros(flat.shape[0], dtype=np.uint32)
        normalized = (flat.numpy() - t_min) / (t_max - t_min)
        quantized = np.round(normalized * levels).astype(np.uint32)
        return np.clip(quantized, 0, levels)

    def _dequantize(self, quantized: np.ndarray, t_min: float, t_max: float) -> torch.Tensor:
        levels = (1 << self.bits) - 1
        normalized = quantized.astype(np.float32) / levels
        return torch.from_numpy(normalized * (t_max - t_min) + t_min)

    def _pack_bits(self, quantized: np.ndarray) -> bytes:
        if self.bits == 8:
            return quantized.astype(np.uint8).tobytes()
        if self.bits == 16:
            return quantized.astype(np.uint16).tobytes()
        result = bytearray()
        buf = 0
        bits_in_buf = 0
        for val in quantized:
            buf = (buf << self.bits) | int(val)
            bits_in_buf += self.bits
            while bits_in_buf >= 8:
                bits_in_buf -= 8
                result.append((buf >> bits_in_buf) & 0xFF)
        if bits_in_buf > 0:
            result.append((buf << (8 - bits_in_buf)) & 0xFF)
        return bytes(result)

    def _unpack_bits(self, data: bytes, n_elements: int) -> np.ndarray:
        if self.bits == 8:
            return np.frombuffer(data, dtype=np.uint8)[:n_elements].astype(np.uint32)
        if self.bits == 16:
            return np.frombuffer(data, dtype=np.uint16)[:n_elements].astype(np.uint32)
        mask = (1 << self.bits) - 1
        result = []
        buf = 0
        bits_in_buf = 0
        for byte in data:
            buf = (buf << 8) | byte
            bits_in_buf += 8
            while bits_in_buf >= self.bits and len(result) < n_elements:
                bits_in_buf -= self.bits
                result.append((buf >> bits_in_buf) & mask)
        return np.array(result[:n_elements], dtype=np.uint32)
