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
        if self.bits == 1:
            return np.packbits(quantized.astype(np.uint8)).tobytes()
        # 2-bit: pack 4 values per byte
        # 4-bit: pack 2 values per byte
        vals_per_byte = 8 // self.bits
        n = len(quantized)
        pad = (-n) % vals_per_byte
        padded = np.zeros(n + pad, dtype=np.uint8)
        padded[:n] = quantized.astype(np.uint8)
        grouped = padded.reshape(-1, vals_per_byte)
        packed = np.zeros(len(grouped), dtype=np.uint8)
        for i in range(vals_per_byte):
            packed |= (grouped[:, i] << (8 - self.bits * (i + 1)))
        return packed.tobytes()

    def _unpack_bits(self, data: bytes, n_elements: int) -> np.ndarray:
        if self.bits == 8:
            return np.frombuffer(data, dtype=np.uint8)[:n_elements].astype(np.uint32)
        if self.bits == 16:
            return np.frombuffer(data, dtype=np.uint16)[:n_elements].astype(np.uint32)
        if self.bits == 1:
            return np.unpackbits(np.frombuffer(data, dtype=np.uint8))[:n_elements].astype(np.uint32)
        # 2-bit: unpack 4 values per byte
        # 4-bit: unpack 2 values per byte
        vals_per_byte = 8 // self.bits
        mask = np.uint8((1 << self.bits) - 1)
        packed = np.frombuffer(data, dtype=np.uint8)
        grouped = np.zeros((len(packed), vals_per_byte), dtype=np.uint8)
        for i in range(vals_per_byte):
            grouped[:, i] = (packed >> (8 - self.bits * (i + 1))) & mask
        return grouped.reshape(-1)[:n_elements].astype(np.uint32)
