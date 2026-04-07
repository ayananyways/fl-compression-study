import pytest
import torch

from src.compressors.no_compression import NoCompression
from src.compressors.quantization import QuantizationCompressor


class TestNoCompression:

    def test_roundtrip(self) -> None:
        compressor = NoCompression()
        original = torch.tensor([1.0, 2.5, -3.7, 0.0, 100.0])
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert torch.allclose(original, reconstructed)

    def test_2d_roundtrip(self) -> None:
        compressor = NoCompression()
        original = torch.randn(4, 8)
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert torch.allclose(original, reconstructed)


class TestQuantization:

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_roundtrip_within_noise(self, bits: int) -> None:
        compressor = QuantizationCompressor(bits=bits)
        original = torch.randn(256)
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert reconstructed.shape == original.shape
        levels = (1 << bits) - 1
        t_min = original.min().item()
        t_max = original.max().item()
        expected_max_error = (t_max - t_min) / levels
        max_error = (original - reconstructed).abs().max().item()
        assert max_error <= expected_max_error * 1.01 + 1e-6

    def test_invalid_bits(self) -> None:
        with pytest.raises(ValueError):
            QuantizationCompressor(bits=3)

    def test_constant_tensor(self) -> None:
        compressor = QuantizationCompressor(bits=8)
        original = torch.ones(64) * 5.0
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert torch.allclose(original, reconstructed)

    def test_16bit(self) -> None:
        compressor = QuantizationCompressor(bits=16)
        original = torch.randn(128)
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert reconstructed.shape == original.shape


class TestSZCompressor:

    def test_roundtrip(self) -> None:
        try:
            import sz  # noqa: F401
        except ImportError:
            pytest.skip("pyszz not installed")

        from src.compressors.sz import SZCompressor

        error_bound = 0.01
        compressor = SZCompressor(error_bound=error_bound)
        original = torch.randn(256)
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        max_abs_error = (original - reconstructed).abs().max().item()
        assert max_abs_error <= error_bound


class TestHybridCompressor:

    def test_roundtrip(self) -> None:
        try:
            import sz  # noqa: F401
        except ImportError:
            pytest.skip("pyszz not installed")

        from src.compressors.hybrid import HybridCompressor

        compressor = HybridCompressor(bits=8, error_bound=0.01)
        original = torch.randn(256)
        compressed = compressor.compress(original)
        reconstructed = compressor.decompress(compressed, original.shape, original.dtype)
        assert reconstructed.shape == original.shape


class TestCompressionRatio:

    def test_no_compression_ratio_equals_one(self) -> None:
        compressor = NoCompression()
        tensor = torch.randn(64)
        compressed = compressor.compress(tensor)
        ratio = compressor.compression_ratio(tensor, compressed)
        assert abs(ratio - 1.0) < 1e-6

    @pytest.mark.parametrize("bits", [1, 2, 4, 8])
    def test_quantization_ratio_greater_than_one(self, bits: int) -> None:
        compressor = QuantizationCompressor(bits=bits)
        tensor = torch.randn(1024)
        compressed = compressor.compress(tensor)
        ratio = compressor.compression_ratio(tensor, compressed)
        assert ratio > 1.0, f"Expected ratio > 1.0 for {bits}-bit, got {ratio}"
