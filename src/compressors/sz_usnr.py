"""
USNR-RMS compressor: per-tensor SZ error bound derived from the update RMS.

For each trainable tensor with ndim >= 2:
    delta = W_local - W_global
    rms   = sqrt(mean(delta^2))
    eb    = clip(alpha * rms, eb_min, eb_max)
    compress W_local with SZ3 at that eb

Tensors with ndim < 2 (biases, BN scalars) are kept as raw FP32.
"""

import pickle
import struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import Compressor

# Wire-format constants
_HDR_FMT  = "!I"   # 4 bytes: total number of tensors
_CHUNK_FMT = "!BI"  # 1 byte flag + 4 bytes chunk length


class SZUsnrRmsCompressor(Compressor):
    """
    Update-Signal-to-Noise-Ratio guided SZ compression.

    compress_tensors(local_params, global_params) → (payload, metrics)
    decompress_tensors(payload, shapes)           → List[Tensor]
    """

    def __init__(
        self,
        alpha: float = 0.1,
        eb_min: float = 1e-6,
        eb_max: float = 1.0,
        diagnostics: bool = False,
    ) -> None:
        try:
            import pysz as _pysz  # noqa: F401
        except ImportError:
            raise ImportError("pysz is required for SZUsnrRmsCompressor. Install with: pip install pysz")
        self.alpha       = alpha
        self.eb_min      = eb_min
        self.eb_max      = eb_max
        self.diagnostics = diagnostics

    @property
    def name(self) -> str:
        return f"sz_usnr_rms_a{self.alpha}"

    # ── Flat interface required by base class — not used for USNR ─────────────

    def compress(self, tensor: torch.Tensor) -> bytes:
        raise NotImplementedError(
            "SZUsnrRmsCompressor requires global params. "
            "Call compress_tensors(local_params, global_params) instead."
        )

    def decompress(self, data: bytes, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        raise NotImplementedError("Call decompress_tensors(data, shapes) instead.")

    # ── USNR-specific interface ───────────────────────────────────────────────

    def compress_tensors(
        self,
        local_params: List[torch.Tensor],
        global_params: List[torch.Tensor],
    ) -> Tuple[bytes, Dict]:
        """
        Compress each parameter tensor independently.

        Returns
        -------
        payload : bytes
            Header (4B tensor count) + per-tensor records
            (1B flag | 4B chunk_len | chunk_bytes).
        metrics : dict
            Aggregated per-round stats ready to pass back via Flower metrics.
            Includes optional '_usnr_diag' key (bytes) if diagnostics=True.
        """
        import pysz

        chunks: List[bytes] = []
        ebs:   List[float]  = []
        rmss:  List[float]  = []
        n_compressed = n_uncompressed = 0
        diag_rows: List[Dict] = []
        total_original = 0

        for i, (lp, gp) in enumerate(zip(local_params, global_params)):
            local_arr  = np.ascontiguousarray(lp.float().numpy())
            global_arr = np.ascontiguousarray(gp.float().numpy())
            total_original += local_arr.nbytes

            if lp.ndim >= 2:
                delta = local_arr - global_arr
                rms   = float(np.sqrt(np.mean(delta ** 2)))
                eb    = float(np.clip(self.alpha * rms, self.eb_min, self.eb_max))

                try:
                    cfg = pysz.szConfig()
                    cfg.errorBoundMode = pysz.szErrorBoundMode.ABS
                    cfg.absErrorBound  = eb
                    compressed_arr, _ = pysz.sz.compress(local_arr, cfg)
                    chunk = compressed_arr.tobytes()
                    ebs.append(eb)
                    rmss.append(rms)
                    n_compressed += 1
                    flag = 1
                except (ValueError, RuntimeError):
                    # SZ3 can't compress tensors that are too small or high-entropy;
                    # fall back to FP32 for this tensor.
                    chunk = local_arr.tobytes()
                    rms = eb = float("nan")
                    n_uncompressed += 1
                    flag = 0
            else:
                chunk = local_arr.tobytes()
                rms = eb = float("nan")
                n_uncompressed += 1
                flag = 0

            chunks.append(struct.pack(_CHUNK_FMT, flag, len(chunk)) + chunk)

            if self.diagnostics:
                diag_rows.append({
                    "tensor_idx":       i,
                    "shape":            str(tuple(local_arr.shape)),
                    "ndim":             int(lp.ndim),
                    "original_bytes":   int(local_arr.nbytes),
                    "compressed_bytes": int(len(chunk)),
                    "ratio": (local_arr.nbytes / len(chunk)) if len(chunk) > 0 else 1.0,
                    "rms":              rms,
                    "eb":               eb,
                })

        payload = struct.pack(_HDR_FMT, len(local_params)) + b"".join(chunks)

        metrics: Dict = {
            "usnr_alpha":               self.alpha,
            "usnr_eb_mean":             float(np.mean(ebs))  if ebs  else float("nan"),
            "usnr_eb_min":              float(np.min(ebs))   if ebs  else float("nan"),
            "usnr_eb_max":              float(np.max(ebs))   if ebs  else float("nan"),
            "usnr_rms_mean":            float(np.mean(rmss)) if rmss else float("nan"),
            "usnr_rms_min":             float(np.min(rmss))  if rmss else float("nan"),
            "usnr_rms_max":             float(np.max(rmss))  if rmss else float("nan"),
            "compressed_tensors_count":   float(n_compressed),
            "uncompressed_tensors_count": float(n_uncompressed),
            "bytes_sent":               float(len(payload)),
            "original_bytes":           float(total_original),
        }
        if self.diagnostics:
            metrics["_usnr_diag"] = pickle.dumps(diag_rows)

        return payload, metrics

    def decompress_tensors(
        self,
        data: bytes,
        shapes: List[Tuple],
        dtype: torch.dtype = torch.float32,
    ) -> List[torch.Tensor]:
        """Unpack payload produced by compress_tensors back to a list of tensors."""
        import pysz

        offset = 0
        (n,) = struct.unpack_from(_HDR_FMT, data, offset)
        offset += struct.calcsize(_HDR_FMT)

        tensors: List[torch.Tensor] = []
        chunk_step = struct.calcsize(_CHUNK_FMT)

        for i in range(n):
            flag, chunk_len = struct.unpack_from(_CHUNK_FMT, data, offset)
            offset += chunk_step
            chunk  = data[offset: offset + chunk_len]
            offset += chunk_len

            shape = shapes[i]
            if flag == 1:
                compressed = np.frombuffer(chunk, dtype=np.uint8)
                decompressed, _ = pysz.sz.decompress(compressed, np.float32, shape)
                t = torch.from_numpy(decompressed.copy()).to(dtype)
            else:
                arr = np.frombuffer(chunk, dtype=np.float32).reshape(shape)
                t = torch.from_numpy(arr.copy()).to(dtype)

            tensors.append(t)

        return tensors
