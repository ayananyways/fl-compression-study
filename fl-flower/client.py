import math
import sys, os, time
import numpy as np
import torch
import torch.nn as nn
import flwr as fl
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
from src.compressors.base import Compressor
from models import get_parameters, set_parameters


def cosine_lr(
    round_num: int,
    lr_max: float = 0.01,
    lr_min: float = 0.001,
    total_rounds: int = 100,
) -> float:
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * round_num / total_rounds))


class FlowerClient(fl.client.NumPyClient):
    """
    One simulated FL client.

    On each fit() call:
      1. Load global weights from server.
      2. Train locally for `local_epochs`.
      3. Compress the updated weights, then immediately decompress them.
         This simulates the lossy effect of compression on the model
         (critical for SZ — quantization error accumulates across rounds).
      4. Return the (decompressed) weights plus compression metrics.

    Config dict keys read from server each round:
      server_round (int)  : current round number, used for cosine LR
      lr_decay (bool)     : if True, use cosine LR schedule instead of fixed
      eb (float, optional): override compressor error bound (for adaptive SZ)
    """

    def __init__(
        self,
        cid: int,
        model: nn.Module,
        trainloader: torch.utils.data.DataLoader,
        local_epochs: int,
        compressor: Compressor,
        device: torch.device,
    ) -> None:
        self.cid = cid
        self.model = model.to(device)
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.compressor = compressor
        self.device = device

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        set_parameters(self.model, parameters)

        server_round = int(config.get("server_round", 1))
        lr = cosine_lr(server_round) if config.get("lr_decay", False) else 0.01

        # Override compressor error bound for adaptive SZ scheduling
        eb = config.get("eb", None)
        if eb is not None and hasattr(self.compressor, "error_bound"):
            self.compressor.error_bound = float(eb)

        self._train(lr=lr)

        # Compress ONLY trainable parameters (nn.Parameters), not BN buffers.
        # Including buffers like num_batches_tracked (large int) in the flat
        # vector would blow up the quantization range and destroy weight fidelity.
        # BN buffers are returned uncompressed; Flower aggregates the full state_dict.
        param_flat = np.concatenate([p.detach().cpu().numpy().flatten()
                                     for p in self.model.parameters()])
        param_shapes = [p.shape for p in self.model.parameters()]
        tensor = torch.from_numpy(param_flat.copy())

        t0 = time.perf_counter()
        compressed = self.compressor.compress(tensor)
        compress_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        decompressed = self.compressor.decompress(
            compressed, tuple(tensor.shape), tensor.dtype
        )
        decompress_time = time.perf_counter() - t0

        # Reconstruct compressed parameters into model, then return full state_dict
        compressed_params = self._unflatten(decompressed.numpy(), param_shapes)
        state = self.model.state_dict()
        for (name, _), arr in zip(self.model.named_parameters(), compressed_params):
            state[name] = torch.tensor(arr)
        params_out = [v.cpu().float().numpy() for v in state.values()]

        metrics = {
            "bytes_sent":     float(len(compressed)),
            "original_bytes": float(param_flat.nbytes),
            "compress_time":  compress_time,
            "decompress_time": decompress_time,
        }
        return params_out, len(self.trainloader.dataset), metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        # Server-side centralized evaluation is used; this is a no-op.
        return 0.0, len(self.trainloader.dataset), {}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _train(self, lr: float = 0.01) -> None:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4
        )
        for _ in range(self.local_epochs):
            for x, y in self.trainloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                criterion(self.model(x), y).backward()
                optimizer.step()

    @staticmethod
    def _unflatten(flat: np.ndarray, shapes: List[tuple]) -> List[np.ndarray]:
        result, idx = [], 0
        for shape in shapes:
            n = int(np.prod(shape))
            result.append(flat[idx : idx + n].reshape(shape))
            idx += n
        return result
