from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ── ResNet-20 for CIFAR-10/100 (He et al. 2016) ──────────────────────────────
# Standard CIFAR variant: no MaxPool stem, 3 stages (16→32→64 channels),
# global average pooling. ~272K parameters.
# Reference: https://arxiv.org/abs/1512.03385 (Table 6)

class _BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            # Option A shortcut: 1×1 conv to match dimensions (used in original paper)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet20(nn.Module):
    """
    ResNet-20 for CIFAR-10/100 (32×32 input).
    Architecture: 3 stages × 3 blocks, channels 16→32→64, global avg pool.
    Parameters: ~272K (CIFAR-10) / ~273K (CIFAR-100).
    Centralized accuracy: ~91.5% on CIFAR-10 (He et al. 2016).
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1  = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, 16, n=3, stride=1)
        self.layer2 = self._make_layer(16, 32, n=3, stride=2)
        self.layer3 = self._make_layer(32, 64, n=3, stride=2)
        self.fc     = nn.Linear(64, num_classes)

        # Weight initialisation matching the original paper
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _make_layer(in_planes: int, planes: int, n: int, stride: int) -> nn.Sequential:
        layers = [_BasicBlock(in_planes, planes, stride)]
        for _ in range(n - 1):
            layers.append(_BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def build_model(dataset: str) -> nn.Module:
    num_classes = 100 if dataset == "cifar100" else 10
    return ResNet20(num_classes=num_classes)


# ── Parameter helpers (Flower uses List[np.ndarray]) ─────────────────────────

def get_parameters(model: nn.Module) -> List[np.ndarray]:
    # Include full state_dict (parameters + BN running stats) so FedAvg
    # aggregates BN statistics correctly. All values cast to float32 for
    # uniform compression; set_parameters restores original dtypes.
    return [v.cpu().float().numpy() for v in model.state_dict().values()]


def set_parameters(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state = OrderedDict(
        (k, torch.tensor(arr, dtype=v.dtype).to(v.device))
        for (k, v), arr in zip(model.state_dict().items(), parameters)
    )
    model.load_state_dict(state, strict=True)
