import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple


_CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
_CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def _transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
    ])


def _load_cifar(dataset: str):
    cls = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10
    train = cls(root="./data", train=True,  download=True, transform=_transforms(True))
    test  = cls(root="./data", train=False, download=True, transform=_transforms(False))
    return train, test


def dirichlet_partition(
    dataset,
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> List[List[int]]:
    """
    Split dataset indices across clients using a Dirichlet(alpha) distribution
    per class.  Lower alpha → more non-IID (each client sees fewer classes).
    alpha=1000 ≈ IID.
    """
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = int(labels.max()) + 1
    rng = np.random.RandomState(seed)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        rng.shuffle(class_idx)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(class_idx)).astype(int)
        # make counts sum exactly to len(class_idx)
        counts[-1] = len(class_idx) - counts[:-1].sum()
        start = 0
        for client_id, count in enumerate(counts):
            client_indices[client_id].extend(class_idx[start : start + count].tolist())
            start += count

    return client_indices


def load_datasets(
    dataset: str,
    num_clients: int,
    alpha: float,
    batch_size: int = 64,
    seed: int = 42,
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Returns per-client train DataLoaders and a single test DataLoader.
    Data is partitioned IID (alpha=1000) or non-IID (alpha≈0.1–0.5).
    """
    train_ds, test_ds = _load_cifar(dataset)

    partition = dirichlet_partition(train_ds, num_clients, alpha, seed)

    train_loaders = [
        DataLoader(
            Subset(train_ds, idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        for idx in partition
    ]
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    sizes = [len(idx) for idx in partition]
    print(f"[data] {dataset} | {num_clients} clients | alpha={alpha} | "
          f"samples/client: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}")

    return train_loaders, test_loader
