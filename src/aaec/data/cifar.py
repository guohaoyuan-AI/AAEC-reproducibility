from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10, CIFAR100

from .common import LoaderBundle, image_transform


def _build_cifar_loaders(
    dataset_cls,
    root,
    image_size,
    train_interpolation,
    val_interpolations,
    test_interpolations,
    batch_size,
    num_workers,
    val_ratio,
    seed,
    download,
    pin_memory,
):
    root = str(Path(root))
    base = dataset_cls(root=root, train=True, transform=None, download=bool(download))
    labels = list(base.targets)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
    train_idx, val_idx = next(splitter.split(range(len(labels)), labels))

    train_ds = Subset(
        dataset_cls(root=root, train=True, transform=image_transform(image_size, train_interpolation), download=False),
        train_idx.tolist(),
    )
    val = {
        interp: DataLoader(
            Subset(
                dataset_cls(root=root, train=True, transform=image_transform(image_size, interp), download=False),
                val_idx.tolist(),
            ),
            batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
        )
        for interp in val_interpolations
    }
    test = {
        interp: DataLoader(
            dataset_cls(root=root, train=False, transform=image_transform(image_size, interp), download=False),
            batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
        )
        for interp in test_interpolations
    }
    train = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers),
        pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
    )
    return LoaderBundle(train=train, val=val, test=test)


def build_cifar100_loaders(**kwargs):
    return _build_cifar_loaders(CIFAR100, **kwargs)


def build_cifar10_loaders(**kwargs):
    return _build_cifar_loaders(CIFAR10, **kwargs)
