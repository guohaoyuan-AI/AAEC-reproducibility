from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from .common import LoaderBundle, image_transform


def _find_dir(root: Path, names):
    for name in names:
        path = root / name
        if path.is_dir():
            return path
    return None


def build_imagefolder_classification_loaders(
    root,
    image_size=224,
    train_interpolation="bicubic",
    val_interpolations=("bicubic", "bilinear"),
    test_interpolations=("bicubic", "bilinear", "nearest"),
    batch_size=64,
    num_workers=4,
    val_ratio=0.1,
    seed=42,
    download=False,
    pin_memory=True,
    **_,
):
    del download
    root = Path(root)
    train_dir = _find_dir(root, ("train", "training"))
    test_dir = _find_dir(root, ("test", "val", "validation"))
    if train_dir is None:
        raise FileNotFoundError(f"Expected train/ under {root}")
    if test_dir is None:
        raise FileNotFoundError(f"Expected test/ or val/ under {root}")

    base = ImageFolder(train_dir)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=float(val_ratio), random_state=int(seed))
    train_idx, val_idx = next(splitter.split(range(len(base.targets)), base.targets))

    train = DataLoader(
        Subset(ImageFolder(train_dir, transform=image_transform(image_size, train_interpolation)), train_idx.tolist()),
        batch_size=int(batch_size), shuffle=True, num_workers=int(num_workers),
        pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
    )
    val = {
        interp: DataLoader(
            Subset(ImageFolder(train_dir, transform=image_transform(image_size, interp)), val_idx.tolist()),
            batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
        )
        for interp in val_interpolations
    }
    test = {
        interp: DataLoader(
            ImageFolder(test_dir, transform=image_transform(image_size, interp)),
            batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers),
            pin_memory=bool(pin_memory), persistent_workers=int(num_workers) > 0,
        )
        for interp in test_interpolations
    }
    return LoaderBundle(train=train, val=val, test=test)
