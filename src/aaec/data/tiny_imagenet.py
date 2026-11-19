from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from .imagefolder_datasets import build_imagefolder_classification_loaders


class TinyImageNetValidation(Dataset):
    """Reader for the original Tiny-ImageNet val/images + val_annotations.txt layout."""

    def __init__(self, root: Path, class_to_idx: dict[str, int], transform=None):
        self.transform = transform
        ann = root / "val" / "val_annotations.txt"
        image_dir = root / "val" / "images"
        if not ann.exists() or not image_dir.is_dir():
            raise FileNotFoundError("Original Tiny-ImageNet validation layout not found.")
        samples = []
        for line in ann.read_text(encoding="utf-8").splitlines():
            parts = line.split("	")
            if len(parts) >= 2 and parts[1] in class_to_idx:
                samples.append((image_dir / parts[0], class_to_idx[parts[1]]))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_tiny_imagenet_loaders(**kwargs):
    # The formal repository uses the same generic train-split/held-out-test
    # interface as ImageNet100. Reorganize Tiny-ImageNet val into class folders,
    # or provide a prepared test/ directory before calling this loader.
    return build_imagefolder_classification_loaders(**kwargs)
