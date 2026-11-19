from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from torch.utils.data import DataLoader
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class LoaderBundle:
    train: DataLoader
    val: Dict[str, DataLoader]
    test: Dict[str, DataLoader]


def interpolation_mode(name: str):
    name = str(name).strip().lower()
    mapping = {
        "bicubic": transforms.InterpolationMode.BICUBIC,
        "bilinear": transforms.InterpolationMode.BILINEAR,
        "nearest": transforms.InterpolationMode.NEAREST,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported interpolation: {name}")
    return mapping[name]


def image_transform(image_size: int, interpolation: str):
    return transforms.Compose(
        [
            transforms.Resize(
                (int(image_size), int(image_size)),
                interpolation=interpolation_mode(interpolation),
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
