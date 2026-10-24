"""PyTorch Dataset / DataLoaders for the BTD CT slices.

Preprocessing, in order (every image):
    1. load the .npy slice and cast to float32   (int16 / float64 -> one dtype)
    2. clip to a fixed CT window and scale to [0, 1]
    3. extract a 96x96 patch at the CSV coordinates, then resize to image_size
    4. copy the single channel to 3 channels      (pretrained CNNs expect RGB)
    5. training set only: random flips + small rotation
    6. optional ImageNet mean/std normalization   (for pretrained weights)

Each item is (image, label, meta):
    image  float tensor [3, S, S]
    label  0 = authentic, 1 = manipulated
    meta   dict with path, scan_id, scanner, cur_slice and x / y, the CSV ROI
           location in the RESIZED PATCH (column x, row y), for BOTH classes.
           Authentic removal coordinates select healthy tissue, not a lesion.
           source_x/y and crop_left/top describe the original slice geometry.
           x / y are only valid when augmentation is off (val / test).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import DataConfig, get_data_root
from .splits import build_splits

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_slice(path: str | Path) -> np.ndarray:
    """Load one .npy slice as a 2D float32 array."""
    arr = np.squeeze(np.load(path))
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D slice, got shape {arr.shape} for {path}")
    return arr.astype(np.float32)


def window_and_scale(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip to [lo, hi] and scale to [0, 1]."""
    return (np.clip(arr, lo, hi) - lo) / (hi - lo)


def extract_patch(arr: np.ndarray, x: int, y: int, patch_size: int):
    """Shift an ROI inward at image edges, as in the authors' evaluation loader."""
    h, w = arr.shape
    if patch_size <= 0 or patch_size % 2 or min(h, w) < patch_size:
        raise ValueError(f"Expected a positive even patch size <= image dimensions, got {patch_size} for {arr.shape}")
    if not (0 <= x < w and 0 <= y < h):
        raise ValueError(f"ROI coordinate {(x, y)} is outside image shape {arr.shape}")
    half = patch_size // 2
    cx, cy = min(max(x, half), w - half), min(max(y, half), h - half)
    left, top = cx - half, cy - half
    return arr[top:top + patch_size, left:left + patch_size], left, top


class BTDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path, cfg: DataConfig, train: bool = False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg
        self.train = train
        self.augment = T.Compose(
            [
                T.RandomHorizontalFlip(cfg.flip_prob),
                T.RandomVerticalFlip(cfg.flip_prob),
                T.RandomRotation(cfg.rotation_deg),
            ]
        )
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.df)

    def label_counts(self) -> Dict[int, int]:
        return {int(k): int(v) for k, v in self.df["label"].value_counts().items()}

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        arr = load_slice(self.root / row["path"])
        arr = window_and_scale(arr, self.cfg.hu_min, self.cfg.hu_max)
        source_x, source_y = int(row["x"]), int(row["y"])
        arr, left, top = extract_patch(arr, source_x, source_y, self.cfg.patch_size)

        size = self.cfg.image_size
        img = torch.from_numpy(arr)[None, None]  # [1, 1, H, W]
        img = F.interpolate(img, size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0]
        img = img.repeat(3, 1, 1)  # [3, S, S]

        if self.train:
            img = self.augment(img)
        if self.cfg.imagenet_norm:
            img = (img - self.mean) / self.std

        x = (source_x - left) * size / self.cfg.patch_size
        y = (source_y - top) * size / self.cfg.patch_size
        meta = {
            "path": str(row["path"]),
            "scan_id": str(row["scan_id"]),
            "scanner": str(row["scanner"]),
            "cur_slice": int(row["cur_slice"]),
            "x": x,
            "y": y,
            "source_x": source_x,
            "source_y": source_y,
            "crop_left": left,
            "crop_top": top,
        }
        return img, int(row["label"]), meta


def get_dataloaders(
    task: str,
    train_method: str = "CTGAN",
    root: Optional[str | Path] = None,
    cfg: Optional[DataConfig] = None,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for 'train', 'val', 'test_same' and 'test_other'.

    task:          'injection' or 'removal'
    train_method:  'CTGAN' (default) or 'SD'. The other method is the
                   generalization test set ('test_other').
    """
    cfg = cfg or DataConfig()
    root = get_data_root(root)
    splits = build_splits(task, train_method, root, cfg)
    loaders = {}
    for name, df in splits.items():
        is_train = name == "train"
        loaders[name] = DataLoader(
            BTDDataset(df, root, cfg, train=is_train),
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
        )
    return loaders
