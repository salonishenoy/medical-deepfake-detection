"""MRI PNG data pipeline with shared source-folder splits across both tasks.

Scaling and 128px ROIs follow the BTD evaluation notebook:
https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/
0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb
Default: preserve PNG orientation; the unstable border-based flip is reference-only.
If explicitly enabled, reflected y = height - 1 - y, rather than upstream 512-y.
CNN resizing, augmentation, normalization, metadata, and splits are project additions.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .config import get_data_root
from .data_loader import IMAGENET_MEAN, IMAGENET_STD, extract_patch
from .splits import assign_scan_splits

LABELS = {"injection": {"TM": 0, "FM": 1}, "removal": {"TB": 0, "FB": 1}}


@dataclass
class MRIDataConfig:
    orientation_policy: str = "preserve"
    patch_size: int = 128
    image_size: int = 224
    imagenet_norm: bool = True
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15
    flip_prob: float = 0.5
    rotation_deg: float = 10.0

    def __post_init__(self):
        if self.orientation_policy not in ("preserve", "author_border"):
            raise ValueError("orientation_policy must be 'preserve' or 'author_border'")
        if self.patch_size <= 0 or self.patch_size % 2 or self.image_size <= 0:
            raise ValueError("patch_size must be positive and even; image_size must be positive")
        if not (0 < self.val_frac < 1 and 0 < self.test_frac < 1 and self.val_frac + self.test_frac < 1):
            raise ValueError("Validation/test fractions must be positive and sum to less than 1")


def load_mri_table(task, root=None):
    """Preserve CSV ROI rows; validate labels and released source-folder layout."""
    if task not in LABELS:
        raise ValueError(f"task must be one of {list(LABELS)}")
    root = get_data_root(root, modality="MRI")
    df = pd.read_csv(root / "MRI" / f"MRI_{task}.csv")
    required = ["type", "path", "x", "y", "scanner"]
    if not set(required).issubset(df.columns) or df[required].isna().any().any() or df.empty:
        raise ValueError(f"{task}: empty manifest or missing required fields")
    df["path"] = df.path.str.replace("\\", "/", regex=False)
    parts = df.path.str.extract(r"^MRI/([^/]+)/(Breast_MRI_\d+)/([^/]+\.png)$")
    if parts.isna().any().any() or not (parts[0] == df.scanner).all():
        raise ValueError(f"{task}: unexpected MRI path or scanner mismatch")
    if not df.type.isin(LABELS[task]).all():
        raise ValueError(f"{task}: unknown class label")
    coords = df[["x", "y"]].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(coords).all() or not (coords == np.floor(coords)).all():
        raise ValueError(f"{task}: ROI coordinates must be finite integers")
    df[["x", "y"]] = coords.astype(np.int64)
    df["scan_id"] = parts[1]
    slices = parts[2].str.extract(r"^(?:injection_)?(\d+)(?:_fake)?\.png$")[0]
    if slices.isna().any():
        raise ValueError(f"{task}: unexpected slice filename")
    df["cur_slice"] = slices.astype(int)
    df["label"] = df.type.map(LABELS[task])
    df["task"] = task
    return df


def build_mri_splits(task, root=None, cfg=None):
    """SD-only train/val/test; always assign the union of both manifests together.

    Both CSVs are required even when loading one task. This prevents a source
    shared by injection/removal from receiving different split assignments.
    """
    if task not in LABELS:
        raise ValueError(f"task must be one of {list(LABELS)}")
    cfg = cfg or MRIDataConfig()
    tables = {name: load_mri_table(name, root) for name in LABELS}
    combined = pd.concat(tables.values(), ignore_index=True)
    if (combined.groupby("scan_id").scanner.nunique() != 1).any():
        raise ValueError("A source folder has inconsistent scanners across MRI manifests")
    assignment = assign_scan_splits(combined, cfg)
    df = tables[task].copy()
    df["split"] = df.scan_id.map(assignment)
    splits = {name: df[df.split == name].reset_index(drop=True) for name in ("train", "val", "test")}
    if any(d.empty or d.label.nunique() != 2 for d in splits.values()):
        raise ValueError("MRI split is empty or lacks a class; inspect source groups and split fractions")
    return splits


def load_mri_patch(path, x, y, patch_size=128, orientation_policy="preserve"):
    """Return a float32 native ROI and its geometry before random augmentation."""
    if orientation_policy not in ("preserve", "author_border"):
        raise ValueError("orientation_policy must be 'preserve' or 'author_border'")
    with Image.open(path) as im:
        if im.mode != "L":
            raise ValueError(f"Expected grayscale uint8 PNG (mode L), got {im.mode}: {path}")
        image = np.array(im).astype(np.float32) / 255.0
    h, w = image.shape
    if not (np.isfinite(x) and np.isfinite(y) and x == int(x) and y == int(y)
            and 0 <= x < w and 0 <= y < h):
        raise ValueError(f"ROI coordinate {(x, y)} is outside image shape {image.shape} or non-integral")
    x, y = int(x), int(y)
    flipped = orientation_policy == "author_border" and bool(image[-30:].sum() < image[:30].sum())
    oriented_y = h - 1 - y if flipped else y
    if flipped:
        image = np.flip(image, axis=0).copy()
    patch, left, top = extract_patch(image, x, oriented_y, patch_size)
    geometry = dict(source_x=x, source_y=y, oriented_x=x, oriented_y=oriented_y,
                    orientation_flipped=flipped, orientation_policy=orientation_policy,
                    source_height=h, source_width=w,
                    crop_left=left, crop_top=top)
    return patch, geometry


class MRIDataset(Dataset):
    def __init__(self, df, root, cfg=None, train=False):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.cfg = cfg or MRIDataConfig()
        self.train = train
        self.augment = T.Compose([T.RandomHorizontalFlip(self.cfg.flip_prob),
                                  T.RandomVerticalFlip(self.cfg.flip_prob),
                                  T.RandomRotation(self.cfg.rotation_deg)])
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        patch, meta = load_mri_patch(self.root / row.path, row.x, row.y, self.cfg.patch_size,
                                    self.cfg.orientation_policy)
        size = self.cfg.image_size
        image = F.interpolate(torch.from_numpy(patch)[None, None], size=(size, size),
                              mode="bilinear", align_corners=False, antialias=True)[0].repeat(3, 1, 1)
        if self.train:
            image = self.augment(image)
        if self.cfg.imagenet_norm:
            image = (image - self.mean) / self.std
        meta.update(path=str(row.path), scan_id=str(row.scan_id), scanner=str(row.scanner),
                    cur_slice=int(row.cur_slice), task=str(row.task),
                    x=(meta["oriented_x"] - meta["crop_left"] + 0.5) * size / self.cfg.patch_size - 0.5,
                    y=(meta["oriented_y"] - meta["crop_top"] + 0.5) * size / self.cfg.patch_size - 0.5,
                    coordinate_convention="pixel_center",
                    coordinates_valid=not self.train)
        return image, int(row.label), meta


def get_mri_dataloaders(task, root=None, cfg=None, batch_size=32, num_workers=0):
    """Return train/val/test loaders. MRI has no CTGAN test_other population."""
    root = get_data_root(root, modality="MRI")
    cfg = cfg or MRIDataConfig()
    splits = build_mri_splits(task, root, cfg)
    return {name: DataLoader(MRIDataset(df, root, cfg, train=name == "train"),
                            batch_size=batch_size, shuffle=name == "train", num_workers=num_workers)
            for name, df in splits.items()}
