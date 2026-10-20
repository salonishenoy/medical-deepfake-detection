"""Central settings for the data pipeline.

Nothing machine-specific is hard-coded. Point the code at the dataset folder
(the one that contains the ``CT`` and ``MRI`` folders) in one of two ways:

    * set the environment variable ``BTD_DATA_ROOT``, or
    * pass ``root=...`` to the functions that need it.

Windows (PowerShell) example:
    $env:BTD_DATA_ROOT = "C:\\data\\btd\\archive"
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

ENV_VAR = "BTD_DATA_ROOT"


def get_data_root(root: Optional[str | Path] = None) -> Path:
    """Return the dataset root folder, or raise a clear error if it is not set/valid."""
    value = root or os.environ.get(ENV_VAR)
    if not value:
        raise ValueError(
            f"Dataset folder not set. Set the {ENV_VAR} environment variable "
            "or pass root=... (the folder that contains 'CT' and 'MRI')."
        )
    path = Path(value)
    if not (path / "CT").is_dir():
        raise FileNotFoundError(f"No 'CT' folder inside {path}. Check the dataset path.")
    return path


@dataclass
class DataConfig:
    # --- CT intensity window (Hounsfield units). Everything is clipped to this
    # range and then scaled to [0, 1], so the same tissue always gets the same
    # brightness. -1000..400 covers lung and soft tissue.
    hu_min: float = -1000.0
    hu_max: float = 400.0

    # --- Model input
    image_size: int = 224          # ResNet / EfficientNet input size
    imagenet_norm: bool = True     # subtract ImageNet mean/std (needed for pretrained weights)

    # --- Splitting (done by scan, never by slice)
    seed: int = 42
    val_frac: float = 0.15
    test_frac: float = 0.15

    # --- Optional slice-position filter, e.g. (34, 132). OFF by default so no
    # data is thrown away. See src/check_data.py to see what it would cost.
    slice_range: Optional[Tuple[int, int]] = None

    # --- Training-only augmentation (kept mild so tampered lesions are not removed)
    flip_prob: float = 0.5
    rotation_deg: float = 10.0
