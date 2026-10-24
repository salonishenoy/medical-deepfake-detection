"""Read the BTD CT CSVs and build leakage-safe train / val / test splits.

Key ideas
---------
* Every CSV row defines an ROI patch. A slice can contribute multiple patches.
* The scan ID is the folder name in the path.
* Splits are made by SCAN, so slices from one scan never appear in two splits.
* The same source slices are reused across the CTGAN and Stable Diffusion sets,
  so the scan split is decided once per task and shared by both methods.
* Train/val come from one tamper method (default CTGAN). Two test sets are
  returned, both from held-out scans only:
      test_same  - same method as training
      test_other - the other method (the generalization test)
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import DataConfig, get_data_root

# task name -> CSV file name (inside root/CT/<task>/)
TASKS = {
    "injection": "CT_injection.csv",
    "removal": "CT_removal.csv",
}
FAKE_SUFFIXES = ["FM", "FB"]  # FM = fake malignant (injected), FB = fake benign (removed)


def _find_flag_column(df: pd.DataFrame, prefix: str) -> str:
    """Find the 0/1 column for a tamper method (e.g. 'CTGAN-CT-injection')."""
    matches = [c for c in df.columns if c.upper().startswith(prefix.upper() + "-")]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one '{prefix}-...' column, found {matches}")
    return matches[0]


def load_task_table(task: str, root: Optional[str | Path] = None) -> pd.DataFrame:
    """Load one task's CSV, preserving the authors' one-row-per-ROI samples.

    Columns: path, type, label (1 = fake), scan_id, scanner, cur_slice,
             x, y (ROI location for authentic and manipulated samples),
             in_ctgan, in_sd (1 if the row belongs to that method's test set).
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {list(TASKS)}")
    root = get_data_root(root)
    df = pd.read_csv(root / "CT" / task / TASKS[task])

    df["path"] = df["path"].str.replace("\\", "/", regex=False)
    ctgan_col = _find_flag_column(df, "CTGAN")
    sd_col = _find_flag_column(df, "SD")

    # Match the reference loader's row sampling, including exact duplicate rows.
    # Grouping by path would discard distinct coordinates on the same slice.
    out = df.rename(columns={ctgan_col: "in_ctgan", sd_col: "in_sd"}).copy()
    out["label"] = out["type"].str.strip().str[-2:].isin(FAKE_SUFFIXES).astype(int)
    out["scan_id"] = out["path"].str.split("/").str[-2]
    return out


def assign_scan_splits(df: pd.DataFrame, cfg: DataConfig) -> Dict[str, str]:
    """Map each scan_id to 'train' / 'val' / 'test', balanced across scanner models."""
    scan_scanner = df.groupby("scan_id")["scanner"].agg(lambda s: s.mode().iloc[0])
    rng = random.Random(cfg.seed)
    assignment: Dict[str, str] = {}
    for scanner in sorted(scan_scanner.unique()):
        ids = sorted(scan_scanner.index[scan_scanner == scanner])
        rng.shuffle(ids)
        n = len(ids)
        n_test = round(n * cfg.test_frac)
        n_val = round(n * cfg.val_frac)
        if n >= 3:  # make sure small scanner groups still reach val and test
            n_test, n_val = max(1, n_test), max(1, n_val)
        for i, scan in enumerate(ids):
            if i < n_test:
                assignment[scan] = "test"
            elif i < n_test + n_val:
                assignment[scan] = "val"
            else:
                assignment[scan] = "train"
    return assignment


def build_splits(
    task: str,
    train_method: str = "CTGAN",
    root: Optional[str | Path] = None,
    cfg: Optional[DataConfig] = None,
) -> Dict[str, pd.DataFrame]:
    """Return {'train', 'val', 'test_same', 'test_other'} DataFrames."""
    cfg = cfg or DataConfig()
    method = train_method.upper()
    if method not in ("CTGAN", "SD"):
        raise ValueError("train_method must be 'CTGAN' or 'SD'")

    df = load_task_table(task, root)
    # Decide the scan split BEFORE any filtering so a filter never reshuffles it.
    df["split"] = df["scan_id"].map(assign_scan_splits(df, cfg))

    if cfg.slice_range is not None:
        lo, hi = cfg.slice_range
        df = df[df["cur_slice"].between(lo, hi)]

    own, other = ("in_ctgan", "in_sd") if method == "CTGAN" else ("in_sd", "in_ctgan")
    main, rest = df[df[own] == 1], df[df[other] == 1]

    def pick(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        return frame[frame["split"] == name].reset_index(drop=True)

    return {
        "train": pick(main, "train"),
        "val": pick(main, "val"),
        "test_same": pick(main, "test"),
        "test_other": pick(rest, "test"),
    }


def check_no_leakage(splits: Dict[str, pd.DataFrame]) -> None:
    """Raise if any scan appears in more than one of train / val / test."""
    scans = {k: set(v["scan_id"]) for k, v in splits.items()}
    test_scans = scans["test_same"] | scans["test_other"]
    pairs = [
        ("train", "val", scans["train"] & scans["val"]),
        ("train", "test", scans["train"] & test_scans),
        ("val", "test", scans["val"] & test_scans),
    ]
    for a, b, overlap in pairs:
        if overlap:
            raise AssertionError(f"Leakage: {len(overlap)} scans appear in both {a} and {b}")


def summarize(splits: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Counts of authentic (0) / fake (1) ROI samples and scans per split."""
    rows = []
    for name, d in splits.items():
        rows.append(
            {
                "split": name,
                "scans": d["scan_id"].nunique(),
                "authentic": int((d["label"] == 0).sum()),
                "fake": int((d["label"] == 1).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("split")
