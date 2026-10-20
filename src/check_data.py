"""Sanity checks for the data pipeline. Run it once the dataset is downloaded:

    python -m src.check_data --root "C:\\data\\btd\\archive" --task both

For each task (injection / removal) it:
  1. builds the splits and prints how many authentic / fake slices each has
  2. checks that no scan appears in two splits (leakage)
  3. checks every file in the splits exists
  4. shows the raw array types per class (catches file-type shortcuts)
  5. checks preprocessed images are 3 x S x S and within [0, 1]
  6. shows what the slice-position filter WOULD remove (nothing is removed by default)
  7. trains two tiny "shortcut" models (slice number only, scanner only); an AUC
     well above 0.5 means that detail alone gives away real vs fake
  8. saves a sample image grid with the lesion position marked on fakes
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig, get_data_root
from .data_loader import BTDDataset
from .splits import TASKS, build_splits, check_no_leakage, load_task_table, summarize


def raw_dtypes(df: pd.DataFrame, root: Path, per_class: int = 30, seed: int = 0) -> None:
    rng = random.Random(seed)
    for label, name in ((0, "authentic"), (1, "fake")):
        paths = list(df.loc[df["label"] == label, "path"])
        rng.shuffle(paths)
        counts = Counter(str(np.load(root / p, mmap_mode="r").dtype) for p in paths[:per_class])
        print(f"   raw dtype, {name:9s}: {dict(counts)}")


def preprocessing_check(df: pd.DataFrame, root: Path, cfg: DataConfig, n: int = 20) -> None:
    ds = BTDDataset(df, root, replace(cfg, imagenet_norm=False), train=False)
    idx = random.Random(0).sample(range(len(ds)), min(n, len(ds)))
    lo, hi, shape = 1.0, 0.0, None
    for i in idx:
        img, _, _ = ds[i]
        lo, hi, shape = min(lo, img.min().item()), max(hi, img.max().item()), tuple(img.shape)
    print(f"   preprocessed: shape={shape}, value range=[{lo:.3f}, {hi:.3f}] (expect within [0, 1])")


def slice_filter_cost(task: str, root: Path, slice_range) -> None:
    full = load_task_table(task, root)
    lo, hi = slice_range
    print(f"   slice filter {lo}-{hi}: what it WOULD keep (not applied unless you set slice_range)")
    for method, col in (("CTGAN", "in_ctgan"), ("SD", "in_sd")):
        d = full[full[col] == 1]
        for label, name in ((0, "authentic"), (1, "fake")):
            sub = d[d["label"] == label]
            kept = int(sub["cur_slice"].between(lo, hi).sum())
            print(f"      {method:5s} {name:9s}: keep {kept} of {len(sub)}")


def shortcut_baselines(train: pd.DataFrame, val: pd.DataFrame) -> None:
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        print("   (scikit-learn not installed, skipping shortcut baselines)")
        return
    if val["label"].nunique() < 2 or train["label"].nunique() < 2:
        print("   (a split has only one class, skipping shortcut baselines)")
        return
    both = pd.concat([train, val])
    features = {
        "slice number only": both[["cur_slice"]].astype(float),
        "scanner only": pd.get_dummies(both["scanner"]).astype(float),
    }
    n = len(train)
    for name, X in features.items():
        clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X.iloc[:n], train["label"])
        auc = roc_auc_score(val["label"], clf.predict_proba(X.iloc[n:])[:, 1])
        print(f"   shortcut model, {name:17s}: val AUC = {auc:.3f}  (0.5 = no leak)")


def save_samples(df: pd.DataFrame, root: Path, cfg: DataConfig, out_file: Path, k: int = 4) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = BTDDataset(df, root, replace(cfg, imagenet_norm=False), train=False)
    rng = random.Random(0)
    picks = []
    for label in (0, 1):
        idx = list(df.index[df["label"] == label])
        rng.shuffle(idx)
        picks += idx[:k]
    fig, axes = plt.subplots(2, k, figsize=(3 * k, 6))
    for ax, i in zip(axes.ravel(), picks):
        img, label, meta = ds[i]
        ax.imshow(img[0], cmap="gray", vmin=0, vmax=1)
        if label == 1:
            ax.add_patch(plt.Circle((meta["x"], meta["y"]), 12, fill=False, color="red", lw=1.5))
        ax.set_title(("FAKE" if label else "real") + f" | slice {meta['cur_slice']}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=100)
    plt.close(fig)
    print(f"   saved sample grid: {out_file} (red circle = lesion position on fakes)")


def check_task(task: str, method: str, root: Path, cfg: DataConfig, out_dir: Path, slice_range) -> None:
    print(f"\n=== {task} | train on {method} ===")
    splits = build_splits(task, method, root, cfg)
    print(summarize(splits).to_string())
    check_no_leakage(splits)
    print("   leakage check: OK (no scan appears in two splits)")

    everything = pd.concat(splits.values()).drop_duplicates("path")
    missing = [p for p in everything["path"] if not (root / p).exists()]
    print(f"   missing files: {len(missing)}" + (f" e.g. {missing[:3]}" if missing else ""))
    if missing:
        return

    raw_dtypes(everything, root)
    preprocessing_check(splits["train"], root, cfg)
    slice_filter_cost(task, root, slice_range)
    shortcut_baselines(splits["train"], splits["val"])
    save_samples(splits["train"], root, cfg, out_dir / f"samples_{task}_{method.lower()}.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=None, help="dataset folder (or set BTD_DATA_ROOT)")
    p.add_argument("--task", choices=[*TASKS, "both"], default="both")
    p.add_argument("--train-method", choices=["CTGAN", "SD"], default="CTGAN")
    p.add_argument("--out", default="sample_images", help="where to save the sample grid")
    p.add_argument("--slice-min", type=int, default=34)
    p.add_argument("--slice-max", type=int, default=132)
    args = p.parse_args()

    root = get_data_root(args.root)
    tasks = list(TASKS) if args.task == "both" else [args.task]
    for task in tasks:
        check_task(task, args.train_method, root, DataConfig(), Path(args.out), (args.slice_min, args.slice_max))


if __name__ == "__main__":
    main()
