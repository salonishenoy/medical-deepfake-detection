"""Exhaustive CT data checks; no detector training or interpretability work.

Run: python -m src.audit_data --root /path/to/archive --out results/ct_audit
Detailed manifests stay local. summary.json is an aggregate handoff artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from torch.utils.data import DataLoader

from .config import DataConfig, get_data_root
from .data_loader import BTDDataset
from .splits import TASKS, assign_scan_splits, build_splits, check_no_leakage, load_task_table, summarize


def reference_patch(raw, x, y):
    """Independent reference: BTD_evaluation.ipynb, commit 0fa74da, dataset class.

    The authors normalize BEFORE casting to float32 and shift edge ROIs inward.
    This helper deliberately does not call the pipeline's window/crop functions.
    """
    image = ((np.clip(raw, -700, 2000) + 700) / 2700).astype(np.float32)
    x, y = int(x), int(y)
    if 48 - y > 0:
        y += 48 - y
    if 48 - x > 0:
        x += 48 - x
    if y + 48 > image.shape[0]:
        y -= abs(y + 48 - image.shape[0])
    if x + 48 > image.shape[1]:
        x -= abs(x + 48 - image.shape[1])
    return image[y - 48:y + 48, x - 48:x + 48]


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def duplicate_summary(frame, column):
    groups = frame.groupby(column)
    duplicated = groups.filter(lambda g: len(g) > 1)
    conflicts = groups.filter(lambda g: g.label.nunique() > 1)
    leakage = groups.filter(lambda g: g.split.nunique() > 1)
    return {"duplicate_groups": int(duplicated[column].nunique()),
            "conflicting_label_groups": int(conflicts[column].nunique()),
            "cross_split_groups": int(leakage[column].nunique())}, conflicts, leakage


def shortcut_checks(frame):
    """Depth-3 nuisance-feature probes fitted on train and scored on validation."""
    result = {}
    for method, flag in (("CTGAN", "in_ctgan"), ("SD", "in_sd")):
        subset = frame[frame[flag] == 1]
        train, val = subset[subset.split == "train"], subset[subset.split == "val"]
        if min(train.label.nunique(), val.label.nunique()) < 2:
            result[method] = {"skipped": "One class missing"}
            continue
        probes = {"slice": ["cur_slice"], "brightness": ["mean"],
                  "texture": ["std", "gradient"],
                  "clipping": ["zero_fraction", "one_fraction"],
                  "image_statistics": ["mean", "std", "gradient", "zero_fraction", "one_fraction"]}
        scores = {}
        for name, columns in probes.items():
            clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(train[columns], train.label)
            scores[name] = float(roc_auc_score(val.label, clf.predict_proba(val[columns])[:, 1]))
        train_scanner = pd.get_dummies(train.scanner)
        val_scanner = pd.get_dummies(val.scanner).reindex(columns=train_scanner.columns, fill_value=False)
        clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(train_scanner, train.label)
        scores["scanner"] = float(roc_auc_score(val.label, clf.predict_proba(val_scanner)[:, 1]))
        result[method] = scores
    return result


def audit_task(task, root, out):
    cfg = DataConfig()
    frame = load_task_table(task, root)
    frame["split"] = frame.scan_id.map(assign_scan_splits(frame, cfg))
    expected_labels = {"injection": {"TM": 0, "FM": 1}, "removal": {"TB": 0, "FB": 1}}[task]
    if not frame.type.isin(expected_labels).all():
        raise ValueError(f"Unknown class in {task}")
    assert (frame.label == frame.type.map(expected_labels)).all()
    assert frame[["in_ctgan", "in_sd"]].isin([0, 1]).all().all()
    assert frame[["in_ctgan", "in_sd"]].sum(axis=1).ge(1).all()
    native = BTDDataset(frame, root, replace(cfg, image_size=96, imagenet_norm=False))
    cnn = BTDDataset(frame, root, cfg)
    findings, errors, file_info = [], [], {}
    max_error = 0.0
    edge_count = 0
    for i, row in frame.iterrows():
        try:
            raw = np.load(root / row.path, allow_pickle=False)
            if raw.ndim != 2 or not np.isfinite(raw).all():
                raise ValueError(f"Invalid array: shape={raw.shape}, finite={np.isfinite(raw).all()}")
            h, w = raw.shape
            if not (np.isfinite(row.x) and np.isfinite(row.y) and 0 <= row.x < w and 0 <= row.y < h):
                raise ValueError("Invalid CSV coordinates")
            image, label, meta = native[i]
            patch = image[0].numpy()
            expected = reference_patch(raw, row.x, row.y)
            assert patch.shape == (96, 96)
            assert image.dtype == torch.float32 and torch.isfinite(image).all()
            assert 0 <= patch.min() <= patch.max() <= 1
            np.testing.assert_allclose(patch, expected, rtol=1e-6, atol=1e-7)
            max_error = max(max_error, float(np.abs(patch - expected).max()))
            assert label == expected_labels[row.type]
            assert meta["x"] == int(row.x) - meta["crop_left"]
            assert meta["y"] == int(row.y) - meta["crop_top"]
            assert 0 <= meta["x"] < 96 and 0 <= meta["y"] < 96
            edge_count += int((meta["x"], meta["y"]) != (48, 48))
            full, _, cnn_meta = cnn[i]
            assert full.shape == (3, 224, 224) and torch.isfinite(full).all()
            restored = full * cnn.std + cnn.mean
            assert restored.min() >= -1e-6 and restored.max() <= 1.000001
            assert abs(cnn_meta["x"] * 96 / 224 + meta["crop_left"] - int(row.x)) < 1e-6
            assert abs(cnn_meta["y"] * 96 / 224 + meta["crop_top"] - int(row.y)) < 1e-6
            if row.path not in file_info:
                file_info[row.path] = {"dtype": str(raw.dtype), "shape": list(raw.shape),
                                       "raw_hash": digest(raw.astype(np.float32))}
            findings.append(dict(row_id=int(i), path=row.path, label=int(label), type=row.type,
                x=int(row.x), y=int(row.y), scan_id=row.scan_id, scanner=row.scanner,
                cur_slice=int(row.cur_slice), split=row.split, in_ctgan=int(row.in_ctgan), in_sd=int(row.in_sd),
                native_hash=digest(patch), cnn_hash=digest(full.numpy()),
                raw_hash=file_info[row.path]["raw_hash"], mean=float(patch.mean()), std=float(patch.std()),
                gradient=float((np.abs(np.diff(patch, axis=0)).mean() + np.abs(np.diff(patch, axis=1)).mean()) / 2),
                zero_fraction=float((patch == 0).mean()), one_fraction=float((patch == 1).mean())))
        except Exception as error:
            errors.append({"row_id": int(i), "path": row.path, "error": str(error)[:800]})
        if (i + 1) % 500 == 0:
            print(f"{task}: {i + 1}/{len(frame)} rows checked", flush=True)
    records = pd.DataFrame(findings)
    records.to_csv(out / f"{task}_rows.csv", index=False)
    summary = {"csv_rows": len(frame), "unique_files": frame.path.nunique(), "scans": frame.scan_id.nunique(),
        "exact_duplicate_csv_rows": int(frame.duplicated().sum()), "rows_checked": len(records),
        "errors": len(errors), "max_reference_absolute_error": max_error, "edge_shift_rows": edge_count,
        "raw_dtype_file_counts": pd.Series([r["dtype"] for r in file_info.values()]).value_counts().to_dict()}
    (out / f"{task}_errors.json").write_text(json.dumps(errors, indent=2))
    summary["hash_checks"] = {}
    for column in ("native_hash", "cnn_hash", "raw_hash"):
        values, conflicts, leakage = duplicate_summary(records, column)
        summary["hash_checks"][column] = values
        conflicts.to_csv(out / f"{task}_{column}_label_conflicts.csv", index=False)
        leakage.to_csv(out / f"{task}_{column}_split_overlap.csv", index=False)
    summary["splits"] = {}
    for method in ("CTGAN", "SD"):
        splits = build_splits(task, method, root)
        check_no_leakage(splits)
        summary["splits"][method] = summarize(splits).to_dict("index")
        for name, split in splits.items():
            split.to_csv(out / f"{task}_{method}_{name}.csv", index=False)
    summary["validation_shortcut_auc"] = shortcut_checks(records) if not errors else {}
    summary["batch_rows"] = {}
    if not errors:
        for train in (False, True):
            torch.manual_seed(42)
            seen = 0
            for images, labels, meta in DataLoader(BTDDataset(frame, root, cfg, train=train), batch_size=32):
                assert images.shape[1:] == (3, 224, 224) and torch.isfinite(images).all()
                assert labels.dtype == torch.int64 and len(labels) == len(meta["path"])
                seen += len(labels)
            summary["batch_rows"]["train" if train else "eval"] = seen
    # Whole-array conflicts are expected for different ROI labels. Patch conflicts are not.
    summary["passed"] = not errors and all(
        summary["hash_checks"][c]["conflicting_label_groups"] == 0 and
        summary["hash_checks"][c]["cross_split_groups"] == 0
        for c in ("native_hash", "cnn_hash"))
    print(task, "complete:", summary["passed"], flush=True)
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default="results/ct_audit")
    args = parser.parse_args()
    torch.set_num_threads(1)
    root, out = get_data_root(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"config": asdict(DataConfig()), "reference_commit": "0fa74daef6802d305dcca4e5e37b4393fdf89c5b",
              "tasks": {}}
    records = {}
    for task in TASKS:
        report["tasks"][task], records[task] = audit_task(task, root, out)
    cross = {}
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            cross[a + "/" + b] = len(set(records["injection"].query("split == @a").scan_id) &
                                      set(records["removal"].query("split == @b").scan_id))
    report["cross_task_scan_overlap_injection_vs_removal"] = cross
    report["passed"] = all(t["passed"] for t in report["tasks"].values())
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    print("Summary:", out / "summary.json", flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
