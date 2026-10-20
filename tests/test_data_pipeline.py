"""Tests for the data pipeline, using a tiny synthetic copy of the BTD layout.

Run from the repo root:   python -m pytest tests -q
No real data is needed or touched.
"""
import numpy as np
import pandas as pd
import pytest

from src.config import DataConfig
from src.data_loader import BTDDataset, get_dataloaders, load_slice, window_and_scale
from src.splits import build_splits, check_no_leakage, load_task_table

SCANNERS = ["LightSpeed16", "Sensation 16", "LightSpeed QX"]
N_SCANS = 18
SLICES = [20, 50, 90, 200]  # 200 is outside the 34-132 range


def _make_task(root, task):
    """Write fake slices + a combined CSV shaped like CT_injection.csv / CT_removal.csv."""
    fake_t, real_t = ("FM", "TM") if task == "injection" else ("FB", "TB")
    ctgan_col, sd_col = f"CTGAN-CT-{task}", f"SD-CT-{task}"
    rng = np.random.default_rng(0)
    rows = []
    for s in range(N_SCANS):
        scan = f"1.3.6.1.4.1.14519.5.2.1.6279.6001.{1000 + s}"
        scanner = SCANNERS[s % len(SCANNERS)]
        for sl in SLICES:
            for folder, typ, flags, dtype in (
                (real_t, real_t, (1, 1), np.int16),
                (f"{fake_t}_CTGAN", fake_t, (1, 0), np.int16),
                (f"{fake_t}_SD", fake_t, (0, 1), np.float64),
            ):
                rel = f"CT/{task}/{folder}/{scan}/{sl}.npy"
                f = root / rel
                f.parent.mkdir(parents=True, exist_ok=True)
                np.save(f, rng.integers(-1000, 400, (64, 64)).astype(dtype))
                row = dict(type=typ, path=rel, x=40, y=20, scanner=scanner, cur_slice=sl)
                row[ctgan_col], row[sd_col] = flags
                rows.append(row)
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[[1]]])  # a duplicated path (second lesion on a slice)
    df.iloc[-1, df.columns.get_loc("x")] = 10
    df.to_csv(root / "CT" / task / f"CT_{task}.csv", index=False)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    r = tmp_path_factory.mktemp("btd")
    for task in ("injection", "removal"):
        _make_task(r, task)
    return r


@pytest.mark.parametrize("task", ["injection", "removal"])
@pytest.mark.parametrize("method", ["CTGAN", "SD"])
def test_no_scan_leakage(root, task, method):
    splits = build_splits(task, method, root)
    check_no_leakage(splits)  # raises on leakage
    assert all(len(d) > 0 for d in splits.values())
    # the generalization test only uses held-out scans
    held_out = set(splits["train"]["scan_id"]) | set(splits["val"]["scan_id"])
    assert not set(splits["test_other"]["scan_id"]) & held_out


def test_methods_are_separated(root):
    s = build_splits("injection", "CTGAN", root)
    assert s["train"]["path"].str.contains("FM_SD").sum() == 0
    assert s["test_other"]["path"].str.contains("FM_CTGAN").sum() == 0
    assert (s["test_other"]["label"] == 1).any() and (s["test_other"]["label"] == 0).any()


def test_duplicates_merged_and_coords_kept(root):
    df = load_task_table("injection", root)
    assert df["path"].is_unique
    dup = df[df["xs"].str.len() == 2]
    assert len(dup) == 1 and set(dup.iloc[0]["xs"]) == {40, 10}


def test_split_is_reproducible(root):
    a = build_splits("removal", "CTGAN", root)["test_same"]["scan_id"].tolist()
    b = build_splits("removal", "CTGAN", root)["test_same"]["scan_id"].tolist()
    assert a == b


def test_slice_filter_off_by_default_and_works_when_on(root):
    full = build_splits("injection", "CTGAN", root)
    assert (pd.concat(full.values())["cur_slice"] == 200).any()
    filt = build_splits("injection", "CTGAN", root, DataConfig(slice_range=(34, 132)))
    assert not (pd.concat(filt.values())["cur_slice"] == 200).any()
    # the scan assignment must not change when the filter is switched on
    assert set(filt["test_same"]["scan_id"]) == set(full["test_same"]["scan_id"])


def test_window_and_scale():
    out = window_and_scale(np.array([-3000.0, -1000.0, -300.0, 400.0, 2000.0]), -1000, 400)
    assert out.min() == 0.0 and out.max() == 1.0 and out[2] == pytest.approx(0.5)


def test_item_shape_range_and_dtype_normalised(root):
    cfg = DataConfig(image_size=32, imagenet_norm=False)
    splits = build_splits("injection", "SD", root, cfg)  # SD fakes are float64 on disk
    ds = BTDDataset(splits["train"], root, cfg, train=False)
    for i in range(len(ds)):
        img, label, meta = ds[i]
        assert img.shape == (3, 32, 32) and img.dtype.is_floating_point
        assert 0.0 <= img.min() and img.max() <= 1.0
        assert torch_equal_channels(img)
        if label == 1:
            assert meta["x"] == pytest.approx(40 * 32 / 64) and meta["y"] == pytest.approx(20 * 32 / 64)
        else:
            assert meta["x"] == -1.0 and meta["y"] == -1.0


def torch_equal_channels(img):
    return bool((img[0] == img[1]).all() and (img[1] == img[2]).all())


def test_dataloader_batches_and_augmentation_only_in_train(root):
    cfg = DataConfig(image_size=32)
    loaders = get_dataloaders("removal", "CTGAN", root, cfg, batch_size=4)
    assert set(loaders) == {"train", "val", "test_same", "test_other"}
    x, y, meta = next(iter(loaders["train"]))
    assert x.shape[1:] == (3, 32, 32) and set(y.tolist()) <= {0, 1}
    assert isinstance(meta["path"][0], str)
    assert loaders["train"].dataset.train and not loaders["val"].dataset.train
    # val items are deterministic (no augmentation)
    a = loaders["val"].dataset[0][0]
    b = loaders["val"].dataset[0][0]
    assert (a == b).all()


def test_load_slice_casts_to_float32(root):
    f = next((root / "CT" / "injection").rglob("*.npy"))
    assert load_slice(f).dtype == np.float32
