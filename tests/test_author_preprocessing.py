"""Regression checks for the dataset authors' CT window and ROI semantics."""
import numpy as np
import pandas as pd
import pytest
import torch

from src.config import DataConfig
from src.data_loader import BTDDataset, window_and_scale


def test_default_window_matches_authors():
    cfg = DataConfig()
    raw = np.array([-3024, -1000, -700, 650, 2000, 3071], dtype=np.float32)
    np.testing.assert_allclose(
        window_and_scale(raw, cfg.hu_min, cfg.hu_max), [0, 0, 0, 0.5, 1, 1]
    )


def test_same_array_with_different_regions_produces_different_inputs(tmp_path):
    # Reproduces the real removal data: same array, different ROI and label.
    raw = np.tile(np.arange(256, dtype=np.float32) * 10 - 700, (256, 1))
    np.save(tmp_path / "slice.npy", raw)
    rows = pd.DataFrame([
        dict(path="slice.npy", scan_id="scan", scanner="scanner", cur_slice=1,
             x=x, y=128, label=label)
        for x, label in [(64, 0), (192, 1)]
    ])
    ds = BTDDataset(rows, tmp_path, DataConfig(image_size=96, imagenet_norm=False))
    real, real_label, real_meta = ds[0]
    fake, fake_label, fake_meta = ds[1]
    assert (real_label, fake_label) == (0, 1)
    assert not torch.equal(real, fake)
    np.testing.assert_allclose(real[0], (raw[80:176, 16:112] + 700) / 2700)
    np.testing.assert_allclose(fake[0], (raw[80:176, 144:240] + 700) / 2700)
    assert (real_meta["x"], real_meta["y"]) == (48, 48)
    assert (fake_meta["x"], fake_meta["y"]) == (48, 48)


@pytest.mark.parametrize("x,y,left,top", [(0, 0, 0, 0), (255, 255, 160, 160),
                                          (10, 200, 0, 152), (200, 10, 152, 0)])
def test_edge_patch_matches_author_boundary_shift(tmp_path, x, y, left, top):
    raw = np.random.default_rng(0).uniform(-700, 2000, (256, 256)).astype(np.float32)
    np.save(tmp_path / "slice.npy", raw)
    df = pd.DataFrame([dict(path="slice.npy", scan_id="s", scanner="m", cur_slice=1,
                           label=1, x=x, y=y)])
    ds = BTDDataset(df, tmp_path, DataConfig(image_size=96, imagenet_norm=False))
    image, _, meta = ds[0]
    expected = (raw[top:top+96, left:left+96] + 700) / 2700
    np.testing.assert_allclose(image[0], expected)
    assert (meta["x"], meta["y"]) == (x-left, y-top)
