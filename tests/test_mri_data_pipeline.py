"""MRI geometry and cross-task splitting regressions; no downloaded data needed."""
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

from src.config import get_data_root
from src.mri_data_loader import (MRIDataConfig, MRIDataset, build_mri_splits,
                                 get_mri_dataloaders, load_mri_patch, load_mri_table)


@pytest.mark.parametrize("height,x,y", [(256, 0, 0), (256, 255, 255), (192, 128, 70)])
def test_standard_reflection_tracks_original_pixel_and_edges(tmp_path, height, x, y):
    raw = np.random.default_rng(1).integers(0, 200, (height, 256), dtype=np.uint8)
    raw[:30] = 255
    raw[-30:] = 0
    path = tmp_path / "image.png"
    Image.fromarray(raw).save(path)
    patch, meta = load_mri_patch(path, x, y, orientation_policy='author_border')
    reflected_y = height - 1 - y
    assert meta['orientation_flipped'] and meta['oriented_y'] == reflected_y
    expected_left = min(max(x - 64, 0), 128)
    expected_top = min(max(reflected_y - 64, 0), height - 128)
    expected = raw[::-1][expected_top:expected_top+128, expected_left:expected_left+128].astype(np.float32) / 255
    np.testing.assert_array_equal(patch, expected)
    assert patch[reflected_y - expected_top, x - expected_left] == np.float32(raw[y, x]) / np.float32(255)
    assert 0 <= meta['oriented_y'] < height


@pytest.mark.parametrize("equal_borders", [True, False])
def test_no_flip_for_tied_or_brighter_bottom_border(tmp_path, equal_borders):
    raw = np.full((256, 256), 64, dtype=np.uint8)
    if not equal_borders:
        raw[-30:] = 128
    path = tmp_path / 'image.png'
    Image.fromarray(raw).save(path)
    patch, meta = load_mri_patch(path, 128, 128, orientation_policy='author_border')
    assert not meta['orientation_flipped'] and meta['oriented_y'] == 128
    np.testing.assert_array_equal(patch, raw[64:192, 64:192].astype(np.float32)/255)
    assert patch.min() == pytest.approx(64/255)  # No per-image min/max or HU window.


@pytest.mark.parametrize("x,y", [(-1, 0), (0, 256), (1.5, 2), (2, float('nan'))])
def test_invalid_roi_rejected_before_reflection(tmp_path, x, y):
    path = tmp_path / 'image.png'
    Image.fromarray(np.zeros((256,256), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match='ROI coordinate'):
        load_mri_patch(path, x, y)


def test_rgb_is_not_silently_converted(tmp_path):
    path = tmp_path / 'rgb.png'
    Image.fromarray(np.zeros((256,256,3), dtype=np.uint8)).save(path)
    with pytest.raises(ValueError, match='mode L'):
        load_mri_patch(path, 128, 128)


def test_default_preserves_native_orientation_even_with_bright_top(tmp_path):
    raw = np.tile(np.arange(256,dtype=np.uint8)[:,None],(1,256))
    raw[:30] = 255
    raw[-30:] = 0
    path = tmp_path / 'image.png'
    Image.fromarray(raw).save(path)
    patch, meta = load_mri_patch(path, 128, 90)
    assert MRIDataConfig().orientation_policy == meta['orientation_policy'] == 'preserve'
    assert not meta['orientation_flipped'] and meta['oriented_y'] == meta['source_y'] == 90
    np.testing.assert_array_equal(patch,raw[26:154,64:192].astype(np.float32)/255)


def test_unknown_orientation_policy_rejected():
    with pytest.raises(ValueError,match='orientation_policy'):
        MRIDataConfig(orientation_policy='unknown')


@pytest.fixture
def mri_root(tmp_path):
    for task, labels in [('injection', ['TM','FM']), ('removal', ['TB','FB'])]:
        rows = []
        for i in range(9):
            # Task-specific inventories differ: splits must still use their union.
            if task == 'removal' and i == 0:
                continue
            folder = f'MRI/scanner{i%3}/Breast_MRI_{i:03d}'
            for label in labels:
                name = '001.png' if label[0] == 'T' else 'injection_001.png' if task == 'injection' else '001_fake.png'
                path = tmp_path / folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                raw = np.tile(np.arange(256,dtype=np.uint8), (256,1))
                Image.fromarray(raw).save(path)
                rows.append(dict(type=label, path=f'{folder}/{name}', x=130, y=140, scanner=f'scanner{i%3}'))
        pd.DataFrame(rows).to_csv(tmp_path / 'MRI' / f'MRI_{task}.csv', index=False)
    return tmp_path


def test_joint_splits_reproducible_and_cross_task_safe(mri_root):
    assert get_data_root(mri_root, modality='MRI') == mri_root  # No CT directory needed.
    frames = []
    for task in ['injection','removal']:
        first = build_mri_splits(task, mri_root)
        second = build_mri_splits(task, mri_root)
        assert set(first) == {'train','val','test'}
        for name, data in first.items():
            pd.testing.assert_frame_equal(data, second[name])
            assert data.label.nunique() == 2
            frames.append(data)
    combined = pd.concat(frames)
    assert combined.groupby('scan_id').split.nunique().max() == 1
    assert combined.groupby('path').split.nunique().max() == 1


def test_dataset_propagates_explicit_reference_orientation_policy(mri_root):
    table = load_mri_table('injection', mri_root).iloc[[0]]
    raw = np.full((256,256),30,dtype=np.uint8)
    raw[:30] = 255
    raw[-30:] = 0
    Image.fromarray(raw).save(mri_root/table.iloc[0].path)
    native = MRIDataset(table,mri_root)[0][2]
    reference = MRIDataset(table,mri_root,MRIDataConfig(orientation_policy='author_border'))[0][2]
    assert not native['orientation_flipped'] and native['oriented_y'] == 140
    assert reference['orientation_flipped'] and reference['oriented_y'] == 115


def test_preserves_multiple_regions_and_rejects_unknown_labels(mri_root):
    path = mri_root / 'MRI/MRI_injection.csv'
    original = pd.read_csv(path)
    additional = original.iloc[[0]].copy()
    additional['x'] = 25
    pd.concat([original, additional]).to_csv(path,index=False)
    table = load_mri_table('injection', mri_root)
    assert len(table) == len(original)+1 and table.path.duplicated().sum() == 1
    original.loc[0,'type'] = 'UNKNOWN'
    original.to_csv(path,index=False)
    with pytest.raises(ValueError, match='unknown class'):
        load_mri_table('injection', mri_root)


def test_cnn_adapter_metadata_and_training_only_augmentation(mri_root):
    cfg = MRIDataConfig(image_size=64)
    loaders = get_mri_dataloaders('injection', mri_root, cfg, batch_size=2)
    for split, loader in loaders.items():
        image, label, meta = next(iter(loader))
        assert image.shape == (2,3,64,64) and image.dtype == torch.float32
        assert torch.isfinite(image).all() and label.dtype == torch.int64
        assert bool(meta['coordinates_valid'].all()) == (split != 'train')
        assert loader.dataset.train == (split == 'train')
    val = loaders['val']
    assert isinstance(val.sampler, SequentialSampler)
    torch.testing.assert_close(val.dataset[0][0], val.dataset[0][0], rtol=0, atol=0)
    def forbidden_augmentation(_):
        raise AssertionError('Validation must not augment')
    val.dataset.augment = forbidden_augmentation
    image, _, meta = val.dataset[0]
    assert meta['x'] == meta['y'] == 31.75
    assert meta['coordinate_convention'] == 'pixel_center' and meta['coordinates_valid']
    train = loaders['train'].dataset
    called = []
    train.augment = lambda tensor: (called.append(True) or tensor)
    train[0]
    assert called == [True]
    unnormalized = MRIDataset(val.dataset.df, mri_root, MRIDataConfig(image_size=64,imagenet_norm=False))[0][0]
    expected = (unnormalized-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    torch.testing.assert_close(image, expected)
