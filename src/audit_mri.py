"""Historical border-rule MRI audit against original/corrected references.

Uses author_border explicitly, not the current preserve-orientation default.
Requires a local copy of the pinned BTD notebook (not included in this repo).
No training or model evaluation. Detailed row/image exports remain local.
"""
import argparse
import ast
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor

from .config import get_data_root
from .mri_data_loader import LABELS, MRIDataConfig, MRIDataset, build_mri_splits, get_mri_dataloaders

NOTEBOOK_SHA256 = 'a0c06362bb7b6e525b7061bf431145e33cbc8cf3d2cda06469831ca4ccda2892'


def reference_classes(notebook, root):
    """Execute only the audited dataset class from the pinned upstream notebook."""
    content = notebook.read_bytes()
    if hashlib.sha256(content).hexdigest() != NOTEBOOK_SHA256:
        raise ValueError('Reference notebook does not match audited commit 0fa74da')
    nb = json.loads(content)
    source = next(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'
                  and 'class MedicalDeepfakeDataset' in ''.join(c['source']))
    if source.count('y = 512-y') != 1:
        raise ValueError('Expected exactly one upstream reflection expression')
    classes = []
    for code in (source, source.replace('y = 512-y', 'y = image.shape[0] - 1 - y')):
        module = ast.parse(code)
        assert len(module.body) == 1 and isinstance(module.body[0], ast.ClassDef)
        ns = dict(Dataset=Dataset, os=os, pd=pd, np=np, torch=torch, Image=Image, root_dir=str(root))
        exec(compile(module, str(notebook), 'exec'), ns)
        classes.append(ns['MedicalDeepfakeDataset'])
    return classes


def audit(root, notebook, out):
    root = get_data_root(root, modality='MRI')
    out.mkdir(parents=True, exist_ok=True)
    original_cls, corrected_cls = reference_classes(notebook, root)
    # Historical compatibility audit, deliberately separate from the current
    # preserve-orientation default. See audit_mri_mitigation for that policy.
    cfg = MRIDataConfig(orientation_policy='author_border')
    records, summaries = [], {}
    for task in LABELS:
        splits = build_mri_splits(task, root, cfg)
        frame = pd.concat(splits.values(), ignore_index=True)
        reference = original_cls(str(root), 'MRI', task, None, 128, ToTensor())
        corrected = corrected_cls(str(root), 'MRI', task, None, 128, ToTensor())
        reference.data_frame = corrected.data_frame = frame
        native = MRIDataset(frame, root, replace(cfg, image_size=128, imagenet_norm=False))
        cnn = MRIDataset(frame, root, cfg)
        for i, row in frame.iterrows():
            image, label, meta = native[i]
            expected = corrected[i]['image']
            upstream = reference[i]['image']
            np.testing.assert_array_equal(image.numpy(), np.repeat(expected, 3, axis=0))
            assert image.dtype == torch.float32 and torch.isfinite(image).all()
            assert 0 <= image.min() <= image.max() <= 1
            assert label == LABELS[task][corrected[i]['label']]
            assert meta['oriented_y'] == (meta['source_height']-1-row.y if meta['orientation_flipped'] else row.y)
            assert 0 <= meta['x'] < 128 and 0 <= meta['y'] < 128
            assert meta['x'] == row.x-meta['crop_left']
            assert meta['y'] == meta['oriented_y']-meta['crop_top']
            same_original = bool(np.array_equal(image[0], upstream[0]))
            if not meta['orientation_flipped']:
                assert same_original
            tensor, _, cnn_meta = cnn[i]
            assert tensor.shape == (3,224,224) and torch.isfinite(tensor).all()
            assert cnn_meta['x'] == (meta['x']+.5)*224/128-.5
            assert cnn_meta['y'] == (meta['y']+.5)*224/128-.5
            with Image.open(root / row.path) as im:
                raw = np.array(im)
                assert im.mode == 'L' and im.size == (512,512)
            digest = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
            records.append(dict(task=task, path=row.path, scan_id=row.scan_id,
                split=row.split, label=label, cur_slice=int(row.cur_slice),
                orientation_flipped=meta['orientation_flipped'],
                original_patch_equal=same_original, raw_hash=digest(raw),
                native_hash=digest(image[0].numpy()), cnn_hash=digest(tensor.numpy()),
                mean=float(image[0].mean()), zero_fraction=float((image[0] == 0).float().mean())))
        batch_counts = {}
        for train in (False, True):
            count = 0
            for x, y, meta in DataLoader(MRIDataset(frame, root, cfg, train=train), batch_size=16):
                assert x.shape[1:] == (3,224,224) and x.dtype == torch.float32 and torch.isfinite(x).all()
                assert y.dtype == torch.int64 and len(y) == len(meta['path'])
                assert bool(meta['coordinates_valid'].all()) == (not train)
                count += len(y)
            assert count == len(frame)
            batch_counts['train' if train else 'eval'] = count
        public_loaders = get_mri_dataloaders(task, root, cfg, batch_size=16)
        assert set(public_loaders) == {'train','val','test'}
        for name, loader in public_loaders.items():
            x, y, meta = next(iter(loader))
            assert len(loader.dataset) == len(splits[name]) and x.shape[1:] == (3,224,224)
            assert torch.isfinite(x).all() and bool(meta['coordinates_valid'].all()) == (name != 'train')
        summaries[task] = dict(rows=len(frame), batch_rows=batch_counts,
            splits={name: dict(scans=d.scan_id.nunique(), authentic=int((d.label == 0).sum()),
                               fake=int((d.label == 1).sum())) for name,d in splits.items()})
    rows = pd.DataFrame(records)
    rows.to_csv(out / 'rows.csv', index=False)
    disk = {str(p.relative_to(root)) for p in (root/'MRI').rglob('*.png')}
    assert disk == set(rows.path)
    assert rows.groupby('scan_id').split.nunique().max() == 1
    hashes = {}
    for field in ['raw_hash','native_hash','cnn_hash']:
        cross_split = int((rows.groupby(field).split.nunique() > 1).sum())
        assert cross_split == 0
        hashes[field] = dict(cross_split_groups=cross_split)
        for task in LABELS:
            subset = rows[rows.task == task]
            conflicts = int((subset.groupby(field).label.nunique() > 1).sum())
            assert conflicts == 0
            hashes[field][task+'_conflicting_label_groups'] = conflicts
    for task in LABELS:
        data = rows[rows.task == task]
        pairs = data[data.label == 0].merge(data[data.label == 1], on=['scan_id','cur_slice'],
                                           suffixes=('_real','_fake'), validate='one_to_one')
        summaries[task]['paired_flip_disagreements'] = int((pairs.orientation_flipped_real != pairs.orientation_flipped_fake).sum())
        summaries[task]['patch_statistics_by_label'] = data.groupby('label')[['mean','zero_fraction']].median().to_dict('index')
    report = dict(passed=True, scope='Historical border-rule MRI engineering; not the preserve-orientation default',
        dataset_version=7, config=asdict(cfg), unique_files=len(disk), rows=len(rows),
        source_groups=rows.scan_id.nunique(), reference_commit='0fa74daef6802d305dcca4e5e37b4393fdf89c5b',
        notebook_sha256=NOTEBOOK_SHA256, reference_change='y = 512-y -> y = image.shape[0] - 1 - y',
        corrected_reference_exact_matches=len(rows), original_reference_exact_matches=int(rows.original_patch_equal.sum()),
        different_from_original=int((~rows.original_patch_equal).sum()), flipped_rows=int(rows.orientation_flipped.sum()),
        hash_checks=hashes, tasks=summaries,
        source_sha256={str(p.relative_to(Path(__file__).parent.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), Path(__file__).with_name('mri_data_loader.py'),
                                 Path(__file__).with_name('config.py'), Path(__file__).with_name('data_loader.py'),
                                 Path(__file__).with_name('splits.py')]},
        manifest_sha256={f'MRI_{task}.csv': hashlib.sha256((root/'MRI'/f'MRI_{task}.csv').read_bytes()).hexdigest()
                         for task in LABELS},
        caveats=['Image-dependent orientation heuristic still disagrees for seven paired source slices.',
                 'Only 25 source groups; patient identity not independently verified.',
                 'MRI is SD-only; train/val/test share source assignments across tasks.',
                 'Random training augmentation invalidates returned ROI coordinates.',
                 'MRI shortcut assessment remains a separate increment.'])
    (out/'summary.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path)
    parser.add_argument('--reference-notebook', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=Path('results/mri_audit'))
    args = parser.parse_args()
    result = audit(args.root, args.reference_notebook, args.out)
    print(json.dumps({k:result[k] for k in ['passed','unique_files','rows','corrected_reference_exact_matches',
                                          'original_reference_exact_matches','different_from_original']}, indent=2))
