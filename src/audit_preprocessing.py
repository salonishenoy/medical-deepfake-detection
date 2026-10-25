"""Complete mechanical audit of frozen Team A preprocessing. No model scores.

python -m src.audit_preprocessing --root /path/to/archive --out results/preprocessing_v1
Detailed sample records stay local; summary.json contains aggregate evidence.
"""
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from .config import DataConfig
from .data_loader import BTDDataset, get_dataloaders, IMAGENET_MEAN, IMAGENET_STD
from .mri_data_loader import MRIDataConfig, MRIDataset, build_mri_splits, get_mri_dataloaders
from .splits import assign_scan_splits, load_task_table

REPO = Path(__file__).resolve().parents[1]


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_snapshot(contract, root, repo=REPO):
    require(contract['contract_id'] == 'team-a-preprocessing-v1', 'Unsupported contract ID')
    require(contract['defaults'] == {'CT':asdict(DataConfig()),'MRI':asdict(MRIDataConfig())},
            'Runtime defaults differ from frozen contract')
    for name,digest in contract['provenance']['source_sha256'].items():
        require(file_digest(repo/name) == digest, f'Frozen source changed: {name}')
    for name,digest in contract['dataset']['manifest_sha256'].items():
        require(file_digest(root/name) == digest, f'Dataset manifest changed: {name}')


def hash_summary(frame, column):
    grouped = frame.groupby(column)
    return dict(duplicate_groups=int((grouped.size()>1).sum()),
                conflicting_label_groups=int((grouped.label.nunique()>1).sum()),
                cross_split_groups=int((grouped.split.nunique()>1).sum()))


def sample_key(path,x,y,label):
    return str(path),int(x),int(y),int(label)


def audit_loader(loader, modality, schema, reference, split_name):
    """Verify the actual delivered multiset, metadata, and unaugmented tensors."""
    training = split_name == 'train'
    require(loader.dataset.train == training, 'Wrong training/evaluation mode')
    require(loader.batch_size == 32 and loader.num_workers == 0 and not loader.drop_last,
            'Public loader defaults differ from contract')
    require(isinstance(loader.sampler,RandomSampler if training else SequentialSampler), 'Wrong shuffle policy')
    transforms = loader.dataset.augment.transforms
    require([type(t).__name__ for t in transforms] == ['RandomHorizontalFlip','RandomVerticalFlip','RandomRotation'],
            'Augmentation order changed')
    require(transforms[0].p == transforms[1].p == .5 and transforms[2].degrees == [-10.,10.]
            and transforms[2].interpolation.value == 'nearest' and transforms[2].fill == 0
            and not transforms[2].expand, 'Augmentation defaults changed')
    expected = Counter(sample_key(r.path,r.x,r.y,r.label) for r in loader.dataset.df.itertuples())
    delivered, count = Counter(), 0
    for images,labels,meta in loader:
        require(images.dtype == torch.float32 and images.shape[1:] == (3,224,224)
                and bool(torch.isfinite(images).all()), 'Invalid image batch')
        require(labels.dtype == torch.int64 and labels.ndim == 1 and len(labels) == len(images), 'Invalid labels')
        require(set(meta) == set(schema), 'Batch metadata fields differ from contract')
        require(all(len(value) == len(labels) for value in meta.values()), 'Metadata batch length mismatch')
        require(bool((meta['coordinates_valid'] == (not training)).all()), 'Coordinate validity flag mismatch')
        restored = images*torch.tensor(IMAGENET_STD)[None,:,None,None]+torch.tensor(IMAGENET_MEAN)[None,:,None,None]
        require(float(restored.min()) >= -1e-6 and float(restored.max()) <= 1.000001, 'Invalid scaled intensities')
        for i in range(len(labels)):
            key = sample_key(meta['path'][i],meta['source_x'][i],meta['source_y'][i],labels[i])
            require(key in reference, 'Public loader delivered unexpected sample')
            ref = reference[key]
            for field,value in ref['meta'].items():
                actual = meta[field][i]
                if isinstance(actual,torch.Tensor):
                    actual = actual.item()
                require(actual == (not training) if field == 'coordinates_valid' else actual == value,
                        f'Public metadata mismatch: {field}')
            if not training:
                require(array_digest(images[i].numpy()) == ref['cnn_hash'], 'Evaluation tensor differs from audited item')
            delivered[key] += 1
        count += len(labels)
    require(delivered == expected, 'Public loader dropped, duplicated, or substituted samples')
    return dict(rows=count,authentic=sum(n for k,n in delivered.items() if k[-1] == 0),
                fake=sum(n for k,n in delivered.items() if k[-1] == 1),
                source_groups=loader.dataset.df.scan_id.nunique(),
                mode='training augmentation' if training else 'deterministic evaluation',
                delivered_sample_multiset_matches=True)


def run(root, contract_path, out):
    contract = json.loads(contract_path.read_text())
    contract_sha = file_digest(contract_path)
    verify_snapshot(contract,root)
    records, reports, references, image_inventory = [],{}, {}, {}
    for modality in ['CT','MRI']:
        reports[modality] = {}
        for task in ['injection','removal']:
            if modality == 'CT':
                cfg,cls = DataConfig(),BTDDataset
                table = load_task_table(task,root)
                table['split'] = table.scan_id.map(assign_scan_splits(table,cfg))
            else:
                cfg,cls = MRIDataConfig(),MRIDataset
                table = pd.concat(build_mri_splits(task,root,cfg).values(),ignore_index=True)
            labels = {'TM':0,'FM':1} if task == 'injection' else {'TB':0,'FB':1}
            require(table.type.isin(labels).all() and (table.label == table.type.map(labels)).all(), 'Unknown/mismapped labels')
            manifest_path = root/(f'CT/{task}/CT_{task}.csv' if modality == 'CT' else f'MRI/MRI_{task}.csv')
            raw_manifest = pd.read_csv(manifest_path)
            require(len(table) == len(raw_manifest), 'CSV sample rows were removed/merged')
            require(table[['path','x','y','scanner','scan_id','split']].notna().all().all(), 'Missing metadata')
            require(table.groupby('scan_id').split.nunique().max() == 1, 'Source split leakage')
            if modality == 'CT':
                for flag in ['in_ctgan','in_sd']:
                    require(table[flag].isin([0,1]).all(), 'Invalid method flag')
                require((table[['in_ctgan','in_sd']].sum(axis=1)>0).all(), 'Sample belongs to no fake-method subset')
            native = cls(table,root,replace(cfg,image_size=cfg.patch_size,imagenet_norm=False))
            cnn = cls(table,root,cfg)
            schema = contract['metadata']['item_field_types'][modality]
            refmap,max_error,max_inverse,edge_rows = {},0.0,0.0,0
            for i,row in table.iterrows():
                path = root/row.path
                if modality == 'CT':
                    raw = np.load(path,allow_pickle=False)
                    require(raw.ndim == 2 and np.isfinite(raw).all(), 'Invalid CT source array')
                    scaled = ((np.clip(raw,-700,2000)+700)/2700).astype(np.float32)
                else:
                    with Image.open(path) as im:
                        require(im.mode == 'L', 'MRI must be grayscale mode L')
                        raw = np.array(im)
                    require(raw.dtype == np.uint8, 'MRI must be uint8')
                    scaled = raw.astype(np.float32)/255
                require(list(raw.shape) == contract['inputs'][modality]['observed_shape'], 'Unexpected released image shape')
                x,y = int(row.x),int(row.y)
                require(x == row.x and y == row.y and 0 <= x < raw.shape[1] and 0 <= y < raw.shape[0], 'Invalid CSV coordinates')
                p = cfg.patch_size
                left,top = min(max(x-p//2,0),raw.shape[1]-p),min(max(y-p//2,0),raw.shape[0]-p)
                expected = scaled[top:top+p,left:left+p]
                image,label,meta = native[i]
                require(image.shape == (3,p,p) and image.dtype == torch.float32, 'Invalid native tensor')
                np.testing.assert_allclose(image[0].numpy(),expected,atol=1e-7,rtol=1e-6)
                require(torch.equal(image[0],image[1]) and torch.equal(image[1],image[2]), 'Native channels differ')
                if modality == 'MRI':
                    np.testing.assert_array_equal(image[0].numpy(),expected)
                    require(not meta['orientation_flipped'] and meta['orientation_policy'] == 'preserve', 'MRI default orientation changed')
                    require(meta['oriented_x'] == x and meta['oriented_y'] == y, 'MRI point changed')
                max_error = max(max_error,float(np.abs(image[0].numpy()-expected).max()))
                require(label == int(row.label) and meta['source_x'] == x and meta['source_y'] == y, 'Label/source point mismatch')
                require(meta['crop_left'] == left and meta['crop_top'] == top and meta['x'] == x-left and meta['y'] == y-top, 'Native crop geometry mismatch')
                full,full_label,cm = cnn[i]
                require(full.shape == (3,224,224) and full.dtype == torch.float32 and torch.isfinite(full).all(), 'Invalid CNN tensor')
                require(full_label == label and {k:type(v).__name__ for k,v in cm.items()} == schema, 'Item metadata types mismatch')
                for axis,origin,source in [('x',left,x),('y',top,y)]:
                    require(cm[axis] == (source-origin+.5)*224/p-.5, 'Resized coordinate mismatch')
                    require(abs((cm[axis]+.5)*p/224-.5+origin-source)<1e-6, 'Coordinate inversion mismatch')
                require(cm['coordinate_convention'] == 'pixel_center' and cm['coordinates_valid'], 'Coordinate semantics mismatch')
                resized = F.interpolate(image[None],size=(224,224),mode='bilinear',align_corners=False,antialias=True)[0]
                inverse = full*cnn.std+cnn.mean
                torch.testing.assert_close(inverse,resized,atol=2e-7,rtol=1e-6)
                max_inverse = max(max_inverse,float((inverse-resized).abs().max()))
                if i%100 == 0:
                    torch.testing.assert_close(full,cnn[i][0],atol=0,rtol=0)
                edge_rows += int(left != x-p//2 or top != y-p//2)
                key = sample_key(row.path,x,y,label)
                entry = dict(meta=cm,cnn_hash=array_digest(full.numpy()))
                if key in refmap:
                    require(refmap[key] == entry, 'Repeated sample has inconsistent outputs')
                refmap[key] = entry
                record = dict(modality=modality,task=task,path=row.path,scan_id=row.scan_id,split=row.split,
                    label=label,source_x=x,source_y=y,crop_left=left,crop_top=top,x=cm['x'],y=cm['y'],
                    raw_hash=array_digest(raw),native_hash=array_digest(image[0].numpy()),cnn_hash=entry['cnn_hash'])
                records.append(record)
                if (i+1)%500 == 0:
                    print(f'{modality}/{task}: {i+1}/{len(table)} rows checked',flush=True)
            references[modality,task] = refmap
            reports[modality][task] = dict(rows=len(table),unique_files=table.path.nunique(),source_groups=table.scan_id.nunique(),
                exact_duplicate_csv_rows=int(raw_manifest.duplicated().sum()),edge_shift_rows=edge_rows,
                native_max_abs_reference_error=max_error,normalization_inverse_max_abs_error=max_inverse,
                labels=table.type.value_counts().to_dict(),held_out_rows=int((table.split == 'test').sum()))
            print(f'{modality}/{task}: all {len(table)} row checks passed',flush=True)
        referenced = {r['path'] for r in records if r['modality'] == modality}
        disk = {str(p.relative_to(root)) for p in (root/modality).rglob('*.npy' if modality == 'CT' else '*.png')}
        require(disk == referenced, f'{modality}: missing or unreferenced image files')
        image_inventory[modality] = dict(files=len(disk),missing_files=0,unreferenced_files=0)
    rows = pd.DataFrame(records)
    rows.to_csv(out/'sample_checks.csv',index=False)
    hashes = {}
    for (modality,task),frame in rows.groupby(['modality','task']):
        checks = {field:hash_summary(frame,field) for field in ['raw_hash','native_hash','cnn_hash']}
        require(all(checks[c]['cross_split_groups'] == 0 for c in checks), 'Exact image crosses splits')
        require(all(checks[c]['conflicting_label_groups'] == 0 for c in ['native_hash','cnn_hash']), 'Exact input has conflicting labels')
        hashes[f'{modality}/{task}'] = checks
    mri = rows[rows.modality == 'MRI']
    require(mri.groupby('scan_id').split.nunique().max() == 1, 'MRI cross-task source leakage')
    for field in ['raw_hash','native_hash','cnn_hash']:
        require(hash_summary(mri,field)['cross_split_groups'] == 0, 'MRI cross-task exact-image leakage')
    ct = rows[rows.modality == 'CT']
    a,b = ct[ct.task == 'injection'],ct[ct.task == 'removal']
    overlap = {f'{sa}/{sb}':len(set(a[a.split == sa].scan_id)&set(b[b.split == sb].scan_id))
               for sa in ['train','val','test'] for sb in ['train','val','test']}
    interfaces = []
    for modality in ['CT','MRI']:
        for task in ['injection','removal']:
            for method in (['CTGAN','SD'] if modality == 'CT' else [None]):
                get = lambda: get_dataloaders(task,method,root) if modality == 'CT' else get_mri_dataloaders(task,root)
                loaders,again = get(),get()
                require(list(loaders) == contract['splits'][modality]['keys'], 'Wrong split keys')
                require(pd.concat([v.dataset.df for v in loaders.values()]).groupby('scan_id').split.nunique().max() == 1, 'Public source leakage')
                summaries = {}
                for name,loader in loaders.items():
                    pd.testing.assert_frame_equal(loader.dataset.df,again[name].dataset.df)
                    if modality == 'CT':
                        expected_method = method if name != 'test_other' else ('SD' if method == 'CTGAN' else 'CTGAN')
                        flag = 'in_ctgan' if expected_method == 'CTGAN' else 'in_sd'
                        require((loader.dataset.df[flag] == 1).all(), 'Wrong CT fake-method membership')
                    torch.manual_seed(42)  # Audit harness seeding; no production defaults changed.
                    summaries[name] = audit_loader(loader,modality,contract['metadata']['item_field_types'][modality],
                                                   references[modality,task],name)
                interfaces.append(dict(modality=modality,task=task,train_method=method,splits=summaries))
                print(f'Public API {modality}/{task}/{method or "SD-only"}: all splits passed',flush=True)
    verify_snapshot(contract,root)
    require(file_digest(contract_path) == contract_sha, 'Contract changed during audit')
    report = dict(passed=True,contract_id=contract['contract_id'],contract_sha256=contract_sha,
        scope='Complete mechanical preprocessing audit; no model fitting, performance scores, or parameter selection',
        dataset_version=contract['dataset']['version'],rows_checked=len(rows),image_inventory=image_inventory,
        tasks=reports,hash_checks=hashes,public_loader_interfaces=interfaces,
        public_loader_rows=sum(split['rows'] for i in interfaces for split in i['splits'].values()),
        held_out_rows_checked=int((rows.split == 'test').sum()),
        mri_cross_task_group_and_exact_image_isolation=True,
        ct_cross_task_scan_overlap_injection_vs_removal=overlap,
        frozen_sources_and_manifests_unchanged=True,
        source_sha256=contract['provenance']['source_sha256'],manifest_sha256=contract['dataset']['manifest_sha256'],
        audit_script_sha256=file_digest(Path(__file__)),python=platform.python_version(),
        packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','torch','torchvision','pillow']},
        qualifications=['CT task-specific splits cannot be pooled without a new joint allocation.',
                        'Exact hashes do not rule out near-duplicates or verify patient identity.',
                        'ROI coordinates are supplied annotations, not independently verified clinical truth.',
                        'Preserve native MRI orientation does not establish canonical anatomy.',
                        'Training metadata precedes random augmentation and is flagged invalid for that tensor.',
                        'Teammate reproduction/review and publication remain pending.'])
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--contract',type=Path,default=REPO/'docs/preprocessing_contract_v1.json')
    parser.add_argument('--out',type=Path,default=Path('results/preprocessing_v1'))
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    # A failed rerun must not leave a stale success summary.
    (args.out/'summary.json').write_text(json.dumps({'passed':False,'status':'running'})+'\n')
    torch.set_num_threads(1)
    try:
        result = run(args.root,args.contract,args.out)
    except Exception as error:
        (args.out/'summary.json').write_text(json.dumps({'passed':False,'status':'failed','error':str(error)[:1600]},indent=2)+'\n')
        raise
    print(json.dumps({k:result[k] for k in ['passed','rows_checked','held_out_rows_checked','public_loader_rows']},indent=2))


if __name__ == '__main__':
    main()
