"""MRI data-quality probes and paired orientation diagnostics; no detector training.

Fit fixed depth-3 trees on train, score validation only. Test images participate
only in the existing inventory/orientation audit, never in probe fitting/scoring.
"""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from .config import get_data_root
from .mri_data_loader import LABELS, MRIDataConfig, MRIDataset, build_mri_splits, load_mri_patch


def image_features(image):
    return dict(mean=float(image.mean()), std=float(image.std()),
                gradient=float((np.abs(np.diff(image,axis=0)).mean()+np.abs(np.diff(image,axis=1)).mean())/2),
                zero_fraction=float((image == 0).mean()),
                near_zero_fraction=float((image <= 1/255).mean()),
                saturated_fraction=float((image == 1).mean()))


def probe_scores(frame):
    """No test labels/features are used; predictions are fixed before bootstrapping."""
    train, val = frame[frame.split == 'train'], frame[frame.split == 'val']
    if train.label.nunique() != 2 or val.label.nunique() != 2:
        raise ValueError('Both classes required in training and validation')
    if set(train.scan_id) & set(val.scan_id):
        raise ValueError('Source-group overlap between training and validation')
    probes = dict(slice=['cur_slice'], orientation=['flipped'], brightness=['mean'],
                  texture=['std','gradient'], background=['zero_fraction','near_zero_fraction','saturated_fraction'],
                  all_image_statistics=['mean','std','gradient','zero_fraction','near_zero_fraction','saturated_fraction'])
    results, predictions = {}, []
    for name in [*probes, 'scanner']:
        if name == 'scanner':
            x_train = pd.get_dummies(train.scanner)
            x_val = pd.get_dummies(val.scanner).reindex(columns=x_train.columns,fill_value=False)
        else:
            x_train, x_val = train[probes[name]], val[probes[name]]
        classifier = DecisionTreeClassifier(max_depth=3,random_state=0).fit(x_train,train.label)
        pred = classifier.predict_proba(x_val)[:,1]
        auc = float(roc_auc_score(val.label,pred))
        per_group = [float(roc_auc_score(val.label[val.scan_id == group],pred[(val.scan_id == group).to_numpy()]))
                     for group in sorted(val.scan_id.unique())]
        # Resample whole validation source folders, preserving within-folder correlation.
        groups = sorted(val.scan_id.unique())
        indices = [np.flatnonzero((val.scan_id == group).to_numpy()) for group in groups]
        rng = np.random.default_rng(42)
        boot = []
        for _ in range(1000):
            idx = np.concatenate([indices[g] for g in rng.integers(0,len(groups),len(groups))])
            boot.append(roc_auc_score(val.label.iloc[idx],pred[idx]))
        keep = ~val.paired_flip_disagreement.to_numpy(dtype=bool)
        without_flagged = float(roc_auc_score(val.label.iloc[np.flatnonzero(keep)],pred[keep])) if len(np.unique(val.label[keep])) == 2 else None
        results[name] = dict(validation_auc=auc,
            source_bootstrap_95_interval=[float(x) for x in np.quantile(boot,[.025,.975])],
            within_source_auc_range=[min(per_group),max(per_group)],
            validation_auc_excluding_flagged_pairs=without_flagged,
            validation_rows_excluding_flagged_pairs=int(keep.sum()))
        predictions.extend(dict(row=int(i),probe=name,label=int(y),score=float(p))
                           for i,y,p in zip(val.index,val.label,pred))
    return results, pd.DataFrame(predictions)


def run(root, out):
    root = get_data_root(root,modality='MRI')
    out.mkdir(parents=True,exist_ok=True)
    # Reproduce stage 3's historical border-rule baseline explicitly.
    cfg = MRIDataConfig(orientation_policy='author_border')
    all_features, disagreements, summaries = [], [], {}
    for task in LABELS:
        splits = build_mri_splits(task,root,cfg)
        frame = pd.concat(splits.values(),ignore_index=True)
        geometry_rows = []
        for _, row in frame.iterrows():
            with Image.open(root/row.path) as im:
                raw = np.array(im).astype(np.float32)/255
            geometry_rows.append(dict(task=task,path=row.path,scan_id=row.scan_id,
                cur_slice=int(row.cur_slice),scanner=row.scanner,label=int(row.label),split=row.split,
                flipped=bool(raw[-30:].sum() < raw[:30].sum()),
                top_minus_bottom_mean_uint8=float((raw[:30].mean()-raw[-30:].mean())*255),
                x=int(row.x),y=int(row.y)))
        geometry = pd.DataFrame(geometry_rows)
        pairs = geometry[geometry.label == 0].merge(geometry[geometry.label == 1],
            on=['scan_id','cur_slice'],suffixes=('_real','_fake'),validate='one_to_one')
        assert len(pairs)*2 == len(frame)
        bad = pairs[pairs.flipped_real != pairs.flipped_fake]
        flagged = set(zip(bad.scan_id,bad.cur_slice))
        geometry['paired_flip_disagreement'] = [(r.scan_id,r.cur_slice) in flagged for r in geometry.itertuples()]
        for _, pair in bad.iterrows():
            with Image.open(root/pair.path_real) as im:
                real = np.array(im).astype(np.float32)/255
            with Image.open(root/pair.path_fake) as im:
                fake = np.array(im).astype(np.float32)/255
            disagreements.append(dict(task=task,scan_id=pair.scan_id,cur_slice=int(pair.cur_slice),
                split=pair.split_real, real_path=pair.path_real,fake_path=pair.path_fake,
                real_x=int(pair.x_real),real_y=int(pair.y_real),fake_x=int(pair.x_fake),fake_y=int(pair.y_fake),
                real_flipped=bool(pair.flipped_real),fake_flipped=bool(pair.flipped_fake),
                real_border_mean_difference=float(pair.top_minus_bottom_mean_uint8_real),
                fake_border_mean_difference=float(pair.top_minus_bottom_mean_uint8_fake),
                raw_mae=float(np.abs(real-fake).mean()),opposite_orientation_mae=float(np.abs(real-fake[::-1]).mean())))
        geometry.to_csv(out/f'{task}_orientation.csv',index=False)
        development = frame[frame.split.isin(['train','val'])].reset_index(drop=True)
        cnn = MRIDataset(development,root,replace(cfg,imagenet_norm=False))
        features = []
        for i,row in development.iterrows():
            native, meta = load_mri_patch(root/row.path,row.x,row.y,orientation_policy='author_border')
            resized = cnn[i][0][0].numpy()
            for stage, image in [('native',native),('cnn',resized)]:
                features.append(dict(task=task,stage=stage,path=row.path,scan_id=row.scan_id,
                    cur_slice=int(row.cur_slice),scanner=row.scanner,label=int(row.label),split=row.split,
                    flipped=int(meta['orientation_flipped']),
                    paired_flip_disagreement=(row.scan_id,row.cur_slice) in flagged,**image_features(image)))
        features = pd.DataFrame(features)
        all_features.append(features)
        results, distributions = {}, {}
        for stage in ['native','cnn']:
            subset = features[features.stage == stage].reset_index(drop=True)
            results[stage], predictions = probe_scores(subset)
            predictions.to_csv(out/f'{task}_{stage}_validation_predictions.csv',index=False)
            distributions[stage] = subset.groupby(['split','label'])[['mean','zero_fraction','near_zero_fraction']].median().reset_index().to_dict('records')
        summaries[task] = dict(train_rows=len(splits['train']),val_rows=len(splits['val']),
            train_source_groups=splits['train'].scan_id.nunique(),val_source_groups=splits['val'].scan_id.nunique(),
            paired_flip_disagreements=len(bad),disagreement_sources=bad.scan_id.nunique(),
            disagreements_by_split=bad.split_real.value_counts().to_dict(),
            border_sign_real_positive_fake_negative=bool(((bad.top_minus_bottom_mean_uint8_real > 0)&(bad.top_minus_bottom_mean_uint8_fake < 0)).all()),
            probes=results,median_features=distributions)
    pd.concat(all_features).to_csv(out/'development_features.csv',index=False)
    mismatch = pd.DataFrame(disagreements)
    mismatch.to_csv(out/'orientation_disagreements.csv',index=False)
    report = dict(scope='Team A MRI data-quality audit; no neural detector development',
        seed=42,probe='DecisionTreeClassifier(max_depth=3, random_state=0); fixed features; no tuning',
        evaluated_split='val only',test_probe_rows=0,
        uncertainty='1000 source-folder bootstrap resamples of fixed validation predictions, seed 42. Only four validation groups: descriptive, not a reliable population confidence guarantee.',
        interpretation='AUC above chance can indicate dataset cues; it does not prove which cues a CNN uses or that texture information is spurious.',
        orientation=dict(task_pair_disagreements=len(mismatch),unique_source_slices=len(mismatch[['scan_id','cur_slice']].drop_duplicates()),
            source_groups=mismatch.scan_id.nunique(),all_real_flip_fake_do_not=bool((mismatch.real_flipped&~mismatch.fake_flipped).all()),
            raw_pairs_closer_before_opposite_flips=int((mismatch.raw_mae < mismatch.opposite_orientation_mae).sum()),
            real_border_mean_difference_range=[float(mismatch.real_border_mean_difference.min()),float(mismatch.real_border_mean_difference.max())],
            fake_border_mean_difference_range=[float(mismatch.fake_border_mean_difference.min()),float(mismatch.fake_border_mean_difference.max())]),
        tasks=summaries,
        source_sha256={str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('mri_data_loader.py'),Path(__file__).with_name('config.py'),Path(__file__).with_name('splits.py')]},
        manifest_sha256={f'MRI_{task}.csv':hashlib.sha256((root/'MRI'/f'MRI_{task}.csv').read_bytes()).hexdigest() for task in LABELS},
        loader_changed=False,rows_removed=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,default=Path('results/mri_shortcuts'))
    args=parser.parse_args()
    report=run(args.root,args.out)
    print(json.dumps({task:{name:round(value['validation_auc'],4) for name,value in data['probes']['cnn'].items()}
                      for task,data in report['tasks'].items()},indent=2))
