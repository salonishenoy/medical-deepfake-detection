"""Development-only orientation/background ablation; no held-out images read.

Three fixed paths: historical border-rule baseline; preserve native orientation;
preserve plus a training-corner 99th-percentile intensity floor. The last path is
an audit candidate only, never applied by the production loader.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from .audit_mri_shortcuts import image_features, probe_scores
from .mri_data_loader import MRIDataConfig, MRIDataset, build_mri_splits, load_mri_patch


def read_png(path):
    with Image.open(path) as im:
        if im.mode != 'L':
            raise ValueError('Expected mode-L PNG')
        return np.array(im)


def training_corner_threshold(root, frames):
    """No label conditioning; each unique training image contributes once."""
    paths = sorted(set(pd.concat(frames).query("split == 'train'").path))
    values = []
    for path in paths:
        image = read_png(root/path)
        values.append(np.concatenate([image[:30,:30].ravel(),image[:30,-30:].ravel(),
                                      image[-30:,:30].ravel(),image[-30:,-30:].ravel()]))
    return int(np.quantile(np.concatenate(values),.99,method='higher')), len(paths)


def floor_candidate(patch, threshold):
    """Audit-only: suppress intensities <= threshold/255; leave others unchanged."""
    return np.where(patch <= threshold/255, 0.0, patch).astype(np.float32)


def run(root, out):
    out.mkdir(parents=True,exist_ok=True)
    development = {}
    for task in ['injection','removal']:
        splits = build_mri_splits(task,root)
        development[task] = pd.concat([splits['train'],splits['val']],ignore_index=True)
    threshold, training_files = training_corner_threshold(root,list(development.values()))
    features, geometry, changes = [], [], []
    for task,frame in development.items():
        for row in frame.itertuples():
            raw = read_png(root/row.path)
            image = raw.astype(np.float32)/255
            mass = image.sum(axis=1,dtype=np.float64)
            centroid = float(np.dot(np.arange(len(mass)),mass)/mass.sum()) if mass.sum() else (len(mass)-1)/2
            native,meta = load_mri_patch(root/row.path,row.x,row.y)
            # Independent direct indexing check of the preserve-orientation default.
            top = min(max(int(row.y)-64,0),raw.shape[0]-128)
            left = min(max(int(row.x)-64,0),raw.shape[1]-128)
            np.testing.assert_array_equal(native,image[top:top+128,left:left+128])
            assert not meta['orientation_flipped'] and meta['oriented_y'] == row.y
            border,border_meta = load_mri_patch(root/row.path,row.x,row.y,orientation_policy='author_border')
            masked = floor_candidate(native,threshold)
            changed = native != masked
            core = changed[48:80,48:80]
            changes.append(dict(task=task,split=row.split,path=row.path,scan_id=row.scan_id,
                fraction_changed=float(changed.mean()),central_32_fraction_changed=float(core.mean()),
                roi_center_changed=bool(changed[int(row.y)-top,int(row.x)-left]),
                mean_absolute_change=float(np.abs(masked-native).mean())))
            geometry.append(dict(task=task,path=row.path,scan_id=row.scan_id,cur_slice=row.cur_slice,
                label=row.label,split=row.split,border_flip=border_meta['orientation_flipped'],
                centroid_flip=centroid < (raw.shape[0]-1)/2,centroid_margin=abs(centroid-(raw.shape[0]-1)/2),
                preserve_flip=False,x=row.x,y=row.y))
            for variant,patch,flipped in [('border',border,border_meta['orientation_flipped']),
                                          ('preserve',native,False),('preserve_floor',masked,False)]:
                resized = F.interpolate(torch.from_numpy(patch)[None,None],size=(224,224),
                                        mode='bilinear',align_corners=False,antialias=True)[0,0].numpy()
                features.append(dict(task=task,variant=variant,path=row.path,scan_id=row.scan_id,
                    scanner=row.scanner,cur_slice=row.cur_slice,split=row.split,label=row.label,
                    flipped=int(flipped),**image_features(resized)))
        # Exercise the default CNN interface only on development images.
        for train in (False,True):
            count = 0
            for x,y,meta in DataLoader(MRIDataset(frame,root,MRIDataConfig(),train=train),batch_size=16):
                assert x.shape[1:] == (3,224,224) and torch.isfinite(x).all()
                assert not meta['orientation_flipped'].any()
                assert y.dtype == torch.int64 and bool(meta['coordinates_valid'].all()) == (not train)
                count += len(y)
            assert count == len(frame)
    features,geometry,changes = pd.DataFrame(features),pd.DataFrame(geometry),pd.DataFrame(changes)
    task_reports, pairs_details = {}, []
    for task in development:
        g = geometry[geometry.task == task]
        pairs = g[g.label == 0].merge(g[g.label == 1],on=['scan_id','cur_slice'],suffixes=('_real','_fake'),validate='one_to_one')
        flagged = set(zip(pairs[pairs.border_flip_real != pairs.border_flip_fake].scan_id,
                          pairs[pairs.border_flip_real != pairs.border_flip_fake].cur_slice))
        comparisons = []
        for p in pairs.itertuples():
            real = read_png(root/p.path_real).astype(np.float32)/255
            fake = read_png(root/p.path_fake).astype(np.float32)/255
            same = float(np.abs(real-fake).mean()); opposite = float(np.abs(real-fake[::-1]).mean())
            comparisons.append(opposite > same)
            pairs_details.append(dict(task=task,scan_id=p.scan_id,cur_slice=p.cur_slice,
                                      raw_mae=same,opposite_orientation_mae=opposite))
        probes = {}
        for variant in ['border','preserve','preserve_floor']:
            subset = features[(features.task == task)&(features.variant == variant)].copy().reset_index(drop=True)
            subset['paired_flip_disagreement'] = [(r.scan_id,r.cur_slice) in flagged for r in subset.itertuples()]
            probes[variant],predictions = probe_scores(subset)
            predictions.to_csv(out/f'{task}_{variant}_validation_predictions.csv',index=False)
        ch = changes[changes.task == task]
        task_reports[task] = dict(development_rows=len(g),paired_source_slices=len(pairs),
            raw_orientation_closer_pairs=sum(comparisons),
            border_disagreements=int((pairs.border_flip_real != pairs.border_flip_fake).sum()),
            centroid_disagreements=int((pairs.centroid_flip_real != pairs.centroid_flip_fake).sum()),
            preserve_disagreements=0,changed_vs_border_rows=int(g.border_flip.sum()),
            probes=probes,
            floor_effect=ch.groupby('split')[['fraction_changed','central_32_fraction_changed','roi_center_changed','mean_absolute_change']].mean().to_dict('index'))
    features.to_csv(out/'features.csv',index=False)
    geometry.to_csv(out/'geometry.csv',index=False)
    changes.to_csv(out/'floor_changes.csv',index=False)
    pd.DataFrame(pairs_details).to_csv(out/'raw_pair_orientation.csv',index=False)
    report = dict(scope='Team A development-only MRI mitigation study',test_images_read=0,test_probe_rows=0,
        development_rows=len(geometry),unique_development_images=geometry.path.nunique(),source_groups=geometry.scan_id.nunique(),
        default_policy='preserve native PNG orientation; no canonical anatomy claim',
        centroid_candidate_adopted=False,background_candidate_adopted=False,
        background_candidate=dict(rule='Set intensities <= training-corner 99th percentile to zero before resizing',
                                  threshold_uint8=threshold,unique_training_files=training_files,corner_size=30),
        model_protocol='Same fixed depth-3 data-quality probes; fit train, score validation. Repeated validation inspection is exploratory.',
        tasks=task_reports,
        limitations=['Raw-pair agreement is an audit measure only; the loader never reads counterparts or labels to choose orientation.',
                     'Only four validation source groups; no test-set or new-data confirmation.',
                     'Preserving native orientation removes preprocessing-induced flips, not all acquisition orientation variation.',
                     'Intensity thresholding is not a tissue/background segmentation; central-pixel changes flag possible information loss.'],
        source_sha256={str(p.relative_to(Path(__file__).parent.parent)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('mri_data_loader.py'),Path(__file__).with_name('audit_mri_shortcuts.py')]})
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({task:{variant:round(probes['background']['validation_auc'],4)
                           for variant,probes in result['probes'].items()} for task,result in task_reports.items()},indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=Path('results/mri_mitigation'))
    args = parser.parse_args()
    run(args.root,args.out)
