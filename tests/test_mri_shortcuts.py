"""Ensure data-quality probes preserve the train/validation/test boundary."""
import numpy as np
import pandas as pd
import pytest

from src.audit_mri_shortcuts import probe_scores


def probe_frame():
    rows = []
    for split in ('train','val','test'):
        for group in range(2):
            for label in (0,1):
                rows.append(dict(split=split,scan_id=f'{split}_{group}',label=label,
                    scanner='scanner',cur_slice=1,flipped=0,
                    paired_flip_disagreement=(split == 'val' and group == 0),
                    mean=float(label),std=float(label),gradient=float(label),
                    zero_fraction=float(label),near_zero_fraction=float(label),saturated_fraction=float(label)))
    return pd.DataFrame(rows)


def test_test_rows_never_enter_probe_fit_or_score():
    frame = probe_frame()
    # These values would fail fitting/scoring if the held-out rows were consumed.
    frame.loc[frame.split == 'test', ['mean','std','gradient','zero_fraction','near_zero_fraction','saturated_fraction']] = np.inf
    frame.loc[frame.split == 'test','label'] = 2
    scores, predictions = probe_scores(frame)
    assert set(predictions.row) == set(frame.index[frame.split == 'val'])
    assert scores['brightness']['validation_auc'] == 1.0
    assert scores['brightness']['source_bootstrap_95_interval'] == [1.0,1.0]
    assert scores['brightness']['validation_rows_excluding_flagged_pairs'] == 2
    assert scores['brightness']['validation_auc_excluding_flagged_pairs'] == 1.0
    assert scores['scanner']['validation_auc'] == .5


def test_probe_rejects_source_overlap():
    frame = probe_frame()
    frame.loc[frame.split == 'val','scan_id'] = 'train_0'
    with pytest.raises(ValueError, match='Source-group overlap'):
        probe_scores(frame)


def test_probe_rejects_missing_class():
    frame = probe_frame()
    frame.loc[frame.split == 'train','label'] = 0
    with pytest.raises(ValueError, match='Both classes required'):
        probe_scores(frame)
