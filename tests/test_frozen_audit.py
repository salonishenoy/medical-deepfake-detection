"""The full audit must fail closed on drift or unsafe duplicate inputs."""
from dataclasses import asdict
import hashlib

import pandas as pd
import pytest

from src.audit_preprocessing import hash_summary, verify_snapshot
from src.config import DataConfig
from src.mri_data_loader import MRIDataConfig


def test_snapshot_guard_rejects_source_and_manifest_drift(tmp_path):
    repo,root = tmp_path/'repo',tmp_path/'data'
    repo.mkdir();root.mkdir()
    source,manifest = repo/'loader.py',root/'manifest.csv'
    source.write_text('frozen source')
    manifest.write_text('frozen manifest')
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    contract=dict(contract_id='team-a-preprocessing-v1',
        defaults={'CT':asdict(DataConfig()),'MRI':asdict(MRIDataConfig())},
        provenance={'source_sha256':{'loader.py':digest(source)}},
        dataset={'manifest_sha256':{'manifest.csv':digest(manifest)}})
    verify_snapshot(contract,root,repo)
    source.write_text('changed source')
    with pytest.raises(ValueError,match='Frozen source changed'):
        verify_snapshot(contract,root,repo)
    source.write_text('frozen source')
    manifest.write_text('changed manifest')
    with pytest.raises(ValueError,match='Dataset manifest changed'):
        verify_snapshot(contract,root,repo)


def test_snapshot_guard_rejects_unrecorded_default_change(tmp_path):
    contract=dict(contract_id='team-a-preprocessing-v1',
                  defaults={'CT':asdict(DataConfig()),'MRI':asdict(MRIDataConfig())})
    contract['defaults']['MRI']['orientation_policy']='author_border'
    with pytest.raises(ValueError,match='Runtime defaults differ'):
        verify_snapshot(contract,tmp_path)


def test_hash_checks_distinguish_repeat_weight_from_conflicts_and_leakage():
    frame=pd.DataFrame([
        dict(hash='repeat',label=0,split='train'),dict(hash='repeat',label=0,split='train'),
        dict(hash='conflict',label=0,split='val'),dict(hash='conflict',label=1,split='val'),
        dict(hash='leaked',label=0,split='train'),dict(hash='leaked',label=0,split='test')])
    assert hash_summary(frame,'hash')==dict(duplicate_groups=3,conflicting_label_groups=1,cross_split_groups=1)
