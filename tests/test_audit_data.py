"""The audit must distinguish expected shared arrays from bad sample labels."""
import pandas as pd

from src.audit_data import duplicate_summary


def test_hash_audit_flags_conflicting_labels_and_cross_split_duplicates():
    rows = pd.DataFrame([
        {"hash": "shared", "label": 0, "split": "train"},
        {"hash": "shared", "label": 1, "split": "train"},
        {"hash": "leaked", "label": 0, "split": "train"},
        {"hash": "leaked", "label": 0, "split": "test"},
        {"hash": "unique", "label": 1, "split": "val"},
    ])
    result, conflicts, leaks = duplicate_summary(rows, "hash")
    assert result == {"duplicate_groups": 2, "conflicting_label_groups": 1, "cross_split_groups": 1}
    assert set(conflicts["hash"]) == {"shared"}
    assert set(leaks["hash"]) == {"leaked"}


def test_same_label_repeated_csv_rows_are_not_conflicts_or_leakage():
    row = {"hash": "repeat", "label": 1, "split": "train"}
    result, conflicts, leaks = duplicate_summary(pd.DataFrame([row, row]), "hash")
    assert result["duplicate_groups"] == 1
    assert conflicts.empty and leaks.empty
