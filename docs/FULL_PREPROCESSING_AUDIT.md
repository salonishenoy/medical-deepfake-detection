# Complete preprocessing audit: contract v1

The full result has also been reproduced in a fresh isolated environment with
34 pinned packages: [setup and replay evidence](REPRODUCIBILITY.md). All 5,791
sample records and the aggregate summaries match the original audit exactly.

**Result: passed.** The frozen `team-a-preprocessing-v1` implementation passed the
complete mechanical audit. All preprocessing source hashes, configuration defaults,
and dataset manifest hashes matched the [frozen contract](PREPROCESSING_CONTRACT.md)
before and after the run. Runtime preprocessing was not changed during the audit.

Aggregate evidence, contract hash, audit-script hash, and package versions:
[full_preprocessing_audit.json](full_preprocessing_audit.json).
The JSON contract's verification section records the earlier freeze-time checks;
this report supplies the subsequent complete-data evidence for that exact snapshot.

## Coverage

| Modality / task | CSV-defined regions | Unique files |
|---|---:|---:|
| CT injection | 3,945 | 3,635 |
| CT removal | 1,146 | 1,125 |
| MRI injection | 350 | 350 |
| MRI removal | 350 | 350 |
| Total regions | **5,791** | **5,285 unique files across the release** |

The MRI tasks reuse 175 authentic files, so task-level unique-file counts must
not be added to obtain the release inventory. Actual inventory is 4,760 CT arrays
and 525 MRI PNGs, with no missing or unreferenced image files.

All splits were checked, including 1,102 held-out CSV rows. Held-out images were
used only to verify the frozen implementation. No detector was fitted, no model
performance was scored, and no preprocessing parameter was selected from this run.

## Checks that passed

- **Files and labels:** every referenced file loads; released dimensions/dtypes
  are valid; source coordinates and class mappings are valid. CSV rows are retained.
- **Scaling and native crops:** independently indexed source regions agree with
  native loader output. MRI agrees exactly after division by 255; CT agrees with
  the reference arithmetic within float32 tolerance (maximum 1.20e-7).
- **CNN tensors:** finite float32 3x224x224; ImageNet normalization inverts back to
  the intended resized grayscale channels within 5.97e-8.
- **Metadata:** exact contract field names/types, pixel-center resize coordinates,
  source/crop mapping, and coordinate inversion pass for every region. MRI keeps
  native orientation under the frozen default.
- **Batches:** all six public loader combinations and every returned split were
  traversed: 8,574 delivered samples. The delivered multiset exactly matches each
  split manifest, including repeated-row weights. Evaluation tensors match the
  independently checked item hashes. Training batches have the correct augmentation
  mode and invalid-coordinate flag.
- **Integrity and isolation:** no exact native-patch or CNN-input label conflicts;
  no source or exact image/input crosses train/validation/test within either task.
  MRI's joint source allocation also passes across tasks.
- **Repeatability:** rebuilding split frames gives identical membership; repeated
  evaluation-item checks give identical tensors. Training RNG was seeded explicitly
  by the audit harness; production defaults were not changed.
- **Regression suite:** 57 tests pass, including frozen-source/default/manifest
  drift rejection and checks for duplicate conflicts and split leakage.

## Expected reuse and remaining qualifications

The CT injection CSV retains 150 exact duplicate rows as specified. Full CT removal
arrays can have opposite labels because different coordinates select different
regions; there are 375 such raw-array hash groups. Their native/CNN patches have
no exact conflicting labels. The public CT test interfaces intentionally share
authentic test samples across fake methods.

CT task allocations remain independent: 22 injection-train source groups occur
in removal test, and 20 injection-test groups occur in removal train. This matches
the contract's restriction: do not pool these task-specific splits. A combined
experiment requires a new joint allocation from Team A.

Passing this audit establishes implementation correctness under the recorded
contract. It does not verify patient identity, clinical annotation truth, canonical
anatomical orientation, absence of near-duplicates, or full training reproducibility.
Background label information remains a documented dataset observation; AUC was
not an acceptance criterion. Teammate reproduction/review and publication remain
separate release tasks.

## Reproduce

```bash
python -m src.audit_preprocessing \
  --root "$BTD_DATA_ROOT" \
  --contract docs/preprocessing_contract_v1.json \
  --out results/preprocessing_v1
python -m pytest tests -q
```

The command fails if a frozen preprocessing source/default/manifest drifts or any
mechanical check fails. A failed rerun writes `passed=false`, so a previous success
cannot remain as its apparent result. Detailed row hashes and coordinates are
written only to the local output directory; only aggregate evidence belongs in Git.

Changes are prepared for feature-branch review. No Team B/C implementation was performed.
