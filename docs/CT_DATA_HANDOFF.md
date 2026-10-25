# Team A CT data handoff

> Authoritative current behavior: [preprocessing contract v1](PREPROCESSING_CONTRACT.md).
> This handoff preserves the original full CT audit and subsequent metadata correction.
> The [complete current-baseline audit](FULL_PREPROCESSING_AUDIT.md) now passes
> all 5,091 CT rows with the frozen pixel-center metadata; 57 regression tests pass.

> Metadata correction after the original full audit: resized x/y now follow
> interpolation pixel centers. All 4,101 CT development rows were rechecked;
> raw/native/CNN pixel hashes, labels, and splits are unchanged. The complete
> suite now has 54 passing tests. Original full-audit results below remain
> historical evidence for their recorded implementation.

Audit date: 2026-10-24. Scope: CT data engineering only. No detector training,
hyperparameter tuning, Grad-CAM implementation, or MRI image audit was performed.

## Status

The corrected CT pipeline passes the full loading, preprocessing, coordinate,
batch-interface, and within-task split audit. It is ready for teammate code review
and downstream integration, with the dataset limitations below. A passing audit
does **not** certify an unbiased benchmark or model quality.

- Dataset: [BTD MRI and CT Deepfake Test Sets](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets), version **7**.
- Reference: [reference implementation's evaluation notebook](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb).
- Aggregate results: [ct_audit_summary.json](ct_audit_summary.json).
- Runtime and source fingerprints: [ct_audit_provenance.json](ct_audit_provenance.json).

## Verified coverage

| Check | Result |
|---|---|
| CT file inventory | 4,760 `.npy` files; every file referenced; no missing files |
| CSV-defined samples | 5,091 rows: 3,945 injection + 1,146 removal |
| Native patch reference comparison | All 5,091 match; maximum absolute difference 1.20e-7 |
| Independent comparison to original notebook class | All 5,091 match within float32 tolerance |
| Tensor shape, dtype, finiteness, normalization | Pass for every sample |
| Training/evaluation batching | All 5,091 rows exercised in each mode |
| Native and CNN-input hashes | Zero exact conflicting-label groups; zero cross-split groups within either task |
| Source-scan separation | No train/validation/test overlap within each task, for both training methods |
| Coordinates | All within source bounds; all crop/resize mappings verified; 42 edge-shifted rows covered |
| Visual geometry check | Interior and edge overlays inspected for both classes in both tasks |
| Automated tests | 20 passing |

Coordinate validation establishes agreement with CSV annotations and crop geometry;
it does not independently establish the medical correctness or extent of every annotation.
The hash checks detect exact duplicates, not all possible near-duplicates or additional
series belonging to the same patient. Splitting uses the released source-scan identifier;
the release does not supply a separately verified patient mapping here.

## Data contract

1. Load CT `.npy` slices and clip to **[-700, 2000]**, then scale to [0, 1].
2. Extract a native **96x96** patch using the CSV's **x = column, y = row**, for
   authentic and manipulated samples. Shift edge crops inward, without padding.
3. Preserve every CSV row, as the reference implementation's loader does. Multiple coordinates on
   one file are separate samples. The 150 exact duplicate injection rows are also
   preserved for reference fidelity; they are repeated weighting, not extra independent data.
4. CNN adaptation: resize each patch to 224x224 and copy to three channels.
5. Training only: horizontal/vertical flips and rotations up to 10 degrees.
6. Default output applies ImageNet mean/std normalization. Consequently, final
   tensors are **not** constrained to [0, 1], and channels differ after normalization.

The reference implementation's detector uses one-channel native patches. Resizing, channel copying,
ImageNet normalization, and training augmentation are the project's CNN adapter.
Use `DataConfig(image_size=96, imagenet_norm=False)` and `train=False` to inspect
native patch values; all three channels then contain the reference patch.

Each batch is `(images, labels, meta)`:

- `images`: float32 `[B, 3, 224, 224]` by default.
- `labels`: int64 `[B]`, **0 = authentic** (TM/TB), **1 = manipulated** (FM/FB).
- `meta`: path, scan_id, scanner, cur_slice, x, y, source_x, source_y, crop_left, crop_top.
- `x/y`: CSV ROI location within the resized patch for **both** classes. They are
  valid only without spatial augmentation (validation/test). Training coordinates
  currently describe the pre-augmentation patch and must not be used as transformed targets.
- `source_x/y`: original CSV coordinates. `crop_left/top`: crop origin in the original slice.
- Coordinate convention: `x_patch = (source_x - crop_left + 0.5) * image_size / 96 - 0.5`,
  analogously for y. Integer input coordinates are zero-indexed pixel centers.
  Metadata includes `coordinate_convention="pixel_center"` and
  `coordinates_valid` (false during training). These are points, not masks.
  Downsampled boundary points can lie between -0.5 and 0, or between the last
  center and the outer edge. Do not clamp them; doing so breaks coordinate inversion.
- Authentic removal coordinates select healthy tissue. Use fake samples for
  later tamper-region comparisons; an ROI center alone does not specify a segmentation mask.

## Split contract and counts

Seed 42, approximate 70/15/15 source-scan allocation, stratified by scanner. Keep
injection and removal experiments separate. Training and validation contain only
the selected fake-generation method plus authentic samples.

| Task | Split | Scans | Authentic ROI rows | Fake ROI rows |
|---|---|---:|---:|---:|
| Injection | train | 180 | 903 | 828 |
| Injection | val | 38 | 153 | 231 |
| Injection | test_same | 38 | 141 | 315 |
| Injection | test_other | 38 | 141 | 315 |
| Removal | train | 143 | 255 | 255 |
| Removal | val | 32 | 54 | 54 |
| Removal | test_same | 32 | 73 | 73 |
| Removal | test_other | 32 | 73 | 73 |

For `train_method="CTGAN"`, `test_same` uses CTGAN fakes and `test_other` uses SD fakes.
`train_method="SD"` reverses them. The authentic test images are deliberately shared
between the two test sets. They are not two independent authentic test populations.
Counts are CSV ROI samples, not unique slices or patients.

## Dataset limitations to carry into downstream work

### Injection scanner and slice bias remains

Depth-3 decision trees, fit only on training data and evaluated on validation, give
AUC **0.765** from slice number and **0.722** from scanner. These are data-quality
diagnostics, not detector results. Cropping does not repair the underlying real/fake
source-scan mismatch. Scanner and slice number are metadata, not classifier inputs.

### SD removal has a strong image-statistics cue

The same diagnostic procedure using patch mean, standard deviation, adjacent-pixel
difference magnitude, and fractions clipped to 0/1 gives these validation AUCs:

| Task / fake method | Image statistics | Clipping fractions only |
|---|---:|---:|
| Injection / CTGAN | 0.589 | 0.517 |
| Injection / SD | 0.754 | 0.646 |
| Removal / CTGAN | 0.591 | 0.552 |
| Removal / SD | **0.999** | **0.969** |

For removal, the median zero-pixel fraction of native patches is approximately
52.9% for TB, 55.8% for FB_CTGAN, and 1.46% for FB_SD. This difference survives
the reference implementation's preprocessing. It is an observed distribution cue; its exact generation
cause has not been established. A high SD-removal score alone is not evidence that
a model learned lesion-local forensic cues or will generalize to other datasets.
These estimates use one validation split (only 54 examples per class for removal),
and repeated source scans/rows mean examples are not statistically independent.
No thresholds were selected using test predictions, and no additional transformation
was added to hide these cues.

### Do not concatenate the task-specific splits

There are **22** source scans in injection training that occur in removal test,
and **20** in injection test that occur in removal training. Each task's independent
split is valid; mixing them would leak source scans. A combined model or cross-task
evaluation requires a new joint source-scan allocation from Team A first.

### Full-slice labels are unsafe for removal

375 full-array hash groups contain TB and FB labels. Their coordinate-centered
patches have no exact conflicting labels. Whole-slice inputs would reintroduce the
previous contradiction. Do not substitute full images for the specified patches.

## Reproduce and consume

Keep the dataset outside Git and set `BTD_DATA_ROOT` to the folder containing `CT/`.
The audit writes detailed local CSV manifests, error lists, and an aggregate JSON
summary under its output directory. Detailed row exports and images remain local.

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.audit_data --root "$BTD_DATA_ROOT" --out results/ct_audit
python -m src.check_data --root "$BTD_DATA_ROOT" --task both
```

```python
from src.data_loader import get_dataloaders

loaders = get_dataloaders("removal", train_method="CTGAN", batch_size=32)
images, labels, meta = next(iter(loaders["train"]))
validation = loaders["val"]
same_method_test = loaders["test_same"]
generalization_test = loaders["test_other"]
```

The audit exits nonzero for row-level failures or exact patch conflicts/leakage.
Its `passed` value covers engineering checks, not the absence of dataset biases.
Use the table above and `validation_shortcut_auc` when interpreting the release.

## Remaining Team A release steps

- Teammate review and merge of the corrected preprocessing and audit changes.
- Confirm colleagues use the same dataset version and configuration when reproducing.
- MRI loading is implemented locally; see [MRI_DATA_HANDOFF.md](MRI_DATA_HANDOFF.md).
- Mechanical audit complete: [frozen CT/MRI baseline results](FULL_PREPROCESSING_AUDIT.md).

Detector development belongs to Team B; Grad-CAM and reporting belong to Team C.
