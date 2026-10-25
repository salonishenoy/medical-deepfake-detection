# Team A MRI data handoff

> Authoritative current behavior: [preprocessing contract v1](PREPROCESSING_CONTRACT.md).
> The [complete current-baseline audit](FULL_PREPROCESSING_AUDIT.md) passed all
> 700 MRI rows, every MRI split, and the frozen metadata contract. Historical
> reports below retain their original configuration and coverage.

> Current review: mitigation experiments are paused. See
> [preprocessing verification](PREPROCESSING_VERIFICATION.md) for correctness
> checks and qualifications. Background AUC is a dataset observation, not a
> preprocessing acceptance criterion or an established requirement to suppress it.
> Resized coordinates now use pixel-center geometry; the metadata correction
> passed the development audit and subsequent full audit; 57 tests now pass.

Status: implemented and mechanically verified; review/merge pending. This
increment covers loading, geometry, source splitting, and the CNN batch interface.
The [mitigation study](MRI_MITIGATION_STUDY.md) changes the default to preserve
native PNG orientation. Background label information is documented; its removal is
not a preprocessing acceptance requirement. The tested intensity floor is inactive.
No Team B/C implementation.

## Source and deliberate correction

Dataset: [BTD MRI and CT Deepfake Test Sets](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets), version 7.
Reference methodology: [BTD evaluation notebook](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb).

PNG scaling and native 128x128 crops follow the BTD reference. The default now preserves
native orientation, removing the unstable border-based flip decision. An explicit
`MRIDataConfig(orientation_policy="author_border")` option retains that heuristic
for reference comparisons. When that option flips an image, we use standard zero-index reflection:
**`oriented_y = height - 1 - source_y`** when flipped. On a 512-pixel image this
is `511 - y`, rather than their `512 - y`. This change was explicitly selected
after the [stage-1 audit](MRI_AUDIT_STAGE1.md). Neither this corrected reference
option nor the preserve-orientation default should be claimed as universally
identical to the reference implementation's original preprocessing.

## Verified results

Historical full-reference results: [mri_audit_summary.json](mri_audit_summary.json).
Current frozen-default full results: [full_preprocessing_audit.json](full_preprocessing_audit.json).
Earlier development study: [mri_mitigation_summary.json](mri_mitigation_summary.json).

| Check | Result |
|---|---|
| Inventory | 525 grayscale uint8 PNGs, all 512x512; no missing/unreferenced files |
| Samples | 700 CSV rows: 350 injection and 350 removal |
| Grouping | 25 source folders across four scanners |
| Historical border-rule comparison | All 700 patches matched the coordinate-corrected reference; 175 matched the unmodified reference |
| Current preserve-orientation default | All 700 patches match direct native-image crops exactly |
| Coordinates | All 700 rows pass source/native/resized pixel-center and metadata-schema checks |
| Tensor interface | float32, finite, 3x224x224; int64 labels; every MRI public split traversed |
| Historical border-rule hash audit | No conflicting labels or cross-split exact arrays/patches/CNN inputs in the full 700-row audit |
| Current source splits | Unchanged joint source-folder allocation; no source crosses train/validation/test, including across tasks |
| Automated tests | 57 passing, including CT, geometry, frozen-contract guard, and split/probe isolation tests |

The historical comparison executes the pinned original dataset class twice: unmodified,
and with only `y = 512-y` replaced by `y = image.shape[0] - 1 - y`. This isolates
the intended behavioral change. It does not establish medical annotation truth.

## Data contract

1. Read mode-L grayscale PNGs; reject other modes rather than silently converting.
2. Scale uint8 intensities to float32 by dividing by 255. No CT window and no
   per-image min/max normalization.
3. Default: preserve native PNG orientation and coordinates. The explicit
   `author_border` comparison option flips if the normalized bottom 30-row sum
   is less than the top 30-row sum, using standard reflection; ties do not flip.
4. Crop a native 128x128 ROI using each CSV row's coordinates for both classes.
   Shift edge crops inward without padding. Real data has no edge-shifted MRI
   samples; synthetic tests cover boundaries and non-512 image heights.
5. Default CNN adaptation: resize to 224x224, copy to three channels, apply
   train-only random horizontal/vertical flips and up-to-10-degree rotation,
   then optional ImageNet normalization. These are project adaptations.

Labels are 0 authentic (TM/TB), 1 manipulated (FM/FB). Each batch returns
`(images, labels, meta)` with images `[B,3,224,224]` and labels `[B]`.

Metadata includes `path`, `scan_id`, `scanner`, `cur_slice`, `task`, original
`source_x/y`, source dimensions, `orientation_policy`, `orientation_flipped`, `oriented_x/y`, crop
origin `crop_left/top`, and resized annotation `x/y`. Crop origins refer to the
oriented image. Resized coordinates use zero-indexed pixel centers, as in CT:
`(oriented_coordinate - crop_origin + 0.5) * image_size / patch_size - 0.5`.
`coordinate_convention="pixel_center"` identifies this contract. When downsampling,
boundary points may lie between -0.5 and 0 or beyond the final center within the
outer pixel edge. Keep those fractional positions; do not clamp them.
These are points, not segmentation masks.

`coordinates_valid` is false for training: metadata precedes random spatial
augmentation. Original/source geometry remains available, but do not treat
returned `x/y` as transformed training targets.

## Shared source splits

Both `MRI_injection.csv` and `MRI_removal.csv` are required, even when requesting
one task. Assign source folders from the union once, stratified by scanner with
seed 42, then select the requested task. The MRI API exposes `train`, `val`,
`test`; no MRI CTGAN samples exist to construct a `test_other` set.

| Split | Source folders | Authentic per task | Fake per task |
|---|---:|---:|---:|
| Train | 17 | 119 | 119 |
| Validation | 4 | 28 | 28 |
| Test | 4 | 28 | 28 |

Source-folder grouping is not an independently verified patient mapping. There
are seven slices per class per source group; do not treat them as independent
patients. The same 175 authentic files are used at different ROIs across tasks.

## Usage and reproduction

```python
from src.mri_data_loader import MRIDataConfig, get_mri_dataloaders

loaders = get_mri_dataloaders("removal", root="/path/to/archive", batch_size=16)
images, labels, meta = next(iter(loaders["val"]))
# Native unnormalized patches for inspection:
native = get_mri_dataloaders("removal", root="/path/to/archive",
    cfg=MRIDataConfig(image_size=128, imagenet_norm=False))
```

```bash
pip install -r requirements.txt
python -m pytest tests -q
python -m src.audit_preprocessing --root "$BTD_DATA_ROOT" --out results/preprocessing_v1
# Optional historical reference audit, not the preserve-orientation default:
python -m src.audit_mri --root "$BTD_DATA_ROOT" \
  --reference-notebook /path/to/pinned/BTD_evaluation.ipynb \
  --out results/mri_audit
```

Obtain the notebook from the pinned source above. The audit checks its SHA-256
before executing the reviewed dataset class; it does not execute other cells.
Detailed row hashes remain in the local output directory; only aggregate results
belong in the repository. The loader itself does not need the reference notebook.

## Remaining limitations and next increment

- The original orientation heuristic has seven paired disagreements. The default
  now avoids that heuristic entirely, preserving native orientation. This does
  not establish canonical anatomical orientation; the optional reference mode
  retains the original decision issue while using correct reflection geometry.
- Four validation/test source groups each give limited evidence of generalization.
- Exact hashes do not rule out near-duplicate or correlated samples.
- The [mitigation study](MRI_MITIGATION_STUDY.md) finds removal background-only
  validation AUC 0.929 under the preserve-orientation default, versus 0.833 for
  the historical border-rule baseline. Only four validation source groups are
  available. This is a dataset observation rather than a correctness failure.
  Further mitigation is paused; the attempted intensity floor was rejected.
- Teammate reproduction and review/merge remain.
