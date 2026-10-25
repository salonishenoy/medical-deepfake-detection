# Team A preprocessing contract v1

**ID: `team-a-preprocessing-v1`. Status: frozen and mechanically verified;
branch review pending.** See the [complete audit](FULL_PREPROCESSING_AUDIT.md).
This is the authoritative specification of current
default behavior. Historical audit reports describe their recorded snapshots.

The [machine-readable snapshot](preprocessing_contract_v1.json) records all
configuration defaults, metadata types, source hashes, and dataset manifest hashes.
Creating this contract did not change runtime preprocessing.

## Acceptance and inputs

Team A prepares files, tensors, labels, coordinates, and source-separated splits.
Acceptance depends on correct scaling/cropping, coordinate correspondence, label
integrity, separation of source groups, repeatable evaluation, and clear provenance.
AUC is not an acceptance criterion. Further mitigation is paused; no foreground
mask or intensity floor is active.

Use [BTD dataset](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets)
version **7**, with `BTD_DATA_ROOT` or an explicit `root` argument.

| Default property | CT | MRI |
|---|---|---|
| Input | `.npy`, squeezed to 2D and cast to float32 | uint8 grayscale PNG, mode `L` |
| Observed release dimensions | 512x512 | 512x512 |
| Scaling | Clip to [-700,2000], then `(value+700)/2700` | Cast to float32, divide by 255 |
| Native ROI | 96x96 | 128x128 |
| Deterministic orientation | Preserve array orientation | Preserve native PNG orientation |
| CNN output | 3x224x224 | 3x224x224 |
| ImageNet normalization | Enabled | Enabled |
| Split seed / validation / test fractions | 42 / 0.15 / 0.15 | 42 / 0.15 / 0.15 |
| Slice filtering | Off (`slice_range=None`) | No slice filter option |

Each CSV row is one sample, including authentic rows, repeated paths, and exact
duplicate rows. Do not merge by path. Repeated rows add sample weight, not independent
images. Use each row's own coordinates. CT requires its `CT/` tree; MRI requires
`MRI/` and both task manifests, even when requesting a single MRI task.

MRI's explicit comparison option `orientation_policy="author_border"` flips if
the bottom 30-row intensity sum is less than the top 30-row sum; ties do not flip.
It uses **`oriented_y = height - 1 - source_y`**. It is not the default and does
not restore the notebook's one-pixel reflection convention.

## Ordered image operations and output

1. Load and scale as above.
2. Apply MRI orientation only if the explicit comparison option was selected.
3. Crop at the CSV point: **x = column, y = row**, zero-indexed pixel centers.
   Shift boundary crops inward without padding. Keep the annotation point even
   if the crop center has to shift.
4. Resize: bilinear, `align_corners=False`, `antialias=True`.
5. Repeat grayscale into three channels.
6. Training items only: horizontal flip p=0.5; independently vertical flip p=0.5;
   then rotation uniformly from -10 to +10 degrees. Rotation uses nearest
   interpolation, fill 0, and no expansion. Metadata precedes these operations.
7. Optional ImageNet normalization: `(channel-mean)/std`, using mean
   `(0.485,0.456,0.406)` and std `(0.229,0.224,0.225)`.

These augmentation choices are project additions; anatomical plausibility has not
been clinically established. Evaluation items have no random augmentation.

Each batch is `(images, labels, meta)`. Images are float32 `[B,3,224,224]` by
default; labels are int64 `[B]`: **0 authentic** (TM/TB), **1 manipulated** (FM/FB).
Final normalized tensors need not lie in [0,1]. Channels are equal before
normalization and generally differ afterward.

Loader defaults: batch size 32, zero workers, no dropped last batch, shuffle
training only. Native inspection explicitly sets `image_size=patch_size`,
`imagenet_norm=False`, and uses an unaugmented item.

## Coordinate and metadata contract

Both modalities return these 12 per-item fields:

| Fields | Types | Meaning |
|---|---|---|
| `path`, `scan_id`, `scanner` | str | Relative source path and source/scanner identifiers |
| `cur_slice` | int | Released/filename-derived slice number |
| `source_x/y` | int | Original CSV point |
| `crop_left/top` | int | Crop origin in the oriented image |
| `x/y` | float | Resized point before random augmentation |
| `coordinate_convention` | str | `"pixel_center"` |
| `coordinates_valid` | bool | False whenever `train=True`; true for evaluation |

MRI additionally returns seven fields: `task` (str), `source_height/width` (int),
`oriented_x/y` (int), `orientation_policy` (str), and `orientation_flipped` (bool).
CT's oriented point equals its source point; MRI's does too under the default.
Default batching turns numeric/bool fields into tensors and strings into sequences.

For oriented coordinate `u`, crop origin `c`, native patch size `P`, and output `S`:

```text
output point = (u - c + 0.5) * S / P - 0.5
oriented point = (output point + 0.5) * P / S - 0.5 + c
```

For the explicit MRI reflection option, invert y with `height-1-y` to recover the
source point. Coordinates are continuous pixel centers. When downsampling, a
boundary point may be slightly negative within the outer image extent
`[-0.5,S-0.5]`. Do not clamp or round metadata. These are ROI points, not masks.

Training x/y remain pre-augmentation coordinates. Do not use them as transformed
targets. `coordinates_valid=False` throughout training, including random draws
that happen to leave the image unchanged. Original source geometry remains available.

## Split and interface contract

| Interface | Keys | Source allocation |
|---|---|---|
| `get_dataloaders` in `src.data_loader` | train, val, test_same, test_other | CT: per task, shared across CTGAN/SD |
| `get_mri_dataloaders` in `src.mri_data_loader` | train, val, test | MRI: union of both task manifests |

Both accept injection/removal tasks. CT accepts CTGAN/SD as `train_method`;
training/validation contain the selected method's fakes plus authentic rows.
The two CT tests use the same held-out source allocation, different fake methods,
and shared authentic test rows. They are not independent authentic populations.

MRI is SD-only; there is no MRI `test_other` population. Shared authentic files
and source groups stay in one split across both MRI tasks. CT injection/removal
allocations are independent: **keep those experiments separate**. Pooling requires
a new joint split from Team A.

Allocation is approximately 70/15/15 by source folder, stratified by scanner,
using sorted IDs and seed 42, with rounding per scanner group. It is not an exact
row fraction or an independently verified patient-level separation.

## Reproducibility, provenance, and release status

The config seed controls split membership. It does not automatically seed training
augmentation or shuffle; callers must seed their training RNGs and record worker/
device settings. Evaluation is deterministic for the same files, configuration,
and runtime. Full model-training reproducibility is outside this interface.
The [reproducibility handoff](REPRODUCIBILITY.md) now pins and verifies the
same-platform environment; the JSON records the baseline and verification history.

Native scaling/crop parameters come from the reference implementation's
[pinned notebook](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb).
MRI native-orientation preservation, corrected optional reflection, CNN adaptation,
output-coordinate metadata, and source splits are explicit project choices.

Evidence: [correctness review](PREPROCESSING_VERIFICATION.md), 4,689 development
rows verified, 54 tests passing, and six public loader interfaces checked against
the snapshot (two tasks x two CT methods, plus two MRI tasks). The subsequent
[complete-data audit](FULL_PREPROCESSING_AUDIT.md) passed all 5,791 regions and
8,574 public-loader sample deliveries; the regression suite now has 57 passing tests.
The JSON snapshot records the frozen behavior and publication-time fingerprints;
the complete audit binds its subsequent results to that snapshot's SHA-256.

Default pixel operations, labels, metadata semantics, or source allocation may
change only with a contract revision and renewed checks. Record configuration
overrides explicitly. Historical results retain their recorded configurations.
Data files, detailed row exports, and image artifacts remain outside Git.
