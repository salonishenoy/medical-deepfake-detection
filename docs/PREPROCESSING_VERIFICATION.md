# Preprocessing verification: correctness criteria

Current default behavior is frozen in [contract v1](PREPROCESSING_CONTRACT.md).
This review supplies development correctness evidence and its limits. The
subsequent [complete-data audit](FULL_PREPROCESSING_AUDIT.md) passed all 5,791
regions under the same frozen baseline.

Status: mitigation experiments paused. This review checks whether the pipeline
implements its stated operations. No AUC target, classifier, foreground mask, or
new intensity treatment is part of the review. A subsequent metadata-only
correction now aligns resized coordinates with interpolation pixel centers;
image tensors, labels, crops, intensity handling, and splits are unchanged.

Evidence: [preprocessing_verification.json](preprocessing_verification.json).

## What was checked, in simple terms

| Question | Result | Meaning |
|---|---|---|
| Do we read the right pixels and labels? | Pass on 4,689 development samples | 4,101 CT and 588 MRI training/validation rows were read and checked against the released files/CSVs. |
| Is intensity handling what we intended? | Pass under the released-data contract | CT follows the reference implementation's [-700,2000] clipping/scaling; MRI is uint8 divided by 255. No background suppression is active. |
| Does a native crop contain the stated region? | Pass | Direct source-array indexing matches the loader; the selected source pixel is at the reported native crop coordinate. MRI is exact; CT differs from reference arithmetic by at most 1.20e-7. |
| Do resizing, channels, and normalization behave as specified? | Pass | Outputs have finite float32 3x224x224 values. Undoing ImageNet normalization recovers the intended resized grayscale channels within 5.97e-8. |
| Are coordinates aligned after every operation? | Pass with training restriction | Native crops and reflection are correct. Resized x/y now match interpolation pixel centers. Training x/y precede random augmentation and are flagged invalid for the augmented tensor. |
| Are source groups separated? | Pass within the stated experiment boundaries | CT groups stay separate within each task. MRI uses a joint source allocation across tasks. CT injection/removal cannot be pooled using their independent splits. |
| Can we repeat the result? | Pass for splits/evaluation; training needs caller seeding | Rebuilding splits gives identical frames, repeated evaluation samples are identical, and explicitly reseeding torch reproduces augmentation. The config seed alone controls only splits. |
| Are methods and departures documented? | Pass with limits | Source methods and departures are documented; clinical orientation/tissue truth is unverified. |

## Scope of the evidence

- CT injection: 3,174 development rows; CT removal: 927.
- MRI injection/removal: 294 development rows each.
- Rebuilt split membership includes held-out metadata, but no held-out image
  files were opened and no test/model scores were computed.
- Exact native-patch and CNN-input hashes show no conflicting labels or crossing
  between training and validation within either task. MRI source groups and patch
  hashes are also isolated across tasks. Raw CT removal arrays can have opposite
  labels because their ROI coordinates differ; that is why coordinate crops matter.
- The earlier full-data audits remain historical evidence for their recorded
  configurations. This review verifies the current defaults on development data.

Agreement with the release is not independent clinical validation. We have not
verified patient identities against original DICOM records, the medical accuracy
of lesion annotations, or whether every training augmentation is anatomically
appropriate. Source-folder isolation is the guarantee we can currently support.

## Coordinate convention: resolved

Previous metadata used simple scaling:

```text
reported x = native x * output_size / patch_size
```

Current metadata treats integer input indices as zero-indexed pixel centers,
matching bilinear resize with `align_corners=False`:

```text
pixel-center x = (native x + 0.5) * output_size / patch_size - 0.5
```

The same applies to y. An independent linear-ramp check confirms the interpolation
convention in the installed runtime. At the default output size 224, the offset
from the previous metadata was 0.375 pixels for MRI (128px patch) and 0.667 pixels
for CT (96px patch). The correction removes that offset. At native size the two formulas agree.

The correction changes metadata only. A repeat audit of all 4,689 development rows
confirms identical raw/native/CNN pixel hashes, labels, and split records before
and after. New linear-plane tests sample the actual resized image at its reported
coordinate; marked-pixel tests compare metadata with the resized marker centroid.
Boundary tests verify source-coordinate round trips when shrinking images, and
reflection tests cover MRI's explicit reference option. All 54 tests pass.

Both modalities expose `coordinate_convention="pixel_center"` and
`coordinates_valid`. When downsampling a boundary pixel, its continuous coordinate
can be slightly below zero or beyond the last output pixel center. It remains
inside the outer image edge, from -0.5 to `output_size - 0.5`. Metadata is not
clamped: clamping would change the represented source point.

Training spatial augmentation adds another constraint: returned x/y describe the
pre-augmentation crop. Both modalities expose `coordinates_valid=False` during training.
Downstream localization should use unaugmented samples
unless coordinates are transformed with the image.

## Reproducibility detail

`cfg.seed` determines source-group allocation; it does not automatically seed
random training augmentations or DataLoader shuffle. The training entry point
must set its RNG seeds. Resetting `torch.manual_seed` reproduced the same sampled
augmentation in this audit. That is a Team B integration requirement, not a claim
that complete training runs are reproducible already.

The JSON evidence records source hashes and installed package versions. Exact
bitwise agreement across different package versions, platforms, or devices has
not been established. A locked environment and teammate reproduction remain
release tasks.

## Basis for the preprocessing choices

- **CT window and ROI sizes:** adopted from the reference implementation for a traceable starting
  point; crop/scale correctness is verified against the released arrays.
- **MRI native orientation:** preserves supplied pixels and avoids an uncertain
  automatic flip. It is a conservative policy, not proven canonical anatomy.
- **Reference reflection option:** explicitly uses `height - 1 - y`, with synthetic
  pixel correspondence/edge tests. Its border-based choice to flip is still a
  heuristic and is not the default.
- **CNN adapter:** resizing, channel copying, ImageNet normalization, and training
  augmentation are project additions. They must be described separately from the
  reference implementation's native diffusion-model input.
- **Rejected intensity floor:** it altered many central image pixels and visibly
  removed dim structures. The relevant concern was unvalidated information loss;
  its effect on a nuisance AUC does not determine preprocessing correctness.

## Corrected interpretation of the background findings

Background-only ranking above chance establishes that background appearance carries
label information for these probes and this split. It does not, by itself, establish
a loading bug, an invalid detection signal, or a requirement to erase it. Generator
artifacts may be relevant to deepfake detection. The experiment's intended claim
determines whether reliance on them limits generalization.

The previous mitigation reports are exploratory history. There is no current
requirement or approved plan to mask tissue or reduce background AUC toward 0.5.
Document the observation and evaluate model reliance/generalization later under
the project's agreed research question. Further pixel modification is paused.

## Before declaring the handoff complete

1. State split/evaluation reproducibility and training seeding responsibilities clearly.
2. Complete teammate reproduction of the documented environment.
3. Keep orientation, patient identity, and clinical annotation limits explicit.

These are correctness and provenance requirements. AUC is not an acceptance gate.
