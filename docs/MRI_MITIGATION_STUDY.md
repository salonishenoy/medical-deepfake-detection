# MRI stage 4: orientation policy and background mitigation

> Exploratory history. Further mitigation is paused. The
> [correctness review](PREPROCESSING_VERIFICATION.md) supersedes the framing of
> background suppression as a required fix. AUC alone does not establish that
> the preprocessing is wrong or that a background signal must be removed.

**Decision:** preserve native PNG orientation by default. Keep the intensity
values unchanged. The background shortcut is unresolved; the tested suppression
candidate was rejected because it removed substantial ROI information.

Evidence: [mri_mitigation_summary.json](mri_mitigation_summary.json).
Reproduce: `python -m src.audit_mri_mitigation --root "$BTD_DATA_ROOT" --out results/mri_mitigation`.
This is Team A data engineering, not detector development.

## Scope and safeguards

Used only existing training/validation groups: 588 CSV rows, 441 unique images,
21 source folders. No held-out test images were opened, and no test scores were
computed. Split membership is unchanged. The same fixed depth-3 diagnostic trees
were fitted on training and scored on validation for three predeclared paths:
the historical border rule; preserve native orientation; preserve plus a training-
derived intensity floor. Repeated validation inspection is exploratory and is
not independent confirmation. There are still only four validation source groups.

## Orientation: preserve native images

The full-image intensity-centroid candidate was tested first. It gave eight paired
disagreements (two injection, six removal), versus seven for the border heuristic.
It was rejected, and is not a production option.

All 294 development authentic/fake pairs are closer in their existing orientation
than when one image is vertically reflected. This supports preserving native
orientation instead of trying to infer a canonical orientation from intensity.
The default loader performs no deterministic orientation flip. It does not inspect
labels, filenames, source identity, or an authentic counterpart to make this choice.

This removes the seven preprocessing-induced paired flip disagreements, without
claiming a medically canonical orientation or uniform acquisition orientation.
Random training augmentation remains separate and applies equally to both classes.

An explicit `MRIDataConfig(orientation_policy="author_border")` option reproduces
the earlier border decision with **standard `height - 1 - y` reflection**. The
notebook's one-pixel coordinate convention is never restored.

## Background: the simple candidate did not solve it

The candidate threshold was the 99th percentile of the four 30x30 corners of each
unique training image (357 images), without conditioning on its label. This gave
29 on the uint8 0–255 scale. Native patch values <=29 were set to zero before
resizing; values above 29 were unchanged. This is not an anatomical background mask.

| CNN-sized background probe | Injection validation AUC | Removal validation AUC |
|---|---:|---:|
| Historical border-rule baseline | 0.500 | 0.833 |
| Preserve native orientation | 0.473 | **0.929** |
| Preserve + intensity floor (rejected) | **0.677** | 0.732 |

The orientation change did not remove the removal background cue. The numbers
are for refitted simple probes on the same small split, not a causal estimate of
orientation effects or expected neural-model performance. Reversing an even-sized
coordinate-centered crop can change a boundary row; resizing and refitted tree
thresholds can amplify small feature differences.

The floor lowered the removal score but increased the injection background score.
It changed **49.9% of pixels in the central 32x32 region** of removal validation
crops on average, and changed the exact ROI-center pixel in **51.8% of those crops**.
For injection validation, the central-region change was **23.0%**. Whole-patch
changes averaged 64.7% for removal and 57.4% for injection validation.

Training-only visual examples, selected nearest the median central change within
each task/class, show widespread removal of dim image structure. These observations
do not prove which pixels are clinically relevant; they are sufficient to reject
this candidate as a background-only treatment. It remains in the audit script only,
not in the production loader.

## Engineering checks

- All 588 development patches equal independently indexed native image crops.
- All 588 development rows exercise the default CNN batches in both train/eval modes.
- Both native orientation and the explicit reference option have regression tests;
  standard reflection remains tested at image edges and non-512 image heights.
- All **41 tests pass**, including the original CT suite, split isolation, probe
  isolation, and a check that threshold calibration reads unique training files only.
- Previous full-700 results belong to the historical border-rule implementation.
  The new default has not been run on the held-out test images in this increment.

## What remains before claiming the cue is fixed

There is no validated background fix yet. Do not tune more intensity cutoffs until
a low validation AUC appears, or erase the effect with a paired authentic image
that would be unavailable for a new input.

A next Team A experiment can assess a label-independent foreground mask that
distinguishes external air from dark tissue, with explicit checks for ROI coverage
and tissue preservation. It must be evaluated for both tasks and for new artifacts
introduced by mask boundaries. A thresholded intensity floor is not such a mask.
If a defensible mask cannot be verified, retain the unaltered data and disclose
the cue; cleaner or independently sourced evaluation data would be needed to
establish robustness. Team B/C should receive these limitations alongside the
data contract, rather than interpret high removal performance as tissue-specific
forensic detection.

Study results are retained for review alongside the current baseline.
