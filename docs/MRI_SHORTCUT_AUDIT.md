# MRI stage 3: shortcut cues and orientation disagreements

> Historical baseline. The subsequent [mitigation study](MRI_MITIGATION_STUDY.md)
> changes the default to preserve native orientation. The 0.833 result below
> belongs to the original border-rule path; current-default removal background
> AUC is 0.929 on the same validation split. The tested intensity floor was rejected.

Team A data-quality audit. The production loader remains unchanged: standard
`height - 1 - y` reflection, the original border-intensity flip decision, and
128x128 native crops. No rows were removed. No neural detector was trained.

Aggregate evidence: [mri_shortcut_summary.json](mri_shortcut_summary.json).
Reproduce with `python -m src.audit_mri_shortcuts --root "$BTD_DATA_ROOT" --out results/mri_shortcuts`.
Detailed features, predictions, and paired-source diagnostics stay local.

## What the seven disagreements mean

The orientation heuristic compares the top and bottom 30 image rows. It flips
when the top is brighter. In seven authentic/fake pairs from the same source
folder and slice, that brightness comparison changes sign:

- One injection pair and six removal pairs.
- Six unique source slices in three source folders; one slice occurs in both tasks.
- All seven authentic images flip, while their manipulated counterparts do not.
- Three removal pairs are in training; one injection and three removal pairs are
  in validation. None of the disagreements are in the held-out test split.

The borders are nearly tied. Mean top-minus-bottom intensity differences are
only +0.109 to +0.387 grayscale levels for the authentic images and -0.216 to
-0.002 for the manipulated images, on a 0–255 scale. For example, one pair changes
from +0.112 to -0.009. A small border change reverses a whole-image flip decision.

All seven raw pairs have lower whole-image absolute difference before imposing
opposite flips. Visual inspection of all seven supports consistent raw orientation
within each pair. This is evidence of an unstable orientation heuristic, not an
independent determination of correct anatomical orientation.

Standard reflection still maps the selected source point correctly through each
flip. The issue is the choice of whether to flip, which can produce inconsistent
patch orientations. Fixing `512-y` to `511-y` does not change that binary choice.
Paired authentic and fake ROIs can be in different parts of the same image;
the image panels are not expected to depict identical tissue.

## Probe protocol

- Existing seed-42 split: 17 training source folders (238 rows per task), four
  validation source folders (56 rows per task), four held-out test source folders.
- Fixed depth-3 decision trees, random state 0, no parameter/threshold tuning.
- Fit on training only; report validation AUC only. Test features and labels do
  not enter fitting or scoring. The pair inventory includes all source images.
- Probe metadata (scanner, slice number, flip decision) separately from image
  statistics: mean, standard deviation, adjacent-pixel gradients, exact-zero,
  near-zero (<=1/255), and saturated-pixel fractions.
- Measure both native 128px patches and resized 224px grayscale patches before
  ImageNet normalization. The latter are the actual CNN adapter's pixel values
  before its fixed channel-wise affine transform, without random augmentation.

## Validation results

| Probe | Injection native | Injection resized | Removal native | Removal resized |
|---|---:|---:|---:|---:|
| Scanner | 0.500 | 0.500 | 0.500 | 0.500 |
| Slice number | 0.500 | 0.500 | 0.500 | 0.500 |
| Flip decision | 0.500 | 0.500 | 0.554 | 0.554 |
| Mean brightness | 0.500 | 0.500 | 0.500 | 0.500 |
| Standard deviation + gradients | 0.640 | 0.637 | 0.594 | 0.647 |
| Background/saturation fractions | 0.491 | 0.500 | **0.897** | **0.833** |
| All image statistics | 0.571 | 0.500 | **0.962** | **0.825** |

AUC 0.5 is chance ranking. These are nuisance-feature diagnostics, not the
project's detector results. Above-chance texture signals are not automatically
spurious; a low score from these particular trees does not establish a clean
dataset or rule out other learnable cues. Resizing changes which cues these
simple probes exploit, so native scores should not be substituted for CNN scores.

Removal validation patches have a median exact-zero fraction of **3.67% authentic
versus 0% fake**, and a near-zero fraction of **8.84% versus 0.027%** after resizing.
That makes background appearance a plausible shortcut requiring disclosure.

## Are the seven orientation cases causing the signal?

Using the same already-fitted probes and excluding both members of flagged pairs
from validation scoring (a diagnostic calculation only):

- Removal flip-only AUC falls from 0.554 to 0.500.
- Removal background AUC remains **0.812**, versus 0.833 on all validation rows.
- Removal all-statistics AUC remains **0.805**, versus 0.825.
- Removal has 50 validation rows after this diagnostic exclusion; injection has 54.

Thus the paired disagreements explain the observed flip-only ranking signal,
but do not account for the broader removal background cue in these probes.
No samples were excluded from the actual loaders, and no probe was retrained
using validation performance to choose a preprocessing change.

## Uncertainty and next decision

Only four validation source folders are available. Resampling whole folders
1,000 times with fixed predictions gives a descriptive 95% interval of **0.470–1.000**
for resized removal background AUC. Its within-folder AUC spans **0.286–1.000**.
Slices within a source are correlated; these intervals are not a reliable
population-level guarantee. The point estimate merits investigation, not a claim
of robust performance. No held-out test score was computed.

Before changing orientation, define and validate a label-independent policy that
also works when an authentic counterpart is unavailable. Do not select flips from
real/fake labels or silently borrow the paired authentic image at inference time.
For now, preserve the decisions in metadata and document the seven cases. Keep
background-bias findings in the Team B/C handoff; fixing orientation alone will
not establish an unbiased MRI benchmark.

All 36 tests pass, including checks that the probes exclude test rows and reject
train/validation source overlap. Publication, teammate reproduction, and a decision
on the orientation policy remained pending at this stage.
