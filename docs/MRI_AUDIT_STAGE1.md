# MRI stage 1: inventory and reference preprocessing

Date: 2026-10-24. Team A only. This is a completed source-data audit, not a
production MRI loader release. CT implementation is unchanged.

> Subsequent decision: use standard reflection `y' = height - 1 - y` rather
> than the notebook's `512 - y`. The implementation and corrected-reference
> validation are recorded in [MRI_DATA_HANDOFF.md](MRI_DATA_HANDOFF.md).
> The measurements below describe the original reference, before that correction.

## Sources and scope

- Dataset: [BTD MRI and CT Deepfake Test Sets](https://www.kaggle.com/datasets/freddiegraboski/btd-mri-and-ct-deepfake-test-sets), version 7.
- Reference: [BTD_evaluation.ipynb](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb), commit `0fa74daef6802d305dcca4e5e37b4393fdf89c5b`.
- Aggregate evidence: [mri_audit_stage1_summary.json](mri_audit_stage1_summary.json).

MRI files were extracted from the already downloaded public dataset archive.
The reference dataset class was executed directly from the pinned notebook, with
its `ToTensor()` transform. An audit calculation using uint8 scaling, the reference implementation's
orientation rule, and the existing parameterized crop helper was compared against
every reference output. No model was trained. Detailed row exports and image
overlays remain local, outside the repository.

## Results

| Check | Result |
|---|---|
| Unique images | 525 PNG files, all grayscale `L`, uint8, 512×512 |
| Injection | 350 rows: 175 TM / 175 FM |
| Removal | 350 rows: 175 TB / 175 FB |
| Source groups | 25 `Breast_MRI_###` folders, four scanner models |
| Missing / unreferenced files | 0 / 0 |
| Invalid labels, coordinates, or unreadable images | 0 |
| Exact duplicate CSV rows | 0 in either task |
| Reference patches | All 700 match exactly; maximum absolute error 0 |
| Exact conflicting-label hashes | 0 within either task, for full arrays and patches |
| Edge-shifted crops | 0; all released coordinates fit 128×128 crops |
| Visual spot check | Eight examples: both classes, both tasks, flipped/unflipped |

There are seven slices per class per source folder in each task. The 700 rows are
not independent patients. Folder IDs are usable grouping keys; patient identity
has not been independently verified against the source MRI collection.

## Preprocessing contract to implement next

1. Read the grayscale PNG and convert uint8 values to float32 divided by 255.
   Do not apply the CT HU window or per-image min/max normalization.
2. Apply the reference implementation's heuristic: if the bottom 30 rows have less total intensity
   than the top 30 rows, flip vertically and replace `y` with `512 - y`.
3. Crop a 128×128 region centered on the CSV coordinate after orientation.
   Use the reference implementation's inward shift at boundaries; synthetic boundary tests are
   needed because this release does not exercise that branch.
4. Preserve every CSV row and its coordinates, including authentic samples.
5. Add the project's CNN resizing/channel adapter separately, documenting it as
   a project adaptation. Carry source coordinates, orientation decisions, and
   crop coordinates through metadata for downstream consumers.

The native preprocessing follows the BTD reference. This audit establishes
agreement with their code, not originality or clinical annotation correctness.

## Findings that must not be hidden

### One-pixel reference convention

For a zero-indexed 512-pixel image, an exact vertical reflection maps `y` to
`511 - y`. The notebook uses `512 - y`. Its rule flips 525 sample rows involving
392 unique files, and the one-pixel difference changes all 525 affected patches.
The requested reference-compatible pipeline should preserve and disclose this
convention. A corrected geometric variant would be a separate experiment.

### Orientation heuristic instability

Among 175 authentic/fake source-slice pairs in each task, flip decisions disagree
for one injection pair and six removal pairs. All seven raw pairs have lower
whole-image mean absolute difference before imposing opposite orientations.
This suggests the border-intensity rule can change its decision after manipulation;
it does not establish the true anatomical orientation. Preserve the reference
behavior and expose the decision in metadata so later comparisons are interpretable.

### Shared authentic images have different ROIs

The same 175 authentic PNGs appear in both tasks, with different coordinates in
every case. Only 91 of those pairs have x coordinates summing to 511, so do not
derive one task's coordinate by assuming a universal mirror operation. Use each
CSV's coordinates directly. Keep these shared source folders in the same split.

### MRI is SD-only

This release has no MRI CTGAN subset. An MRI API should expose train/validation/test;
it must not invent a GAN-to-diffusion `test_other` set.

## Provisional shared split

Applying the existing scan/scanner grouping function to the union of both MRI
manifests, with seed 42, gives the same allocation for both tasks:

| Split | Source folders | Authentic rows per task | Fake rows per task |
|---|---:|---:|---:|
| Train | 17 | 119 | 119 |
| Validation | 4 | 28 | 28 |
| Test | 4 | 28 | 28 |

No source group or exact patch crosses these provisional splits, including across
tasks. Only four validation and four test groups make performance estimates
sensitive to split choice. This allocation is audit evidence; a production MRI
split API and its regression tests remain to be implemented.

## Next increment

Implement the MRI Dataset and shared split API against this contract; add tests
for orientation, reference coordinate convention, grayscale scaling, crop edges,
shared-group isolation, and CNN batching. Re-run the complete MRI comparison and
existing CT tests. Then audit MRI image-statistics shortcuts before declaring the
MRI handoff ready. Production loader, batching, shortcut assessment, and
publication remained pending at this stage.
