# Handoff: Medical Deepfake Detection (Team A, data engineering)

Written 2026-10-24 for another agent (or person) who will continue this work. It records
everything decided, found, built and left open up to that date. Nothing here is a summary
of a guess unless it is marked **ASSUMPTION** or **UNVERIFIED**.

---

## 0. TL;DR

- **Project:** a 2D deep-learning pipeline that tells authentic CT/MRI slices from AI-manipulated
  ones (tumours added or removed), then tests whether a detector trained on GAN fakes also
  catches diffusion fakes, then uses Grad-CAM to check whether the model looks at the tampered spot.
- **Course / deadline:** CSCI 566 (Deep Learning) group project. Poster defense **November 30**.
  Midterm report is due in the weeks 4-6 window of the timeline.
- **Our role:** Team A (data engineering, 2 people: Saloni and Rohan). We build the data
  ingestion, preprocessing and splits that Teams B (modeling) and C (interpretability/reporting) use.
- **State:** the CT data pipeline is written, unit-tested on synthetic data, and **has run
  successfully on the real data** on Saloni's machine (`python -m src.check_data --task both`).
  It lives on the branch `feature/data-loader` (commit `2f393c8` plus this file). Not yet
  merged to `main`; no PR was known to exist when this was written.
- **Biggest finding:** CT **removal** is a clean benchmark. CT **injection** has real shortcuts
  (slice number alone gives AUC 0.74, scanner alone 0.72; real and fake images come mostly from
  different patients). See section 4.
- **Waiting on Saloni:** (1) confirm the red circle in `sample_images/*.png` lands on the lesion in
  fake images; (2) say yes/no to the proposed next changes in section 9 (balanced metrics, class
  weights, slice-filter default, shared-scan test set).

---

## 1. People, roles, process

- 7-person team. Team A = data engineering (2: Saloni, Rohan). Team B = modeling and generalization
  (3). Team C = interpretability and reporting (2). Assignments are tracked in `PROJECT_TRACKER.md`
  (Team A line there is still the template "[Assign 2 Members]").
- Saloni is the user. She codes on **Windows**, in PowerShell, inside a `.venv`, using Cursor plus
  Claude. The repo sits inside a OneDrive folder (be careful: OneDrive syncing `.git` or generated
  images can cause trouble); the dataset is kept **outside** the repo.
- Absolute local paths are deliberately **not** written here (CONTRIBUTING says not to commit
  machine-specific paths). Ask Saloni for the dataset root and repo path, or read them from the
  `BTD_DATA_ROOT` environment variable on her machine.
- How she likes to work: **explain before executing**, keep explanations **simple** (she asked this
  at the start and again when she said "explain the output report in relatively simple words").
  She asks "what do you mean?" when a term is unclear, so define terms. She wants pros and cons
  when a choice is open. She is worried about having **limited data**, so avoid throwing data away
  without showing the cost first. She wants decisions surfaced, not made silently.
- Git workflow (from `CONTRIBUTING.md`): short-lived branch from `main`, focused commits with
  imperative messages, PR against `main`, at least one teammate reviews, update
  `PROJECT_TRACKER.md` in the same PR when a task changes status, never commit data / `.npy` /
  weights / checkpoints / credentials.
- The Claude Code session that wrote the code **could push to `feature/data-loader`** (it cannot
  and should not push to `main`). Commits carry the attribution trailers the harness asks for.

---

## 2. Source documents and what they say

Both PDFs are in the claude.ai Project named "CSCI 566" ("Deep_Learning_Project_Proposal (2).pdf"
and "Team Division & Scope Document.pdf"). They were read on 2026-10-24 (an earlier message in
the session described them before they had actually been opened; they have since been read and
checked against the code).

### 2.1 Project proposal
- Title: "Deep Learning for Detection of Manipulated Medical Images".
- Goal: unified 2D pipeline to separate authentic from manipulated medical images; compare CNNs;
  evaluate **generalization across manipulation types**; use interpretability to see whether
  predictions focus on the manipulated region.
- Data processing: normalization, resizing, cropping, **limited** augmentation; train/val/test
  separated **at the patient or source-scan level where possible** to reduce leakage.
- Models: ResNet and EfficientNet for binary classification. Optionally combine neighbouring
  slices as input for spatial context (not planned yet).
- Interpretability: Grad-CAM; check whether hot regions match the altered lesion.
- Metrics: **Accuracy, Precision, Recall, F1, Specificity, AUC-ROC**, with particular attention to
  **false negatives** (a manipulated image classified as authentic).
- Timeline: weeks 1-3 dataset preparation; weeks 4-6 baselines and midterm report; weeks 7-10
  tuning, generalization, interpretability, error analysis; week 11 final report and the
  **November 30** poster defense.
- References: LIDC-IDRI (Armato et al., 2011), MedForge (Chen et al., 2026, arXiv:2603.18577),
  CT-GAN (Mirsky et al., USENIX Security 2019).

### 2.2 Team Division and Scope document
- Consolidates everything around **one dataset: BTD MRI and CT Deepfake Test Sets (Kaggle)**,
  chosen because it ships pre-extracted 2D slices, covers CT and MRI, covers GAN (CTGAN) and
  diffusion (Stable Diffusion) tampering, and has CSVs with (x, y) lesion coordinates for Grad-CAM checks.
- Team A tasks: download from Kaggle; script to load `.npy`; preprocessing (below); parse the CSVs
  and split train/val/test **based on tamper method** (separate CTGAN from Stable Diffusion).
- Team B tasks: baseline ResNet and EfficientNet, binary classification, generalization experiments
  (train on CTGAN fakes, test on Stable Diffusion fakes), track metrics with focus on minimizing false negatives.
- Team C tasks: Grad-CAM, heatmaps for manipulated images, script that checks whether the hottest
  part of the heatmap aligns with the (x, y) coordinates in the CSVs.
- Preprocessing spec for Team A: (1) load with `numpy.load()` and convert to tensors; (2) intensity
  normalization with **Min-Max scaling** to [0, 1] or [0, 255]; (3) duplicate the grayscale channel
  to 3 channels (`[3, 224, 224]`); (4) resize to the CNN input size (e.g. 224x224) and, **for the
  training set only**, limited augmentation: random horizontal/vertical flips and small rotations
  (about +/-10 degrees); **no aggressive warping or cropping**, because that could distort or remove
  the small tampered lesions.

### 2.3 Where reality differs from the scope document
1. **Normalization.** The scope doc says Min-Max. The raw CT arrays are in Hounsfield-like units whose
   minimum differs per scan (-3024, -2048, -1024), so per-image Min-Max would give the same tissue
   different brightness in different scans. The code uses a **fixed CT window** instead (clip to
   [-1000, 400], scale to [0, 1]). This is a deliberate deviation. **It needs the team's agreement**
   because Team B/C will see different pixel values than the doc describes. It is a one-line change
   in `src/config.py` (`hu_min`, `hu_max`) if the team prefers something else.
2. **"No HU windowing needed."** The scope doc implies pre-extracted slices avoid HU windowing. In
   practice the arrays are still raw HU-scale values, so some form of fixed scaling is required.
3. **MRI is not `.npy`.** The scope doc says the dataset is `.npy`. CT slices are `.npy` (4,760
   files), but the MRI slices are **PNG** (525 files). MRI needs its own loader (not written).
4. **Models.** The scope doc mentions ResNet50; the course feedback (below) suggests ResNet-18/34
   starting points. The model choice belongs to Team B.
5. **Framework.** The scope doc leaves PyTorch/TensorFlow open. The code uses **PyTorch** (Claude
   suggested it; Saloni proceeded with it).

### 2.4 Feedback received on the pre-proposal (verbatim from Saloni)
> This project is well scoped and the research objective is meaningful. One risk might be that unlike
> ordinary image classification tasks, the feature difference between a real and synthesized medical
> image might be much more subtle and harder to capture, and the model may be prune to overfitting to
> irrelevant visual cues, which require more careful hyperparameter selection and regularization
> technique. You may also try starting with a pretrained CNN model like ResNet-18/34 and see if it
> yields better results than one trained from scratch. The project can be further strengthened by
> analyzing what image patterns your model is good at differentiating and what patterns your model
> struggles with, by sampling from the success and failure sets and see what are in common in each set.

How the work so far responds to it: the shortcut checks (slice number, scanner, file type) exist
precisely because of the "overfitting to irrelevant visual cues" warning. The shortcut AUCs become
the bar the model must beat. Pretrained-vs-scratch and success/failure analysis are for Team B/C
(see section 9).

---

## 3. The dataset

### 3.1 What it is (Kaggle description)
"BTD MRI and CT Deepfake Test Sets": CT and MRI images in which cancerous tumours were **added or
removed** by generative models. Companion paper: *Back-in-Time Diffusion: Unsupervised Detection of
Medical Deepfakes* (arXiv 2407.15169). Demo/eval script from the authors:
`github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes`.

- Tamper methods: **CTGAN** (GAN, CT only) and **Stable Diffusion** (CT and MRI).
- Classes: **TM** true-malign (authentic, has a tumour), **FM** fake-malign (tumour injected),
  **TB** true-benign (authentic, no tumour), **FB** fake-benign (tumour removed).
- Six test sets (each has two classes): CTGAN-CT-Inject (TM, FM), SD-CT-Inject (TM, FM),
  CTGAN-CT-Remove (TB, FB), SD-CT-Remove (TB, FB), SD-MRI-Inject (TM, FM), SD-MRI-Remove (TB, FB).
- CSV columns per the description: Y, X (centre of the area of interest, i.e. the injected or
  removed tumour), scanner, cur_slice, path, type.
- This is a **test-set-sized** release, so the amount of data is small (see counts below). That
  matches the feedback's advice to start from a pretrained CNN and regularize.

### 3.2 What is actually on disk (from the dataset report Saloni generated)
- File types: 4,760 `.npy`, 525 `.png`, 10 `.csv`, 1 `.txt`, 1 `.py` (the last two are the report
  helper outputs/script).
- Folder layout (Windows paths in the report; the CSVs use forward slashes):
  - `CT/injection/FM_CTGAN/<scan_id>/<slice>.npy` : CTGAN fakes
  - `CT/injection/FM_SD/<scan_id>/<slice>.npy`    : Stable Diffusion fakes
  - `CT/injection/TM/<scan_id>/<slice>.npy`       : authentic with tumour
  - `CT/removal/FB_CTGAN/...`, `CT/removal/FB_SD/...`, `CT/removal/TB/...` (same pattern)
  - `CT/injection/CT_injection.csv`, `CT/removal/CT_removal.csv`: combined CSVs (**these are what the code uses**)
  - per-folder `data.csv` files (`FM_CTGAN/data.csv`, `FM_SD/data.csv`, `TM/data.csv`, `FB_*`, `TB`):
    **not used** (see 3.5)
  - `MRI/<scanner>/Breast_MRI_###/...png`, `MRI/MRI_injection.csv`, `MRI/MRI_removal.csv`
- `<scan_id>` is a long DICOM-style UID such as `1.3.6.1.4.1.14519.5.2.1.6279.6001.<digits>` (this
  is the LIDC-IDRI lung CT series naming). It identifies the patient scan. **It is the key used for leakage-safe splitting.**
- Slice arrays: shape (512, 512). Stored dtypes differ: injection TM **int16**, injection FM_CTGAN
  **int16**, injection FM_SD **float64**; removal TB, FB_CTGAN and FB_SD are all **float64**.
- Intensity ranges: minimum value is typically -3024, -2048 or -1024 depending on the scan, maximum
  roughly 1100-3100. Values are Hounsfield-like. Mean around -450 to -1090.

### 3.3 Counts
Scans (distinct `scan_id` folders):

| Task | Fake (CTGAN / SD) | Authentic | Shared between fake and authentic |
|---|---|---|---|
| Injection | 50 / 50 | TM: 220 | **14** |
| Removal | 207 / 207 | TB: 207 | 207 (all) |

`CT_injection.csv`: 3,945 rows, 8 columns (`type, path, x, y, scanner, cur_slice, CTGAN-CT-injection,
SD-CT-injection`). `type` is FM (2,748 rows) or TM (1,197 rows). Each method's flag column marks
2,571 rows (1,374 FM + 1,197 TM); the TM rows have both flags set to 1, i.e. **the same authentic
images are shared by the CTGAN and SD test sets**. Unique authentic paths: 1,141; unique fake paths
per method: 1,247. Duplicated paths inside the CSV: 310 (254 fake + 56 authentic rows).
Scanners (5): LightSpeed16 1,200; LightSpeed Ultra 869; Sensation 16 763; LightSpeed QX 573;
LightSpeed Pro 16 540. Scanner x type (FM / TM): LightSpeed Pro 16 480/60, LightSpeed QX 486/87,
LightSpeed Ultra 576/293, LightSpeed16 588/612, Sensation 16 618/145.
Slice index (`cur_slice`): TM 15-132; FM 34-475.

`CT_removal.csv`: 1,146 rows, columns `type, path, x, y, scanner, cur_slice, CTGAN-CT-removal,
SD-CT-removal`. FB 764 rows, TB 382 rows; each method's set is 382 FB + 382 TB. Unique authentic
paths 375 (shared by both methods). Duplicated paths: 21. Scanners (4, no Pro 16): LightSpeed16
612, LightSpeed Ultra 294, Sensation 16 153, LightSpeed QX 87 (in the combined table; per method
these are halved for FB). Slice index 16-132 for both real and fake. x range 17-495, y 118-434.

MRI: `MRI_injection.csv` and `MRI_removal.csv`, 350 rows each, 175 fake + 175 authentic, perfectly
balanced per scanner (Avanto 35+35, Optima MR450w 49+49, SIGNA HDx 42+42, Signa HDxt 49+49).
Columns `type, path, x, y, scanner` (**no `cur_slice` column**). PNG files named like
`MRI/Signa HDxt/Breast_MRI_025/injection_034.png` (fake injection), `.../034_fake.png` (fake
removal), and plain `036.png` (authentic). x range 81-430, y 64-389. The patient/scan key for MRI
would be the `Breast_MRI_###` folder (**ASSUMPTION**, not yet used).

### 3.4 Coordinates (important for Team C's Grad-CAM check)
- `x` is the **column** (left-right) and `y` the **row** (top-bottom) in the original 512x512 image
  (this is how the code and the sample-image overlay treat them; **the overlay circle position still
  needs Saloni's visual confirmation**, see section 7).
- For fake images the (x, y) is the injected or removed tumour location.
- In **removal**, the authentic (TB) row for the same scan and slice has a **mirrored x**:
  x_TB = 512 - x_FB (e.g. 295 vs 217, 154 vs 358, 370 vs 142; the y values are identical). So the
  authentic image's coordinates are **not** a tumour location. Only use fakes' coordinates for the
  Grad-CAM alignment check. (**OPEN QUESTION**: why they are mirrored, i.e. whether the TB image is
  a flipped counterpart or simply a differently-indexed location, was not determined. Viewing
  matched TB / FB_CTGAN / FB_SD images side by side, or reading the BTD paper, would settle it. If
  TB turns out to be a horizontally flipped version of the same slice, horizontal-flip
  augmentation would interact with that cue.)
- In injection, authentic (TM) rows also have coordinates (a real tumour). The loader sets x, y to
  -1 for **all** authentic images; this is fine for Grad-CAM-on-fakes but loses the real-tumour location if anyone needs it.
- MRI injection and removal coordinates for the same slice also look mirrored (e.g. 430 vs 81,
  which sum to 511).

### 3.5 CSV and path quirks the code handles
- Kaggle's description names the flag columns `CTGAN-CT-Inject`, `SD-CT-Inject`, etc. The real
  columns are `CTGAN-CT-injection`, `SD-CT-injection`, `CTGAN-CT-removal`, `SD-CT-removal`.
  `src/splits.py` finds them by prefix (`CTGAN-`, `SD-`).
- The combined CSVs hold full relative paths (`CT/injection/FM_SD/<scan>/<slice>.npy`), resolvable
  under the dataset root. The per-folder `data.csv` files hold partial paths and, for the CTGAN and
  TB removal folders, an extra subfolder (`<scan>/44/044.npy`). They also use different `type`
  strings (`diffusion_FM`) and the TM one has an `Unnamed: 0` column. Because of this, the code only
  uses the combined CSVs. `check_data` confirmed there are 0 missing files with this approach.
- Some removal filenames are zero-padded (`088.npy`) while others are not (`44.npy`); the CSV paths
  already match the disk, so nothing needs normalizing.
- Duplicate paths (310 in injection, 21 in removal): the code merges them into one row per path and
  keeps every coordinate in `xs` / `ys` lists. **ASSUMPTION:** the duplicates are multiple lesions
  on one slice. This has **not** been verified. Check whether the duplicate rows differ in x/y (if
  they are exact copies, drop them instead). The `meta` returned by the dataset only carries the
  **first** lesion's x, y.

---

## 4. Findings so far

### 4.1 Real `check_data` output (run by Saloni on the real data, 2026-10-21)
```
=== injection | train on CTGAN ===
            scans  authentic  fake
split
train         180        850   769
val            38        152   187
test_same      38        139   291
test_other     38        139   291
   leakage check: OK (no scan appears in two splits)
   missing files: 0
   raw dtype, authentic: {'int16': 30}
   raw dtype, fake     : {'int16': 23, 'float64': 7}
   preprocessed: shape=(3, 224, 224), value range=[0.000, 1.000] (expect within [0, 1])
   slice filter 34-132: what it WOULD keep (not applied unless you set slice_range)
      CTGAN authentic: keep 1121 of 1141
      CTGAN fake     : keep 575 of 1247
      SD    authentic: keep 1121 of 1141
      SD    fake     : keep 575 of 1247
   shortcut model, slice number only: val AUC = 0.742  (0.5 = no leak)
   shortcut model, scanner only     : val AUC = 0.717  (0.5 = no leak)
   saved sample grid: sample_images\samples_injection_ctgan.png (red circle = lesion position on fakes)

=== removal | train on CTGAN ===
            scans  authentic  fake
split
train         143        250   250
val            32         53    53
test_same      32         72    72
test_other     32         72    72
   leakage check: OK (no scan appears in two splits)
   missing files: 0
   raw dtype, authentic: {'float64': 30}
   raw dtype, fake     : {'float64': 30}
   preprocessed: shape=(3, 224, 224), value range=[0.000, 1.000] (expect within [0, 1])
   slice filter 34-132: what it WOULD keep (not applied unless you set slice_range)
      CTGAN authentic: keep 368 of 375
      CTGAN fake     : keep 368 of 375
      SD    authentic: keep 368 of 375
      SD    fake     : keep 368 of 375
   shortcut model, slice number only: val AUC = 0.500  (0.5 = no leak)
   shortcut model, scanner only     : val AUC = 0.500  (0.5 = no leak)
   saved sample grid: sample_images\samples_removal_ctgan.png (red circle = lesion position on fakes)
```
(Note: `test_other` has identical authentic images to `test_same`, because the authentic images
are shared by both methods; only the fakes differ.)

### 4.2 Interpretation
- **Pipeline works end to end on the real data:** no missing files, no scan in two splits, images
  come out as 3x224x224 with values within [0, 1].
- **Removal is clean.** Real and fake slices come from the same 207 scans, are balanced 1:1, cover
  the same slice range, and the shortcut models score 0.500 (a coin flip). Anything a model scores
  here is real detection.
- **Injection has shortcuts.**
  - Slice number alone: AUC 0.742. Scanner alone: AUC 0.717 (0.5 would be no leak).
  - Fakes come from only 50 scans, authentic from 220, and only 14 scans appear on both sides, so
    in injection the real and the fake images are mostly **different patients** (this is why scanner
    leaks the label). **UNVERIFIED hypothesis** for why: fakes were probably generated from
    patients/slices without a nodule at that spot, while the TM images are real nodule slices; the
    BTD paper should confirm.
  - Slice position: authentic 15-132 vs fake 34-475. The slice filter (34-132) would keep 1,121 of
    1,141 authentic but only 575 of 1,247 fakes (46%), and it does **not** fix the different-patients problem.
- **Class imbalance in injection evaluation sets:** val 152 real / 187 fake; test 139 real / 291
  fake. A classifier that always says "fake" gets about 68% accuracy on test. Judge by AUC, F1,
  balanced accuracy, specificity and recall on fakes, not plain accuracy. Training is 850/769 (close to balanced).
- **dtype:** the raw dtype differs by class in injection (int16 vs float64 for SD fakes). The loader
  casts everything to float32 so the model cannot see the type directly, but float64 SD fakes may
  hold non-integer values (CT originals are integers), a subtle cue **UNVERIFIED**.
- **Data are small.** Removal: 250 train images per class. This supports pretrained
  ResNet-18/34, strong regularization, careful hyperparameters (feedback), and reporting variance across seeds.

---

## 5. Decisions made (and why)

| # | Decision | Reason | Status |
|---|---|---|---|
| 1 | Use **PyTorch** | Suggested by Claude; Saloni proceeded | Settled |
| 2 | Pipeline order: float32 cast, **fixed CT window** [-1000, 400] clipped and scaled to [0,1], resize 224 (bilinear, antialias), copy to 3 channels, train-only augmentation, then ImageNet mean/std | Consistent brightness across scans (min differs per scan); pretrained weights expect ImageNet-normalized input | Deviates from scope doc's Min-Max; **needs team sign-off** |
| 3 | Augmentation: horizontal + vertical flip (p=0.5), rotation +/-10 degrees, **train only** | Matches scope doc; avoids destroying small lesions | Settled (but see the TB mirrored-x open question) |
| 4 | **Split by scan ID**, stratified by scanner, seed 42, 70/15/15 | Proposal requires patient/source-scan level separation; the same slices are reused across CTGAN and SD sets, so a scan must never straddle train and test | Settled |
| 5 | Train and validate on **CTGAN**; **Stable Diffusion = held-out generalization test** (`test_other`, held-out scans only); `train_method="SD"` reverses it | Scope doc's generalization experiment | Settled as default |
| 6 | Use the **combined CSVs** only, not per-folder `data.csv` | Per-folder files have inconsistent paths and types | Settled |
| 7 | Scope: **CT injection and CT removal now**; **MRI later as a stretch** | Same `.npy` loader and splits for CT; MRI is PNG, SD-only (cannot test GAN to diffusion), needs its own loader | Saloni: "do both if possible"; MRI deferred, revisit |
| 8 | **Separate** models/sets for injection and removal first; a **mixed** model as a later experiment | Easier to interpret; injection and removal have different shortcuts. Mixed: more data but harder to debug | Proposed, accepted implicitly by "go"; revisit |
| 9 | **Never feed slice number or scanner to the model** | They are shortcuts | Settled |
| 10 | **Slice filter OFF by default** (`slice_range=None`) | Saloni worried about losing data; the filter would drop 54% of injection fakes | Settled for now; **final default undecided** |
| 11 | **Evaluate both ways** (full test set and slice-matched subset 34-132) and run a **"dumb baseline"** that sees only the slice number (and one with scanner only) | Shows whether a model leans on the shortcut without throwing data away | Baselines implemented in `check_data`; slice-matched evaluation **not yet built** |
| 12 | Use **removal as the main clean experiment** for the midterm; injection with filter + class weights, plus a shared-scan (14 scans) evaluation | Removal has no shortcuts; injection does | **Proposed, awaiting Saloni's yes/no** |
| 13 | Work happens on **`feature/data-loader`**, merged by PR after teammate review | CONTRIBUTING.md; Saloni asked for a feature branch | In progress |
| 14 | Tracker: only the Kaggle-download box was ticked; the code boxes stay unticked until the real-data run is confirmed by Saloni and merged | Honest status | See section 6.4 |

---

## 6. Code state (branch `feature/data-loader`)

### 6.1 Files added or changed
```
src/__init__.py              empty package marker
src/config.py                settings: dataset root, CT window, image size, split fractions, augmentation
src/splits.py                CSV cleaning, scan-level splits, leakage check, summary
src/data_loader.py           BTDDataset + get_dataloaders
src/check_data.py            sanity checks + sample images (the script that produced section 4.1)
tests/test_data_pipeline.py  12 tests on a synthetic copy of the BTD layout (no real data)
requirements.txt             numpy, pandas, torch, torchvision, scikit-learn, matplotlib, pytest
README.md                    added a "Data pipeline (CT)" usage section
PROJECT_TRACKER.md           ticked the download box; added notes and two new Team A items
.gitignore                   added sample_images/
docs/HANDOFF.md              this file
```

### 6.2 How it works
- **`src/config.py`**
  - `get_data_root(root=None)`: dataset folder from the argument or env var `BTD_DATA_ROOT`; errors
    clearly if unset or if there is no `CT` folder inside.
  - `DataConfig` defaults: `hu_min=-1000`, `hu_max=400`, `image_size=224`, `imagenet_norm=True`,
    `seed=42`, `val_frac=0.15`, `test_frac=0.15`, `slice_range=None`, `flip_prob=0.5`, `rotation_deg=10`.
- **`src/splits.py`**
  - `load_task_table(task, root)` (`task` is `"injection"` or `"removal"`): reads the combined CSV,
    normalizes backslashes, merges duplicate paths, adds `label` (1 = fake, 0 = authentic), `scan_id`
    (the folder name before the file), `in_ctgan`, `in_sd` flags, `xs`/`ys` lesion lists.
  - `assign_scan_splits(df, cfg)`: per scanner group, shuffles the scans with the seed and assigns
    test/val/train (at least one scan each to val/test when a scanner group has 3 or more scans).
    The assignment is made **before** any slice filtering so switching the filter on never reshuffles the split.
  - `build_splits(task, train_method="CTGAN", root, cfg)` returns `{"train", "val", "test_same", "test_other"}`.
  - `check_no_leakage(splits)` raises if a scan sits in two groups (train, val, and test as a union of both test sets).
  - `summarize(splits)` returns counts of scans, authentic, fake.
- **`src/data_loader.py`**
  - `load_slice`, `window_and_scale` (clip and scale), `BTDDataset(df, root, cfg, train)`.
  - Each item is `(image [3,S,S] float tensor, label int, meta dict)`; `meta` has `path`, `scan_id`,
    `scanner`, `cur_slice`, `x`, `y` (lesion centre rescaled to the resized image, -1 for authentic).
    **x, y are only valid when augmentation is off** (val/test), since flips and rotation move the lesion.
  - `get_dataloaders(task, train_method, root, cfg, batch_size=32, num_workers=0)` returns a dict of
    `DataLoader`s (shuffle only for train). `num_workers=0` is the safe default on Windows.
- **`src/check_data.py`** (`python -m src.check_data --root <dir> --task both|injection|removal`, also
  `--train-method`, `--out`, `--slice-min`, `--slice-max`): prints split table, leakage check,
  missing-file check, raw dtypes by class, preprocessed range and shape, what the slice filter
  would remove, two shortcut models (depth-3 decision trees on slice number only and on scanner
  only, reported as validation AUC), and saves a sample-image grid (red circle at the lesion on fakes).
- **Tests:** `python -m pytest tests -q` gives 12 passing tests on synthetic data: leakage for both
  tasks and methods, method separation, duplicate merging, reproducibility, filter off by default and
  split unchanged when switched on, window scaling, item shape/range/channel equality, and
  augmentation only in train. They were run in the Claude sandbox (Python 3.13, CPU torch 2.14).
  Saloni was asked to run them on her machine too; her result for the tests was not reported (she
  reported the `check_data` run).

### 6.3 Usage snippet
```python
from src.data_loader import get_dataloaders
loaders = get_dataloaders("removal", train_method="CTGAN", batch_size=32)
# keys: train, val, test_same (CTGAN fakes), test_other (Stable Diffusion fakes)
# batch: images [B,3,224,224], labels (1 = manipulated), meta (path, scan_id, scanner, cur_slice, x, y)
```

### 6.4 Branch and PR status
- Remote branch: `origin/feature/data-loader`. Base `main` is at `150313f`; the code commit is
  `2f393c8`; this document is a later commit on the same branch.
- Saloni had first created her own local branch of the same name from `main`, so it lacked the
  commit; the fix was `git fetch origin` then `git merge --ff-only origin/feature/data-loader`.
  She has since run the code, so her local copy has the files.
- No PR was known to be open as of this writing (the Claude session cannot see GitHub PRs). A teammate (Rohan) should review it before merging, per CONTRIBUTING.
- `PROJECT_TRACKER.md`: the Team A code items are intentionally still unticked; tick them in the PR once Saloni has confirmed the checks and sample images.

---

## 7. Known limitations, assumptions, unverified items

1. **Sample-image overlay not yet visually confirmed.** The red circle uses the CSV's (x, y) scaled
   to the resized image. If it does not sit on the tumour, the x/y convention (column vs row) or
   the scaling is wrong and Team C's Grad-CAM check is affected. Saloni was asked to look at both PNGs.
2. **Duplicate rows assumed to be multiple lesions** (section 3.5). Unverified.
3. **`meta` carries only the first lesion's coordinates**; `xs` / `ys` exist in the table but are not passed through.
4. **Authentic images get x=y=-1**, even TM images that contain a real tumour.
5. **Removal TB x is mirrored** and its meaning is unexplained (section 3.4).
6. **The shortcut checks only test slice number and scanner.** They do not test noise level, global
   brightness, patient-specific appearance, or the dtype/integer-valued cue. A CNN could still
   find other cues (injection especially).
7. **Injection real-vs-fake come from mostly different patients** (50 fake scans, 220 real, 14
   shared). Even with the slice filter this remains. The paired comparison exists only in removal
   and in the 14 shared injection scans.
8. **Stratification is by scanner only**, not by class, so the injection val/test sets came out fake-heavy.
9. **Small data:** removal train has 250 images per class; metrics will be noisy. A single split can be misleading, so multiple seeds or cross-validation by scan are worth considering.
10. **Windowing choice** [-1000, 400] covers lung and soft tissue but clips bone/metal and very dense structures; fine for lung nodules, not validated for anything else.
11. **MRI is untouched** (no loader, no splits, no checks).
12. **Rotation fill value** is 0 (which is air after windowing, so fairly harmless) and torchvision's default nearest interpolation is used for rotation on tensors.
13. The tests only use synthetic data; they cannot catch problems specific to the real files (e.g. an unusual array shape). The real run did not hit any.
14. Local hardware/OS differences: code was developed on Linux and run by Saloni on Windows; no Windows-specific issue has appeared yet. Watch `num_workers>0` on Windows.

---

## 8. Open questions and decisions for the team

1. **Normalization:** keep the fixed CT window (current) or revert to the scope doc's per-image Min-Max? (Needs Team B/C awareness either way.)
2. **Slice filter default for injection:** keep OFF, or turn ON (drops 54% of injection fakes but removes the slice shortcut)? Current proposal: OFF for training, then report results on both the full test set and the slice-matched subset.
3. **Is injection worth reporting as a headline benchmark** given the patient mismatch, or should it be framed as a "shortcut analysis" finding with removal as the main benchmark?
4. **Mixed injection+removal model:** run as a later experiment?
5. **MRI:** when to add it, and as what (a separate SD-only experiment, since it cannot test GAN to diffusion)?
6. **Metrics definition:** the proposal lists Accuracy, Precision, Recall, F1, Specificity, AUC-ROC with emphasis on false negatives. Decide the primary metric for model selection (suggest AUC-ROC and recall on fakes, plus balanced accuracy).
7. **Seeds / repeated splits** for stability given the small data.
8. **What do TB images in removal actually represent?** (Mirrored x; same scan and slice as FB.)
9. **Ownership:** which Team A member does what (tracker lists "[Assign 2 Members]").

---

## 9. Next steps

### Proposed and waiting on Saloni's yes (Team A, same branch)
1. **Balanced-metrics helper** (`src/metrics.py`): accuracy, precision, recall (sensitivity), specificity,
   F1, balanced accuracy, AUC-ROC, confusion matrix with the **false-negative count** highlighted, matching the proposal's list.
2. **Class weights** (or `pos_weight`) computed from the training split, exposed from the loader module.
3. **Slice filter as a clear switch** (it already exists as `DataConfig.slice_range`; add an evaluation
   helper that scores the same predictions on the slice-matched subset 34-132).
4. **Shared-scan evaluation set** for injection (the 14 scans that have both real and fake) as an extra
   test group, plus a `check_data` line reporting its size.
5. Update `PROJECT_TRACKER.md` boxes after Saloni confirms the run and sample images.

### Other Team A follow-ups
6. Verify the duplicate-row assumption and pass **all** lesion coordinates (`xs`, `ys`) through `meta`.
7. Resolve the TB mirrored-x question by viewing matched TB / FB_CTGAN / FB_SD images; decide whether
   horizontal-flip augmentation is safe for removal.
8. Extra shortcut checks: simple image-statistics baselines (mean brightness, noise level, integer-valuedness), maybe a scan-level baseline.
9. MRI loader: PNG reader, splits keyed by the `Breast_MRI_###` folder, same preprocessing, SD-only experiments.
10. Optional: cross-validation or multiple split seeds; cache preprocessed tensors for speed if training is slow.
11. Open the PR for `feature/data-loader` and ask Rohan to review; add a short note on the normalization deviation in the PR description.

### Hand-offs to other teams (so Team A's output is used correctly)
- **Team B:** start with a pretrained ResNet-18/34 vs from-scratch comparison (the feedback), strong
  regularization (weight decay, dropout, early stopping on validation), careful learning rates; use
  `test_same` and `test_other` for the CTGAN to SD generalization result; report the shortcut AUCs
  (0.74 slice, 0.72 scanner for injection) as the floor a model must beat; do a success/failure
  analysis (what patterns it separates well vs poorly, as the feedback asked); track false negatives.
- **Team C:** Grad-CAM on the trained models; the (x, y) coordinates in `meta` are in the **resized
  224x224 image space** and valid only for val/test items (no augmentation); use **fakes only**; in
  removal ignore authentic x/y (mirrored); handle multi-lesion slices once `xs`/`ys` are exposed.

---

## 10. Chronology of the conversation (for context)

1. Saloni, on Team A, shared the project documents, the pre-proposal feedback and the repo URL, asked
   for an understanding of the project and scope, and asked to **explain before executing, in simple words**.
2. The repo was cloned read-only. It held only scaffolding (README, tracker, contributing guide,
   `.gitignore`, empty `src/`, `docs/`, `results/`); every Team A tracker box was open.
3. Claude summarized the project and flagged: leakage risk from reused slices, label mapping, the
   "overfitting to irrelevant cues" warning, and the normalization question. It asked scoping questions.
4. Saloni first generated a **dataset report** (file types, per-folder file counts, sample array
   stats, CSV profiles). Claude analysed it (sections 3 and 4) and asked a second round of questions.
5. Saloni's answers: do both CT tasks if possible; wanted pros and cons for separate vs mixed
   models; asked what "normalization" meant (explained: fixed window vs per-image min-max);
   agreed to casting dtypes, stratifying by scanner and reporting the slice issue, and asked whether
   slice position needs to be a parameter at all (answer: never an input, but the image reveals it
   anyway); gave the dataset and repo locations; asked for a feature branch first.
6. Saloni was nervous about dropping data via the slice filter; the agreed approach: filter OFF by
   default, evaluate both ways, run a slice-number-only "dumb baseline".
7. On "go", Claude wrote the code (section 6), tested it on synthetic data (12 tests), pushed to
   `feature/data-loader` and told her how to fetch it (her local branch had the same name but was
   created from `main`).
8. Saloni ran `check_data` on the real data (section 4.1). Claude interpreted it and proposed the
   next changes (section 9). Saloni asked for a simpler explanation of the output, then asked for this handoff.

---

## 11. Glossary (the plain-words versions given to Saloni)

- **Slice:** one 2D image cut from a 3D scan.
- **Scan / patient ID:** the long number in the folder name; all slices from one patient scan share it.
- **Leakage:** the same patient appearing in both training and test, so the model looks smarter than it is.
- **Shortcut:** a clue unrelated to tampering (slice number, scanner) that still separates real from fake.
- **AUC:** score where 0.5 is a coin flip and 1.0 is perfect. The "shortcut AUC" is how well a tiny model can separate real from fake from the clue alone.
- **Fixed CT window:** pick one brightness range for every image (here -1000 to 400), clip to it, then scale to 0-1, so the same tissue always gets the same brightness.
- **Per-image Min-Max:** stretch each image using its own lowest and highest value (brightness then depends on that image's extremes).
- **False negative:** a fake image the model calls real (the costly error in this project).
- **Specificity:** fraction of authentic images correctly called authentic.
- **Generalization test:** train on one fake type (CTGAN), test on another (Stable Diffusion).
- **Grad-CAM:** a heatmap of where the model looked; Team C checks whether it points at the tampered spot.
