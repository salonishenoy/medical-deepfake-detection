# Medical Deepfake Detection: Project Tracker

## Project Overview

**Goal:** Develop a 2D deep learning classification pipeline to distinguish authentic medical scans (CT/MRI) from synthetically manipulated ones (deepfakes).

**Dataset:** BTD MRI and CT Deepfake Test Sets (2D `.npy` slices).

**Core Models:** ResNet, EfficientNet.

**Key Deliverables:** Binary classifier, generalization experiments (GAN vs. Diffusion), and Grad-CAM interpretability heatmaps.

## Team Assignments

- **Team A (Data Engineering):** [Assign 2 Members]
- **Team B (Modeling & Generalization):** [Assign 3 Members]
- **Team C (Interpretability & Reporting):** [Assign 2 Members]

## Master To-Do List

Team A environment reproduction is documented in the
[setup/replay guide](docs/REPRODUCIBILITY.md). Python 3.12.13 and all 34 packages
are pinned for macOS arm64, with a dedicated environment checker. See its evidence
report for the completed clean-environment validation results.

Team A's [preprocessing contract v1](docs/PREPROCESSING_CONTRACT.md) is frozen for
the final mechanical audit. The [JSON snapshot](docs/preprocessing_contract_v1.json)
records defaults, schemas, and source/manifest hashes. Six public loader interfaces
were checked against it; runtime preprocessing was unchanged during the freeze.
The [complete-data mechanical audit](docs/FULL_PREPROCESSING_AUDIT.md) now passes
all 5,791 regions / 5,285 files and 8,574 public-loader deliveries. All 57 tests pass.
Review/publication remain pending.

Current Team A review: further mitigation experiments are paused. The
[preprocessing correctness review](docs/PREPROCESSING_VERIFICATION.md) checks
4,689 CT/MRI development rows without an AUC criterion. Loading, scaling, native
ROI geometry, source splitting, and evaluation determinism pass. Resized
pixel-center metadata is now corrected and verified; 57 tests pass. Training
seeding responsibilities and teammate reproduction remain to confirm. Clinical annotation/orientation truth is not verified.

### Phase 1: Initiation & Dataset Preparation (Weeks 1-3)

**Goal:** Set up the repository and build the data ingestion pipeline.

CT correction prepared for `feature/data-loader`: use the reference [-700, 2000]
window and 96x96 CSV-centered patches for both classes, preserving every CSV row. The CNN
adapter still produces 3x224x224 tensors. All 4,760 CT files / 5,091 ROI rows passed the
frozen-baseline audit; 57 tests pass across CT/MRI. Review and merge remain pending. See
[CT_DATA_HANDOFF.md](docs/CT_DATA_HANDOFF.md) for results and remaining dataset biases.

- [x] Finalize project scope and proposal.
- [x] Initialize GitHub repository (`README.md`, `.gitignore`).
- [x] **Team A:** Download the BTD dataset from Kaggle to local machines (do not push to GitHub).
- [ ] **Team A:** Write `src/data_loader.py` to load `.npy` files. *(Implemented; complete frozen-baseline mechanical audit passes. Review/merge pending.)*
- [ ] **Team A:** Implement intensity normalization and 1-to-3 channel duplication. *(Implemented and mechanically verified on every CT/MRI row. Review/merge pending.)*
- [ ] **Team A:** Parse dataset CSVs to create distinct Training, Validation, and Test splits. Ensure splits are isolated by manipulation type (CTGAN vs. Diffusion). *(`src/splits.py`; all public split deliveries and source isolation verified. CT tasks must remain separate. Review/merge pending.)*
- [ ] **Team A:** Run the shortcut checks (slice position, scanner, file type) and decide whether to turn on the slice filter.
- [ ] **Team A (stretch):** Add MRI (`.png`) loading after the CT pipeline is confirmed. *(Implemented locally; all 700 rows and every MRI public split pass the frozen-default mechanical audit. Resized metadata uses pixel-center geometry; 57 tests pass. Background AUC is an observation, not an acceptance gate; mitigation is paused. See [MRI_DATA_HANDOFF.md](docs/MRI_DATA_HANDOFF.md). Review/merge pending.)*

### Phase 2: Baselines & Midterm Report (Weeks 4-6)

**Goal:** Achieve initial classification results and submit the Midterm.

- [ ] **Team B:** Build the standard ResNet and EfficientNet architectures in `src/models.py`.
- [ ] **Team B:** Write the training loop in `src/train.py`.
- [ ] **Team B:** Train the baseline binary classifier (Authentic vs. Manipulated) using the CTGAN training split.
- [ ] **Team B:** Record initial Accuracy, Precision, Recall, and F1-Score (focus heavily on tracking False Negatives).
- [ ] **Team C:** Outline and draft the Midterm Report, incorporating the literature review and baseline metrics.
- [ ] **All:** Review and submit Midterm Report.

### Phase 3: Generalization & Interpretability (Weeks 7-10)

**Goal:** Test model robustness and implement Explainable AI.

- [ ] **Team B:** Conduct Generalization Experiments (e.g., test the CTGAN-trained model on the Stable Diffusion test set).
- [ ] **Team B:** Perform hyperparameter tuning to optimize performance and reduce false negatives.
- [ ] **Team C:** Implement Grad-CAM in `src/interpretability.py`.
- [ ] **Team C:** Generate visual heatmaps for true positive predictions (correctly identified deepfakes).
- [ ] **Team C:** Write a script/function to verify if the "hottest" heatmap regions mathematically align with the $(x, y)$ coordinates provided in the dataset CSVs.

### Phase 4: Dissemination & Poster Defense (Week 11)

**Goal:** Finalize documentation and prep for the presentation.

- [ ] **Team C:** Draft the Final Report (Methodology, Results, Error Analysis).
- [ ] **All:** Design the presentation poster.
- [ ] **All:** Conduct internal Q&A sessions. Ensure every member can confidently explain the data pipeline, the CNN architecture, and the Grad-CAM results.
- [ ] **All:** November 30 - Poster Defense!

