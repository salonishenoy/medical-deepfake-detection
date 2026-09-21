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

### Phase 1: Initiation & Dataset Preparation (Weeks 1-3)

**Goal:** Set up the repository and build the data ingestion pipeline.

- [x] Finalize project scope and proposal.
- [x] Initialize GitHub repository (`README.md`, `.gitignore`).
- [ ] **Team A:** Download the BTD dataset from Kaggle to local machines (do not push to GitHub).
- [ ] **Team A:** Write `src/data_loader.py` to load `.npy` files.
- [ ] **Team A:** Implement intensity normalization and 1-to-3 channel duplication.
- [ ] **Team A:** Parse dataset CSVs to create distinct Training, Validation, and Test splits. Ensure splits are isolated by manipulation type (CTGAN vs. Diffusion).

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

