# Medical Deepfake Detection

A collaborative research project for developing a 2D deep-learning pipeline that distinguishes authentic CT/MRI scans from synthetically manipulated scans.

## Project status

Repository setup is complete. Data ingestion, modeling, generalization experiments, interpretability work, and reporting are tracked in [PROJECT_TRACKER.md](PROJECT_TRACKER.md).

## Planned approach

- Load 2D `.npy` slices from the BTD MRI and CT Deepfake Test Sets.
- Train ResNet and EfficientNet binary classifiers.
- Evaluate generalization across GAN- and diffusion-based manipulations.
- Generate Grad-CAM heatmaps for model interpretation.
- Track accuracy, precision, recall, F1 score, and false negatives.

## Repository layout

```text
src/       Source code
docs/      Project documentation and reports
results/   Generated summaries and lightweight result artifacts
```

## Data pipeline (CT)

```bash
pip install -r requirements.txt
# point the code at the dataset folder (the one containing CT/ and MRI/)
export BTD_DATA_ROOT=/path/to/archive          # PowerShell: $env:BTD_DATA_ROOT = "C:\data\btd\archive"

python -m src.check_data --task both           # splits, leakage, shortcut checks, sample images
python -m pytest tests -q                      # unit tests (synthetic data, no real scans needed)
```

```python
from src.data_loader import get_dataloaders

loaders = get_dataloaders("injection", train_method="CTGAN", batch_size=32)
# keys: train, val, test_same (CTGAN), test_other (Stable Diffusion = generalization test)
# each batch: images [B, 3, 224, 224], labels (1 = manipulated), meta (path, scan_id, x, y, ...)
```

Splits are made by scan ID (never by slice), stratified by scanner. Settings live in `src/config.py`.

CT preprocessing follows the dataset authors' [evaluation notebook](https://github.com/FreddieMG/BTD--Unsupervised-Detection-of-Medical-Deepfakes/blob/0fa74daef6802d305dcca4e5e37b4393fdf89c5b/BTD_evaluation.ipynb):
clip to **[-700, 2000]**, scale to [0, 1], and extract a **96x96 patch at each CSV (x, y)**
for both authentic and manipulated rows. Patches near an edge shift inward rather than
being padded. Each CSV row remains a sample, including repeated paths and exact duplicate
rows, matching the reference loader. Counts therefore refer to ROI samples, not unique files.

The CNN adapter then resizes the patch to 224x224, duplicates it to three channels, applies
training-only augmentation, and optionally applies ImageNet normalization. These are CNN
adaptations, not part of the authors' one-channel diffusion-model input. To inspect the native
patch values, use `DataConfig(image_size=96, imagenet_norm=False)`; each channel then contains
the reference patch, without augmentation when `train=False`.

`meta.x/y` are ROI coordinates within the resized patch for **both** classes; authentic
removal coordinates identify healthy tissue rather than a lesion. `source_x/y` and
`crop_left/top` preserve original-slice geometry. Patch coordinates are valid only without
augmentation (validation/test). Grad-CAM checks on manipulated regions should use fake rows.

Whole-slice classification is not valid for the sampled CTGAN removal pairs: TB and FB files
can contain identical full arrays with different ROI coordinates and labels. Preserve the
coordinate-defined samples when changing this pipeline.

The full CT audit and downstream contract are documented in [Team A's CT handoff](docs/CT_DATA_HANDOFF.md).
Run `python -m src.audit_data --root "$BTD_DATA_ROOT" --out results/ct_audit` for exhaustive
reference, coordinate, hash, split, and batch checks. The handoff also records remaining
dataset biases and why injection/removal splits must not be concatenated.

## Dataset and artifact policy

Do not commit medical-image data, `.npy` files, CSV dataset exports containing sensitive or restricted information, trained model weights, checkpoints, or generated experiment output. Obtain the dataset through its authorized source and keep it outside version control. Review the dataset license and institutional requirements before use.

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

