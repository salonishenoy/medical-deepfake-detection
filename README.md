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

## Dataset and artifact policy

Do not commit medical-image data, `.npy` files, CSV dataset exports containing sensitive or restricted information, trained model weights, checkpoints, or generated experiment output. Obtain the dataset through its authorized source and keep it outside version control. Review the dataset license and institutional requirements before use.

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

