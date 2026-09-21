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

## Dataset and artifact policy

Do not commit medical-image data, `.npy` files, CSV dataset exports containing sensitive or restricted information, trained model weights, checkpoints, or generated experiment output. Obtain the dataset through its authorized source and keep it outside version control. Review the dataset license and institutional requirements before use.

## Collaboration

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch and pull-request workflow.

