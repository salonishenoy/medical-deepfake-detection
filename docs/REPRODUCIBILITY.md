# Reproduce Team A preprocessing

The verified environment is **CPython 3.12.13, macOS arm64, CPU preprocessing**.
The [frozen preprocessing contract](PREPROCESSING_CONTRACT.md) remains unchanged.
This guide covers environment setup and loader replay, not model training.

## What is pinned

- [.python-version](../.python-version): Python 3.12.13.
- [requirements.txt](../requirements.txt): the eight direct dependency versions
  used in the successful audit.
- [requirements-macos-arm64-py312.lock](../requirements-macos-arm64-py312.lock):
  all 34 direct/transitive package versions from that environment.
- [preprocessing_contract_v1.json](preprocessing_contract_v1.json): configuration
  defaults, source hashes, manifest hashes, and interface schema.

The lock is a platform-scoped version snapshot. It does not claim cross-platform
bitwise equivalence or lock wheel-file hashes. Linux, Windows, Intel Macs, and GPU
environments need separate installation and preprocessing verification. The direct
pins can be a starting point there, but their transitive resolution is not this lock.

## Fresh setup on the verified platform

Start in the repository directory with a new virtual environment. With uv:

```bash
uv venv --python 3.12.13 .venv
uv pip sync --python .venv/bin/python --strict requirements-macos-arm64-py312.lock
uv pip check --python .venv/bin/python
source .venv/bin/activate
python -m src.check_environment
python -m pytest tests -q
```

The `sync` command makes the new environment match the lock, including removal
of packages not listed. Use it on a dedicated project environment.

Alternatively, with an existing CPython **3.12.13** installation:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-macos-arm64-py312.lock
python -m pip check
python -m src.check_environment
python -m pytest tests -q
```

The pip route may retain installer tools such as pip itself; the environment
checker reports extra packages separately. Every locked dependency must match.
The clean validation run used uv and had exactly the 34 locked packages.

`src.check_environment` checks Python/platform, isolation from system site packages,
all locked package versions, distribution locations, and frozen preprocessing source
hashes. It writes `results/environment.json` and exits nonzero on a mismatch. It
does not install packages, modify preprocessing, or set RNG seeds.

## Dataset and complete mechanical audit

Use dataset version 7, extracted so the configured root contains `CT/` and `MRI/`.
Both MRI task manifests are required. Keep the files outside the repository.

```bash
export BTD_DATA_ROOT=/path/to/archive
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m src.audit_preprocessing \
  --root "$BTD_DATA_ROOT" \
  --contract docs/preprocessing_contract_v1.json \
  --out results/preprocessing_v1
```

The audit validates the source/manifest fingerprints before and after checking
every released sample and public loader split. Detailed hashes and sample records
remain in the local output directory. Aggregate expected coverage is:

- 5,285 unique image files and 5,791 CSV-defined regions.
- 1,102 held-out rows checked mechanically, with no model scores.
- Six public loader combinations and 8,574 sample deliveries.
- 57 passing tests in the current suite.

See the [full audit report](FULL_PREPROCESSING_AUDIT.md) for check definitions and
experiment boundaries. A mechanical pass is not clinical annotation validation.

## Seed behavior and replay

`DataConfig.seed` / `MRIDataConfig.seed` fix source-group split allocation only.
Loading an unaugmented evaluation item uses no random image transformation.
Training-item augmentation and shuffled sample order use PyTorch randomness.

For a repeatable loader-only start with the verified zero-worker setting:

```python
import torch
from src.mri_data_loader import get_mri_dataloaders

torch.manual_seed(42)
loaders = get_mri_dataloaders("injection", root="/path/to/archive", num_workers=0)
images, labels, meta = next(iter(loaders["train"]))
```

To replay that start, reset the seed immediately before constructing fresh loaders
and iterating them. Other random draws between seeding and iteration change the
sequence. Do not expect an existing, already-consumed iterator to rewind.
Record explicit overrides of configuration, workers, and thread settings.

The clean-environment check compared the first seed-42 training batch for each of
the six loader combinations: augmented image bytes, labels, and metadata matched
exactly between environments. This does not promise reproducibility for arbitrary
workers, devices, resumed training states, or different package/runtime versions.

## Recorded evidence

The fresh environment was empty, had system-site-package access disabled, and
received the 34 packages through an offline uv installation from cached artifacts.
Installed site-packages were not copied from the working environment. Both
environments used the same host and base Python build.

The checker also rejected an intentionally incorrect package pin, demonstrating
that it detects version drift. Dependency compatibility was checked separately.
All 5,791 per-sample audit records, including source/native/CNN hashes, labels,
splits, and corrected coordinates, match the original audit exactly. The complete
sample-record CSV is byte-identical and the aggregate audit summaries also match.
All 57 tests passed in the new environment. Full replay results and source/lock hashes are recorded in
[reproducibility_verification.json](reproducibility_verification.json).

The original audit reports and frozen JSON snapshot remain unchanged historical
records. This environment evidence is an additional handoff artifact, not a change
to preprocessing or its frozen contract. Teammate reproduction on another machine
remains distinct from this same-host clean-environment validation.
