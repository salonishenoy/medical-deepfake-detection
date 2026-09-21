# Contributing

## Workflow

1. Create a short-lived branch from `main`, such as `feature/data-loader` or `docs/midterm-outline`.
2. Make focused commits with clear, imperative messages.
3. Push the branch and open a pull request against `main`.
4. Describe what changed, how it was checked, and any known limitations.
5. Ask at least one teammate to review before merging.
6. Update `PROJECT_TRACKER.md` in the same pull request when a tracked task changes status.

## Repository hygiene

- Never commit datasets, medical scans, `.npy` arrays, credentials, model weights, or large generated artifacts.
- Keep reusable code in `src/`, documentation in `docs/`, and only lightweight result summaries in `results/`.
- Avoid committing machine-specific paths or environment files.
- Preserve isolated training, validation, and test splits when working with dataset code.

## Before opening a pull request

- Confirm that no restricted or sensitive data is included.
- Run the relevant tests or checks when they become available.
- Keep the pull request small enough for a teammate to review.
- Link the change to the applicable item in `PROJECT_TRACKER.md`.

