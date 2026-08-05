# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A flower classification system (Oxford-102 dataset, 102 classes) built on PyTorch/Lightning, served via FastAPI. Two API versions coexist in `api/app/`: `v1` (legacy, self-contained) and `v2` (current, in active development — see "v2 API" below).

## Commands

```bash
uv sync                          # install deps
uv sync --dev                    # install with dev deps

uv run pytest                    # run all tests
uv run pytest tests/test_api.py  # single file
uv run pytest tests/test_api.py::test_name -v   # single test

uv run ruff check .              # lint
uv run ruff check --fix .
uv run pyright                   # typecheck (basic mode, targets src/)
uv run pre-commit run --all-files

uv run python -m api.app.v2.main # run v2 API (port 8000)
uv run train/cli/custom.py train # run a training job
uv run train/gui/app.py          # training dashboard (localhost:8000)
```

CI (`.github/workflows/ci.yml`) runs lint → typecheck → test → docker build, in that order, on push/PR to `main` and `hf-release`. Docker build only packages **v2** (`Dockerfile` copies `api/app/v2` and `src/`, not `v1` or `train/`).

## Architecture

### Training core (`src/`)
Framework-agnostic Lightning code shared by both training and (indirectly, via checkpoint weights) the APIs:
- `src/data.py` — `FlowerDataset` (wraps Oxford-102, downloads via `torchvision.datasets.Flowers102` if missing, parses `imagelabels.mat`), `SubsetWithTransform` (per-split transforms over the same base dataset), `FlowerDataModule` (LightningDataModule).
- `src/classifier.py` — `FlowerClassifier(L.LightningModule)`: wraps an arbitrary pretrained torchvision backbone, swaps its head (`head_name` points at the attribute, e.g. `"classifier"`, `"fc"`, `"heads"`) for a 102-class layer. Logs loss/acc/macro-F1 per stage via `self.log` (logger-agnostic); `test_per_class_f1` is accumulated but never logged here — the caller computes and exports it. Keep MLflow-client calls out of `src/`.
- `src/callbacks.py` — `BackboneFinetuning`: two-stage fine-tuning callback (freeze backbone → unfreeze at `unfreeze_at_epoch` with a lower LR and a separate param group).
- `src/utils.py` — device selection, `get_transforms()` (the canonical 224×224 resize/crop/normalize pipeline, mean/std precomputed from the dataset), class-weight computation for the imbalanced label distribution.

### Training pipeline (`train/`)
- `train/config.py` — `TrainConfig` dataclass, the single source of truth for hyperparameters. Optimizer/scheduler/pretrained-model choices are stored as *name strings* and resolved via registries (`OPTIMIZER_REGISTRY`, `SCHEDULER_REGISTRY`, `PRETRAINED_MODEL_REGISTRY`) — add a model/optimizer by adding a registry entry, not by editing call sites. `resolve()` must be called before the config is usable (materializes the actual classes/instances).
- `train/run_training.py` — `run_training(cfg)`: builds the datamodule, model, callbacks (checkpointing, early stopping, LR monitor), MLflow logger, and Lightning `Trainer`, then fits, runs `trainer.test(ckpt_path="best")`, and exports the per-class F1 vector as the `per_class_f1.json` MLflow artifact. Requires GPU (`accelerator="gpu"` hardcoded).
- `train/cli/custom.py` — argparse CLI over `TrainConfig`, driven by the registries in `config.py`.
- `train/cli/sweep.py` — runs a curated grid of configs across three tiers (A=max accuracy, B=balanced, C=fast).
- `train/gui/app.py` — FastAPI + Jinja dashboard that launches CLI training runs and reads results straight from `mlflow.db` (sqlite).
- Checkpoints land in `models_checkpoints/`, named `{model}-epoch={n}-val_acc={acc}-{run_id}.ckpt`; MLflow tracking DB is `mlflow.db` at repo root, artifacts (`per_class_f1.json`, etc.) in `artifacts/`.

### v1 API (`api/app/v1/flowers/`)
Self-contained legacy module (own `models.py` with `FlowerDataset`/`SimpleCNN` duplicated rather than importing `src/`). Serves both a from-scratch `SimpleCNN` (`/predict/scratch`) and a fine-tuned EfficientNet-B0 (`/predict`) from `app.state`. Not part of the Docker image; kept for reference/comparison.

### v2 API (`api/app/v2/`) — actively evolving, model choice not yet finalized
Restructured into routers/core/utils rather than one flat file:
- `core/config.py` — `AppState` singleton (`state`), loaded/cleared in the FastAPI `lifespan`. Holds `classifier`, `class_names`, `transform`.
- `core/logging_config.py`, `core/middleware.py` — structured logging (overrides uvicorn's default logger, unlike v1) and a request-logging middleware.
- `routers/classify.py` — `POST /classify`; `routers/system.py` — `GET /health`.
- `utils/dependencies.py` — `validate_and_convert_file`: FastAPI dependency doing content-type/size/dimension validation, returns a `ValidatedImage`.
- `utils/helpers.py` — `load_model()` currently hardcodes `torchvision.models.efficientnet_b0` and downloads weights (`ft_EfficientNet-B0.pth`) from the `bengid/flower-classifier` HF Hub repo if not found at `FT_MODEL_PATH`/repo root.
- `utils/model_def.py` — defines a `FlowerClassifier(nn.Module)` wrapper that is currently **unused** by `helpers.load_model()` (which builds the raw `efficientnet_b0` directly). The model architecture for v2 is still being decided — check whether this wrapper is meant to replace the direct-load path before assuming either is canonical.

Env vars: `DATA_ROOT` (default `./data`), `FT_MODEL_PATH` (default `./ft_EfficientNet-B0.pth`), `HOST`/`PORT` (v1 only).

### Tests (`tests/`)
`test_api.py` targets v2 (`api.app.v2.main:app`) exclusively, mocking `AppState` via `unittest.mock.patch` so no GPU/weights are needed. `test_src.py`, `test_train_cli.py`, `test_train_config.py` cover `src/` and `train/` respectively.

## Notes

- The root `README.md` describes an older `src/flowers/` package layout (`src/flowers/api.py`, `models.py`, etc.) that no longer exists — the actual layout is `api/app/v{1,2}/` + `src/` as described above. Treat the README's Project Structure, API, and "Model Weights" sections as stale; verify against the code before relying on either.
- `pyproject.toml` `pythonpath` includes `.`, `src`, and `train`, which is why `train/*.py` files import `config`/`run_training` as top-level modules rather than `train.config`.
