![CI Status](https://github.com/ben-gid/flowers/actions/workflows/ci.yml/badge.svg)

# Flower Classification API 🌸

A flower classification system (Oxford-102, 102 classes) built on PyTorch Lightning and served via FastAPI. The project covers the full ML lifecycle: a custom dataset, a CNN trained from scratch, transfer learning across 11 backbone architectures tracked in MLflow, and a restructured REST API shipped in Docker.

**Best result:** fine-tuned **ViT-B/16** at **0.9796 accuracy / 0.9637 macro-F1** on a held-out test split. **EfficientNetV2-S** (0.9642 / 0.9364) is the serving default — 4x smaller and 3x faster for ~2.7pp of F1.

> ### ⚠️ Read this first: the headline numbers changed
>
> This project previously reported **1.0 validation accuracy** for its best models. That was a data leak, not a result. The numbers above are the corrected, honestly-measured ones. See [The data leak](#the-data-leak) — it's the most instructive thing in this repo.

---

## Table of contents

- [The data leak](#the-data-leak)
- [Results](#results)
- [Project structure](#project-structure)
- [How the project evolved](#how-the-project-evolved)
- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [API](#api)
- [Tests — incomplete, read the caveat](#tests--incomplete-read-the-caveat)
- [CI](#ci)
- [Roadmap](#roadmap)

---

## The data leak

Every Lightning-era model in this repo — 71 MLflow runs — reported validation accuracy between 0.99 and 1.0. That looked like a triumph of full-backbone fine-tuning. It was a bug.

`FlowerDataModule.setup()` in `src/data.py` built all three splits from the same subset:

```python
train_subset, val_subset, test_subset = random_split(full_ds, (0.7, 0.15, 0.15), generator=generator)

self.train_set = SubsetWithTransform(train_subset, self.transform_train)
self.val_set   = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be val_subset
self.test_set  = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be test_subset
```

`random_split` correctly produced three disjoint subsets. Then `val_subset` and `test_subset` were **created and thrown away**, and validation and test both scored the model against the exact images it had trained on. This is total leakage, not partial — a ~1.0 score was the arithmetic consequence of memorization and carried no information about generalization whatsoever.

### What it did and didn't corrupt

**The weights are fine.** Training only ever read `train_subset`, so no held-out image ever contributed to a gradient. Nothing needs to be thrown away.

**The measurements are worthless**, and so is everything downstream of them:

- Reported `val_acc` / `val_f1` on all 71 runs were really *train* accuracy.
- `ModelCheckpoint` selected "best" epochs on that signal — so `models_checkpoints/*.ckpt` filenames advertise a `val_acc` that means nothing, and the saved epoch is unlikely to be the genuinely best one.
- `EarlyStopping` monitored the same signal. Train accuracy rarely degrades, so early stopping had no reason to fire when it should have.
- Every architecture and hyperparameter comparison in the sweep was decided by a metric that couldn't distinguish a good model from an overfit one.

### The fix

One line each for val and test, plus wiring up the `train_split`/`val_split`/`test_split` constructor args that were being ignored in favour of a hardcoded `(0.7, 0.15, 0.15)`. A regression test (`tests/test_src.py::test_splits_are_disjoint`) now asserts the three index sets are pairwise disjoint and sum to the dataset size, so this can't come back silently.

The two published models were re-scored against genuinely held-out data in `notebooks/reevaluate_published_models.ipynb`; those are the numbers in [Results](#results) and in the model cards under `model-cards/`.

### Why it survived so long

Nothing screamed. A 1.0 validation score on a 102-class fine-grained task should have been implausible on its face — a fine-tuned backbone on ~8k images does not perfectly separate 102 flower species — but it arrived looking like the reward for doing transfer learning properly, right after a round of real improvements (full-backbone unfreezing, better architectures) that genuinely did help. The bug also produced no error, no warning, and no test failure: `SubsetWithTransform(train_subset, ...)` is a perfectly valid line of code. The leak was caught by disbelief at the number, not by any tooling.

### Planned: retrain

**These corrected numbers still understate what this recipe can do.** They come from checkpoints selected by a callback reading a leaked metric, so the saved epoch is essentially arbitrary within its run, and no hyperparameter in any of the 71 runs was ever validated against held-out data. A full retrain on the corrected splits — with checkpoint selection and early stopping driven by a real validation signal — is planned, and should beat these figures. The sweep results should be re-run too: the tier comparisons that picked these architectures were themselves decided on leaked metrics.

---

## Results

All figures are **test-split** accuracy, on held-out data, after the split fix:

| Model | Test Acc | Test F1 | Params | Size | Latency (mean) | Notes |
|---|---|---|---|---|---|---|
| SimpleCNN (scratch) | ~0.63 | — | — | — | — | v1 baseline |
| EfficientNet-B0 (v1, partial unfreeze) | >0.93 | — | — | — | — | v1 baseline |
| **EfficientNetV2-S** | **0.9642** | **0.9364** | 20.3M | 81.8 MB | 29.6 ms | **serving default** |
| **ViT-B/16** | **0.9796** | **0.9637** | 85.9M | 343.5 MB | 86.5 ms | best accuracy |

Latency measured on a Ryzen 5600X CPU at batch size 1. Split: 5,733 train / 1,228 val / 1,228 test.

The two v1 models were trained by the older, pre-Lightning pipeline (`api/app/v1/flowers/train_scratch.py`), which split train/val/test **correctly** — the leak was introduced later, in the Lightning migration. Their numbers were never affected and are directly comparable to the corrected v2 figures.

Two things worth noting now that the numbers are honest:

- **Full-backbone unfreezing gained ~3–5 points over v1's partial unfreeze, not the ~7 the leaked metrics implied.** It's still the right call, just a much less dramatic one than a jump from 93% to "100%".
- **ViT's lead over EfficientNetV2-S is real (~2.7pp test F1), where the leaked numbers hid it entirely** by pinning both at ~1.0. EfficientNetV2-S remains the serving default on size and latency grounds, but that's now a deliberate trade rather than a free win.

Macro F1 trails accuracy on both models (~1.6pp for ViT, ~2.8pp for EfficientNetV2-S). Oxford-102 has 40–258 images per class, so that gap is the rare classes underperforming — and it says EfficientNetV2-S's long-tail deficit is the wider of the two.

---

## Project structure

```text
flowers/
├── src/                       # framework-agnostic Lightning core, shared by training + API
│   ├── data.py                # FlowerDataset, SubsetWithTransform, FlowerDataModule
│   ├── classifier.py          # FlowerClassifier(L.LightningModule) — wraps any tv backbone
│   ├── callbacks.py           # BackboneFinetuning — two-stage freeze/unfreeze
│   └── utils.py               # device selection, get_transforms(), class weights
├── train/
│   ├── config.py              # TrainConfig + OPTIMIZER/SCHEDULER/PRETRAINED_MODEL registries
│   ├── run_training.py        # run_training(cfg): datamodule → model → callbacks → Trainer
│   ├── cli/custom.py          # argparse CLI over TrainConfig
│   ├── cli/sweep.py           # curated grid across three tiers (A=accuracy, B=balanced, C=fast)
│   └── gui/app.py             # FastAPI + Jinja dashboard; launches runs, reads mlflow.db
├── api/app/
│   ├── v1/flowers/            # legacy, self-contained; not in the Docker image
│   └── v2/                    # current API
│       ├── core/              # AppState singleton, logging config, request middleware
│       ├── routers/           # classify.py (POST /classify), system.py (GET /health)
│       ├── utils/             # dependencies.py (validation), helpers.py (MODEL_REGISTRY + HF)
│       └── models.py          # PredictionResponse, ValidatedImage
├── model-cards/               # HF model cards for the two published models
├── notebooks/                 # EDA, experiments, publishing, re-evaluation
├── tests/                     # see the caveat below
├── models_checkpoints/        # {model}-epoch={n}-val_acc={acc}-{run_id}.ckpt
├── mlflow.db                  # sqlite tracking DB
└── Dockerfile                 # packages v2 + src only
```

---

## How the project evolved

### 1. v1 — hand-rolled PyTorch

The original pipeline was a self-contained package: a custom `SimpleCNN` trained from scratch (~63%), then an ImageNet-pretrained EfficientNet-B0 fine-tuned in two manual stages with hand-written training loops, `copy.deepcopy` checkpointing, and per-layer-group learning rates set by hand. It only ever unfroze the **last 3 backbone blocks** — the earlier layers never adapted to flower-specific features, capping it above 93%.

It still lives at `api/app/v1/flowers/` for reference. It is not in the Docker image and not covered by tests.

### 2. The move to PyTorch Lightning

The custom training loops became:

- **`FlowerDataModule`** (`src/data.py`) — a `LightningDataModule` owning download, splits, transforms, and dataloaders. (This is also where the leak was introduced. The refactor that made the pipeline cleaner also made it wrong.)
- **`FlowerClassifier`** (`src/classifier.py`) — a `LightningModule` that wraps an *arbitrary* torchvision backbone and swaps its head for a 102-class layer. `head_name` points at the head attribute (`"classifier"`, `"fc"`, `"heads"`), which is what makes one class work across every architecture in the registry. Logs loss, accuracy, and macro-F1 to MLflow.
- **`BackboneFinetuning`** (`src/callbacks.py`) — replaces v1's manual staged unfreeze: the backbone starts frozen (head-only warm-up), then unfreezes at `unfreeze_at_epoch` into its own param group at a lower LR.
- **MLflow** — `MLFlowLogger` for metrics and hyperparameters, sqlite-backed at `mlflow.db`, artifacts in `artifacts/`.

### 3. Model architecture improvements

`train/config.py` turns architecture choice into data. Optimizers, schedulers, and backbones are stored as **name strings** resolved through registries — you add a model by adding a registry entry, not by editing call sites:

```python
# name -> (factory_callable, head_name, backbone_name)
PRETRAINED_MODEL_REGISTRY = {
    "efficientnet_b0": (lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"), "classifier", "features"),
    "vit_b_16":        (lambda: tv_models.vit_b_16(weights="IMAGENET1K_V1"),        "heads",      None),
    ...
}
```

11 backbones are registered: EfficientNet-B0/B1/B2/B3, EfficientNetV2-S, ResNet-50/101, MobileNetV3-Large, VGG16, ConvNeXt-Tiny, ViT-B/16. `train/cli/sweep.py` runs a curated grid across three tiers (A = max accuracy, B = balanced, C = fast), which produced the 71 runs in `mlflow.db`.

Two changes drove the real gains over v1: **unfreezing the entire backbone** rather than the last 3 blocks, and **trying architectures beyond EfficientNet-B0**. Both hold up under the corrected metrics — just less spectacularly than the leaked ones suggested. *(Caveat: the sweep that ranked these was itself scored on leaked metrics. The rankings need re-running.)*

### 4. The v2 API

v1 was one flat module that duplicated `FlowerDataset` and `SimpleCNN` rather than importing `src/`. v2 is restructured:

| Concern | v1 | v2 |
|---|---|---|
| Layout | one flat `api.py` | `core/` + `routers/` + `utils/` |
| Model source | duplicated in `models.py` | imports `src/`, loads from HF Hub |
| Model choice | hardcoded | `MODEL_REGISTRY` + `MODEL_NAME` env var |
| State | module globals | `AppState` singleton, loaded/cleared in `lifespan` |
| Validation | inline in the handler | `validate_and_convert_file` FastAPI dependency |
| Logging | uvicorn default | structured logging + request middleware |
| Endpoints | `/predict`, `/predict/scratch` | `/classify` |

Only v2 is packaged in the Docker image (`Dockerfile` copies `api/app/v2` and `src/`).

---

## Setup

```bash
uv sync                                  # install deps
uv sync --dev                            # with dev deps

uv run python -m api.app.v2.main         # run v2 API (port 8000)
uv run train/cli/custom.py train         # run a training job
uv run train/gui/app.py                  # training dashboard

uv run pytest                            # tests
uv run ruff check .                       # lint
uv run pyright                            # typecheck
```

### Docker

```bash
docker build -t flower-api .
docker run -p 8000:8000 flower-api
```

---

## Dataset

**Oxford-102 Flowers** — 8,189 images, 102 species, downloaded automatically via `torchvision.datasets.Flowers102` if not found locally.

`FlowerDataset` (`src/data.py`) wraps the raw dataset rather than using the torchvision loader directly: it parses the MATLAB `imagelabels.mat` via `scipy.io.loadmat` and converts to 0-indexed labels, lazily loads images with PIL converting to RGB, and composes with standard DataLoader tooling. `SubsetWithTransform` applies different transforms to splits of the same base dataset, which `torch.utils.data.Subset` doesn't support natively.

Class counts range from 40 to 258 images, so `CrossEntropyLoss` is class-weighted (`get_class_weights` in `src/utils.py`). This imbalance is why **macro-F1 is the metric to trust here, not accuracy**.

### Transforms

Images arrive at inconsistent sizes; `get_transforms()` (`src/utils.py`) is the canonical pipeline:

```python
# Uniformize: resize → center-crop → tensor
uniformize = Compose([Resize((256, 256)), CenterCrop((224, 224)), ToTensor()])

# Train: uniformize + augmentation + normalize (mean/std precomputed over the dataset)
train_transform = Compose([uniformize, RandomHorizontalFlip(0.5), RandomVerticalFlip(0.5),
                           RandomRotation(15), Normalize(mean, std)])

# Val/Test: uniformize + normalize only
val_transform = Compose([uniformize, Normalize(mean, std)])
```

---

## Training

`TrainConfig` (`train/config.py`) is the single source of truth for hyperparameters. `resolve()` must be called before use — it materializes registry name-strings into actual classes.

```bash
uv run train/cli/custom.py train --pretrained-model vit_b_16 --max-epochs 50
uv run train/cli/sweep.py --tier B
uv run train/gui/app.py                  # dashboard: launch runs, browse mlflow.db
```

`run_training(cfg)` builds the datamodule, model, callbacks (checkpointing, early stopping, LR monitor), MLflow logger, and Lightning `Trainer`, then fits. **Requires a GPU** (`accelerator="gpu"` is hardcoded).

Checkpoints land in `models_checkpoints/` as `{model}-epoch={n}-val_acc={acc}-{run_id}.ckpt`. **The `val_acc` in existing filenames is a leaked metric — ignore it.**

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Classify an uploaded image |
| `/health` | GET | Reports loaded model / transform status |

Model weights are pulled from HF Hub on startup (cached by `hf_hub_download`) and held in the `AppState` singleton for the process lifetime.

### Validation

Enforced by the `validate_and_convert_file` dependency (`api/app/v2/utils/dependencies.py`):

- **File type**: JPEG / PNG only → 400
- **File size**: ≤ 5 MB → 413
- **Dimensions**: ≥ 224×224 → 400

### Example

```bash
curl -X POST 'http://localhost:8000/classify' \
  -H 'accept: application/json' \
  -F 'file=@rose.jpg'
```

```json
{
  "filename": "rose.jpg",
  "content_type": "image/jpeg",
  "prediction": "rose",
  "confidence": 0.97
}
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `./data` | Dataset root (needs `Oxford-102_Flower_dataset_labels.txt` for class names) |
| `MODEL_NAME` | `efficientnet_v2_s` | Architecture to serve; one of `MODEL_REGISTRY` (`efficientnet_v2_s`, `vit_b_16`) |

---

## Tests — incomplete, read the caveat

> **The test suite is not trustworthy yet, and I know it.**
>
> Most of `tests/` was generated by Claude and **I have not reviewed it line by line**. It has not been audited for whether the assertions are meaningful, whether the mocks reflect real behaviour, or whether passing actually implies working. Treat a green run as weak evidence. **Reviewing and completing this suite is on the roadmap and I plan to fix it properly.**
>
> This isn't hypothetical. `tests/test_helpers.py` was passing against an API that no longer existed — it set an `FT_MODEL_PATH` env var `load_model()` had stopped reading, and saved fixtures with `torch.save` when the loader had moved to safetensors. It only surfaced when the registry gained a field and the tuple unpack broke. That's a test that would have kept "passing" while asserting nothing about the real code path.
>
> More to the point: **the test suite did not catch the data leak.** Nothing in `tests/` checked that the splits were disjoint until after the bug was found by hand.

| File | Covers |
|---|---|
| `test_api.py` | v2 API — lifespan, `/classify`, `/health`, size/dimension/MIME rejection. Mocks `AppState`, so no GPU or weights needed. |
| `test_src.py` | `src/utils.py` transforms and class weights, plus `test_splits_are_disjoint` — the leak regression test. |
| `test_helpers.py` | v2 `load_model()` architecture selection and unknown-name rejection. |
| `test_train_cli.py`, `test_train_config.py` | `train/` CLI parsing and `TrainConfig` resolution. |

Known state: **33 passing**, but `ruff check .` currently reports 22 errors repo-wide (mostly import ordering and line length in the generated test files), so **CI is red on the lint gate**. v1 has no test coverage at all.

---

## CI

`.github/workflows/ci.yml`, on push/PR to `main` and `hf-release`:

```
Lint (Ruff) ─┐
             ├─→ Run Tests (Pytest) ─→ Verify Docker Build
Typecheck ───┘   (Pyright, basic mode, targets src/)
```

Lint and typecheck run in parallel; tests need both; the Docker build needs tests.

---

## Roadmap

### Correctness (current priority)
- [x] Fix the train/val/test leak in `FlowerDataModule.setup()`
- [x] Add a regression test asserting splits are disjoint
- [x] Re-evaluate both published models on honest splits
- [x] Correct the published model cards
- [ ] **Retrain on corrected splits** — real checkpoint selection and early stopping
- [ ] Re-run the sweep; the tier rankings were decided on leaked metrics
- [ ] **Review the Claude-generated test suite properly** and fix the lint gate
- [ ] Export per-class F1 (computed in `src/classifier.py`, not currently surfaced)

### Model experiments
- [x] Migrate training pipeline to PyTorch Lightning
- [x] Use MLflow for tracking and model documentation
- [x] Fully unfreeze the backbone
- [x] Experiment with other architectures (11 registered)
- [x] Add model cards
- [ ] Stratified splitting — 102 imbalanced classes, a random 15% may underrepresent rare ones

### API
- [x] Restructure into routers / core / utils (v2)
- [x] Make the served architecture configurable (`MODEL_NAME`)
- [ ] Rate limiting (e.g. `slowapi`) and authentication
- [ ] Deploy

### Other
- [x] Add a CLI
- [x] Add a training GUI

---

## Model weights & HuggingFace Hub

Published models — the v2 API auto-downloads from these:

- **[bengid/efficientnetv2-s-flower-classifier](https://huggingface.co/bengid/efficientnetv2-s-flower-classifier)** — serving default
- **[bengid/vit-flower-classifier](https://huggingface.co/bengid/vit-flower-classifier)** — best accuracy

Full model cards under `model-cards/`. Legacy v1 weights remain at [bengid/flower-classifier](https://huggingface.co/bengid/flower-classifier).

---

## License

Apache 2.0.
