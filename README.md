![CI Status](https://github.com/ben-gid/flowers/actions/workflows/ci.yml/badge.svg)

# Flower Classification API 🌸

Oxford-102 flower classification (102 classes) on PyTorch Lightning, served via FastAPI. Full lifecycle: custom dataset, a CNN from scratch, transfer learning across 11 backbones tracked in MLflow, and a REST API shipped in Docker.

**Best result:** fine-tuned **ViT-B/16** — 0.9837 accuracy / 0.9722 macro-F1 on a held-out test split. **ConvNeXt-Tiny** (0.9821 / 0.9694) is the serving default: a third of the weights for 0.3pp of F1.

> ⚠️ This project previously reported **1.0 validation accuracy**. That was a data leak, not a result. See [The data leak](#the-data-leak) — it's the most instructive thing in this repo.

---

## Results

Test-split metrics on genuinely held-out data, after the split fix and retrain:

| Model | Test Acc | Test F1 | Params | Size | Latency | Notes |
|---|---|---|---|---|---|---|
| SimpleCNN (scratch) | ~0.63 | — | — | — | — | v1 baseline |
| EfficientNet-B0 (v1, partial unfreeze) | >0.93 | — | — | — | — | v1 baseline |
| EfficientNetV2-S | 0.9691 | 0.9433 | 20.3M | 81.8 MB | 11.3 ms | smallest weights |
| **ConvNeXt-Tiny** | **0.9821** | **0.9694** | 27.9M | 111.6 MB | 3.5 ms | **serving default** |
| **ViT-B/16** | **0.9837** | **0.9722** | 85.9M | 343.5 MB | 5.4 ms | best accuracy |

Latency is mean over batch size 1 on an RTX 5070 and does **not** transfer to CPU (on CPU, ViT runs ~3x the other two). Split: 5,733 train / 1,228 val / 1,228 test.

- **Full-backbone unfreezing gained ~3–5 points over v1's partial unfreeze**, not the ~7 the leaked metrics implied. Still the right call, less dramatic.
- **EfficientNetV2-S is the smallest and the slowest.** Its long chain of small depthwise blocks leaves the GPU launch-bound at batch size 1. Don't infer latency from parameter count.
- **Macro-F1 trails accuracy on all three** (~1.2pp ViT, ~1.3pp ConvNeXt, ~2.6pp EfficientNetV2-S). Class counts run 40–258, so that gap is the rare classes — and EfficientNetV2-S's long-tail deficit is the widest.

The v1 models came from the older pre-Lightning pipeline (`api/app/v1/flowers/train_scratch.py`), which split correctly. Their numbers were never affected and are directly comparable.

---

## The data leak

Every Lightning-era model in this repo — 71 MLflow runs — reported validation accuracy between 0.99 and 1.0. That looked like a triumph of full-backbone fine-tuning. It was a bug in `FlowerDataModule.setup()`:

```python
train_subset, val_subset, test_subset = random_split(full_ds, (0.7, 0.15, 0.15), generator=generator)

self.train_set = SubsetWithTransform(train_subset, self.transform_train)
self.val_set   = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be val_subset
self.test_set  = SubsetWithTransform(train_subset, self.transform_test)  # bug — should be test_subset
```

`random_split` correctly produced three disjoint subsets; `val_subset` and `test_subset` were then **created and thrown away**. Validation and test both scored the model against the exact images it trained on — total leakage, so ~1.0 was the arithmetic consequence of memorization and carried no information about generalization.

**The weights were never contaminated.** Training only ever read `train_subset`, so no held-out image reached a gradient. What the bug corrupted was *measurement* — and through it, model selection: `ModelCheckpoint` saved on a leaked signal, `EarlyStopping` monitored a metric that rarely degrades so it had no reason to fire, and every architecture comparison in the sweep was decided by a number that couldn't tell a good model from an overfit one.

**The fix:** one line each for val and test, plus wiring up the `train_split`/`val_split`/`test_split` args that were being ignored for a hardcoded `(0.7, 0.15, 0.15)`. `tests/test_src.py::test_splits_are_disjoint` now asserts the three index sets are pairwise disjoint and sum to the dataset size, so it can't come back silently.

**Why it survived:** nothing screamed. A 1.0 score on a 102-class fine-grained task should have been implausible on its face, but it arrived right after a round of changes (full unfreezing, better architectures) that genuinely did help. No error, no warning, no test failure — `SubsetWithTransform(train_subset, ...)` is a perfectly valid line. It was caught by disbelief at the number, not by tooling.

**What the retrain recovered:** less than expected. Re-scored honestly on the same held-out data, the leak-era ViT checkpoint hit 0.9637 test macro-F1 against the retrained model's 0.9722. The bug inflated the *reported* numbers substantially; it cost the *weights* comparatively little.

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
│   └── v2/                    # current API: core/ (AppState, logging, middleware),
│                              # routers/ (classify, system), utils/ (validation, MODEL_REGISTRY + HF)
├── model-cards/               # HF model cards for the three published models
├── notebooks/                 # EDA, experiments, publishing, re-evaluation
├── models_checkpoints/        # {model}-epoch={n}-val_acc={acc}-{run_id}.ckpt
├── mlflow.db                  # sqlite tracking DB
└── Dockerfile                 # packages v2 + src only
```

---

## How it evolved

**1. v1 — hand-rolled PyTorch.** A custom `SimpleCNN` from scratch (~63%), then an ImageNet-pretrained EfficientNet-B0 fine-tuned in two manual stages with hand-written loops, `copy.deepcopy` checkpointing, and per-layer-group LRs set by hand. It only ever unfroze the **last 3 backbone blocks**, capping it above 93%. Still at `api/app/v1/flowers/` for reference; not in Docker, not tested.

**2. PyTorch Lightning.** `FlowerDataModule` owns download/splits/transforms/dataloaders (and is where the leak was introduced — the refactor that made the pipeline cleaner also made it wrong). `FlowerClassifier` wraps an *arbitrary* torchvision backbone and swaps its head for a 102-class layer; `head_name` points at the head attribute (`"classifier"`, `"fc"`, `"heads"`), which is what makes one class work across every architecture. `BackboneFinetuning` replaces v1's manual staged unfreeze. MLflow logs metrics and hyperparameters to `mlflow.db`.

**3. Architecture as data.** Optimizers, schedulers, and backbones are name strings resolved through registries in `train/config.py` — add a model by adding a registry entry, not by editing call sites:

```python
# name -> (factory_callable, head_name, backbone_name)
PRETRAINED_MODEL_REGISTRY = {
    "efficientnet_b0": (lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"), "classifier", "features"),
    "vit_b_16":        (lambda: tv_models.vit_b_16(weights="IMAGENET1K_V1"),        "heads",      None),
    ...
}
```

11 backbones registered: EfficientNet-B0/B1/B2/B3, EfficientNetV2-S, ResNet-50/101, MobileNetV3-Large, VGG16, ConvNeXt-Tiny, ViT-B/16. Two changes drove the real gains over v1: **unfreezing the entire backbone** and **trying architectures beyond EfficientNet-B0**.

**4. The v2 API.** v1 was one flat module duplicating `FlowerDataset`/`SimpleCNN` instead of importing `src/`:

| Concern | v1 | v2 |
|---|---|---|
| Layout | one flat `api.py` | `core/` + `routers/` + `utils/` |
| Model source | duplicated in `models.py` | imports `src/`, loads from HF Hub |
| Model choice | hardcoded | `MODEL_REGISTRY` + `MODEL_NAME` env var |
| State | module globals | `AppState` singleton, loaded/cleared in `lifespan` |
| Validation | inline in the handler | `validate_and_convert_file` dependency |
| Logging | uvicorn default | structured logging + request middleware |
| Endpoints | `/predict`, `/predict/scratch` | `/classify` |

---

## Setup

```bash
uv sync --dev                            # install deps

uv run python -m api.app.v2.main         # run v2 API (port 8000)
uv run train/cli/custom.py train         # run a training job
uv run train/gui/app.py                  # training dashboard

uv run pytest                            # tests
uv run ruff check .                      # lint
uv run pyright                           # typecheck

docker build -t flower-api . && docker run -p 8000:8000 flower-api
```

---

## Dataset

**Oxford-102 Flowers** — 8,189 images, 102 species, downloaded automatically via `torchvision.datasets.Flowers102` if not found locally.

`FlowerDataset` (`src/data.py`) wraps the raw dataset rather than using the torchvision loader: it parses `imagelabels.mat` via `scipy.io.loadmat` into 0-indexed labels and lazily loads images with PIL. `SubsetWithTransform` applies different transforms to splits of the same base dataset, which `torch.utils.data.Subset` doesn't support natively.

Class counts range from 40 to 258, so `CrossEntropyLoss` is class-weighted (`get_class_weights`). That imbalance is why **macro-F1 is the metric to trust here, not accuracy**.

`get_transforms()` (`src/utils.py`) is the canonical pipeline — resize to a fixed 256×256 (aspect ratio *not* preserved), center-crop to 224, tensor, then augmentation on train only, then normalize with dataset-precomputed mean/std.

---

## Training

`TrainConfig` (`train/config.py`) is the single source of truth for hyperparameters. `resolve()` must be called before use — it materializes registry name-strings into actual classes.

```bash
uv run train/cli/custom.py train --pretrained-model vit_b_16 --max-epochs 50
uv run train/cli/sweep.py --tier B
uv run train/gui/app.py                  # dashboard: launch runs, browse mlflow.db
```

`run_training(cfg)` builds the datamodule, model, callbacks (checkpointing, early stopping, LR monitor), MLflow logger, and Lightning `Trainer`, then fits. **Requires a GPU** (`accelerator="gpu"` is hardcoded).

Checkpoints land in `models_checkpoints/` as `{model}-epoch={n}-val_acc={acc}-{run_id}.ckpt`. **The `val_acc` in pre-fix filenames is a leaked metric — ignore it.**

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/classify` | POST | Classify an uploaded image |
| `/health` | GET | Reports loaded model / transform status |

Weights are pulled from HF Hub on startup (cached by `hf_hub_download`) and held in the `AppState` singleton for the process lifetime. Validation is enforced by the `validate_and_convert_file` dependency: JPEG/PNG only (400), ≤ 5 MB (413), ≥ 224×224 (400).

```bash
curl -X POST 'http://localhost:8000/classify' -F 'file=@rose.jpg'
```

```json
{"filename": "rose.jpg", "content_type": "image/jpeg", "prediction": "rose", "confidence": 0.97}
```

Interactive docs at [http://localhost:8000/docs](http://localhost:8000/docs).

| Env var | Default | Description |
|---|---|---|
| `DATA_ROOT` | `./data` | Dataset root (needs `Oxford-102_Flower_dataset_labels.txt` for class names) |
| `MODEL_NAME` | `convnext_tiny` | One of `MODEL_REGISTRY`: `convnext_tiny`, `efficientnet_v2_s`, `vit_b_16` |

---

## Tests — read the caveat

> **The test suite is not trustworthy yet, and I know it.** Most of `tests/` was generated by Claude and **I have not reviewed it line by line** — not for whether the assertions are meaningful, whether the mocks reflect real behaviour, or whether passing implies working. Treat a green run as weak evidence.
>
> This isn't hypothetical. `tests/test_helpers.py` was passing against an API that no longer existed — setting an `FT_MODEL_PATH` env var `load_model()` had stopped reading, saving fixtures with `torch.save` after the loader moved to safetensors. It only surfaced when the registry gained a field and the tuple unpack broke.
>
> More to the point: **the suite did not catch the data leak.** Nothing checked that the splits were disjoint until after the bug was found by hand.

| File | Covers |
|---|---|
| `test_api.py` | v2 API — lifespan, `/classify`, `/health`, size/dimension/MIME rejection. Mocks `AppState`, so no GPU or weights needed. |
| `test_src.py` | `src/utils.py` transforms and class weights, plus `test_splits_are_disjoint` — the leak regression test. |
| `test_helpers.py` | v2 `load_model()` architecture selection and unknown-name rejection. |
| `test_train_cli.py`, `test_train_config.py` | `train/` CLI parsing and `TrainConfig` resolution. |

Current state: **34 passing**, lint and typecheck clean. v1 has no coverage at all.

---

## CI

`.github/workflows/ci.yml`, on push/PR to `main` and `hf-release`: lint (Ruff) and typecheck (Pyright, basic mode, targets `src/`) run in parallel → tests (Pytest) → Docker build.

---

## Roadmap

**Correctness**
- [x] Fix the train/val/test leak, add a disjoint-splits regression test
- [x] Re-evaluate published models on honest splits, correct the model cards
- [x] Retrain on corrected splits with real checkpoint selection and early stopping
- [x] Re-run the sweep — the tier rankings had been decided on leaked metrics
- [ ] **Review the Claude-generated test suite properly**
- [ ] Export per-class F1 (computed in `src/classifier.py`, not surfaced)

**Model**
- [x] Lightning migration, MLflow tracking, full-backbone unfreeze, 11 architectures, model cards
- [ ] Stratified splitting — with 102 imbalanced classes, a random 15% may underrepresent rare ones

**API**
- [x] Restructure into routers / core / utils (v2), configurable architecture via `MODEL_NAME`
- [ ] Rate limiting (e.g. `slowapi`) and authentication
- [ ] Deploy

---

## Model weights

Published to HF Hub; the v2 API auto-downloads from these. Full cards under `model-cards/`.

- **[bengid/convnext-tiny-flower-classifier](https://huggingface.co/bengid/convnext-tiny-flower-classifier)** — serving default
- **[bengid/vit-flower-classifier](https://huggingface.co/bengid/vit-flower-classifier)** — best accuracy
- **[bengid/efficientnetv2-s-flower-classifier](https://huggingface.co/bengid/efficientnetv2-s-flower-classifier)** — smallest weights

Legacy v1 weights remain at [bengid/flower-classifier](https://huggingface.co/bengid/flower-classifier).

## License

Apache 2.0.
