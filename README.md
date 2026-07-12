---
license: apache-2.0
library_name: pytorch
tags:
- image-classification
- flowers
- computer-vision
- transfer-learning
- efficientnet
metrics:
- accuracy
---
![CI Status](https://github.com/ben-gid/flowers/actions/workflows/ci.yml/badge.svg)

# Flower Classification API 🌸

A production-ready flower classification system built on PyTorch and served via FastAPI. The project covers the full ML lifecycle: building a custom dataset, training a CNN from scratch, applying transfer learning to close a 30-point accuracy gap, and shipping both models behind a tested REST API.

**Best result:** fine-tuned **EfficientNet-B0** achieves **>93% accuracy** on 102 flower classes, up from **63%** with a custom SimpleCNN trained from scratch.

---

## Project Structure

```text
flowers/
├── data/
│   ├── Oxford-102_Flower_dataset_labels.txt   # Human-readable class names
│   └── flowers-102/                           # Raw Oxford-102 images & label .mat files
├── model_weights/
│   ├── flower_model_weights.pth               # SimpleCNN (trained from scratch)
│   └── ft_EfficientNet-B0.pth                 # Fine-tuned EfficientNet-B0
├── notbooks/
│   ├── flowers.ipynb                          # EDA, dataset stats, prototype training
│   └── fine_tune.ipynb                        # Fine-tuning experiments & analysis
├── src/
│   └── flowers/
│       ├── __init__.py
│       ├── api.py                             # FastAPI app (lifespan, /predict, /predict/scratch, /health)
│       ├── main.py                            # Uvicorn entry point
│       ├── models.py                          # FlowerDataset, SubsetWithTransform, CNNBlock, SimpleCNN
│       ├── fine_tune.py                       # EfficientNet-B0 fine-tuning pipeline
│       ├── train_scratch.py                   # SimpleCNN train-from-scratch pipeline
│       ├── training_utils.py                  # Shared utilities: transforms, training loop, class weights, freeze helpers
│       └── utils.py                           # Logging, model loading, HuggingFace Hub integration
├── tests/
│   └── test_api.py                            # Pytest suite covering lifespan, /predict, /predict/scratch, edge cases
├── .github/workflows/ci.yml                   # GitHub Actions CI (lint → typecheck → test → docker build)
├── .pre-commit-config.yaml                    # Pre-commit hooks (Ruff, Pyright)
├── Dockerfile
├── pyproject.toml                             # uv dependency management
└── README.md
```

---

## Setup and Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management.

```bash
# Install all dependencies
uv sync

# Run the API
uv run python src/flowers/main.py

# Run the test suite
uv run pytest
```

### Docker

```bash
docker build -t flower-api .
docker run -p 8000:8000 flower-api
```

---

## Roadmap
### Model experiments
- [x] Migrate training pipeline to PyTorch Lightning
  - [x] Refactor dataset logic into a reusable `LightningDataModule`.
  - [x] Replace custom training loops with `LightningModule` and the built-in `Trainer`.
  - [x] Use `MLFlowLogger` for seamless metric and hyperparameter tracking.
  - [x] Utilize built-in fine-tuning callbacks for stage-based unfreezing.
- [x] Use MLflow for model documentation and tracking experiment results
  - [x] Set up MLflow as a Model Registry to version weights and manage model lifecycle.
- [x] Update model to fully unfreeze the backbone
- [x] Experiment with other model architectures
- [ ] Add model card(s)

### API improvement
- [ ] Rearrange the API to make it scalable (separate routers, environment variables, etc.)
- [ ] Implement API rate limiting and security
  - [ ] Integrate a rate limiter (e.g., `slowapi`) to prevent abuse and manage load.
  - [ ] Add robust authentication (API keys or OAuth2).
### Other
- [ ] Add a CLI
- [ ] Deploy the API

---

## Dataset

The project uses the **Oxford 102 Flower** dataset, consisting of 8,189 images across 102 flower categories.

A **custom `FlowerDataset`** class (`src/flowers/models.py`) was built from scratch to wrap the raw dataset rather than relying on out-of-the-box `torchvision` loaders. It:

- Parses the `.mat` label file from MATLAB format via `scipy.io.loadmat` and converts labels to 0-indexed
- Lazily loads images from disk using PIL, converting each to RGB regardless of the source format
- Automatically downloads the Oxford-102 data via `torchvision.datasets.Flowers102` if not found locally
- Integrates cleanly with `torch.utils.data.Dataset` so it composes with standard DataLoader tooling

A companion **`SubsetWithTransform`** class makes it possible to apply different augmentation pipelines to train/val/test splits of the same base dataset, since `torch.utils.data.Subset` does not natively support per-split transforms.

### Custom Transforms

Oxford-102 images come in inconsistent sizes and aspect ratios. A **custom transform pipeline** standardizes them to a fixed `224×224` input before either augmentation or inference:

```python
# Uniformize: resize → center-crop → tensor
uniformize = Compose([Resize((256, 256)), CenterCrop((224, 224)), ToTensor()])

# Training: uniformize + augmentation + normalize (dataset mean/std pre-calculated in notebook)
train_transform = Compose([uniformize, RandomHorizontalFlip(0.5),
                           RandomVerticalFlip(0.5), RandomRotation(15),
                           Normalize(mean, std)])

# Val/Test: uniformize + normalize only (no augmentation)
val_transform = Compose([uniformize, Normalize(mean, std)])
```

Mean and standard deviation were calculated over the full dataset in `flowers.ipynb`.

---

## Model Pipelines

### Training from Scratch — SimpleCNN

A custom **SimpleCNN** was designed and trained end-to-end on the Oxford-102 dataset without any pre-trained weights.

**Architecture:** 6 stacked `CNNBlock` modules (Conv2d → BatchNorm2d → ReLU → MaxPool2d) growing from 32 → 1024 channels, followed by a 4-layer fully-connected classifier with Dropout (p=0.5).

**Training pipeline (`train_scratch.py`):**

1. Load `FlowerDataset` and split 70/15/15 into train/val/test using `random_split`
2. Apply train and val transforms via `SubsetWithTransform`
3. Initialize `SimpleCNN` for 102 classes with `torch.Size([3, 224, 224])` input
4. Train with **Adam** optimizer (weight decay 1e-4), **CrossEntropyLoss**, and a **CosineAnnealingLR** scheduler over 50 epochs (η_min = 2e-4)
5. Track best validation accuracy per epoch with `copy.deepcopy` checkpointing
6. Evaluate on held-out test set; save weights to `model_weights/flower_model_weights.pth`

**Result: ~63% test accuracy**

---

### Fine-Tuning EfficientNet-B0

Transfer learning was applied to close the accuracy gap by adapting an ImageNet-pretrained **EfficientNet-B0** to the flower domain in two stages.

**Fine-tuning pipeline (`fine_tune.py`):**

**Stage 1 — Classifier-only warm-up (20 epochs):**
1. Load `EfficientNet-B0` with `weights='IMAGENET1K_V1'`
2. Freeze all backbone parameters; swap the final linear layer for a 102-class output head
3. Compute **class-balanced sample weights** with `sklearn.utils.compute_class_weight` to handle Oxford-102's uneven class distribution
4. Train only the classifier with **Adam** (lr = 1e-4, weight decay 5e-4), weighted **CrossEntropyLoss**, and **CosineAnnealingLR** (T_max=10)

**Stage 2 — Partial unfreeze (20 epochs):**
1. Selectively unfreeze the last 3 feature blocks of the EfficientNet backbone
2. Apply **differential learning rates** per layer group to preserve low-level features while adapting high-level ones:
   - `features[6]`: lr = 1e-5
   - `features[7]`: lr = 1e-5
   - `features[8]`: lr = 1e-4
   - `classifier`: lr = 1e-3
3. Continue training with weighted CrossEntropyLoss and CosineAnnealingLR
4. Save best checkpoint to `model_weights/ft_EfficientNet-B0.pth`

**Result: >93% test accuracy (+30 points over SimpleCNN)**

---

## API

Both models are served simultaneously via FastAPI with a shared lifespan context that loads and tears down models cleanly at startup and shutdown.

| Endpoint | Model | Description |
|---|---|---|
| `POST /predict` | EfficientNet-B0 (fine-tuned) | Primary prediction endpoint |
| `POST /predict/scratch` | SimpleCNN | Prediction using the scratch-trained model |
| `GET /health` | — | Reports loaded model and transform status |

### Request Validation

The API enforces:
- **File type**: JPEG or PNG only (HTTP 400)
- **File size**: ≤ 5 MB (HTTP 413)
- **Image dimensions**: minimum 224×224 pixels (HTTP 400)

### Example

```bash
curl -X POST 'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@rose.jpg'
```

**Response:**
```json
{
  "filename": "rose.jpg",
  "content_type": "image/jpeg",
  "prediction": "rose",
  "confidence": 0.97
}
```

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## CI Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request to `main` and `hf-release`, enforcing a strict quality gate before any code merges:

```
┌─────────────────┐     ┌──────────────────────┐
│  Lint (Ruff)    │──┐  │ Type Check (Pyright) │──┐
└─────────────────┘  │  └──────────────────────┘  │
                     ▼                             ▼
              ┌──────────────────────────────────────┐
              │         Run Tests (Pytest)            │
              └──────────────────────────┬───────────┘
                                         ▼
                              ┌─────────────────────┐
                              │  Docker Build Verify │
                              └─────────────────────┘
```

| Stage | Tool | Notes |
|---|---|---|
| Lint | Ruff | Fast Python linter & formatter |
| Type Check | Pyright | Static type analysis (`basic` mode) |
| Tests | Pytest | Must pass before Docker stage runs |
| Docker Build | Docker | Validates `Dockerfile` builds cleanly |

The test suite (`tests/test_api.py`) covers:
- Lifespan state initialization and teardown
- `/predict` and `/predict/scratch` with valid input
- File-size rejection (> 5 MB)
- Minimum dimension rejection (< 224px)
- Invalid MIME type rejection

Model loading is mocked via `unittest.mock.patch` so tests run without GPU or model weight files in CI.

---

## Development & Code Quality

```bash
# Install dev dependencies
uv sync --dev

# Install pre-commit hooks (runs on every commit)
uv run pre-commit install

# Run the full quality suite manually
uv run pre-commit run --all-files
uv run ruff check .
uv run pyright
```

---

## Model Weights & HuggingFace Hub

The API auto-downloads weights from HuggingFace Hub if not found locally:

👉 **[bengid/flower-classifier](https://huggingface.co/bengid/flower-classifier)**

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `./model_weights/flower_model_weights.pth` | Path to SimpleCNN weights |
| `DATA_ROOT` | `./data` | Path to dataset root |
