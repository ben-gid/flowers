---
license: apache-2.0
library_name: pytorch
tags:
- image-classification
- flowers
- computer-vision
metrics:
- accuracy
---
![CI Status](https://github.com/ben-gid/flowers/actions/workflows/ci.yml/badge.svg)
# Flower Classification API 🌸

A PyTorch-based flower classification pipeline and FastAPI service for identifying flower species using the Oxford-102 dataset.

## Project Structure

```text
flowers/
├── data/                 # Dataset & Label files (Oxford-102)
├── src/
│   └── flowers/          # Core package
│       ├── __init__.py
│       ├── api.py        # FastAPI logic (Prediction & Health)
│       ├── main.py       # API Entry point
│       ├── models.py     # SimpleCNN & FlowerDataset definitions
│       ├── train.py      # Training utilities
│       └── utils.py      # Logging & Model loading utilities
├── tests/                # Automated pytest suite
├── flower_model_weights.pth # Best model weights (only in huggingface repo unless trained locally)
├── Dockerfile            # Container configuration
├── pyproject.toml        # Dependency management (uv)
└── README.md
```

## Setup and Installation

### Local Setup
This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the API:
   ```bash
   uv run python src/flowers/main.py
   ```

3. Run Tests:
   ```bash
   uv run pytest
   ```

## Model Weights & Training

### Automatic Weight Loading
The API is designed to be "plug-and-play". If the `flower_model_weights.pth` file is not found locally, the application will automatically download the latest weights from the Hugging Face Hub:
👉 **[bengid/flower-classifier](https://huggingface.co/bengid/flower-classifier)**

### Manual Training
If you wish to train the model yourself (e.g., to experiment with the architecture or hyperparameters), you can run the training script:

```bash
uv run python src/flowers/train.py
```

This script will:
1. Download the Oxford-102 dataset if not found in `data/`.
2. Train the model using CUDA (if available) or CPU.
3. Save the best model weights to `flower_model_weights.pth`.

## Docker Setup

To run the application in a container:

1. Build the image:
   ```bash
   docker build -t flower-api .
   ```

2. Run the container:
   ```bash
   docker run -p 8000:8000 flower-api
   ```

## API Documentation

Once the server is running, you can access:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### Prediction Endpoint
**POST** `/predict`

**Parameters:**
- `file`: Image file (JPG, PNG)

**Example with curl:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@your_flower.jpg'
```

## Development & Quality Control

This project uses a suite of tools to maintain code quality:
- **Ruff**: Extremely fast Python linter and formatter.
- **Pyright**: Static type checker for Python.
- **Pre-commit**: Automatically runs checks before every commit.

### Setup Development Environment
1. Install dev dependencies:
   ```bash
   uv sync --dev
   ```

2. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

### Manual Checks
You can run the quality suite manually at any time:
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files

# Run specific tools
uv run ruff check .
uv run pyright
```

## GitHub / Deployment Notes
- **Large Files**: The model weights (`.pth`) are currently included in the Docker build but excluded from Git via `.gitignore` to avoid repository bloat.
- **Environment Variables**:
  - `MODEL_PATH`: Path to the weights file (default: `./flower_model_weights.pth`).
  - `DATA_ROOT`: Path to the dataset labels directory (default: `./data`).
