# Flower Classification API 🌸

A PyTorch-based flower classification pipeline and FastAPI service for identifying flower species using the Oxford-102 dataset.

## Project Structure

```text
flowers/
├── data/                 # Dataset & Label files (Oxford-102)
├── src/
│   └── flowers/          # Core package
│       ├── __init__.py
│       ├── api.py        # FastAPI endpoints
│       ├── main.py       # API Entry point
│       ├── models.py     # Model architecture & Dataset logic
│       └── train.py      # Model training script
├── tests/                # Automated pytest suite
├── flower_model_weights.pth # Trained model weights (Git Ignored)
├── Dockerfile            # Container configuration
├── pyproject.toml        # Dependency management
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

## Training the Model

The project includes a `train.py` script to train the CNN from scratch. 

> **Important**: The `flower_model_weights.pth` file is required for the API to make predictions. If this file is missing, you must run the training script first:

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

## GitHub / Deployment Notes
- **Large Files**: The model weights (`.pth`) are currently included in the Docker build but excluded from Git via `.gitignore` to avoid repository bloat. 
- **Environment Variables**:
  - `MODEL_PATH`: Path to the weights file (default: `./flower_model_weights.pth`).
  - `DATA_ROOT`: Path to the dataset labels directory (default: `./data`).
