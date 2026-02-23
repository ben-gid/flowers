# Flower Classification API 🌸

A PyTorch-based flower classification pipeline and FastAPI service.

## Project Structure
- `api.py`: FastAPI service for model inference.
- `models.py`: CNN model architecture and Dataset definitions.
- `train.py`: Training script and utilities.
- `flower_model_weights.pth`: Trained model weights.
- `data/`: Oxford-102 Flower dataset and label files.
- `tests/`: Automated tests for API and lifespan logic.
- `Dockerfile`: Containerization setup using `uv`.

## Setup and Installation

### Local Setup
This project uses [uv](https://github.com/astral-sh/uv) for lightning-fast dependency management.

1. Install dependencies:
   ```bash
   uv sync
   ```

2. Run the API:
   ```bash
   uv run uvicorn api:app --reload
   ```

3. Run Tests:
   ```bash
   uv run pytest
   ```

### Docker Setup
To containerize the application:

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
- **Large Files**: The model weights (`.pth`) are currently included in the Docker build but excluded from Git via `.gitignore` to avoid repository bloat. For production, consider using Git LFS or hosting weights on an external bucket (S3/GCS).
- **Environment Variables**:
  - `MODEL_PATH`: Path to the weights file.
  - `DATA_ROOT`: Path to the dataset labels directory.
