import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from flowers.api import app

# Add the root and src directories to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))


@pytest.fixture
def mock_api_startup():
    """Mock out model loading for all tests using TestClient lifespan."""
    with (
        patch("flowers.api.load_model") as mock_load_model,
        patch("flowers.api.load_class_names") as mock_load_classes,
        patch("flowers.api.get_transforms") as mock_transforms,
        patch("builtins.open", MagicMock()),
    ):
        # Setup mocks to return dummy data
        mock_model = MagicMock()
        mock_model.parameters.return_value = iter([MagicMock(device="cpu")])
        mock_model.return_value = torch.tensor(
            [[10.0, 0.0, 0.0]]
        )  # high score for class 0

        mock_load_model.return_value = mock_model
        mock_load_classes.return_value = ["rose", "daisy", "tulip"]
        mock_transforms.return_value = (None, lambda x: torch.zeros(3, 224, 224))
        yield


def test_lifespan_state_initialization(mock_api_startup):
    """Test that startup correctly populates app.state and shutdown clears it."""
    with TestClient(app):
        # Check if startup populated the state
        assert hasattr(app.state, "model")
        assert hasattr(app.state, "class_names")
        assert hasattr(app.state, "transform")

    # After exiting the context (shutdown), check if state is cleared
    assert not hasattr(app.state, "model")
    assert not hasattr(app.state, "class_names")


def create_mock_image(width=224, height=224, fmt="JPEG"):
    """Create a mock image as bytes for testing /predict endpoint."""
    file = io.BytesIO()
    image = Image.new("RGB", (width, height), color="red")
    image.save(file, format=fmt)
    file.seek(0)
    return file.read()


def test_health(mock_api_startup):
    """Test the /health endpoint."""
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


def test_predict_success(mock_api_startup):
    """Test the /predict endpoint with a valid image."""
    img_bytes = create_mock_image(224, 224)
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prediction"] == "rose"
        assert "confidence" in data


def test_predict_too_small(mock_api_startup):
    """Test the /predict endpoint with an image that is too small."""
    img_bytes = create_mock_image(100, 100)
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", files={"file": ("small.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Image too small" in response.json()["detail"]


def test_predict_too_large(mock_api_startup):
    """Test the /predict endpoint with an image that is too large."""
    # 6MB dummy data
    large_data = b"0" * (6 * 1024 * 1024)
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", files={"file": ("large.jpg", large_data, "image/jpeg")}
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]


def test_predict_invalid_type(mock_api_startup):
    """Test the /predict endpoint with an invalid file type."""
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict",
            files={"file": ("not_an_image.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
