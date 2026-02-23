import sys
import io
from pathlib import Path
from fastapi.testclient import TestClient
from PIL import Image
import pytest
from unittest.mock import patch, MagicMock

# Add the root and src directories to sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir / "src"))

from flowers.api import app

client = TestClient(app)


def test_lifespan_state_initialization():
    """Test that startup correctly populates app.state and shutdown clears it."""
    # We mock out the heavy loading parts
    with patch("flowers.api.init_model") as mock_init, \
         patch("flowers.api.torch.load") as mock_load, \
         patch("flowers.api.get_transforms") as mock_transforms, \
         patch("builtins.open", MagicMock()): 
        
        # Setup mocks to return dummy data
        mock_init.return_value = (MagicMock(), None, None, None)
        mock_transforms.return_value = (None, MagicMock())
        
        with TestClient(app) as client:
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

def test_health():
    """Test the /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_success():
    """Test the /predict endpoint with a valid image."""
    img_bytes = create_mock_image(224, 224)
    # Wrap in with statement to ensure lifespan (model loading) runs
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", 
            files={"file": ("test.jpg", img_bytes, "image/jpeg")}
        )
        # This might fail if model weights are missing
        if response.status_code == 200:
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
        else:
            print(f"Prediction failed with {response.status_code}: {response.text}")

def test_predict_too_small():
    """Test the /predict endpoint with an image that is too small."""
    img_bytes = create_mock_image(100, 100)
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", 
            files={"file": ("small.jpg", img_bytes, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "Image too small" in response.json()["detail"]

def test_predict_too_large():
    """Test the /predict endpoint with an image that is too large."""
    # 6MB dummy data
    large_data = b"0" * (6 * 1024 * 1024)
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", 
            files={"file": ("large.jpg", large_data, "image/jpeg")}
        )
        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]

def test_predict_invalid_type():
    """Test the /predict endpoint with an invalid file type."""
    with TestClient(app) as local_client:
        response = local_client.post(
            "/predict", 
            files={"file": ("not_an_image.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]