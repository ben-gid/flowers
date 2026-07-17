"""Tests for the v2 API (api/app/v2/main.py)."""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from api.app.v2.main import app
from api.app.v2.core.config import state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_jpeg(width: int = 224, height: int = 224) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color="red").save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def loaded_state():
    """Stub AppState with mocks; prevent lifespan from overwriting them."""
    mock_model = MagicMock()
    mock_model.parameters.return_value = iter([MagicMock(device=torch.device("cpu"))])
    mock_model.return_value = torch.tensor([[10.0] + [0.0] * 101])  # class 0 wins

    class_names = [f"Class {i}" for i in range(102)]
    class_names[0] = "rose"

    def _fake_load(logger=None):
        state.classifier = mock_model
        state.class_names = class_names
        state.transform = lambda img: torch.zeros(3, 224, 224)

    with patch("api.app.v2.core.config.AppState.load", side_effect=_fake_load):
        yield
    state.clear(logger=MagicMock())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_health_ok(loaded_state):
    with TestClient(app) as client:
        r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_classify_success(loaded_state):
    with TestClient(app) as client:
        r = client.post(
            "/classify",
            files={"file": ("flower.jpg", _make_jpeg(), "image/jpeg")},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["prediction"] == "rose"
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["filename"] == "flower.jpg"
    assert data["content_type"] == "image/jpeg"


def test_classify_image_too_small(loaded_state):
    with TestClient(app) as client:
        r = client.post(
            "/classify",
            files={"file": ("small.jpg", _make_jpeg(100, 100), "image/jpeg")},
        )
    assert r.status_code == 400
    assert "too small" in r.json()["detail"].lower()


def test_classify_file_too_large(loaded_state):
    large = b"0" * (6 * 1024 * 1024)  # 6 MB > 5 MB limit
    with TestClient(app) as client:
        r = client.post(
            "/classify",
            files={"file": ("big.jpg", large, "image/jpeg")},
        )
    assert r.status_code == 413
    assert "too large" in r.json()["detail"].lower()


def test_classify_invalid_file_type(loaded_state):
    with TestClient(app) as client:
        r = client.post(
            "/classify",
            files={"file": ("doc.txt", b"not an image", "text/plain")},
        )
    assert r.status_code == 400
    assert "invalid file type" in r.json()["detail"].lower()


def test_classify_no_model_loaded():
    """Classifier is None → 500."""
    def _load_without_classifier(logger=None):
        state.classifier = None
        state.class_names = ["rose"]
        state.transform = lambda img: torch.zeros(3, 224, 224)

    with patch("api.app.v2.core.config.AppState.load", side_effect=_load_without_classifier):
        with TestClient(app) as client:
            r = client.post(
                "/classify",
                files={"file": ("flower.jpg", _make_jpeg(), "image/jpeg")},
            )
        assert r.status_code == 500
    state.clear(logger=MagicMock())
