"""Tests for src/ utilities (src/utils.py) and src/data.py split logic."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from src.data import FlowerDataModule
from utils import get_class_weights, get_transforms

# ---------------------------------------------------------------------------
# get_transforms
# ---------------------------------------------------------------------------


def test_get_transforms_returns_two_composes():
    train_t, val_t = get_transforms()
    assert train_t is not None
    assert val_t is not None


def test_val_transform_shape():
    _, val_t = get_transforms()
    img = Image.new("RGB", (256, 256), color="green")
    tensor = val_t(img)
    assert tensor.shape == (3, 224, 224), f"unexpected shape {tensor.shape}"


def test_train_transform_shape():
    train_t, _ = get_transforms()
    img = Image.new("RGB", (256, 256), color="blue")
    tensor = train_t(img)
    assert tensor.shape == (3, 224, 224)


def test_val_transform_normalizes():
    """The transform the API serves with must apply (x - mean) / std."""
    mean = torch.tensor([0.4727, 0.3996, 0.3193])
    std = torch.tensor([0.2965, 0.2471, 0.2812])

    _, val_t = get_transforms()
    img = Image.new("RGB", (256, 256), color=(255, 128, 0))
    tensor = val_t(img)

    expected = (torch.tensor([255, 128, 0]) / 255 - mean) / std
    per_channel = tensor.flatten(1).mean(dim=1)
    assert torch.allclose(per_channel, expected, atol=1e-4), per_channel


# ---------------------------------------------------------------------------
# get_class_weights
# ---------------------------------------------------------------------------


def test_class_weights_shape():
    labels = np.array([0, 0, 1, 2, 2, 2])
    weights = get_class_weights(labels)
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (3,)


def test_class_weights_balanced():
    """Balanced dataset → all weights equal."""
    labels = np.array([0, 1, 2] * 10)
    weights = get_class_weights(labels)
    assert torch.allclose(weights, weights[0].expand_as(weights), atol=1e-5)


def test_class_weights_minority_gets_more():
    """Minority class should have a higher weight."""
    labels = np.array([0] * 90 + [1] * 10)
    weights = get_class_weights(labels)
    assert weights[1] > weights[0]


# ---------------------------------------------------------------------------
# FlowerDataModule splits
# ---------------------------------------------------------------------------


def test_splits_are_disjoint():
    class FakeDS:
        def __len__(self):
            return 1000

        def __getitem__(self, i):
            return i, 0

    with patch("src.data.FlowerDataset", return_value=FakeDS()):
        dm = FlowerDataModule(Path("."), batch_size=1)
        dm.setup("fit")

    idx = {
        name: set(getattr(dm, name).subset.indices)
        for name in ("train_set", "val_set", "test_set")
    }
    assert not idx["train_set"] & idx["val_set"], "train/val leakage"
    assert not idx["train_set"] & idx["test_set"], "train/test leakage"
    assert not idx["val_set"] & idx["test_set"], "val/test leakage"
    assert sum(len(s) for s in idx.values()) == 1000
