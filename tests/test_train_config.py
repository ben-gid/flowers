"""Tests for train/config.py — TrainConfig dataclass."""

import argparse

import pytest
from config import (
    OPTIMIZER_REGISTRY,
    SCHEDULER_REGISTRY,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


def test_resolve_defaults():
    cfg = TrainConfig().resolve()
    assert cfg.optimizer_cls is OPTIMIZER_REGISTRY["adamw"]
    assert cfg.scheduler_cls is SCHEDULER_REGISTRY["cosine"]
    assert cfg.pretrained_model_instance is not None
    assert cfg.head_name == "classifier"  # efficientnet_b0
    assert cfg.optimizer_kwargs  # filled by resolve
    assert cfg.scheduler_kwargs  # filled by resolve


def test_resolve_all_optimizers():
    for name in OPTIMIZER_REGISTRY:
        cfg = TrainConfig(optimizer=name).resolve()
        assert cfg.optimizer_cls is OPTIMIZER_REGISTRY[name]


def test_resolve_all_schedulers():
    for name in SCHEDULER_REGISTRY:
        cfg = TrainConfig(scheduler=name).resolve()
        assert cfg.scheduler_cls is SCHEDULER_REGISTRY[name]


def test_resolve_bad_optimizer():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        TrainConfig(optimizer="nope").resolve()


def test_resolve_bad_scheduler():
    with pytest.raises(ValueError, match="Unknown scheduler"):
        TrainConfig(scheduler="nope").resolve()


def test_resolve_bad_model():
    with pytest.raises(ValueError, match="Unknown pretrained_model"):
        TrainConfig(pretrained_model="nope").resolve()


def test_resolve_bad_precision():
    with pytest.raises(ValueError, match="Unknown precision"):
        TrainConfig(precision="nope").resolve()  # type: ignore[arg-type]


def test_resolve_resnet_head_name():
    cfg = TrainConfig(pretrained_model="resnet50").resolve()
    assert cfg.head_name == "fc"
    assert cfg.backbone_name is None


# ---------------------------------------------------------------------------
# from_namespace()
# ---------------------------------------------------------------------------


def test_from_namespace_roundtrip():
    ns = argparse.Namespace(
        exp_name="test-run",
        mlflow_db_uri="sqlite:///test.db",
        lr_head_stage_1=1e-4,
        lr_head_stage_2=1e-4,
        lr_backbone=1e-6,
        unfreeze_at_epoch=3,
        max_epochs=10,
        batch_size=32,
        accumulate_grad_batches=2,
        early_stopping_patience=3,
        precision="32",
        optimizer="adam",
        scheduler="step",
        pretrained_model="efficientnet_b0",
    )
    cfg = TrainConfig.from_namespace(ns)
    assert cfg.exp_name == "test-run"
    assert cfg.max_epochs == 10
    assert cfg.optimizer == "adam"
    assert cfg.scheduler == "step"


def test_from_namespace_ignores_none():
    """None values in namespace should not override defaults."""
    ns = argparse.Namespace(
        exp_name=None,
        mlflow_db_uri=None,
        lr_head_stage_1=None,
        lr_head_stage_2=None,
        lr_backbone=None,
        unfreeze_at_epoch=None,
        max_epochs=None,
        batch_size=None,
        accumulate_grad_batches=None,
        early_stopping_patience=None,
        precision=None,
        optimizer=None,
        scheduler=None,
        pretrained_model=None,
    )
    cfg = TrainConfig.from_namespace(ns)
    default = TrainConfig()
    assert cfg.max_epochs == default.max_epochs
    assert cfg.optimizer == default.optimizer


# ---------------------------------------------------------------------------
# summary()
# ---------------------------------------------------------------------------


def test_summary_contains_key_fields():
    summary = TrainConfig().summary()
    assert "TrainConfig" in summary
    assert "optimizer" in summary
    assert "max_epochs" in summary
    # resolved fields (repr=False) must NOT appear
    assert "optimizer_cls" not in summary
