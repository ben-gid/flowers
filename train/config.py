"""Training configuration for the flower classifier.

The canonical representation uses a ``dataclass`` (the ML-community standard
— mirroring HuggingFace ``TrainingArguments``, Lightning ``Trainer`` args,
etc.).  Complex objects that cannot be expressed as plain strings (optimizer
class, scheduler class, pretrained model) are stored as *name strings* and
resolved to real Python objects by :meth:`TrainConfig.resolve` before training
starts.  This keeps the config fully serialisable and trivially editable from
the CLI.
"""

from __future__ import annotations

import argparse
import typing
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin

from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from torch import nn
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LRScheduler,
    ReduceLROnPlateau,
    StepLR,
)
from torchvision import models as tv_models

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
MLFLOW_DB_PATH = BASE_DIR / "mlflow.db"

# ---------------------------------------------------------------------------
# Name -> class registries  (add entries here to expose them via the CLI)
# ---------------------------------------------------------------------------

OPTIMIZER_REGISTRY: dict[str, type[Optimizer]] = {
    "adamw": AdamW,
    "adam": Adam,
    "sgd": SGD,
}

SCHEDULER_REGISTRY: dict[str, type[LRScheduler]] = {
    "cosine": CosineAnnealingLR,
    "step": StepLR,
    "plateau": ReduceLROnPlateau,
}

# Pretrained model names -> (factory_callable, head_name, backbone_name)
PRETRAINED_MODEL_REGISTRY: dict[
    str, tuple[Callable[[], nn.Module], str, str | None]
] = {
    "efficientnet_b0": (
        lambda: tv_models.efficientnet_b0(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "efficientnet_b1": (
        lambda: tv_models.efficientnet_b1(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "efficientnet_b2": (
        lambda: tv_models.efficientnet_b2(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "efficientnet_b3": (
        lambda: tv_models.efficientnet_b3(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "efficientnet_v2_s": (
        lambda: tv_models.efficientnet_v2_s(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "resnet50": (lambda: tv_models.resnet50(weights="IMAGENET1K_V2"), "fc", None),
    "resnet101": (lambda: tv_models.resnet101(weights="IMAGENET1K_V2"), "fc", None),
    "mobilenet_v3_large": (
        lambda: tv_models.mobilenet_v3_large(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "vgg16": (
        lambda: tv_models.vgg16(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "convnext_tiny": (
        lambda: tv_models.convnext_tiny(weights="IMAGENET1K_V1"),
        "classifier",
        "features",
    ),
    "vit_b_16": (lambda: tv_models.vit_b_16(weights="IMAGENET1K_V1"), "heads", None),
}


# Valid precision strings accepted by Lightning
def _extract_valid_precisions(precision_type: UnionType) -> list:
    """Extracts valid precision strings from the _PRECISION_INPUT type hint."""
    valid = []
    # get_args on Union returns the inner types (e.g. Literal)
    # get_args on Literal returns the string/int values
    for arg in get_args(precision_type):
        if get_origin(arg) is getattr(typing, "Literal", None):
            valid.extend(get_args(arg))
        else:
            valid.append(arg)

    return [str(v) for v in valid]


VALID_PRECISION = _extract_valid_precisions(_PRECISION_INPUT)

# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyper-parameters and experiment settings for a training run.

    String fields (``optimizer``, ``scheduler``, ``pretrained_model``) are
    *name keys* into the registries above.  Call :meth:`resolve` to
    obtain the actual class / :class:`nn.Module` instances before passing
    the config to the trainer.
    """

    # ── Experiment ──────────────────────────────────────────────────────────
    exp_name: str = "flower-classification-v2"
    mlflow_db_uri: str = f"sqlite:///{MLFLOW_DB_PATH}"

    # ── Learning rates ───────────────────────────────────────────────────────
    lr_head_stage_1: float = 1e-3
    lr_head_stage_2: float = 1e-3
    lr_backbone: float = 1e-5

    # ── Training schedule ────────────────────────────────────────────────────
    unfreeze_at_epoch: int = 5
    max_epochs: int = 30
    batch_size: int = 64
    accumulate_grad_batches: int = 1
    early_stopping_patience: int = 5

    # ── Precision ────────────────────────────────────────────────────────────
    precision: _PRECISION_INPUT = "16-mixed"  # e.g. "32", "16-mixed", "bf16-mixed"

    # ── Components (resolved from registry at runtime) ───────────────────────
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    pretrained_model: str = "efficientnet_b0"

    # ── Kwargs for components ────────────────────────────────────────────────
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    scheduler_kwargs: dict[str, Any] = field(default_factory=dict)

    # ── Resolved objects (populated by resolve(), never set by user) ─────────
    optimizer_cls: type[Optimizer] = field(default=AdamW, init=False, repr=False)
    scheduler_cls: type[LRScheduler] = field(
        default=CosineAnnealingLR, init=False, repr=False
    )
    pretrained_model_instance: nn.Module | None = field(
        default=None, init=False, repr=False
    )
    head_name: str = field(default="", init=False, repr=False)
    backbone_name: str | None = field(default=None, init=False, repr=False)

    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> TrainConfig:
        """Build a :class:`TrainConfig` from an :class:`argparse.Namespace`."""
        init_field_names = {f.name for f in fields(cls) if f.init}
        kwargs = {
            k: v for k, v in vars(ns).items() if k in init_field_names and v is not None
        }
        return cls(**kwargs)

    def resolve(self) -> TrainConfig:
        """Materialise registry name strings into real class / object refs.

        Call this once before passing the config to the trainer.

        Raises:
            ValueError: if any registry key is unknown.
        """
        if self.optimizer not in OPTIMIZER_REGISTRY:
            raise ValueError(
                f"Unknown optimizer '{self.optimizer}'. "
                f"Choose from: {list(OPTIMIZER_REGISTRY)}"
            )
        if self.scheduler not in SCHEDULER_REGISTRY:
            raise ValueError(
                f"Unknown scheduler '{self.scheduler}'. "
                f"Choose from: {list(SCHEDULER_REGISTRY)}"
            )
        if self.pretrained_model not in PRETRAINED_MODEL_REGISTRY:
            raise ValueError(
                f"Unknown pretrained_model '{self.pretrained_model}'. "
                f"Choose from: {list(PRETRAINED_MODEL_REGISTRY)}"
            )
        if self.precision not in VALID_PRECISION:
            raise ValueError(
                f"Unknown precision '{self.precision}'. "
                f"Choose from: {list(VALID_PRECISION)}"
            )

        self.optimizer_cls = OPTIMIZER_REGISTRY[self.optimizer]
        self.scheduler_cls = SCHEDULER_REGISTRY[self.scheduler]

        factory, self.head_name, self.backbone_name = PRETRAINED_MODEL_REGISTRY[
            self.pretrained_model
        ]

        self.pretrained_model_instance = factory()

        if not self.optimizer_kwargs:
            self.optimizer_kwargs = {"weight_decay": 1e-2}

        if not self.scheduler_kwargs:
            match self.scheduler:
                case "cosine":
                    self.scheduler_kwargs = {"T_max": self.max_epochs, "eta_min": 1e-6}
                case "plateau":
                    self.scheduler_kwargs = {
                        "mode": "max",
                        "patience": 2,
                        "factor": 0.5,
                        "min_lr": 1e-6,
                    }
                case "step":
                    self.scheduler_kwargs = {
                        "step_size": max(1, self.max_epochs // 3),
                        "gamma": 0.1,
                    }

        return self

    def summary(self) -> str:
        """Return a table of the current config values."""
        lines = ["TrainConfig", "─" * 46]
        for f in fields(self):
            if not f.repr:
                continue
            lines.append(f"  {f.name:<30} {getattr(self, f.name)!r}")
        return "\n".join(lines)
