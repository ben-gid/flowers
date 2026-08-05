"""Training script for the flower classifier.

This module exposes a single :func:`run_training` function that accepts a
resolved :class:`~config.TrainConfig` and orchestrates the full Lightning
training loop.  It is called by the CLI (``cli/main.py``) but can also be
imported and called programmatically (e.g. from a notebook or a sweep).
"""

from __future__ import annotations

import sys
from dataclasses import fields as ds_fields
from pathlib import Path

# Ensure repo-root packages are importable when running directly
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "cli"))  # for config
sys.path.insert(0, str(_REPO_ROOT))  # for src.* imports

import lightning as L  # noqa: E402
import mlflow  # noqa: E402
from config import TrainConfig  # noqa: E402
from lightning.pytorch.callbacks import (  # noqa: E402
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import MLFlowLogger  # noqa: E402

from src.callbacks import BackboneFinetuning  # noqa: E402
from src.classifier import FlowerClassifier  # noqa: E402
from src.data import FlowerDataModule, FlowerDataset  # noqa: E402
from src.utils import get_class_weights  # noqa: E402

BASE_DIR = _REPO_ROOT
DATA_ROOT = BASE_DIR / "data"
MODEL_CHX_DIR = BASE_DIR / "models_checkpoints"


def run_training(cfg: TrainConfig) -> None:
    """Run a full training job described by *cfg*.

    Parameters
    ----------
    cfg:
        A **resolved** :class:`TrainConfig` (i.e. :meth:`~TrainConfig.resolve`
        must have been called beforehand so that ``cfg.optimizer_cls``,
        ``cfg.scheduler_cls``, and ``cfg.pretrained_model_instance`` are set).
    """
    if cfg.pretrained_model_instance is None:
        raise RuntimeError(
            "cfg.pretrained_model_instance is None — did you call cfg.resolve()?"
        )

    # ── MLflow setup ─────────────────────────────────────────────────────────
    mlflow.set_tracking_uri(cfg.mlflow_db_uri)
    experiment = mlflow.get_experiment_by_name(cfg.exp_name)
    if experiment is None:
        mlflow.create_experiment(
            cfg.exp_name,
            artifact_location=str(BASE_DIR / "artifacts"),
        )

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.exp_name,
        tracking_uri=cfg.mlflow_db_uri,
        log_model=True,
    )

    run_id = mlflow_logger.run_id or "unknown"

    # ── Data ─────────────────────────────────────────────────────────────────
    flower_dm = FlowerDataModule(DATA_ROOT, batch_size=cfg.batch_size)
    dataset = FlowerDataset(DATA_ROOT)

    # ── Model ─────────────────────────────────────────────────────────────────
    pl_model = FlowerClassifier(
        pretrained_model=cfg.pretrained_model_instance,
        num_classes=len(dataset.classes),
        class_weights=get_class_weights(dataset.labels),
        lr_head=cfg.lr_head_stage_1,
        optimizer=cfg.optimizer_cls,
        scheduler=cfg.scheduler_cls,
        head_name=cfg.head_name,
        optimizer_kwargs=cfg.optimizer_kwargs,
        scheduler_kwargs=cfg.scheduler_kwargs,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        BackboneFinetuning(
            unfreeze_at_epoch=cfg.unfreeze_at_epoch,
            lr_backbone=cfg.lr_backbone,
            new_lr_head=cfg.lr_head_stage_2,
            backbone_name=cfg.backbone_name,
        ),
        ModelCheckpoint(
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            filename=f"{cfg.pretrained_model}-{{epoch}}-{{val_acc:.3f}}-{run_id[:8]}",
            dirpath=MODEL_CHX_DIR,
        ),
        EarlyStopping(
            monitor="val_acc",
            mode="max",
            patience=cfg.early_stopping_patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        logger=mlflow_logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        precision=cfg.precision,
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
    )

    mlflow_logger.log_hyperparams(
        {
            "effective_batch_size": (
                flower_dm.batch_size * trainer.accumulate_grad_batches
            ),
            **{f.name: getattr(cfg, f.name) for f in ds_fields(cfg) if f.repr},
        }
    )

    trainer.fit(pl_model, datamodule=flower_dm)
    trainer.test(pl_model, datamodule=flower_dm, ckpt_path="best")

    # per-class F1 is a 102-vector, so it goes out as an artifact rather than a metric
    mlflow_logger.experiment.log_dict(
        run_id,
        dict(
            zip(
                dataset.classes,
                pl_model.test_per_class_f1.compute().tolist(),
                strict=True,
            )
        ),
        "per_class_f1.json",
    )


if __name__ == "__main__":
    print(f"Base dir: {_REPO_ROOT}")
    print(f"Data dir: {DATA_ROOT}")
    print(f"Model check dir: {MODEL_CHX_DIR}")
