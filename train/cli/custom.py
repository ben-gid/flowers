"""CLI entrypoint for the flower classifier.

Usage examples
--------------
Train with defaults::

    uv run train/cli.py train

Override specific fields::

    uv run train/cli.py train \\
        --max-epochs 50 \\
        --batch-size 32 \\
        --optimizer adamw \\
        --lr-head-stage-1 5e-4 \\
        --pretrained-model resnet50 \\
        --precision bf16-mixed

Dry-run (print resolved config without training)::

    uv run train/cli.py train --dry-run

List available choices for registry fields::

    uv run train/cli.py choices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run directly
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "train"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from config import (  # noqa: E402
    OPTIMIZER_REGISTRY,
    PRETRAINED_MODEL_REGISTRY,
    SCHEDULER_REGISTRY,
    VALID_PRECISION,
    TrainConfig,
)

# ---------------------------------------------------------------------------
# Argument parser builders
# ---------------------------------------------------------------------------


def _build_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``train`` sub-command."""
    p = subparsers.add_parser(
        "train",
        help="Run a training job with the given hyper-parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment ──────────────────────────────────────────────────────────
    exp = p.add_argument_group("experiment")
    exp.add_argument(
        "--exp-name",
        default=TrainConfig.exp_name,
        metavar="NAME",
        help="MLflow experiment name.",
    )
    exp.add_argument(
        "--mlflow-db-uri",
        default=TrainConfig.mlflow_db_uri,
        metavar="URI",
        help="SQLAlchemy tracking URI for MLflow.",
    )

    # ── Learning rates ───────────────────────────────────────────────────────
    lr = p.add_argument_group("learning rates")
    lr.add_argument(
        "--lr-head-stage-1",
        type=float,
        default=TrainConfig.lr_head_stage_1,
        metavar="LR",
        help="Head learning rate during stage 1 (backbone frozen).",
    )
    lr.add_argument(
        "--lr-head-stage-2",
        type=float,
        default=TrainConfig.lr_head_stage_2,
        metavar="LR",
        help="Head learning rate during stage 2 (after unfreeze).",
    )
    lr.add_argument(
        "--lr-backbone",
        type=float,
        default=TrainConfig.lr_backbone,
        metavar="LR",
        help="Backbone learning rate during stage 2.",
    )

    # ── Training schedule ────────────────────────────────────────────────────
    sched = p.add_argument_group("training schedule")
    sched.add_argument(
        "--unfreeze-at-epoch",
        type=int,
        default=TrainConfig.unfreeze_at_epoch,
        metavar="N",
        help="Epoch at which backbone is unfrozen.",
    )
    sched.add_argument(
        "--max-epochs",
        type=int,
        default=TrainConfig.max_epochs,
        metavar="N",
        help="Maximum number of training epochs.",
    )
    sched.add_argument(
        "--batch-size",
        type=int,
        default=TrainConfig.batch_size,
        metavar="N",
        help="Dataloader batch size per GPU.",
    )
    sched.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=TrainConfig.accumulate_grad_batches,
        metavar="N",
        help="Gradient accumulation steps (effective batch = batch_size x N).",
    )
    sched.add_argument(
        "--early-stopping-patience",
        type=int,
        default=TrainConfig.early_stopping_patience,
        metavar="N",
        help="Number of epochs with no val_acc improvement before stopping.",
    )

    # ── Precision ────────────────────────────────────────────────────────────
    hw = p.add_argument_group("hardware / precision")
    hw.add_argument(
        "--precision",
        default=TrainConfig.precision,
        choices=VALID_PRECISION,
        help="Lightning trainer precision.",
    )

    # ── Components ───────────────────────────────────────────────────────────
    comp = p.add_argument_group("components")
    comp.add_argument(
        "--optimizer",
        default=TrainConfig.optimizer,
        choices=list(OPTIMIZER_REGISTRY),
        help="Optimiser to use.",
    )
    comp.add_argument(
        "--scheduler",
        default=TrainConfig.scheduler,
        choices=list(SCHEDULER_REGISTRY),
        help="LR scheduler to use.",
    )
    comp.add_argument(
        "--pretrained-model",
        default=TrainConfig.pretrained_model,
        choices=list(PRETRAINED_MODEL_REGISTRY),
        help="Pretrained backbone to fine-tune.",
    )

    # ── Misc ─────────────────────────────────────────────────────────────────
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved config without running training.",
    )


def _build_choices_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Register the ``choices`` sub-command."""
    subparsers.add_parser(
        "choices",
        help="List valid choices for registry fields (optimizers, schedulers, models).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="flowers",
        description="Flower classification training CLI.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True
    _build_train_parser(sub)
    _build_choices_parser(sub)
    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _argparse_key(name: str) -> str:
    """Convert argparse dest (underscores) back to CLI flag name."""
    return name.replace("_", "-")


def _namespace_to_config(ns: argparse.Namespace) -> TrainConfig:
    """Map the flat argparse namespace to a :class:`TrainConfig` instance."""
    # argparse stores --foo-bar as ns.foo_bar; TrainConfig uses foo_bar too.
    return TrainConfig(
        exp_name=ns.exp_name,
        mlflow_db_uri=ns.mlflow_db_uri,
        lr_head_stage_1=ns.lr_head_stage_1,
        lr_head_stage_2=ns.lr_head_stage_2,
        lr_backbone=ns.lr_backbone,
        unfreeze_at_epoch=ns.unfreeze_at_epoch,
        max_epochs=ns.max_epochs,
        batch_size=ns.batch_size,
        accumulate_grad_batches=ns.accumulate_grad_batches,
        early_stopping_patience=ns.early_stopping_patience,
        precision=ns.precision,
        optimizer=ns.optimizer,
        scheduler=ns.scheduler,
        pretrained_model=ns.pretrained_model,
    )


def cmd_train(ns: argparse.Namespace) -> None:
    cfg = _namespace_to_config(ns)
    cfg.resolve()

    print(cfg.summary())
    print()

    if ns.dry_run:
        print("[dry-run] Skipping training.")
        return

    # late import for heavy function
    from run_training import run_training  # noqa: PLC0415
    run_training(cfg)


def cmd_choices(_ns: argparse.Namespace) -> None:
    print("Optimizers:      ", ", ".join(OPTIMIZER_REGISTRY))
    print("Schedulers:      ", ", ".join(SCHEDULER_REGISTRY))
    print("Pretrained models:", ", ".join(PRETRAINED_MODEL_REGISTRY))
    print("Precision:       ", ", ".join(VALID_PRECISION))


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

COMMAND_HANDLERS = {
    "train": cmd_train,
    "choices": cmd_choices,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    ns = parser.parse_args(argv)
    handler = COMMAND_HANDLERS[ns.command]
    handler(ns)


if __name__ == "__main__":
    main()
