"""Hyperparameter sweep for the flower classifier.

Runs :func:`~train.run_training` sequentially across a curated set of
configurations that have a high probability of producing the best F1 /
accuracy while also covering efficient low-cost options.

Configs are grouped into three tiers:
  • Tier A – max accuracy   (larger backbones, longer schedules)
  • Tier B – balanced       (medium backbones, reasonable schedules)
  • Tier C – efficient      (lightweight backbones, fast schedules)

Usage
-----
Run all configs::

    python train/sweep.py

Dry-run (print configs without training)::

    python train/sweep.py --dry-run

Run only a specific tier (A / B / C)::

    python train/sweep.py --tier B

Skip the first N configs (resume after a crash)::

    python train/sweep.py --skip 3

Log to a different MLflow experiment::

    python train/sweep.py --exp-name my-sweep
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Path bootstrap – identical to train.py so this file is runnable on its own
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "train"))  # train.py lives here
sys.path.insert(0, str(_REPO_ROOT))  # src.* imports

from config import TrainConfig  # noqa: E402
from run_training import run_training  # noqa: E402

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

Tier = Literal["A", "B", "C"]


@dataclass(frozen=True)
class SweepConfig:
    """A single point in the sweep grid.

    All fields map 1-to-1 with :class:`~config.TrainConfig` init parameters.
    ``label`` is a human-readable name shown in the console header.
    ``tier`` controls which group this config belongs to.
    """

    label: str
    tier: Tier

    # backbone
    pretrained_model: str = "efficientnet_b2"

    # optimizer + scheduler
    optimizer: str = "adamw"
    scheduler: str = "cosine"

    # learning rates
    lr_head_stage_1: float = 1e-3
    lr_head_stage_2: float = 5e-4
    lr_backbone: float = 1e-5

    # schedule
    unfreeze_at_epoch: int = 5
    max_epochs: int = 40
    batch_size: int = 64
    accumulate_grad_batches: int = 1
    early_stopping_patience: int = 7

    # hardware
    precision: str = "16-mixed"

    def to_train_config(self, exp_name: str | None = None) -> TrainConfig:
        return TrainConfig(
            exp_name=exp_name or TrainConfig.exp_name,
            pretrained_model=self.pretrained_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            lr_head_stage_1=self.lr_head_stage_1,
            lr_head_stage_2=self.lr_head_stage_2,
            lr_backbone=self.lr_backbone,
            unfreeze_at_epoch=self.unfreeze_at_epoch,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            accumulate_grad_batches=self.accumulate_grad_batches,
            early_stopping_patience=self.early_stopping_patience,
            precision=self.precision,  # type: ignore
        )


# ── Tier A: Max accuracy ────────────────────────────────────────────────────
#
# Strategy: large EfficientNet / ResNet backbones, long warm-up + cosine LR,
# moderate backbone LR after unfreeze, mixed precision throughout.

TIER_A: list[SweepConfig] = [
    # A1 – EfficientNet-B3 + AdamW/cosine — strong baseline for accuracy
    SweepConfig(
        label="A1 | EffNet-B3 | AdamW | Cosine | bs64 | e50",
        tier="A",
        pretrained_model="efficientnet_b3",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=50,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A2 – EfficientNet-B3 + AdamW/cosine, early unfreeze + higher backbone LR
    SweepConfig(
        label="A2 | EffNet-B3 | AdamW | Cosine | early-unfreeze | bs64",
        tier="A",
        pretrained_model="efficientnet_b3",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=3,
        max_epochs=50,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A3 – ResNet-101 + AdamW/cosine — classic workhorse
    SweepConfig(
        label="A3 | ResNet-101 | AdamW | Cosine | bs32 | accum2",
        tier="A",
        pretrained_model="resnet101",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=50,
        batch_size=32,
        accumulate_grad_batches=2,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A4 – EfficientNet-B3 + AdamW/plateau — adaptive LR reduction
    SweepConfig(
        label="A4 | EffNet-B3 | AdamW | Plateau | bs64 | e40",
        tier="A",
        pretrained_model="efficientnet_b3",
        optimizer="adamw",
        scheduler="plateau",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=40,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=10,
        precision="16-mixed",
    ),
    # A5 – EfficientNet-B2 + AdamW/cosine, bigger effective batch via accumulation
    SweepConfig(
        label="A5 | EffNet-B2 | AdamW | Cosine | bs32 | accum4",
        tier="A",
        pretrained_model="efficientnet_b2",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=45,
        batch_size=32,
        accumulate_grad_batches=4,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A6 – ResNet-50 + AdamW/cosine — lighter than 101 but still high accuracy
    SweepConfig(
        label="A6 | ResNet-50 | AdamW | Cosine | bs64 | e40",
        tier="A",
        pretrained_model="resnet50",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=40,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A7 – EfficientNet-B3 + SGD/cosine — SGD often edges AdamW on fine-tuning
    SweepConfig(
        label="A7 | EffNet-B3 | SGD | Cosine | bs64 | e50",
        tier="A",
        pretrained_model="efficientnet_b3",
        optimizer="sgd",
        scheduler="cosine",
        lr_head_stage_1=1e-2,
        lr_head_stage_2=5e-3,
        lr_backbone=1e-4,
        unfreeze_at_epoch=5,
        max_epochs=50,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=10,
        precision="16-mixed",
    ),
    # A8 – ConvNeXt-Tiny + AdamW/cosine — modern ConvNet design (depthwise convs +
    # LayerNorm + GELU); was already in the registry but unused in the sweep.
    # Similar param count to EffNet-B3/ResNet-50 but tends to out-perform both on
    # fine-grained classification tasks.
    SweepConfig(
        label="A8 | ConvNeXt-Tiny | AdamW | Cosine | bs64 | e45",
        tier="A",
        pretrained_model="convnext_tiny",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=45,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # A9 – EfficientNetV2-S + AdamW/cosine — V2's fused-MBConv stem + stronger
    # ImageNet pretraining should beat the V1 EfficientNet line at similar cost.
    # Low backbone LR since V2 features are already well-tuned.
    SweepConfig(
        label="A9 | EffNetV2-S | AdamW | Cosine | bs64 | e45",
        tier="A",
        pretrained_model="efficientnet_v2_s",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=5,
        max_epochs=45,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
]

# ── Tier B: Balanced accuracy + efficiency ──────────────────────────────────
#
# Strategy: EfficientNet-B1/B2 + medium schedules — good accuracy in fewer
# GPU-hours.

TIER_B: list[SweepConfig] = [
    # B1 – EfficientNet-B2 + AdamW/cosine — default "go-to" balanced config
    SweepConfig(
        label="B1 | EffNet-B2 | AdamW | Cosine | bs64 | e35",
        tier="B",
        pretrained_model="efficientnet_b2",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B2 – ConvNeXt-Tiny + AdamW/cosine — lighter-schedule sibling of A8, testing
    # whether the modern-architecture edge shows up even at "balanced" budget
    SweepConfig(
        label="B2 | ConvNeXt-Tiny | AdamW | Cosine | bs64 | e35",
        tier="B",
        pretrained_model="convnext_tiny",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B3 – EfficientNet-B1 + AdamW/cosine — slightly lighter than B2
    SweepConfig(
        label="B3 | EffNet-B1 | AdamW | Cosine | bs64 | e35",
        tier="B",
        pretrained_model="efficientnet_b1",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B4 – ResNet-50 + AdamW/plateau — adaptive scheduler, robust to noisy val
    SweepConfig(
        label="B4 | ResNet-50 | AdamW | Plateau | bs64 | e35",
        tier="B",
        pretrained_model="resnet50",
        optimizer="adamw",
        scheduler="plateau",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # B5 – EfficientNet-B2 + AdamW/step — simple schedule, often underrated
    SweepConfig(
        label="B5 | EffNet-B2 | AdamW | Step | bs64 | e35",
        tier="B",
        pretrained_model="efficientnet_b2",
        optimizer="adamw",
        scheduler="step",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B6 – EfficientNet-B2 + Adam/cosine — compare plain Adam vs AdamW
    SweepConfig(
        label="B6 | EffNet-B2 | Adam | Cosine | bs64 | e35",
        tier="B",
        pretrained_model="efficientnet_b2",
        optimizer="adam",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B7 – EfficientNet-B2 + AdamW/cosine, late unfreeze
    SweepConfig(
        label="B7 | EffNet-B2 | AdamW | Cosine | late-unfreeze | bs64",
        tier="B",
        pretrained_model="efficientnet_b2",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=5e-6,
        unfreeze_at_epoch=10,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
    # B8 – ResNet-50 + SGD/cosine — momentum-based vs adaptive comparison
    SweepConfig(
        label="B8 | ResNet-50 | SGD | Cosine | bs64 | e40",
        tier="B",
        pretrained_model="resnet50",
        optimizer="sgd",
        scheduler="cosine",
        lr_head_stage_1=1e-2,
        lr_head_stage_2=5e-3,
        lr_backbone=1e-4,
        unfreeze_at_epoch=5,
        max_epochs=40,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=8,
        precision="16-mixed",
    ),
    # B9 – EfficientNetV2-S + AdamW/cosine — this is probably your best
    # "balanced" pick: V2-S is roughly EffNet-B2-cost but with the training
    # improvements from V2, so it's the one I'd watch closest against B1/B2.
    SweepConfig(
        label="B9 | EffNetV2-S | AdamW | Cosine | bs64 | e35",
        tier="B",
        pretrained_model="efficientnet_v2_s",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=35,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=7,
        precision="16-mixed",
    ),
]

# ── Tier C: Efficient / fast ─────────────────────────────────────────────────
#
# Strategy: EfficientNet-B0 / MobileNetV3 — minimal compute, suitable for
# rapid iteration, CI checks, or resource-constrained environments.

TIER_C: list[SweepConfig] = [
    # C1 – EfficientNet-B0 + AdamW/cosine — fastest EfficientNet
    SweepConfig(
        label="C1 | EffNet-B0 | AdamW | Cosine | bs64 | e30",
        tier="C",
        pretrained_model="efficientnet_b0",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=30,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=6,
        precision="16-mixed",
    ),
    # C2 – MobileNetV3-Large + AdamW/cosine — very fast, surprisingly capable
    SweepConfig(
        label="C2 | MobileNetV3 | AdamW | Cosine | bs128 | e30",
        tier="C",
        pretrained_model="mobilenet_v3_large",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=5,
        max_epochs=30,
        batch_size=128,
        accumulate_grad_batches=1,
        early_stopping_patience=6,
        precision="16-mixed",
    ),
    # C3 – EfficientNet-B1 + AdamW/step — aggressive step decay for a fast finish
    SweepConfig(
        label="C3 | EffNet-B1 | AdamW | Step | bs128 | e25",
        tier="C",
        pretrained_model="efficientnet_b1",
        optimizer="adamw",
        scheduler="step",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=4,
        max_epochs=25,
        batch_size=128,
        accumulate_grad_batches=1,
        early_stopping_patience=5,
        precision="16-mixed",
    ),
    # C4 – EfficientNet-B0 + AdamW/plateau — short but adaptive
    SweepConfig(
        label="C4 | EffNet-B0 | AdamW | Plateau | bs64 | e25",
        tier="C",
        pretrained_model="efficientnet_b0",
        optimizer="adamw",
        scheduler="plateau",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=3e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=4,
        max_epochs=25,
        batch_size=64,
        accumulate_grad_batches=1,
        early_stopping_patience=5,
        precision="16-mixed",
    ),
    # C5 – MobileNetV3 + AdamW/step — maximum speed with decisive LR drops
    SweepConfig(
        label="C5 | MobileNetV3 | AdamW | Step | bs128 | e20",
        tier="C",
        pretrained_model="mobilenet_v3_large",
        optimizer="adamw",
        scheduler="step",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=3,
        max_epochs=20,
        batch_size=128,
        accumulate_grad_batches=1,
        early_stopping_patience=5,
        precision="16-mixed",
    ),
    # C6 – EfficientNetV2-S + AdamW/cosine, short schedule — tests whether V2's
    # training-efficiency gains hold up even under a tight epoch budget.
    SweepConfig(
        label="C6 | EffNetV2-S | AdamW | Cosine | bs96 | e25",
        tier="C",
        pretrained_model="efficientnet_v2_s",
        optimizer="adamw",
        scheduler="cosine",
        lr_head_stage_1=1e-3,
        lr_head_stage_2=5e-4,
        lr_backbone=1e-5,
        unfreeze_at_epoch=4,
        max_epochs=25,
        batch_size=96,
        accumulate_grad_batches=1,
        early_stopping_patience=6,
        precision="16-mixed",
    ),
]

ALL_CONFIGS: list[SweepConfig] = TIER_A + TIER_B + TIER_C


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _separator(label: str, width: int = 70) -> str:
    bar = "─" * width
    return f"\n{bar}\n  {label}\n{bar}"


def run_sweep(
    configs: list[SweepConfig],
    dry_run: bool = False,
    skip: int = 0,
    exp_name: str | None = None,
) -> None:
    """Run (or print) every config in *configs* sequentially.

    Parameters
    ----------
    configs:
        Ordered list of :class:`SweepConfig` objects to run.
    dry_run:
        If *True*, print each resolved config but skip actual training.
    skip:
        Skip the first *skip* configs (useful to resume after a crash).
    exp_name:
        Override the MLflow experiment name for every config in the sweep.
    """
    total = len(configs)
    if skip:
        print(f"[sweep] Skipping first {skip} config(s) as requested.")
        configs = configs[skip:]

    results: list[dict] = []

    for idx, sc in enumerate(configs, start=skip + 1):
        print(_separator(f"[{idx}/{total}] {sc.label}  (Tier {sc.tier})"))

        cfg: TrainConfig = sc.to_train_config(exp_name).resolve()
        print(cfg.summary())
        print()

        if dry_run:
            print("  [dry-run] Skipping training.\n")
            results.append({"label": sc.label, "status": "skipped"})
            continue

        t0 = time.perf_counter()
        try:
            run_training(cfg)
            elapsed = time.perf_counter() - t0
            results.append({"label": sc.label, "status": "ok", "elapsed_s": elapsed})
            print(f"\n  Finished in {elapsed / 60:.1f} min")
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            print(f"\n  FAILED after {elapsed / 60:.1f} min: {exc}")
            results.append({"label": sc.label, "status": "error", "error": str(exc)})

    # ── Summary ──────────────────────────────────────────────────────────────
    print(_separator("Sweep complete"))
    for r in results:
        if r["status"] == "ok":
            status_icon = "ok  "
        elif r["status"] == "skipped":
            status_icon = "skip"
        else:
            status_icon = "FAIL"
        mins = f"  {r['elapsed_s'] / 60:.1f} min" if "elapsed_s" in r else ""
        print(f"  [{status_icon}]  {r['label']}{mins}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="sweep",
        description=textwrap.dedent(
            """\
            Hyperparameter sweep for the flower classifier.

            Runs train.run_training sequentially across curated configs grouped
            into three tiers:
              Tier A – max accuracy   (9 configs)
              Tier B – balanced       (9 configs)
              Tier C – efficient/fast (6 configs)
            """
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print each resolved config but skip actual training.",
    )
    p.add_argument(
        "--tier",
        choices=["A", "B", "C"],
        metavar="TIER",
        help="Run only configs in the specified tier (A / B / C).",
    )
    p.add_argument(
        "--skip",
        type=int,
        default=0,
        metavar="N",
        help="Skip the first N configs globally (useful to resume after a crash).",
    )
    p.add_argument(
        "--exp-name",
        default=None,
        metavar="NAME",
        help=f"MLflow experiment name for every run (default: {TrainConfig.exp_name}).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List all configs without running them.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    # Filter by tier if requested
    if args.tier:
        configs = [c for c in ALL_CONFIGS if c.tier == args.tier]
    else:
        configs = ALL_CONFIGS

    if args.list_only:
        print(f"{'#':<4} {'Tier':<6} {'Label'}")
        print("-" * 80)
        for i, sc in enumerate(configs, 1):
            print(f"{i:<4} {sc.tier:<6} {sc.label}")
        return

    print(f"[sweep] {len(configs)} config(s) queued")
    if args.tier:
        print(f"[sweep] Tier filter: {args.tier}")
    if args.dry_run:
        print("[sweep] DRY RUN -- no training will occur")
    print()

    run_sweep(configs, dry_run=args.dry_run, skip=args.skip, exp_name=args.exp_name)


if __name__ == "__main__":
    main()
