"""Tests for train/cli/custom.py — argument parser and command handlers."""

import pytest
from cli.custom import (
    build_parser,
    cmd_choices,
    main,
)
from config import TrainConfig

# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_parser_train_defaults():
    parser = build_parser()
    ns = parser.parse_args(["train"])
    assert ns.command == "train"
    assert ns.max_epochs == TrainConfig.max_epochs
    assert ns.optimizer == TrainConfig.optimizer
    assert ns.scheduler == TrainConfig.scheduler
    assert ns.pretrained_model == TrainConfig.pretrained_model
    assert ns.dry_run is False


def test_parser_train_overrides():
    parser = build_parser()
    ns = parser.parse_args(
        [
            "train",
            "--max-epochs",
            "5",
            "--batch-size",
            "16",
            "--optimizer",
            "adam",
            "--scheduler",
            "step",
            "--pretrained-model",
            "resnet50",
            "--dry-run",
        ]
    )
    assert ns.max_epochs == 5
    assert ns.batch_size == 16
    assert ns.optimizer == "adam"
    assert ns.scheduler == "step"
    assert ns.pretrained_model == "resnet50"
    assert ns.dry_run is True


def test_parser_choices_subcommand():
    parser = build_parser()
    ns = parser.parse_args(["choices"])
    assert ns.command == "choices"


def test_parser_invalid_optimizer_rejected():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--optimizer", "badopt"])


# ---------------------------------------------------------------------------
# cmd_choices
# ---------------------------------------------------------------------------


def test_cmd_choices_output(capsys):
    cmd_choices(None)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "adamw" in out
    assert "cosine" in out
    assert "efficientnet_b0" in out


# ---------------------------------------------------------------------------
# main() dry-run integration
# ---------------------------------------------------------------------------


def test_main_dry_run(capsys):
    """main() with --dry-run should print config summary and return without training."""
    main(["train", "--dry-run", "--max-epochs", "3"])
    out = capsys.readouterr().out
    assert "TrainConfig" in out
    assert "[dry-run]" in out
