# Training Infrastructure

This directory contains the training infrastructure for the Flowers classification project. Most of the code here, including the GUI and CLI, was built with the help of **Antigravity IDE**.

## CLI (`cli.py`)
The command-line interface lets you run training jobs with custom hyperparameters. It reads all default values and available options directly from `config.py`.

```bash
# Run with defaults
uv run train/cli/custom.py train

# Run with custom parameters
uv run train/cli/custom.py train --max-epochs 50 --batch-size 32 --optimizer adamw

# See all available models, optimizers, etc.
uv run train/cli/custom.py choices
```

## GUI (`gui/app.py`)
A frosted glass web dashboard that provides a user-friendly visual wrapper around the CLI. It allows you to launch training runs from your browser and view live metrics/results pulled directly from the MLflow database.

```bash
# Launch the web UI
uv run train/gui/app.py
```
Then open **http://localhost:8000** in your browser.

## Sweep (`sweep.py`)
Runs 20 curated hyperparameter configurations ordered into three tiers: A (max accuracy), B (balanced), and C (fast/efficient). Useful for finding the best model without manually running `cli.py`.

```bash
# Run all configurations
uv run train/cli/sweep.py

# Run a specific tier (A, B, or C)
uv run train/cli/sweep.py --tier A

# Dry run to see what will run
uv run train/cli/sweep.py --dry-run
```
