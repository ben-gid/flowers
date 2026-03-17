import logging
import os
from pathlib import Path
from typing import Any

import torch
import torchvision
from huggingface_hub import hf_hub_download

from .fine_tune import init_model_fine_tune
from .models import SimpleCNN
from .train_scratch import init_model
from .training_utils import change_classifier

project_root = Path(__file__).parent.parent.parent
data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
model_weights_root = project_root / "model_weights"


def init_logger(name: str = "api") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    return logger


def load_class_names(logger: logging.Logger | None = None) -> list[Any]:
    # We only need get_classes, we don't need the full Dataset instance logic
    if logger is not None:
        logger.info(f"Loading Class Names from {data_root}")
    try:
        classes_path = data_root / "Oxford-102_Flower_dataset_labels.txt"
        if not classes_path.exists():
            if logger is not None:
                logger.error(
                    f"Class names file NOT FOUND at {classes_path}. \
                        Please ensure it is present."
                )
            class_names = [f"Class {i}" for i in range(102)]
        else:
            with open(classes_path) as f:
                class_names = f.read().splitlines()
            if logger is not None:
                logger.info("Class Names loaded")
    except Exception as e:
        if logger is not None:
            logger.error(f"Failed to load class names: {e}")
        class_names = [f"Class {i}" for i in range(102)]
    return class_names


def load_scratch_model(logger: logging.Logger | None = None) -> SimpleCNN:
    scratch_model_path = Path(
        os.getenv("SCRATCH_MODEL_PATH", model_weights_root / "flower_model_weights.pth")
    )

    # check if model weights exist on disk
    if not scratch_model_path.exists():
        scratch_model_path = hf_hub_download(
            repo_id="bengid/flower-classifier",
            filename="flower_model_weights.pth",
        )

    if logger:
        logger.info(f"Loading CNN model from {scratch_model_path}...")

    # init model
    model, _, _ = init_model()

    # load weights to model
    model.load_state_dict(
        torch.load(scratch_model_path, map_location="cpu", weights_only=True)
    )
    # set model to eval mode
    model.eval()

    return model


def load_ft_model(
    logger: logging.Logger | None = None,
) -> torchvision.models.EfficientNet:
    ft_model_path = Path(
        os.getenv("FT_MODEL_PATH", project_root / "ft_EfficientNet-B0.pth")
    )

    # check if model weights exist on disk
    if not ft_model_path.exists():
        ft_model_path = hf_hub_download(
            repo_id="bengid/flower-classifier",
            filename="ft_EfficientNet-B0.pth",
        )

    if logger:
        logger.info(f"Loading EfficientNet model from {ft_model_path}...")

    # init model
    model = init_model_fine_tune()
    model = change_classifier(model=model, num_classes=102)  # type: ignore

    # load weights to model
    model.load_state_dict(
        torch.load(ft_model_path, map_location="cpu", weights_only=True)
    )
    # set model to eval mode
    model.eval()

    return model  # type: ignore
