import os
from collections.abc import Callable
from logging import Logger
from pathlib import Path

from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torchvision import models as tv_models

project_root = Path(__file__).parent.parent.parent.parent.parent
data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))

# name -> (architecture builder, head attribute name, HF repo id, weights filename)
MODEL_REGISTRY: dict[str, tuple[Callable[[], nn.Module], str, str, str]] = {
    "efficientnet_v2_s": (
        lambda: tv_models.efficientnet_v2_s(weights=None),
        "classifier",
        "bengid/efficientnetv2-s-flower-classifier",
        "efficientnetv2-s-flower-classifier.safetensors",
    ),
    "vit_b_16": (
        lambda: tv_models.vit_b_16(weights=None),
        "heads",
        "bengid/vit-flower-classifier",
        "vit-flower-classifier.safetensors",
    ),
    "convnext_tiny": (
        lambda: tv_models.convnext_tiny(weights=None),
        "classifier",
        "bengid/convnext-tiny-flower-classifier",
        "convnext-tiny-flower-classifier.safetensors",
    ),
}
DEFAULT_MODEL_NAME = "convnext_tiny"


def load_class_names(logger: Logger | None = None) -> list[str]:
    """loads class names from data/"Oxford-102_Flower_dataset_labels.txt"
    if its available else create a list of "Class i"

    Args:
        logger (Logger | None, optional): api logger. Defaults to None.

    Returns:
        list[str]: list of class names
    """
    # We only need get_classes, we don't need the full Dataset instance logic
    if logger is not None:
        logger.info(f"Loading Class Names from {data_root}")
    try:
        classes_path = data_root / "Oxford-102_Flower_dataset_labels.txt"
        if not classes_path.exists():
            if logger is not None:
                logger.error(
                    f"Class names file NOT FOUND at {classes_path}."
                    "Please ensure it is present."
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


def load_model(
    logger: Logger | None = None,
) -> nn.Module:
    """loads finetuned flower classifier weights from huggingface.

    Model architecture is selected via the MODEL_NAME env var (one of
    MODEL_REGISTRY, defaults to DEFAULT_MODEL_NAME).

    Args:
        logger (Logger | None, optional): api logger. Defaults to None.

    Returns:
        nn.Module: fine tuned classifier
    """
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown MODEL_NAME '{model_name}'. Choose from: {list(MODEL_REGISTRY)}"
        )
    build_model, head_name, repo_id, hf_filename = MODEL_REGISTRY[model_name]

    if logger:
        logger.info(f"Downloading {model_name} weights from {repo_id}...")

    # fetch weights from HF Hub (cached locally by hf_hub_download after first call)
    ft_model_path = hf_hub_download(repo_id=repo_id, filename=hf_filename)

    # init model and swap head to the 102-class layer the weights expect
    model = build_model()
    head = getattr(model, head_name)
    if isinstance(head, nn.Sequential):
        head[-1] = nn.Linear(head[-1].in_features, 102)  # type: ignore
    else:
        setattr(model, head_name, nn.Linear(head.in_features, 102))  # type: ignore

    # load weights to model
    model.load_state_dict(load_file(ft_model_path, device="cpu"))
    # set model to eval mode
    model.eval()

    return model
