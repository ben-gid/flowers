import os
from logging import Logger
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from torch import nn
from torchvision import models as tv_models

project_root = Path(__file__).parent.parent.parent.parent.parent
data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))

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
) -> tv_models.EfficientNet:
    """loads finetuned flower classifier weights from huggingface to lean 
        FlowerClassifier

    Args:
        logger (Logger | None, optional): api logger. Defaults to None.

    Returns:
        FlowerClassifier: fine tuned classifier
    """
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
    model = tv_models.efficientnet_b0(weights=None)
    
    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features, # type: ignore
        102
    )

    # load weights to model
    model.load_state_dict(
        torch.load(ft_model_path, map_location="cpu", weights_only=True)
    )
    # set model to eval mode
    model.eval()

    return model