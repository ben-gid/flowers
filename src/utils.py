import logging
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torchvision import transforms


def get_device() -> torch.device:
    """Gets the available PyTorch device (CUDA or CPU).

    Returns:
        torch.device: The available device.
    """
    # use cuda gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return device


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Gets the train and validation/test transforms for the dataset.

    Returns:
        tuple[transforms.Compose, transforms.Compose]: A tuple containing the
            training transform and the validation/test transform.
    """
    # precalculated mean and std from notebook
    mean = torch.tensor([0.4727, 0.3996, 0.3193])
    std = torch.tensor([0.2965, 0.2471, 0.2812])

    uniformize_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_transform = transforms.Compose(
        [
            uniformize_transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    # val and test; no augmentation
    val_transform = transforms.Compose(
        [uniformize_transform, transforms.Normalize(mean=mean, std=std)]
    )

    return train_transform, val_transform


project_root = Path(__file__).parent.parent.parent
data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
model_weights_root = project_root / "model_weights"


def init_logger(name: str = "api") -> logging.Logger:
    """Initializes and returns a logger instance.

    Args:
        name (str, optional): Name of the logger. Defaults to "api".

    Returns:
        logging.Logger: The configured logger instance.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)
    return logger


def get_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Computes class weight distribution for an uneven dataset.

    Args:
        labels (np.ndarray): Array of integer labels for the dataset.

    Returns:
        torch.Tensor: Computed class weights as a 1D tensor.
    """
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    return class_weights
