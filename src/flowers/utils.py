import os
from pathlib import Path
import logging
from typing import Any, Optional

from huggingface_hub import hf_hub_download
import torch
from .models import SimpleCNN
from .train import init_model

project_root = Path(__file__).parent.parent.parent
data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))

def init_logger(name: str="api") -> logging.Logger:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)
    return logger

def load_class_names(logger: Optional[logging.Logger]=None) -> list[Any]:
    # We only need get_classes, we don't need the full Dataset instance logic
    if logger is not None:
        logger.info(f"Loading Class Names from {data_root}")
    try:
        classes_path = data_root / "Oxford-102_Flower_dataset_labels.txt"
        if not classes_path.exists():
            if logger is not None:
                logger.error(f"Class names file NOT FOUND at {classes_path}. Please ensure it is present.")
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

def load_model(logger: Optional[logging.Logger]=None) -> SimpleCNN:
    model_path = Path(os.getenv("MODEL_PATH", project_root/"flower_model_weights.pth"))
    
    # check if model weights exist on disk
    if not model_path.exists():
        model_path = hf_hub_download(
            repo_id="bengid/flower-classifier",
            filename="flower_model_weights.pth"
        )
    
    if logger:
        logger.info(f"Loading CNN model from {model_path}...")
    
    # init model
    model, _ , _ = init_model()
    
    # load weights to model
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    # set model to eval mode
    model.eval()
    
    return model
    