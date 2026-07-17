import sys
from logging import Logger
from pathlib import Path

from torch import nn
from torchvision import transforms

from ..utils.helpers import load_class_names, load_model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from src.utils import get_transforms


class AppState:
    classifier: nn.Module | None = None
    class_names: list[str] | None = None
    transform: transforms.Compose | None = None
    
    def load(self, logger: Logger) -> None:
        """load classifier, transform and class names into app state memory

        Args:
            logger (Logger): api logger
        """
        logger.info("Loading model...")
        self.classifier = load_model(logger)
        logger.info("Model loaded")
        
        self.class_names = load_class_names(logger)
        _, val_transform = get_transforms()
        self.transform = val_transform
        
    def clear(self, logger) -> None:
        """clear state memory
        """
        self.classifier = None
        self.class_names = None
        self.transform = None
        logger.info("Model, Transform, and Class Names cleared from memory.")
        
state = AppState()