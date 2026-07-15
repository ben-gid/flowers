from torch import Tensor, nn
from torchvision import models as tv_models


class FlowerClassifier(nn.Module):
    """Lean nn.Module for api

    Args:
        nn (_type_): _description_
    """
    def __init__(self, num_classes: int=102) -> None:
        super().__init__()
        # load model
        self.model = tv_models.efficientnet_b0(weights=None)
        
        # update classifier head
        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features, # type: ignore
            num_classes
        )
        
    def forward(self, x:Tensor) -> Tensor:
        return self.model(x)