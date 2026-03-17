import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision import models

sys.path.append(str(Path(__file__).parent.parent))

try:
    from .models import FlowerDataset, SubsetWithTransform
    from .training_utils import (
        change_classifier,
        get_class_weights,
        get_device,
        get_transforms,
        partial_freeze,
        split_dataset,
        train,
        val_epoch,
    )
except (ImportError, ValueError):
    from models import FlowerDataset, SubsetWithTransform
    from training_utils import (
        change_classifier,
        get_class_weights,
        get_device,
        get_transforms,
        partial_freeze,
        split_dataset,
        train,
        val_epoch,
    )


def main():
    device = get_device()
    project_root = Path(__file__).parent.parent.parent
    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
    model_weights_root = project_root / "model_weights"

    # get transforms
    train_transform, val_transform = get_transforms()
    # get oxford flowers dataset
    dataset = FlowerDataset(data_root)
    # get subsets
    train_subset, val_subset, test_subset = split_dataset(dataset, 0.7, 0.15, 0.15)

    # apply transforms
    train_subset = SubsetWithTransform(train_subset, train_transform)
    val_subset = SubsetWithTransform(val_subset, val_transform)
    test_subset = SubsetWithTransform(test_subset, val_transform)

    # get loaders
    batch_size = 64
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    class_weights = get_class_weights(dataset=dataset, device=device)

    num_epochs = 20
    model, _, loss_function = train_only_classifier(
        train_loader, val_loader, num_epochs, device, class_weights
    )

    model, _ = fine_tune(
        model,  # type: ignore
        train_loader,
        val_loader,
        num_epochs,
        device,
        class_weights,
    )
    test_loss, test_accuracy = val_epoch(model, test_loader, loss_function, device)
    print(f"{test_loss=}\n{test_accuracy=}")
    torch.save(model.state_dict(), model_weights_root / "ft_EfficientNet-B0.pth")


def train_only_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    class_weights: Tensor | None = None,
) -> tuple[nn.Module, list, nn.CrossEntropyLoss]:
    """freezes backbone and fine tunes efficientnet_b0 with an updated classifier

    Args:
        train_loader (DataLoader): training loader
        val_loader (DataLoader): validation loader
        num_epochs (int): number of epochs
        device (torch.device): device running model
        class_weights (Optional[Tensor]): class weights for uneven dataset

    Returns:
        tuple[nn.Module, list, nn.CrossEntropyLoss]: model, metrics, loss_function
    """
    model = init_model_fine_tune()
    model = change_classifier(model=model, num_classes=102)  # type: ignore
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.0005, lr=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model, metrics = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
    )
    return model, metrics, loss_function


def fine_tune(
    model: models.EfficientNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    class_weights: Tensor | None = None,
) -> tuple[models.EfficientNet, list]:
    model = partial_freeze(model, layers_to_unfreeze=3)

    optimizer = torch.optim.Adam(
        [
            {"params": model.features[6].parameters(), "lr": 1e-5},
            {"params": model.features[7].parameters(), "lr": 1e-5},
            {"params": model.features[8].parameters(), "lr": 1e-4},
            {"params": model.classifier.parameters(), "lr": 1e-3},
        ]
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    model, metrics = train(
        model=model,  # type: ignore
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=num_epochs,
        device=device,
    )
    return model, metrics


def init_model_fine_tune() -> nn.Module:
    """gets efficientnet_b0 model with weights='IMAGENET1K_V1'

    Returns:
        nn.Module: model
    """
    # init fine tuning model
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    return model


if __name__ == "__main__":
    main()
