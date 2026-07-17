import os
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .models import FlowerDataset, SimpleCNN, SubsetWithTransform
from .training_utils import (
    get_device,
    get_transforms,
    split_dataset,
    train,
    val_epoch,
)


def main():
    project_root = Path(__file__).parent.parent.parent
    data_root = Path(os.getenv("DATA_ROOT", project_root / "data"))
    model_weights_root = project_root / "model_weights"

    device = get_device()

    train_transform, val_transform = get_transforms()

    dataset = FlowerDataset(data_root)
    train_dataset, val_dataset, test_dataset = split_dataset(dataset, 0.7, 0.15, 0.15)

    train_dataset = SubsetWithTransform(train_dataset, train_transform)
    val_dataset = SubsetWithTransform(val_dataset, val_transform)
    test_dataset = SubsetWithTransform(test_dataset, val_transform)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model, loss_function, optimizer = init_model()

    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=0.0002
    )

    model, metrics = train(
        model,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        scheduler,
        num_epochs,
        device,
    )

    test_loss, test_accuracy = val_epoch(model, test_loader, loss_function, device)
    print(f"{test_loss=}\n{test_accuracy=}\n\nTrain Metrics:{metrics}")
    torch.save(model.state_dict(), model_weights_root / "flower_model_weights.pth")


def init_model() -> tuple[SimpleCNN, nn.CrossEntropyLoss, optim.Adam]:
    """initialize simple cnn to train from scratch

    Returns:
        tuple[SimpleCNN, nn.CrossEntropyLoss, optim.Adam]:
        model, loss_function, optimizer
    """
    # taken from notebook
    # num_classes = len(dataset.classes)
    # single_img_shape = train_dataset[0][0].shape
    num_classes = 102
    single_img_shape = torch.Size([3, 224, 224])
    model = SimpleCNN(single_img_shape=single_img_shape, num_classes=num_classes)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), weight_decay=0.0005)

    return model, loss_function, optimizer


if __name__ == "__main__":
    main()
