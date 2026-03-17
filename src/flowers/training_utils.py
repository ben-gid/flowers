import copy

import numpy as np
import torch
from sklearn.utils import compute_class_weight
from torch import nn, optim
from torch._tensor import Tensor
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from torchvision import models, transforms

from flowers.models import FlowerDataset


def get_device() -> torch.device:
    # use cuda gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    return device


def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """transforms to prepare image for training

    Returns:
        tuple[transforms.Compose, transforms.Compose]: train_transform, val_transform
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


def split_dataset(
    dataset: Dataset, train_fraction: float, val_fraction: float, test_fraction: float
) -> tuple[Subset, Subset, Subset]:
    """splits the dataset into training, validation and testing Subsets

    Args:
        dataset (Dataset): original dataset
        train_fraction (float): fraction of original dataset to be training subset
        val_fraction (float): fraction of original dataset to be validation subset
        test_fraction (float): fraction of original dataset to be testing subset

    Returns:
        tuple[Subset, Subset, Subset]: train_subset, val_subset, test_subset
    """
    train_subset, val_subset, test_subset = random_split(
        dataset, [train_fraction, val_fraction, test_fraction]
    )

    return train_subset, val_subset, test_subset


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    device: torch.device,
):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        # move data to device
        images, labels = images.to(device), labels.to(device)
        # reset grads
        optimizer.zero_grad()
        # compute predictions
        outputs = model(images)
        # compute loss
        loss = loss_function(outputs, labels)
        # compute grads
        loss.backward()
        # update model weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss


def val_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    device: torch.device,
):
    model.eval()
    running_val_loss = 0.0
    correct = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)

        val_loss = loss_function(outputs, labels)

        running_val_loss += val_loss.item() * images.size(0)

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_accuracy: float = 100.0 * correct / total

    return epoch_val_loss, epoch_accuracy


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    scheduler: optim.lr_scheduler.LRScheduler | None,
    num_epochs: int,
    device: torch.device,
):
    # Move the model to the specified device (CPU or GPU)
    model.to(device)

    # Initialize variables to track the best performing model
    best_val_accuracy = 0.0
    best_model_state = None
    best_epoch = 0

    # Initialize lists to store training and validation metrics
    train_losses, val_losses, val_accuracies = [], [], []

    print("--- Training Started ---")

    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        # Perform one epoch of training
        epoch_loss = train_epoch(model, train_loader, loss_function, optimizer, device)
        train_losses.append(epoch_loss)

        # Perform one epoch of validation
        epoch_val_loss, epoch_accuracy = val_epoch(
            model, val_loader, loss_function, device
        )
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)

        # Print the metrics for the current epoch
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, \
                Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%"
        )

        # Update the learning rate
        if scheduler is not None:
            scheduler.step()

        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())

    print("--- Finished Training ---")

    # Load the best model weights before returning
    if best_model_state:
        print(
            f"\n--- Returning best model with {best_val_accuracy:.2f}% validation \
                accuracy, achieved at epoch {best_epoch} ---"
        )
        model.load_state_dict(best_model_state)

    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]

    # Return the trained model and the collected metrics
    return model, metrics


# ============= from fine_tune.ipynb ===========


def get_class_weights(dataset: FlowerDataset, device: torch.device) -> Tensor:
    """computes class weight distribution for uneven dataset

    Args:
        dataset (FlowerDataset): custom oxford flower 102 dataset
        device (torch.device): device

    Returns:
        Tensor: class weights
    """
    labels = dataset.labels
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=labels
    )
    # convert to tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return class_weights


def change_classifier(model: models.EfficientNet, num_classes: int) -> nn.Module:
    """freezes all parameters of the model,
    changes the classifiers out_features to num classes
    unfreezes the classifier for fine tuning

    Args:
        model (models.EfficientNet): model to update
        num_classes (int): number of classes in dataset

    Returns:
        nn.Module: updated model
    """
    for param in model.parameters():
        param.requires_grad = False

        old_classifier = model.classifier

        new_classifier_lin_lay = nn.Linear(old_classifier[1].in_features, num_classes)  # type: ignore

        model.classifier[1] = new_classifier_lin_lay
        for param in model.classifier.parameters():
            param.requires_grad = True
    return model


def partial_freeze(
    model: models.EfficientNet, layers_to_unfreeze: int
) -> models.EfficientNet:
    """partially freeze layers, classifier will be unfrozen

    Args:
        model (models.EfficientNet): model to partially freeze
        layers_to_unfreeze (int): number of last layers to unfreeze

    Returns:
        nn.Module: updated model
    """
    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    conv_layers = model.features

    # unfreeze layers in reverse
    for i in range(layers_to_unfreeze):
        layer_to_unfreeze = conv_layers[-(i + 1)]

        for param in layer_to_unfreeze.parameters():
            param.requires_grad = True

    # unfreeze classifier
    for param in model.classifier.parameters():
        param.requires_grad = True

    # verify
    for idx, feat in enumerate(model.features):
        if any(param.requires_grad for param in feat.parameters()):
            print(f"layer {idx} requires grad")
    if all(param.requires_grad for param in model.classifier.parameters()):
        print("classifier requires grad")

    return model
