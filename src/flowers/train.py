import sys
import os
import copy
from pathlib import Path
from typing import Optional, Any

import torch
from torch._tensor import Tensor
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from torchvision import transforms

# Add 'src' directory to sys.path
# Path(__file__).parent is src/flowers/
sys.path.append(str(Path(__file__).parent.parent))

try:
    from .models import FlowerDataset, SubsetWithTransform, SimpleCNN
except (ImportError, ValueError):
    from models import FlowerDataset, SubsetWithTransform, SimpleCNN


def main():
    data_dir = Path(os.getcwd()) / "data"

    device = get_device()
    
    train_transform, val_transform = get_transforms()
    
    dataset = FlowerDataset(data_dir)
    train_dataset, val_dataset, test_dataset = split_dataset(
        dataset, 0.7, 0.15, 0.15)
    
    train_dataset = SubsetWithTransform(train_dataset, train_transform)
    val_dataset = SubsetWithTransform(val_dataset, val_transform)
    test_dataset = SubsetWithTransform(test_dataset, val_transform)
    
    batch_size = 100
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    num_epochs = 20
    
    model, loss_function, optimizer, scheduler = init_model()
    
    model, metrics = train(model, train_loader, val_loader, loss_function, optimizer, scheduler, num_epochs, device)
    
    test_loss, test_accuracy = val_epoch(model, test_loader, loss_function, device)
    print(f"{test_loss=}\n{test_accuracy=}\n\nTrain Metrics:{metrics}")
    torch.save(model.state_dict(), "flower_model_weights.pth")

def init_model():
    # taken from notebook
    # num_classes = len(dataset.classes)
    # single_img_shape = train_dataset[0][0].shape
    num_classes = 102
    single_img_shape = torch.Size([3, 224, 224])
    model = SimpleCNN(single_img_shape=single_img_shape, num_classes=num_classes)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.0005)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    return model, loss_function, optimizer, scheduler

    
def get_device() -> torch.device:
   # use cuda gpu if available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    return device  

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    # precalculated mean and std from notebook 
    mean = torch.tensor([0.4727, 0.3996, 0.3193])
    std = torch.tensor([0.2965, 0.2471, 0.2812])
    
    uniformize_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor()
    ])
    
    train_transform = transforms.Compose([
        uniformize_transform,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=mean, std=std)
    ])

    # val and test; no augmentation
    val_transform = transforms.Compose([
        uniformize_transform,
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transform, val_transform

def split_dataset(dataset: Dataset, train_fraction: float, val_fraction: float, 
                  test_fraction: float) -> tuple[Subset, Subset, Subset]:
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_fraction, val_fraction, test_fraction]
    )
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(
    model: nn.Module, 
    train_loader: DataLoader, 
    loss_function: nn.CrossEntropyLoss, 
    optimizer:optim.Adam, 
    device: torch.device
):
    model.train()
    running_loss = 0.
    
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
    device: torch.device
):
    model.eval()
    running_val_loss = 0.
    correct = 0
    total = 0
    
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        
        val_loss = loss_function(outputs, labels)
        
        running_val_loss += val_loss.item() * images.size(0)
        
        predicted  = outputs.argmax(dim=1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_val_loss = running_val_loss / len(val_loader)
    epoch_accuracy: float = 100. * correct / total
    
    return epoch_val_loss, epoch_accuracy

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_function: nn.CrossEntropyLoss,
    optimizer: optim.Adam,
    scheduler: optim.lr_scheduler.LRScheduler,
    num_epochs: int,
    device: torch.device
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
        epoch_val_loss, epoch_accuracy = val_epoch(model, val_loader, loss_function, device)
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_accuracy)
        
        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Update the learning rate based on validation accuracy
        scheduler.step(int(epoch_accuracy))
        
        # Check if the current model is the best one so far
        if epoch_accuracy > best_val_accuracy:
            best_val_accuracy = epoch_accuracy
            best_epoch = epoch + 1
            # Save the state of the best model in memory
            best_model_state = copy.deepcopy(model.state_dict())
            
    print("--- Finished Training ---")
    
    # Load the best model weights before returning
    if best_model_state:
        print(f"\n--- Returning best model with {best_val_accuracy:.2f}% validation accuracy, achieved at epoch {best_epoch} ---")
        model.load_state_dict(best_model_state)
    
    # Consolidate all metrics into a single list
    metrics = [train_losses, val_losses, val_accuracies]
    
    # Return the trained model and the collected metrics
    return model, metrics

if __name__ == "__main__":
    main()