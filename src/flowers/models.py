from pathlib import Path
import os
from PIL import Image

from typing import Optional, Any

from scipy.io import loadmat

import torch
from torch._tensor import Tensor
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from torchvision import transforms

# create custom flower dataset
class FlowerDataset(Dataset):
    def __init__(self, root_dir: Path, transform: Optional[transforms.Compose]=None):
        self.root_dir = root_dir
        self.classes_path = root_dir/"Oxford-102_Flower_dataset_labels.txt"
        self.flowers_path = root_dir/"flowers-102"
        
        # Download dataset if not found
        if not self.flowers_path.exists():
            print(f"Dataset not found at {self.flowers_path}. Downloading using torchvision...")
            from torchvision import datasets
            # This will download and extract the .mat and .jpg files into root_dir/flowers-102
            # We call it once to trigger the download logic
            datasets.Flowers102(root=str(root_dir), split="train", download=True)
            datasets.Flowers102(root=str(root_dir), split="test", download=True)
            datasets.Flowers102(root=str(root_dir), split="val", download=True)

        self.transform = transform
        self.image_dir = self.flowers_path/"jpg"
        self.labels = self.load_labels()
        self.classes = self.get_classes()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx) -> tuple[Any, int]:
        if torch.is_tensor(idx):
            idx = idx.item()
        image = self.get_image(idx)
        
        #apply optional transform
        if self.transform is not None:
            image = self.transform(image)
        
        label = self.labels[idx]
        
        return image, label
    
    def load_labels(self):
        self.labels_mat = loadmat(self.flowers_path/"imagelabels.mat")
        # subtract one from labels to make them 0 indexed
        labels = self.labels_mat["labels"][0] - 1
        return labels
    
    def get_image(self, idx):
        # image index in name has 5 digits and is 1 indexed 
        # (eg. image_00001.jpg)
        img_name = f"image_{idx + 1:05d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        return image

    def get_classes(self):
        classes_path = self.root_dir/"Oxford-102_Flower_dataset_labels.txt"
        with open(classes_path) as f:
            classes = f.read().splitlines()
        return classes
    
class SubsetWithTransform(Dataset):
    """Subset that applies transforms to the data"""
    def __init__(self, subset: Subset, transform: Optional[transforms.Compose] = None) -> None:
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index) -> Any:
        image, label = self.subset[index]  # type: ignore
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    
class CNNBlock(nn.Module):
    """a simple CNN Block with 4 layers:
    conv2d: for learning filters
    batchnorm2d: to stabilize and accelerate training
    relu: to make non-linear
    maxpool2d: to downsample the feature map
    """
    def __init__(self,in_channels: int, out_channels: int, kernel_size: int=3,
                 padding:int=1, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x) -> Tensor:
        return self.block(x)
    
class SimpleCNN(nn.Module):
    def __init__(self, single_img_shape: torch.Size, num_classes: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        
        self.features = nn.Sequential(
            CNNBlock(in_channels=single_img_shape[0], out_channels=32),
            CNNBlock(in_channels=32, out_channels=64),
            CNNBlock(in_channels=64, out_channels=128),
            CNNBlock(in_channels=128, out_channels=256),
            CNNBlock(in_channels=256, out_channels=512),
            CNNBlock(in_channels=512, out_channels=1024)
        )
        
        # height and width are reduced to half 6 times
        # use floor division because that is what pytorch pooling uses 
        # to prevent mismatched shapes if image cant be divided by 2**6
        reduced_h = single_img_shape[1] // 2**6
        reduced_w = single_img_shape[2] // 2**6
        
        # input to first fully connected layer
        self.fcl_in = reduced_h * reduced_w * 1024
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fcl_in, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
            