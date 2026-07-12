import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms

from flowers.training_utils import get_transforms


class FlowerDataset(Dataset):
    """Custom torch dataset for Oxford-102 Flowers."""

    def __init__(
        self, root_dir: Path, transform: transforms.Compose | None = None
    ) -> None:
        """Initializes the FlowerDataset.

        Args:
            root_dir (Path): The root directory where data should be stored/found.
            transform (transforms.Compose | None, optional): Optional image
                transformations. Defaults to None.
        """
        self.root_dir = root_dir
        self.classes_path = root_dir / "Oxford-102_Flower_dataset_labels.txt"
        self.flowers_path = root_dir / "flowers-102"

        # Download dataset if not found
        if not self.flowers_path.exists():
            print(
                f"Dataset not found at {self.flowers_path}. \
                Downloading using torchvision..."
            )
            # This will download and extract the .mat and .jpg files into
            # root_dir/flowers-102
            # We call it once to trigger the download logic
            datasets.Flowers102(root=str(root_dir), split="train", download=True)
            datasets.Flowers102(root=str(root_dir), split="test", download=True)
            datasets.Flowers102(root=str(root_dir), split="val", download=True)

        self.transform = transform
        self.image_dir = self.flowers_path / "jpg"
        self.labels = self._load_labels()
        self.classes: list[str] = self._get_classes()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset.

        Returns:
            int: The dataset size.
        """
        return len(self.labels)

    def __getitem__(self, idx) -> tuple[Any, int]:
        """Retrieves a single sample and its label from the dataset.

        Args:
            idx (int | torch.Tensor): Index of the sample to retrieve.

        Returns:
            tuple[Any, int]: A tuple of the (possibly transformed) image and its class
                label.
        """
        if torch.is_tensor(idx):
            idx = idx.item()
        idx = int(idx)
        image = self._get_image(idx)

        # apply optional transform
        if self.transform is not None:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label

    def _load_labels(self) -> np.ndarray:
        """Gets all the labels from the dataset.

        Returns:
            np.ndarray: An array of integer labels (0-indexed).
        """
        self.labels_mat = loadmat(self.flowers_path / "imagelabels.mat")
        # subtract one from labels to make them 0 indexed
        labels: np.ndarray = self.labels_mat["labels"][0] - 1
        return labels

    def _get_image(self, idx: int) -> Image.Image:
        """Loads a single image from disk.

        Args:
            idx (int): The 0-indexed index of the image to load.

        Returns:
            Image.Image: The loaded PIL Image in RGB format.
        """
        # image index in name has 5 digits and is 1 indexed
        # (eg. image_00001.jpg)
        img_name = f"image_{idx + 1:05d}.jpg"
        img_path = os.path.join(self.image_dir, img_name)
        with Image.open(img_path) as img:
            image = img.convert("RGB")
        return image

    def _get_classes(self) -> list[str]:
        """Loads the class names from the dataset labels text file.

        Returns:
            list[str]: A list of class names.
        """
        classes_path = self.root_dir / "Oxford-102_Flower_dataset_labels.txt"
        with open(classes_path) as f:
            classes = f.read().splitlines()
        return classes


class SubsetWithTransform(Dataset):
    """Subset that applies transforms to the data.

    This is a separate class to allow different transforms between train and val/test
    subsets.
    """

    def __init__(
        self, subset: Subset, transform: transforms.Compose | None = None
    ) -> None:
        """Initializes the SubsetWithTransform.

        Args:
            subset (Subset): The original dataset subset.
            transform (transforms.Compose | None, optional): The transformations to
                apply. Defaults to None.
        """
        super().__init__()
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        """Returns the number of samples in the subset.

        Returns:
            int: The subset size.
        """
        return len(self.subset)

    def __getitem__(self, index: int) -> Any:
        """Retrieves a sample from the subset, applying the transform if available.

        Args:
            index (int): Index of the sample.

        Returns:
            Any: A tuple of the (possibly transformed) image and its label.
        """
        image, label = self.subset[index]  # type: ignore
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def seed_worker(worker_id: int) -> None:
    """Sets the seed for a DataLoader worker to ensure reproducibility.

    Args:
        worker_id (int): ID of the worker process.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class FlowerDataModule(L.LightningDataModule):
    """LightningDataModule for managing the Oxford-102 Flowers dataset."""

    def __init__(
        self,
        data_root: Path,
        batch_size: int,
        num_workers: int = 4,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        get_transforms: Callable[
            ..., tuple[transforms.Compose, transforms.Compose]
        ] = get_transforms,
        seed: int = 42,
    ) -> None:
        """Initializes the LightningDataModule.

        Args:
            data_root (Path): Directory where data is stored.
            batch_size (int): Number of samples per batch.
            num_workers (int, optional): Number of subprocesses for data loading.
                Defaults to 4.
            train_split (float, optional): Proportion of the data to use for training.
                Defaults to 0.7.
            val_split (float, optional): Proportion of the data to use for validation.
                Defaults to 0.15.
            test_split (float, optional): Proportion of the data to use for testing.
                Defaults to 0.15.
            get_transforms (Callable, optional): Function returning train and test
                transforms. Defaults to get_transforms.
            seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        super().__init__()
        self.data_root = data_root

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        self.transform_train, self.transform_test = get_transforms()

        self.seed = seed

        # populated in setup()
        self.train_set: SubsetWithTransform | None = None
        self.val_set: SubsetWithTransform | None = None
        self.test_set: SubsetWithTransform | None = None

    def prepare_data(self) -> None:
        """Downloads the dataset if it doesn't already exist."""
        FlowerDataset(self.data_root)  # triggers download only

    def setup(self, stage: str) -> None:
        """Sets up the dataset splits for training, validation, and testing.

        Args:
            stage (str): The current stage (e.g., 'fit', 'test').
        """
        full_ds = FlowerDataset(self.data_root)
        generator = torch.Generator().manual_seed(self.seed)
        train_subset, val_subset, test_subset = random_split(
            full_ds, (0.7, 0.15, 0.15), generator=generator
        )  # type: ignore

        self.train_set = SubsetWithTransform(train_subset, self.transform_train)
        self.val_set = SubsetWithTransform(train_subset, self.transform_test)
        self.test_set = SubsetWithTransform(train_subset, self.transform_test)

    def train_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the training set.

        Returns:
            DataLoader: The training DataLoader.
        """
        return DataLoader(
            self.train_set,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def val_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the validation set.

        Returns:
            DataLoader: The validation DataLoader.
        """
        return DataLoader(
            self.val_set,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )

    def test_dataloader(self) -> DataLoader:
        """Creates the DataLoader for the test set.

        Returns:
            DataLoader: The test DataLoader.
        """
        return DataLoader(
            self.test_set,  # type: ignore
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(self.seed),
        )
