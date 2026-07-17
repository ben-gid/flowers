from typing import Any

import lightning as L
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler
from torchmetrics import F1Score
from torchmetrics.classification import Accuracy


class FlowerClassifier(L.LightningModule):
    """LightningModule for flower classification using a pretrained backbone."""

    def __init__(
        self,
        pretrained_model: nn.Module,
        num_classes: int,
        class_weights: torch.Tensor | None,
        lr_head: float = 1e-3,
        weight_decay: float = 1e-2,
        optimizer: type[optim.Optimizer] = optim.AdamW,
        scheduler: type[LRScheduler] = CosineAnnealingLR,
        class_names: list[str] | None = None,
        head_name: str = "classifier",
        optimizer_kwargs: dict[str, Any] | None = None,
        scheduler_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Initializes the FlowerClassifier module.

        Args:
            pretrained_model (nn.Module): The pretrained PyTorch model
                (e.g., from torchvision).
            num_classes (int): Number of target classes.
            class_weights (torch.Tensor | None): Optional tensor of weights for the
                cross-entropy loss.
            lr_head (float, optional): Learning rate for the classification head.
                Defaults to 1e-3.
            weight_decay (float, optional): Weight decay for the optimizer.
                Defaults to 1e-2.
            optimizer (type[optim.Optimizer], optional): Optimizer class to use.
                Defaults to optim.AdamW.
            scheduler (type[LRScheduler], optional): Learning rate scheduler class.
                Defaults to CosineAnnealingLR.
            class_names (list[str] | None, optional): List of class names.
                Defaults to None.
            head_name (str): Name of the classification head attribute.
                Defaults to "classifier".
            backbone_name (str | None): Name of the backbone attribute.
                Defaults to "features".
            optimizer_kwargs (dict | None): Kwargs for the optimizer.
            scheduler_kwargs (dict | None): Kwargs for the scheduler.
        """
        super().__init__()
        self.save_hyperparameters(
            "lr_head",
            "num_classes",
            "weight_decay",
            "head_name",
            ignore=["backbone", "class_weights", "pretrained_model"],
        )
        self.optimizer_kwargs = optimizer_kwargs or {"weight_decay": weight_decay}
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None else torch.ones(num_classes),
        )
        self.model = pretrained_model
        # dynamically replace head
        head = getattr(self.model, head_name)
        if isinstance(head, nn.Sequential):
            in_features = head[-1].in_features
            head[-1] = nn.Linear(in_features, num_classes)  # type: ignore
        elif isinstance(head, nn.Linear):
            in_features = head.in_features
            setattr(self.model, head_name, nn.Linear(in_features, num_classes))
        else:
            raise ValueError(f"Unsupported head type: {type(head)}")

        self.optimizer = optimizer
        self.lr_scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.class_names = class_names or [str(i) for i in range(num_classes)]

        self.train_acc = Accuracy("multiclass", num_classes=num_classes)
        self.train_f1 = F1Score("multiclass", num_classes=num_classes, average="macro")

        self.val_acc = Accuracy("multiclass", num_classes=num_classes)
        self.val_f1 = F1Score("multiclass", num_classes=num_classes, average="macro")

        self.test_acc = Accuracy("multiclass", num_classes=num_classes)
        self.test_f1 = F1Score("multiclass", num_classes=num_classes, average="macro")
        self.test_per_class_f1 = F1Score(
            task="multiclass", num_classes=num_classes, average=None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Model logits.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Executes a single training step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple of (images, labels).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc(logits, y), on_epoch=True, prog_bar=True)
        self.log("train_f1", self.train_f1(logits, y), prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """Executes a single validation step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple of (images, labels).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc(logits, y), prog_bar=True)
        self.log("val_f1", self.val_f1(logits, y), prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        """Executes a single test step.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor]): A tuple of (images, labels).
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc(logits, y), prog_bar=True)
        self.log("test_f1", self.test_f1(logits, y), prog_bar=True)
        # per-class F1 is a vector, not a scalar, so accumulate rather than self.log it
        self.test_per_class_f1.update(logits, y)

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        """Configures the optimizer and learning rate scheduler.

        Returns:
            dict[str, Any]: A dictionary containing the configured optimizer and
            scheduler.
        """
        # first only classifier params
        head = getattr(self.model, self.hparams.head_name)  # type: ignore
        optimizer = self.optimizer(
            head.parameters(),
            lr=self.hparams.lr_head,  # pyright: ignore
            **self.optimizer_kwargs,
        )

        lr_scheduler = self.lr_scheduler(optimizer, **self.scheduler_kwargs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "val_loss",
        }

