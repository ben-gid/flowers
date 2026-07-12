from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim import Optimizer

from classifier import FlowerClassifier


class BackboneFinetuning(BaseFinetuning):
    """
    Callback to finetune a pretrained backbone by unfreezing it at a specific epoch.
    """

    def __init__(
        self,
        unfreeze_at_epoch: int = 5,
        lr_backbone: float = 1e-5,
        new_lr_head: float | None = None,
    ) -> None:
        """Initializes the BackboneFinetuning callback.

        Args:
            unfreeze_at_epoch (int, optional): Epoch at which to unfreeze the backbone.
                Defaults to 5.
            lr_backbone (float, optional): Learning rate for the unfreezed backbone.
                Defaults to 1e-5.
            new_lr_head (float | None, optional):
                Optional new learning rate for the classification head.
                Defaults to None.
        """
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.lr_backbone = lr_backbone
        self.new_lr_head = new_lr_head

    def freeze_before_training(self, pl_module: FlowerClassifier) -> None:
        """Freezes the backbone for phase 1 of training.

        Args:
            pl_module (FlowerClassifier): The LightningModule containing the model.
        """
        self.freeze(pl_module.model.features)

    def finetune_function(
        self, pl_module: FlowerClassifier, epoch: int, optimizer: Optimizer
    ) -> None:
        """Unfreezes the backbone at the specified epoch and updates the optimizer.

        Args:
            pl_module (FlowerClassifier): The LightningModule containing the model.
            epoch (int): The current epoch index.
            optimizer (Optimizer): The optimizer used for training.
        """
        if epoch == self.unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.model.features,
                optimizer=optimizer,
                lr=self.lr_backbone,
            )

            if self.new_lr_head is not None:
                optimizer.param_groups[0]["lr"] = self.new_lr_head

            # Patch the scheduler so it tracks the backbone
            scheduler = pl_module.lr_schedulers()
            scheduler.base_lrs.append(self.lr_backbone)  # type:ignore
