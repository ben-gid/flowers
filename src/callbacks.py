from lightning.pytorch.callbacks import BaseFinetuning
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .classifier import FlowerClassifier


class BackboneFinetuning(BaseFinetuning):
    """
    Callback to finetune a pretrained backbone by unfreezing it at a specific epoch.
    """

    def __init__(
        self,
        unfreeze_at_epoch: int = 5,
        lr_backbone: float = 1e-5,
        new_lr_head: float | None = None,
        backbone_name: str | None = "features",
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
            backbone_name (str | None, optional): Name of the backbone module.
                Defaults to "features".
        """
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.lr_backbone = lr_backbone
        self.new_lr_head = new_lr_head
        self.backbone_name = backbone_name

    def freeze_before_training(self, pl_module: FlowerClassifier) -> None:
        """Freezes the backbone for phase 1 of training.

        Args:
            pl_module (FlowerClassifier): The LightningModule containing the model.
        """
        if self.backbone_name is not None:
            modules = getattr(pl_module.model, self.backbone_name)
        else:
            head = getattr(pl_module.model, pl_module.hparams.head_name)  # type: ignore
            modules = [m for m in pl_module.model.children() if m is not head]
        self.freeze(modules)

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
            if self.backbone_name is not None:
                modules = getattr(pl_module.model, self.backbone_name)
            else:
                head = getattr(pl_module.model, pl_module.hparams.head_name)  # type: ignore
                modules = [m for m in pl_module.model.children() if m is not head]

            self.unfreeze_and_add_param_group(
                modules=modules,
                optimizer=optimizer,
                lr=self.lr_backbone,
            )

            if self.new_lr_head is not None:
                optimizer.param_groups[0]["lr"] = self.new_lr_head

            
            scheduler = pl_module.lr_schedulers()
            if not isinstance(scheduler, ReduceLROnPlateau):
                # Patch the scheduler so it tracks the backbone
                scheduler.base_lrs.append(self.lr_backbone)  # type:ignore
