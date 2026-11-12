import torch
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

from asgnet import ASGNet_SC
from asgnet.losses import BoundaryLaplacianLoss, fr2_loss


class nnUNetTrainerASGNet(nnUNetTrainer):
    """ASG-Net trainer for binary whole-lesion segmentation and 3D plans."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        if self.is_ddp:
            raise ValueError("The ASG-Net trainer supports one device per run.")
        if self.label_manager.has_regions or set(self.label_manager.all_labels) != {0, 1}:
            raise ValueError("ASG-Net requires binary class labels 0 (background) and 1 (lesion).")
        if self.label_manager.has_ignore_label:
            raise ValueError("The ASG-Net trainer does not support an ignore label.")
        patch_size = self.configuration_manager.patch_size
        if len(patch_size) != 3 or any(size < 24 or size % 8 for size in patch_size):
            raise ValueError("ASG-Net needs a 3D patch with axes of at least 24 and divisible by 8.")
        self.enable_deep_supervision = False
        self.num_epochs = 200
        self.initial_lr = 1e-3
        self.weight_decay = 0.01

    @staticmethod
    def build_network_architecture(
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision=False,
    ):
        return ASGNet_SC(in_chans=num_input_channels, num_classes=num_output_channels)

    def set_deep_supervision_enabled(self, enabled):
        if enabled:
            raise ValueError("ASG-Net uses auxiliary losses instead of nnU-Net deep supervision.")

    def configure_optimizers(self):
        params = (p for p in self.network.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW(params, lr=self.initial_lr, weight_decay=self.weight_decay)
        return optimizer, PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)

    def _build_loss(self):
        segmentation = DC_and_CE_loss(
            {"batch_dice": self.configuration_manager.batch_dice,
             "smooth": 1e-5, "do_bg": False, "ddp": False},
            {},
            dice_class=MemoryEfficientSoftDiceLoss,
        )
        boundary = BoundaryLaplacianLoss().to(self.device)

        def loss(output, target):
            if torch.is_tensor(output):
                return segmentation(output, target)
            logits, features, encoder_logits, diversity = output
            bfa = torch.stack([boundary(level, target) for level in encoder_logits]).mean()
            alpha = min(1.0, self.current_epoch / 30)
            return segmentation(logits, target) + alpha * bfa + 0.3 * fr2_loss(features, target) + 1e-3 * diversity

        return loss
