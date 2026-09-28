import torch
import torch.nn as nn
import torchvision


def get_loss_function(loss_name, **args):
    """
    Factory function to retrieve loss function modules.
    Args:
        loss_name (str): The name of the loss function.
        **args: Parameters to initialize the loss function.
    """
    if loss_name == "SigmoidFocalLoss":
        return FocalLossModule(**args)
    elif loss_name == "DiceLoss":
        # Make sure only accepted args are passed to DiceLossModule
        valid_args = {k: v for k, v in args.items() if k in ['reduction', 'smooth']}
        return DiceLossModule(**valid_args)
    elif loss_name == "DiceFocalLoss":
        return DiceFocalLossModule(**args)
    elif loss_name == "CombinedBCEDiceLoss":
        return CombinedBCEDiceLossModule(**args)
    else:
        # Fallback to torch.nn modules
        if hasattr(nn, loss_name):
            try:
                return getattr(nn, loss_name)(**args)
            except TypeError:
                 # If initialization fails, try without args
                 return getattr(nn, loss_name)()
        raise ValueError(f"Loss function {loss_name} not found in torch.nn or custom modules")

class DiceFocalLossModule(torch.nn.Module):
    """
    Combined Dice and Focal Loss.
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean", focal_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super(DiceFocalLossModule, self).__init__()
        self.focal_loss = FocalLossModule(alpha=alpha, gamma=gamma, reduction=reduction)
        self.dice_loss = DiceLossModule(smooth=smooth, reduction=reduction)
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        focal_l = self.focal_loss(logits, targets)
        dice_l = self.dice_loss(logits, targets)
        
        # If reduction is 'none', both return tensors of same shape
        # Focal returns per-pixel loss for 'none' -> (B, C, D, H, W)
        # Dice returns per-sample loss for 'none' -> (B)
        
        if self.focal_loss.reduction == 'none':
             # Average focal loss over spatial/channel dims to match Dice shape (B)
             if focal_l.ndim > 1:
                focal_l = focal_l.mean(dim=list(range(1, focal_l.ndim)))
        
        return self.focal_weight * focal_l + self.dice_weight * dice_l


class FocalLossModule(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLossModule, self).__init__()
        self.alpha = alpha  # optional alpha parameter for class weights
        self.gamma = gamma  # gamma parameter for focal loss
        self.reduction = reduction  # specifies reduction to apply to the output

    def forward(self, logits, targets):
        # Calculate the sigmoid focal loss using torchvision's function
        loss = torchvision.ops.sigmoid_focal_loss(
            logits,
            targets,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
        )
        return loss


class DiceLossModule(torch.nn.Module):
    def __init__(self, smooth=1e-4, reduction="mean"):
        super(DiceLossModule, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, logits, targets):
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten spatial dims: (B, C, D, H, W) -> (B, -1)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        # Vectorized intersection and union across spatial dims
        intersection = (probs * targets).sum(dim=1)
        total = probs.sum(dim=1) + targets.sum(dim=1)

        # Robust Dice: smooth avoids div-by-zero even when both are empty
        dice = (2.0 * intersection + self.smooth) / (total + self.smooth)
        loss = 1.0 - dice  # Shape: [batch_size]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss  # Shape: [batch_size]


class CombinedBCEDiceLossModule(torch.nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-4, reduction="mean"):
        super(CombinedBCEDiceLossModule, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.reduction = reduction
        
        # Initialize BCE loss with same reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        
        # Initialize Dice loss with same reduction
        self.dice_loss = DiceLossModule(smooth=smooth, reduction=reduction)

    def forward(self, logits, targets):
        # Calculate BCE loss
        bce = self.bce_loss(logits, targets)
        
        # Calculate Dice loss
        dice = self.dice_loss(logits, targets)
        
        # Combine losses
        if self.reduction == "none":
            # bce is likely (B, C, D, H, W) or similar
            # dice is (B,) from our implementation
            
            if bce.ndim > 1:
                # Flatten spatial/channel dimensions and mean
                # This handles any number of dimensions (3D, 2D, etc)
                bce = bce.view(bce.size(0), -1).mean(dim=1)
                
            # Now bce is (B,) and dice is (B,)
            combined_loss = self.bce_weight * bce + self.dice_weight * dice
        else:
            combined_loss = self.bce_weight * bce + self.dice_weight * dice
        
        return combined_loss

