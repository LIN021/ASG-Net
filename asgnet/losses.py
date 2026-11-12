import torch
from torch import nn
from torch.nn import functional as F


class BoundaryLaplacianLoss(nn.Module):
    """Binary cross-entropy and soft Dice on the Laplacian boundary target."""

    def __init__(self):
        super().__init__()
        kernel = torch.full((1, 1, 3, 3, 3), -1.0)
        kernel[0, 0, 1, 1, 1] = 26.0
        self.register_buffer("kernel", kernel)

    def forward(self, logits, target):
        with torch.no_grad():
            foreground = (target == 1).float()
            edges = F.conv3d(foreground, self.kernel, padding=1) > 0.1
        if logits.shape[2:] != edges.shape[2:]:
            logits = F.interpolate(logits, size=edges.shape[2:], mode="trilinear", align_corners=False)
        logits = logits.float()
        edges = edges.float()
        pred = logits.sigmoid()
        dice_loss = 1 - (2 * (pred * edges).sum() + 1) / (pred.sum() + edges.sum() + 1)
        return F.binary_cross_entropy_with_logits(logits, edges) + dice_loss


def _sample(features, max_points):
    if len(features) <= max_points:
        return features
    indices = torch.randperm(len(features), device=features.device)[:max_points]
    return features[indices]


def fr2_loss(features, target, dilate_iters=6, max_points=4096):
    """Foreground compactness and repulsion of near-boundary background."""
    foreground = target[:, 0] == 1
    voxels = features.movedim(1, -1)
    positive = voxels[foreground].float()
    if positive.numel() == 0:
        return features.new_zeros(())

    positive = F.normalize(positive, dim=1)
    prototype = F.normalize(positive.mean(dim=0), dim=0)
    positive_similarity = (_sample(positive, max_points) * prototype).sum(dim=1)
    compactness = (1 - positive_similarity).mean()

    with torch.no_grad():
        dilated = foreground.unsqueeze(1).float()
        for _ in range(dilate_iters):
            dilated = F.max_pool3d(dilated, kernel_size=3, stride=1, padding=1)
        ring = (dilated[:, 0] > 0) & ~foreground

    negative = voxels[ring]
    if negative.numel() == 0:
        return compactness
    negative = F.normalize(_sample(negative, max_points).float(), dim=1)
    repulsion = (negative * prototype).sum(dim=1).relu().mean()
    return compactness + repulsion
