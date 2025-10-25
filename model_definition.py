import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
from typing import List

class FocalLoss(nn.Module):
    """
    Multiclass Focal Loss implementation.
    Works with softmax logits (for CrossEntropy-style classification).
    """
    def __init__(self, alpha: List[float], gamma: float = 2.0):
        """
        Args:
            alpha: List of weighting factor for class imbalance
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = torch.tensor(alpha, dtype = torch.float32)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: model outputs before softmax, shape (N, C)
            targets: ground-truth class indices
        """
        # Logits to probabilities
        probs = F.softmax(logits, dim = 1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Select alpha value
        alpha_t = self.alpha.to(logits.device)[targets]

        # Compute focal loss
        focal_term = (1 - pt) ** self.gamma
        loss = -alpha_t * focal_term * torch.log(pt + 1e-8)

        return loss.mean()
    
# Functions to create mobels with pretrained weights
def create_mobilenet_v3(num_classes: int, weights = "IMAGENET1K_V1"):
    model = models.mobilenet_v3_small(weights = weights)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model

def create_resnet18(num_classes: int, weights = "IMAGENET1K_V1"):
    model = models.resnet18(weights = weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def create_convnext_tiny(num_classes: int, weights = "IMAGENET1K_V1"):
    model = models.convnext_tiny(weights = weights)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model