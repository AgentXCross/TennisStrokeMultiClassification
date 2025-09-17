import torch
from torch import nn
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def return_model():
    model = efficientnet_b0(weights = EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)
    return model