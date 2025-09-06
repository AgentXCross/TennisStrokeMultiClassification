# Model Class Definition
import torch
from torch import nn

class TennisStrokeClassification(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.features = nn.Sequential(
            # [32, 3, 128, 128]
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1),

            # [32, 32, 126, 126]
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 1),

            # [32, 64, 124, 124]
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # [32, 128, 62, 62]
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # [32, 256, 31, 31]
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # [32, 512, 15, 15]
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Pool it down to 8x8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 8 * 8, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        if x.device.type == 'mps':
            x = x.to('cpu')
            x = self.adaptive_pool(x)
            x = x.to('mps')
        else:
            x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x