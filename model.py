# Model Class Definition
import torch
from torch import nn

class TennisStrokeClassification(nn.Module):
    '''
    Tennis Stroke Multiclass Classification Convolutional Neural Network (CNN).
    Takes `num_classes` as input. Takes RGB inputs, thus 3 in_channels on first convolutional layer. 
    Each block of `self.features` applies a `nn.Conv2d()` filter, normalizes the batch using `nn.BatchNorm2d()`,
    applies a rectified linear unit activation layer `nn.ReLU()`, and performs max pooling using `nn.MaxPool2d()`.
    `self.adaptive_pool` forces the final feature map to have size 8x8. Finally, `self.classifier` flattens the image and
    applies a linear layer with 512 * 8 * 8 input to `num_classes` outputs.
    '''
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
        """Defines the `forward()` pass of the CNN model. Device is moved to `cpu` for `self.adaptive_pool` as `mps` does not support it."""
        x = self.features(x)
        if x.device.type == 'mps':
            x = x.to('cpu')
            x = self.adaptive_pool(x)
            x = x.to('mps')
        else:
            x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x