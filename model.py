import torch
from torch import nn

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        # Convolutional path
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut path (identity/nothing or 1x1 conv to match shapes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out += self.shortcut(x)   # skip connection
        out = self.relu(out)      # apply ReLU() after the shortcut
        return out
    
class TennisStrokeClassifier(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.layer1 = ConvResidualBlock(in_channels = 3, out_channels = 64)
        self.layer2 = ConvResidualBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.layer3 = ConvResidualBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
