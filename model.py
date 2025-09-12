import torch
from torch import nn

class ConvResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        
        # Convolutional Path
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut Path (nothing or 1x1 Conv)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        out = self.conv_layers(x)
        out += self.shortcut(x)   # Skip Connection
        out = self.relu(out)      # Apply ReLU() after the Shortcut
        return out
    
class TennisStrokeClassifier(nn.Module):
    def __init__(self, num_classes = 4):
        super().__init__()
        self.layer1 = ConvResidualBlock(in_channels = 3, out_channels = 64)
        self.layer2 = ConvResidualBlock(in_channels = 64, out_channels = 128, stride = 2)
        self.layer3 = ConvResidualBlock(in_channels = 128, out_channels = 256, stride = 2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2)
        self.pool2 = nn.MaxPool2d(kernel_size = 5)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x): # Start shape [32, 3, 320, 320]
        x = self.layer1(x) # [32, 64, 320, 320]
        x = self.layer2(x) # [32, 128, 160, 160]
        x = self.layer3(x) # [32, 256, 80, 80]
        x = self.pool1(x) # [32, 256, 40, 40]
        x = self.pool1(x) # [32, 256, 20, 20]
        x = self.pool1(x) # [32, 256, 10, 10]
        x = self.pool1(x) # [32, 256, 5, 5]
        x = self.pool2(x) # [32, 256, 1, 1]
        x = torch.flatten(x, 1) # 256 * 1 * 1
        x = self.fc(x) # 256 -> 4
        return x
