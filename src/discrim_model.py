import torch
import torch.nn as nn


"""
DiscriminatorBlock 
https://arxiv.org/pdf/1609.04802 
"""
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels=64,out_channels=64, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.a = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.a(self.bn(self.conv(x)))

"""
DiscriminatorModel 
https://arxiv.org/pdf/1609.04802 
"""
class DiscriminatorModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Initial block: No BN per paper
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            DiscriminatorBlock(64, 64, stride=2),
            
            DiscriminatorBlock(64, 128, stride=1),
            DiscriminatorBlock(128, 128, stride=2),
            
            DiscriminatorBlock(128, 256, stride=1),
            DiscriminatorBlock(256, 256, stride=2),
            
            DiscriminatorBlock(256, 512, stride=1),
            DiscriminatorBlock(512, 512, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Robust to input size
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1)
            # Sigmoid is omitted here; use BCEWithLogitsLoss for stability
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
