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
        raise NotImplementedError
    
    def forward(self, x):
        raise NotImplementedError
