import torch
import torchvision
import torch.nn as nn


color_channels = 3
"""
BottleneckResidualBlock 
https://arxiv.org/pdf/1609.04802 
"""
class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels=64,out_channels=64):
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(color_channels)
        self.a0 = nn.PReLU()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(color_channels)
        self.a1 = nn.PReLU()

    def forward(self, x):
        identity = x
        x = self.a0(self.bn0(self.conv0(x)))
        x = self.bn1(self.conv1(x))
        x = x + identity
        return x

"""
UpscalerBlock
https://arxiv.org/pdf/1609.04802
"""
class UpscalerBlock(nn.Module):
    def __init__(self, in_channels=256,out_channels=256):
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,padding=1, bias=False)
        self.ps = nn.PixelShuffle(2)
        self.a = nn.PReLU()

    def forward(self, x):
        return self.a(self.ps(self.conv(x)))

"""
GenerativeModel
https://arxiv.org/pdf/1609.04802
"""
class GenerativeModel(nn.module):
    def __init__(self, num_blocks=5, block_channels=64):

        self.input_conv = nn.Conv2d(block_channels, block_channels, kernel_size=9, stride=1, padding=4)
        self.inputa = nn.PReLU()

        self.blocks = nn.Sequential(*[BottleneckResidualBlock(block_channels, block_channels) for _ in range(num_blocks)])
        self.after_blocks = nn.Sequential(
            nn.Conv2d(block_channels, 4*block_channels, kernel_size=3,padding=1,),
            nn.BatchNorm2d(color_channels)
        )
        self.upscaler = UpscalerBlock()
        self.final_conv = nn.Conv2d(4*block_channels, 3, kernel_size=9,padding=4)
    
    def forward(self, x):
        input_image = x
        x = self.inputa(self.input_conv(x))
        x = self.blocks(x)
        x = self.after_blocks(x) + input_image
        x = self.upscaler(x)
        x = self.final_conv(x)
        return x
