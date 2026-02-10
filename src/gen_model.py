import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GeneratorModel']
UPSCALE_FACTOR = 2

class BottleneckResidualBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        # Use out_channels for BN, not color_channels (3)
        self.conv0 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels) 
        self.a0 = nn.PReLU()

        self.conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        x = self.a0(self.bn0(self.conv0(x)))
        x = self.bn1(self.conv1(x))
        return x + identity

class UpscalerBlock(nn.Module):
    def __init__(self, in_channels=64, upscale_factor=2):
        super().__init__()
        # PixelShuffle(r) requires in_channels = out_channels * (r^2)
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor**2), kernel_size=3, padding=1)
        self.ps = nn.PixelShuffle(upscale_factor)
        self.a = nn.PReLU()

    def forward(self, x):
        return self.a(self.ps(self.conv(x)))

class GeneratorModel(nn.Module):
    def __init__(self, num_blocks=16, block_channels=64):
        super().__init__()

        # Initial Convolution
        self.input_conv = nn.Conv2d(3, block_channels, kernel_size=9, stride=1, padding=4)
        self.inputa = nn.PReLU()

        # Residual Blocks
        self.blocks = nn.Sequential(*[BottleneckResidualBlock(block_channels, block_channels) for _ in range(num_blocks)])
        
        # Post-residual block (element-wise sum is with the output of input_conv)
        self.after_blocks = nn.Sequential(
            nn.Conv2d(block_channels, block_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(block_channels)
        )

        # Upsampling 
        self.upscaler = nn.Sequential(
            UpscalerBlock(block_channels, upscale_factor=UPSCALE_FACTOR)
        )
        
        # Final Layer
        self.final_conv = nn.Conv2d(block_channels, 3, kernel_size=9, padding=4)
    
    def forward(self, x):
            bicubic_base = F.interpolate(x, scale_factor=UPSCALE_FACTOR, mode='bicubic', align_corners=False)
            
            x = self.inputa(self.input_conv(x))
            initial_feat = x
            
            x = self.blocks(x)
            x = self.after_blocks(x)
            x = x + initial_feat 
            
            # 3. Upscale learned features
            x = self.upscaler(x)
            residual = self.final_conv(x)
            
            # 4. Final summation
            return bicubic_base + residual