"""
ufno3d.py
=========

U-FNO 3D for wavefield prediction
---------------------------------

Input:
    [B, T, 96, 96, 96]

where
    B = num_of_src
    T = num_of_snapshot

Example:
    x.shape = [20, 40, 96, 96, 96]

This implementation treats the snapshot dimension as channels.

Compared with standard FNO3D:
--------------------------------
UFNO adds:
    1. U-Net style encoder-decoder
    2. Local convolution refinement
    3. Multi-scale feature extraction
    4. Better high-frequency recovery

Architecture:
--------------------------------
Input
  ↓
Lift Conv
  ↓
[FNO + Conv] × N
  ↓
Downsample
  ↓
[FNO + Conv]
  ↓
Upsample
  ↓
Skip Connection Fusion
  ↓
Projection

Good default settings for 96³:
--------------------------------
width       = 32
modes       = 12
n_layers    = 4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Spectral Convolution
# ======================================================================

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes_x, modes_y, modes_z):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_x = modes_x
        self.modes_y = modes_y
        self.modes_z = modes_z
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.ParameterList([
            nn.Parameter(scale * torch.randn(in_channels, out_channels, modes_x, modes_y, modes_z, dtype=torch.cfloat))
            for _ in range(4)
        ])

    @staticmethod
    def compl_mul3d(a, b):
        return torch.einsum("bixyz,ioxyz->boxyz", a, b)

    def forward(self, x):
        B = x.shape[0]
        NX, NY, NZ = x.shape[-3:]
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(B, self.out_channels, NX, NY, NZ // 2 + 1, dtype=torch.cfloat, device=x.device)
        mx, my, mz = self.modes_x, self.modes_y, self.modes_z
        out_ft[:, :, :mx, :my, :mz] = self.compl_mul3d(x_ft[:, :, :mx, :my, :mz], self.weights[0])
        out_ft[:, :, -mx:, :my, :mz] = self.compl_mul3d(x_ft[:, :, -mx:, :my, :mz], self.weights[1])
        out_ft[:, :, :mx, -my:, :mz] = self.compl_mul3d(x_ft[:, :, :mx, -my:, :mz], self.weights[2])
        out_ft[:, :, -mx:, -my:, :mz] = self.compl_mul3d(x_ft[:, :, -mx:, -my:, :mz], self.weights[3])
        x = torch.fft.irfftn(out_ft, s=(NX, NY, NZ), dim=(-3, -2, -1))
        return x


# ======================================================================
# UFNO Block
# ======================================================================

class UFNOBlock3d(nn.Module):
    def __init__(self, width, modes_x, modes_y, modes_z):
        super().__init__()
        self.spectral = SpectralConv3d(width, width, modes_x, modes_y, modes_z)
        self.skip = nn.Conv3d(width, width, 1)
        self.local = nn.Sequential(
            nn.Conv3d(width, width, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(width, width, 3, padding=1),
        )
        self.norm = nn.InstanceNorm3d(width, affine=True)

    def forward(self, x):
        spectral = self.spectral(x)
        skip = self.skip(x)
        local = self.local(x)
        x = spectral + skip + local
        x = self.norm(x)
        return F.gelu(x)


# ======================================================================
# Down Block
# ======================================================================

class DownBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU()
        )

    def forward(self, x):
        return self.down(x)


# ======================================================================
# Up Block
# ======================================================================

class UpBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2),
            nn.GELU()
        )

    def forward(self, x):
        return self.up(x)


# ======================================================================
# UFNO 3D
# ======================================================================

class UFNO3d(nn.Module):
    def __init__(self, in_channels=40, out_channels=40, width=32, modes_x=12, modes_y=12, modes_z=12):
        super().__init__()
        self.width = width
        # Lift
        self.lift = nn.Conv3d(in_channels, width, kernel_size=1)
        # Encoder
        self.block1 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        self.block2 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        self.down1 = DownBlock(width)
        # Bottleneck
        self.block3 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        self.block4 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        # Decoder
        self.up1 = UpBlock(width)
        self.block5 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        self.block6 = UFNOBlock3d(width, modes_x, modes_y, modes_z)
        # Projection
        self.proj = nn.Sequential(
            nn.Conv3d(width, 128, 1),
            nn.GELU(),
            nn.Conv3d(128, out_channels, 1)
        )

    def forward(self, x):
        """
        x: [B, T, 96, 96, 96]
        output: [B, T, 96, 96, 96]
        """
        # Lift
        x = self.lift(x)
        # Encoder
        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.down1(x1)
        # Bottleneck
        x2 = self.block3(x2)
        x2 = self.block4(x2)
        # Decoder
        x3 = self.up1(x2)
        x3 = x3 + x1  # skip connection
        x3 = self.block5(x3)
        x3 = self.block6(x3)
        # Projection
        out = self.proj(x3)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)