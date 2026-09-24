from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

class ConvBlock(nn.Module):
    """Two 3x3 convolutions used throughout the network."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention Gate to suppress irrelevant background features.
    
    It takes the gating signal (from the decoder) and the skip connection
    (from the encoder) and computes an attention coefficient (alpha) applied
    to the skip connection.
    """
    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        # W_g transforms the gating signal
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )
        
        # W_x transforms the skip connection
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(f_int)
        )

        # Psi computes the attention weights
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # Transform gating signal and skip connection
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # Resize g1 to match x1 if necessary (should match in standard U-Net)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
            
        # Add them and apply ReLU
        psi = self.relu(g1 + x1)
        
        # Compute attention map (alpha)
        alpha = self.psi(psi)
        
        # Apply attention to the original skip connection
        return x * alpha


class AttentionUNet2D(nn.Module):
    """Attention U-Net baseline for BraTS-GLI slice segmentation.
    
    Drop-in replacement for the standard UNet2D. Includes Attention Gates
    on all skip connections.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        # Encoder
        self.encoder1 = ConvBlock(in_channels, channels[0])
        self.encoder2 = ConvBlock(channels[0], channels[1])
        self.encoder3 = ConvBlock(channels[1], channels[2])
        self.encoder4 = ConvBlock(channels[2], channels[3])
        self.bottleneck = ConvBlock(channels[3], channels[4])

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder & Attention Gates
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.att4 = AttentionGate(f_g=channels[3], f_l=channels[3], f_int=channels[2])
        self.decoder4 = ConvBlock(channels[4], channels[3])

        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.att3 = AttentionGate(f_g=channels[2], f_l=channels[2], f_int=channels[1])
        self.decoder3 = ConvBlock(channels[3], channels[2])

        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.att2 = AttentionGate(f_g=channels[1], f_l=channels[1], f_int=channels[0])
        self.decoder2 = ConvBlock(channels[2], channels[1])

        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.att1 = AttentionGate(f_g=channels[0], f_l=channels[0], f_int=channels[0] // 2)
        self.decoder1 = ConvBlock(channels[1], channels[0])

        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        # Decoder 4
        g4 = self.upconv4(bottleneck)
        x4 = self.att4(g=g4, x=enc4)
        dec4 = torch.cat((g4, x4), dim=1)
        dec4 = self.decoder4(dec4)

        # Decoder 3
        g3 = self.upconv3(dec4)
        x3 = self.att3(g=g3, x=enc3)
        dec3 = torch.cat((g3, x3), dim=1)
        dec3 = self.decoder3(dec3)

        # Decoder 2
        g2 = self.upconv2(dec3)
        x2 = self.att2(g=g2, x=enc2)
        dec2 = torch.cat((g2, x2), dim=1)
        dec2 = self.decoder2(dec2)

        # Decoder 1
        g1 = self.upconv1(dec2)
        x1 = self.att1(g=g1, x=enc1)
        dec1 = torch.cat((g1, x1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.classifier(dec1)

__all__ = ["ConvBlock", "AttentionGate", "AttentionUNet2D"]