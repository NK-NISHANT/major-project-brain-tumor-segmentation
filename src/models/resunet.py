"""Deep Residual 2D U-Net (ResUNet) for multi-channel brain tumor segmentation.

B.Tech Major Project:
Brain Tumor Segmentation Using 2D CNNs and a Reliability-Aware AI Agent
Workspace: D:\MajorProject
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _get_group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    """Construct GroupNorm ensuring num_channels is divisible by num_groups.

    Finds the largest divisor of channels that is <= preferred_groups.
    For standard channel progressions [32, 64, 128, 256, 512], num_groups is 8.
    For input channels (e.g. 4), num_groups is 4 (or 2).
    """
    groups = min(preferred_groups, channels)
    while channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class PreActResidualBlock(nn.Module):
    """Full pre-activation residual convolutional block with Group Normalization.

    Topology:
        Main path F(x):
            Norm1 -> ReLU -> Conv3x3 (stride) -> Norm2 -> ReLU -> Conv3x3 (stride=1)
        Shortcut path:
            If stride != 1 or in_channels != out_channels:
                Conv1x1 (stride, bias=False) -> Norm
            Else:
                Identity
        Output:
            F(x) + shortcut(x)

    Design rationale:
        1. Pre-activation leaves the residual identity addition unobstructed by
           a trailing non-linearity, preserving the linear gradient highway.
        2. Group Normalization computes statistics per instance across channel
           groups, making training and evaluation dynamics invariant to batch size
           (critical for small batch size B=2).
        3. All convolutions use bias=False because normalization layers track
           affine scale and shift parameters directly.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        self.norm1 = _get_group_norm(in_channels, num_groups)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.norm2 = _get_group_norm(out_channels, num_groups)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                _get_group_norm(out_channels, num_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out + residual


class ResUNet2D(nn.Module):
    """5-Stage Deep Residual 2D U-Net for BraTS-GLI slice segmentation.

    The input has 4 channels corresponding to the four co-registered MRI
    modalities: t1n, t1c, t2f, and t2w. The output has 4 unnormalized logits
    for classes 0 (background), 1 (necrotic core), 2 (peritumoral edema),
    and 3 (enhancing tumor).

    Architectural properties:
        - Level 1 operates at stride 1 (native 240x240 resolution) to preserve
          small enhancing tumor (ET) boundary structures.
        - Residual units use full pre-activation and Group Normalization.
        - Skip connections use concatenation followed by residual projection.
        - Decoder upconvolutions handle non-divisible spatial dimensions via
          adaptive bilinear interpolation alignment when necessary.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
        num_groups: int = 8,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        channels = [
            base_channels,        # Level 1: 32
            base_channels * 2,    # Level 2: 64
            base_channels * 4,    # Level 3: 128
            base_channels * 8,    # Level 4: 256
            base_channels * 16,   # Bottleneck Level 5: 512
        ]

        # Encoder path (Level 1 stride=1 preserves full 240x240 spatial detail)
        self.encoder1 = PreActResidualBlock(in_channels, channels[0], stride=1, num_groups=num_groups)
        self.encoder2 = PreActResidualBlock(channels[0], channels[1], stride=2, num_groups=num_groups)
        self.encoder3 = PreActResidualBlock(channels[1], channels[2], stride=2, num_groups=num_groups)
        self.encoder4 = PreActResidualBlock(channels[2], channels[3], stride=2, num_groups=num_groups)
        self.bottleneck = PreActResidualBlock(channels[3], channels[4], stride=2, num_groups=num_groups)

        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.decoder4 = PreActResidualBlock(channels[4], channels[3], stride=1, num_groups=num_groups)

        self.upconv3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.decoder3 = PreActResidualBlock(channels[3], channels[2], stride=1, num_groups=num_groups)

        self.upconv2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.decoder2 = PreActResidualBlock(channels[2], channels[1], stride=1, num_groups=num_groups)

        self.upconv1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.decoder1 = PreActResidualBlock(channels[1], channels[0], stride=1, num_groups=num_groups)

        # 1x1 Conv classification head returning unnormalized logits
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[2:]
        # Encoder forward pass
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        bottleneck = self.bottleneck(enc4)

        # Decoder forward pass with spatial alignment and skip concatenation
        dec4 = self.upconv4(bottleneck)
        dec4 = self._match_and_cat(dec4, enc4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._match_and_cat(dec3, enc3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._match_and_cat(dec2, enc2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = self._match_and_cat(dec1, enc1)
        dec1 = self.decoder1(dec1)

        out = self.classifier(dec1)
        if out.shape[2:] != orig_shape:
            out = F.interpolate(out, size=orig_shape, mode="bilinear", align_corners=False)
        return out

    @staticmethod
    def _match_and_cat(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Align spatial dimensions safely if odd or non-divisible before concatenation."""
        if source.shape[2:] != target.shape[2:]:
            source = F.interpolate(source, size=target.shape[2:], mode="bilinear", align_corners=False)
        return torch.cat([source, target], dim=1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters matching project convention."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["PreActResidualBlock", "ResUNet2D", "count_parameters"]
