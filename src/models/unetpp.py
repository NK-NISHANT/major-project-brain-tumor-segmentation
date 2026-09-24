from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions used to refine each U-Net++ node."""

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


class UNetPlusPlus2D(nn.Module):
    """Readable 2D U-Net++ for controlled BraTS-GLI architecture benchmarking.

    Deep supervision is intentionally disabled for the first benchmark. The
    model returns one final logits tensor, matching the baseline U-Net training
    loop, loss, and metric interface.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if deep_supervision:
            raise ValueError("Deep supervision is not enabled for this controlled benchmark.")

        self.deep_supervision = deep_supervision
        channels = [
            base_channels,
            base_channels * 2,
            base_channels * 4,
            base_channels * 8,
            base_channels * 16,
        ]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.x00 = ConvBlock(in_channels, channels[0])
        self.x10 = ConvBlock(channels[0], channels[1])
        self.x20 = ConvBlock(channels[1], channels[2])
        self.x30 = ConvBlock(channels[2], channels[3])
        self.x40 = ConvBlock(channels[3], channels[4])

        self.x01 = ConvBlock(channels[0] + channels[1], channels[0])
        self.x11 = ConvBlock(channels[1] + channels[2], channels[1])
        self.x21 = ConvBlock(channels[2] + channels[3], channels[2])
        self.x31 = ConvBlock(channels[3] + channels[4], channels[3])

        self.x02 = ConvBlock((channels[0] * 2) + channels[1], channels[0])
        self.x12 = ConvBlock((channels[1] * 2) + channels[2], channels[1])
        self.x22 = ConvBlock((channels[2] * 2) + channels[3], channels[2])

        self.x03 = ConvBlock((channels[0] * 3) + channels[1], channels[0])
        self.x13 = ConvBlock((channels[1] * 3) + channels[2], channels[1])

        self.x04 = ConvBlock((channels[0] * 4) + channels[1], channels[0])
        self.classifier = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        # U-Net++ uses nested dense skip pathways. Each decoder node receives
        # earlier same-resolution nodes plus an upsampled deeper feature map.
        x01 = self.x01(torch.cat([x00, self._upsample_like(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._upsample_like(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._upsample_like(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self._upsample_like(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self._upsample_like(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._upsample_like(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self._upsample_like(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self._upsample_like(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._upsample_like(x22, x10)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self._upsample_like(x13, x00)], dim=1))
        return self.classifier(x04)

    @staticmethod
    def _upsample_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[2:], mode="bilinear", align_corners=False)


__all__ = ["ConvBlock", "UNetPlusPlus2D"]
