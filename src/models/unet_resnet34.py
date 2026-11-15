"""2D U-Net with ResNet-34 encoder for multi-channel brain tumor segmentation.

B.Tech Major Project:
Brain Tumor Segmentation Using 2D CNNs and a Reliability-Aware AI Agent
Workspace: D:\MajorProject
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.models
from torch import nn

logger = logging.getLogger(__name__)


class DecoderBlock(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU for decoder feature refinement."""

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


class UNetResNet34(nn.Module):
    """2D U-Net utilizing a ResNet-34 encoder and symmetric decoder.

    Contract:
        Input:  [B, 4, H, W] float32 tensor (t1n, t1c, t2f, t2w).
        Output: [B, 4, H, W] unnormalized logits for classes {0, 1, 2, 3}.

    Architectural Design Choices:
        1. 4-Channel Input Surgery:
           The torchvision 3-channel conv1 (7x7, stride 2) is surgically adapted
           to 4 channels. When loading pretrained ImageNet weights, the 4th channel
           is initialized as the mean across RGB channels and scaled by (3/4).
           This mathematically conserves filter DC gain and activation variance,
           protecting pretrained bn1 running statistics from distribution shock.
        2. Small-Lesion Spatial Preservation:
           In standard ResNet-34, conv1 (stride 2) followed immediately by maxpool
           (stride 2) collapses 240x240 -> 60x60, obliterating tiny enhancing tumor
           (ET) foci (3x3 voxels) before layer1. Here, maxpool is replaced with
           nn.Identity(). The 3 residual blocks of layer1 operate directly at 120x120,
           preserving spatial fidelity. All subsequent stages (15 -> 30 -> 60 -> 120 -> 240)
           are exact power-of-2 integer multiples.
        3. Pretrained Weight Handling:
           Checks local cache (~/.cache/torch/hub/checkpoints/resnet34-b627a593.pth).
           If present and requested, loads weights strictly offline without network calls.
           If absent or pretrained=False, initializes with Kaiming normal.
        4. Differential Learning Rates:
           Provides get_parameter_groups() to allow fine-tuning the pretrained backbone
           at a reduced learning rate (e.g. 0.1x) while training the decoder at 1.0x.
    """

    def __init__(
        self,
        in_channels: int = 4,
        num_classes: int = 4,
        base_channels: int = 32,  # Kept for signature compatibility with ModelRegistry
        pretrained: bool = False,
        weights_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1. Instantiate ResNet-34 backbone strictly without auto-downloading
        backbone = torchvision.models.resnet34(weights=None)

        # 2. Optionally load locally cached pretrained weights
        if pretrained:
            self._load_local_pretrained_weights(backbone, weights_path)

        # 3. Adapt conv1 to 4 channels with 3/4 energy scaling
        self._adapt_conv1(backbone, in_channels, pretrained)

        # 4. Replace initial maxpool with Identity to preserve 120x120 spatial resolution
        backbone.maxpool = nn.Identity()

        # Extract encoder submodules
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 channels, spatial 120x120
        self.layer2 = backbone.layer2  # 128 channels, spatial 60x60
        self.layer3 = backbone.layer3  # 256 channels, spatial 30x30
        self.layer4 = backbone.layer4  # 512 channels, spatial 15x15 (Bottleneck)

        # 5. Symmetric Decoder stages
        # Level 4 -> Level 3 (spatial 15 -> 30)
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = DecoderBlock(512, 256)  # 256 (upconv) + 256 (layer3 skip)

        # Level 3 -> Level 2 (spatial 30 -> 60)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = DecoderBlock(256, 128)  # 128 (upconv) + 128 (layer2 skip)

        # Level 2 -> Level 1 (spatial 60 -> 120)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = DecoderBlock(128, 64)  # 64 (upconv) + 64 (layer1 skip)

        # Level 1 -> Level 0 (spatial 120 -> 240)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = DecoderBlock(32, 32)

        # Final 1x1 classification head
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        # Initialize decoder parameters
        self._init_decoder_weights()

    def _load_local_pretrained_weights(
        self,
        backbone: nn.Module,
        weights_path: Path | str | None,
    ) -> None:
        """Load pretrained ResNet-34 weights from local cache without network access."""
        resolved_path: Path | None = None
        if weights_path is not None:
            resolved_path = Path(weights_path)
        else:
            candidates = [
                Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet34-b627a593.pth",
                Path(torch.hub.get_dir()) / "checkpoints" / "resnet34-b627a593.pth",
            ]
            for p in candidates:
                if p.is_file():
                    resolved_path = p
                    break

        if resolved_path is not None and resolved_path.is_file():
            try:
                state_dict = torch.load(resolved_path, map_location="cpu", weights_only=True)
                # Load backbone weights strictly before conv1 surgery (ignore classifier fc)
                backbone.load_state_dict(state_dict, strict=False)
                logger.info("Loaded pretrained ResNet-34 weights from %s", resolved_path)
            except Exception as e:
                logger.warning("Failed to load local weights from %s: %s. Using random init.", resolved_path, e)
        else:
            logger.info("Local pretrained ResNet-34 weights not found. Using random Kaiming initialization.")

    def _adapt_conv1(self, backbone: nn.Module, in_channels: int, pretrained: bool) -> None:
        """Adapt conv1 from 3 to in_channels (4) with 3/4 energy scaling."""
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained and old_conv.weight is not None:
            w_old = old_conv.weight.data  # [64, 3, 7, 7]
            w_mean = w_old.mean(dim=1, keepdim=True)  # [64, 1, 7, 7]
            # Conserve DC gain entering bn1: (3/4) * (W_rgb + W_mean)
            w_new = torch.cat([w_old, w_mean], dim=1) * (3.0 / 4.0)
            new_conv.weight.data.copy_(w_new)
        else:
            nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")

        backbone.conv1 = new_conv

    def _init_decoder_weights(self) -> None:
        """Initialize decoder convolutions with Kaiming normal."""
        for m in [
            self.upconv4, self.decoder4,
            self.upconv3, self.decoder3,
            self.upconv2, self.decoder2,
            self.upconv1, self.decoder1,
            self.classifier,
        ]:
            for submodule in m.modules():
                if isinstance(submodule, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(submodule.weight, mode="fan_out", nonlinearity="relu")
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
                elif isinstance(submodule, nn.BatchNorm2d):
                    nn.init.ones_(submodule.weight)
                    nn.init.zeros_(submodule.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape[2:]
        # Encoder forward pass
        x = self.conv1(x)         # [B, 64, 120, 120]
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)       # Identity, preserves [B, 64, 120, 120]

        enc1 = self.layer1(x)     # [B, 64, 120, 120]
        enc2 = self.layer2(enc1)  # [B, 128, 60, 60]
        enc3 = self.layer3(enc2)  # [B, 256, 30, 30]
        enc4 = self.layer4(enc3)  # [B, 512, 15, 15]

        # Decoder forward pass
        dec4 = self.upconv4(enc4)
        dec4 = self._match_and_cat(dec4, enc3)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = self._match_and_cat(dec3, enc2)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = self._match_and_cat(dec2, enc1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
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

    def get_parameter_groups(
        self,
        base_lr: float,
        encoder_lr_ratio: float = 0.1,
        weight_decay: float = 1e-4,
    ) -> list[dict[str, Any]]:
        """Return parameter groups for differential learning rates.

        Pretrained backbone parameters use base_lr * encoder_lr_ratio (e.g. 1e-4).
        Decoder and classification head parameters use base_lr (e.g. 1e-3).
        """
        encoder_modules = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]
        encoder_param_ids = {id(p) for m in encoder_modules for p in m.parameters()}

        encoder_params = [p for p in self.parameters() if id(p) in encoder_param_ids and p.requires_grad]
        decoder_params = [p for p in self.parameters() if id(p) not in encoder_param_ids and p.requires_grad]

        return [
            {
                "params": encoder_params,
                "lr": base_lr * encoder_lr_ratio,
                "weight_decay": weight_decay,
                "name": "encoder",
            },
            {
                "params": decoder_params,
                "lr": base_lr,
                "weight_decay": weight_decay,
                "name": "decoder",
            },
        ]


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters matching project convention."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = ["UNetResNet34", "DecoderBlock", "count_parameters"]
