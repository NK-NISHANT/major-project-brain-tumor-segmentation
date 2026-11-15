"""Bounded unit test suite for Phase B model implementations (ResUNet2D & UNetResNet34).

Verifies 10 core criteria:
1. Model construction with valid configurations.
2. Expected four-channel output and spatial alignment.
3. Finite output logits.
4. Backward propagation with finite gradients on synthetic targets containing valid label IDs.
5. Parameter count and trainable parameter enumeration.
6. Correct behavior at the project's intended input resolution (240x240).
7. Compatibility with the existing loss function (DiceCrossEntropyLoss).
8. Safe behavior for small or odd-sized spatial inputs.
9. Model-factory and configuration integration.
10. Deterministic basic behavior under fixed seeds in eval mode.
"""

from __future__ import annotations

import unittest
import torch
from torch import nn

from src.agent.models import MODEL_FACTORIES, normalize_model_name
from src.models.resunet import ResUNet2D, count_parameters as count_resunet_params
from src.models.unet_resnet34 import UNetResNet34, count_parameters as count_resnet34_params
from src.training.train import DiceCrossEntropyLoss, TrainingConfig, create_model


class TestNewSegmentationModels(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    # 1. Model construction with valid configurations
    def test_01_model_construction(self) -> None:
        model_res = ResUNet2D(in_channels=4, num_classes=4, base_channels=32)
        self.assertIsInstance(model_res, nn.Module)

        model_resnet34_rand = UNetResNet34(in_channels=4, num_classes=4, pretrained=False)
        self.assertIsInstance(model_resnet34_rand, nn.Module)

        model_resnet34_pre = UNetResNet34(in_channels=4, num_classes=4, pretrained=True)
        self.assertIsInstance(model_resnet34_pre, nn.Module)

    # 2. Expected four-channel output and spatial alignment
    def test_02_output_channels_and_spatial_alignment(self) -> None:
        small_batch = torch.randn(2, 4, 64, 64, device=self.device)
        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=False),
        ]:
            model.eval()
            with torch.no_grad():
                out = model(small_batch)
            self.assertEqual(tuple(out.shape), (2, 4, 64, 64))

    # 3. Finite output logits
    def test_03_finite_output_logits(self) -> None:
        inputs = torch.randn(1, 4, 64, 64, device=self.device)
        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=True),
        ]:
            model.eval()
            with torch.no_grad():
                logits = model(inputs)
            self.assertTrue(torch.isfinite(logits).all().item(), "Logits contain NaN or Inf!")

    # 4. Backward propagation with finite gradients
    def test_04_backward_propagation_finite_gradients(self) -> None:
        inputs = torch.randn(2, 4, 64, 64, device=self.device)
        targets = torch.randint(0, 4, (2, 64, 64), device=self.device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=False),
        ]:
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()

            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
                    self.assertTrue(torch.isfinite(param.grad).all().item(), f"Gradient non-finite for {name}")

    # 5. Parameter count and trainable parameter enumeration
    def test_05_parameter_counts(self) -> None:
        model_res = ResUNet2D(in_channels=4, num_classes=4, base_channels=32)
        n_res = count_resunet_params(model_res)
        self.assertEqual(n_res, 8114604, f"ResUNet2D param count mismatch: {n_res} != 8114604")

        model_resnet34 = UNetResNet34(in_channels=4, num_classes=4, pretrained=False)
        n_resnet34 = count_resnet34_params(model_resnet34)
        self.assertEqual(n_resnet34, 24327524, f"UNetResNet34 param count mismatch: {n_resnet34} != 24327524")

    # 6. Correct behavior at the project's intended input resolution (240x240)
    def test_06_intended_resolution_240x240(self) -> None:
        full_slice = torch.randn(2, 4, 240, 240, device=self.device)
        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=True),
        ]:
            model.eval()
            with torch.no_grad():
                out = model(full_slice)
            self.assertEqual(tuple(out.shape), (2, 4, 240, 240))
            self.assertTrue(torch.isfinite(out).all().item())

    # 7. Compatibility with the existing loss function (DiceCrossEntropyLoss)
    def test_07_dice_cross_entropy_loss_compatibility(self) -> None:
        criterion = DiceCrossEntropyLoss(num_classes=4, dice_weight=1.0, ce_weight=1.0)
        inputs = torch.randn(2, 4, 64, 64, device=self.device)
        targets = torch.randint(0, 4, (2, 64, 64), device=self.device, dtype=torch.long)

        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=False),
        ]:
            model.train()
            logits = model(inputs)
            loss = criterion(logits, targets)
            self.assertTrue(torch.isfinite(loss).item())
            self.assertGreater(float(loss.item()), 0.0)

    # 8. Safe behavior for small or odd-sized spatial inputs
    def test_08_odd_and_non_divisible_spatial_inputs(self) -> None:
        odd_input = torch.randn(1, 4, 65, 63, device=self.device)
        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=False),
        ]:
            model.eval()
            with torch.no_grad():
                out = model(odd_input)
            self.assertEqual(tuple(out.shape), (1, 4, 65, 63))

    # 9. Model-factory and configuration integration
    def test_09_model_factory_and_registry_integration(self) -> None:
        cfg_res = TrainingConfig(model_name="resunet", base_channels=32)
        model_res = create_model(cfg_res)
        self.assertIsInstance(model_res, ResUNet2D)

        cfg_resnet34 = TrainingConfig(model_name="unet_resnet34")
        model_resnet34 = create_model(cfg_resnet34)
        self.assertIsInstance(model_resnet34, UNetResNet34)

        self.assertEqual(normalize_model_name("resunet"), "resunet")
        self.assertEqual(normalize_model_name("resunet2d"), "resunet")
        self.assertEqual(normalize_model_name("unet_resnet34"), "unet_resnet34")
        self.assertEqual(normalize_model_name("resnet34_unet"), "unet_resnet34")

        self.assertIn("resunet", MODEL_FACTORIES)
        self.assertIn("unet_resnet34", MODEL_FACTORIES)

    # 10. Deterministic basic behavior under fixed seeds in eval mode
    def test_10_deterministic_eval_behavior(self) -> None:
        x = torch.randn(1, 4, 64, 64, device=self.device)
        for model in [
            ResUNet2D(in_channels=4, num_classes=4, base_channels=32),
            UNetResNet34(in_channels=4, num_classes=4, pretrained=False),
        ]:
            model.eval()
            with torch.no_grad():
                out1 = model(x)
                out2 = model(x)
            self.assertTrue(torch.equal(out1, out2), "Consecutive eval forward passes are non-deterministic!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
