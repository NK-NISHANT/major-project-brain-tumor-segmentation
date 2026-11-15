import math
import unittest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.train import TrainingConfig, validate_config, run_train_epoch, DiceCrossEntropyLoss


class TestStabilizationFeatures(unittest.TestCase):
    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        self.assertEqual(cfg.warmup_epochs, 0)
        self.assertIsNone(cfg.max_grad_norm)
        validate_config(cfg)

    def test_training_config_validation(self):
        with self.assertRaises(ValueError):
            validate_config(TrainingConfig(warmup_epochs=-1))
        with self.assertRaises(ValueError):
            validate_config(TrainingConfig(max_grad_norm=-0.5))
        with self.assertRaises(ValueError):
            validate_config(TrainingConfig(max_grad_norm=0.0))

    def test_run_train_epoch_with_gradient_clipping(self):
        torch.manual_seed(42)
        model = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=1),
        )
        x = torch.randn(4, 4, 32, 32)
        y = torch.randint(0, 4, (4, 32, 32), dtype=torch.long)
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=2)
        criterion = DiceCrossEntropyLoss(num_classes=4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Run with max_grad_norm = 0.5 (should clip)
        avg_loss, pre_norm, post_norm, clipped = run_train_epoch(
            model=model,
            data_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            device=torch.device("cpu"),
            epoch=1,
            total_epochs=1,
            max_grad_norm=0.5,
        )
        self.assertTrue(math.isfinite(avg_loss))
        self.assertTrue(math.isfinite(pre_norm))
        self.assertTrue(math.isfinite(post_norm))
        self.assertLessEqual(post_norm, 0.55)

    def test_linear_warmup_scheduler(self):
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW([
            {"params": [model.weight], "lr": 1e-4},
            {"params": [model.bias], "lr": 1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=2,
        )
        # Initial lr (before step):
        # Param group 0 starts at 1e-5 (0.1 * 1e-4)
        # Param group 1 starts at 1e-4 (0.1 * 1e-3)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-4)

        scheduler.step()
        # After 1 step: factor is 0.55
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 5.5e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 5.5e-4)

        scheduler.step()
        # After 2 steps: factor is 1.0
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-4)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 1e-3)


if __name__ == "__main__":
    unittest.main()
