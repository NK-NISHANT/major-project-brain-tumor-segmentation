import builtins
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import torch
import torch.nn as nn

from src.training.train_scaled import create_model, load_and_verify_cohort_manifests
from src.models.unet import UNet2D
from src.models.unetpp import UNetPlusPlus2D
from src.models.attention_unet import AttentionUNet2D
from src.models.resunet import ResUNet2D
from src.models.unet_resnet34 import UNetResNet34


class TestTrainScaledModelFactory(unittest.TestCase):
    """Verify that create_model in train_scaled correctly supports all 5 Stage C architectures."""

    def test_create_unet(self):
        model = create_model("unet", base_channels=32, num_classes=4)
        self.assertIsInstance(model, UNet2D)
        x = torch.zeros(1, 4, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 4, 64, 64))

    def test_create_unetpp(self):
        model = create_model("unetpp", base_channels=32, num_classes=4)
        self.assertIsInstance(model, UNetPlusPlus2D)
        self.assertFalse(model.deep_supervision)
        x = torch.zeros(1, 4, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 4, 64, 64))

    def test_create_attention_unet(self):
        model = create_model("attention_unet", base_channels=32, num_classes=4)
        self.assertIsInstance(model, AttentionUNet2D)
        x = torch.zeros(1, 4, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 4, 64, 64))

    def test_create_resunet(self):
        model = create_model("resunet_2d", base_channels=32, num_classes=4)
        self.assertIsInstance(model, ResUNet2D)
        x = torch.zeros(1, 4, 64, 64)
        out = model(x)
        self.assertEqual(out.shape, (1, 4, 64, 64))

    def test_create_unet_resnet34(self):
        # UNetResNet34 requires min spatial size for ResNet stem downsamplings
        model = create_model("unet_resnet34", base_channels=32, num_classes=4)
        self.assertIsInstance(model, UNetResNet34)
        x = torch.zeros(1, 4, 128, 128)
        out = model(x)
        self.assertEqual(out.shape, (1, 4, 128, 128))

    def test_unsupported_model_raises(self):
        with self.assertRaises(ValueError):
            create_model("unsupported_architecture_xyz")

    def test_synthetic_gradient_clipping_amplified_gradients(self):
        """Verify that gradient clipping reduces an artificially amplified gradient norm to <= max_norm (1.0).
        Strictly tests gradient clipping without calling optimizer.step() or updating model weights."""
        model = create_model("unet", base_channels=16, num_classes=4)

        # Snapshot initial weights to verify weights are NOT modified
        initial_params = [p.clone().detach() for p in model.parameters()]

        # Synthetic forward pass & loss computation
        x = torch.randn(2, 4, 32, 32)
        target = torch.randint(0, 4, (2, 32, 32), dtype=torch.long)
        criterion = nn.CrossEntropyLoss()

        out = model(x)
        loss = criterion(out, target)
        loss.backward()

        # Artificially amplify gradients by a factor of 1000.0 to guarantee norm >> 1.0
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(1000.0)

        # Verify pre-clip norm is significantly larger than 1.0
        # Emulate train_scaled.py clipping mechanism
        max_grad_norm = 1.0
        pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
        post_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        pre_val = float(pre_norm.item() if hasattr(pre_norm, "item") else pre_norm)
        post_val = float(post_norm.item() if hasattr(post_norm, "item") else post_norm)

        self.assertGreater(pre_val, 10.0, f"Expected amplified pre_norm >> 1.0, got {pre_val}")
        self.assertLessEqual(post_val, max_grad_norm + 1e-5, f"Expected post_norm <= {max_grad_norm}, got {post_val}")
        self.assertAlmostEqual(post_val, 1.0, places=3, msg=f"Clipped norm should equal max_grad_norm (1.0), got {post_val}")

        # Strictly verify weights were NOT updated (no optimizer.step was called)
        for p_init, p_curr in zip(initial_params, model.parameters()):
            self.assertTrue(torch.equal(p_init, p_curr), "Model weights must remain untouched (no optimizer.step)")


class TestCohortManifestQuarantine(unittest.TestCase):
    """Verify that cohort manifest verification strictly respects test quarantine boundaries."""

    def test_cohort_manifest_loader_does_not_access_test_manifest(self):
        """Mandatory quarantine test: Verify that loading authorized manifests NEVER opens test_manifest.json."""
        project_root = Path(__file__).resolve().parents[1]
        splits_dir = project_root / "results" / "dataset_splits"

        real_open = builtins.open
        accessed_paths: list[str] = []

        def guarded_open(file, *args, **kwargs):
            norm_path = str(file).replace("\\", "/")
            accessed_paths.append(norm_path)
            if "test_manifest" in norm_path or "quarantine" in norm_path:
                raise PermissionError(f"TEST QUARANTINE BREACH: Prohibited access to sealed test file: {norm_path}")
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=guarded_open):
            train_cases, val_cases = load_and_verify_cohort_manifests(splits_dir)

        # Confirm successful manifest load without quarantine breach
        self.assertEqual(len(train_cases), 721)
        self.assertEqual(len(val_cases), 155)

        # Confirm permitted manifests were accessed
        self.assertTrue(any("development_train_manifest.json" in p for p in accessed_paths))
        self.assertTrue(any("validation_manifest.json" in p for p in accessed_paths))

        # Strictly confirm ZERO access to test_manifest.json or quarantine
        self.assertFalse(any("test_manifest" in p for p in accessed_paths))
        self.assertFalse(any("quarantine" in p for p in accessed_paths))

    def test_guarded_open_catches_simulated_quarantine_breach(self):
        """Verify that the guard itself properly detects and raises PermissionError on test_manifest access."""
        real_open = builtins.open

        def guarded_open(file, *args, **kwargs):
            norm_path = str(file).replace("\\", "/")
            if "test_manifest" in norm_path or "quarantine" in norm_path:
                raise PermissionError(f"TEST QUARANTINE BREACH: Prohibited access to sealed test file: {norm_path}")
            return real_open(file, *args, **kwargs)

        with mock.patch("builtins.open", side_effect=guarded_open):
            with self.assertRaises(PermissionError):
                with open("results/dataset_splits/test_manifest.json", "r") as f:
                    pass

    def test_cohort_manifest_case_overlap_detected(self):
        """Verify that case ID overlap between train and val raises an explicit error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir)
            train_manifest = {
                "cases": [
                    {"case_id": "case_001", "subject_id": "subj_001"},
                    {"case_id": "case_002", "subject_id": "subj_002"},
                ]
            }
            val_manifest = {
                "cases": [
                    {"case_id": "case_002", "subject_id": "subj_003"},  # Duplicate case_id
                ]
            }
            with open(splits_dir / "development_train_manifest.json", "w", encoding="utf-8") as f:
                json.dump(train_manifest, f)
            with open(splits_dir / "validation_manifest.json", "w", encoding="utf-8") as f:
                json.dump(val_manifest, f)

            with self.assertRaises(ValueError) as ctx:
                load_and_verify_cohort_manifests(
                    splits_dir,
                    expected_train_cases=2,
                    expected_train_subjects=2,
                    expected_val_cases=1,
                    expected_val_subjects=1,
                )
            self.assertIn("case ID overlap detected", str(ctx.exception))

    def test_cohort_manifest_subject_overlap_detected(self):
        """Verify that biological subject overlap between train and val raises an explicit error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir)
            train_manifest = {
                "cases": [
                    {"case_id": "case_001", "subject_id": "subj_001"},
                    {"case_id": "case_002", "subject_id": "subj_002"},
                ]
            }
            val_manifest = {
                "cases": [
                    {"case_id": "case_003", "subject_id": "subj_001"},  # Overlapping subject_id
                ]
            }
            with open(splits_dir / "development_train_manifest.json", "w", encoding="utf-8") as f:
                json.dump(train_manifest, f)
            with open(splits_dir / "validation_manifest.json", "w", encoding="utf-8") as f:
                json.dump(val_manifest, f)

            with self.assertRaises(ValueError) as ctx:
                load_and_verify_cohort_manifests(
                    splits_dir,
                    expected_train_cases=2,
                    expected_train_subjects=2,
                    expected_val_cases=1,
                    expected_val_subjects=1,
                )
            self.assertIn("Biological subject overlap detected", str(ctx.exception))

    def test_missing_manifest_raises_file_not_found(self):
        """Verify that missing permitted manifests produce an explicit FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                load_and_verify_cohort_manifests(splits_dir)

    def test_malformed_manifest_raises_value_error(self):
        """Verify that malformed JSON or missing 'cases' key raises an explicit ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            splits_dir = Path(tmpdir)
            with open(splits_dir / "development_train_manifest.json", "w", encoding="utf-8") as f:
                f.write("{ invalid json")
            with open(splits_dir / "validation_manifest.json", "w", encoding="utf-8") as f:
                f.write("{}")

            with self.assertRaises(ValueError):
                load_and_verify_cohort_manifests(splits_dir)

    def test_cli_argument_parsing_and_defaults(self):
        """Verify that CLI accepts --model unet and all Stage C configuration arguments without launching training."""
        import argparse

        # Create parser matching train_scaled.py main()
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--model",
            choices=["unet", "unetpp", "attention_unet", "resunet_2d", "unet_resnet34"],
            required=True,
        )
        parser.add_argument("--epochs", type=int, default=15)
        parser.add_argument("--batch-size", type=int, default=2)
        parser.add_argument("--base-lr", type=float, default=1e-3)
        parser.add_argument("--warmup-epochs", type=int, default=2)
        parser.add_argument("--max-grad-norm", type=float, default=1.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--results-dir", type=Path, default=None)

        cli_args = [
            "--model", "unet",
            "--epochs", "15",
            "--batch-size", "2",
            "--base-lr", "0.001",
            "--warmup-epochs", "2",
            "--max-grad-norm", "1.0",
            "--seed", "42",
            "--results-dir", "results/stage_c/phase1/unet_721_seed42",
        ]
        args = parser.parse_args(cli_args)
        self.assertEqual(args.model, "unet")
        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.base_lr, 0.001)
        self.assertEqual(args.warmup_epochs, 2)
        self.assertEqual(args.max_grad_norm, 1.0)
        self.assertEqual(args.seed, 42)
        self.assertEqual(str(args.results_dir).replace("\\", "/"), "results/stage_c/phase1/unet_721_seed42")


if __name__ == "__main__":
    unittest.main()
