from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from src.agent.models import DEFAULT_CHECKPOINTS, ModelRegistry, normalize_model_name
from src.agent.runner import InferenceAgent
from src.agent.state import AgentAction, PolicyConfig, SlicePrediction


class DummyModel(nn.Module):
    """Dummy module returning controlled logits for fast CPU unit testing."""

    def __init__(self, dominant_class: int) -> None:
        super().__init__()
        self.dominant_class = dominant_class
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        logits = torch.zeros((b, 4, h, w), dtype=torch.float32, device=x.device)
        logits[:, self.dominant_class, :, :] = 10.0
        return logits


class TestModelRegistryCPU(unittest.TestCase):
    """Fast CPU unit tests for ModelRegistry without loading heavy disk checkpoints."""

    def setUp(self) -> None:
        self.registry = ModelRegistry(default_device=torch.device("cpu"))

    def test_normalize_model_name(self) -> None:
        self.assertEqual(normalize_model_name("unet"), "unet")
        self.assertEqual(normalize_model_name("U-Net"), "unet")
        self.assertEqual(normalize_model_name("UNetPlusPlus"), "unetpp")
        self.assertEqual(normalize_model_name("Attention_UNet2D"), "attention_unet")

        with self.assertRaises(ValueError):
            normalize_model_name("unknown_transformer")

    def test_shape_and_type_validation(self) -> None:
        with self.assertRaises(TypeError):
            self.registry.predict_slice("unet", [[1.0, 2.0]])  # type: ignore

        with self.assertRaises(ValueError):
            # 5D tensor
            self.registry.predict_slice("unet", torch.randn(1, 4, 32, 32, 1))

        with self.assertRaises(ValueError):
            # Batch size > 1
            self.registry.predict_slice("unet", torch.randn(2, 4, 32, 32))

        with self.assertRaises(ValueError):
            # Channel count != 4
            self.registry.predict_slice("unet", torch.randn(1, 3, 32, 32))

        with self.assertRaises(ValueError):
            # Non-positive spatial dimension
            self.registry.predict_slice("unet", torch.randn(1, 4, 0, 32))

    def test_accepts_3d_slice_tensor(self) -> None:
        # 3D tensor [4, H, W] should be accepted and unsqueezed
        # We inject a dummy model into the cache to avoid disk reads
        self.registry._loaded_models["unet"] = DummyModel(dominant_class=1)
        slice_3d = torch.zeros(4, 16, 16)
        pred = self.registry.predict_slice("unet", slice_3d)

        self.assertEqual(pred.model_name, "unet")
        self.assertEqual(pred.pred_foreground_pixels, 16 * 16)

    def test_lazy_loading_and_unloading(self) -> None:
        self.registry._loaded_models["unet"] = DummyModel(dominant_class=0)
        self.registry._loaded_models["unetpp"] = DummyModel(dominant_class=1)

        self.assertIn("unet", self.registry.loaded_model_names)
        self.assertIn("unetpp", self.registry.loaded_model_names)

        # Unload single model
        self.registry.unload("unet")
        self.assertNotIn("unet", self.registry.loaded_model_names)
        self.assertIn("unetpp", self.registry.loaded_model_names)

        # Unload all
        self.registry.unload()
        self.assertEqual(len(self.registry.loaded_model_names), 0)


class TestInferenceAgentCPU(unittest.TestCase):
    """Fast CPU unit tests for InferenceAgent on-demand candidate fallback."""

    def setUp(self) -> None:
        self.registry = ModelRegistry(default_device=torch.device("cpu"))
        # unetpp predicts class 0 (empty foreground)
        self.registry._loaded_models["unetpp"] = DummyModel(dominant_class=0)
        # unet predicts class 1 (valid foreground)
        self.registry._loaded_models["unet"] = DummyModel(dominant_class=1)
        # attention_unet predicts class 2 (valid foreground)
        self.registry._loaded_models["attention_unet"] = DummyModel(dominant_class=2)

    def test_on_demand_fallback_executes_candidate(self) -> None:
        agent = InferenceAgent(
            registry=self.registry,
            primary_model="unetpp",
            config=PolicyConfig(
                enable_empty_prediction_fallback=True,
                fallback_priority=("unet",),
            ),
            device=torch.device("cpu"),
        )
        dummy_input = torch.zeros(1, 4, 16, 16)
        decision = agent.run_slice(dummy_input, case_id="mock_case", slice_index=10)

        # unetpp had empty foreground; agent should have obtained unet prediction and switched
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("warning_complete_empty_prediction", decision.reasons)
        self.assertIn("fallback_selected_unet", decision.reasons)

    def test_no_fallback_when_primary_has_foreground(self) -> None:
        agent = InferenceAgent(
            registry=self.registry,
            primary_model="unet",  # unet produces foreground class 1
            config=PolicyConfig(
                enable_empty_prediction_fallback=True,
                fallback_priority=("unetpp",),
            ),
            device=torch.device("cpu"),
        )
        dummy_input = torch.zeros(1, 4, 16, 16)
        decision = agent.run_slice(dummy_input, case_id="mock_case", slice_index=10)

        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "unet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("nominal_acceptance", decision.reasons)


class TestRealCheckpointSmoke(unittest.TestCase):
    """Smoke tests verifying that actual disk checkpoints can be loaded and run."""

    def test_load_and_run_all_checkpoints(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        registry = ModelRegistry(default_device=device)

        # Check if checkpoints exist before running
        for name, ckpt_path in DEFAULT_CHECKPOINTS.items():
            if not ckpt_path.exists():
                raise unittest.SkipTest(f"Checkpoint for '{name}' not found at {ckpt_path}.")

        dummy_slice = torch.zeros((1, 4, 240, 240), dtype=torch.float32)

        for name in ("unet", "unetpp", "attention_unet"):
            pred = registry.predict_slice(name, dummy_slice, device=device)
            self.assertIsInstance(pred, SlicePrediction)
            self.assertEqual(pred.model_name, name)
            self.assertEqual(tuple(pred.pred_mask.shape), (240, 240))
            self.assertEqual(pred.pred_mask.dtype, torch.uint8)

        # Clean up memory
        registry.unload()


class TestRegressionParity(unittest.TestCase):
    """Exact regression comparison against precomputed benchmark in model_uncertainty_comparison.csv."""

    def test_parity_slice_30(self) -> None:
        case_id = "BraTS-GLI-00045-001"
        slice_index = 30
        expected_unet_fg_pixels = 5
        expected_unet_entropy = 0.4576052129268646  # From model_uncertainty_comparison.csv

        ckpt_path = DEFAULT_CHECKPOINTS["unet"]
        if not ckpt_path.exists():
            raise unittest.SkipTest(f"Checkpoint {ckpt_path} not found.")

        from src.data.dataset import discover_valid_cases, _load_and_normalize_case

        try:
            cases = discover_valid_cases()
            case = next((c for c in cases if c.case_id == case_id), None)
        except Exception as e:
            raise unittest.SkipTest(f"Unable to discover case data: {e}")

        if case is None:
            raise unittest.SkipTest(f"Case {case_id} not found in raw data directory.")

        vol = _load_and_normalize_case(case)
        slice_tensor = torch.from_numpy(vol.image[:, :, :, slice_index]).unsqueeze(0)

        device = torch.device("cpu")
        registry = ModelRegistry(default_device=device)
        pred = registry.predict_slice("unet", slice_tensor, device=device)

        # Assert exact parity on predicted foreground pixels
        self.assertEqual(
            pred.pred_foreground_pixels,
            expected_unet_fg_pixels,
            f"Expected {expected_unet_fg_pixels} foreground pixels, got {pred.pred_foreground_pixels}",
        )

        # Assert floating-point parity within 1e-4 tolerance on entropy
        self.assertIsNotNone(pred.mean_foreground_entropy)
        assert pred.mean_foreground_entropy is not None
        self.assertAlmostEqual(
            pred.mean_foreground_entropy,
            expected_unet_entropy,
            places=4,
            msg=f"Entropy mismatch: got {pred.mean_foreground_entropy}, expected {expected_unet_entropy}",
        )

        registry.unload()


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
