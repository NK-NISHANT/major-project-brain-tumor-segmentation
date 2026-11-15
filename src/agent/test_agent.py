from __future__ import annotations

import unittest
import torch

from src.agent.policy import evaluate_slice_policy, extract_slice_prediction
from src.agent.state import (
    AgentAction,
    AgentState,
    PolicyConfig,
    SlicePrediction,
)


def _make_synthetic_logits(
    dominant_class: int,
    spatial_size: tuple[int, int] = (32, 32),
    confidence_scale: float = 10.0,
    num_classes: int = 4,
) -> torch.Tensor:
    """Helper creating controlled synthetic logits [1, C, H, W]."""
    h, w = spatial_size
    logits = torch.zeros((1, num_classes, h, w), dtype=torch.float32)
    logits[:, dominant_class, :, :] = confidence_scale
    return logits


class TestAgentStateAndPolicy(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    # 1. Invalid tensor shape and type checks
    def test_invalid_logits_type(self) -> None:
        with self.assertRaises(TypeError):
            extract_slice_prediction("unet", [[1.0, 0.0], [0.0, 1.0]])  # type: ignore

    def test_invalid_logits_ndim(self) -> None:
        invalid_3d = torch.randn(4, 32, 32)
        with self.assertRaises(ValueError):
            extract_slice_prediction("unet", invalid_3d)

    def test_invalid_batch_dimension(self) -> None:
        invalid_batch = torch.randn(2, 4, 32, 32)
        with self.assertRaises(ValueError):
            extract_slice_prediction("unet", invalid_batch)

    def test_invalid_channel_dimension(self) -> None:
        invalid_channels = torch.randn(1, 3, 32, 32)
        with self.assertRaises(ValueError):
            extract_slice_prediction("unet", invalid_channels, num_classes=4)

    def test_invalid_spatial_dimensions(self) -> None:
        invalid_spatial = torch.randn(1, 4, 0, 32)
        with self.assertRaises(ValueError):
            extract_slice_prediction("unet", invalid_spatial)

    # 2. Slice prediction extraction
    def test_extract_prediction_with_foreground(self) -> None:
        # Create logits where class 1 (tumor) dominates
        logits = _make_synthetic_logits(dominant_class=1, spatial_size=(16, 16), confidence_scale=8.0)
        pred = extract_slice_prediction("unetpp", logits)

        self.assertEqual(pred.model_name, "unetpp")
        self.assertFalse(pred.has_empty_foreground)
        self.assertEqual(pred.pred_foreground_pixels, 16 * 16)
        self.assertIsNotNone(pred.mean_foreground_entropy)
        assert pred.mean_foreground_entropy is not None
        self.assertGreaterEqual(pred.mean_foreground_entropy, 0.0)
        self.assertLessEqual(pred.mean_foreground_entropy, 1.0)

    def test_extract_prediction_empty_foreground(self) -> None:
        # Create logits where class 0 (background) dominates everywhere
        logits = _make_synthetic_logits(dominant_class=0, spatial_size=(16, 16), confidence_scale=8.0)
        pred = extract_slice_prediction("unetpp", logits)

        self.assertEqual(pred.model_name, "unetpp")
        self.assertTrue(pred.has_empty_foreground)
        self.assertEqual(pred.pred_foreground_pixels, 0)
        self.assertIsNone(pred.mean_foreground_entropy)

    # 3. Normal acceptance
    def test_normal_acceptance(self) -> None:
        logits = _make_synthetic_logits(dominant_class=2, confidence_scale=10.0)
        primary = extract_slice_prediction("unetpp", logits)
        state = AgentState(case_id="case_001", slice_index=50, primary_prediction=primary)

        # Config with explicit threshold well above the near-zero entropy
        config = PolicyConfig(high_uncertainty_threshold=0.20)
        decision = evaluate_slice_policy(state, config)

        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "unetpp")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("nominal_acceptance", decision.reasons)

    # 4. Empty primary prediction triggers fallback selection
    def test_empty_primary_prediction_fallback_selected(self) -> None:
        empty_logits = _make_synthetic_logits(dominant_class=0)
        primary = extract_slice_prediction("unetpp", empty_logits)

        cand_logits = _make_synthetic_logits(dominant_class=1)
        candidate_unet = extract_slice_prediction("unet", cand_logits)

        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=primary,
            candidate_predictions={"unet": candidate_unet},
        )
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
        )
        decision = evaluate_slice_policy(state, config)

        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("warning_complete_empty_prediction", decision.reasons)
        self.assertIn("fallback_selected_unet", decision.reasons)

    # 5. Fallback selection respects explicit deterministic priority
    def test_fallback_selection_priority_order(self) -> None:
        empty_logits = _make_synthetic_logits(dominant_class=0)
        primary = extract_slice_prediction("primary_empty", empty_logits)

        # Both candidates have valid foreground
        cand_b = extract_slice_prediction("model_b", _make_synthetic_logits(dominant_class=1))
        cand_a = extract_slice_prediction("model_a", _make_synthetic_logits(dominant_class=2))

        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=primary,
            candidate_predictions={"model_b": cand_b, "model_a": cand_a},
        )

        # Priority specifies model_a first
        config_a = PolicyConfig(fallback_priority=("model_a", "model_b"))
        decision_a = evaluate_slice_policy(state, config_a)
        self.assertEqual(decision_a.selected_model, "model_a")

        # Priority specifies model_b first
        config_b = PolicyConfig(fallback_priority=("model_b", "model_a"))
        decision_b = evaluate_slice_policy(state, config_b)
        self.assertEqual(decision_b.selected_model, "model_b")

    # 6. Empty primary prediction without candidate -> FLAG_UNRELIABLE
    def test_empty_primary_without_candidate(self) -> None:
        empty_logits = _make_synthetic_logits(dominant_class=0)
        primary = extract_slice_prediction("unetpp", empty_logits)
        state = AgentState(case_id="case_001", slice_index=50, primary_prediction=primary)

        config = PolicyConfig(enable_empty_prediction_fallback=True)
        decision = evaluate_slice_policy(state, config)

        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "unetpp")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIn("warning_complete_empty_prediction", decision.reasons)
        self.assertIn("no_viable_fallback_candidate_with_foreground", decision.reasons)

    # 7. High uncertainty triggers FLAG_UNRELIABLE
    def test_high_uncertainty_flagging(self) -> None:
        # Uniform logits produce maximum entropy ~ 1.0
        uniform_logits = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
        # Add slight non-zero bias so argmax chooses class 1 (foreground)
        uniform_logits[:, 1, :, :] = 0.01
        primary = extract_slice_prediction("unet", uniform_logits)

        self.assertFalse(primary.has_empty_foreground)
        assert primary.mean_foreground_entropy is not None
        self.assertGreater(primary.mean_foreground_entropy, 0.90)

        # Experimental threshold set at 0.15
        state = AgentState(case_id="case_001", slice_index=50, primary_prediction=primary)
        config = PolicyConfig(high_uncertainty_threshold=0.15)
        decision = evaluate_slice_policy(state, config)

        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "unet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertTrue(any("warning_high_uncertainty" in r for r in decision.reasons))

    # 8. Missing / disabled uncertainty threshold accepts prediction
    def test_disabled_uncertainty_threshold(self) -> None:
        # Even with high entropy, if threshold is None, uncertainty flagging is inactive
        uniform_logits = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
        uniform_logits[:, 1, :, :] = 0.01
        primary = extract_slice_prediction("unet", uniform_logits)

        state = AgentState(case_id="case_001", slice_index=50, primary_prediction=primary)
        config = PolicyConfig(high_uncertainty_threshold=None)
        decision = evaluate_slice_policy(state, config)

        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "unet")
        self.assertIn("nominal_acceptance", decision.reasons)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestAgentStateAndPolicy)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
