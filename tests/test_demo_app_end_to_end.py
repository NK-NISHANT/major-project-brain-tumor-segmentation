"""Controlled end-to-end smoke tests and deterministic safety tests for tools/demo_app.py.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import numpy as np
import torch

from src.agent.models import ModelRegistry
from src.agent.policy import evaluate_slice_policy, extract_slice_prediction
from src.agent.runner import InferenceAgent
from src.agent.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.uncertainty.entropy import softmax_entropy
from tools.demo_app import (
    compute_slice_dice_metrics,
    create_policy_config_for_model,
    format_decision_badge_html,
    get_axial_context_predictions,
    get_validation_cases,
    load_calibrated_policy_config,
    load_case_volume,
    validate_input_slice_tensor,
)


def _make_synthetic_logits(
    dominant_class: int,
    spatial_size: tuple[int, int] = (16, 16),
    confidence_scale: float = 8.0,
    num_classes: int = 4,
) -> torch.Tensor:
    """Create controlled synthetic logits of shape [1, C, H, W]."""
    h, w = spatial_size
    logits = torch.zeros((1, num_classes, h, w), dtype=torch.float32)
    logits[:, dominant_class, :, :] = confidence_scale
    return logits


def _make_prediction(
    model_name: str,
    dominant_class: int,
    spatial_size: tuple[int, int] = (16, 16),
    confidence_scale: float = 8.0,
) -> SlicePrediction:
    """Create typed SlicePrediction with deterministic foreground/background."""
    logits = _make_synthetic_logits(dominant_class, spatial_size, confidence_scale)
    return extract_slice_prediction(model_name=model_name, logits=logits)


class TestDemoAppEndToEndRealVolume(unittest.TestCase):
    """Real-volume pipeline smoke tests on an approved validation case."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.val_cases = get_validation_cases()
        if not cls.val_cases:
            raise unittest.SkipTest("No validation cases found.")
        cls.test_case = cls.val_cases[0]
        cls.image_vol, cls.mask_vol = load_case_volume(str(cls.test_case.case_dir))
        cls.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        cls.registry = ModelRegistry(default_device=cls.device_str)

        base_cfg, _ = load_calibrated_policy_config(tau_threshold=0.35)
        cls.policy_config = create_policy_config_for_model(
            primary_model="resunet",
            base_policy_config=base_cfg,
            tau_threshold=0.35,
        )
        cls.agent = InferenceAgent(
            registry=cls.registry,
            primary_model="resunet",
            config=cls.policy_config,
            device=cls.registry.default_device,
        )

    def test_e2e_real_volume_tumor_slice_execution(self) -> None:
        """Run complete agent on an axial slice containing tumor."""
        # Find middle tumor slice from ground truth
        fg_slices = [z for z in range(155) if (self.mask_vol[:, :, z] > 0).any()]
        self.assertGreater(len(fg_slices), 0, "Selected validation case must have tumor slices.")
        target_z = fg_slices[len(fg_slices) // 2]

        slice_img_np = self.image_vol[:, :, :, target_z]
        slice_gt_np = self.mask_vol[:, :, target_z]
        slice_img_t = torch.from_numpy(slice_img_np).unsqueeze(0).to(self.registry.default_device)

        # 1. Interface validation
        validate_input_slice_tensor(slice_img_t)

        # 2. Axial context neighbor computation
        neighbor_preds = get_axial_context_predictions(
            registry=self.registry,
            primary_model="resunet",
            image_vol=self.image_vol,
            center_z=target_z,
            window_k=2,
            total_slices=155,
            device=self.registry.default_device,
            temperature=self.policy_config.primary_temperature,
        )
        self.assertGreaterEqual(len(neighbor_preds), 1)

        # 3. Agent execution
        decision: AgentDecision = self.agent.run_slice(
            image=slice_img_t,
            case_id=self.test_case.case_id,
            slice_index=target_z,
            case_primary_predictions=neighbor_preds,
            case_total_slices=155,
        )

        # Assertions on AgentDecision
        self.assertIn(decision.action, (AgentAction.ACCEPT, AgentAction.FALLBACK_SWITCH, AgentAction.FLAG_UNRELIABLE))
        self.assertIn(decision.selected_model, ("resunet", "unet_resnet34"))
        self.assertEqual(decision.final_mask.shape, (240, 240))
        self.assertAlmostEqual(decision.calibrated_temperature, 2.8808, places=3)
        self.assertGreater(len(decision.reasons), 0)

        # 4. Calibrated entropy map computation
        selected_pred = self.registry.predict_slice(
            model_name=decision.selected_model,
            image=slice_img_t,
            device=self.registry.default_device,
            temperature=decision.calibrated_temperature,
        )
        entropy_tensor = softmax_entropy(selected_pred.logits)
        self.assertEqual(entropy_tensor.shape, (1, 240, 240))
        self.assertGreaterEqual(entropy_tensor.min().item(), 0.0)
        self.assertLessEqual(entropy_tensor.max().item(), 1.0)

        # 5. Dice metric computation
        dice_metrics = compute_slice_dice_metrics(decision.final_mask.cpu().numpy(), slice_gt_np)
        self.assertIn("wt_dice", dice_metrics)
        self.assertIn("tc_dice", dice_metrics)
        self.assertIn("et_dice", dice_metrics)
        self.assertIn("macro_fg_dice", dice_metrics)

        # 6. HTML badge formatting
        html_badge = format_decision_badge_html(decision, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION:", html_badge)

    def test_e2e_real_volume_boundary_slice_smoke(self) -> None:
        """Run complete agent on an anatomical boundary slice without ground-truth tumor (z=5).

        NOTE: This test validates pipeline robustness on boundary anatomy. It does not
        claim to guarantee model emptiness; deterministic policy safety is verified in
        TestDeterministicEmptyPredictionPolicy.
        """
        empty_z = 5
        self.assertFalse((self.mask_vol[:, :, empty_z] > 0).any(), f"Slice {empty_z} ground truth must be background.")

        slice_img_np = self.image_vol[:, :, :, empty_z]
        slice_img_t = torch.from_numpy(slice_img_np).unsqueeze(0).to(self.registry.default_device)

        validate_input_slice_tensor(slice_img_t)

        neighbor_preds = get_axial_context_predictions(
            registry=self.registry,
            primary_model="resunet",
            image_vol=self.image_vol,
            center_z=empty_z,
            window_k=2,
            total_slices=155,
            device=self.registry.default_device,
            temperature=self.policy_config.primary_temperature,
        )

        decision: AgentDecision = self.agent.run_slice(
            image=slice_img_t,
            case_id=self.test_case.case_id,
            slice_index=empty_z,
            case_primary_predictions=neighbor_preds,
            case_total_slices=155,
        )

        self.assertIsInstance(decision, AgentDecision)
        self.assertIn(decision.action, (AgentAction.ACCEPT, AgentAction.FALLBACK_SWITCH, AgentAction.FLAG_UNRELIABLE))
        self.assertEqual(decision.final_mask.shape, (240, 240))

        # Branch invariant verification: safety assertions are unconditional on whichever branch was executed
        if decision.has_empty_foreground:
            self.assertIsNone(decision.mean_uncertainty, "Empty prediction must have mean_uncertainty is None, never 0.0.")
            self.assertNotEqual(decision.action, AgentAction.ACCEPT, "Empty prediction must never be automatically ACCEPTED.")
            self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
            self.assertTrue(decision.is_human_review_recommended)
        else:
            self.assertIsNotNone(decision.mean_uncertainty)
            assert decision.mean_uncertainty is not None
            self.assertGreaterEqual(decision.mean_uncertainty, 0.0)
            self.assertLessEqual(decision.mean_uncertainty, 1.0)


class TestDeterministicEmptyPredictionPolicy(unittest.TestCase):
    """Deterministic, unconditional tests guaranteeing empty primary predictions and policy enforcement."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        # Class 0 dominant everywhere guarantees empty foreground
        self.primary_empty = _make_prediction("resunet", dominant_class=0)
        self.candidate_fg = _make_prediction("unet_resnet34", dominant_class=1)
        self.candidate_empty = _make_prediction("unet_resnet34", dominant_class=0)

        self.config = PolicyConfig(
            high_uncertainty_threshold=0.35,
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet_resnet34",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
            fallback_when_context_unavailable=False,
            primary_temperature=2.8808,
            candidate_temperatures={"unet_resnet34": 2.6267},
        )

    def test_deterministic_empty_prediction_never_synthetic_zero(self) -> None:
        """An empty prediction must strictly produce mean_foreground_entropy is None, never 0.0."""
        self.assertTrue(self.primary_empty.has_empty_foreground)
        self.assertEqual(self.primary_empty.pred_foreground_pixels, 0)
        self.assertIsNone(self.primary_empty.mean_foreground_entropy)

    def test_deterministic_empty_primary_unresolved_flag_unreliable(self) -> None:
        """Primary empty prediction with no axial foreground support must be FLAG_UNRELIABLE."""
        ctx_all_empty = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=0,
            empty_neighbors_count=4,
        )
        state = AgentState(
            case_id="case_det_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_fg},
            axial_context=ctx_all_empty,
        )

        decision = evaluate_slice_policy(state, self.config)

        # UNCONDITIONAL SAFETY ASSERTIONS
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertNotEqual(decision.action, AgentAction.ACCEPT, "Empty primary prediction must never be ACCEPTED.")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIsNone(decision.mean_uncertainty, "Empty foreground uncertainty must be None, never 0.0.")
        self.assertTrue(decision.is_human_review_recommended)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)
        self.assertIn("warning_complete_empty_prediction", decision.reasons)

    def test_deterministic_empty_primary_axial_fallback_switch(self) -> None:
        """Primary empty prediction with axial foreground support and non-empty candidate produces FALLBACK_SWITCH."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_det_002",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_fg},
            axial_context=ctx_supported,
        )

        decision = evaluate_slice_policy(state, self.config)

        # UNCONDITIONAL SAFETY ASSERTIONS
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertNotEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "unet_resnet34")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIsNotNone(decision.mean_uncertainty)
        assert decision.mean_uncertainty is not None
        self.assertGreaterEqual(decision.mean_uncertainty, 0.0)
        self.assertIn("fallback_selected_unet_resnet34", decision.reasons)
        self.assertFalse(decision.is_human_review_recommended)

    def test_deterministic_empty_primary_fallback_exhaustion_flag_unreliable(self) -> None:
        """Primary empty prediction where fallback candidate ALSO predicts empty produces FLAG_UNRELIABLE."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_det_003",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_empty},
            axial_context=ctx_supported,
        )

        decision = evaluate_slice_policy(state, self.config)

        # UNCONDITIONAL SAFETY ASSERTIONS
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertNotEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIsNone(decision.mean_uncertainty, "Empty prediction must not have synthetic zero uncertainty.")
        self.assertTrue(decision.is_human_review_recommended)
        self.assertIn("no_viable_fallback_candidate_with_foreground", decision.reasons)

    def test_deterministic_mocked_inference_agent_safety(self) -> None:
        """Verify InferenceAgent.run_slice with mocked ModelRegistry guarantees empty prediction handling."""
        mock_registry = MagicMock(spec=ModelRegistry)
        mock_registry.default_device = torch.device("cpu")

        # Mock predict_slice: always returns empty prediction for both primary and candidate
        mock_registry.predict_slice.side_effect = [
            self.primary_empty,    # Primary model call
            self.candidate_empty,  # Candidate model call
        ]

        agent = InferenceAgent(
            registry=mock_registry,
            primary_model="resunet",
            config=self.config,
            device="cpu",
        )

        # Case predictions: 2 neighbors have foreground support -> triggers fallback query
        neighbor_preds = {
            48: _make_prediction("resunet", dominant_class=0),
            49: _make_prediction("resunet", dominant_class=1),
            51: _make_prediction("resunet", dominant_class=1),
            52: _make_prediction("resunet", dominant_class=0),
        }

        dummy_image = torch.zeros((1, 4, 16, 16), dtype=torch.float32)
        decision = agent.run_slice(
            image=dummy_image,
            case_id="case_mock_001",
            slice_index=50,
            case_primary_predictions=neighbor_preds,
            case_total_slices=155,
        )

        # UNCONDITIONAL SAFETY ASSERTIONS
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertNotEqual(decision.action, AgentAction.ACCEPT)
        self.assertTrue(decision.has_empty_foreground)
        self.assertIsNone(decision.mean_uncertainty)
        self.assertTrue(decision.is_human_review_recommended)
        self.assertIn("no_viable_fallback_candidate_with_foreground", decision.reasons)


if __name__ == "__main__":
    unittest.main()
