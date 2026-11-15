"""Unit tests for tools/demo_app.py agent integration, channel layout, and policy correctness.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
"""

from __future__ import annotations

import unittest
from pathlib import Path
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
from src.data.dataset import MODALITIES
from tools.demo_app import (
    MODALITY_DISPLAY_SPECS,
    compute_slice_dice_metrics,
    create_policy_config_for_model,
    format_decision_badge_html,
    get_validation_cases,
    load_calibrated_policy_config,
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


class TestModalityChannelOrdering(unittest.TestCase):
    """Verify MRI channel mapping matches canonical dataset specification exactly."""

    def test_channel_order_matches_dataset_modalities(self) -> None:
        """Channel 0=t1n, 1=t1c, 2=t2f, 3=t2w must strictly match MODALITIES."""
        self.assertEqual(len(MODALITY_DISPLAY_SPECS), 4)
        for expected_ch, (label, code, ch_idx, desc) in enumerate(MODALITY_DISPLAY_SPECS):
            self.assertEqual(ch_idx, expected_ch)
            self.assertEqual(code, MODALITIES[expected_ch])

    def test_specific_flair_and_t2w_channel_assignment(self) -> None:
        """Explicitly confirm T2-FLAIR is channel 2 and T2-Weighted is channel 3."""
        ch2 = MODALITY_DISPLAY_SPECS[2]
        ch3 = MODALITY_DISPLAY_SPECS[3]

        self.assertEqual(ch2[1], "t2f")
        self.assertIn("FLAIR", ch2[0])
        self.assertEqual(ch2[2], 2)

        self.assertEqual(ch3[1], "t2w")
        self.assertIn("T2-Weighted", ch3[0])
        self.assertEqual(ch3[2], 3)


class TestInputTensorValidation(unittest.TestCase):
    """Verify strict input tensor validation contract."""

    def test_non_tensor_raises_type_error(self) -> None:
        with self.assertRaises(TypeError):
            validate_input_slice_tensor(np.zeros((1, 4, 32, 32)))  # type: ignore

    def test_invalid_ndim_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(torch.zeros((4, 32, 32)))

    def test_invalid_batch_dimension_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(torch.zeros((2, 4, 32, 32)))

    def test_invalid_channels_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(torch.zeros((1, 3, 32, 32)))
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(torch.zeros((1, 5, 32, 32)))

    def test_non_positive_spatial_dimensions_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(torch.zeros((1, 4, 0, 32)))

    def test_nan_or_inf_raises_value_error(self) -> None:
        t_nan = torch.zeros((1, 4, 32, 32))
        t_nan[0, 0, 5, 5] = float("nan")
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(t_nan)

        t_inf = torch.zeros((1, 4, 32, 32))
        t_inf[0, 0, 5, 5] = float("inf")
        with self.assertRaises(ValueError):
            validate_input_slice_tensor(t_inf)

    def test_valid_tensor_passes(self) -> None:
        t_valid = torch.randn((1, 4, 240, 240), dtype=torch.float32)
        # Should execute without raising
        validate_input_slice_tensor(t_valid)


class TestCalibrationConfigurationLoading(unittest.TestCase):
    """Verify loading and adaptation of calibrated policy configurations."""

    def test_load_calibrated_policy_config(self) -> None:
        cfg, meta = load_calibrated_policy_config(tau_threshold=0.35)
        self.assertIsInstance(cfg, PolicyConfig)
        self.assertAlmostEqual(cfg.primary_temperature, 2.8808, places=3)
        self.assertIn("unet_resnet34", cfg.candidate_temperatures)
        self.assertAlmostEqual(cfg.candidate_temperatures["unet_resnet34"], 2.6267, places=3)
        self.assertEqual(cfg.high_uncertainty_threshold, 0.35)
        self.assertTrue(cfg.enable_empty_prediction_fallback)
        self.assertTrue(cfg.enable_axial_context)

    def test_create_policy_config_for_model(self) -> None:
        base_cfg, _ = load_calibrated_policy_config()

        # ResUNet configuration
        cfg_resunet = create_policy_config_for_model("resunet", base_cfg, tau_threshold=0.35)
        self.assertAlmostEqual(cfg_resunet.primary_temperature, 2.8808, places=3)
        self.assertEqual(cfg_resunet.fallback_priority[0], "unet_resnet34")

        # UNetResNet34 configuration
        cfg_resnet34 = create_policy_config_for_model("unet_resnet34", base_cfg, tau_threshold=0.35)
        self.assertAlmostEqual(cfg_resnet34.primary_temperature, 2.6267, places=3)
        self.assertEqual(cfg_resnet34.fallback_priority[0], "resunet")

        # Standard UNet configuration
        cfg_unet = create_policy_config_for_model("unet", base_cfg, tau_threshold=0.35)
        self.assertEqual(cfg_unet.primary_temperature, 1.0)


class TestEmptyPredictionAndUncertaintyPolicy(unittest.TestCase):
    """Verify correct policy actions for empty predictions and high uncertainty."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.primary_empty = _make_prediction("resunet", dominant_class=0)
        self.candidate_fg = _make_prediction("unet_resnet34", dominant_class=1)
        self.candidate_empty = _make_prediction("unet_resnet34", dominant_class=0)

    def test_empty_prediction_never_receives_synthetic_zero_entropy(self) -> None:
        """An empty foreground prediction must strictly have mean_foreground_entropy is None, never 0.0."""
        self.assertTrue(self.primary_empty.has_empty_foreground)
        self.assertEqual(self.primary_empty.pred_foreground_pixels, 0)
        self.assertIsNone(self.primary_empty.mean_foreground_entropy)

    def test_nominal_acceptance_low_entropy(self) -> None:
        """Low uncertainty non-empty prediction produces ACCEPT action."""
        primary_fg = _make_prediction("resunet", dominant_class=2, confidence_scale=12.0)
        state = AgentState(case_id="case_001", slice_index=72, primary_prediction=primary_fg)
        config = PolicyConfig(high_uncertainty_threshold=0.35)

        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIsNotNone(decision.mean_uncertainty)
        assert decision.mean_uncertainty is not None
        self.assertLess(decision.mean_uncertainty, 0.35)
        self.assertFalse(decision.is_human_review_recommended)

    def test_high_entropy_triggers_flag_unreliable(self) -> None:
        """High uncertainty non-empty prediction produces FLAG_UNRELIABLE action."""
        # Dominant class with low confidence scale creates high entropy across classes
        h_logits = torch.ones((1, 4, 16, 16), dtype=torch.float32)
        h_logits[:, 1, :, :] = 1.05  # Class 1 barely dominant -> near-maximum entropy
        primary_uncertain = extract_slice_prediction("resunet", h_logits)

        state = AgentState(case_id="case_001", slice_index=72, primary_prediction=primary_uncertain)
        config = PolicyConfig(high_uncertainty_threshold=0.35)

        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.is_human_review_recommended)
        self.assertIn("warning_high_uncertainty", decision.reasons[0])

    def test_empty_primary_with_axial_context_triggers_fallback_switch(self) -> None:
        """Empty primary supported by adjacent axial tumor triggers FALLBACK_SWITCH."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=72,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_fg},
            axial_context=ctx_supported,
        )
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet_resnet34",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet_resnet34")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("fallback_selected_unet_resnet34", decision.reasons)
        self.assertFalse(decision.is_human_review_recommended)

    def test_empty_primary_without_axial_context_suppresses_fallback(self) -> None:
        """Empty primary with only empty neighbors suppresses fallback and flags UNRELIABLE."""
        ctx_all_empty = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=0,
            empty_neighbors_count=4,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=72,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_fg},
            axial_context=ctx_all_empty,
        )
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet_resnet34",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIsNone(decision.mean_uncertainty)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)
        self.assertTrue(decision.is_human_review_recommended)

    def test_empty_primary_fallback_failure_triggers_referral(self) -> None:
        """When candidate model also predicts empty foreground, agent flags UNRELIABLE."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=72,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_empty},
            axial_context=ctx_supported,
        )
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet_resnet34",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIn("no_viable_fallback_candidate_with_foreground", decision.reasons)
        self.assertTrue(decision.is_human_review_recommended)


class TestStructuredDecisionFormatting(unittest.TestCase):
    """Verify HTML decision badge generation for Streamlit UI."""

    def test_format_accept_badge(self) -> None:
        dec = AgentDecision(
            action=AgentAction.ACCEPT,
            selected_model="resunet",
            final_mask=torch.ones((16, 16), dtype=torch.uint8),
            mean_uncertainty=0.1852,
            has_empty_foreground=False,
            reasons=("nominal_acceptance",),
            calibrated_temperature=2.8808,
            is_human_review_recommended=False,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: ACCEPT", html)
        self.assertIn("Autonomous Retention", html)
        self.assertIn("0.1852", html)
        self.assertIn("resunet", html)

    def test_format_fallback_badge(self) -> None:
        dec = AgentDecision(
            action=AgentAction.FALLBACK_SWITCH,
            selected_model="unet_resnet34",
            final_mask=torch.ones((16, 16), dtype=torch.uint8),
            mean_uncertainty=0.2215,
            has_empty_foreground=False,
            reasons=("fallback_selected_unet_resnet34",),
            calibrated_temperature=2.6267,
            is_human_review_recommended=False,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: FALLBACK_SWITCH", html)
        self.assertIn("Model Substitution", html)
        self.assertIn("unet_resnet34", html)

    def test_format_flag_unreliable_badge(self) -> None:
        dec = AgentDecision(
            action=AgentAction.FLAG_UNRELIABLE,
            selected_model="resunet",
            final_mask=torch.zeros((16, 16), dtype=torch.uint8),
            mean_uncertainty=None,
            has_empty_foreground=True,
            reasons=("warning_complete_empty_prediction",),
            calibrated_temperature=2.8808,
            is_human_review_recommended=True,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: FLAG_UNRELIABLE", html)
        self.assertIn("Specialist Review Required", html)


class TestValidationCohortIntegrity(unittest.TestCase):
    """Verify strict validation cohort discovery without uncontrolled directory fallback."""

    def test_missing_archive_dir_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_validation_cases(archive_dir_override=Path("data/non_existent_archive"))

    def test_missing_splits_manifest_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            get_validation_cases(
                archive_dir_override=Path("data/extracted_archive"),
                splits_path_override=Path("results/non_existent_splits.json"),
            )


class TestSliceDiceMetrics(unittest.TestCase):
    """Verify slice-level Dice calculation accuracy."""

    def test_exact_match_dice(self) -> None:
        mask = np.array([[0, 1], [2, 3]], dtype=np.uint8)
        metrics = compute_slice_dice_metrics(mask, mask)
        self.assertAlmostEqual(metrics["wt_dice"], 1.0)
        self.assertAlmostEqual(metrics["tc_dice"], 1.0)
        self.assertAlmostEqual(metrics["et_dice"], 1.0)
        self.assertAlmostEqual(metrics["macro_fg_dice"], 1.0)

    def test_complete_mismatch_dice(self) -> None:
        pred = np.zeros((2, 2), dtype=np.uint8)
        gt = np.ones((2, 2), dtype=np.uint8)
        metrics = compute_slice_dice_metrics(pred, gt)
        self.assertAlmostEqual(metrics["wt_dice"], 0.0)
        self.assertAlmostEqual(metrics["macro_fg_dice"], 0.0)

    def test_both_empty_dice(self) -> None:
        zeros = np.zeros((2, 2), dtype=np.uint8)
        metrics = compute_slice_dice_metrics(zeros, zeros)
        self.assertAlmostEqual(metrics["wt_dice"], 1.0)
        self.assertTrue(np.isnan(metrics["macro_fg_dice"]))


if __name__ == "__main__":
    unittest.main()
