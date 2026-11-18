"""Unit tests for tools/demo_app.py agent integration, channel layout, and policy correctness.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from typing import Any
import numpy as np
import torch

from src.agent.calibration import STAGE_C2_REFERENCE_TEMPERATURES
from src.agent.models import CheckpointProfile, ModelRegistry
from src.agent.policy import evaluate_slice_policy, extract_slice_prediction
from src.agent.runner import InferenceAgent
from src.agent.state import (
    STAGE_C_PRIMARY_MODEL,
    STAGE_C_SECONDARY_MODEL,
    STAGE_C_TAU,
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.data.dataset import MODALITIES
from tools.demo_app import (
    HEALTHY_SLICE_DICE_NOTICE,
    MODALITY_DISPLAY_SPECS,
    NON_TUMOR_SLICE_DICE_NOTICE,
    compute_slice_dice_metrics,
    format_decision_badge_html,
    format_healthy_slice_notice_html,
    get_model_registry,
    get_stage_c_agent,
    get_validation_cases,
    load_calibrated_policy_config,
    should_display_healthy_slice_notice,
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
    """Verify loading and fail-closed behavior of calibrated policy configurations."""

    def test_load_calibrated_policy_config(self) -> None:
        cfg, meta = load_calibrated_policy_config(tau_threshold=0.35)
        self.assertIsInstance(cfg, PolicyConfig)
        self.assertAlmostEqual(cfg.primary_temperature, STAGE_C2_REFERENCE_TEMPERATURES["resunet"], places=4)
        self.assertIn("unet_resnet34", cfg.candidate_temperatures)
        self.assertAlmostEqual(
            cfg.candidate_temperatures["unet_resnet34"],
            STAGE_C2_REFERENCE_TEMPERATURES["unet_resnet34"],
            places=4,
        )
        self.assertEqual(cfg.high_uncertainty_threshold, 0.35)
        self.assertTrue(cfg.enable_empty_prediction_fallback)
        self.assertTrue(cfg.enable_axial_context)
        self.assertEqual(cfg.axial_context_window, 2)
        self.assertEqual(cfg.min_neighbor_foreground_slices, 1)

    def test_load_calibrated_policy_config_fails_closed_on_missing_artifact(self) -> None:
        """Must fail closed with FileNotFoundError if calibration artifact is missing."""
        with self.assertRaises(FileNotFoundError):
            load_calibrated_policy_config(calib_path=Path("results/non_existent_calibration.json"))

    def test_load_calibrated_policy_config_rejects_tau_override(self) -> None:
        """Must fail closed with ValueError if non-authoritative tau is requested."""
        with self.assertRaises(ValueError) as cm:
            load_calibrated_policy_config(tau_threshold=0.40)
        self.assertIn("Stage C policy requires high_uncertainty_threshold tau=0.35", str(cm.exception))


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
            calibrated_temperature=2.8805,
            is_human_review_recommended=False,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: ACCEPT", html)
        self.assertIn("Autonomous Retention", html)
        self.assertIn("H = 0.1852 < τ = 0.35", html)
        self.assertNotIn("≤", html)
        self.assertIn("resunet", html)

    def test_format_fallback_badge(self) -> None:
        dec = AgentDecision(
            action=AgentAction.FALLBACK_SWITCH,
            selected_model="unet_resnet34",
            final_mask=torch.ones((16, 16), dtype=torch.uint8),
            mean_uncertainty=0.2215,
            has_empty_foreground=False,
            reasons=("fallback_selected_unet_resnet34",),
            calibrated_temperature=2.6261,
            is_human_review_recommended=False,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: FALLBACK_SWITCH", html)
        self.assertIn("Model Substitution", html)
        self.assertIn("unet_resnet34", html)

    def test_format_fallback_badge_with_candidate_high_uncertainty(self) -> None:
        dec = AgentDecision(
            action=AgentAction.FALLBACK_SWITCH,
            selected_model="unet_resnet34",
            final_mask=torch.ones((16, 16), dtype=torch.uint8),
            mean_uncertainty=0.3842,
            has_empty_foreground=False,
            reasons=("fallback_selected_unet_resnet34", "candidate_high_uncertainty_warning"),
            calibrated_temperature=2.6261,
            is_human_review_recommended=False,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: FALLBACK_SWITCH", html)
        self.assertIn("Candidate High Uncertainty Notice", html)
        self.assertNotIn("ACTION: FLAG_UNRELIABLE", html)

    def test_format_flag_unreliable_badge(self) -> None:
        dec = AgentDecision(
            action=AgentAction.FLAG_UNRELIABLE,
            selected_model="resunet",
            final_mask=torch.zeros((16, 16), dtype=torch.uint8),
            mean_uncertainty=None,
            has_empty_foreground=True,
            reasons=("warning_complete_empty_prediction",),
            calibrated_temperature=2.8805,
            is_human_review_recommended=True,
        )
        html = format_decision_badge_html(dec, tau=0.35, primary_model="ResUNet-2D")
        self.assertIn("ACTION: FLAG_UNRELIABLE", html)
        self.assertIn("Specialist Review Required", html)
        self.assertIn("Clinical Notice", html)
        self.assertIn("statistical prediction uncertainty", html)


class TestEntropyDecisionBoundary(unittest.TestCase):
    """Explicit boundary tests for the frozen C3 v1.1.3 decision boundary (tau = 0.35)."""

    def test_entropy_immediately_below_tau_accepts(self) -> None:
        """Entropy immediately below 0.35 (e.g. 0.349999) must produce ACCEPT."""
        pred = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.349999,
            has_empty_foreground=False,
        )
        state = AgentState(case_id="case_boundary_01", slice_index=50, primary_prediction=pred)
        config = PolicyConfig(high_uncertainty_threshold=0.35)
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertFalse(decision.is_human_review_recommended)
        self.assertIn("nominal_acceptance", decision.reasons)

    def test_entropy_exactly_at_tau_flags_unreliable(self) -> None:
        """Entropy exactly equal to 0.350000 must produce FLAG_UNRELIABLE."""
        pred = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.350000,
            has_empty_foreground=False,
        )
        state = AgentState(case_id="case_boundary_02", slice_index=50, primary_prediction=pred)
        config = PolicyConfig(high_uncertainty_threshold=0.35)
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.is_human_review_recommended)
        self.assertTrue(any("warning_high_uncertainty" in r for r in decision.reasons))

    def test_entropy_immediately_above_tau_flags_unreliable(self) -> None:
        """Entropy immediately above 0.35 (e.g. 0.350001) must produce FLAG_UNRELIABLE."""
        pred = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.350001,
            has_empty_foreground=False,
        )
        state = AgentState(case_id="case_boundary_03", slice_index=50, primary_prediction=pred)
        config = PolicyConfig(high_uncertainty_threshold=0.35)
        decision = evaluate_slice_policy(state, config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.is_human_review_recommended)
        self.assertTrue(any("warning_high_uncertainty" in r for r in decision.reasons))


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


class TestHealthySliceDiceInterpretationNotice(unittest.TestCase):
    """Verify conditional display and exact text of the non-tumor slice Dice interpretation notice."""

    def test_both_masks_empty_displays_notice(self) -> None:
        """Condition 1: Both ground truth and predicted masks empty -> notice displayed."""
        gt_empty = np.zeros((240, 240), dtype=np.uint8)
        pred_empty = np.zeros((240, 240), dtype=np.uint8)
        self.assertTrue(should_display_healthy_slice_notice(pred_empty, gt_empty))
        html = format_healthy_slice_notice_html()
        self.assertIn(NON_TUMOR_SLICE_DICE_NOTICE, html)
        self.assertIn("Non-Tumor Slice Metric Notice:", html)
        self.assertIn("Dice = 1.000 reflects empty-empty agreement", html)

    def test_ground_truth_empty_prediction_non_empty_omits_notice(self) -> None:
        """Condition 2: Ground truth empty, prediction non-empty -> notice not displayed."""
        gt_empty = np.zeros((240, 240), dtype=np.uint8)
        pred_tumor = np.zeros((240, 240), dtype=np.uint8)
        pred_tumor[50:60, 50:60] = 2  # Edema
        self.assertFalse(should_display_healthy_slice_notice(pred_tumor, gt_empty))

    def test_ground_truth_non_empty_prediction_empty_omits_notice(self) -> None:
        """Condition 3: Ground truth non-empty, prediction empty -> notice not displayed."""
        gt_tumor = np.zeros((240, 240), dtype=np.uint8)
        gt_tumor[50:60, 50:60] = 1  # Necrotic core
        pred_empty = np.zeros((240, 240), dtype=np.uint8)
        self.assertFalse(should_display_healthy_slice_notice(pred_empty, gt_tumor))

    def test_both_masks_contain_tumor_omits_notice(self) -> None:
        """Condition 4: Both ground truth and prediction contain tumor -> notice not displayed."""
        gt_tumor = np.zeros((240, 240), dtype=np.uint8)
        gt_tumor[50:60, 50:60] = 3  # Enhancing tumor
        pred_tumor = np.zeros((240, 240), dtype=np.uint8)
        pred_tumor[50:60, 50:60] = 3
        self.assertFalse(should_display_healthy_slice_notice(pred_tumor, gt_tumor))

    def test_notice_text_exact_wording(self) -> None:
        """Notice text must match the exact clinical specification verbatim."""
        expected = (
            "Both ground-truth and predicted masks are empty on this slice. "
            "Dice = 1.000 reflects empty-empty agreement, not demonstrated tumor segmentation accuracy."
        )
        self.assertEqual(NON_TUMOR_SLICE_DICE_NOTICE, expected)
        html = format_healthy_slice_notice_html()
        self.assertIn(f"<b>Non-Tumor Slice Metric Notice:</b> {expected}", html)


class TestDemoAppStageCIntegration(unittest.TestCase):
    """Verify Streamlit Stage C agent factory and policy invariant enforcement."""

    def test_get_model_registry_requires_stage_c_15ep(self) -> None:
        """get_model_registry must reject legacy or historical profiles."""
        reg = get_model_registry("cpu", profile=CheckpointProfile.STAGE_C_15EP)
        self.assertEqual(reg.profile, CheckpointProfile.STAGE_C_15EP)

        with self.assertRaises(ValueError) as cm:
            get_model_registry("cpu", profile=CheckpointProfile.HISTORICAL_5EP)
        self.assertIn("strictly requires CheckpointProfile.STAGE_C_15EP", str(cm.exception))

    def test_get_stage_c_agent_invariants(self) -> None:
        """get_stage_c_agent must produce an InferenceAgent with frozen C3 invariants."""
        agent = get_stage_c_agent("cpu")
        self.assertIsInstance(agent, InferenceAgent)
        self.assertEqual(agent.primary_model, STAGE_C_PRIMARY_MODEL)
        self.assertEqual(agent.config.fallback_priority, (STAGE_C_SECONDARY_MODEL,))
        self.assertEqual(agent.config.high_uncertainty_threshold, STAGE_C_TAU)
        self.assertEqual(agent.config.axial_context_window, 2)
        self.assertEqual(agent.config.min_neighbor_foreground_slices, 1)
        self.assertTrue(agent.config.enable_axial_context)
        self.assertTrue(agent.config.enable_empty_prediction_fallback)
        self.assertAlmostEqual(agent.primary_temperature, STAGE_C2_REFERENCE_TEMPERATURES["resunet"], places=4)
        self.assertAlmostEqual(
            agent.config.candidate_temperatures[STAGE_C_SECONDARY_MODEL],
            STAGE_C2_REFERENCE_TEMPERATURES["unet_resnet34"],
            places=4,
        )

    def test_mocked_stage_c_agent_execution_all_actions(self) -> None:
        """Verify mocked run_slice execution under all three Stage C actions."""
        agent = get_stage_c_agent("cpu")
        dummy_slice = torch.randn((1, 4, 32, 32), dtype=torch.float32)

        # 1. Nominal ACCEPT
        low_ent_pred = _make_prediction("resunet", dominant_class=2, confidence_scale=12.0)
        with patch.object(agent.registry, "predict_slice", return_value=low_ent_pred):
            dec = agent.run_slice(
                image=dummy_slice,
                case_id="case_test",
                slice_index=50,
                case_primary_predictions={},
                case_total_slices=155,
            )
            self.assertEqual(dec.action, AgentAction.ACCEPT)
            self.assertEqual(dec.selected_model, "resunet")
            self.assertAlmostEqual(dec.calibrated_temperature, STAGE_C2_REFERENCE_TEMPERATURES["resunet"], places=6)
            self.assertFalse(dec.is_human_review_recommended)

        # 2. FALLBACK_SWITCH with axial context support
        empty_pred = _make_prediction("resunet", dominant_class=0)
        cand_fg_pred = _make_prediction("unet_resnet34", dominant_class=1, confidence_scale=10.0)

        def mock_predict(model_name: str, **kwargs: Any) -> SlicePrediction:
            if model_name == "resunet":
                return empty_pred
            return cand_fg_pred

        neighbor_preds = {
            49: _make_prediction("resunet", dominant_class=1),
            51: _make_prediction("resunet", dominant_class=0),
        }
        with patch.object(agent.registry, "predict_slice", side_effect=mock_predict):
            dec = agent.run_slice(
                image=dummy_slice,
                case_id="case_test",
                slice_index=50,
                case_primary_predictions=neighbor_preds,
                case_total_slices=155,
            )
            self.assertEqual(dec.action, AgentAction.FALLBACK_SWITCH)
            self.assertEqual(dec.selected_model, "unet_resnet34")
            self.assertAlmostEqual(
                dec.calibrated_temperature, STAGE_C2_REFERENCE_TEMPERATURES["unet_resnet34"], places=6
            )
            self.assertFalse(dec.is_human_review_recommended)

        # 3. FALLBACK_SWITCH with high candidate uncertainty (Option 1 behavior)
        h_logits = torch.ones((1, 4, 16, 16), dtype=torch.float32)
        h_logits[:, 1, :, :] = 1.05  # Class 1 barely dominant -> high entropy >= 0.35
        cand_high_ent = extract_slice_prediction("unet_resnet34", h_logits)
        self.assertGreaterEqual(cand_high_ent.mean_foreground_entropy or 0.0, 0.35)

        def mock_predict_high_ent(model_name: str, **kwargs: Any) -> SlicePrediction:
            if model_name == "resunet":
                return empty_pred
            return cand_high_ent

        with patch.object(agent.registry, "predict_slice", side_effect=mock_predict_high_ent):
            dec = agent.run_slice(
                image=dummy_slice,
                case_id="case_test",
                slice_index=50,
                case_primary_predictions=neighbor_preds,
                case_total_slices=155,
            )
            self.assertEqual(dec.action, AgentAction.FALLBACK_SWITCH)
            self.assertEqual(dec.selected_model, "unet_resnet34")
            self.assertAlmostEqual(
                dec.calibrated_temperature, STAGE_C2_REFERENCE_TEMPERATURES["unet_resnet34"], places=6
            )
            self.assertIn("candidate_high_uncertainty_warning", dec.reasons)
            self.assertFalse(dec.is_human_review_recommended)

        # 4. FLAG_UNRELIABLE due to unsupported empty prediction
        empty_neighbors = {
            49: _make_prediction("resunet", dominant_class=0),
            51: _make_prediction("resunet", dominant_class=0),
        }
        with patch.object(agent.registry, "predict_slice", side_effect=mock_predict):
            dec = agent.run_slice(
                image=dummy_slice,
                case_id="case_test",
                slice_index=50,
                case_primary_predictions=empty_neighbors,
                case_total_slices=155,
            )
            self.assertEqual(dec.action, AgentAction.FLAG_UNRELIABLE)
            self.assertEqual(dec.selected_model, "resunet")
            self.assertAlmostEqual(dec.calibrated_temperature, STAGE_C2_REFERENCE_TEMPERATURES["resunet"], places=6)
            self.assertTrue(dec.is_human_review_recommended)

    def test_stage_c_agent_entropy_boundary_execution(self) -> None:
        """Verify InferenceAgent boundary execution: 0.3499 -> ACCEPT, 0.3500 & 0.3501 -> FLAG_UNRELIABLE."""
        agent = get_stage_c_agent("cpu")
        dummy_slice = torch.randn((1, 4, 16, 16), dtype=torch.float32)

        # 1. H = 0.3499 -> strictly below tau=0.35 -> ACCEPT
        pred_below = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.3499,
            has_empty_foreground=False,
        )
        with patch.object(agent.registry, "predict_slice", return_value=pred_below):
            dec = agent.run_slice(image=dummy_slice, case_id="c_test", slice_index=50)
            self.assertEqual(dec.action, AgentAction.ACCEPT)
            self.assertEqual(dec.selected_model, "resunet")
            self.assertFalse(dec.is_human_review_recommended)

        # 2. H = 0.3500 -> exactly at tau=0.35 -> FLAG_UNRELIABLE
        pred_exact = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.3500,
            has_empty_foreground=False,
        )
        with patch.object(agent.registry, "predict_slice", return_value=pred_exact):
            dec = agent.run_slice(image=dummy_slice, case_id="c_test", slice_index=50)
            self.assertEqual(dec.action, AgentAction.FLAG_UNRELIABLE)
            self.assertEqual(dec.selected_model, "resunet")
            self.assertTrue(dec.is_human_review_recommended)

        # 3. H = 0.3501 -> strictly above tau=0.35 -> FLAG_UNRELIABLE
        pred_above = SlicePrediction(
            model_name="resunet",
            logits=torch.zeros((1, 4, 16, 16)),
            pred_mask=torch.ones((16, 16), dtype=torch.uint8),
            pred_foreground_pixels=256,
            mean_foreground_entropy=0.3501,
            has_empty_foreground=False,
        )
        with patch.object(agent.registry, "predict_slice", return_value=pred_above):
            dec = agent.run_slice(image=dummy_slice, case_id="c_test", slice_index=50)
            self.assertEqual(dec.action, AgentAction.FLAG_UNRELIABLE)
            self.assertEqual(dec.selected_model, "resunet")
            self.assertTrue(dec.is_human_review_recommended)

    def test_stage_c_agent_runtime_inference_error_propagates(self) -> None:
        """Verify that neural network inference errors propagate rather than silently substituting models."""
        agent = get_stage_c_agent("cpu")
        dummy_slice = torch.randn((1, 4, 16, 16), dtype=torch.float32)

        with patch.object(
            agent.registry, "predict_slice", side_effect=RuntimeError("Simulated neural network forward failure")
        ):
            with self.assertRaises(RuntimeError) as cm:
                agent.run_slice(image=dummy_slice, case_id="c_test", slice_index=50)
            self.assertIn("Simulated neural network forward failure", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
