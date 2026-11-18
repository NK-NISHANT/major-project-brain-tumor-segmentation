"""Stage C Reliability Agent Integration & Frozen C3 Policy Verification Suite.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Scientific Authority: Stage C3 v1.1.3 (Option 1 - Preserved C3 Decision Semantics)
Batch 1C: Frozen C3 Reliability Agent Integration.

Requirements Verified:
  6.1 Initialization tests (1-10)
  6.2 Decision tests (11-20, including Option 1 Test 18)
  6.3 Axial context tests (21-25)
  6.4 Regression & safety tests (26-30)

All tests use synthetic tensors, mocked registry calls, and CPU-only fixtures.
No GPU memory or real MRI volumes are accessed.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from types import MappingProxyType

import torch

from src.agent.calibration import (
    DEFAULT_C2_CALIBRATION_JSON,
    DEFAULT_C2_RUN_MANIFEST,
    EXPECTED_CALIBRATION_JSON_SHA256,
    EXPECTED_C2_MANIFEST_SHA256,
    STAGE_C2_REFERENCE_TEMPERATURES,
    StageCCalibrationConfig,
    load_stage_c_calibration,
)
from src.agent.models import (
    STAGE_C_EXPECTED_HASHES,
    CheckpointProfile,
    ModelRegistry,
    normalize_model_name,
)
from src.agent.policy import evaluate_slice_policy, extract_slice_prediction
from src.agent.runner import (
    InferenceAgent,
    create_stage_c_agent,
    create_stage_c_policy_config,
)
from src.agent.state import (
    STAGE_C_ALLOWED_MODELS,
    STAGE_C_AXIAL_CONTEXT_RADIUS,
    STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
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


def _make_synthetic_logits(
    dominant_class: int,
    spatial_size: tuple[int, int] = (16, 16),
    confidence_scale: float = 8.0,
    num_classes: int = 4,
) -> torch.Tensor:
    """Create controlled synthetic logits tensor of shape [1, C, H, W]."""
    h, w = spatial_size
    logits = torch.zeros((1, num_classes, h, w), dtype=torch.float32)
    logits[:, dominant_class, :, :] = confidence_scale
    return logits


def _make_synthetic_prediction(
    model_name: str,
    dominant_class: int,
    spatial_size: tuple[int, int] = (16, 16),
    confidence_scale: float = 8.0,
    forced_entropy: float | None = None,
) -> SlicePrediction:
    """Create typed SlicePrediction with deterministic foreground or background."""
    logits = _make_synthetic_logits(dominant_class, spatial_size, confidence_scale)
    pred = extract_slice_prediction(model_name=model_name, logits=logits)
    if forced_entropy is not None:
        object.__setattr__(pred, "mean_foreground_entropy", forced_entropy)
    return pred


def _make_mock_registry(
    profile: CheckpointProfile = CheckpointProfile.STAGE_C_15EP,
) -> MagicMock:
    """Create a lightweight mock ModelRegistry with correct profile and paths."""
    mock_reg = MagicMock(spec=ModelRegistry)
    mock_reg.profile = profile
    mock_reg.default_device = torch.device("cpu")
    mock_reg.checkpoint_paths = {
        m: Path(f"mock/path/{m}/best_model.pt") for m in STAGE_C_EXPECTED_HASHES
    }
    return mock_reg


class TestStageCAgentInitialization(unittest.TestCase):
    """Section 6.1: Initialization tests (Tests 1-10)."""

    def setUp(self) -> None:
        self.calib_config = load_stage_c_calibration()

    def test_01_agent_selects_stage_c_15ep(self) -> None:
        """Test 1: Agent created via create_stage_c_agent selects CheckpointProfile.STAGE_C_15EP."""
        mock_reg = _make_mock_registry(CheckpointProfile.STAGE_C_15EP)
        agent = create_stage_c_agent(
            registry=mock_reg,
            calibration_config=self.calib_config,
        )
        self.assertEqual(agent.registry.profile, CheckpointProfile.STAGE_C_15EP)
        self.assertTrue(agent.is_stage_c)

    def test_02_legacy_default_not_used(self) -> None:
        """Test 2: Agent rejects legacy profiles (LEGACY_DEFAULT, GENUINE_5EP)."""
        mock_legacy = _make_mock_registry(CheckpointProfile.LEGACY_DEFAULT)
        with self.assertRaises(ValueError) as ctx:
            create_stage_c_agent(registry=mock_legacy, calibration_config=self.calib_config)
        self.assertIn("requires CheckpointProfile.STAGE_C_15EP", str(ctx.exception))

        mock_5ep = _make_mock_registry(CheckpointProfile.GENUINE_5EP)
        with self.assertRaises(ValueError) as ctx2:
            create_stage_c_agent(registry=mock_5ep, calibration_config=self.calib_config)
        self.assertIn("requires CheckpointProfile.STAGE_C_15EP", str(ctx2.exception))

    def test_03_primary_model_is_resunet(self) -> None:
        """Test 3: Primary model is strictly ResUNet-2D ('resunet')."""
        mock_reg = _make_mock_registry()
        agent = create_stage_c_agent(registry=mock_reg, calibration_config=self.calib_config)
        self.assertEqual(agent.primary_model, "resunet")

        # Directly creating a Stage C agent with unapproved primary model raises ValueError
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=mock_reg,
                primary_model="unet",
                config=agent.config,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("primary model must be 'resunet'", str(ctx.exception))

    def test_04_secondary_model_is_unet_resnet34(self) -> None:
        """Test 4: Secondary model is strictly UNetResNet34 ('unet_resnet34')."""
        mock_reg = _make_mock_registry()
        agent = create_stage_c_agent(registry=mock_reg, calibration_config=self.calib_config)
        self.assertEqual(agent.secondary_model, "unet_resnet34")
        self.assertEqual(agent.config.fallback_priority, ("unet_resnet34",))

    def test_05_calibration_loader_uses_strict_default_hash_verification(self) -> None:
        """Test 5: Calibration loader uses strict default cryptographic SHA-256 validation."""
        self.assertEqual(
            self.calib_config.calibration_json_sha256,
            EXPECTED_CALIBRATION_JSON_SHA256,
        )
        self.assertEqual(
            self.calib_config.manifest_sha256,
            EXPECTED_C2_MANIFEST_SHA256,
        )

    def test_06_primary_temperature_retrieved_dynamically(self) -> None:
        """Test 6: Primary temperature is retrieved dynamically from Stage C2 calibration."""
        mock_reg = _make_mock_registry()
        agent = create_stage_c_agent(registry=mock_reg, calibration_config=self.calib_config)
        expected_t = self.calib_config.temperature_for("resunet")
        self.assertAlmostEqual(agent.primary_temperature, expected_t, places=6)
        self.assertAlmostEqual(agent.primary_temperature, 2.880459, places=5)

    def test_07_secondary_temperature_retrieved_dynamically(self) -> None:
        """Test 7: Secondary temperature is retrieved dynamically from Stage C2 calibration."""
        mock_reg = _make_mock_registry()
        agent = create_stage_c_agent(registry=mock_reg, calibration_config=self.calib_config)
        expected_t = self.calib_config.temperature_for("unet_resnet34")
        self.assertIsNotNone(agent.secondary_temperature)
        assert agent.secondary_temperature is not None
        self.assertAlmostEqual(agent.secondary_temperature, expected_t, places=6)
        self.assertAlmostEqual(agent.secondary_temperature, 2.626091, places=5)

    def test_08_calibration_checkpoint_identity_mismatch_rejected(self) -> None:
        """Test 8: Identity mismatch between calibration config and registry profile raises ValueError."""
        # Forge a calibration config with corrupted checkpoint hash
        corrupt_hashes = dict(self.calib_config.checkpoint_hashes)
        corrupt_hashes["resunet"] = "0000000000000000000000000000000000000000000000000000000000000000"
        corrupt_calib = StageCCalibrationConfig(
            temperatures=self.calib_config.temperatures,
            checkpoint_hashes=corrupt_hashes,
            display_names=self.calib_config.display_names,
            fit_diagnostics=self.calib_config.fit_diagnostics,
            calibration_json_sha256=self.calib_config.calibration_json_sha256,
            manifest_sha256=self.calib_config.manifest_sha256,
            splits_json_sha256=self.calib_config.splits_json_sha256,
            timestamp=self.calib_config.timestamp,
        )
        mock_reg = _make_mock_registry()
        with self.assertRaises(ValueError) as ctx:
            create_stage_c_agent(registry=mock_reg, calibration_config=corrupt_calib)
        self.assertIn("identity mismatch for 'resunet'", str(ctx.exception))

    def test_09_invalid_calibration_configuration_fails_initialization(self) -> None:
        """Test 9: Invalid calibration path or missing calibration fails initialization."""
        mock_reg = _make_mock_registry()
        with self.assertRaises(FileNotFoundError):
            create_stage_c_agent(
                registry=mock_reg,
                calib_path="results/non_existent_calibration.json",
            )

    def test_10_no_alternative_architecture_silently_introduced(self) -> None:
        """Test 10: Attempting to introduce unapproved architectures in fallback raises ValueError."""
        mock_reg = _make_mock_registry()
        agent = create_stage_c_agent(registry=mock_reg, calibration_config=self.calib_config)

        # Attempting fallback priority with unapproved models
        invalid_cfg = PolicyConfig(
            high_uncertainty_threshold=0.35,
            fallback_priority=("unet",),
            enable_axial_context=True,
            primary_temperature=agent.primary_temperature,
            candidate_temperatures={"unet": 2.6267},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=mock_reg,
                primary_model="resunet",
                config=invalid_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("fallback priority must be strictly ('unet_resnet34',)", str(ctx.exception))


class TestStageCDecisionSemantics(unittest.TestCase):
    """Section 6.2: Decision tests (Tests 11-20, Option 1 Semantics)."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.calib_config = load_stage_c_calibration()
        self.mock_reg = _make_mock_registry()
        self.agent = create_stage_c_agent(
            registry=self.mock_reg,
            calibration_config=self.calib_config,
        )
        self.primary_empty = _make_synthetic_prediction("resunet", dominant_class=0)
        self.candidate_fg = _make_synthetic_prediction(
            "unet_resnet34", dominant_class=1, confidence_scale=10.0, forced_entropy=0.22
        )
        self.candidate_empty = _make_synthetic_prediction("unet_resnet34", dominant_class=0)

    def test_11_valid_primary_prediction_produces_accept(self) -> None:
        """Test 11: Valid non-empty primary with entropy < 0.35 produces ACCEPT action."""
        primary_fg = _make_synthetic_prediction(
            "resunet", dominant_class=2, confidence_scale=12.0, forced_entropy=0.20
        )
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=primary_fg,
        )
        decision = evaluate_slice_policy(state, self.agent.config)
        self.assertEqual(decision.action, AgentAction.ACCEPT)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertAlmostEqual(decision.mean_uncertainty, 0.20)
        self.assertFalse(decision.is_human_review_recommended)
        self.assertIn("nominal_acceptance", decision.reasons)
        self.assertEqual(decision.metadata["primary_model"], "resunet")
        self.assertFalse(decision.metadata["fallback_invoked"])

    def test_12_high_primary_entropy_produces_flag_unreliable(self) -> None:
        """Test 12: Primary prediction with entropy >= 0.35 produces FLAG_UNRELIABLE."""
        primary_high_ent = _make_synthetic_prediction(
            "resunet", dominant_class=2, confidence_scale=1.0, forced_entropy=0.45
        )
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=primary_high_ent,
        )
        decision = evaluate_slice_policy(state, self.agent.config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertFalse(decision.has_empty_foreground)
        self.assertTrue(decision.is_human_review_recommended)
        self.assertTrue(any("warning_high_uncertainty" in r for r in decision.reasons))
        self.assertFalse(decision.metadata["fallback_invoked"])

    def test_13_high_entropy_alone_does_not_trigger_fallback(self) -> None:
        """Test 13: High primary entropy alone NEVER triggers secondary fallback."""
        primary_high_ent = _make_synthetic_prediction(
            "resunet", dominant_class=2, confidence_scale=1.0, forced_entropy=0.45
        )
        self.mock_reg.predict_slice.side_effect = [primary_high_ent]

        dummy_img = torch.randn((1, 4, 32, 32))
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 1),
            69: _make_synthetic_prediction("resunet", 1),
        }
        decision = self.agent.run_slice(
            image=dummy_img,
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        # Secondary model was NOT called
        self.assertEqual(self.mock_reg.predict_slice.call_count, 1)
        self.assertFalse(decision.metadata["fallback_invoked"])

    def test_14_empty_primary_with_axial_support_triggers_fallback(self) -> None:
        """Test 14: Empty primary with qualifying axial context triggers fallback to UNetResNet34."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty, self.candidate_fg]

        dummy_img = torch.randn((1, 4, 32, 32))
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 1),
            69: _make_synthetic_prediction("resunet", 1),
        }
        decision = self.agent.run_slice(
            image=dummy_img,
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet_resnet34")
        self.assertEqual(self.mock_reg.predict_slice.call_count, 2)
        self.assertTrue(decision.metadata["fallback_invoked"])
        self.assertTrue(decision.metadata["axial_support_confirmed"])

    def test_15_empty_primary_without_axial_support_follows_authoritative_rule(self) -> None:
        """Test 15: Empty primary without axial context support produces FLAG_UNRELIABLE (suppressed)."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty]

        dummy_img = torch.randn((1, 4, 32, 32))
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 0),
            69: _make_synthetic_prediction("resunet", 0),
            71: _make_synthetic_prediction("resunet", 0),
            72: _make_synthetic_prediction("resunet", 0),
        }
        decision = self.agent.run_slice(
            image=dummy_img,
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)
        # Secondary model was NOT called
        self.assertEqual(self.mock_reg.predict_slice.call_count, 1)
        self.assertFalse(decision.metadata["fallback_invoked"])

    def test_16_successful_secondary_prediction_produces_fallback_switch(self) -> None:
        """Test 16: Successful secondary recovery produces FALLBACK_SWITCH, not ACCEPT."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_fg},
            axial_context=ctx_supported,
        )
        decision = evaluate_slice_policy(state, self.agent.config)
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet_resnet34")
        self.assertFalse(decision.has_empty_foreground)
        self.assertIn("fallback_selected_unet_resnet34", decision.reasons)

    def test_17_empty_secondary_prediction_produces_flag_unreliable(self) -> None:
        """Test 17: When secondary model also predicts empty foreground, agent flags UNRELIABLE."""
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": self.candidate_empty},
            axial_context=ctx_supported,
        )
        decision = evaluate_slice_policy(state, self.agent.config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "resunet")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIn("no_viable_fallback_candidate_with_foreground", decision.reasons)
        self.assertTrue(decision.is_human_review_recommended)

    def test_18_secondary_entropy_above_tau_retains_fallback_switch_with_warning(self) -> None:
        """Test 18 (Revised Option 1): Secondary prediction with H >= 0.35 retains FALLBACK_SWITCH with warning.

        Preserves exact authoritative Stage C3 v1.1.3 scientific semantics (where 146 of 147 fallback
        switches exhibited secondary entropy >= 0.35 and were retained as FALLBACK_SWITCH).
        """
        candidate_high_ent = _make_synthetic_prediction(
            "unet_resnet34", dominant_class=1, confidence_scale=1.0, forced_entropy=0.48
        )
        ctx_supported = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=2,
            empty_neighbors_count=2,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet_resnet34": candidate_high_ent},
            axial_context=ctx_supported,
        )
        decision = evaluate_slice_policy(state, self.agent.config)
        # Option 1 Mandate: Action must remain FALLBACK_SWITCH
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet_resnet34")
        self.assertFalse(decision.has_empty_foreground)
        self.assertAlmostEqual(decision.mean_uncertainty, 0.48)
        # Must record high-uncertainty warning
        self.assertIn("candidate_high_uncertainty_warning", decision.reasons)
        self.assertIn("fallback_selected_unet_resnet34", decision.reasons)

    def test_19_empty_primary_never_receives_synthetic_zero_entropy(self) -> None:
        """Test 19: Empty foreground prediction strictly has mean_foreground_entropy is None, never 0.0."""
        self.assertTrue(self.primary_empty.has_empty_foreground)
        self.assertEqual(self.primary_empty.pred_foreground_pixels, 0)
        self.assertIsNone(self.primary_empty.mean_foreground_entropy)

    def test_20_primary_and_secondary_entropy_evaluated_with_correct_temperatures(self) -> None:
        """Test 20: Primary and secondary temperatures are passed to predict_slice during workflow."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty, self.candidate_fg]

        dummy_img = torch.randn((1, 4, 32, 32))
        ctx_map = {69: _make_synthetic_prediction("resunet", 1)}
        self.agent.run_slice(
            image=dummy_img,
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        # Check call arguments to predict_slice
        calls = self.mock_reg.predict_slice.call_args_list
        self.assertEqual(len(calls), 2)

        # Primary call args: model='resunet', temperature=primary_temperature
        p_call_kwargs = calls[0].kwargs
        self.assertEqual(p_call_kwargs["model_name"], "resunet")
        self.assertAlmostEqual(p_call_kwargs["temperature"], 2.880459, places=5)

        # Secondary call args: model='unet_resnet34', temperature=secondary_temperature
        s_call_kwargs = calls[1].kwargs
        self.assertEqual(s_call_kwargs["model_name"], "unet_resnet34")
        self.assertAlmostEqual(s_call_kwargs["temperature"], 2.626091, places=5)


class TestStageCAxialContext(unittest.TestCase):
    """Section 6.3: Axial context tests (Tests 21-25)."""

    def setUp(self) -> None:
        torch.manual_seed(42)
        self.calib_config = load_stage_c_calibration()
        self.mock_reg = _make_mock_registry()
        self.agent = create_stage_c_agent(
            registry=self.mock_reg,
            calibration_config=self.calib_config,
        )
        self.primary_empty = _make_synthetic_prediction("resunet", dominant_class=0)
        self.candidate_fg = _make_synthetic_prediction("unet_resnet34", dominant_class=1)

    def test_21_radius_k2_correctly_respected(self) -> None:
        """Test 21: Context inspects exactly z-2, z-1, z+1, z+2."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty, self.candidate_fg]
        # Only z+2 has foreground
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 0),
            69: _make_synthetic_prediction("resunet", 0),
            71: _make_synthetic_prediction("resunet", 0),
            72: _make_synthetic_prediction("resunet", 1),  # z + 2
        }
        decision = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertIsNotNone(decision.axial_context)
        assert decision.axial_context is not None
        self.assertEqual(decision.axial_context.evaluated_neighbors_count, 4)
        self.assertEqual(decision.axial_context.foreground_neighbors_count, 1)

    def test_22_slices_outside_radius_do_not_trigger_fallback(self) -> None:
        """Test 22: Foreground slices at delta outside [-2, 2] do NOT provide support."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty]
        # Only z+3 has foreground, [-2, 2] are all empty
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 0),
            69: _make_synthetic_prediction("resunet", 0),
            71: _make_synthetic_prediction("resunet", 0),
            72: _make_synthetic_prediction("resunet", 0),
            73: _make_synthetic_prediction("resunet", 1),  # z + 3 (outside radius 2!)
        }
        decision = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)
        self.assertEqual(self.mock_reg.predict_slice.call_count, 1)

    def test_23_current_slice_not_counted_as_neighbor(self) -> None:
        """Test 23: Current slice z is strictly excluded from neighbor accounting."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty]
        # Current slice z=70 in predictions map with foreground should be ignored
        ctx_map = {
            68: _make_synthetic_prediction("resunet", 0),
            69: _make_synthetic_prediction("resunet", 0),
            70: _make_synthetic_prediction("resunet", 1),  # Self-slice!
            71: _make_synthetic_prediction("resunet", 0),
            72: _make_synthetic_prediction("resunet", 0),
        }
        decision = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=70,
            case_primary_predictions=ctx_map,
        )
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        assert decision.axial_context is not None
        self.assertEqual(decision.axial_context.foreground_neighbors_count, 0)
        self.assertEqual(decision.axial_context.empty_neighbors_count, 4)

    def test_24_boundary_slices_handled_safely(self) -> None:
        """Test 24: Slices at z=0 and z=154 evaluate safely without index out of bounds."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty, self.primary_empty]

        # Slice z=0: can only evaluate z=1 and z=2 (2 neighbors evaluated, 2 unavailable)
        ctx_z0 = {
            1: _make_synthetic_prediction("resunet", 0),
            2: _make_synthetic_prediction("resunet", 0),
        }
        dec_z0 = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=0,
            case_primary_predictions=ctx_z0,
            case_total_slices=155,
        )
        self.assertEqual(dec_z0.action, AgentAction.FLAG_UNRELIABLE)
        assert dec_z0.axial_context is not None
        self.assertEqual(dec_z0.axial_context.evaluated_neighbors_count, 2)
        self.assertEqual(dec_z0.axial_context.unavailable_neighbors_count, 2)

        # Slice z=154: can only evaluate z=152 and z=153
        ctx_z154 = {
            152: _make_synthetic_prediction("resunet", 0),
            153: _make_synthetic_prediction("resunet", 0),
        }
        dec_z154 = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=154,
            case_primary_predictions=ctx_z154,
            case_total_slices=155,
        )
        self.assertEqual(dec_z154.action, AgentAction.FLAG_UNRELIABLE)
        assert dec_z154.axial_context is not None
        self.assertEqual(dec_z154.axial_context.evaluated_neighbors_count, 2)
        self.assertEqual(dec_z154.axial_context.unavailable_neighbors_count, 2)

    def test_25_missing_neighbor_context_handled_per_frozen_semantics(self) -> None:
        """Test 25: Completely missing/unevaluated neighbor context suppresses fallback."""
        self.mock_reg.predict_slice.side_effect = [self.primary_empty]
        # None passed for case_primary_predictions
        dec = self.agent.run_slice(
            image=torch.randn((1, 4, 16, 16)),
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=None,
        )
        self.assertEqual(dec.action, AgentAction.FLAG_UNRELIABLE)
        assert dec.axial_context is not None
        self.assertEqual(dec.axial_context.evaluated_neighbors_count, 0)
        self.assertEqual(dec.axial_context.unavailable_neighbors_count, 4)
        self.assertIn("axial_context_suppressed_fallback", dec.reasons)
        self.assertEqual(self.mock_reg.predict_slice.call_count, 1)


class TestStageCRegressionAndSafety(unittest.TestCase):
    """Section 6.4: Regression and safety invariant tests (Tests 26-30)."""

    def setUp(self) -> None:
        self.calib_config = load_stage_c_calibration()
        self.mock_reg = _make_mock_registry()
        self.agent = create_stage_c_agent(
            registry=self.mock_reg,
            calibration_config=self.calib_config,
        )

    def test_26_registry_profile_invariants_pass(self) -> None:
        """Test 26: Stage C profile invariants and SHA-256 expectations remain intact."""
        self.assertEqual(len(STAGE_C_EXPECTED_HASHES), 5)
        for model in ("resunet", "unet_resnet34", "unet", "unetpp", "attention_unet"):
            self.assertIn(model, STAGE_C_EXPECTED_HASHES)
            self.assertEqual(len(STAGE_C_EXPECTED_HASHES[model]), 64)

    def test_27_calibration_loader_properties_and_immutability_pass(self) -> None:
        """Test 27: Calibration loader returns strictly frozen mapping proxy structures."""
        self.assertIsInstance(self.calib_config.temperatures, MappingProxyType)
        with self.assertRaises(TypeError):
            self.calib_config.temperatures["resunet"] = 1.0  # type: ignore

    def test_28_existing_agent_input_validation_remains_functional(self) -> None:
        """Test 28: Input tensor validation in run_slice rejects invalid types and dimensions."""
        # Non-tensor
        with self.assertRaises(TypeError):
            self.agent.run_slice(
                image="invalid_input",  # type: ignore
                case_id="case_001",
                slice_index=0,
            )

    def test_29_typed_state_action_interfaces_remain_compatible(self) -> None:
        """Test 29: AgentDecision, AgentState, and PolicyConfig retain full backward compatibility."""
        dec = AgentDecision(
            action=AgentAction.ACCEPT,
            selected_model="resunet",
            final_mask=torch.zeros((16, 16), dtype=torch.uint8),
            mean_uncertainty=0.15,
            has_empty_foreground=False,
            reasons=("nominal_acceptance",),
        )
        self.assertEqual(dec.action, AgentAction.ACCEPT)
        self.assertEqual(dec.selected_model, "resunet")
        self.assertFalse(dec.is_human_review_recommended)

    def test_30_only_permitted_models_in_live_fallback_chain(self) -> None:
        """Test 30: Only ResUNet-2D and UNetResNet34 are permitted in the Stage C live chain."""
        self.assertEqual(self.agent.primary_model, "resunet")
        self.assertEqual(self.agent.secondary_model, "unet_resnet34")

        # Prohibited primary model
        with self.assertRaises(ValueError):
            InferenceAgent(
                registry=self.mock_reg,
                primary_model="unetpp",
                config=self.agent.config,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )

        # Prohibited fallback priority
        invalid_cfg = PolicyConfig(
            high_uncertainty_threshold=0.35,
            fallback_priority=("unetpp",),
            enable_axial_context=True,
            primary_temperature=self.agent.primary_temperature,
            candidate_temperatures={"unetpp": 2.5647},
        )
        with self.assertRaises(ValueError):
            InferenceAgent(
                registry=self.mock_reg,
                primary_model="resunet",
                config=invalid_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )


class TestStageCConfigurationInvariantHardening(unittest.TestCase):
    """Section 6.5: Batch 1C.2 Invariant Hardening & Immutability Regression Tests (Tests 31-42)."""

    def setUp(self) -> None:
        self.calib_config = load_stage_c_calibration()
        self.mock_reg = _make_mock_registry()
        self.agent = create_stage_c_agent(
            registry=self.mock_reg,
            calibration_config=self.calib_config,
        )

    def test_31_non_authoritative_tau_rejected(self) -> None:
        """Test 31: Overriding tau in factories or InferenceAgent with is_stage_c=True raises ValueError."""
        with self.assertRaises(ValueError) as ctx1:
            create_stage_c_policy_config(self.calib_config, tau=0.50)
        self.assertIn("requires high_uncertainty_threshold tau=0.35", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            create_stage_c_agent(registry=self.mock_reg, calibration_config=self.calib_config, tau=0.20)
        self.assertIn("requires high_uncertainty_threshold tau=0.35", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            InferenceAgent.create_stage_c(registry=self.mock_reg, calibration_config=self.calib_config, tau=0.40)
        self.assertIn("requires high_uncertainty_threshold tau=0.35", str(ctx3.exception))

        # Direct construction with invalid tau
        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=0.50,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx4:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires high_uncertainty_threshold=0.35", str(ctx4.exception))

    def test_32_non_authoritative_axial_window_rejected(self) -> None:
        """Test 32: Overriding axial_context_window in factories or InferenceAgent raises ValueError."""
        with self.assertRaises(ValueError) as ctx1:
            create_stage_c_policy_config(self.calib_config, axial_context_window=3)
        self.assertIn("requires axial_context_window=2", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            create_stage_c_agent(registry=self.mock_reg, calibration_config=self.calib_config, axial_context_window=1)
        self.assertIn("requires axial_context_window=2", str(ctx2.exception))

        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=3,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx3:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires axial_context_window=2", str(ctx3.exception))

    def test_33_non_authoritative_min_foreground_neighbors_rejected(self) -> None:
        """Test 33: Overriding min_neighbor_foreground_slices in factories or InferenceAgent raises ValueError."""
        with self.assertRaises(ValueError) as ctx1:
            create_stage_c_policy_config(self.calib_config, min_neighbor_foreground_slices=2)
        self.assertIn("requires min_neighbor_foreground_slices=1", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            create_stage_c_agent(registry=self.mock_reg, calibration_config=self.calib_config, min_neighbor_foreground_slices=2)
        self.assertIn("requires min_neighbor_foreground_slices=1", str(ctx2.exception))

        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=2,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx3:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires min_neighbor_foreground_slices=1", str(ctx3.exception))

    def test_34_disabled_axial_context_rejected(self) -> None:
        """Test 34: Stage C agent rejects enable_axial_context=False."""
        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=False,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires enable_axial_context=True", str(ctx.exception))

    def test_35_disabled_empty_prediction_fallback_rejected(self) -> None:
        """Test 35: Stage C agent rejects enable_empty_prediction_fallback=False."""
        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=False,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires enable_empty_prediction_fallback=True", str(ctx.exception))

    def test_36_fallback_when_context_unavailable_true_rejected(self) -> None:
        """Test 36: Stage C agent rejects fallback_when_context_unavailable=True."""
        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            fallback_when_context_unavailable=True,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires fallback_when_context_unavailable=False", str(ctx.exception))

    def test_37_uncertainty_gated_fallback_true_rejected(self) -> None:
        """Test 37: Stage C agent rejects enable_uncertainty_fallback=True."""
        bad_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            enable_uncertainty_fallback=True,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("does not permit uncertainty-gated fallback (must be False)", str(ctx.exception))

    def test_38_non_four_classes_rejected(self) -> None:
        """Test 38: Stage C agent rejects num_classes != 4."""
        bad_cfg2 = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=2,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_cfg2,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("requires num_classes=4", str(ctx.exception))

    def test_39_temperature_mismatch_with_c2_rejected(self) -> None:
        """Test 39: Temperature mismatch with verified C2 calibration artifact is rejected."""
        # Primary temperature mismatch
        bad_p_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=1.0,
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL)},
        )
        with self.assertRaises(ValueError) as ctx1:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_p_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("does not match verified Stage C2 calibration temperature", str(ctx1.exception))

        # Secondary temperature mismatch
        bad_s_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={STAGE_C_SECONDARY_MODEL: 1.0},
        )
        with self.assertRaises(ValueError) as ctx2:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_s_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("does not match verified Stage C2 calibration temperature", str(ctx2.exception))

        # Extra candidate model in temperature mapping
        bad_extra_cfg = PolicyConfig(
            high_uncertainty_threshold=STAGE_C_TAU,
            enable_empty_prediction_fallback=True,
            fallback_priority=(STAGE_C_SECONDARY_MODEL,),
            num_classes=4,
            enable_axial_context=True,
            axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
            min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
            primary_temperature=self.calib_config.temperature_for(STAGE_C_PRIMARY_MODEL),
            candidate_temperatures={
                STAGE_C_SECONDARY_MODEL: self.calib_config.temperature_for(STAGE_C_SECONDARY_MODEL),
                "unet": 2.6267,
            },
        )
        with self.assertRaises(ValueError) as ctx3:
            InferenceAgent(
                registry=self.mock_reg,
                primary_model=STAGE_C_PRIMARY_MODEL,
                config=bad_extra_cfg,
                calibration_config=self.calib_config,
                is_stage_c=True,
            )
        self.assertIn("must contain exclusively", str(ctx3.exception))

    def test_40_stage_c2_authoritative_temperatures_retained(self) -> None:
        """Test 40: Stage C agent retains exact authoritative C2 calibration temperatures."""
        ref_p = STAGE_C2_REFERENCE_TEMPERATURES["resunet"]
        ref_s = STAGE_C2_REFERENCE_TEMPERATURES["unet_resnet34"]
        self.assertAlmostEqual(self.agent.primary_temperature, ref_p, places=6)
        self.assertAlmostEqual(self.agent.secondary_temperature, ref_s, places=6)
        self.assertEqual(self.agent.primary_temperature, 2.8804590702056885)
        self.assertEqual(self.agent.secondary_temperature, 2.6260907649993896)

    def test_41_candidate_temperatures_mapping_immutability(self) -> None:
        """Test 41: candidate_temperatures is genuinely immutable and defensively copied."""
        # 1. Returned mapping is MappingProxyType
        self.assertIsInstance(self.agent.config.candidate_temperatures, MappingProxyType)

        # 2. Item assignment rejected
        with self.assertRaises(TypeError):
            self.agent.config.candidate_temperatures["unet_resnet34"] = 999.0  # type: ignore

        # 3. New key insertion rejected
        with self.assertRaises(TypeError):
            self.agent.config.candidate_temperatures["new_model"] = 1.0  # type: ignore

        # 4. Deletion rejected
        with self.assertRaises(TypeError):
            del self.agent.config.candidate_temperatures["unet_resnet34"]  # type: ignore

        # 5. Clear rejected
        with self.assertRaises(AttributeError):
            self.agent.config.candidate_temperatures.clear()  # type: ignore

        # 6. Defensive copying: external dict mutation does not affect PolicyConfig
        external_dict = {"unet_resnet34": 2.6260907649993896}
        cfg = PolicyConfig(candidate_temperatures=external_dict)
        external_dict["unet_resnet34"] = 999.0
        self.assertEqual(cfg.candidate_temperatures["unet_resnet34"], 2.6260907649993896)

    def test_42_historical_and_custom_agent_compatibility_preserved(self) -> None:
        """Test 42: Non-Stage C agent (is_stage_c=False) preserves historical flexibility."""
        hist_reg = _make_mock_registry(CheckpointProfile.HISTORICAL_5EP)
        hist_cfg = PolicyConfig(
            high_uncertainty_threshold=0.50,
            enable_axial_context=False,
            fallback_priority=("unet", "unetpp"),
            primary_temperature=1.2,
            candidate_temperatures={"unet": 1.1, "unetpp": 1.0},
        )
        hist_agent = InferenceAgent(
            registry=hist_reg,
            primary_model="unetpp",
            config=hist_cfg,
            is_stage_c=False,
        )
        self.assertEqual(hist_agent.primary_model, "unetpp")
        self.assertEqual(hist_agent.secondary_model, "unet")
        self.assertEqual(hist_agent.primary_temperature, 1.2)
        self.assertEqual(hist_agent.secondary_temperature, 1.1)


if __name__ == "__main__":
    unittest.main()
