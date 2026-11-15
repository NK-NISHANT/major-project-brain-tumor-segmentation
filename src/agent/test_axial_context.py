from __future__ import annotations

import unittest
from unittest.mock import MagicMock
import torch

from src.agent.models import ModelRegistry
from src.agent.policy import evaluate_slice_policy, extract_slice_prediction
from src.agent.runner import InferenceAgent
from src.agent.state import (
    AgentAction,
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
    h, w = spatial_size
    logits = torch.zeros((1, num_classes, h, w), dtype=torch.float32)
    logits[:, dominant_class, :, :] = confidence_scale
    return logits


def _make_prediction(
    model_name: str,
    dominant_class: int,
    spatial_size: tuple[int, int] = (16, 16),
) -> SlicePrediction:
    logits = _make_synthetic_logits(dominant_class, spatial_size)
    return extract_slice_prediction(model_name=model_name, logits=logits)


class TestAxialContextPolicy(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.primary_empty = _make_prediction("unetpp", dominant_class=0)
        self.candidate_fg = _make_prediction("unet", dominant_class=1)

    def test_default_axial_context_disabled_backward_compatibility(self) -> None:
        """With enable_axial_context=False (default), empty-primary fallback proceeds normally."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=False,
        )
        # Even if axial_context is passed in state, policy ignores it when enable_axial_context=False
        ctx_all_empty = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=0,
            empty_neighbors_count=4,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_all_empty,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet")
        self.assertIn("fallback_selected_unet", decision.reasons)

    def test_axial_context_suppresses_fallback_when_neighbors_empty(self) -> None:
        """When enable_axial_context=True and evaluated neighbors are all empty, fallback is suppressed."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        ctx_all_empty = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=0,
            empty_neighbors_count=4,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_all_empty,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertEqual(decision.selected_model, "unetpp")
        self.assertTrue(decision.has_empty_foreground)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)
        self.assertIn("evaluated_empty_neighbors_4", decision.reasons)
        self.assertTrue(decision.metadata.get("fallback_suppressed_by_context"))

    def test_axial_context_allows_fallback_when_neighbor_has_foreground(self) -> None:
        """When at least min_neighbor_foreground_slices has foreground, fallback is allowed."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        ctx_with_fg = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=1,
            empty_neighbors_count=3,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_with_fg,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision.selected_model, "unet")
        self.assertIn("axial_context_supported_by_1_neighbors", decision.reasons)
        self.assertIn("fallback_selected_unet", decision.reasons)

    def test_min_neighbor_foreground_threshold_boundary(self) -> None:
        """When min_neighbor_foreground_slices=2, 1 neighbor with foreground is insufficient."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            min_neighbor_foreground_slices=2,
        )
        ctx_1_fg = AxialContextSummary(
            evaluated_neighbors_count=4,
            unavailable_neighbors_count=0,
            foreground_neighbors_count=1,
            empty_neighbors_count=3,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_1_fg,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)

    def test_all_unavailable_fallback_permitted(self) -> None:
        """When all neighbors are unavailable and fallback_when_context_unavailable=True, fallback proceeds."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            fallback_when_context_unavailable=True,
        )
        ctx_unavail = AxialContextSummary(
            evaluated_neighbors_count=0,
            unavailable_neighbors_count=4,
            foreground_neighbors_count=0,
            empty_neighbors_count=0,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_unavail,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FALLBACK_SWITCH)
        self.assertIn("axial_context_unavailable_fallback_permitted", decision.reasons)

    def test_all_unavailable_fallback_suppressed(self) -> None:
        """When all neighbors are unavailable and fallback_when_context_unavailable=False, fallback is suppressed."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            fallback_when_context_unavailable=False,
        )
        ctx_unavail = AxialContextSummary(
            evaluated_neighbors_count=0,
            unavailable_neighbors_count=4,
            foreground_neighbors_count=0,
            empty_neighbors_count=0,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_unavail,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)

    def test_all_unavailable_fallback_suppressed_by_default(self) -> None:
        """In Milestone 5 Option A, fallback_when_context_unavailable defaults to False, suppressing fallback."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            # fallback_when_context_unavailable omitted -> must default to False
        )
        self.assertFalse(config.fallback_when_context_unavailable)

        ctx_unavail = AxialContextSummary(
            evaluated_neighbors_count=0,
            unavailable_neighbors_count=4,
            foreground_neighbors_count=0,
            empty_neighbors_count=0,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_unavail,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)

    def test_config_validation(self) -> None:
        """Validation of axial context configuration parameters."""
        with self.assertRaises(ValueError):
            PolicyConfig(axial_context_window=0)
        with self.assertRaises(ValueError):
            PolicyConfig(min_neighbor_foreground_slices=0)

    def test_option_b_explicit_configuration_and_isolation(self) -> None:
        """Option B (k=3) is enabled only when explicitly configured; Option A (k=2) remains default."""
        default_cfg = PolicyConfig()
        self.assertEqual(default_cfg.axial_context_window, 2)
        self.assertFalse(default_cfg.fallback_when_context_unavailable)

        option_b_cfg = PolicyConfig(axial_context_window=3)
        self.assertEqual(option_b_cfg.axial_context_window, 3)
        self.assertFalse(option_b_cfg.fallback_when_context_unavailable)

    def test_option_b_unavailable_context_suppresses_fallback(self) -> None:
        """Under Option B (k=3, unavail=False), completely unavailable context suppresses fallback."""
        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=3,
            fallback_when_context_unavailable=False,
        )
        ctx_unavail = AxialContextSummary(
            evaluated_neighbors_count=0,
            unavailable_neighbors_count=6,
            foreground_neighbors_count=0,
            empty_neighbors_count=0,
        )
        state = AgentState(
            case_id="case_001",
            slice_index=50,
            primary_prediction=self.primary_empty,
            candidate_predictions={"unet": self.candidate_fg},
            axial_context=ctx_unavail,
        )
        decision = evaluate_slice_policy(state=state, config=config)
        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision.reasons)


class TestInferenceAgentAxialContext(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.dummy_image = torch.zeros((1, 4, 16, 16), dtype=torch.float32)

    def test_runner_computes_axial_context_window_and_status(self) -> None:
        """InferenceAgent correctly counts evaluated empty and foreground neighbors within window k=2."""
        registry = MagicMock(spec=ModelRegistry)
        registry.default_device = torch.device("cpu")

        # Mock primary prediction to return empty foreground
        empty_pred = _make_prediction("unetpp", dominant_class=0)
        registry.predict_slice.return_value = empty_pred

        config = PolicyConfig(
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        agent = InferenceAgent(registry=registry, primary_model="unetpp", config=config)

        # Neighbor predictions for slices 48, 49, 51, 52 (slice 50 is center)
        # Slices 48, 49, 52 are empty; slice 51 has foreground
        case_preds = {
            48: _make_prediction("unetpp", dominant_class=0),
            49: _make_prediction("unetpp", dominant_class=0),
            51: _make_prediction("unetpp", dominant_class=1),
            52: _make_prediction("unetpp", dominant_class=0),
        }

        decision = agent.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )

        ctx = decision.axial_context
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.evaluated_neighbors_count, 4)
        self.assertEqual(ctx.unavailable_neighbors_count, 0)
        self.assertEqual(ctx.foreground_neighbors_count, 1)
        self.assertEqual(ctx.empty_neighbors_count, 3)

    def test_runner_distinguishes_unavailable_from_evaluated_neighbors(self) -> None:
        """Missing slice predictions are marked unavailable, NOT evaluated empty."""
        registry = MagicMock(spec=ModelRegistry)
        registry.default_device = torch.device("cpu")
        empty_pred = _make_prediction("unetpp", dominant_class=0)
        registry.predict_slice.return_value = empty_pred

        config = PolicyConfig(
            enable_axial_context=True,
            axial_context_window=2,
        )
        agent = InferenceAgent(registry=registry, primary_model="unetpp", config=config)

        # Only slices 49 and 51 are available in case_preds; 48 and 52 are missing
        case_preds = {
            49: _make_prediction("unetpp", dominant_class=0),
            51: _make_prediction("unetpp", dominant_class=0),
        }

        decision = agent.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )

        ctx = decision.axial_context
        self.assertIsNotNone(ctx)
        assert ctx is not None
        self.assertEqual(ctx.evaluated_neighbors_count, 2)
        self.assertEqual(ctx.unavailable_neighbors_count, 2)
        self.assertEqual(ctx.empty_neighbors_count, 2)
        self.assertEqual(ctx.foreground_neighbors_count, 0)

    def test_runner_handles_volume_physical_boundaries(self) -> None:
        """Slices outside [0, case_total_slices - 1] are excluded from neighbor window."""
        registry = MagicMock(spec=ModelRegistry)
        registry.default_device = torch.device("cpu")
        empty_pred = _make_prediction("unetpp", dominant_class=0)
        registry.predict_slice.return_value = empty_pred

        config = PolicyConfig(
            enable_axial_context=True,
            axial_context_window=2,
        )
        agent = InferenceAgent(registry=registry, primary_model="unetpp", config=config)

        # Slice 0: neighbors at -2, -1 are out of range. In range: slices 1, 2
        case_preds = {
            1: _make_prediction("unetpp", dominant_class=0),
        }
        decision = agent.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=0,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )
        ctx = decision.axial_context
        self.assertIsNotNone(ctx)
        assert ctx is not None
        # In-bounds neighbors are only slice 1 and 2 (total 2 neighbors)
        self.assertEqual(ctx.evaluated_neighbors_count, 1)  # slice 1 was in case_preds
        self.assertEqual(ctx.unavailable_neighbors_count, 1)  # slice 2 was missing
        self.assertEqual(ctx.evaluated_neighbors_count + ctx.unavailable_neighbors_count, 2)

    def test_runner_skips_candidate_query_when_context_suppresses_fallback(self) -> None:
        """Optimization: If axial context suppresses fallback, runner does not query candidate models."""
        registry = MagicMock(spec=ModelRegistry)
        registry.default_device = torch.device("cpu")

        primary_empty = _make_prediction("unetpp", dominant_class=0)
        registry.predict_slice.return_value = primary_empty

        config = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet", "attention_unet"),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
        )
        agent = InferenceAgent(registry=registry, primary_model="unetpp", config=config)

        # All neighbors evaluated and empty -> fallback suppressed
        case_preds = {
            48: _make_prediction("unetpp", dominant_class=0),
            49: _make_prediction("unetpp", dominant_class=0),
            51: _make_prediction("unetpp", dominant_class=0),
            52: _make_prediction("unetpp", dominant_class=0),
        }

        decision = agent.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )

        self.assertEqual(decision.action, AgentAction.FLAG_UNRELIABLE)
        # Verify predict_slice was called EXACTLY ONCE (for primary unetpp only), never for unet or attention_unet
        self.assertEqual(registry.predict_slice.call_count, 1)
        registry.predict_slice.assert_called_once_with(
            model_name="unetpp",
            image=self.dummy_image,
            device=torch.device("cpu"),
        )

    def test_option_b_runner_window_k3_recovers_wider_continuity(self) -> None:
        """With k=3 (Option B), foreground support at delta=+3 enables fallback where k=2 (Option A) would not."""
        registry = MagicMock(spec=ModelRegistry)
        registry.default_device = torch.device("cpu")
        empty_pred = _make_prediction("unetpp", dominant_class=0)
        fg_pred = _make_prediction("unet", dominant_class=1)
        registry.predict_slice.side_effect = lambda model_name, **kwargs: empty_pred if model_name == "unetpp" else fg_pred

        # Neighbors: 48, 49, 51, 52 are empty; slice 53 has foreground
        case_preds = {
            48: _make_prediction("unetpp", dominant_class=0),
            49: _make_prediction("unetpp", dominant_class=0),
            51: _make_prediction("unetpp", dominant_class=0),
            52: _make_prediction("unetpp", dominant_class=0),
            53: _make_prediction("unetpp", dominant_class=1),
        }

        # Under Option A (k=2): slice 53 is out of range, all evaluated neighbors in range are empty -> suppressed
        config_a = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=2,
            fallback_when_context_unavailable=False,
        )
        agent_a = InferenceAgent(registry=registry, primary_model="unetpp", config=config_a)
        decision_a = agent_a.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )
        self.assertEqual(decision_a.action, AgentAction.FLAG_UNRELIABLE)
        self.assertIn("axial_context_suppressed_fallback", decision_a.reasons)

        # Under Option B (k=3): slice 53 is in range -> foreground support detected -> fallback permitted
        config_b = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=3,
            fallback_when_context_unavailable=False,
        )
        agent_b = InferenceAgent(registry=registry, primary_model="unetpp", config=config_b)
        decision_b = agent_b.run_slice(
            image=self.dummy_image,
            case_id="case_001",
            slice_index=50,
            case_primary_predictions=case_preds,
            case_total_slices=155,
        )
        self.assertEqual(decision_b.action, AgentAction.FALLBACK_SWITCH)
        self.assertEqual(decision_b.selected_model, "unet")
        self.assertIn("axial_context_supported_by_1_neighbors", decision_b.reasons)


class TestGlobalMetricAccumulator(unittest.TestCase):
    def test_global_accumulator_perfect_match(self) -> None:
        from src.agent.evaluate_milestone4 import GlobalMetricAccumulator, compute_slice_metrics

        acc = GlobalMetricAccumulator(num_classes=4)
        mask = torch.zeros((16, 16), dtype=torch.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels NCR
        mask[2:4, 2:4] = 2  # 4 pixels ED
        mask[4:6, 4:6] = 3  # 4 pixels ET

        acc.update(mask, mask)
        metrics = acc.compute()

        self.assertAlmostEqual(metrics["global_micro_dice_foreground"], 1.0, places=5)
        self.assertAlmostEqual(metrics["global_dice_wt"], 1.0, places=5)
        self.assertAlmostEqual(metrics["global_dice_tc"], 1.0, places=5)
        self.assertAlmostEqual(metrics["global_dice_et"], 1.0, places=5)

    def test_global_micro_dice_vs_slice_macro_dice(self) -> None:
        from src.agent.evaluate_milestone4 import GlobalMetricAccumulator

        acc = GlobalMetricAccumulator(num_classes=4)
        # Slice 1: Big slice (100 pixels), perfect overlap
        t1 = torch.zeros((20, 20), dtype=torch.uint8)
        t1[0:10, 0:10] = 1  # 100 pixels
        p1 = t1.clone()

        # Slice 2: Small slice (10 pixels), complete miss (predicted 0)
        t2 = torch.zeros((20, 20), dtype=torch.uint8)
        t2[0:2, 0:5] = 1  # 10 pixels
        p2 = torch.zeros((20, 20), dtype=torch.uint8)

        acc.update(p1, t1)
        acc.update(p2, t2)
        metrics = acc.compute()

        # Slice 1 Dice = 1.0, Slice 2 Dice = 0.0 -> Slice Macro Mean = 0.50
        # Global Micro Dice = 2 * 100 / (2 * 100 + 0 + 10) = 200 / 210 = 0.9524!
        # This proves the exact mathematical behavior of volumetric vs slice-level aggregation
        self.assertAlmostEqual(metrics["global_micro_dice_foreground"], 200.0 / 210.0, places=4)

    def test_compute_slice_composite_metrics(self) -> None:
        from src.agent.evaluate_milestone4 import compute_slice_metrics

        tgt = torch.zeros((10, 10), dtype=torch.uint8)
        tgt[0:2, 0:2] = 1  # 4 pixels NCR
        tgt[2:4, 0:2] = 2  # 4 pixels ED
        tgt[4:6, 0:2] = 3  # 4 pixels ET
        # Total WT = 12 pixels, TC = 8 pixels, ET = 4 pixels

        # Pred matches WT perfectly, but predicts all foreground as class 3
        pred = torch.zeros((10, 10), dtype=torch.uint8)
        pred[0:6, 0:2] = 3  # 12 pixels ET

        mean_fg_dice, _, wt_dice, tc_dice, et_dice = compute_slice_metrics(pred, tgt)
        # WT overlap is 12 / 12 = 1.0
        self.assertAlmostEqual(wt_dice, 1.0, places=5)
        # ET: target has 4, pred has 12, overlap is 4 -> Dice = 2*4 / (12 + 4) = 0.50
        self.assertAlmostEqual(et_dice, 0.50, places=5)


class TestMilestone4SafetyAndIntegrity(unittest.TestCase):
    def test_ground_truth_isolation_signature(self) -> None:
        import inspect
        sig = inspect.signature(InferenceAgent.run_slice)
        param_names = list(sig.parameters.keys())
        self.assertIn("image", param_names)
        self.assertIn("case_id", param_names)
        self.assertIn("slice_index", param_names)
        self.assertIn("case_primary_predictions", param_names)

        # Confirm NO ground-truth parameter exists
        for forbidden in ["target", "gt", "ground_truth", "mask", "target_mask", "label"]:
            self.assertNotIn(forbidden, param_names)

    def test_milestone3_artifacts_preserved(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        m3_dir = root / "results" / "agent_evaluation"
        self.assertTrue((m3_dir / "evaluation_summary.json").exists())
        self.assertTrue((m3_dir / "report.md").exists())
        self.assertTrue((m3_dir / "audit_report.md").exists())
        self.assertTrue((m3_dir / "slice_evaluations.csv").exists())

    def test_milestone4_artifacts_preserved(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        m4_dir = root / "results" / "agent_evaluation" / "milestone4"
        self.assertTrue((m4_dir / "evaluation_summary.json").exists())
        self.assertTrue((m4_dir / "report.md").exists())
        self.assertTrue((m4_dir / "slice_evaluations.csv").exists())

    def test_milestone5_artifacts_preserved(self) -> None:
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        m5_dir = root / "results" / "agent_evaluation" / "milestone5"
        self.assertTrue((m5_dir / "evaluation_summary.json").exists())
        self.assertTrue((m5_dir / "report.md").exists())
        self.assertTrue((m5_dir / "slice_evaluations.csv").exists())


if __name__ == "__main__":
    unittest.main()

