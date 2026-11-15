from __future__ import annotations

import csv
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.agent.evaluate_full_volume import (
    GlobalMetricAccumulator,
    SliceEvaluationResult,
    compute_slice_metrics,
    evaluate_slice_record,
    generate_full_volume_markdown_report,
    run_full_volume_evaluation,
    sanitize_for_json,
    summarize_system_evaluation,
)
from src.agent.models import DEFAULT_CHECKPOINTS, ModelRegistry
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
from src.data.dataset import CachedCaseVolume, CaseFiles


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


def _get_agent_context(
    slice_index: int,
    case_primary_predictions: dict[int, SlicePrediction],
    context_window: int = 2,
    case_total_slices: int = 155,
) -> AxialContextSummary | None:
    """Helper to query InferenceAgent's axial context computation directly."""
    registry = MagicMock(spec=ModelRegistry)
    registry.default_device = torch.device("cpu")
    primary = case_primary_predictions.get(
        slice_index, _make_prediction("unetpp", dominant_class=0)
    )
    registry.predict_slice.return_value = primary

    config = PolicyConfig(
        enable_axial_context=True,
        axial_context_window=context_window,
        min_neighbor_foreground_slices=1,
    )
    agent = InferenceAgent(registry=registry, primary_model="unetpp", config=config)
    dummy_img = torch.zeros((4, 16, 16), dtype=torch.float32)
    decision = agent.run_slice(
        image=dummy_img,
        case_id="case_test",
        slice_index=slice_index,
        case_primary_predictions=case_primary_predictions,
        case_total_slices=case_total_slices,
    )
    return decision.axial_context


class TestFullVolumeGeometryAndPhysicalBoundaries(unittest.TestCase):
    """Verifies complete 155-slice volume geometry and physical boundary handling."""

    def test_volume_depth_155_slices_ascending(self) -> None:
        """Confirms that standard BraTS volumes have 155 slices evaluated ascending from 0 to 154."""
        total_z = 155
        slices_to_eval = list(range(total_z))
        self.assertEqual(len(slices_to_eval), 155)
        self.assertEqual(slices_to_eval[0], 0)
        self.assertEqual(slices_to_eval[-1], 154)
        self.assertEqual(slices_to_eval, sorted(slices_to_eval))

    def test_physical_boundary_slice_zero_no_false_unavailable(self) -> None:
        """At z=0, backward neighbors (z < 0) are physical volume boundaries and MUST NOT count as unavailable."""
        total_slices = 155
        # Provide predictions for z in [0, 5]
        preds = {z: _make_prediction("unetpp", dominant_class=0) for z in range(6)}

        # Option A: k=2
        ctx_k2 = _get_agent_context(
            slice_index=0,
            case_primary_predictions=preds,
            context_window=2,
            case_total_slices=total_slices,
        )
        self.assertIsNotNone(ctx_k2)
        assert ctx_k2 is not None
        self.assertEqual(ctx_k2.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k2.evaluated_neighbors_count, 2)  # slices 1 and 2
        self.assertEqual(ctx_k2.empty_neighbors_count, 2)
        self.assertEqual(ctx_k2.foreground_neighbors_count, 0)

        # Option B: k=3
        ctx_k3 = _get_agent_context(
            slice_index=0,
            case_primary_predictions=preds,
            context_window=3,
            case_total_slices=total_slices,
        )
        self.assertIsNotNone(ctx_k3)
        assert ctx_k3 is not None
        self.assertEqual(ctx_k3.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k3.evaluated_neighbors_count, 3)  # slices 1, 2, 3
        self.assertEqual(ctx_k3.empty_neighbors_count, 3)
        self.assertEqual(ctx_k3.foreground_neighbors_count, 0)

    def test_physical_boundary_slice_154_no_false_unavailable(self) -> None:
        """At z=154 (last slice), forward neighbors (z >= 155) MUST NOT count as unavailable."""
        total_slices = 155
        # Provide predictions for z in [150, 154]
        preds = {z: _make_prediction("unetpp", dominant_class=0) for z in range(150, 155)}

        # Option A: k=2
        ctx_k2 = _get_agent_context(
            slice_index=154,
            case_primary_predictions=preds,
            context_window=2,
            case_total_slices=total_slices,
        )
        self.assertIsNotNone(ctx_k2)
        assert ctx_k2 is not None
        self.assertEqual(ctx_k2.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k2.evaluated_neighbors_count, 2)  # slices 152 and 153
        self.assertEqual(ctx_k2.empty_neighbors_count, 2)
        self.assertEqual(ctx_k2.foreground_neighbors_count, 0)

        # Option B: k=3
        ctx_k3 = _get_agent_context(
            slice_index=154,
            case_primary_predictions=preds,
            context_window=3,
            case_total_slices=total_slices,
        )
        self.assertIsNotNone(ctx_k3)
        assert ctx_k3 is not None
        self.assertEqual(ctx_k3.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k3.evaluated_neighbors_count, 3)  # slices 151, 152, 153
        self.assertEqual(ctx_k3.empty_neighbors_count, 3)
        self.assertEqual(ctx_k3.foreground_neighbors_count, 0)

    def test_interior_slice_full_neighborhood(self) -> None:
        """For interior slices, evaluated neighbors count equals 2*k and unavailable count is 0."""
        total_slices = 155
        preds = {z: _make_prediction("unetpp", dominant_class=0) for z in range(40, 60)}

        # Option A: k=2 at z=50 -> 4 neighbors: [48, 49, 51, 52]
        ctx_k2 = _get_agent_context(
            slice_index=50,
            case_primary_predictions=preds,
            context_window=2,
            case_total_slices=total_slices,
        )
        assert ctx_k2 is not None
        self.assertEqual(ctx_k2.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k2.evaluated_neighbors_count, 4)

        # Option B: k=3 at z=50 -> 6 neighbors: [47, 48, 49, 51, 52, 53]
        ctx_k3 = _get_agent_context(
            slice_index=50,
            case_primary_predictions=preds,
            context_window=3,
            case_total_slices=total_slices,
        )
        assert ctx_k3 is not None
        self.assertEqual(ctx_k3.unavailable_neighbors_count, 0)
        self.assertEqual(ctx_k3.evaluated_neighbors_count, 6)

    def test_complete_volume_zero_unavailable_across_all_155_slices(self) -> None:
        """When all 155 slices have primary predictions, unavailable count is 0 for EVERY slice."""
        total_slices = 155
        all_preds = {z: _make_prediction("unetpp", dominant_class=0) for z in range(total_slices)}

        for z in range(total_slices):
            for k in [2, 3]:
                ctx = _get_agent_context(
                    slice_index=z,
                    case_primary_predictions=all_preds,
                    context_window=k,
                    case_total_slices=total_slices,
                )
                self.assertIsNotNone(ctx, f"Context was None for slice {z} with k={k}")
                assert ctx is not None
                self.assertEqual(
                    ctx.unavailable_neighbors_count,
                    0,
                    f"Slice {z} reported {ctx.unavailable_neighbors_count} unavailable neighbors (k={k})",
                )
                expected_evaluated = min(z + k, total_slices - 1) - max(z - k, 0)
                self.assertEqual(
                    ctx.evaluated_neighbors_count,
                    expected_evaluated,
                    f"Slice {z} expected {expected_evaluated} evaluated neighbors, got {ctx.evaluated_neighbors_count}",
                )


class TestGroundTruthIsolationAndPolicySafety(unittest.TestCase):
    """Verifies that ground truth masks are strictly isolated from agent inference and policies."""

    def test_run_slice_signature_has_no_target_mask(self) -> None:
        """InferenceAgent.run_slice must NEVER accept target_mask or ground truth."""
        sig = inspect.signature(InferenceAgent.run_slice)
        forbidden_params = {"target_mask", "ground_truth", "target", "label", "gt", "mask"}
        for p in sig.parameters:
            self.assertNotIn(
                p.lower(),
                forbidden_params,
                f"InferenceAgent.run_slice contains forbidden ground-truth parameter: {p}",
            )

    def test_evaluate_slice_policy_signature_has_no_target_mask(self) -> None:
        """Policy evaluation logic must NEVER accept ground-truth data."""
        sig = inspect.signature(evaluate_slice_policy)
        forbidden_params = {"target_mask", "ground_truth", "target", "label", "gt"}
        for p in sig.parameters:
            self.assertNotIn(
                p.lower(),
                forbidden_params,
                f"evaluate_slice_policy contains forbidden ground-truth parameter: {p}",
            )

    def test_policy_config_defaults(self) -> None:
        """PolicyConfig() default constructor must maintain Option A defaults without side-effects."""
        default_config = PolicyConfig()
        self.assertFalse(default_config.enable_axial_context)
        self.assertEqual(default_config.axial_context_window, 2)
        self.assertEqual(default_config.min_neighbor_foreground_slices, 1)
        self.assertFalse(default_config.fallback_when_context_unavailable)

    def test_agent_option_a_vs_option_b_configurations(self) -> None:
        """Option A (k=2) and Option B (k=3) configs must match their specifications."""
        opt_a = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=2,
            min_neighbor_foreground_slices=1,
            fallback_when_context_unavailable=False,
        )
        opt_b = PolicyConfig(
            enable_empty_prediction_fallback=True,
            fallback_priority=("unet",),
            enable_axial_context=True,
            axial_context_window=3,
            min_neighbor_foreground_slices=1,
            fallback_when_context_unavailable=False,
        )

        self.assertEqual(opt_a.axial_context_window, 2)
        self.assertFalse(opt_a.fallback_when_context_unavailable)
        self.assertEqual(opt_b.axial_context_window, 3)
        self.assertFalse(opt_b.fallback_when_context_unavailable)
        self.assertNotEqual(opt_a.axial_context_window, opt_b.axial_context_window)


class TestFullVolumeMetricsAndEdgeCases(unittest.TestCase):
    """Verifies metric computation, NaN handling on all-zero slices, and composite regions."""

    def test_both_empty_background_slice(self) -> None:
        """When GT and Pred are both background (all zeros), macro Dice is NaN and is_true_negative_slice=True."""
        gt = torch.zeros((32, 32), dtype=torch.int64)
        pred = torch.zeros((32, 32), dtype=torch.int64)

        mean_dice, mean_iou, wt_dice, tc_dice, et_dice = compute_slice_metrics(pred, gt)
        self.assertTrue(np.isnan(mean_dice))
        self.assertTrue(np.isnan(mean_iou))
        self.assertTrue(np.isnan(wt_dice))
        self.assertTrue(np.isnan(tc_dice))
        self.assertTrue(np.isnan(et_dice))

        rec = evaluate_slice_record(
            method="test_method",
            case_id="BraTS_TEST_001",
            slice_index=10,
            pred_mask=pred,
            target_mask=gt,
            latency_ms=12.5,
        )
        self.assertFalse(rec.has_foreground)
        self.assertTrue(rec.is_empty_prediction)
        self.assertFalse(rec.is_confirmed_miss)
        self.assertFalse(rec.is_false_positive_slice)
        self.assertTrue(rec.is_true_negative_slice)
        self.assertEqual(rec.gt_foreground_pixels, 0)
        self.assertEqual(rec.pred_foreground_pixels, 0)

    def test_confirmed_complete_miss_slice(self) -> None:
        """When GT has tumor and Pred is empty, macro Dice is 0.0 and is_confirmed_miss=True."""
        gt = torch.zeros((32, 32), dtype=torch.int64)
        gt[8:16, 8:16] = 1  # 64 pixels of NCR
        pred = torch.zeros((32, 32), dtype=torch.int64)

        mean_dice, mean_iou, wt_dice, tc_dice, et_dice = compute_slice_metrics(pred, gt)
        self.assertEqual(mean_dice, 0.0)
        self.assertEqual(mean_iou, 0.0)
        self.assertEqual(wt_dice, 0.0)
        self.assertEqual(tc_dice, 0.0)
        self.assertTrue(np.isnan(et_dice))  # ET (class 3) has no ground truth or pred -> NaN

        rec = evaluate_slice_record(
            method="test_method",
            case_id="BraTS_TEST_001",
            slice_index=55,
            pred_mask=pred,
            target_mask=gt,
            latency_ms=10.0,
        )
        self.assertTrue(rec.has_foreground)
        self.assertTrue(rec.is_empty_prediction)
        self.assertTrue(rec.is_confirmed_miss)
        self.assertFalse(rec.is_false_positive_slice)
        self.assertFalse(rec.is_true_negative_slice)
        self.assertEqual(rec.gt_foreground_pixels, 64)
        self.assertEqual(rec.pred_foreground_pixels, 0)

    def test_false_positive_healthy_slice(self) -> None:
        """When GT is background and Pred has hallucinated tumor, is_false_positive_slice=True."""
        gt = torch.zeros((32, 32), dtype=torch.int64)
        pred = torch.zeros((32, 32), dtype=torch.int64)
        pred[5:10, 5:10] = 2  # 25 pixels of ED

        mean_dice, mean_iou, wt_dice, tc_dice, et_dice = compute_slice_metrics(pred, gt)
        # Class 2 has 25 false positive pixels and 0 true positives -> Dice is 0.0
        self.assertEqual(mean_dice, 0.0)
        self.assertEqual(wt_dice, 0.0)

        rec = evaluate_slice_record(
            method="test_method",
            case_id="BraTS_TEST_001",
            slice_index=5,
            pred_mask=pred,
            target_mask=gt,
            latency_ms=8.5,
        )
        self.assertFalse(rec.has_foreground)
        self.assertFalse(rec.is_empty_prediction)
        self.assertFalse(rec.is_confirmed_miss)
        self.assertTrue(rec.is_false_positive_slice)
        self.assertFalse(rec.is_true_negative_slice)
        self.assertEqual(rec.gt_foreground_pixels, 0)
        self.assertEqual(rec.pred_foreground_pixels, 25)

    def test_composite_clinical_regions_wt_tc_et(self) -> None:
        """Verifies exact arithmetic for WT {1,2,3}, TC {1,3}, and ET {3} regions."""
        gt = torch.zeros((16, 16), dtype=torch.int64)
        gt[0:4, 0:4] = 1  # 16 pixels of NCR
        gt[4:8, 0:4] = 2  # 16 pixels of ED
        gt[8:12, 0:4] = 3  # 16 pixels of ET
        # Total GT: WT = 48, TC = 32, ET = 16

        pred = torch.zeros((16, 16), dtype=torch.int64)
        pred[0:4, 0:4] = 1  # 16 pixels overlap
        pred[4:8, 0:4] = 2  # 16 pixels overlap
        # Pred misses ET (class 3) entirely

        _, _, wt_dice, tc_dice, et_dice = compute_slice_metrics(pred, gt)
        # WT: TP=32, FP=0, FN=16 -> 2*32 / (64 + 16) = 64/80 = 0.80
        self.assertAlmostEqual(wt_dice, 0.80, places=4)
        # TC: TP=16, FP=0, FN=16 -> 2*16 / (32 + 16) = 32/48 = 0.6667
        self.assertAlmostEqual(tc_dice, 2.0 / 3.0, places=4)
        # ET: TP=0, FP=0, FN=16 -> 0.0
        self.assertAlmostEqual(et_dice, 0.0, places=4)

    def test_global_metric_accumulator(self) -> None:
        """GlobalMetricAccumulator correctly sums voxel-level TP, FP, FN across slices."""
        acc = GlobalMetricAccumulator(num_classes=4)

        # Slice 1: perfect ET match
        gt1 = torch.zeros((10, 10), dtype=torch.int64)
        gt1[0:2, 0:2] = 3  # 4 pixels ET
        pred1 = gt1.clone()
        acc.update(pred1, gt1)

        # Slice 2: complete miss of NCR
        gt2 = torch.zeros((10, 10), dtype=torch.int64)
        gt2[0:2, 0:2] = 1  # 4 pixels NCR
        pred2 = torch.zeros((10, 10), dtype=torch.int64)
        acc.update(pred2, gt2)

        res = acc.compute()
        # Per-class ET (class 3): TP=4, FP=0, FN=0 -> Dice 1.0
        self.assertEqual(res["per_class_dice"][3], 1.0)
        # Per-class NCR (class 1): TP=0, FP=0, FN=4 -> Dice 0.0
        self.assertEqual(res["per_class_dice"][1], 0.0)
        # Per-class ED (class 2): No pixels in either -> NaN
        self.assertTrue(np.isnan(res["per_class_dice"][2]))
        # Foreground micro Dice: (Dice(1) + Dice(3)) / 2 = (0.0 + 1.0) / 2 = 0.5
        self.assertAlmostEqual(res["global_micro_dice_foreground"], 0.5, places=4)


class TestCSVRowAlignmentAndOrdering(unittest.TestCase):
    """Verifies that all evaluated systems log slices in identical ascending axial order (Finding M-1)."""

    def test_csv_row_alignment_uniformity(self) -> None:
        """Every system must produce identical (case_id, slice_index) sequence in slice_evaluations.csv."""
        methods = ["unet", "unetpp", "attention_unet", "agent_baseline", "agent_option_a", "agent_option_b"]
        cases = ["BraTS_001", "BraTS_002"]
        slices_per_case = 5

        all_results: dict[str, list[SliceEvaluationResult]] = {m: [] for m in methods}
        dummy_mask = torch.zeros((8, 8), dtype=torch.int64)

        for case_id in cases:
            for z in range(slices_per_case):
                for m in methods:
                    rec = evaluate_slice_record(
                        method=m,
                        case_id=case_id,
                        slice_index=z,
                        pred_mask=dummy_mask,
                        target_mask=dummy_mask,
                        latency_ms=5.0,
                    )
                    all_results[m].append(rec)

        # Check that for any two methods, the sequence of (case_id, slice_index) keys matches 100%
        reference_keys = [(r.case_id, r.slice_index) for r in all_results["unetpp"]]
        self.assertEqual(len(reference_keys), len(cases) * slices_per_case)

        for m in methods:
            m_keys = [(r.case_id, r.slice_index) for r in all_results[m]]
            self.assertEqual(
                m_keys,
                reference_keys,
                f"Method {m} did not log slices in identical (case_id, slice_index) order",
            )


class TestHistoricalMilestonePreservation(unittest.TestCase):
    """Verifies that historical checkpoints and Milestone 1-6 evaluation artifacts remain intact."""

    def test_preservation_of_milestones_1_to_6(self) -> None:
        """Milestones 1-6 summary files and baseline weights must exist and be valid non-empty files."""
        project_root = Path(__file__).resolve().parents[2]

        # 1. Baseline model checkpoints
        for name, ckpt in DEFAULT_CHECKPOINTS.items():
            ckpt_path = project_root / ckpt
            self.assertTrue(ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}")
            self.assertGreater(ckpt_path.stat().st_size, 10_000_000, f"Checkpoint suspiciously small: {ckpt_path}")

        # 2. Historical Milestone evaluation outputs
        # Milestone 3 artifacts are stored directly in results/agent_evaluation/
        m3_dir = project_root / "results" / "agent_evaluation"
        self.assertTrue((m3_dir / "slice_evaluations.csv").exists(), "Milestone 3 slice_evaluations.csv missing")
        self.assertTrue((m3_dir / "evaluation_summary.json").exists(), "Milestone 3 evaluation_summary.json missing")

        # Milestones 4, 5, 6 are in dedicated subdirectories
        for milestone_dir in ["milestone4", "milestone5", "milestone6_option_b"]:
            m_path = project_root / "results" / "agent_evaluation" / milestone_dir
            self.assertTrue(m_path.exists(), f"Historical directory missing: {m_path}")

            csv_file = m_path / "slice_evaluations.csv"
            self.assertTrue(csv_file.exists(), f"Missing historical CSV: {csv_file}")
            self.assertGreater(csv_file.stat().st_size, 100_000, f"Historical CSV suspiciously small: {csv_file}")

            json_file = m_path / "evaluation_summary.json"
            self.assertTrue(json_file.exists(), f"Missing historical JSON: {json_file}")
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertIn("systems", data, f"JSON {json_file} missing 'systems' key")


class TestEndToEndEvaluatorSynthetic(unittest.TestCase):
    """Verifies end-to-end execution of run_full_volume_evaluation using synthetic cases and mock registry."""

    def test_synthetic_end_to_end_flow(self) -> None:
        """Executes a complete 1-case 4-slice synthetic evaluation and checks all generated artifacts."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)

            # Create mock registry
            mock_registry = MagicMock(spec=ModelRegistry)
            mock_registry.get_model.return_value = MagicMock()

            # Mock slice prediction returning 4 classes logits
            def mock_predict(*args, **kwargs) -> SlicePrediction:
                # Extract image shape from positional or keyword arguments
                img = kwargs.get("image", args[1] if len(args) > 1 else None)
                h = img.shape[1] if img is not None else 16
                w = img.shape[2] if img is not None else 16
                name = kwargs.get("model_name", args[0] if len(args) > 0 else "mock_model")
                logits = _make_synthetic_logits(dominant_class=0, spatial_size=(h, w))
                return extract_slice_prediction(name, logits)

            mock_registry.predict_slice.side_effect = mock_predict

            # Create synthetic case
            mock_case = MagicMock(spec=CaseFiles)
            mock_case.case_id = "BraTS_MOCK_001"
            mock_volume = CachedCaseVolume(
                image=np.zeros((4, 16, 16, 4), dtype=np.float32),
                mask=np.zeros((16, 16, 4), dtype=np.int64),
            )

            with patch("src.agent.evaluate_full_volume._load_and_normalize_case", return_value=mock_volume):
                summary = run_full_volume_evaluation(
                    output_dir=out_dir,
                    device_name="cpu",
                    max_cases=1,
                    max_slices_per_case=4,
                    registry=mock_registry,
                    cases=[mock_case],
                )

            # 1. Summary structure checks
            self.assertIn("configuration", summary)
            self.assertIn("systems", summary)
            self.assertIn("comparative_analysis", summary)

            systems = summary["systems"]
            for m in ["unet", "unetpp", "attention_unet", "agent_baseline", "agent_option_a", "agent_option_b"]:
                self.assertIn(m, systems)
                self.assertEqual(systems[m]["total_slices"], 4)

            # 2. CSV checks
            csv_path = out_dir / "slice_evaluations.csv"
            self.assertTrue(csv_path.exists())
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            # 6 systems * 4 slices = 24 rows
            self.assertEqual(len(rows), 24)

            # 3. JSON checks (Strict RFC 8259 validation: reject NaN / Infinity constants)
            def _reject_constant(c: str) -> None:
                raise ValueError(f"Encountered non-standard JSON constant: {c}")

            json_path = out_dir / "evaluation_summary.json"
            self.assertTrue(json_path.exists())
            with json_path.open("r", encoding="utf-8") as f:
                json_data = json.load(f, parse_constant=_reject_constant)
            self.assertEqual(json_data["configuration"]["validation_case_count"], 1)

            # Slices were all background (GT=0), so macro foreground Dice must be null (None in Python)
            self.assertIsNone(json_data["systems"]["unet"]["slice_macro_mean_dice_foreground"])
            self.assertIsNone(json_data["systems"]["unetpp"]["global_micro_dice_foreground"])
            # Finite metrics must remain intact
            self.assertEqual(json_data["systems"]["unet"]["total_slices"], 4)
            self.assertEqual(json_data["systems"]["unet"]["background_slices"], 4)
            self.assertEqual(json_data["systems"]["unet"]["true_negative_rate_percent"], 100.0)

            # Raw file string check: unquoted NaN or Infinity must not appear
            raw_json_str = json_path.read_text(encoding="utf-8")
            self.assertNotIn(": NaN", raw_json_str)
            self.assertNotIn(": Infinity", raw_json_str)
            self.assertNotIn(": -Infinity", raw_json_str)

            # 4. Report markdown checks
            report_path = out_dir / "report.md"
            self.assertTrue(report_path.exists())
            report_content = report_path.read_text(encoding="utf-8")
            self.assertIn("Milestone 7: Full-Volume", report_content)
            self.assertIn("Agent Option A", report_content)
            self.assertIn("Agent Option B", report_content)


class TestUndefinedMetricJSONSerialization(unittest.TestCase):
    """Unit tests verifying RFC 8259 strict JSON serialization for undefined and finite metrics."""

    def test_scalar_nan_and_inf_converted_to_none(self) -> None:
        """Standard and numpy NaN/Inf values are converted to None."""
        self.assertIsNone(sanitize_for_json(float("nan")))
        self.assertIsNone(sanitize_for_json(float("inf")))
        self.assertIsNone(sanitize_for_json(float("-inf")))
        self.assertIsNone(sanitize_for_json(np.nan))
        self.assertIsNone(sanitize_for_json(np.float32("nan")))
        self.assertIsNone(sanitize_for_json(np.float64("inf")))
        self.assertIsNone(sanitize_for_json(np.float64("-inf")))

    def test_finite_values_and_primitives_preserved(self) -> None:
        """Finite floats, ints, bools, strings, and None are preserved unchanged."""
        self.assertEqual(sanitize_for_json(0.7937), 0.7937)
        self.assertEqual(sanitize_for_json(0.0), 0.0)
        self.assertEqual(sanitize_for_json(-1.5), -1.5)
        self.assertAlmostEqual(sanitize_for_json(np.float32(0.85)), 0.85, places=5)
        self.assertEqual(sanitize_for_json(np.float64(0.85)), 0.85)
        self.assertEqual(sanitize_for_json(42), 42)
        self.assertEqual(sanitize_for_json(np.int64(155)), 155)
        self.assertEqual(sanitize_for_json("unetpp"), "unetpp")
        self.assertIs(sanitize_for_json(True), True)
        self.assertIs(sanitize_for_json(False), False)
        self.assertIsNone(sanitize_for_json(None))

    def test_nested_structures_sanitization(self) -> None:
        """Nested dicts, lists, and tuples have all NaNs converted to None recursively."""
        raw_nested = {
            "method": "agent_option_a",
            "global_dice": float("nan"),
            "finite_score": 0.825,
            "per_class_dice": [1.0, float("nan"), np.nan, 0.75],
            "details": {
                "sub_metric": float("inf"),
                "valid_count": 10,
            },
        }
        sanitized = sanitize_for_json(raw_nested)

        self.assertEqual(sanitized["method"], "agent_option_a")
        self.assertIsNone(sanitized["global_dice"])
        self.assertEqual(sanitized["finite_score"], 0.825)
        self.assertEqual(sanitized["per_class_dice"], [1.0, None, None, 0.75])
        self.assertIsNone(sanitized["details"]["sub_metric"])
        self.assertEqual(sanitized["details"]["valid_count"], 10)

    def test_strict_json_parser_rejects_raw_nan_and_accepts_sanitized(self) -> None:
        """Strict JSON parsing (rejecting NaN/Infinity) fails on raw float('nan') but succeeds on sanitized data."""
        def _reject_constant(val: str) -> None:
            raise ValueError(f"Non-standard constant rejected: {val}")

        data_with_nan = {"metric": float("nan"), "valid": 0.7937}

        # 1. Unsanitized dumps produces non-standard NaN literal
        raw_json_str = json.dumps(data_with_nan)
        self.assertIn("NaN", raw_json_str)
        with self.assertRaises(ValueError):
            json.loads(raw_json_str, parse_constant=_reject_constant)

        # 2. Sanitized dumps produces standard RFC 8259 null
        sanitized = sanitize_for_json(data_with_nan)
        sanitized_json_str = json.dumps(sanitized)
        self.assertNotIn("NaN", sanitized_json_str)
        self.assertIn('"metric": null', sanitized_json_str)

        # 3. Strict parsing succeeds and restores None
        parsed = json.loads(sanitized_json_str, parse_constant=_reject_constant)
        self.assertIsNone(parsed["metric"])
        self.assertEqual(parsed["valid"], 0.7937)


if __name__ == "__main__":
    unittest.main()
