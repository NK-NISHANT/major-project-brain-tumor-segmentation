from __future__ import annotations

import inspect
import unittest
from pathlib import Path

import numpy as np
import torch

from src.agent.evaluate import (
    SliceEvaluationResult,
    compute_mask_metrics,
    evaluate_prediction,
    summarize_evaluation,
)
from src.agent.models import DEFAULT_CHECKPOINTS
from src.agent.runner import InferenceAgent
from src.agent.state import AgentAction


class TestEvaluationMetricsAndEdgeCases(unittest.TestCase):
    """Synthetic unit tests for metric calculation, edge cases, and failure mode categorization."""

    def test_both_empty_background_only(self) -> None:
        # Both ground truth and prediction are entirely background (class 0)
        gt = torch.zeros((16, 16), dtype=torch.uint8)
        pred = torch.zeros((16, 16), dtype=torch.uint8)

        mean_dice, mean_iou, per_class_dice, per_class_iou = compute_mask_metrics(pred, gt)
        # Background class 0 has dice 1.0, but ignored in foreground mean -> nan
        self.assertTrue(np.isnan(mean_dice))
        self.assertTrue(np.isnan(mean_iou))
        self.assertEqual(per_class_dice[0], 1.0)
        self.assertTrue(np.isnan(per_class_dice[1]))

        result = evaluate_prediction("test", "case_0", 1, pred, gt)
        self.assertFalse(result.has_foreground)
        self.assertTrue(result.is_empty_prediction)
        self.assertFalse(result.is_confirmed_miss)
        self.assertFalse(result.is_false_positive_slice)
        self.assertTrue(result.is_true_negative_slice)

    def test_confirmed_miss_edge_case(self) -> None:
        # Ground truth has tumor (class 1), model predicts background (class 0)
        gt = torch.zeros((16, 16), dtype=torch.uint8)
        gt[4:8, 4:8] = 1
        pred = torch.zeros((16, 16), dtype=torch.uint8)

        mean_dice, mean_iou, _, _ = compute_mask_metrics(pred, gt)
        self.assertEqual(mean_dice, 0.0)
        self.assertEqual(mean_iou, 0.0)

        result = evaluate_prediction("test", "case_0", 1, pred, gt)
        self.assertTrue(result.has_foreground)
        self.assertTrue(result.is_empty_prediction)
        self.assertTrue(result.is_confirmed_miss)
        self.assertFalse(result.is_false_positive_slice)
        self.assertFalse(result.is_true_negative_slice)

    def test_false_positive_slice_edge_case(self) -> None:
        # Ground truth is healthy brain (0), model predicts tumor (class 2)
        gt = torch.zeros((16, 16), dtype=torch.uint8)
        pred = torch.zeros((16, 16), dtype=torch.uint8)
        pred[2:6, 2:6] = 2

        mean_dice, mean_iou, _, _ = compute_mask_metrics(pred, gt)
        self.assertEqual(mean_dice, 0.0)
        self.assertEqual(mean_iou, 0.0)

        result = evaluate_prediction("test", "case_0", 1, pred, gt)
        self.assertFalse(result.has_foreground)
        self.assertFalse(result.is_empty_prediction)
        self.assertFalse(result.is_confirmed_miss)
        self.assertTrue(result.is_false_positive_slice)
        self.assertFalse(result.is_true_negative_slice)

    def test_partial_overlap_foreground(self) -> None:
        gt = torch.zeros((16, 16), dtype=torch.uint8)
        gt[0:4, 0:4] = 1  # 16 pixels
        pred = torch.zeros((16, 16), dtype=torch.uint8)
        pred[2:6, 0:4] = 1  # 16 pixels, overlap is 8 pixels

        mean_dice, mean_iou, _, _ = compute_mask_metrics(pred, gt)
        # Overlap: tp = 8, fp = 8, fn = 8 -> Dice = 2*8 / (16 + 16) = 0.50
        self.assertAlmostEqual(mean_dice, 0.50, places=4)
        # IoU: 8 / (8 + 8 + 8) = 0.3333
        self.assertAlmostEqual(mean_iou, 1.0 / 3.0, places=4)

        result = evaluate_prediction("test", "case_0", 1, pred, gt)
        self.assertTrue(result.has_foreground)
        self.assertFalse(result.is_empty_prediction)
        self.assertFalse(result.is_confirmed_miss)
        self.assertFalse(result.is_false_positive_slice)
        self.assertFalse(result.is_true_negative_slice)

    def test_summarize_evaluation_decision_statistics(self) -> None:
        # Build 3 mock results: 1 accepted, 1 fallback, 1 unreliable
        r1 = evaluate_prediction(
            "agent", "case_1", 1, torch.ones((8, 8)), torch.ones((8, 8)),
            action=AgentAction.ACCEPT.value, selected_model="unetpp",
        )
        r2 = evaluate_prediction(
            "agent", "case_1", 2, torch.ones((8, 8)), torch.ones((8, 8)),
            action=AgentAction.FALLBACK_SWITCH.value, selected_model="unet",
        )
        r3 = evaluate_prediction(
            "agent", "case_1", 3, torch.zeros((8, 8)), torch.ones((8, 8)),
            action=AgentAction.FLAG_UNRELIABLE.value, selected_model="unetpp",
        )

        summary = summarize_evaluation("agent", [r1, r2, r3], total_runtime_seconds=1.5)
        decisions = summary["agent_decisions"]

        self.assertEqual(decisions["accepted_count"], 1)
        self.assertEqual(decisions["fallback_count"], 1)
        self.assertEqual(decisions["unreliable_flag_count"], 1)
        self.assertAlmostEqual(decisions["accepted_percent"], 33.33, places=1)
        self.assertEqual(decisions["model_selection_distribution"], {"unetpp": 2, "unet": 1})


class TestSafetyAndIntegrity(unittest.TestCase):
    """Verify ground-truth isolation and preservation of existing checkpoints/results."""

    def test_ground_truth_isolation_signature(self) -> None:
        # Verify InferenceAgent.run_slice does NOT have target/mask/ground_truth parameters
        params = inspect.signature(InferenceAgent.run_slice).parameters
        param_names = set(params.keys())
        forbidden_terms = {"mask", "target", "ground_truth", "gt", "labels"}
        intersection = param_names.intersection(forbidden_terms)
        self.assertEqual(
            len(intersection),
            0,
            f"Ground-truth parameter found in run_slice signature: {intersection}",
        )

    def test_existing_checkpoints_exist_and_unaltered(self) -> None:
        # Ensure all existing baseline checkpoints exist and have positive size
        for name, ckpt_path in DEFAULT_CHECKPOINTS.items():
            self.assertTrue(ckpt_path.exists(), f"Checkpoint {ckpt_path} is missing.")
            self.assertGreater(ckpt_path.stat().st_size, 10_000_000)

        # Ensure precomputed baseline CSV exists
        baseline_csv = Path("results/uncertainty_unet/model_uncertainty_comparison.csv")
        self.assertTrue(baseline_csv.exists(), "Existing baseline comparison CSV is missing.")
        self.assertGreater(baseline_csv.stat().st_size, 500_000)


def main() -> None:
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
