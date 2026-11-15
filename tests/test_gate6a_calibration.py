"""Focused unit tests for Gate 6A uncertainty calibration and reliability policy."""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

from src.agent.models import ModelRegistry
from src.agent.policy import extract_slice_prediction, evaluate_slice_policy
from src.agent.runner import InferenceAgent
from src.agent.state import AgentAction, AgentDecision, AgentState, PolicyConfig, SlicePrediction
from src.uncertainty.calibration import (
    TemperatureScaler,
    compute_multiclass_brier_score,
    compute_multiclass_ece,
    compute_nll,
    compute_risk_coverage_curve,
)


class TestTemperatureScaler(unittest.TestCase):
    def test_initialization_and_forward(self) -> None:
        scaler = TemperatureScaler(initial_temperature=1.5)
        self.assertAlmostEqual(scaler.temperature, 1.5, places=4)
        logits = torch.tensor([[[[2.0, -1.0], [0.5, 3.0]]]])  # [1, 1, 2, 2]
        scaled = scaler(logits)
        expected = logits / 1.5
        self.assertTrue(torch.allclose(scaled, expected, atol=1e-5))

    def test_scale_logits_function(self) -> None:
        scaler = TemperatureScaler(initial_temperature=2.0)
        logits = torch.randn(2, 4, 16, 16)
        scaled = scaler.scale_logits(logits)
        self.assertTrue(torch.allclose(scaled, logits / 2.0, atol=1e-5))

    def test_fit_overconfident_logits(self) -> None:
        """When a model is confident but makes mistakes (label noise), fitting increases T > 1."""
        torch.manual_seed(42)
        n_samples = 400
        n_classes = 4

        # Model is very confident on class 0 for all samples
        sharp_logits = torch.zeros(n_samples, n_classes)
        sharp_logits[:, 0] = 5.0

        # But true labels have 35% error rate (not all class 0)
        targets = torch.zeros(n_samples, dtype=torch.long)
        noise_idx = torch.randperm(n_samples)[: int(0.35 * n_samples)]
        targets[noise_idx] = torch.randint(1, n_classes, (len(noise_idx),))

        scaler = TemperatureScaler(initial_temperature=1.0)
        fit_results = scaler.fit(sharp_logits, targets, max_iter=60)

        # To mitigate large penalties on 35% errors, temperature must soften (T > 1.0)
        self.assertGreater(scaler.temperature, 1.0)
        self.assertTrue(0.05 <= scaler.temperature <= 10.0)
        self.assertIn("initial_nll", fit_results)
        self.assertIn("final_nll", fit_results)
        self.assertLess(fit_results["final_nll"], fit_results["initial_nll"])

    def test_numerical_stability_extreme_logits(self) -> None:
        """Scaler handles extreme logits without producing NaN or Inf."""
        extreme_logits = torch.tensor([[1000.0, -1000.0, 500.0, -500.0]])
        scaler = TemperatureScaler()
        scaled = scaler(extreme_logits)
        self.assertFalse(torch.isnan(scaled).any())
        self.assertFalse(torch.isinf(scaled).any())


class TestCalibrationMetrics(unittest.TestCase):
    def test_compute_multiclass_ece_uniform_and_peaked(self) -> None:
        # Uniform distribution with balanced targets
        logits_uniform = torch.zeros(100, 4)
        probs_uniform = F.softmax(logits_uniform, dim=-1)
        targets = torch.randint(0, 4, (100,))
        ece_dict = compute_multiclass_ece(probs_uniform, targets, n_bins=15)
        self.assertIsInstance(ece_dict, dict)
        self.assertIn("ece", ece_dict)
        self.assertIn("mce", ece_dict)
        self.assertGreaterEqual(ece_dict["ece"], 0.0)
        self.assertLessEqual(ece_dict["ece"], 1.0)
        self.assertEqual(len(ece_dict["bin_counts"]), 15)

    def test_empty_bins_handled_without_nan(self) -> None:
        # Extreme logits where all confidences fall in the top bin
        logits = torch.full((50, 4), -50.0)
        logits[:, 0] = 50.0
        probs = F.softmax(logits, dim=-1)
        targets = torch.zeros(50, dtype=torch.long)
        ece_dict = compute_multiclass_ece(probs, targets, n_bins=15)
        self.assertFalse(torch.isnan(torch.tensor(ece_dict["ece"])))
        self.assertGreaterEqual(ece_dict["ece"], 0.0)
        # Empty bins should have count 0, accuracy 0.0, and confidence at bin midpoint
        for i, count in enumerate(ece_dict["bin_counts"]):
            if count == 0:
                self.assertEqual(ece_dict["bin_accuracies"][i], 0.0)
                expected_midpoint = (ece_dict["bin_edges"][i] + ece_dict["bin_edges"][i + 1]) / 2.0
                self.assertAlmostEqual(ece_dict["bin_confidences"][i], expected_midpoint, places=5)

    def test_compute_multiclass_brier_score(self) -> None:
        probs = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]])
        targets = torch.tensor([0, 0])
        # Sample 0 error: (1-1)^2 + 0^2 + 0^2 + 0^2 = 0
        # Sample 1 error: (0.5-1)^2 + 0.5^2 + 0^2 + 0^2 = 0.25 + 0.25 = 0.5
        # Mean brier: (0 + 0.5) / 2 = 0.25
        brier = compute_multiclass_brier_score(probs, targets)
        self.assertAlmostEqual(brier, 0.25, places=5)

    def test_compute_nll(self) -> None:
        probs = torch.tensor([[0.8, 0.2], [0.9, 0.1]])
        targets = torch.tensor([0, 0])
        nll = compute_nll(probs, targets)
        expected = - (torch.log(torch.tensor(0.8)) + torch.log(torch.tensor(0.9))).item() / 2.0
        self.assertAlmostEqual(nll, expected, places=5)

    def test_compute_risk_coverage_curve(self) -> None:
        uncertainties = [0.1, 0.2, 0.8, 0.9]
        dice_scores = [0.95, 0.90, 0.30, 0.20]
        curve = compute_risk_coverage_curve(uncertainties, dice_scores, n_thresholds=10)
        self.assertIn("coverage", curve)
        self.assertIn("risk", curve)
        self.assertIn("mean_dice", curve)
        self.assertIn("thresholds", curve)
        self.assertGreater(len(curve["coverage"]), 0)


class TestPolicyConfigAndState(unittest.TestCase):
    def test_policy_config_temperature_validation(self) -> None:
        # Valid configs
        cfg = PolicyConfig(primary_temperature=1.2, candidate_temperatures={"unet_resnet34": 0.9})
        self.assertEqual(cfg.primary_temperature, 1.2)
        self.assertEqual(cfg.candidate_temperatures["unet_resnet34"], 0.9)

        # Invalid primary temperature
        with self.assertRaises(ValueError):
            PolicyConfig(primary_temperature=0.0)
        with self.assertRaises(ValueError):
            PolicyConfig(primary_temperature=-1.0)

        # Invalid candidate temperature
        with self.assertRaises(ValueError):
            PolicyConfig(candidate_temperatures={"unet_resnet34": 0.0})

        # Invalid uncertainty threshold
        with self.assertRaises(ValueError):
            PolicyConfig(uncertainty_fallback_threshold=-0.1)

    def test_agent_decision_fields(self) -> None:
        logits = torch.randn(1, 4, 32, 32)
        pred = extract_slice_prediction("resunet_2d", logits)
        state = AgentState(
            case_id="case_001",
            slice_index=70,
            primary_prediction=pred,
            candidate_predictions={},
        )
        cfg = PolicyConfig(primary_temperature=1.35)
        decision = evaluate_slice_policy(state, cfg)

        self.assertIsInstance(decision, AgentDecision)
        self.assertEqual(decision.calibrated_temperature, 1.35)
        self.assertIsInstance(decision.is_human_review_recommended, bool)


class TestQuarantineSecurity(unittest.TestCase):
    def test_test_manifest_quarantined_from_development_and_validation(self) -> None:
        """Confirm that test manifest IDs are strictly distinct from train and val manifests."""
        root = Path(__file__).resolve().parent.parent
        splits_dir = root / "results" / "dataset_splits"

        test_path = splits_dir / "test_manifest.json"
        val_path = splits_dir / "validation_manifest.json"
        train_path = splits_dir / "development_train_manifest.json"

        self.assertTrue(test_path.is_file(), f"Missing {test_path}")
        self.assertTrue(val_path.is_file(), f"Missing {val_path}")
        self.assertTrue(train_path.is_file(), f"Missing {train_path}")

        def get_ids(path: Path) -> set[str]:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cases = data["cases"] if isinstance(data, dict) and "cases" in data else data
            return {c["case_id"] if isinstance(c, dict) else str(c) for c in cases}

        test_ids = get_ids(test_path)
        val_ids = get_ids(val_path)
        train_ids = get_ids(train_path)

        self.assertEqual(len(test_ids), 155)
        self.assertEqual(len(val_ids), 155)
        self.assertEqual(len(train_ids), 721)

        # Zero leakage
        self.assertEqual(len(test_ids.intersection(val_ids)), 0)
        self.assertEqual(len(test_ids.intersection(train_ids)), 0)
        self.assertEqual(len(val_ids.intersection(train_ids)), 0)


if __name__ == "__main__":
    unittest.main()
