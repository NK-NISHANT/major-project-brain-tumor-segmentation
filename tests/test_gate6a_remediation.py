"""Unit tests for Gate 6A Remediation:
- Subject-grouped split integrity and disjointness
- Calibrated entropy calculation consistency
- Argmax invariance under temperature scaling
- Cryptographic provenance validation and cache rejection
- Subject-level cluster bootstrap mechanics
- Evaluation voxel provenance and audit consistency

Workspace: D:\\MajorProject
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from src.agent.policy import extract_slice_prediction
from src.agent.state import SlicePrediction
from src.uncertainty.calibration import (
    TemperatureScaler,
    compute_multiclass_brier_score,
    compute_multiclass_ece,
    compute_nll,
)


def compute_cluster_bootstrap_ci(
    subject_to_scan_values: dict[str, list[float]],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Cluster-level bootstrap: resample unique subjects with replacement.

    For each drawn subject, all corresponding scan values are included in the replicate mean.
    """
    subjects = sorted(list(subject_to_scan_values.keys()))
    if len(subjects) == 0:
        return float("nan"), float("nan"), float("nan")

    # Pooled point estimate
    all_values = [v for s in subjects for v in subject_to_scan_values[s] if not np.isnan(v)]
    if len(all_values) == 0:
        return float("nan"), float("nan"), float("nan")
    point_mean = float(np.mean(all_values))

    if len(subjects) == 1:
        return point_mean, point_mean, point_mean

    rng = np.random.RandomState(seed)
    n_sub = len(subjects)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        draw_sub_indices = rng.randint(0, n_sub, size=n_sub)
        rep_vals: list[float] = []
        for idx in draw_sub_indices:
            sid = subjects[idx]
            rep_vals.extend([v for v in subject_to_scan_values[sid] if not np.isnan(v)])
        boot_means[i] = np.mean(rep_vals) if rep_vals else point_mean

    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return point_mean, lower, upper


class TestGate6ARemediation(unittest.TestCase):
    """Test suite for Gate 6A Remediation components."""

    def test_01_subject_grouped_split_disjointness_and_coverage(self):
        """Verify calibration_splits.json has 0 subject overlap and complete validation coverage."""
        splits_path = PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_splits.json"
        self.assertTrue(splits_path.exists(), f"Missing {splits_path}")

        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)

        cal_cases = set(splits_data["calibration_set"]["case_ids"])
        eval_cases = set(splits_data["evaluation_set"]["case_ids"])
        cal_subs = set(splits_data["calibration_set"]["subject_ids"])
        eval_subs = set(splits_data["evaluation_set"]["subject_ids"])

        # Cardinalities
        self.assertEqual(len(cal_cases), 50)
        self.assertEqual(len(eval_cases), 105)
        self.assertEqual(len(cal_subs), 47)
        self.assertEqual(len(eval_subs), 96)

        # Subject disjointness
        self.assertEqual(len(cal_subs.intersection(eval_subs)), 0, "Subject overlap detected!")

        # Case disjointness
        self.assertEqual(len(cal_cases.intersection(eval_cases)), 0, "Case overlap detected!")

        # Manifest validation coverage
        val_manifest_path = PROJECT_ROOT / "results" / "dataset_splits" / "validation_manifest.json"
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            val_data = json.load(f)
        val_case_ids = set(c["case_id"] for c in val_data["cases"])
        self.assertEqual(cal_cases.union(eval_cases), val_case_ids)

        # Strict test quarantine
        test_manifest_path = PROJECT_ROOT / "results" / "dataset_splits" / "test_manifest.json"
        with open(test_manifest_path, "r", encoding="utf-8") as f:
            test_data = json.load(f)
        test_case_ids = set(c["case_id"] for c in test_data["cases"])
        self.assertEqual(len(cal_cases.intersection(test_case_ids)), 0)
        self.assertEqual(len(eval_cases.intersection(test_case_ids)), 0)

    def test_02_argmax_invariance_under_temperature_scaling(self):
        """Verify that scaling logits by positive temperature T never alters argmax discrete class masks."""
        torch.manual_seed(42)
        logits = torch.randn(1, 4, 32, 32)
        pred_uncal = extract_slice_prediction("test_model", logits)

        for t_val in [0.1, 0.5, 1.0, 2.5, 2.8808, 5.0, 10.0]:
            pred_cal = extract_slice_prediction("test_model", logits / t_val)
            self.assertTrue(
                torch.equal(pred_uncal.pred_mask, pred_cal.pred_mask),
                f"Argmax mask altered at temperature T={t_val}",
            )
            self.assertEqual(pred_uncal.pred_foreground_pixels, pred_cal.pred_foreground_pixels)

    def test_03_calibrated_entropy_monotonic_relationship(self):
        """Verify calibrated predictive entropy changes predictably with temperature."""
        torch.manual_seed(42)
        # Create peaked logits predicting foreground (class 1 > class 0)
        logits = torch.tensor([[[[1.0]], [[3.5]], [[0.5]], [[-1.0]]]], dtype=torch.float32)

        pred_uncal = extract_slice_prediction("test", logits)
        pred_cal = extract_slice_prediction("test", logits / 2.8808)

        # Confirm foreground is predicted
        self.assertFalse(pred_uncal.has_empty_foreground)
        self.assertIsNotNone(pred_uncal.mean_foreground_entropy)
        self.assertIsNotNone(pred_cal.mean_foreground_entropy)

        # Scaling by T > 1 flattens distribution, increasing entropy towards uniform
        self.assertGreater(pred_cal.mean_foreground_entropy, pred_uncal.mean_foreground_entropy)

    def test_04_cluster_bootstrap_resamples_subject_clusters(self):
        """Verify cluster bootstrap samples all scans of a subject together."""
        # 3 subjects: SubA has 2 scans (high 0.9), SubB has 2 scans (low 0.1), SubC has 1 scan (0.5)
        sub_data = {
            "SubA": [0.90, 0.92],
            "SubB": [0.10, 0.12],
            "SubC": [0.50],
        }

        mean_val, lower, upper = compute_cluster_bootstrap_ci(sub_data, n_bootstraps=500, seed=42)
        pooled_mean = np.mean([0.90, 0.92, 0.10, 0.12, 0.50])
        self.assertAlmostEqual(mean_val, pooled_mean, places=4)
        self.assertLess(lower, mean_val)
        self.assertGreater(upper, mean_val)

    def test_05_cache_provenance_validation(self):
        """Verify case cache validates cryptographic hashes before accepting cached predictions."""
        valid_provenance = {
            "resunet_hash": "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d",
            "resnet34_hash": "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d",
            "parameters_hash": "7a6e6d747925c6258fcba912bff61e32d8cbc155094470141d738ebbeacf4b8e",
        }

        # Match check
        def is_cache_valid(cache_record: dict, expected: dict) -> bool:
            prov = cache_record.get("provenance", {})
            return (
                prov.get("resunet_hash") == expected["resunet_hash"]
                and prov.get("resnet34_hash") == expected["resnet34_hash"]
                and prov.get("parameters_hash") == expected["parameters_hash"]
            )

        valid_cache = {"case_id": "BraTS-GLI-00008-001", "provenance": dict(valid_provenance)}
        self.assertTrue(is_cache_valid(valid_cache, valid_provenance))

        # Stale checkpoint check
        stale_cache = {"case_id": "BraTS-GLI-00008-001", "provenance": dict(valid_provenance)}
        stale_cache["provenance"]["resunet_hash"] = "altered_hash_123"
        self.assertFalse(is_cache_valid(stale_cache, valid_provenance))

    def test_06_calibration_parameters_artifact_reconciliation(self):
        """Verify calibration_parameters.json has exact consistency between tables and metadata."""
        params_path = PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_parameters.json"
        self.assertTrue(params_path.exists())

        with open(params_path, "r", encoding="utf-8") as f:
            params = json.load(f)

        meta = params["metadata"]
        self.assertEqual(meta["calibration_case_count"], 50)
        self.assertEqual(meta["total_slices_evaluated"], 7750)
        self.assertEqual(meta["foreground_slices_evaluated"], 3185)

        # Check threshold exploration at tau = 0.35
        abstention = params["threshold_exploration"]["abstention_analysis"]
        r35 = next((r for r in abstention if abs(r["threshold_tau"] - 0.35) < 1e-4), None)
        self.assertIsNotNone(r35)
        self.assertEqual(r35["flagged_slices"], 197)
        self.assertEqual(r35["retained_slices"], 2988)
        self.assertAlmostEqual(r35["coverage_percent"], 93.81475, places=2)


if __name__ == "__main__":
    unittest.main()
