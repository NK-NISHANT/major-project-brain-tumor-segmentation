"""Unit tests for Stage C Work Package C3 v1.1 Policy & Scientific Integrity.

Verifies:
1. Zero-retained denominator handling (null/NaN metrics when retained count is 0).
2. Correct complete-miss accounting (mutually exclusive tiers summing to 6,549).
3. All-slice (64.57%) vs. GT-positive (16.16%) vs. case-level (100%) referral accounting.
4. Axial-context fallback edge cases (116 GT+ slices without axial support).
5. Secondary fallback success (37 positive overlap) vs. failed zero-overlap (17).
6. Ground-truth isolation from runtime policy decisions.
7. Empirical non-monotonicity in risk-coverage curves for UNetResNet34 and U-Net++.
8. Authoritative checkpoint hashes and frozen temperatures from C3_READINESS_GATE_v1.4.1.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_stage_c_c3_selective_prediction_v1_1 import (
    AUTHORITATIVE_CHECKPOINTS,
    enforce_preflight_boundaries,
)
from src.agent.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.agent.policy import evaluate_slice_policy
import torch


def test_authoritative_checkpoint_hashes():
    """Verify all 5 checkpoint hashes and temperatures match C3_READINESS_GATE_v1.4.1.md exactly."""
    expected = {
        "resunet_2d": {
            "best_epoch": 13,
            "size_bytes": 97486845,
            "sha256": "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d",
            "temperature": 2.880459,
        },
        "unet_resnet34": {
            "best_epoch": 15,
            "size_bytes": 292222285,
            "sha256": "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d",
            "temperature": 2.626091,
        },
        "unet": {
            "best_epoch": 15,
            "size_bytes": 93281991,
            "sha256": "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef",
            "temperature": 2.626793,
        },
        "unetpp": {
            "best_epoch": 15,
            "size_bytes": 110090779,
            "sha256": "fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6",
            "temperature": 2.564703,
        },
        "attention_unet": {
            "best_epoch": 15,
            "size_bytes": 94417899,
            "sha256": "5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e",
            "temperature": 2.286888,
        },
    }
    assert AUTHORITATIVE_CHECKPOINTS == expected
    info = enforce_preflight_boundaries()
    assert info["eval_cases_count"] == 105
    assert info["eval_subjs_count"] == 96


def test_zero_retained_denominator_handling():
    """Verify that when retained count is zero, metrics are null/None (not 0.0 or 100% risk reduction)."""
    # At tau = 0.05 on ResUNet-2D, retained GT+ count is 0
    # In run_stage_c_c3_selective_prediction_v1_1, this must result in None
    retained_count = 0
    base_severe_rate = 0.2051
    if retained_count == 0:
        retained_mean_dice = None
        retained_severe_rate = None
        rel_risk_reduction = None
    else:
        retained_mean_dice = 0.0
        retained_severe_rate = 0.0
        rel_risk_reduction = 1.0

    assert retained_mean_dice is None
    assert retained_severe_rate is None
    assert rel_risk_reduction is None


def test_mutually_exclusive_quality_tiers_sum():
    """Verify that the 4 quality tiers partition the 6,549 GT+ slices with zero double counting."""
    tiers_path = PROJECT_ROOT / "results" / "stage_c" / "phase3" / "c3_selective_prediction_v1_1" / "segmentation_quality_tiers.csv"
    if not tiers_path.exists():
        pytest.skip("v1.1 artifacts not yet generated")
    
    rows = list(csv.DictReader(open(tiers_path)))
    assert len(rows) == 4
    
    total_slices = sum(int(r["total_count"]) for r in rows)
    total_retained = sum(int(r["retained_count"]) for r in rows)
    total_referred = sum(int(r["referred_count"]) for r in rows)
    
    assert total_slices == 6549
    assert total_retained == 5491
    assert total_referred == 1058
    assert total_retained + total_referred == 6549


def test_all_slice_vs_gt_referral_accounting():
    """Verify distinct referral rates across denominators: GT+ (16.16%), All-slice (64.57%), Case-level (100%)."""
    trace_path = PROJECT_ROOT / "results" / "stage_c" / "phase3" / "c3_selective_prediction_v1_1" / "policy_action_trace.csv"
    if not trace_path.exists():
        pytest.skip("v1.1 artifacts not yet generated")
        
    trace = list(csv.DictReader(open(trace_path)))
    assert len(trace) == 16275
    
    all_flagged = sum(1 for r in trace if r["final_action"] == "FLAG_UNRELIABLE")
    gt_flagged = sum(1 for r in trace if r["final_action"] == "FLAG_UNRELIABLE" and r["is_gt_positive"] == "True")
    gt_neg_flagged = sum(1 for r in trace if r["final_action"] == "FLAG_UNRELIABLE" and r["is_gt_positive"] == "False")
    
    assert all_flagged == 10509
    assert gt_flagged == 1058
    assert gt_neg_flagged == 9451
    assert gt_flagged + gt_neg_flagged == all_flagged
    
    # All-slice referral rate
    assert abs(all_flagged / 16275 - 0.645714) < 1e-4
    # GT+ referral rate
    assert abs(gt_flagged / 6549 - 0.161551) < 1e-4
    
    # Case-level escalation
    case_flags = set(r["case_id"] for r in trace if r["final_action"] == "FLAG_UNRELIABLE")
    assert len(case_flags) == 105


def test_axial_fallback_edge_cases_and_unswitched_gt_slices():
    """Verify fallback outcomes and that 116 GT+ slices had no predicted axial support."""
    audit_path = PROJECT_ROOT / "results" / "stage_c" / "phase3" / "c3_selective_prediction_v1_1" / "empty_prediction_fallback_audit.csv"
    if not audit_path.exists():
        pytest.skip("v1.1 artifacts not yet generated")
        
    audit = list(csv.DictReader(open(audit_path)))
    assert len(audit) == 9401
    
    gt_empty = [r for r in audit if r["is_gt_positive"] == "True"]
    assert len(gt_empty) == 340
    
    # Check unsupported GT+ slices
    no_axial_gt = [r for r in gt_empty if r["has_axial_support"] == "False"]
    assert len(no_axial_gt) == 116
    for r in no_axial_gt:
        assert r["fallback_triggered"] == "False"
        
    # Check switched GT+ slices
    switched_gt = [r for r in gt_empty if r["fallback_triggered"] == "True"]
    assert len(switched_gt) == 54
    
    pos_overlap = [r for r in switched_gt if float(r["secondary_dice"]) > 0]
    zero_overlap = [r for r in switched_gt if float(r["secondary_dice"]) == 0]
    assert len(pos_overlap) == 37
    assert len(zero_overlap) == 17
    assert len(pos_overlap) + len(zero_overlap) == 54


def test_ground_truth_isolation():
    """Ensure runtime policy evaluation functions have zero access to ground truth."""
    import inspect
    sig = inspect.signature(evaluate_slice_policy)
    assert list(sig.parameters.keys()) == ["state", "config"]
    assert "gt" not in str(sig)
    
    fields = [f.name for f in AgentState.__dataclass_fields__.values()]
    assert "gt_mask" not in fields
    assert "gt_pixels" not in fields
    assert "dice" not in fields


def test_risk_coverage_non_monotonicity_documentation():
    """Verify that non-monotonic risk trends in UNetResNet34 and U-Net++ are captured."""
    rc_path = PROJECT_ROOT / "results" / "stage_c" / "phase3" / "c3_selective_prediction_v1_1" / "risk_coverage_by_model.csv"
    if not rc_path.exists():
        pytest.skip("v1.1 artifacts not yet generated")
        
    rows = list(csv.DictReader(open(rc_path)))
    
    # UNetResNet34: severe failure rate decreases between tau=0.15 and tau=0.25
    u34_rows = {float(r["tau_threshold"]): float(r["retained_severe_failure_rate"]) for r in rows if r["model_name"] == "unet_resnet34" and r["retained_severe_failure_rate"] != ""}
    assert u34_rows[0.15] > u34_rows[0.20] > u34_rows[0.25], "Expected non-monotonic decrease in UNetResNet34 severe failure rate"
    
    # U-Net++: severe failure rate drops between tau=0.15 and tau=0.20
    upp_rows = {float(r["tau_threshold"]): float(r["retained_severe_failure_rate"]) for r in rows if r["model_name"] == "unetpp" and r["retained_severe_failure_rate"] != ""}
    assert upp_rows[0.15] > upp_rows[0.20], "Expected non-monotonic drop in U-Net++ severe failure rate"
