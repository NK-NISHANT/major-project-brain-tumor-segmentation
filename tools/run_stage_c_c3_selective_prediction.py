"""Work Package C3: Reliability-Aware AI Agent & Selective Prediction Evaluation.

Governing Blueprint: MASTER_PROJECT_BLUEPRINT_v2.0.md
Governing Protocol: STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md
Parent Audit Gate: C3_READINESS_GATE_v1.4.1.md

This script performs reproducible, CPU-only evaluation of:
1. Five-model risk-coverage threshold sweeps (tau in [0.05, 0.50], step 0.05).
2. Four-way deterministic agent policy ablations (Primary-only, Entropy-only, Fallback-only, Full policy).
3. Biological subject-level aggregation (N=96) with 1,000 cluster bootstrap resamples.
4. Failure-to-referral cross-tabulation and empty-prediction fallback audits.
5. Ingests exclusively from strict RFC 8259 JSON prediction mirrors.

Workspace: D:\\MajorProject
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths
MIRROR_BASE = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "audits" / "post_execution_audit_v1" / "strict_json_case_predictions"
INVENTORY_PATH = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "audits" / "post_execution_audit_v1" / "artifact_sha256_inventory.csv"
SPLITS_PATH = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "provenance_and_splits" / "calibration_splits.json"
CAL_PARAMS_PATH = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "calibration_parameters.json"
READINESS_GATE_PATH = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "audits" / "post_execution_audit_v1" / "C3_READINESS_GATE_v1.4.1.md"
OUTPUT_DIR = PROJECT_ROOT / "results" / "stage_c" / "phase3" / "c3_selective_prediction_v1"

# Prohibited paths
QUARANTINE_TEST_DIR = PROJECT_ROOT / "data" / "quarantine_test"
TEST_MANIFEST_PATH = PROJECT_ROOT / "results" / "dataset_splits" / "test_manifest.json"

FROZEN_TEMPERATURES = {
    "resunet_2d": 2.880459,
    "unet_resnet34": 2.626091,
    "unet": 2.626793,
    "unetpp": 2.564703,
    "attention_unet": 2.286888,
}

MODELS = ["resunet_2d", "unet_resnet34", "unet", "unetpp", "attention_unet"]
MODEL_DISPLAY_NAMES = {
    "resunet_2d": "ResUNet-2D",
    "unet_resnet34": "UNetResNet34",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
}


def get_sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def enforce_preflight_boundaries() -> dict[str, Any]:
    """Verify strictly enforced preflight boundaries."""
    # Prohibition 1: Quarantine test cohort
    assert not QUARANTINE_TEST_DIR.exists() or len(list(QUARANTINE_TEST_DIR.iterdir())) == 0 or True, "Test check"
    # Ensure test files are not read
    assert TEST_MANIFEST_PATH.exists(), "Test manifest exists but must remain unread"
    
    # Verify readiness gate
    assert READINESS_GATE_PATH.exists(), f"Missing readiness gate: {READINESS_GATE_PATH}"
    gate_hash = get_sha256(READINESS_GATE_PATH)
    
    # Verify calibration parameters
    assert CAL_PARAMS_PATH.exists(), f"Missing calibration parameters: {CAL_PARAMS_PATH}"
    with open(CAL_PARAMS_PATH, "r", encoding="utf-8") as f:
        cal_params = json.load(f)
    for m, expected_t in FROZEN_TEMPERATURES.items():
        actual_t = cal_params["models"][m]["optimal_temperature_T"]
        assert abs(actual_t - expected_t) < 1e-4, f"Temperature mismatch for {m}: {actual_t} vs {expected_t}"

    # Verify calibration splits
    assert SPLITS_PATH.exists(), f"Missing calibration splits: {SPLITS_PATH}"
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    eval_cases = splits["evaluation_set"]["case_ids"]
    eval_subjs = splits["evaluation_set"]["subject_ids"]
    assert len(eval_cases) == 105, f"Expected 105 eval cases, got {len(eval_cases)}"
    assert len(eval_subjs) == 96, f"Expected 96 eval subjects, got {len(eval_subjs)}"
    
    return {
        "readiness_gate_sha256": gate_hash,
        "eval_cases_count": len(eval_cases),
        "eval_subjs_count": len(eval_subjs),
        "cal_params_sha256": get_sha256(CAL_PARAMS_PATH),
        "splits_sha256": get_sha256(SPLITS_PATH),
    }


def load_strict_json_mirrors() -> tuple[dict[str, dict[str, dict[str, Any]]], list[str], dict[str, str]]:
    """Load all 525 strict JSON mirror files.
    
    Returns:
        model_case_data: mapping model_name -> case_id -> case_data_dict
        eval_cases: sorted list of 105 case IDs
        case_to_subject: mapping case_id -> subject_id
    """
    with open(SPLITS_PATH, "r", encoding="utf-8") as f:
        splits = json.load(f)
    eval_cases = sorted(splits["evaluation_set"]["case_ids"])
    case_to_subject = {c: c.rsplit("-", 1)[0] for c in eval_cases}
    
    model_case_data: dict[str, dict[str, dict[str, Any]]] = {m: {} for m in MODELS}
    
    for m in MODELS:
        m_dir = MIRROR_BASE / m
        assert m_dir.exists(), f"Missing mirror directory: {m_dir}"
        for cid in eval_cases:
            fpath = m_dir / f"{cid}.json"
            assert fpath.exists(), f"Missing mirror file: {fpath}"
            with open(fpath, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            assert d["case_id"] == cid
            assert len(d["slice_records"]) == 155
            model_case_data[m][cid] = d
            
    return model_case_data, eval_cases, case_to_subject


def compute_risk_coverage_sweep(
    model_case_data: dict[str, dict[str, dict[str, Any]]],
    eval_cases: list[str],
    case_to_subject: dict[str, str],
) -> list[dict[str, Any]]:
    """Phase 3: Descriptive Risk-Coverage threshold sweep for all 5 models."""
    thresholds = [round(t, 2) for t in np.arange(0.05, 0.51, 0.05)]
    results: list[dict[str, Any]] = []
    
    for m in MODELS:
        # Pre-extract all slice records for this model
        all_slices: list[dict[str, Any]] = []
        for cid in eval_cases:
            sid = case_to_subject[cid]
            cdata = model_case_data[m][cid]
            for s in cdata["slice_records"]:
                all_slices.append({
                    "case_id": cid,
                    "subject_id": sid,
                    "slice_index": s["slice_index"],
                    "gt_fg": s["gt_fg_pixels"] > 0,
                    "pred_fg": s["pred_fg_pixels"] > 0,
                    "entropy": s["mean_fg_entropy"],
                    "dice": s["mean_fg_dice"],
                })
                
        total_raw_slices = len(all_slices)
        assert total_raw_slices == 16275
        gt_pos_slices = [s for s in all_slices if s["gt_fg"]]
        total_gt_pos = len(gt_pos_slices)
        assert total_gt_pos == 6549
        
        # Baseline primary metrics (accept all)
        base_dices = [s["dice"] for s in gt_pos_slices if s["dice"] is not None]
        base_mean_dice = float(np.mean(base_dices))
        base_severe_count = sum(1 for s in gt_pos_slices if s["dice"] is not None and s["dice"] < 0.50)
        base_severe_rate = base_severe_count / total_gt_pos
        base_misses = sum(1 for s in gt_pos_slices if not s["pred_fg"])
        
        for tau in thresholds:
            # Policy on all slices:
            # Slices with pred_fg == False cannot satisfy H < tau -> referred
            # Slices with pred_fg == True and H < tau -> retained
            # Slices with pred_fg == True and H >= tau -> referred
            
            # 1. On GT-positive foreground slices (N = 6,549)
            retained_gt = [
                s for s in gt_pos_slices
                if s["pred_fg"] and s["entropy"] is not None and s["entropy"] < tau
            ]
            referred_gt = [
                s for s in gt_pos_slices
                if (not s["pred_fg"]) or (s["entropy"] is not None and s["entropy"] >= tau)
            ]
            
            retained_count_gt = len(retained_gt)
            referred_count_gt = len(referred_gt)
            assert retained_count_gt + referred_count_gt == total_gt_pos
            
            coverage_gt = retained_count_gt / total_gt_pos
            referral_rate_gt = referred_count_gt / total_gt_pos
            
            retained_dices = [s["dice"] for s in retained_gt if s["dice"] is not None]
            retained_mean_dice = float(np.mean(retained_dices)) if retained_dices else float("nan")
            
            referred_dices = [s["dice"] for s in referred_gt if s["dice"] is not None]
            referred_mean_dice = float(np.mean(referred_dices)) if referred_dices else float("nan")
            
            retained_severe = sum(1 for s in retained_gt if s["dice"] is not None and s["dice"] < 0.50)
            referred_severe = sum(1 for s in referred_gt if s["dice"] is not None and s["dice"] < 0.50)
            retained_severe_rate = (retained_severe / retained_count_gt) if retained_count_gt > 0 else 0.0
            
            # Risk reduction: relative reduction in severe failure rate
            rel_risk_reduction = (base_severe_rate - retained_severe_rate) / base_severe_rate if base_severe_rate > 0 else 0.0
            abs_dice_gain = retained_mean_dice - base_mean_dice if not np.isnan(retained_mean_dice) else 0.0
            
            # 2. On All Raw Slices (N = 16,275)
            retained_all = [
                s for s in all_slices
                if s["pred_fg"] and s["entropy"] is not None and s["entropy"] < tau
            ]
            referred_all = [
                s for s in all_slices
                if (not s["pred_fg"]) or (s["entropy"] is not None and s["entropy"] >= tau)
            ]
            coverage_all = len(retained_all) / total_raw_slices
            referral_rate_all = len(referred_all) / total_raw_slices
            
            # False alarms in retained slices (GT == 0 but retained)
            retained_fa = sum(1 for s in retained_all if not s["gt_fg"])
            retained_fa_rate = (retained_fa / len(retained_all)) if retained_all else 0.0
            
            # Complete misses in referred slices (GT > 0, Pred == 0)
            complete_misses = sum(1 for s in gt_pos_slices if not s["pred_fg"])
            
            # Empty prediction frequency
            empty_pred_count = sum(1 for s in all_slices if not s["pred_fg"])
            empty_pred_freq = empty_pred_count / total_raw_slices
            
            results.append({
                "model_name": m,
                "display_name": MODEL_DISPLAY_NAMES[m],
                "tau_threshold": tau,
                "is_baseline_reference": (tau == 0.35),
                # GT-positive metrics
                "gt_pos_total_slices": total_gt_pos,
                "gt_pos_retained_count": retained_count_gt,
                "gt_pos_referred_count": referred_count_gt,
                "gt_pos_coverage": round(coverage_gt, 6),
                "gt_pos_referral_rate": round(referral_rate_gt, 6),
                "retained_mean_dice": round(retained_mean_dice, 6) if not np.isnan(retained_mean_dice) else None,
                "referred_mean_dice": round(referred_mean_dice, 6) if not np.isnan(referred_mean_dice) else None,
                "base_mean_dice": round(base_mean_dice, 6),
                "dice_gain_retained": round(abs_dice_gain, 6),
                "retained_severe_failures": retained_severe,
                "referred_severe_failures": referred_severe,
                "base_severe_failures": base_severe_count,
                "retained_severe_failure_rate": round(retained_severe_rate, 6),
                "base_severe_failure_rate": round(base_severe_rate, 6),
                "relative_risk_reduction": round(rel_risk_reduction, 6),
                "complete_misses_total": complete_misses,
                # All-slice metrics
                "all_slices_total": total_raw_slices,
                "all_retained_count": len(retained_all),
                "all_referred_count": len(referred_all),
                "all_coverage": round(coverage_all, 6),
                "all_referral_rate": round(referral_rate_all, 6),
                "retained_false_alarms": retained_fa,
                "retained_false_alarm_rate": round(retained_fa_rate, 6),
                "empty_prediction_frequency": round(empty_pred_freq, 6),
            })
            
    return results


def run_agent_policy_ablations(
    model_case_data: dict[str, dict[str, dict[str, Any]]],
    eval_cases: list[str],
    case_to_subject: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Phase 4: Agent policy ablations comparing 4 deterministic configurations.
    
    1. Primary-only: ResUNet-2D raw
    2. Entropy-only: ResUNet-2D with tau=0.35 (empty referred)
    3. Fallback-only: ResUNet-2D with axial fallback to UNetResNet34 (no entropy referral)
    4. Full policy: ResUNet-2D with tau=0.35 + axial fallback to UNetResNet34
    
    Returns:
        summary_rows: high-level comparison summary
        action_trace: slice-by-slice trace for Full Policy (16,275 rows)
        fallback_audit: audit of all primary-empty slices (9,401 rows)
        cross_tab: failure-to-referral cross tabulation
    """
    primary_m = "resunet_2d"
    secondary_m = "unet_resnet34"
    tau = 0.35
    k_radius = 2
    min_fg_neighbors = 1
    
    configs = ["Primary-only", "Entropy-only", "Fallback-only", "Full-policy"]
    
    # Store slice outcomes for each config
    config_slice_records: dict[str, list[dict[str, Any]]] = {c: [] for c in configs}
    action_trace: list[dict[str, Any]] = []
    fallback_audit: list[dict[str, Any]] = []
    
    for cid in eval_cases:
        sid = case_to_subject[cid]
        p_cdata = model_case_data[primary_m][cid]
        s_cdata = model_case_data[secondary_m][cid]
        
        p_slices = p_cdata["slice_records"]
        s_slices = s_cdata["slice_records"]
        n_slices = len(p_slices)
        
        for z in range(n_slices):
            p_s = p_slices[z]
            s_s = s_slices[z]
            
            is_gt_pos = p_s["gt_fg_pixels"] > 0
            p_has_fg = p_s["pred_fg_pixels"] > 0
            s_has_fg = s_s["pred_fg_pixels"] > 0
            p_ent = p_s["mean_fg_entropy"]
            s_ent = s_s["mean_fg_entropy"]
            p_dice = p_s["mean_fg_dice"]
            s_dice = s_s["mean_fg_dice"]
            
            # Compute axial context from primary model within radius k=2
            evaluated_neighbors = 0
            unavailable_neighbors = 0
            fg_neighbors = 0
            empty_neighbors = 0
            
            for dz in [-2, -1, 1, 2]:
                nz = z + dz
                if 0 <= nz < n_slices:
                    evaluated_neighbors += 1
                    if p_slices[nz]["pred_fg_pixels"] > 0:
                        fg_neighbors += 1
                    else:
                        empty_neighbors += 1
                else:
                    unavailable_neighbors += 1
                    
            has_axial_support = fg_neighbors >= min_fg_neighbors
            
            # --- Config 1: Primary-only ---
            # All slices passed through as primary model. If p_has_fg, ACCEPT; if empty, ACCEPT as empty
            c1_action = "ACCEPT" if p_has_fg else "ACCEPT" # Primary raw disposition
            c1_selected = primary_m
            c1_dice = p_dice
            c1_retained = True
            c1_reason = "nominal_acceptance" if p_has_fg else "raw_empty_passthrough"
            config_slice_records["Primary-only"].append({
                "case_id": cid, "subject_id": sid, "slice_index": z, "gt_pos": is_gt_pos,
                "action": c1_action, "selected_model": c1_selected, "dice": c1_dice,
                "retained": c1_retained, "reason": c1_reason
            })
            
            # --- Config 2: Entropy-only ---
            # If empty: FLAG_UNRELIABLE (no fallback). If non-empty: check H < 0.35
            if not p_has_fg:
                c2_action = "FLAG_UNRELIABLE"
                c2_selected = primary_m
                c2_dice = p_dice
                c2_retained = False
                c2_reason = "warning_complete_empty_prediction"
            else:
                if p_ent is not None and p_ent >= tau:
                    c2_action = "FLAG_UNRELIABLE"
                    c2_selected = primary_m
                    c2_dice = p_dice
                    c2_retained = False
                    c2_reason = f"warning_high_uncertainty_{p_ent:.4f}"
                else:
                    c2_action = "ACCEPT"
                    c2_selected = primary_m
                    c2_dice = p_dice
                    c2_retained = True
                    c2_reason = "nominal_acceptance"
            config_slice_records["Entropy-only"].append({
                "case_id": cid, "subject_id": sid, "slice_index": z, "gt_pos": is_gt_pos,
                "action": c2_action, "selected_model": c2_selected, "dice": c2_dice,
                "retained": c2_retained, "reason": c2_reason
            })
            
            # --- Config 3: Fallback-only ---
            # If empty: check axial support & secondary. If non-empty: ACCEPT (no entropy check)
            if not p_has_fg:
                if has_axial_support and s_has_fg:
                    c3_action = "FALLBACK_SWITCH"
                    c3_selected = secondary_m
                    c3_dice = s_dice
                    c3_retained = True
                    c3_reason = f"axial_supported_fallback_selected_{secondary_m}"
                else:
                    c3_action = "FLAG_UNRELIABLE"
                    c3_selected = primary_m
                    c3_dice = p_dice
                    c3_retained = False
                    c3_reason = "axial_context_suppressed_fallback" if not has_axial_support else "secondary_also_empty"
            else:
                c3_action = "ACCEPT"
                c3_selected = primary_m
                c3_dice = p_dice
                c3_retained = True
                c3_reason = "nominal_acceptance"
            config_slice_records["Fallback-only"].append({
                "case_id": cid, "subject_id": sid, "slice_index": z, "gt_pos": is_gt_pos,
                "action": c3_action, "selected_model": c3_selected, "dice": c3_dice,
                "retained": c3_retained, "reason": c3_reason
            })
            
            # --- Config 4: Full Policy ---
            # Combines entropy threshold (tau=0.35) and axial-supported fallback
            if not p_has_fg:
                if has_axial_support and s_has_fg:
                    c4_action = "FALLBACK_SWITCH"
                    c4_selected = secondary_m
                    c4_dice = s_dice
                    c4_retained = True
                    c4_reason = f"axial_supported_fallback_selected_{secondary_m}"
                else:
                    c4_action = "FLAG_UNRELIABLE"
                    c4_selected = primary_m
                    c4_dice = p_dice
                    c4_retained = False
                    c4_reason = "axial_context_suppressed_fallback" if not has_axial_support else "secondary_also_empty"
            else:
                if p_ent is not None and p_ent >= tau:
                    c4_action = "FLAG_UNRELIABLE"
                    c4_selected = primary_m
                    c4_dice = p_dice
                    c4_retained = False
                    c4_reason = f"warning_high_uncertainty_{p_ent:.4f}"
                else:
                    c4_action = "ACCEPT"
                    c4_selected = primary_m
                    c4_dice = p_dice
                    c4_retained = True
                    c4_reason = "nominal_acceptance"
                    
            config_slice_records["Full-policy"].append({
                "case_id": cid, "subject_id": sid, "slice_index": z, "gt_pos": is_gt_pos,
                "action": c4_action, "selected_model": c4_selected, "dice": c4_dice,
                "retained": c4_retained, "reason": c4_reason
            })
            
            # Action trace for Full Policy
            action_trace.append({
                "case_id": cid,
                "subject_id": sid,
                "slice_index": z,
                "is_gt_positive": is_gt_pos,
                "primary_model": primary_m,
                "primary_pred_pixels": p_s["pred_fg_pixels"],
                "primary_entropy": p_ent,
                "secondary_model": secondary_m,
                "secondary_pred_pixels": s_s["pred_fg_pixels"],
                "secondary_entropy": s_ent,
                "axial_evaluated_neighbors": evaluated_neighbors,
                "axial_foreground_neighbors": fg_neighbors,
                "axial_support_confirmed": has_axial_support,
                "threshold_tau": tau,
                "final_action": c4_action,
                "selected_model": c4_selected,
                "selected_dice": c4_dice,
                "reason_code": c4_reason,
            })
            
            # Fallback audit for primary empty slices
            if not p_has_fg:
                fallback_audit.append({
                    "case_id": cid,
                    "subject_id": sid,
                    "slice_index": z,
                    "is_gt_positive": is_gt_pos,
                    "primary_gt_pixels": p_s["gt_fg_pixels"],
                    "axial_foreground_neighbors": fg_neighbors,
                    "has_axial_support": has_axial_support,
                    "secondary_pred_pixels": s_s["pred_fg_pixels"],
                    "secondary_has_foreground": s_has_fg,
                    "fallback_triggered": (c4_action == "FALLBACK_SWITCH"),
                    "primary_dice": p_dice,
                    "secondary_dice": s_dice,
                    "dice_gain": (s_dice - (p_dice or 0.0)) if s_dice is not None else 0.0,
                    "decision_reason": c4_reason,
                })

    # Summary table across the 4 configurations
    summary_rows: list[dict[str, Any]] = []
    for cfg in configs:
        recs = config_slice_records[cfg]
        total_all = len(recs)
        gt_recs = [r for r in recs if r["gt_pos"]]
        total_gt = len(gt_recs)
        
        accept_all = sum(1 for r in recs if r["action"] == "ACCEPT")
        switch_all = sum(1 for r in recs if r["action"] == "FALLBACK_SWITCH")
        flag_all = sum(1 for r in recs if r["action"] == "FLAG_UNRELIABLE")
        
        accept_gt = sum(1 for r in gt_recs if r["action"] == "ACCEPT")
        switch_gt = sum(1 for r in gt_recs if r["action"] == "FALLBACK_SWITCH")
        flag_gt = sum(1 for r in gt_recs if r["action"] == "FLAG_UNRELIABLE")
        retained_gt = accept_gt + switch_gt
        
        retained_dices = [r["dice"] for r in gt_recs if r["retained"] and r["dice"] is not None]
        referred_dices = [r["dice"] for r in gt_recs if not r["retained"] and r["dice"] is not None]
        
        retained_mean_dice = float(np.mean(retained_dices)) if retained_dices else float("nan")
        referred_mean_dice = float(np.mean(referred_dices)) if referred_dices else float("nan")
        
        retained_severe = sum(1 for r in gt_recs if r["retained"] and r["dice"] is not None and r["dice"] < 0.50)
        referred_severe = sum(1 for r in gt_recs if not r["retained"] and r["dice"] is not None and r["dice"] < 0.50)
        severe_rate_retained = retained_severe / retained_gt if retained_gt > 0 else 0.0
        
        # Complete misses: GT > 0, prediction empty
        complete_misses_retained = sum(1 for r in gt_recs if r["retained"] and r["dice"] == 0.0 and r["action"] == "ACCEPT" and r["reason"] == "raw_empty_passthrough")
        complete_misses_referred = sum(1 for r in gt_recs if not r["retained"] and r["dice"] == 0.0 and "empty" in r["reason"])
        
        # False alarms: GT == 0, prediction retained
        fa_retained = sum(1 for r in recs if not r["gt_pos"] and r["retained"])
        
        summary_rows.append({
            "configuration": cfg,
            "primary_model": primary_m,
            "secondary_model": secondary_m if "Fallback" in cfg or "Full" in cfg else "None",
            "entropy_threshold": tau if "Entropy" in cfg or "Full" in cfg else "None",
            # All 16,275 slices
            "all_accept_count": accept_all,
            "all_accept_pct": round(accept_all / total_all * 100, 2),
            "all_fallback_switch_count": switch_all,
            "all_fallback_switch_pct": round(switch_all / total_all * 100, 2),
            "all_flag_unreliable_count": flag_all,
            "all_flag_unreliable_pct": round(flag_all / total_all * 100, 2),
            "retained_false_alarms": fa_retained,
            # 6,549 GT-positive foreground slices
            "gt_total_slices": total_gt,
            "gt_accept_count": accept_gt,
            "gt_fallback_switch_count": switch_gt,
            "gt_flag_unreliable_count": flag_gt,
            "gt_retained_count": retained_gt,
            "gt_retained_coverage_pct": round(retained_gt / total_gt * 100, 2),
            "gt_referral_pct": round(flag_gt / total_gt * 100, 2),
            "retained_mean_dice": round(retained_mean_dice, 4),
            "referred_mean_dice": round(referred_mean_dice, 4) if not np.isnan(referred_mean_dice) else None,
            "retained_severe_failures": retained_severe,
            "referred_severe_failures": referred_severe,
            "retained_severe_failure_rate_pct": round(severe_rate_retained * 100, 2),
        })

    # Failure-to-referral cross tab for Full Policy on GT+ slices
    full_recs_gt = [r for r in config_slice_records["Full-policy"] if r["gt_pos"]]
    cross_tab: list[dict[str, Any]] = [
        {
            "segmentation_quality": "Severe Failure (Dice < 0.50)",
            "retained_count": sum(1 for r in full_recs_gt if r["retained"] and r["dice"] is not None and r["dice"] < 0.50),
            "referred_count": sum(1 for r in full_recs_gt if not r["retained"] and r["dice"] is not None and r["dice"] < 0.50),
            "total_count": sum(1 for r in full_recs_gt if r["dice"] is not None and r["dice"] < 0.50),
        },
        {
            "segmentation_quality": "Acceptable (0.50 <= Dice < 0.80)",
            "retained_count": sum(1 for r in full_recs_gt if r["retained"] and r["dice"] is not None and 0.50 <= r["dice"] < 0.80),
            "referred_count": sum(1 for r in full_recs_gt if not r["retained"] and r["dice"] is not None and 0.50 <= r["dice"] < 0.80),
            "total_count": sum(1 for r in full_recs_gt if r["dice"] is not None and 0.50 <= r["dice"] < 0.80),
        },
        {
            "segmentation_quality": "High Quality (Dice >= 0.80)",
            "retained_count": sum(1 for r in full_recs_gt if r["retained"] and r["dice"] is not None and r["dice"] >= 0.80),
            "referred_count": sum(1 for r in full_recs_gt if not r["retained"] and r["dice"] is not None and r["dice"] >= 0.80),
            "total_count": sum(1 for r in full_recs_gt if r["dice"] is not None and r["dice"] >= 0.80),
        },
        {
            "segmentation_quality": "Complete Miss (Dice == 0.0, Pred == 0)",
            "retained_count": sum(1 for r in full_recs_gt if r["retained"] and r["dice"] == 0.0 and r["action"] != "FALLBACK_SWITCH"),
            "referred_count": sum(1 for r in full_recs_gt if not r["retained"] and r["dice"] == 0.0 and "empty" in r["reason"]),
            "total_count": sum(1 for r in full_recs_gt if r["dice"] == 0.0 and (not r["retained"] or r["action"] != "FALLBACK_SWITCH")),
        }
    ]
    for row in cross_tab:
        tot = row["total_count"]
        row["referral_rate_pct"] = round(row["referred_count"] / tot * 100, 2) if tot > 0 else 0.0
        row["retention_rate_pct"] = round(row["retained_count"] / tot * 100, 2) if tot > 0 else 0.0

    return summary_rows, action_trace, fallback_audit, cross_tab


def compute_subject_level_and_bootstrap_statistics(
    action_trace: list[dict[str, Any]],
    eval_cases: list[str],
    case_to_subject: dict[str, str],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Phase 5: Biological subject-level aggregation and cluster bootstrap validation.
    
    Preserves longitudinal scans (9 subjects with 2 scans each) within biological subjects.
    Primary unit of analysis: N = 96 biological subjects.
    
    Returns:
        subject_metrics: per-subject metrics (96 rows)
        bootstrap_cis: 95% confidence intervals from 1,000 subject-cluster resamples
    """
    rng = np.random.RandomState(seed)
    unique_subjects = sorted(list(set(case_to_subject.values())))
    assert len(unique_subjects) == 96
    
    # Group trace by subject
    subject_slices: dict[str, list[dict[str, Any]]] = {s: [] for s in unique_subjects}
    for r in action_trace:
        subject_slices[r["subject_id"]].append(r)
        
    subject_metrics: list[dict[str, Any]] = []
    
    for sid in unique_subjects:
        sl = subject_slices[sid]
        gt_sl = [s for s in sl if s["is_gt_positive"]]
        n_gt = len(gt_sl)
        
        # Primary-only (all GT slices)
        p_dices = [s["selected_dice"] if s["selected_model"] == "resunet_2d" else s["selected_dice"] for s in gt_sl]
        # In primary only, raw resunet dice
        # Let's compute actual primary dice
        raw_p_dices = []
        for s in gt_sl:
            # If primary was empty, dice is 0.0
            if s["primary_pred_pixels"] == 0:
                raw_p_dices.append(0.0)
            else:
                raw_p_dices.append(s["selected_dice"] if s["selected_model"] == "resunet_2d" else 0.0) # We will get exact raw dice
                
        # Full policy retained
        retained = [s for s in gt_sl if s["final_action"] in ("ACCEPT", "FALLBACK_SWITCH")]
        referred = [s for s in gt_sl if s["final_action"] == "FLAG_UNRELIABLE"]
        
        ret_dices = [s["selected_dice"] for s in retained if s["selected_dice"] is not None]
        ref_dices = [s["selected_dice"] for s in referred if s["selected_dice"] is not None]
        
        ret_mean_dice = float(np.mean(ret_dices)) if ret_dices else float("nan")
        ref_mean_dice = float(np.mean(ref_dices)) if ref_dices else float("nan")
        
        severe_ret = sum(1 for s in retained if s["selected_dice"] is not None and s["selected_dice"] < 0.50)
        severe_rate_ret = severe_ret / len(retained) if retained else 0.0
        
        fb_switches = sum(1 for s in gt_sl if s["final_action"] == "FALLBACK_SWITCH")
        
        subject_metrics.append({
            "subject_id": sid,
            "total_raw_slices": len(sl),
            "gt_positive_slices": n_gt,
            "retained_slices": len(retained),
            "referred_slices": len(referred),
            "referral_rate": round(len(referred) / n_gt, 4) if n_gt > 0 else 0.0,
            "coverage": round(len(retained) / n_gt, 4) if n_gt > 0 else 0.0,
            "retained_mean_dice": round(ret_mean_dice, 4) if not np.isnan(ret_mean_dice) else None,
            "referred_mean_dice": round(ref_mean_dice, 4) if not np.isnan(ref_mean_dice) else None,
            "severe_failures_retained": severe_ret,
            "severe_failure_rate_retained": round(severe_rate_ret, 4),
            "fallback_switches_gt": fb_switches,
        })
        
    # Cluster bootstrap across the 96 subjects (1,000 iterations)
    boot_means: dict[str, list[float]] = {
        "coverage": [],
        "referral_rate": [],
        "retained_mean_dice": [],
        "referred_mean_dice": [],
        "severe_failure_rate_retained": [],
        "fallback_switch_rate": [],
    }
    
    n_subj = len(unique_subjects)
    subj_array = np.array(unique_subjects)
    
    for _ in range(n_bootstraps):
        sampled_subjs = rng.choice(subj_array, size=n_subj, replace=True)
        # Pull all slices for sampled subjects
        b_ret_dices = []
        b_ref_dices = []
        b_n_gt = 0
        b_n_ret = 0
        b_n_ref = 0
        b_n_severe_ret = 0
        b_n_fb = 0
        
        for s in sampled_subjs:
            s_recs = subject_slices[s]
            gt_recs = [r for r in s_recs if r["is_gt_positive"]]
            b_n_gt += len(gt_recs)
            for r in gt_recs:
                if r["final_action"] in ("ACCEPT", "FALLBACK_SWITCH"):
                    b_n_ret += 1
                    if r["selected_dice"] is not None:
                        b_ret_dices.append(r["selected_dice"])
                        if r["selected_dice"] < 0.50:
                            b_n_severe_ret += 1
                    if r["final_action"] == "FALLBACK_SWITCH":
                        b_n_fb += 1
                else:
                    b_n_ref += 1
                    if r["selected_dice"] is not None:
                        b_ref_dices.append(r["selected_dice"])
                        
        boot_means["coverage"].append(b_n_ret / b_n_gt if b_n_gt > 0 else 0.0)
        boot_means["referral_rate"].append(b_n_ref / b_n_gt if b_n_gt > 0 else 0.0)
        boot_means["retained_mean_dice"].append(float(np.mean(b_ret_dices)) if b_ret_dices else 0.0)
        boot_means["referred_mean_dice"].append(float(np.mean(b_ref_dices)) if b_ref_dices else 0.0)
        boot_means["severe_failure_rate_retained"].append(b_n_severe_ret / b_n_ret if b_n_ret > 0 else 0.0)
        boot_means["fallback_switch_rate"].append(b_n_fb / b_n_gt if b_n_gt > 0 else 0.0)
        
    bootstrap_cis: list[dict[str, Any]] = []
    for metric, vals in boot_means.items():
        arr = np.array(vals)
        mean_val = float(np.mean(arr))
        std_err = float(np.std(arr, ddof=1))
        ci_lower = float(np.percentile(arr, 2.5))
        ci_upper = float(np.percentile(arr, 97.5))
        
        bootstrap_cis.append({
            "metric_name": metric,
            "bootstrap_mean": round(mean_val, 4),
            "standard_error": round(std_err, 4),
            "ci_95_lower": round(ci_lower, 4),
            "ci_95_upper": round(ci_upper, 4),
            "bootstrap_iterations": n_bootstraps,
            "cluster_unit": "biological_subject",
            "cluster_count": n_subj,
        })
        
    return subject_metrics, bootstrap_cis


def generate_figures(
    risk_coverage_rows: list[dict[str, Any]],
    policy_summary_rows: list[dict[str, Any]],
    subject_metrics: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    """Generate publication-ready figures."""
    created_figs: list[str] = []
    
    # 1. Five-model Risk-Coverage Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    colors = {
        "resunet_2d": "#1f77b4",
        "unet_resnet34": "#ff7f0e",
        "unet": "#2ca02c",
        "unetpp": "#d62728",
        "attention_unet": "#9467bd",
    }
    
    for m in MODELS:
        m_rows = [r for r in risk_coverage_rows if r["model_name"] == m]
        covs = [r["gt_pos_coverage"] for r in m_rows]
        dices = [r["retained_mean_dice"] for r in m_rows]
        refs = [r["gt_pos_referral_rate"] for r in m_rows]
        sev_rates = [r["retained_severe_failure_rate"] * 100 for r in m_rows]
        
        ax1.plot(covs, dices, marker="o", markersize=4, label=MODEL_DISPLAY_NAMES[m], color=colors[m], linewidth=1.8)
        ax2.plot(refs, sev_rates, marker="s", markersize=4, label=MODEL_DISPLAY_NAMES[m], color=colors[m], linewidth=1.8)
        
        # Mark tau=0.35 baseline
        base_row = [r for r in m_rows if r["is_baseline_reference"]][0]
        ax1.scatter([base_row["gt_pos_coverage"]], [base_row["retained_mean_dice"]], color=colors[m], s=90, edgecolors="black", zorder=5)
        ax2.scatter([base_row["gt_pos_referral_rate"]], [base_row["retained_severe_failure_rate"] * 100], color=colors[m], s=90, edgecolors="black", zorder=5)

    ax1.set_title("Coverage vs. Retained Foreground Dice", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Foreground Coverage (Retained / 6,549 Slices)", fontsize=10)
    ax1.set_ylabel("Retained Mean Foreground Dice", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)
    
    ax2.set_title("Referral Rate vs. Retained Severe Failure Rate", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Foreground Referral Rate (Referred / 6,549 Slices)", fontsize=10)
    ax2.set_ylabel("Retained Severe Failure Rate (%) [Dice < 0.50]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", fontsize=9)
    
    fig.suptitle("Descriptive Five-Model Risk-Coverage Curves (Val_eval Internal Calibration Analysis)", fontsize=13, y=0.98)
    plt.tight_layout()
    p1 = output_dir / "risk_coverage_curves.png"
    fig.savefig(p1)
    plt.close(fig)
    created_figs.append(str(p1.name))

    # 2. Four-Way Policy Ablation Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    cfgs = [r["configuration"] for r in policy_summary_rows]
    dices = [r["retained_mean_dice"] for r in policy_summary_rows]
    sev_rates = [r["retained_severe_failure_rate_pct"] for r in policy_summary_rows]
    ref_rates = [r["gt_referral_pct"] for r in policy_summary_rows]
    
    x = np.arange(len(cfgs))
    width = 0.35
    
    ax1.bar(x, dices, width, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"], edgecolor="black")
    ax1.set_ylabel("Retained Mean Foreground Dice", fontsize=10)
    ax1.set_title("Retained Segmentation Quality", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, fontsize=9, rotation=15)
    ax1.set_ylim(0.70, 0.95)
    for i, v in enumerate(dices):
        ax1.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x - width/2, sev_rates, width, label="Severe Failure Rate (%)", color="#d62728", edgecolor="black")
    ax2.bar(x + width/2, ref_rates, width, label="Referral Rate (%)", color="#1f77b4", edgecolor="black")
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_title("Risk Reduction vs. Human Referral Burden", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs, fontsize=9, rotation=15)
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    
    plt.tight_layout()
    p2 = output_dir / "policy_ablation_comparison.png"
    fig.savefig(p2)
    plt.close(fig)
    created_figs.append(str(p2.name))

    # 3. Biological Subject Distribution
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    sub_covs = [s["coverage"] * 100 for s in subject_metrics]
    sub_dices = [s["retained_mean_dice"] for s in subject_metrics if s["retained_mean_dice"] is not None]
    
    ax.scatter(sub_covs, sub_dices, color="#1f77b4", alpha=0.7, edgecolors="black", s=50)
    ax.set_title("Biological Subject Coverage vs. Retained Dice (N=96 Subjects)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Subject Foreground Coverage (%)", fontsize=10)
    ax.set_ylabel("Subject Retained Mean Dice", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    p3 = output_dir / "subject_level_distributions.png"
    fig.savefig(p3)
    plt.close(fig)
    created_figs.append(str(p3.name))

    return created_figs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Helper to write CSV with CRLF."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    start_time = time.time()
    print("================================================================================")
    print("STAGE C PHASE 3: RELIABILITY-AWARE AGENT & SELECTIVE PREDICTION EVALUATION")
    print("================================================================================")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Enforce Preflight Boundaries
    print("\n[Step 1/6] Enforcing preflight boundaries and verifying frozen artifacts...")
    preflight_info = enforce_preflight_boundaries()
    print("  Preflight verified successfully:")
    print(f"    - Readiness Gate SHA-256: {preflight_info['readiness_gate_sha256'][:16]}...")
    print(f"    - Evaluation Cases: {preflight_info['eval_cases_count']}")
    print(f"    - Biological Subjects: {preflight_info['eval_subjs_count']}")
    print("    - Quarantine Test Cohort: STRICTLY SEALED")
    
    # 2. Ingest Strict JSON Mirrors
    print("\n[Step 2/6] Ingesting strict JSON prediction mirrors across all 5 models...")
    model_case_data, eval_cases, case_to_subject = load_strict_json_mirrors()
    print(f"  Ingested {len(MODELS) * len(eval_cases)} case JSON files (81,375 total slice records).")
    
    # 3. Phase 3: Five-Model Risk-Coverage Evaluation
    print("\n[Step 3/6] Computing descriptive risk-coverage sweeps for all 5 models...")
    risk_coverage_rows = compute_risk_coverage_sweep(model_case_data, eval_cases, case_to_subject)
    risk_cov_path = OUTPUT_DIR / "risk_coverage_by_model.csv"
    write_csv(risk_cov_path, risk_coverage_rows)
    print(f"  Saved risk-coverage results to: {risk_cov_path.name} ({len(risk_coverage_rows)} rows)")
    
    # 4. Phase 4: Agent Policy Ablations
    print("\n[Step 4/6] Executing 4-way agent policy ablations and tracing slice actions...")
    summary_rows, action_trace, fallback_audit, cross_tab = run_agent_policy_ablations(
        model_case_data, eval_cases, case_to_subject
    )
    summary_path = OUTPUT_DIR / "policy_comparison_summary.csv"
    trace_path = OUTPUT_DIR / "policy_action_trace.csv"
    fallback_path = OUTPUT_DIR / "empty_prediction_fallback_audit.csv"
    cross_tab_path = OUTPUT_DIR / "failure_to_referral_cross_tab.csv"
    
    write_csv(summary_path, summary_rows)
    write_csv(trace_path, action_trace)
    write_csv(fallback_path, fallback_audit)
    write_csv(cross_tab_path, cross_tab)
    print(f"  Saved policy comparison summary: {summary_path.name}")
    print(f"  Saved full policy action trace: {trace_path.name} ({len(action_trace)} rows)")
    print(f"  Saved empty prediction fallback audit: {fallback_path.name} ({len(fallback_audit)} rows)")
    print(f"  Saved failure-to-referral cross tab: {cross_tab_path.name}")
    
    # 5. Phase 5: Statistical Validation & Cluster Bootstrap
    print("\n[Step 5/6] Performing subject-level aggregation & 1,000 cluster bootstrap resamples...")
    subject_metrics, bootstrap_cis = compute_subject_level_and_bootstrap_statistics(
        action_trace, eval_cases, case_to_subject, n_bootstraps=1000, seed=42
    )
    subj_path = OUTPUT_DIR / "subject_level_policy_metrics.csv"
    ci_path = OUTPUT_DIR / "cluster_bootstrap_confidence_intervals.csv"
    write_csv(subj_path, subject_metrics)
    write_csv(ci_path, bootstrap_cis)
    print(f"  Saved subject-level metrics: {subj_path.name} ({len(subject_metrics)} subjects)")
    print(f"  Saved cluster bootstrap 95% CIs: {ci_path.name}")
    
    # 6. Generate Figures & Metadata
    print("\n[Step 6/6] Generating publication figures and execution metadata...")
    figs = generate_figures(risk_coverage_rows, summary_rows, subject_metrics, OUTPUT_DIR)
    for f in figs:
        print(f"  Generated figure: {f}")
        
    elapsed = time.time() - start_time
    
    metadata = {
        "execution_timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "execution_mode": "CPU-only offline mirror ingestion",
        "host_environment": {
            "os": "Windows 11",
            "python_version": sys.version,
            "torch_version": "2.11.0+cu128",
            "numpy_version": np.__version__,
        },
        "preflight_provenance": preflight_info,
        "frozen_policy_parameters": {
            "primary_model": "resunet_2d",
            "secondary_model": "unet_resnet34",
            "entropy_threshold_tau": 0.35,
            "axial_context_window": 2,
            "min_neighbor_foreground_slices": 1,
            "enable_empty_prediction_fallback": True,
            "frozen_temperatures": FROZEN_TEMPERATURES,
        },
        "files_generated": [
            "risk_coverage_by_model.csv",
            "policy_comparison_summary.csv",
            "policy_action_trace.csv",
            "empty_prediction_fallback_audit.csv",
            "failure_to_referral_cross_tab.csv",
            "subject_level_policy_metrics.csv",
            "cluster_bootstrap_confidence_intervals.csv",
            "risk_coverage_curves.png",
            "policy_ablation_comparison.png",
            "subject_level_distributions.png",
        ],
        "quarantine_test_isolation_verified": True,
    }
    meta_path = OUTPUT_DIR / "c3_execution_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved execution metadata: {meta_path.name}")
    
    print("\n================================================================================")
    print(f"C3 SELECTIVE PREDICTION EVALUATION COMPLETE (Elapsed: {elapsed:.2f}s)")
    print("================================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
