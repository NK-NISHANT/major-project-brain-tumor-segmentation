"""
Unit tests for Stage C Phase 3: Work Package C3 v1.1.3.
Verifies presentation updates, external legends, footnote placements,
preserved statistical quantities, and artifact integrity.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats

C3_V1_1_3_DIR = "results/stage_c/phase3/c3_selective_prediction_v1_1_3"


def test_directory_and_deliverable_files_exist():
    """Verify all 17 deliverables exist in v1_1_3 directory."""
    expected_files = [
        "c3_execution_metadata.json",
        "c3_input_schema_validation.json",
        "C3_PREFLIGHT_REPORT_v1.1.3.md",
        "C3_SELECTIVE_PREDICTION_REPORT_v1.1.3.md",
        "cluster_bootstrap_confidence_intervals.csv",
        "paired_hypothesis_tests.csv",
        "policy_action_trace.csv",
        "empty_prediction_fallback_audit.csv",
        "failure_to_referral_cross_tab.csv",
        "segmentation_quality_tiers.csv",
        "diagnostic_anatomy_zero_overlap.csv",
        "policy_comparison_summary.csv",
        "risk_coverage_by_model.csv",
        "subject_level_policy_metrics.csv",
        "policy_ablation_comparison.png",
        "risk_coverage_curves.png",
        "subject_level_distributions.png",
    ]
    for fn in expected_files:
        fp = os.path.join(C3_V1_1_3_DIR, fn)
        assert os.path.exists(fp), f"Missing expected file: {fn}"
        assert os.path.getsize(fp) > 0, f"Empty file: {fn}"


def test_scientific_mandate_and_triage_tradeoff_scope():
    """Verify exact scientific interpretation text and workflow triage scope in metadata."""
    meta_path = os.path.join(C3_V1_1_3_DIR, "c3_execution_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    expected_mandate = (
        "Selective prediction significantly increases the mean segmentation quality among retained slices "
        "compared with the unfiltered baseline, while reducing coverage and referring lower-confidence predictions. "
        "This comparison measures the mean quality of the selected/retained subset against the unfiltered primary baseline, "
        "and does not demonstrate that the agent improved the underlying segmentation model or improved Dice over all slices."
    )
    assert meta["scientific_interpretation_mandate"] == expected_mandate
    assert "workflow-level triage trade-offs" in meta["triage_trade_offs_scope"]


def test_preserved_subject_macro_statistics():
    """Verify unchanged subject-macro statistics and bootstrap results."""
    subj_path = os.path.join(C3_V1_1_3_DIR, "subject_level_policy_metrics.csv")
    df = pd.read_csv(subj_path)
    assert len(df) == 96

    subj_ret = df["retained_mean_dice"].values
    subj_base = df["base_mean_dice"].values
    subj_delta = subj_ret - subj_base

    assert np.isclose(np.mean(subj_ret), 0.752105, atol=1e-5)
    assert np.isclose(np.mean(subj_base), 0.704260, atol=1e-5)
    assert np.isclose(np.mean(subj_delta), 0.047845, atol=1e-5)

    # Bootstrap
    np.random.seed(42)
    n = len(df)
    boot_deltas = [np.mean(subj_delta[np.random.choice(n, size=n, replace=True)]) for _ in range(1000)]
    boot_deltas = np.array(boot_deltas)

    assert np.isclose(np.mean(boot_deltas), 0.047455, atol=1e-5)
    assert np.isclose(np.std(boot_deltas, ddof=1), 0.005624, atol=1e-5)
    assert np.isclose(np.percentile(boot_deltas, 2.5), 0.037198, atol=1e-5)
    assert np.isclose(np.percentile(boot_deltas, 97.5), 0.059234, atol=1e-5)


def test_preserved_wilcoxon_signed_rank_test():
    """Verify unchanged paired Wilcoxon test."""
    subj_path = os.path.join(C3_V1_1_3_DIR, "subject_level_policy_metrics.csv")
    df = pd.read_csv(subj_path)
    subj_ret = df["retained_mean_dice"].values
    subj_base = df["base_mean_dice"].values
    diffs = subj_ret - subj_base

    assert (diffs > 0).sum() == 86
    assert (diffs < 0).sum() == 6
    assert (diffs == 0).sum() == 4

    res = stats.wilcoxon(subj_ret, subj_base, zero_method="wilcox", alternative="two-sided")
    assert np.isclose(res.statistic, 229.0, atol=1e-3)
    assert np.isclose(res.pvalue, 1.0270e-13, rtol=1e-2)


def test_preserved_action_and_fallback_counts():
    """Verify unchanged action counts, fallback counts, and quality tier sums."""
    trace_path = os.path.join(C3_V1_1_3_DIR, "policy_action_trace.csv")
    trace_df = pd.read_csv(trace_path)
    assert len(trace_df) == 16275

    assert (trace_df["final_action"] == "ACCEPT").sum() == 5619
    assert (trace_df["final_action"] == "FALLBACK_SWITCH").sum() == 147
    assert (trace_df["final_action"] == "FLAG_UNRELIABLE").sum() == 10509

    gt_pos = trace_df[trace_df["is_gt_positive"] == True]
    assert len(gt_pos) == 6549
    assert (gt_pos["final_action"] == "ACCEPT").sum() == 5437
    assert (gt_pos["final_action"] == "FALLBACK_SWITCH").sum() == 54
    assert (gt_pos["final_action"] == "FLAG_UNRELIABLE").sum() == 1058

    tiers_df = pd.read_csv(os.path.join(C3_V1_1_3_DIR, "segmentation_quality_tiers.csv"))
    assert tiers_df["total_count"].sum() == 6549


def test_plotting_script_exists_and_runs():
    """Verify plotting script exists."""
    script_path = "tools/plot_stage_c_c3_figures_v1_1_3.py"
    assert os.path.exists(script_path)
    assert os.path.getsize(script_path) > 0
