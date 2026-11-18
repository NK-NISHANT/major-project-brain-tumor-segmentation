"""
Unit tests for Stage C Phase 3: Work Package C3 v1.1.1.
Verifies statistical aggregations, subject-macro bootstrap, Wilcoxon signed-rank test,
numerical precision reconciliations, and provenance boundaries.
"""

import os
import json
import numpy as np
import pandas as pd
import pytest
from scipy import stats

C3_V1_1_1_DIR = "results/stage_c/phase3/c3_selective_prediction_v1_1_1"


def test_authoritative_checkpoint_hashes():
    """Verify all 5 checkpoint hashes in execution metadata match Readiness Gate v1.4.1."""
    meta_path = os.path.join(C3_V1_1_1_DIR, "c3_execution_metadata.json")
    assert os.path.exists(meta_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    ckpts = meta["authoritative_checkpoints"]
    assert ckpts["resunet_2d"]["sha256"] == "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d"
    assert ckpts["unet_resnet34"]["sha256"] == "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d"
    assert ckpts["unet"]["sha256"] == "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef"
    assert ckpts["unetpp"]["sha256"] == "fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6"
    assert ckpts["attention_unet"]["sha256"] == "5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e"


def test_subject_macro_statistics_and_bootstrap():
    """Verify subject-macro mean Dice and primary bootstrap calculations."""
    subj_path = os.path.join(C3_V1_1_1_DIR, "subject_level_policy_metrics.csv")
    assert os.path.exists(subj_path)
    df = pd.read_csv(subj_path)
    assert len(df) == 96

    subj_ret = df["retained_mean_dice"].values
    subj_base = df["base_mean_dice"].values
    subj_delta = subj_ret - subj_base

    # Check sample means
    assert np.isclose(np.mean(subj_ret), 0.752105, atol=1e-5)
    assert np.isclose(np.mean(subj_base), 0.704260, atol=1e-5)
    assert np.isclose(np.mean(subj_delta), 0.047845, atol=1e-5)

    # Primary bootstrap (1000 resamples, seed 42)
    np.random.seed(42)
    n = len(df)
    boot_deltas = [np.mean(subj_delta[np.random.choice(n, size=n, replace=True)]) for _ in range(1000)]
    boot_deltas = np.array(boot_deltas)

    b_mean = np.mean(boot_deltas)
    b_se = np.std(boot_deltas, ddof=1)
    ci_lower = np.percentile(boot_deltas, 2.5)
    ci_upper = np.percentile(boot_deltas, 97.5)

    assert np.isclose(b_mean, 0.047455, atol=1e-5)
    assert np.isclose(b_se, 0.005624, atol=1e-5)
    assert np.isclose(ci_lower, 0.037198, atol=1e-5)
    assert np.isclose(ci_upper, 0.059234, atol=1e-5)


def test_paired_wilcoxon_signed_rank_test():
    """Verify paired Wilcoxon signed-rank test on 96 biological subjects."""
    subj_path = os.path.join(C3_V1_1_1_DIR, "subject_level_policy_metrics.csv")
    df = pd.read_csv(subj_path)

    subj_ret = df["retained_mean_dice"].values
    subj_base = df["base_mean_dice"].values
    diffs = subj_ret - subj_base

    pos = (diffs > 0).sum()
    neg = (diffs < 0).sum()
    zero = (diffs == 0).sum()

    assert pos == 86
    assert neg == 6
    assert zero == 4

    res = stats.wilcoxon(subj_ret, subj_base, zero_method="wilcox", alternative="two-sided")
    assert np.isclose(res.statistic, 229.0, atol=1e-3)
    assert np.isclose(res.pvalue, 1.0270e-13, rtol=1e-2)

    # Verify documented in paired_hypothesis_tests.csv
    hyp_path = os.path.join(C3_V1_1_1_DIR, "paired_hypothesis_tests.csv")
    assert os.path.exists(hyp_path)
    hyp_df = pd.read_csv(hyp_path)
    assert len(hyp_df) >= 1
    w_row = hyp_df[hyp_df["test_id"] == "TEST-WILCOXON-WILCOX-01"].iloc[0]
    assert w_row["test_statistic"] == 229.0
    assert np.isclose(w_row["p_value"], 1.0270e-13, rtol=1e-2)


def test_case_flagged_slices_dispersion():
    """Verify case-level flagged slice mean, median, min, max, sample SD, and population SD."""
    trace_path = os.path.join(C3_V1_1_1_DIR, "policy_action_trace.csv")
    assert os.path.exists(trace_path)
    trace_df = pd.read_csv(trace_path)

    case_counts = trace_df.groupby("case_id")["final_action"].apply(lambda x: (x == "FLAG_UNRELIABLE").sum())
    assert len(case_counts) == 105
    assert (case_counts > 0).all()

    assert np.isclose(case_counts.mean(), 100.085714, atol=1e-4)
    assert case_counts.median() == 98.0
    assert case_counts.min() == 67
    assert case_counts.max() == 151
    assert np.isclose(case_counts.std(ddof=1), 19.601371, atol=1e-4)
    assert np.isclose(case_counts.std(ddof=0), 19.507808, atol=1e-4)


def test_fallback_secondary_dice_means():
    """Verify exact mean Dice for fallback switches on GT+ slices."""
    trace_path = os.path.join(C3_V1_1_1_DIR, "policy_action_trace.csv")
    trace_df = pd.read_csv(trace_path)

    switches_gt = trace_df[(trace_df["final_action"] == "FALLBACK_SWITCH") & (trace_df["is_gt_positive"] == True)]
    assert len(switches_gt) == 54

    # Mean across all 54 GT+ switches
    assert np.isclose(switches_gt["secondary_dice"].mean(), 0.261239, atol=1e-4)

    # Mean across the 37 positive overlap recoveries
    pos_overlap = switches_gt[switches_gt["secondary_dice"] > 0]
    assert len(pos_overlap) == 37
    assert np.isclose(pos_overlap["secondary_dice"].mean(), 0.381268, atol=1e-4)


def test_mutually_exclusive_quality_tiers_sum():
    """Verify 4 mutually exclusive quality tiers sum to exactly 6,549 GT+ slices."""
    tiers_path = os.path.join(C3_V1_1_1_DIR, "segmentation_quality_tiers.csv")
    assert os.path.exists(tiers_path)
    df = pd.read_csv(tiers_path)

    assert df["total_count"].sum() == 6549
    assert df["retained_count"].sum() == 5491
    assert df["referred_count"].sum() == 1058

    high = df[df["segmentation_quality_tier"].str.contains("High Quality")].iloc[0]["total_count"]
    acc = df[df["segmentation_quality_tier"].str.contains("Acceptable")].iloc[0]["total_count"]
    poor = df[df["segmentation_quality_tier"].str.contains("Poor")].iloc[0]["total_count"]
    zero = df[df["segmentation_quality_tier"].str.contains("Zero Overlap")].iloc[0]["total_count"]

    assert high == 3665
    assert acc == 1552
    assert poor == 949
    assert zero == 383
    assert high + acc + poor + zero == 6549


def test_quarantine_disclosure_in_metadata():
    """Verify quarantine disclosure accurately states historical parse in Oct 2026 and unopened in C3."""
    meta_path = os.path.join(C3_V1_1_1_DIR, "c3_execution_metadata.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    q = meta["preflight_provenance"]["quarantine_test_isolation"]
    assert "October 2026" in q["historical_manifest_access"]
    assert "not opened or read" in q["current_c3_execution_access"]
    assert q["test_cases_accessed"] == 0
