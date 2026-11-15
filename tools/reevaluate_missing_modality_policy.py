"""Re-evaluation of Missing-Modality Robustness under Full Gate 6A Policy vs Entropy-Only Ablation.

Evaluates the 105 validation cases (96 unique subjects, 16,275 slices) under C0-C4:
- C0: Baseline (4 modalities: T1n, T1c, T2w, T2f)
- C1: Withhold T1n (channel 0 = 0.0)
- C2: Withhold T1c / Contrast (channel 1 = 0.0)
- C3: Withhold T2w (channel 2 = 0.0)
- C4: Withhold T2-FLAIR (channel 3 = 0.0)

Evaluates three distinct policies:
1. Analysis A: Original Entropy-Only Threshold (simplified ablation; empty pred -> entropy=0.0 -> retained)
2. Analysis B1: Gate 6A Standalone Policy (ResUNet-2D with empty-prediction guard, T*=2.880783, tau=0.35)
3. Analysis B2: Full Gate 6A Agent Policy (ResUNet-2D champion + UNetResNet34 fallback + axial context, tau=0.35)

Outputs stored in:
results/final_research_baseline/
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F

from src.agent.policy import extract_slice_prediction, evaluate_slice_policy
from src.agent.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.data.dataset import discover_valid_cases, _load_and_normalize_case
from src.models.resunet import ResUNet2D
from src.models.unet_resnet34 import UNetResNet34
from src.uncertainty.entropy import softmax_entropy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("modality_reeval")

T_RESUNET = 2.8807828426361084
T_RESNET34 = 2.6266958713531494
TAU_ABSTAIN = 0.35


def compute_dice_binary(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute binary Dice coefficient."""
    pred_b = pred > 0
    target_b = target > 0
    intersection = np.logical_and(pred_b, target_b).sum()
    total = pred_b.sum() + target_b.sum()
    if total == 0:
        return 1.0 if intersection == 0 else 0.0
    return float((2.0 * intersection) / total)


def compute_axial_context(
    primary_preds: dict[int, SlicePrediction],
    z: int,
    total_slices: int = 155,
    k: int = 2,
) -> AxialContextSummary:
    """Extract local axial neighbor statistics within +/- k slices."""
    evaluated = 0
    unavailable = 0
    fg_count = 0
    empty_count = 0
    for delta in range(-k, k + 1):
        if delta == 0:
            continue
        n_idx = z + delta
        if 0 <= n_idx < total_slices:
            if n_idx in primary_preds:
                evaluated += 1
                if primary_preds[n_idx].has_empty_foreground:
                    empty_count += 1
                else:
                    fg_count += 1
            else:
                unavailable += 1
        else:
            unavailable += 1
    return AxialContextSummary(
        evaluated_neighbors_count=evaluated,
        unavailable_neighbors_count=unavailable,
        foreground_neighbors_count=fg_count,
        empty_neighbors_count=empty_count,
    )


def cluster_bootstrap_mean_ci(
    values: list[float],
    cluster_ids: list[str],
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Calculate cluster-bootstrap 95% confidence interval at subject level."""
    df = pd.DataFrame({"val": values, "cluster": cluster_ids}).dropna()
    if len(df) == 0:
        return float("nan"), float("nan"), float("nan")

    clusters = df["cluster"].unique()
    n_clusters = len(clusters)
    cluster_map = {c: df[df["cluster"] == c]["val"].values for c in clusters}

    mean_val = float(df["val"].mean())
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_bootstraps):
        sampled_c = rng.choice(clusters, size=n_clusters, replace=True)
        boot_vals = np.concatenate([cluster_map[c] for c in sampled_c])
        boot_means.append(float(np.mean(boot_vals)))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_lower, ci_upper


def main() -> None:
    output_dir = PROJECT_ROOT / "results" / "final_research_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load evaluation split (105 cases, 96 unique subjects)
    splits_path = PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_splits.json"
    with open(splits_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    eval_case_ids = splits["evaluation_set"]["case_ids"]
    logger.info("Loaded %d evaluation case IDs from %s", len(eval_case_ids), splits_path)

    archive_dir = PROJECT_ROOT / "data" / "extracted_archive"
    all_cases = discover_valid_cases(archive_dir)
    case_map = {c.case_id: c for c in all_cases}
    eval_cases = [case_map[cid] for cid in eval_case_ids if cid in case_map]
    logger.info("Discovered %d matching evaluation case files", len(eval_cases))
    assert len(eval_cases) == 105, f"Expected 105 evaluation cases, found {len(eval_cases)}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    # 2. Load Checkpoints
    resunet_path = PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt"
    m_resunet = ResUNet2D(in_channels=4, num_classes=4).to(device)
    ckpt_r = torch.load(resunet_path, map_location=device, weights_only=False)
    m_resunet.load_state_dict(ckpt_r["model_state_dict"] if "model_state_dict" in ckpt_r else ckpt_r)
    m_resunet.eval()

    resnet34_path = PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt"
    m_resnet34 = UNetResNet34(in_channels=4, num_classes=4).to(device)
    ckpt_34 = torch.load(resnet34_path, map_location=device, weights_only=False)
    m_resnet34.load_state_dict(ckpt_34["model_state_dict"] if "model_state_dict" in ckpt_34 else ckpt_34)
    m_resnet34.eval()

    logger.info("Successfully loaded frozen ResUNet-2D and UNetResNet34 models")

    # 3. Policy Configurations
    cfg_gate6a_standalone = PolicyConfig(
        enable_empty_prediction_fallback=False,
        enable_axial_context=False,
        primary_temperature=T_RESUNET,
        enable_uncertainty_fallback=False,
        high_uncertainty_threshold=TAU_ABSTAIN,
    )

    cfg_gate6a_agent = PolicyConfig(
        enable_empty_prediction_fallback=True,
        fallback_priority=("unet_resnet34",),
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
        primary_temperature=T_RESUNET,
        candidate_temperatures={"unet_resnet34": T_RESNET34},
        enable_uncertainty_fallback=False,
        high_uncertainty_threshold=TAU_ABSTAIN,
    )

    conditions = [
        ("C0_baseline", None, "Full 4-Modality mpMRI (Baseline)"),
        ("C1_drop_t1n", 0, "Withhold T1n (Channel 0)"),
        ("C2_drop_t1c", 1, "Withhold T1c / Contrast (Channel 1)"),
        ("C3_drop_t2w", 2, "Withhold T2w (Channel 2)"),
        ("C4_drop_t2f", 3, "Withhold T2-FLAIR (Channel 3)"),
    ]

    all_case_results: list[dict[str, Any]] = []
    all_slice_records: list[dict[str, Any]] = []

    t_start = time.time()
    batch_size = 16

    for idx, case_file in enumerate(eval_cases):
        cid = case_file.case_id
        sid = cid.rsplit("-", 1)[0]
        if (idx + 1) % 10 == 0 or idx == 0:
            logger.info("[%d/105] Evaluating case %s...", idx + 1, cid)

        cached = _load_and_normalize_case(case_file)
        img_raw = cached.image  # [4, H, W, Z]
        gt_vol = cached.mask   # [H, W, Z]
        n_slices = img_raw.shape[3]

        for cond_id, drop_idx, desc in conditions:
            img_cond = img_raw.copy()
            if drop_idx is not None:
                img_cond[drop_idx, :, :, :] = 0.0

            # Store slice-level predictions
            preds_resunet: dict[int, SlicePrediction] = {}
            preds_resnet34: dict[int, SlicePrediction] = {}
            pred_vol_resunet = np.zeros(img_cond.shape[1:], dtype=np.int64)
            pred_vol_agent = np.zeros(img_cond.shape[1:], dtype=np.int64)

            # Slices evaluation for entropy-only (Analysis A)
            entropy_only_retained = []
            entropy_only_entropy = []

            with torch.no_grad():
                for z_start in range(0, n_slices, batch_size):
                    z_end = min(z_start + batch_size, n_slices)
                    batch_np = np.transpose(img_cond[:, :, :, z_start:z_end], (3, 0, 1, 2))
                    batch_t = torch.from_numpy(batch_np).to(device)

                    logits1 = m_resunet(batch_t)
                    logits2 = m_resnet34(batch_t)

                    scaled_l1 = logits1 / T_RESUNET
                    scaled_l2 = logits2 / T_RESNET34

                    ent1_map = softmax_entropy(scaled_l1)

                    for b_i in range(z_end - z_start):
                        z_curr = z_start + b_i
                        l1_slice = scaled_l1[b_i : b_i + 1]
                        l2_slice = scaled_l2[b_i : b_i + 1]

                        sp1 = extract_slice_prediction("resunet_2d", l1_slice, num_classes=4)
                        sp2 = extract_slice_prediction("unet_resnet34", l2_slice, num_classes=4)

                        preds_resunet[z_curr] = sp1
                        preds_resnet34[z_curr] = sp2
                        pred_vol_resunet[:, :, z_curr] = sp1.pred_mask.cpu().numpy()

                        # Analysis A (Simplified Entropy-only)
                        # If pred_fg_pixels == 0 -> mean_ent = 0.0 -> is_retained = True
                        if sp1.pred_foreground_pixels > 0:
                            ent_val = float(sp1.mean_foreground_entropy)
                        else:
                            ent_val = 0.0
                        ret_a = ent_val <= TAU_ABSTAIN
                        entropy_only_retained.append(ret_a)
                        entropy_only_entropy.append(ent_val)

            # Volumetric Ground Truth
            gt_wt = np.isin(gt_vol, [1, 2, 3])
            gt_tc = np.isin(gt_vol, [1, 3])
            gt_et = gt_vol == 3

            # Volumetric Compartment Dice for Champion ResUNet-2D
            resunet_wt = np.isin(pred_vol_resunet, [1, 2, 3])
            resunet_tc = np.isin(pred_vol_resunet, [1, 3])
            resunet_et = pred_vol_resunet == 3

            wt_dice_resunet = compute_dice_binary(resunet_wt, gt_wt)
            tc_dice_resunet = compute_dice_binary(resunet_tc, gt_tc)
            et_dice_resunet = compute_dice_binary(resunet_et, gt_et)
            mean_dice_resunet = (wt_dice_resunet + tc_dice_resunet + et_dice_resunet) / 3.0

            # Pass 2: Evaluate Policy on each slice
            case_fg_slices = 0
            # Counters for Analysis A (Entropy only)
            a_fg_retained = 0
            a_fg_flagged = 0
            a_all_retained = 0
            a_all_flagged = 0
            a_acc_dice_list = []
            a_rej_dice_list = []

            # Counters for Analysis B1 (Gate 6A Standalone ResUNet)
            b1_fg_retained = 0
            b1_fg_flagged = 0
            b1_all_retained = 0
            b1_all_flagged = 0
            b1_empty_pred_fg = 0
            b1_empty_pred_all = 0
            b1_acc_dice_list = []
            b1_rej_dice_list = []

            # Counters for Analysis B2 (Gate 6A Agent with UNetResNet34 Fallback)
            b2_fg_retained = 0
            b2_fg_flagged = 0
            b2_all_retained = 0
            b2_all_flagged = 0
            b2_fallback_switches_fg = 0
            b2_acc_dice_list = []
            b2_rej_dice_list = []

            for z in range(n_slices):
                sp1 = preds_resunet[z]
                sp2 = preds_resnet34[z]
                gt_slice = gt_vol[:, :, z]
                has_gt_fg = bool((gt_slice > 0).any())
                if has_gt_fg:
                    case_fg_slices += 1

                # Slice Dice for ResUNet
                slice_pred_mask = sp1.pred_mask.cpu().numpy()
                sd = compute_dice_binary(slice_pred_mask, gt_slice)

                # 1. Analysis A Evaluation
                ret_a = entropy_only_retained[z]
                if ret_a:
                    a_all_retained += 1
                    if has_gt_fg:
                        a_fg_retained += 1
                        a_acc_dice_list.append(sd)
                else:
                    a_all_flagged += 1
                    if has_gt_fg:
                        a_fg_flagged += 1
                        a_rej_dice_list.append(sd)

                # 2. Analysis B1 Evaluation (Gate 6A Standalone)
                # Standalone state has no candidate predictions
                st_standalone = AgentState(
                    case_id=cid,
                    slice_index=z,
                    primary_prediction=sp1,
                    candidate_predictions={},
                    axial_context=None,
                )
                dec_b1 = evaluate_slice_policy(st_standalone, cfg_gate6a_standalone)
                ret_b1 = not dec_b1.is_human_review_recommended  # True if ACCEPT, False if FLAG_UNRELIABLE
                if sp1.has_empty_foreground:
                    b1_empty_pred_all += 1
                    if has_gt_fg:
                        b1_empty_pred_fg += 1

                if ret_b1:
                    b1_all_retained += 1
                    if has_gt_fg:
                        b1_fg_retained += 1
                        b1_acc_dice_list.append(sd)
                else:
                    b1_all_flagged += 1
                    if has_gt_fg:
                        b1_fg_flagged += 1
                        b1_rej_dice_list.append(sd)

                # 3. Analysis B2 Evaluation (Gate 6A Agent with Fallback)
                ctx = compute_axial_context(preds_resunet, z, total_slices=n_slices, k=2)
                st_agent = AgentState(
                    case_id=cid,
                    slice_index=z,
                    primary_prediction=sp1,
                    candidate_predictions={"unet_resnet34": sp2},
                    axial_context=ctx,
                )
                dec_b2 = evaluate_slice_policy(st_agent, cfg_gate6a_agent)
                pred_vol_agent[:, :, z] = dec_b2.final_mask.cpu().numpy()

                ret_b2 = not dec_b2.is_human_review_recommended
                if dec_b2.action == AgentAction.FALLBACK_SWITCH and has_gt_fg:
                    b2_fallback_switches_fg += 1

                agent_sd = compute_dice_binary(dec_b2.final_mask.cpu().numpy(), gt_slice)
                if ret_b2:
                    b2_all_retained += 1
                    if has_gt_fg:
                        b2_fg_retained += 1
                        b2_acc_dice_list.append(agent_sd)
                else:
                    b2_all_flagged += 1
                    if has_gt_fg:
                        b2_fg_flagged += 1
                        b2_rej_dice_list.append(agent_sd)

            # Volumetric metrics for Agent
            agent_wt = np.isin(pred_vol_agent, [1, 2, 3])
            agent_tc = np.isin(pred_vol_agent, [1, 3])
            agent_et = pred_vol_agent == 3
            wt_dice_agent = compute_dice_binary(agent_wt, gt_wt)
            tc_dice_agent = compute_dice_binary(agent_tc, gt_tc)
            et_dice_agent = compute_dice_binary(agent_et, gt_et)
            mean_dice_agent = (wt_dice_agent + tc_dice_agent + et_dice_agent) / 3.0

            case_record = {
                "case_id": cid,
                "subject_id": sid,
                "condition": cond_id,
                "condition_desc": desc,
                "dropped_channel": drop_idx if drop_idx is not None else -1,
                "total_slices": n_slices,
                "total_fg_slices": case_fg_slices,
                # ResUNet Volumetric Segmentation Quality
                "resunet_wt_dice": wt_dice_resunet,
                "resunet_tc_dice": tc_dice_resunet,
                "resunet_et_dice": et_dice_resunet,
                "resunet_compartment_mean_dice": mean_dice_resunet,
                # Agent Volumetric Segmentation Quality
                "agent_wt_dice": wt_dice_agent,
                "agent_tc_dice": tc_dice_agent,
                "agent_et_dice": et_dice_agent,
                "agent_compartment_mean_dice": mean_dice_agent,
                # Analysis A: Simplified Entropy-Only Policy
                "a_fg_retained": a_fg_retained,
                "a_fg_flagged": a_fg_flagged,
                "a_fg_coverage_pct": (a_fg_retained / case_fg_slices * 100.0) if case_fg_slices > 0 else 100.0,
                "a_fg_referral_pct": (a_fg_flagged / case_fg_slices * 100.0) if case_fg_slices > 0 else 0.0,
                "a_all_retained": a_all_retained,
                "a_all_flagged": a_all_flagged,
                "a_all_coverage_pct": (a_all_retained / n_slices * 100.0),
                "a_all_referral_pct": (a_all_flagged / n_slices * 100.0),
                "a_mean_accepted_slice_dice": float(np.mean(a_acc_dice_list)) if a_acc_dice_list else float("nan"),
                "a_mean_rejected_slice_dice": float(np.mean(a_rej_dice_list)) if a_rej_dice_list else float("nan"),
                # Analysis B1: Gate 6A Standalone Policy (Empty Guard + tau=0.35)
                "b1_fg_retained": b1_fg_retained,
                "b1_fg_flagged": b1_fg_flagged,
                "b1_fg_coverage_pct": (b1_fg_retained / case_fg_slices * 100.0) if case_fg_slices > 0 else 100.0,
                "b1_fg_referral_pct": (b1_fg_flagged / case_fg_slices * 100.0) if case_fg_slices > 0 else 0.0,
                "b1_all_retained": b1_all_retained,
                "b1_all_flagged": b1_all_flagged,
                "b1_all_coverage_pct": (b1_all_retained / n_slices * 100.0),
                "b1_all_referral_pct": (b1_all_flagged / n_slices * 100.0),
                "b1_empty_pred_fg_count": b1_empty_pred_fg,
                "b1_empty_pred_all_count": b1_empty_pred_all,
                "b1_mean_accepted_slice_dice": float(np.mean(b1_acc_dice_list)) if b1_acc_dice_list else float("nan"),
                "b1_mean_rejected_slice_dice": float(np.mean(b1_rej_dice_list)) if b1_rej_dice_list else float("nan"),
                # Analysis B2: Full Gate 6A Multi-Model Agent Policy (Fallback + Context)
                "b2_fg_retained": b2_fg_retained,
                "b2_fg_flagged": b2_fg_flagged,
                "b2_fg_coverage_pct": (b2_fg_retained / case_fg_slices * 100.0) if case_fg_slices > 0 else 100.0,
                "b2_fg_referral_pct": (b2_fg_flagged / case_fg_slices * 100.0) if case_fg_slices > 0 else 0.0,
                "b2_all_retained": b2_all_retained,
                "b2_all_flagged": b2_all_flagged,
                "b2_all_coverage_pct": (b2_all_retained / n_slices * 100.0),
                "b2_all_referral_pct": (b2_all_flagged / n_slices * 100.0),
                "b2_fallback_switches_fg": b2_fallback_switches_fg,
                "b2_mean_accepted_slice_dice": float(np.mean(b2_acc_dice_list)) if b2_acc_dice_list else float("nan"),
                "b2_mean_rejected_slice_dice": float(np.mean(b2_rej_dice_list)) if b2_rej_dice_list else float("nan"),
            }
            all_case_results.append(case_record)

    total_time = time.time() - t_start
    logger.info("Evaluation completed in %.2f seconds (%.2f minutes)", total_time, total_time / 60.0)

    # 4. Save full case-level CSV
    case_df = pd.DataFrame(all_case_results)
    case_csv_path = output_dir / "reevaluated_modality_metrics.csv"
    case_df.to_csv(case_csv_path, index=False)
    logger.info("Saved case-level metrics to %s", case_csv_path)

    # 5. Subject-level Aggregation & Statistical Testing
    subject_rows = []
    unique_subjects = sorted(case_df["subject_id"].unique())
    logger.info("Aggregating across %d unique subjects...", len(unique_subjects))

    # Metric columns to average at subject level
    num_cols = [
        "resunet_wt_dice", "resunet_tc_dice", "resunet_et_dice", "resunet_compartment_mean_dice",
        "agent_wt_dice", "agent_tc_dice", "agent_et_dice", "agent_compartment_mean_dice",
        "a_fg_coverage_pct", "a_fg_referral_pct", "a_all_coverage_pct", "a_all_referral_pct",
        "a_mean_accepted_slice_dice", "a_mean_rejected_slice_dice",
        "b1_fg_coverage_pct", "b1_fg_referral_pct", "b1_all_coverage_pct", "b1_all_referral_pct",
        "b1_empty_pred_fg_count", "b1_empty_pred_all_count",
        "b1_mean_accepted_slice_dice", "b1_mean_rejected_slice_dice",
        "b2_fg_coverage_pct", "b2_fg_referral_pct", "b2_all_coverage_pct", "b2_all_referral_pct",
        "b2_fallback_switches_fg",
        "b2_mean_accepted_slice_dice", "b2_mean_rejected_slice_dice",
    ]

    for sid in unique_subjects:
        s_df = case_df[case_df["subject_id"] == sid]
        for cond_id, _, desc in conditions:
            sc_df = s_df[s_df["condition"] == cond_id]
            s_dict = {
                "subject_id": sid,
                "condition": cond_id,
                "condition_desc": desc,
                "cases_count": len(sc_df),
                "total_slices": int(sc_df["total_slices"].sum()),
                "total_fg_slices": int(sc_df["total_fg_slices"].sum()),
            }
            for col in num_cols:
                s_dict[col] = float(sc_df[col].mean())
            subject_rows.append(s_dict)

    subj_df = pd.DataFrame(subject_rows)
    subj_csv_path = output_dir / "subject_level_modality_stats.csv"
    subj_df.to_csv(subj_csv_path, index=False)
    logger.info("Saved subject-level metrics to %s", subj_csv_path)

    # 6. Aggregate Summary and Statistical Comparisons against C0
    summary_report: dict[str, Any] = {
        "metadata": {
            "evaluation_cohort": "Val_eval_remediation",
            "cases_evaluated": len(eval_cases),
            "unique_subjects": len(unique_subjects),
            "total_slices_per_condition": 16275,
            "total_fg_slices_cohort": int(case_df[case_df["condition"] == "C0_baseline"]["total_fg_slices"].sum()),
            "temperature_resunet": T_RESUNET,
            "temperature_resnet34": T_RESNET34,
            "threshold_tau": TAU_ABSTAIN,
            "elapsed_seconds": total_time,
        },
        "conditions": {},
    }

    # Reference C0 subject rows for paired tests
    c0_subj = subj_df[subj_df["condition"] == "C0_baseline"].sort_values("subject_id")

    for cond_id, _, desc in conditions:
        c_cases = case_df[case_df["condition"] == cond_id]
        c_subj = subj_df[subj_df["condition"] == cond_id].sort_values("subject_id")

        total_fg_cohort = int(c_cases["total_fg_slices"].sum())
        total_slices_cohort = int(c_cases["total_slices"].sum())

        # Analysis A Pooled
        a_fg_ret_pooled = int(c_cases["a_fg_retained"].sum())
        a_fg_flag_pooled = int(c_cases["a_fg_flagged"].sum())
        a_all_ret_pooled = int(c_cases["a_all_retained"].sum())
        a_all_flag_pooled = int(c_cases["a_all_flagged"].sum())

        # Analysis B1 Pooled
        b1_fg_ret_pooled = int(c_cases["b1_fg_retained"].sum())
        b1_fg_flag_pooled = int(c_cases["b1_fg_flagged"].sum())
        b1_all_ret_pooled = int(c_cases["b1_all_retained"].sum())
        b1_all_flag_pooled = int(c_cases["b1_all_flagged"].sum())
        b1_empty_fg_pooled = int(c_cases["b1_empty_pred_fg_count"].sum())
        b1_empty_all_pooled = int(c_cases["b1_empty_pred_all_count"].sum())

        # Analysis B2 Pooled
        b2_fg_ret_pooled = int(c_cases["b2_fg_retained"].sum())
        b2_fg_flag_pooled = int(c_cases["b2_fg_flagged"].sum())
        b2_all_ret_pooled = int(c_cases["b2_all_retained"].sum())
        b2_all_flag_pooled = int(c_cases["b2_all_flagged"].sum())
        b2_fallback_fg_pooled = int(c_cases["b2_fallback_switches_fg"].sum())

        cond_dict: dict[str, Any] = {
            "condition_desc": desc,
            # ResUNet Segmentation
            "resunet_wt_dice": {
                "case_mean": float(c_cases["resunet_wt_dice"].mean()),
                "subject_mean": float(c_subj["resunet_wt_dice"].mean()),
                "ci_95": cluster_bootstrap_mean_ci(c_cases["resunet_wt_dice"].tolist(), c_cases["subject_id"].tolist())[:3],
            },
            "resunet_tc_dice": {
                "case_mean": float(c_cases["resunet_tc_dice"].mean()),
                "subject_mean": float(c_subj["resunet_tc_dice"].mean()),
                "ci_95": cluster_bootstrap_mean_ci(c_cases["resunet_tc_dice"].tolist(), c_cases["subject_id"].tolist())[:3],
            },
            "resunet_et_dice": {
                "case_mean": float(c_cases["resunet_et_dice"].mean()),
                "subject_mean": float(c_subj["resunet_et_dice"].mean()),
                "ci_95": cluster_bootstrap_mean_ci(c_cases["resunet_et_dice"].tolist(), c_cases["subject_id"].tolist())[:3],
            },
            "resunet_compartment_mean_dice": {
                "case_mean": float(c_cases["resunet_compartment_mean_dice"].mean()),
                "subject_mean": float(c_subj["resunet_compartment_mean_dice"].mean()),
                "ci_95": cluster_bootstrap_mean_ci(c_cases["resunet_compartment_mean_dice"].tolist(), c_cases["subject_id"].tolist())[:3],
            },
            # Agent Segmentation
            "agent_compartment_mean_dice": {
                "case_mean": float(c_cases["agent_compartment_mean_dice"].mean()),
                "subject_mean": float(c_subj["agent_compartment_mean_dice"].mean()),
                "ci_95": cluster_bootstrap_mean_ci(c_cases["agent_compartment_mean_dice"].tolist(), c_cases["subject_id"].tolist())[:3],
            },
            # Analysis A (Entropy Only)
            "analysis_a_entropy_only": {
                "pooled_fg_coverage_pct": (a_fg_ret_pooled / total_fg_cohort * 100.0),
                "pooled_fg_referral_pct": (a_fg_flag_pooled / total_fg_cohort * 100.0),
                "numerator_retained_fg": a_fg_ret_pooled,
                "numerator_flagged_fg": a_fg_flag_pooled,
                "denominator_fg": total_fg_cohort,
                "subject_averaged_referral_pct": float(c_subj["a_fg_referral_pct"].mean()),
                "subject_averaged_coverage_pct": float(c_subj["a_fg_coverage_pct"].mean()),
                "all_slice_coverage_pct": (a_all_ret_pooled / total_slices_cohort * 100.0),
                "all_slice_referral_pct": (a_all_flag_pooled / total_slices_cohort * 100.0),
                "mean_accepted_slice_dice": float(c_cases["a_mean_accepted_slice_dice"].mean()),
                "mean_rejected_slice_dice": float(c_cases["a_mean_rejected_slice_dice"].mean()),
            },
            # Analysis B1 (Gate 6A Standalone Policy)
            "analysis_b1_gate6a_standalone": {
                "pooled_fg_coverage_pct": (b1_fg_ret_pooled / total_fg_cohort * 100.0),
                "pooled_fg_referral_pct": (b1_fg_flag_pooled / total_fg_cohort * 100.0),
                "numerator_retained_fg": b1_fg_ret_pooled,
                "numerator_flagged_fg": b1_fg_flag_pooled,
                "denominator_fg": total_fg_cohort,
                "empty_pred_fg_count": b1_empty_fg_pooled,
                "empty_pred_all_count": b1_empty_all_pooled,
                "subject_averaged_referral_pct": float(c_subj["b1_fg_referral_pct"].mean()),
                "subject_averaged_coverage_pct": float(c_subj["b1_fg_coverage_pct"].mean()),
                "all_slice_coverage_pct": (b1_all_ret_pooled / total_slices_cohort * 100.0),
                "all_slice_referral_pct": (b1_all_flag_pooled / total_slices_cohort * 100.0),
                "mean_accepted_slice_dice": float(c_cases["b1_mean_accepted_slice_dice"].mean()),
                "mean_rejected_slice_dice": float(c_cases["b1_mean_rejected_slice_dice"].mean()),
            },
            # Analysis B2 (Full Gate 6A Agent Policy)
            "analysis_b2_gate6a_agent": {
                "pooled_fg_coverage_pct": (b2_fg_ret_pooled / total_fg_cohort * 100.0),
                "pooled_fg_referral_pct": (b2_fg_flag_pooled / total_fg_cohort * 100.0),
                "numerator_retained_fg": b2_fg_ret_pooled,
                "numerator_flagged_fg": b2_fg_flag_pooled,
                "denominator_fg": total_fg_cohort,
                "fallback_switches_fg": b2_fallback_fg_pooled,
                "subject_averaged_referral_pct": float(c_subj["b2_fg_referral_pct"].mean()),
                "subject_averaged_coverage_pct": float(c_subj["b2_fg_coverage_pct"].mean()),
                "all_slice_coverage_pct": (b2_all_ret_pooled / total_slices_cohort * 100.0),
                "all_slice_referral_pct": (b2_all_flag_pooled / total_slices_cohort * 100.0),
                "mean_accepted_slice_dice": float(c_cases["b2_mean_accepted_slice_dice"].mean()),
                "mean_rejected_slice_dice": float(c_cases["b2_mean_rejected_slice_dice"].mean()),
            },
        }

        # Paired Wilcoxon Tests vs C0 (if not C0)
        if cond_id != "C0_baseline":
            # 1. ResUNet Compartment Mean Dice Wilcoxon
            w_dice, p_dice = stats.wilcoxon(c_subj["resunet_compartment_mean_dice"], c0_subj["resunet_compartment_mean_dice"])
            # 2. Analysis A Referral Wilcoxon
            w_ref_a, p_ref_a = stats.wilcoxon(c_subj["a_fg_referral_pct"], c0_subj["a_fg_referral_pct"])
            # 3. Analysis B1 Referral Wilcoxon
            w_ref_b1, p_ref_b1 = stats.wilcoxon(c_subj["b1_fg_referral_pct"], c0_subj["b1_fg_referral_pct"])
            # 4. Analysis B2 Referral Wilcoxon
            w_ref_b2, p_ref_b2 = stats.wilcoxon(c_subj["b2_fg_referral_pct"], c0_subj["b2_fg_referral_pct"])

            cond_dict["statistical_tests_vs_c0"] = {
                "dice": {
                    "wilcoxon_stat": float(w_dice),
                    "p_value_raw": float(p_dice),
                    "p_value_bonferroni": float(min(1.0, 4.0 * p_dice)),
                },
                "analysis_a_referral": {
                    "wilcoxon_stat": float(w_ref_a),
                    "p_value_raw": float(p_ref_a),
                    "p_value_bonferroni": float(min(1.0, 4.0 * p_ref_a)),
                },
                "analysis_b1_referral": {
                    "wilcoxon_stat": float(w_ref_b1),
                    "p_value_raw": float(p_ref_b1),
                    "p_value_bonferroni": float(min(1.0, 4.0 * p_ref_b1)),
                },
                "analysis_b2_referral": {
                    "wilcoxon_stat": float(w_ref_b2),
                    "p_value_raw": float(p_ref_b2),
                    "p_value_bonferroni": float(min(1.0, 4.0 * p_ref_b2)),
                },
            }

        summary_report["conditions"][cond_id] = cond_dict

    summary_json_path = output_dir / "modality_policy_comparison.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)
    logger.info("Saved summary report to %s", summary_json_path)

    # 7. Print Console Verification Table
    print("\n" + "=" * 90)
    print("RE-EVALUATED MISSING MODALITY COMPARISON: BASELINE VS MISSING MODALITIES")
    print("=" * 90)
    print(f"{'Cond':<5} | {'Desc':<25} | {'Mean Dice':<9} | {'A Ref %':<8} | {'B1 Ref %':<9} | {'B2 Ref %':<9} | {'Empty FG':<8}")
    print("-" * 90)
    for c_id, d in summary_report["conditions"].items():
        desc = d["condition_desc"][:25]
        mdice = d["resunet_compartment_mean_dice"]["case_mean"]
        a_ref = d["analysis_a_entropy_only"]["pooled_fg_referral_pct"]
        b1_ref = d["analysis_b1_gate6a_standalone"]["pooled_fg_referral_pct"]
        b2_ref = d["analysis_b2_gate6a_agent"]["pooled_fg_referral_pct"]
        empty_fg = d["analysis_b1_gate6a_standalone"]["empty_pred_fg_count"]
        print(f"{c_id:<5} | {desc:<25} | {mdice:<9.4f} | {a_ref:<8.2f}% | {b1_ref:<9.2f}% | {b2_ref:<9.2f}% | {empty_fg:<8d}")
    print("=" * 90)


if __name__ == "__main__":
    main()
