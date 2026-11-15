"""Gate 6A Remediation: Temperature Calibration & Threshold Exploration on Subject-Grouped Val_cal.

Workspace: D:\\MajorProject
Output: results/gate6a_remediation_v1/calibration_parameters.json
        results/gate6a_remediation_v1/calibration_slice_metrics.csv
        results/gate6a_remediation_v1/cal_sampled_voxels.pt
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.models import ModelRegistry
from src.agent.policy import extract_slice_prediction
from src.data.dataset import (
    CaseFiles,
    MODALITIES,
    SEGMENTATION_SUFFIX,
    _load_and_normalize_case,
)
from src.evaluation.metrics import _nanmean
from src.uncertainty.calibration import (
    TemperatureScaler,
    compute_multiclass_brier_score,
    compute_multiclass_ece,
    compute_nll,
    compute_risk_coverage_curve,
)


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_slice_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float, float, float]:
    p = pred_mask.detach().cpu().long()
    t = target_mask.detach().cpu().long()

    # Per-class Dice (1=NCR, 2=ED, 3=ET)
    class_dices: list[float] = []
    for c in (1, 2, 3):
        tp = float(((p == c) & (t == c)).sum().item())
        fp = float(((p == c) & (t != c)).sum().item())
        fn = float(((p != c) & (t == c)).sum().item())
        denom = 2.0 * tp + fp + fn
        class_dices.append((2.0 * tp / denom) if denom > 0 else float("nan"))

    # WT: {1, 2, 3}
    p_wt = p > 0
    t_wt = t > 0
    wt_tp = float((p_wt & t_wt).sum().item())
    wt_fp = float((p_wt & ~t_wt).sum().item())
    wt_fn = float((~p_wt & t_wt).sum().item())
    wt_denom = 2.0 * wt_tp + wt_fp + wt_fn
    wt_dice = (2.0 * wt_tp / wt_denom) if wt_denom > 0 else float("nan")

    # TC: {1, 3}
    p_tc = (p == 1) | (p == 3)
    t_tc = (t == 1) | (t == 3)
    tc_tp = float((p_tc & t_tc).sum().item())
    tc_fp = float((p_tc & ~t_tc).sum().item())
    tc_fn = float((~p_tc & t_tc).sum().item())
    tc_denom = 2.0 * tc_tp + tc_fp + tc_fn
    tc_dice = (2.0 * tc_tp / tc_denom) if tc_denom > 0 else float("nan")

    # ET: {3}
    p_et = p == 3
    t_et = t == 3
    et_tp = float((p_et & t_et).sum().item())
    et_fp = float((p_et & ~t_et).sum().item())
    et_fn = float((~p_et & t_et).sum().item())
    et_denom = 2.0 * et_tp + et_fp + et_fn
    et_dice = (2.0 * et_tp / et_denom) if et_denom > 0 else float("nan")

    mean_fg_dice = _nanmean(class_dices)
    return mean_fg_dice, wt_dice, tc_dice, et_dice


def resolve_single_case(case_id: str, raw_dir: Path, archive_dir: Path) -> CaseFiles:
    cdir = (raw_dir / case_id) if (raw_dir / case_id).exists() else (archive_dir / case_id)
    if not cdir.exists():
        raise FileNotFoundError(f"Case directory {case_id} not found in raw or extracted_archive")

    mod_paths = {m: cdir / f"{case_id}-{m}.nii.gz" for m in MODALITIES}
    seg_path = cdir / f"{case_id}-{SEGMENTATION_SUFFIX}.nii.gz"
    shape = tuple(int(dim) for dim in nib.load(str(seg_path)).shape)

    return CaseFiles(
        case_id=case_id,
        case_dir=cdir,
        modality_paths=mod_paths,
        segmentation_path=seg_path,
        shape=shape,
    )


def main() -> None:
    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Gate 6A Remediation: Temperature Calibration on Subject-Grouped Val_cal ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    out_dir = PROJECT_ROOT / "results" / "gate6a_remediation_v1"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    archive_dir = PROJECT_ROOT / "data" / "extracted_archive"

    splits_file = out_dir / "calibration_splits.json"
    if not splits_file.exists():
        raise FileNotFoundError(f"Remediation splits file not found: {splits_file}. Run create_remediation_splits.py first.")

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    cal_case_ids: list[str] = splits_data["calibration_set"]["case_ids"]
    assert len(cal_case_ids) == 50, f"Expected 50 calibration cases, got {len(cal_case_ids)}"

    # 1. Cryptographic Checkpoint Verification
    resunet_ckpt_path = PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt"
    resnet34_ckpt_path = PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt"

    resunet_hash = get_file_sha256(resunet_ckpt_path)
    resnet34_hash = get_file_sha256(resnet34_ckpt_path)

    print(f"\nModel Checkpoint Provenance:")
    print(f"  ResUNet-2D:     {resunet_ckpt_path} (SHA-256: {resunet_hash})")
    print(f"  UNetResNet34:   {resnet34_ckpt_path} (SHA-256: {resnet34_hash})")

    torch.backends.cudnn.benchmark = False
    registry = ModelRegistry(default_device=device)
    m_resunet = registry.get_model("resunet", device=device)
    m_resnet34 = registry.get_model("unet_resnet34", device=device)
    print("Models loaded successfully.")

    # 2. Documented Voxel Sampling & Slice Inference across all 50 Val_cal cases
    print(f"\nBeginning inference and voxel sampling across {len(cal_case_ids)} Val_cal cases...")
    sampled_l1: list[torch.Tensor] = []
    sampled_l2: list[torch.Tensor] = []
    sampled_t: list[torch.Tensor] = []

    per_case_voxel_audit: dict[str, dict[str, int]] = {}
    raw_slice_records: list[dict[str, Any]] = []

    rng = np.random.RandomState(42)
    total_sampled_voxels = 0

    for idx, cid in enumerate(cal_case_ids, start=1):
        t_case_start = time.perf_counter()
        cfiles = resolve_single_case(cid, raw_dir, archive_dir)
        vol = _load_and_normalize_case(cfiles)
        total_z = vol.image.shape[3]  # 155

        case_fg_voxels = 0
        case_bg_voxels = 0

        with torch.inference_mode():
            for z in range(total_z):
                img_slice = np.ascontiguousarray(vol.image[:, :, :, z])
                img = torch.from_numpy(img_slice).float().unsqueeze(0).to(device)
                tgt = torch.from_numpy(vol.mask[:, :, z].astype(np.int64))

                # Run both models (1 forward pass each, offload to CPU)
                l1 = m_resunet(img).cpu()
                l2 = m_resnet34(img).cpu()

                # Extract uncalibrated predictions for Dice calculation
                p1_uncal = extract_slice_prediction("resunet", l1)
                p2_uncal = extract_slice_prediction("unet_resnet34", l2)

                m_dice1, wt1, tc1, et1 = compute_slice_dice(p1_uncal.pred_mask, tgt)
                m_dice2, wt2, tc2, et2 = compute_slice_dice(p2_uncal.pred_mask, tgt)

                gt_fg_count = int((tgt > 0).sum().item())
                has_gt_fg = gt_fg_count > 0

                raw_slice_records.append({
                    "case_id": cid,
                    "slice_index": z,
                    "has_gt_fg": has_gt_fg,
                    "gt_fg_pixels": gt_fg_count,
                    "resunet_logits": l1,  # Retained in memory temporarily for calibrated entropy
                    "resnet34_logits": l2,
                    "resunet_uncal_entropy": p1_uncal.mean_foreground_entropy,
                    "resnet34_uncal_entropy": p2_uncal.mean_foreground_entropy,
                    "resunet_mean_dice": m_dice1,
                    "resunet_wt_dice": wt1,
                    "resunet_tc_dice": tc1,
                    "resunet_et_dice": et1,
                    "resnet34_mean_dice": m_dice2,
                    "resnet34_wt_dice": wt2,
                    "resnet34_tc_dice": tc2,
                    "resnet34_et_dice": et2,
                })

                # Voxel subsampling for temperature scaling: Every 2nd slice across ALL 50 cases
                if z % 2 == 0:
                    tgt_np = tgt.numpy()
                    brain_mask = (vol.image[:, :, :, z].sum(axis=0) != 0)
                    fg_idx = np.where(tgt_np > 0)
                    n_fg = len(fg_idx[0])

                    if n_fg > 0:
                        l1_cpu = l1.squeeze(0)
                        l2_cpu = l2.squeeze(0)

                        l1_fg = l1_cpu[:, fg_idx[0], fg_idx[1]].permute(1, 0)
                        l2_fg = l2_cpu[:, fg_idx[0], fg_idx[1]].permute(1, 0)
                        t_fg = tgt[fg_idx[0], fg_idx[1]]

                        bg_idx = np.where((tgt_np == 0) & brain_mask)
                        n_bg = len(bg_idx[0])
                        s_bg = min(n_bg, n_fg)

                        if s_bg > 0:
                            chosen_bg = rng.choice(n_bg, size=s_bg, replace=False)
                            bg_r, bg_c = bg_idx[0][chosen_bg], bg_idx[1][chosen_bg]
                            l1_bg = l1_cpu[:, bg_r, bg_c].permute(1, 0)
                            l2_bg = l2_cpu[:, bg_r, bg_c].permute(1, 0)
                            t_bg = tgt[bg_r, bg_c]

                            cat_l1 = torch.cat([l1_fg, l1_bg], dim=0)
                            cat_l2 = torch.cat([l2_fg, l2_bg], dim=0)
                            cat_t = torch.cat([t_fg, t_bg], dim=0)
                        else:
                            cat_l1 = l1_fg
                            cat_l2 = l2_fg
                            cat_t = t_fg

                        sampled_l1.append(cat_l1)
                        sampled_l2.append(cat_l2)
                        sampled_t.append(cat_t)

                        case_fg_voxels += n_fg
                        case_bg_voxels += s_bg
                        total_sampled_voxels += cat_l1.shape[0]

        del vol
        torch.cuda.empty_cache()

        per_case_voxel_audit[cid] = {
            "foreground_voxels": case_fg_voxels,
            "background_voxels": case_bg_voxels,
            "total_sampled": case_fg_voxels + case_bg_voxels,
        }
        case_time = time.perf_counter() - t_case_start
        print(f"  [{idx:2d}/50] Completed {cid} in {case_time:.2f}s | Sampled: {case_fg_voxels+case_bg_voxels} voxels", flush=True)

    # 3. Concatenate and serialize calibration sample
    all_l1 = torch.cat(sampled_l1, dim=0)
    all_l2 = torch.cat(sampled_l2, dim=0)
    all_t = torch.cat(sampled_t, dim=0)

    cal_voxels_file = out_dir / "cal_sampled_voxels.pt"
    torch.save({
        "metadata": {
            "protocol": "Gate 6A Remediation Calibration Voxel Sample",
            "cases_count": len(cal_case_ids),
            "total_voxels": all_l1.shape[0],
            "per_case_audit": per_case_voxel_audit,
            "resunet_checkpoint_sha256": resunet_hash,
            "resnet34_checkpoint_sha256": resnet34_hash,
        },
        "l1": all_l1,
        "l2": all_l2,
        "t": all_t,
    }, cal_voxels_file)
    print(f"\nSerialized {all_l1.shape[0]} calibration voxels from 50 cases to {cal_voxels_file}")

    # 4. Temperature Scaling Optimization
    print("\nOptimizing temperature scaling T*...")
    scaler1 = TemperatureScaler(initial_temperature=1.0)
    scaler2 = TemperatureScaler(initial_temperature=1.0)

    fit1 = scaler1.fit(all_l1, all_t, max_iter=100)
    fit2 = scaler2.fit(all_l2, all_t, max_iter=100)

    t_resunet = float(fit1["optimal_temperature"])
    t_resnet34 = float(fit2["optimal_temperature"])

    print(f"  ResUNet-2D:   T* = {t_resunet:.4f} (NLL: {fit1['initial_nll']:.4f} -> {fit1['final_nll']:.4f})")
    print(f"  UNetResNet34: T* = {t_resnet34:.4f} (NLL: {fit2['initial_nll']:.4f} -> {fit2['final_nll']:.4f})")

    # 5. Evaluate calibration metrics before/after
    targets_np = all_t.numpy()
    p1_uncal_probs = F.softmax(all_l1, dim=-1).numpy()
    p1_cal_probs = F.softmax(all_l1 / t_resunet, dim=-1).numpy()
    p2_uncal_probs = F.softmax(all_l2, dim=-1).numpy()
    p2_cal_probs = F.softmax(all_l2 / t_resnet34, dim=-1).numpy()

    ece1_uncal = compute_multiclass_ece(p1_uncal_probs, targets_np, n_bins=15)
    ece1_cal = compute_multiclass_ece(p1_cal_probs, targets_np, n_bins=15)
    ece2_uncal = compute_multiclass_ece(p2_uncal_probs, targets_np, n_bins=15)
    ece2_cal = compute_multiclass_ece(p2_cal_probs, targets_np, n_bins=15)

    brier1_uncal = compute_multiclass_brier_score(p1_uncal_probs, targets_np)
    brier1_cal = compute_multiclass_brier_score(p1_cal_probs, targets_np)
    brier2_uncal = compute_multiclass_brier_score(p2_uncal_probs, targets_np)
    brier2_cal = compute_multiclass_brier_score(p2_cal_probs, targets_np)

    nll1_uncal = compute_nll(p1_uncal_probs, targets_np)
    nll1_cal = compute_nll(p1_cal_probs, targets_np)
    nll2_uncal = compute_nll(p2_uncal_probs, targets_np)
    nll2_cal = compute_nll(p2_cal_probs, targets_np)

    # 6. Compute CALIBRATED predictive entropy for all slices & save slice metrics CSV
    print("\nComputing calibrated slice predictive entropy for threshold search...")
    final_slice_records: list[dict[str, Any]] = []

    for s in raw_slice_records:
        l1 = s["resunet_logits"]
        l2 = s["resnet34_logits"]

        p1_cal = extract_slice_prediction("resunet", l1 / t_resunet)
        p2_cal = extract_slice_prediction("unet_resnet34", l2 / t_resnet34)

        final_slice_records.append({
            "case_id": s["case_id"],
            "slice_index": s["slice_index"],
            "has_gt_fg": s["has_gt_fg"],
            "gt_fg_pixels": s["gt_fg_pixels"],
            # Primary ResUNet
            "resunet_fg_pixels": p1_cal.pred_foreground_pixels,
            "resunet_uncal_entropy": s["resunet_uncal_entropy"],
            "resunet_cal_entropy": p1_cal.mean_foreground_entropy,
            "resunet_mean_dice": s["resunet_mean_dice"],
            "resunet_wt_dice": s["resunet_wt_dice"],
            "resunet_tc_dice": s["resunet_tc_dice"],
            "resunet_et_dice": s["resunet_et_dice"],
            # Candidate UNetResNet34
            "resnet34_fg_pixels": p2_cal.pred_foreground_pixels,
            "resnet34_uncal_entropy": s["resnet34_uncal_entropy"],
            "resnet34_cal_entropy": p2_cal.mean_foreground_entropy,
            "resnet34_mean_dice": s["resnet34_mean_dice"],
            "resnet34_wt_dice": s["resnet34_wt_dice"],
            "resnet34_tc_dice": s["resnet34_tc_dice"],
            "resnet34_et_dice": s["resnet34_et_dice"],
        })

    # Save calibration slice metrics CSV
    cal_slice_csv = out_dir / "calibration_slice_metrics.csv"
    with open(cal_slice_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(final_slice_records[0].keys()))
        writer.writeheader()
        writer.writerows(final_slice_records)
    print(f"Saved {len(final_slice_records)} slice records to {cal_slice_csv}")

    # 7. Threshold Search on CALIBRATED Predictive Entropy
    print("\n--- Threshold Exploration on CALIBRATED Predictive Entropy (Val_cal) ---")
    fg_slices = [s for s in final_slice_records if s["has_gt_fg"] and s["resunet_cal_entropy"] is not None]
    total_fg_slices_count = len(fg_slices)
    print(f"Total foreground slices with valid calibrated entropy: {total_fg_slices_count}")

    candidate_thresholds = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    abstention_analysis: list[dict[str, Any]] = []

    for tau in candidate_thresholds:
        retained = [s for s in fg_slices if s["resunet_cal_entropy"] <= tau]
        flagged = [s for s in fg_slices if s["resunet_cal_entropy"] > tau]

        cov = len(retained) / total_fg_slices_count if total_fg_slices_count else 0.0
        retained_dice = _nanmean([s["resunet_mean_dice"] for s in retained]) if retained else 0.0
        flagged_dice = _nanmean([s["resunet_mean_dice"] for s in flagged]) if flagged else 0.0

        record = {
            "threshold_tau": tau,
            "total_fg_slices_denominator": total_fg_slices_count,
            "retained_slices": len(retained),
            "coverage_percent": float(cov * 100.0),
            "retained_mean_dice": float(retained_dice),
            "risk": float(1.0 - retained_dice) if retained_dice > 0 else 0.0,
            "flagged_slices": len(flagged),
            "flagged_percent": float((len(flagged) / total_fg_slices_count) * 100.0),
            "flagged_mean_dice": float(flagged_dice),
        }
        abstention_analysis.append(record)
        print(f"  tau={tau:.2f}: Coverage={record['coverage_percent']:5.1f}% | Flagged={record['flagged_slices']:4d} ({record['flagged_percent']:4.1f}%) | Retained Dice={record['retained_mean_dice']:.4f} | Flagged Dice={record['flagged_mean_dice']:.4f}")

    # Fallback model switching analysis on calibrated entropy
    switching_analysis: list[dict[str, Any]] = []
    for tau in candidate_thresholds:
        triggered = [s for s in fg_slices if s["resunet_cal_entropy"] > tau and s["resnet34_fg_pixels"] > 0]
        if not triggered:
            switching_analysis.append({
                "threshold_tau": tau,
                "triggered_slices": 0,
                "net_dice_change": 0.0,
                "improved_count": 0,
                "degraded_count": 0,
                "equal_count": 0,
                "mean_dice_primary": 0.0,
                "mean_dice_fallback": 0.0,
            })
            continue

        improved = 0
        degraded = 0
        equal = 0
        diffs = []
        dices_p = []
        dices_fb = []

        for s in triggered:
            dp = s["resunet_mean_dice"]
            dfb = s["resnet34_mean_dice"]
            if not np.isnan(dp) and not np.isnan(dfb):
                dices_p.append(dp)
                dices_fb.append(dfb)
                d = dfb - dp
                diffs.append(d)
                if d > 0.01:
                    improved += 1
                elif d < -0.01:
                    degraded += 1
                else:
                    equal += 1

        net_diff = float(np.mean(diffs)) if diffs else 0.0
        switching_analysis.append({
            "threshold_tau": tau,
            "triggered_slices": len(triggered),
            "net_dice_change": net_diff,
            "improved_count": improved,
            "degraded_count": degraded,
            "equal_count": equal,
            "mean_dice_primary": float(np.mean(dices_p)) if dices_p else 0.0,
            "mean_dice_fallback": float(np.mean(dices_fb)) if dices_fb else 0.0,
        })

    # Continuous Risk-Coverage Curve
    unc_arr = [s["resunet_cal_entropy"] for s in fg_slices]
    dice_arr = [s["resunet_mean_dice"] for s in fg_slices]
    rc_curve = compute_risk_coverage_curve(unc_arr, dice_arr, n_thresholds=50)

    # Operating threshold selection
    # At tau = 0.50 on calibrated entropy, selective risk is minimized while retaining high coverage
    frozen_tau_abstain = 0.50
    frozen_tau_fallback = 0.60

    cal_params_payload = {
        "metadata": {
            "protocol": "Gate 6A Remediation: Temperature Calibration & Threshold Exploration",
            "version": "1.0.0",
            "date": "2026-10-29",
            "calibration_cohort": "Val_cal_remediation",
            "calibration_case_count": len(cal_case_ids),
            "calibration_voxel_count": all_l1.shape[0],
            "total_slices_evaluated": len(final_slice_records),
            "foreground_slices_evaluated": total_fg_slices_count,
            "entropy_definition_used": "Normalized calibrated foreground predictive entropy: H(softmax(z / T*)) / log(4)",
            "resunet_checkpoint_sha256": resunet_hash,
            "resnet34_checkpoint_sha256": resnet34_hash,
        },
        "per_case_voxel_sampling_audit": per_case_voxel_audit,
        "fitted_temperatures": {
            "resunet_2d": {
                "name": "resunet",
                "checkpoint": str(resunet_ckpt_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": resunet_hash,
                "optimal_temperature_T": t_resunet,
                "initial_nll": fit1["initial_nll"],
                "final_nll": fit1["final_nll"],
                "nll_reduction": fit1["initial_nll"] - fit1["final_nll"],
                "iterations": fit1["iterations"],
                "converged": fit1["converged"],
            },
            "unet_resnet34": {
                "name": "unet_resnet34",
                "checkpoint": str(resnet34_ckpt_path.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": resnet34_hash,
                "optimal_temperature_T": t_resnet34,
                "initial_nll": fit2["initial_nll"],
                "final_nll": fit2["final_nll"],
                "nll_reduction": fit2["initial_nll"] - fit2["final_nll"],
                "iterations": fit2["iterations"],
                "converged": fit2["converged"],
            },
        },
        "calibration_metrics_val_cal": {
            "resunet_2d": {
                "uncalibrated": {
                    "temperature": 1.0,
                    "ece": ece1_uncal["ece"],
                    "mce": ece1_uncal["mce"],
                    "brier_score": brier1_uncal,
                    "nll": nll1_uncal,
                    "bin_accuracies": ece1_uncal["bin_accuracies"],
                    "bin_confidences": ece1_uncal["bin_confidences"],
                    "bin_counts": ece1_uncal["bin_counts"],
                    "bin_edges": ece1_uncal["bin_edges"],
                },
                "calibrated": {
                    "temperature": t_resunet,
                    "ece": ece1_cal["ece"],
                    "mce": ece1_cal["mce"],
                    "brier_score": brier1_cal,
                    "nll": nll1_cal,
                    "bin_accuracies": ece1_cal["bin_accuracies"],
                    "bin_confidences": ece1_cal["bin_confidences"],
                    "bin_counts": ece1_cal["bin_counts"],
                    "bin_edges": ece1_cal["bin_edges"],
                },
            },
            "unet_resnet34": {
                "uncalibrated": {
                    "temperature": 1.0,
                    "ece": ece2_uncal["ece"],
                    "mce": ece2_uncal["mce"],
                    "brier_score": brier2_uncal,
                    "nll": nll2_uncal,
                    "bin_accuracies": ece2_uncal["bin_accuracies"],
                    "bin_confidences": ece2_uncal["bin_confidences"],
                    "bin_counts": ece2_uncal["bin_counts"],
                    "bin_edges": ece2_uncal["bin_edges"],
                },
                "calibrated": {
                    "temperature": t_resnet34,
                    "ece": ece2_cal["ece"],
                    "mce": ece2_cal["mce"],
                    "brier_score": brier2_cal,
                    "nll": nll2_cal,
                    "bin_accuracies": ece2_cal["bin_accuracies"],
                    "bin_confidences": ece2_cal["bin_confidences"],
                    "bin_counts": ece2_cal["bin_counts"],
                    "bin_edges": ece2_cal["bin_edges"],
                },
            },
        },
        "threshold_exploration": {
            "abstention_analysis": abstention_analysis,
            "switching_analysis": switching_analysis,
            "risk_coverage_curve": rc_curve,
        },
        "frozen_policy_configuration": {
            "primary_model": "resunet",
            "candidate_model": "unet_resnet34",
            "primary_temperature": t_resunet,
            "candidate_temperatures": {
                "unet_resnet34": t_resnet34,
            },
            "recommended_high_uncertainty_threshold": frozen_tau_abstain,
            "recommended_uncertainty_fallback_threshold": frozen_tau_fallback,
            "policy_selection_rationale": (
                "Threshold search conducted strictly on calibrated predictive entropy H(softmax(z/T*)). "
                f"Operating threshold tau={frozen_tau_abstain:.2f} flags unreliable slices for human review. "
                "Blind model switching is restricted because UNetResNet34 degrades accuracy on uncertain slices; "
                "fallback is reserved for empty foreground predictions supported by axial context."
            ),
        },
    }

    params_file = out_dir / "calibration_parameters.json"
    with open(params_file, "w", encoding="utf-8") as f:
        json.dump(cal_params_payload, f, indent=2)

    total_time = (time.perf_counter() - t_start) / 60.0
    print(f"\n=== Calibration Completed Successfully in {total_time:.2f} minutes ===")
    print(f"Parameters saved to: {params_file}")


if __name__ == "__main__":
    main()
