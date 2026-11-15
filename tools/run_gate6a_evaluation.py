"""Gate 6A: Full evaluation of 4 configurations across 105 Val_eval cases.

Optimized for memory efficiency, execution stability, and exact volumetric metrics:
  - 1 forward pass per model per slice in torch.inference_mode().
  - Policy decisions evaluated deterministically from cached slice predictions.
  - Periodic GPU cache cleanup per case volume.
  - Volumetric 3D patient-level Dice (WT, TC, ET, Mean Dice).
  - 95% Bootstrap Confidence Intervals (1,000 resamples across 105 cases).
  - ECE (15 bins), MCE, Brier score, and NLL.
  - Strict quarantine audit against test_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.models import ModelRegistry
from src.agent.policy import extract_slice_prediction, evaluate_slice_policy
from src.agent.state import AgentAction, AgentDecision, AgentState, AxialContextSummary, PolicyConfig, SlicePrediction
from src.data.dataset import (
    CaseFiles,
    MODALITIES,
    SEGMENTATION_SUFFIX,
    _load_and_normalize_case,
)
from src.evaluation.metrics import _nanmean
from src.uncertainty.calibration import (
    compute_multiclass_brier_score,
    compute_multiclass_ece,
    compute_nll,
    compute_risk_coverage_curve,
)


def compute_bootstrap_ci(
    values: Sequence[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval (percentile method)."""
    val_arr = np.array([v for v in values if not np.isnan(v)], dtype=np.float64)
    if len(val_arr) == 0:
        return float("nan"), float("nan"), float("nan")

    mean_val = float(np.mean(val_arr))
    if len(val_arr) == 1:
        return mean_val, mean_val, mean_val

    rng = np.random.RandomState(seed)
    n = len(val_arr)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(val_arr[indices])

    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_val, lower, upper


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


def compute_axial_context(
    primary_preds: dict[int, SlicePrediction],
    z: int,
    total_slices: int = 155,
    k: int = 2,
) -> AxialContextSummary:
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
    return AxialContextSummary(
        evaluated_neighbors_count=evaluated,
        unavailable_neighbors_count=unavailable,
        foreground_neighbors_count=fg_count,
        empty_neighbors_count=empty_count,
    )


class VolumetricMetricAccumulator:
    """Accumulates 3D voxel confusion counts across all 155 slices of a single patient case."""

    def __init__(self) -> None:
        self.wt_tp = 0
        self.wt_fp = 0
        self.wt_fn = 0

        self.tc_tp = 0
        self.tc_fp = 0
        self.tc_fn = 0

        self.et_tp = 0
        self.et_fp = 0
        self.et_fn = 0

    def update_slice(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        p = pred_mask.detach().cpu().long()
        t = target_mask.detach().cpu().long()

        p_wt = p > 0
        t_wt = t > 0
        self.wt_tp += int((p_wt & t_wt).sum().item())
        self.wt_fp += int((p_wt & ~t_wt).sum().item())
        self.wt_fn += int((~p_wt & t_wt).sum().item())

        p_tc = (p == 1) | (p == 3)
        t_tc = (t == 1) | (t == 3)
        self.tc_tp += int((p_tc & t_tc).sum().item())
        self.tc_fp += int((p_tc & ~t_tc).sum().item())
        self.tc_fn += int((~p_tc & t_tc).sum().item())

        p_et = p == 3
        t_et = t == 3
        self.et_tp += int((p_et & t_et).sum().item())
        self.et_fp += int((p_et & ~t_et).sum().item())
        self.et_fn += int((~p_et & t_et).sum().item())

    def compute_case_dice(self) -> dict[str, float]:
        def _dice(tp: int, fp: int, fn: int) -> float:
            denom = 2 * tp + fp + fn
            if denom > 0:
                return (2.0 * tp) / denom
            return 1.0 if (tp + fp + fn) == 0 else 0.0

        wt_d = _dice(self.wt_tp, self.wt_fp, self.wt_fn)
        tc_d = _dice(self.tc_tp, self.tc_fp, self.tc_fn)
        et_d = _dice(self.et_tp, self.et_fp, self.et_fn)

        valid_dices = [d for d in (wt_d, tc_d, et_d) if not np.isnan(d)]
        mean_d = float(np.mean(valid_dices)) if valid_dices else float("nan")

        return {
            "wt_dice": wt_d,
            "tc_dice": tc_d,
            "et_dice": et_d,
            "mean_dice": mean_d,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate 6A evaluation on Val_eval")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index in eval cases (0-based)")
    parser.add_argument("--end-idx", type=int, default=105, help="End index in eval cases (exclusive)")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip inference and aggregate from case_cache")
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Gate 6A: Full Reliability-Aware Evaluation on Val_eval (105 cases) ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Range: [{args.start_idx}:{args.end_idx}] | Aggregate-only: {args.aggregate_only}")

    out_dir = PROJECT_ROOT / "results" / "gate6a_reliability_calibration"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    archive_dir = PROJECT_ROOT / "data" / "extracted_archive"
    case_cache_dir = out_dir / "case_cache"
    case_cache_dir.mkdir(parents=True, exist_ok=True)
    eval_voxels_file = out_dir / "eval_sampled_voxels.pt"

    # 1. Load calibration parameters and splits
    param_file = out_dir / "calibration_parameters.json"
    if not param_file.exists():
        raise FileNotFoundError(f"Calibration parameters file not found: {param_file}. Run calibration first.")

    with open(param_file, "r", encoding="utf-8") as f:
        cal_params = json.load(f)

    splits_file = out_dir / "calibration_splits.json"
    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    eval_case_ids: list[str] = splits_data["evaluation_set"]["case_ids"]
    assert len(eval_case_ids) == 105, f"Expected 105 Val_eval cases, got {len(eval_case_ids)}"

    # Strict quarantine audit
    test_manifest_path = PROJECT_ROOT / "results" / "dataset_splits" / "test_manifest.json"
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_manifest = json.load(f)
    test_case_ids = {c["case_id"] for c in test_manifest["cases"]}
    leaked = set(eval_case_ids).intersection(test_case_ids)
    assert len(leaked) == 0, f"FATAL: Quarantined test cases in evaluation set: {leaked}"
    print(f"Quarantine audit: Verified 0 test cases in Val_eval (105 cases).")

    # Fitted parameters
    t_resunet = cal_params["fitted_temperatures"]["resunet_2d"]["optimal_temperature_T"]
    t_resnet34 = cal_params["fitted_temperatures"]["unet_resnet34"]["optimal_temperature_T"]
    frozen_cfg = cal_params["frozen_policy_configuration"]
    tau_abstain = frozen_cfg["recommended_high_uncertainty_threshold"]
    tau_fallback = frozen_cfg["recommended_uncertainty_fallback_threshold"]

    print(f"\nLoaded Calibration Parameters:")
    print(f"  ResUNet-2D Temperature:          T* = {t_resunet:.4f}")
    print(f"  UNetResNet34 Temperature:        T* = {t_resnet34:.4f}")
    print(f"  High Uncertainty Threshold:      tau = {tau_abstain:.2f}")
    print(f"  Uncertainty Fallback Threshold:  tau = {tau_fallback:.2f}")

    # 2. Load models into registry
    print("\nLoading models into ModelRegistry...")
    torch.backends.cudnn.benchmark = False
    registry = ModelRegistry(default_device=device)
    m_resunet = registry.get_model("resunet", device=device)
    m_resnet34 = registry.get_model("unet_resnet34", device=device)
    print("Models loaded successfully.")

    # 3. Policy Configurations
    cfg_agent_baseline = PolicyConfig(
        enable_empty_prediction_fallback=True,
        fallback_priority=("unet_resnet34",),
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
        enable_uncertainty_fallback=False,
        high_uncertainty_threshold=None,
    )
    cfg_agent_calibrated = PolicyConfig(
        enable_empty_prediction_fallback=True,
        fallback_priority=("unet_resnet34",),
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
        primary_temperature=t_resunet,
        candidate_temperatures={"unet_resnet34": t_resnet34},
        enable_uncertainty_fallback=False,
        high_uncertainty_threshold=tau_abstain,
    )
    cfg_agent_calibrated_fb = PolicyConfig(
        enable_empty_prediction_fallback=True,
        fallback_priority=("unet_resnet34",),
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
        primary_temperature=t_resunet,
        candidate_temperatures={"unet_resnet34": t_resnet34},
        enable_uncertainty_fallback=True,
        uncertainty_fallback_threshold=tau_fallback,
        high_uncertainty_threshold=tau_abstain,
    )

    evaluated_configs = [
        "resunet_uncal",
        "resunet_cal",
        "resnet34_uncal",
        "resnet34_cal",
        "agent_baseline",
        "agent_calibrated",
        "agent_calibrated_fb",
    ]

    case_results: dict[str, list[dict[str, Any]]] = {c: [] for c in evaluated_configs}
    slice_records: list[dict[str, Any]] = []
    latencies_per_case: dict[str, list[float]] = {c: [] for c in evaluated_configs}

    # Calibration voxel collection for Val_eval (CPU)
    eval_sampled_l1: list[torch.Tensor] = []
    eval_sampled_t: list[torch.Tensor] = []
    max_eval_sample_voxels = 500000
    if eval_voxels_file.exists():
        total_sampled_eval_voxels = max_eval_sample_voxels
    else:
        total_sampled_eval_voxels = 0

    rng = np.random.RandomState(42)
    total_slices_count = 0

    print(f"\nEvaluating cases [{args.start_idx}:{args.end_idx}] of {len(eval_case_ids)} Val_eval cases...")

    if not args.aggregate_only:
        for case_idx in range(args.start_idx + 1, min(args.end_idx + 1, len(eval_case_ids) + 1)):
            cid = eval_case_ids[case_idx - 1]
            cache_path = case_cache_dir / f"{cid}.json"
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    case_payload = json.load(f)
                m_res = case_payload["case_results"]["resunet_cal"]["mean_dice"]
                m_agt = case_payload["case_results"]["agent_calibrated"]["mean_dice"]
                print(f"  [{case_idx:3d}/{len(eval_case_ids)}] [CACHED] {cid} | ResUNet={m_res:.4f}, Agent={m_agt:.4f}", flush=True)
                continue

            t_case_start = time.perf_counter()
            cfiles = resolve_single_case(cid, raw_dir, archive_dir)
            vol = _load_and_normalize_case(cfiles)
            total_z = vol.image.shape[3]  # 155

            case_accumulators = {c: VolumetricMetricAccumulator() for c in evaluated_configs}
            case_slice_records: list[dict[str, Any]] = []

            p1_uncal_dict: dict[int, SlicePrediction] = {}
            p1_cal_dict: dict[int, SlicePrediction] = {}
            p2_uncal_dict: dict[int, SlicePrediction] = {}
            p2_cal_dict: dict[int, SlicePrediction] = {}
            case_targets: dict[int, torch.Tensor] = {}

            # Pass 1: Run each model exactly ONCE per slice
            t_models_start = time.perf_counter()
            with torch.inference_mode():
                for z in range(total_z):
                    total_slices_count += 1
                    img_slice = np.ascontiguousarray(vol.image[:, :, :, z])
                    img = torch.from_numpy(img_slice).float().unsqueeze(0).to(device)
                    tgt = torch.from_numpy(vol.mask[:, :, z].astype(np.int64))
                    case_targets[z] = tgt

                    # ResUNet-2D (1 forward pass)
                    l1 = m_resunet(img).cpu()
                    p1_uncal = extract_slice_prediction("resunet", l1)
                    p1_cal = extract_slice_prediction("resunet", l1 / t_resunet)
                    p1_uncal_dict[z] = p1_uncal
                    p1_cal_dict[z] = p1_cal

                    # UNetResNet34 (1 forward pass)
                    l2 = m_resnet34(img).cpu()
                    p2_uncal = extract_slice_prediction("unet_resnet34", l2)
                    p2_cal = extract_slice_prediction("unet_resnet34", l2 / t_resnet34)
                    p2_uncal_dict[z] = p2_uncal
                    p2_cal_dict[z] = p2_cal

                    # Voxel subsampling for ECE (strictly on CPU)
                    if total_sampled_eval_voxels < max_eval_sample_voxels and (z % 4 == 0):
                        tgt_np = tgt.numpy()
                        brain_mask = (vol.image[:, :, :, z].sum(axis=0) != 0)
                        fg_idx = np.where(tgt_np > 0)
                        n_fg = len(fg_idx[0])
                        if n_fg > 0:
                            l1_cpu = l1.squeeze(0)
                            l1_fg = l1_cpu[:, fg_idx[0], fg_idx[1]].permute(1, 0)
                            t_fg = tgt[fg_idx[0], fg_idx[1]]
                            bg_idx = np.where((tgt_np == 0) & brain_mask)
                            n_bg = len(bg_idx[0])
                            s_bg = min(n_bg, n_fg)
                            if s_bg > 0:
                                chosen_bg = rng.choice(n_bg, size=s_bg, replace=False)
                                bg_r, bg_c = bg_idx[0][chosen_bg], bg_idx[1][chosen_bg]
                                l1_bg = l1_cpu[:, bg_r, bg_c].permute(1, 0)
                                t_bg = tgt[bg_r, bg_c]

                                cat_l = torch.cat([l1_fg, l1_bg], dim=0)
                                cat_t = torch.cat([t_fg, t_bg], dim=0)
                            else:
                                cat_l = l1_fg
                                cat_t = t_fg

                            eval_sampled_l1.append(cat_l)
                            eval_sampled_t.append(cat_t)
                            total_sampled_eval_voxels += cat_l.shape[0]

            model_time_ms = (time.perf_counter() - t_models_start) * 1000.0
            slice_lat_est = model_time_ms / total_z / 2.0  # Approx latency per single model

            # Pass 2: Evaluate policy decisions and accumulate volumetric metrics
            with torch.inference_mode():
                for z in range(total_z):
                    tgt = case_targets[z]
                    p1_u = p1_uncal_dict[z]
                    p1_c = p1_cal_dict[z]
                    p2_u = p2_uncal_dict[z]
                    p2_c = p2_cal_dict[z]

                    # Standalone models
                    case_accumulators["resunet_uncal"].update_slice(p1_u.pred_mask, tgt)
                    case_accumulators["resunet_cal"].update_slice(p1_c.pred_mask, tgt)
                    case_accumulators["resnet34_uncal"].update_slice(p2_u.pred_mask, tgt)
                    case_accumulators["resnet34_cal"].update_slice(p2_c.pred_mask, tgt)

                    # Context summaries
                    ctx_uncal = compute_axial_context(p1_uncal_dict, z, total_z, 2)
                    ctx_cal = compute_axial_context(p1_cal_dict, z, total_z, 2)

                    # Agent Baseline
                    st_base = AgentState(cid, z, p1_u, {"unet_resnet34": p2_u}, ctx_uncal)
                    dec_base = evaluate_slice_policy(st_base, cfg_agent_baseline)
                    case_accumulators["agent_baseline"].update_slice(dec_base.final_mask, tgt)

                    # Agent Calibrated (Gate 6A primary)
                    st_cal = AgentState(cid, z, p1_c, {"unet_resnet34": p2_c}, ctx_cal)
                    dec_cal = evaluate_slice_policy(st_cal, cfg_agent_calibrated)
                    case_accumulators["agent_calibrated"].update_slice(dec_cal.final_mask, tgt)

                    # Agent Calibrated with Fallback
                    dec_cal_fb = evaluate_slice_policy(st_cal, cfg_agent_calibrated_fb)
                    case_accumulators["agent_calibrated_fb"].update_slice(dec_cal_fb.final_mask, tgt)

                    # Record slice metrics
                    gt_fg = int((tgt > 0).sum().item())
                    case_slice_records.append({
                        "case_id": cid,
                        "slice_index": z,
                        "has_gt_fg": gt_fg > 0,
                        "gt_fg_pixels": gt_fg,
                        "primary_fg_pixels": p1_c.pred_foreground_pixels,
                        "primary_entropy": p1_c.mean_foreground_entropy,
                        "agent_base_action": dec_base.action.value,
                        "agent_base_model": dec_base.selected_model,
                        "agent_cal_action": dec_cal.action.value,
                        "agent_cal_model": dec_cal.selected_model,
                        "agent_cal_review_flagged": dec_cal.is_human_review_recommended,
                        "agent_cal_fb_action": dec_cal_fb.action.value,
                        "agent_cal_fb_model": dec_cal_fb.selected_model,
                    })

            # Record case metrics and save to case cache
            case_total_time = time.perf_counter() - t_case_start
            computed_case_dices = {cfg: case_accumulators[cfg].compute_case_dice() for cfg in evaluated_configs}
            case_payload = {
                "case_id": cid,
                "case_results": computed_case_dices,
                "slice_records": case_slice_records,
                "slice_lat_est": slice_lat_est,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(case_payload, f)

            # Release case data and clear CUDA cache
            del vol, p1_uncal_dict, p1_cal_dict, p2_uncal_dict, p2_cal_dict, case_targets
            torch.cuda.empty_cache()

            m_res = computed_case_dices["resunet_cal"]["mean_dice"]
            m_agt = computed_case_dices["agent_calibrated"]["mean_dice"]
            print(f"  [{case_idx:3d}/{len(eval_case_ids)}] Completed {cid} in {case_total_time:.2f}s | ResUNet={m_res:.4f}, Agent={m_agt:.4f}", flush=True)

        # Save sampled voxels if freshly gathered
        if not eval_voxels_file.exists() and len(eval_sampled_l1) > 0:
            torch.save({
                "l1": torch.cat(eval_sampled_l1, dim=0),
                "t": torch.cat(eval_sampled_t, dim=0),
            }, eval_voxels_file)
            print(f"Saved {total_sampled_eval_voxels} calibration evaluation voxels to {eval_voxels_file}", flush=True)

    # Check overall cache coverage
    cached_ids = set()
    for cfile in case_cache_dir.glob("*.json"):
        cached_ids.add(cfile.stem)
    print(f"\n[Case Cache Status] {len(cached_ids)} of {len(eval_case_ids)} evaluation cases are cached on disk.", flush=True)

    if len(cached_ids) < len(eval_case_ids):
        print(f"Evaluation chunk completed. Run remaining cases to complete full evaluation aggregation.", flush=True)
        return

    # 4. Statistical Analysis: Aggregate all 105 cached cases
    print("\nAll 105 evaluation cases cached! Aggregating metrics across all cases...")
    for cid in eval_case_ids:
        cache_path = case_cache_dir / f"{cid}.json"
        with open(cache_path, "r", encoding="utf-8") as f:
            cpayload = json.load(f)
        for cfg in evaluated_configs:
            m_dict = dict(cpayload["case_results"][cfg])
            m_dict["case_id"] = cid
            case_results[cfg].append(m_dict)
            latencies_per_case[cfg].append(cpayload["slice_lat_est"])
        slice_records.extend(cpayload["slice_records"])

    print("Computing patient-level 95% bootstrap confidence intervals (1,000 resamples)...")
    statistical_summary: dict[str, Any] = {}

    for cfg in evaluated_configs:
        wt_list = [r["wt_dice"] for r in case_results[cfg]]
        tc_list = [r["tc_dice"] for r in case_results[cfg]]
        et_list = [r["et_dice"] for r in case_results[cfg]]
        mean_list = [r["mean_dice"] for r in case_results[cfg]]

        wt_m, wt_l, wt_u = compute_bootstrap_ci(wt_list)
        tc_m, tc_l, tc_u = compute_bootstrap_ci(tc_list)
        et_m, et_l, et_u = compute_bootstrap_ci(et_list)
        mean_m, mean_l, mean_u = compute_bootstrap_ci(mean_list)

        lats = latencies_per_case[cfg]
        statistical_summary[cfg] = {
            "patient_count": len(eval_case_ids),
            "mean_dice": {"mean": mean_m, "ci_95_lower": mean_l, "ci_95_upper": mean_u},
            "wt_dice": {"mean": wt_m, "ci_95_lower": wt_l, "ci_95_upper": wt_u},
            "tc_dice": {"mean": tc_m, "ci_95_lower": tc_l, "ci_95_upper": tc_u},
            "et_dice": {"mean": et_m, "ci_95_lower": et_l, "ci_95_upper": et_u},
            "latency_ms_per_slice_approx": float(np.mean(lats)),
        }

    # 5. Calibration Evaluation on Val_eval Sampled Voxels
    print("\nComputing calibration metrics on Val_eval...")
    if eval_voxels_file.exists():
        saved_vox = torch.load(eval_voxels_file, map_location="cpu")
        all_eval_l1 = saved_vox["l1"]
        all_eval_t = saved_vox["t"]
    elif len(eval_sampled_l1) > 0:
        all_eval_l1 = torch.cat(eval_sampled_l1, dim=0)
        all_eval_t = torch.cat(eval_sampled_t, dim=0)
        torch.save({"l1": all_eval_l1, "t": all_eval_t}, eval_voxels_file)
    else:
        raise RuntimeError("No calibration evaluation voxels found. Run inference on at least one chunk first.")
    targets_np = all_eval_t.cpu().numpy()

    probs_uncal = F.softmax(all_eval_l1, dim=-1).cpu().numpy()
    probs_cal = F.softmax(all_eval_l1 / t_resunet, dim=-1).cpu().numpy()

    ece_eval_uncal = compute_multiclass_ece(probs_uncal, targets_np, n_bins=15)
    ece_eval_cal = compute_multiclass_ece(probs_cal, targets_np, n_bins=15)
    brier_eval_uncal = compute_multiclass_brier_score(probs_uncal, targets_np)
    brier_eval_cal = compute_multiclass_brier_score(probs_cal, targets_np)
    nll_eval_uncal = compute_nll(probs_uncal, targets_np)
    nll_eval_cal = compute_nll(probs_cal, targets_np)

    calibration_eval_summary = {
        "val_eval_voxel_count": all_eval_l1.shape[0],
        "resunet_uncalibrated": {
            "temperature": 1.0,
            "ece": ece_eval_uncal["ece"],
            "mce": ece_eval_uncal["mce"],
            "brier_score": brier_eval_uncal,
            "nll": nll_eval_uncal,
            "bin_accuracies": ece_eval_uncal["bin_accuracies"],
            "bin_confidences": ece_eval_uncal["bin_confidences"],
            "bin_counts": ece_eval_uncal["bin_counts"],
            "bin_edges": ece_eval_uncal["bin_edges"],
        },
        "resunet_calibrated": {
            "temperature": t_resunet,
            "ece": ece_eval_cal["ece"],
            "mce": ece_eval_cal["mce"],
            "brier_score": brier_eval_cal,
            "nll": nll_eval_cal,
            "bin_accuracies": ece_eval_cal["bin_accuracies"],
            "bin_confidences": ece_eval_cal["bin_confidences"],
            "bin_counts": ece_eval_cal["bin_counts"],
            "bin_edges": ece_eval_cal["bin_edges"],
        },
    }

    # 6. Agent Routing Telemetry Summary
    print("\nAnalyzing agent decision distribution on Val_eval...")
    agent_telemetry: dict[str, Any] = {}
    total_slices_eval = len(slice_records)

    for agent_key, action_col, model_col in [
        ("agent_baseline", "agent_base_action", "agent_base_model"),
        ("agent_calibrated", "agent_cal_action", "agent_cal_model"),
        ("agent_calibrated_fb", "agent_cal_fb_action", "agent_cal_fb_model"),
    ]:
        accept_count = sum(1 for s in slice_records if s[action_col] == AgentAction.ACCEPT.value)
        fallback_count = sum(1 for s in slice_records if s[action_col] == AgentAction.FALLBACK_SWITCH.value)
        unreliable_count = sum(1 for s in slice_records if s[action_col] == AgentAction.FLAG_UNRELIABLE.value)

        model_counts: dict[str, int] = {}
        for s in slice_records:
            m = s[model_col]
            if m:
                model_counts[m] = model_counts.get(m, 0) + 1

        agent_telemetry[agent_key] = {
            "total_slices": total_slices_eval,
            "accepted_slices": accept_count,
            "accepted_percent": (accept_count / total_slices_eval) * 100.0,
            "fallback_switched_slices": fallback_count,
            "fallback_switched_percent": (fallback_count / total_slices_eval) * 100.0,
            "flagged_unreliable_slices": unreliable_count,
            "flagged_unreliable_percent": (unreliable_count / total_slices_eval) * 100.0,
            "model_selection_counts": model_counts,
        }

    # 7. Print Comparative Results Table
    print("\n" + "=" * 80)
    print("GATE 6A EVALUATION RESULTS ON VAL_EVAL (105 CASES)")
    print("=" * 80)
    header = f"{'Configuration':<22} | {'Mean Dice (95% CI)':<26} | {'WT Dice':<10} | {'TC Dice':<10} | {'ET Dice':<10}"
    print(header)
    print("-" * 80)
    for cfg in evaluated_configs:
        s = statistical_summary[cfg]
        m = s["mean_dice"]
        mean_str = f"{m['mean']:.4f} [{m['ci_95_lower']:.4f}, {m['ci_95_upper']:.4f}]"
        wt_str = f"{s['wt_dice']['mean']:.4f}"
        tc_str = f"{s['tc_dice']['mean']:.4f}"
        et_str = f"{s['et_dice']['mean']:.4f}"
        print(f"{cfg:<22} | {mean_str:<26} | {wt_str:<10} | {tc_str:<10} | {et_str:<10}")
    print("=" * 80)

    print(f"\nCalibration Results on Val_eval (ECE, Brier, NLL):")
    print(f"  Uncalibrated ResUNet-2D (T=1.0000): ECE={ece_eval_uncal['ece']:.4f}, Brier={brier_eval_uncal:.4f}, NLL={nll_eval_uncal:.4f}")
    print(f"  Calibrated ResUNet-2D   (T={t_resunet:.4f}): ECE={ece_eval_cal['ece']:.4f}, Brier={brier_eval_cal:.4f}, NLL={nll_eval_cal:.4f}")

    print(f"\nAgent Telemetry:")
    for akey, tinfo in agent_telemetry.items():
        print(f"  {akey:<20}: Accept={tinfo['accepted_percent']:5.1f}% | Fallback={tinfo['fallback_switched_percent']:4.1f}% ({tinfo['fallback_switched_slices']} slices) | Flagged={tinfo['flagged_unreliable_percent']:4.1f}% ({tinfo['flagged_unreliable_slices']} slices)")

    # 8. Save deliverables
    summary_file = out_dir / "evaluation_summary.json"
    full_output = {
        "metadata": {
            "protocol": "Gate 6A Full Reliability-Aware Evaluation on Val_eval",
            "date": "2026-10-29",
            "evaluation_cases_count": len(eval_case_ids),
            "total_slices_evaluated": total_slices_eval,
            "disclosure": (
                "Val_eval (105 cases) is an internal evaluation subset of the 155-case validation partition, "
                "strictly distinct from the 50-case Val_cal set used for temperature fitting. "
                "It is NOT an independent final test set. The 155 test cases in results/dataset_splits/test_manifest.json "
                "remain 100% quarantined and unaccessed."
            ),
        },
        "statistical_summary": statistical_summary,
        "calibration_eval_summary": calibration_eval_summary,
        "agent_telemetry": agent_telemetry,
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    case_csv_file = out_dir / "case_level_metrics.csv"
    with open(case_csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["config", "case_id", "mean_dice", "wt_dice", "tc_dice", "et_dice"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cfg in evaluated_configs:
            for r in case_results[cfg]:
                writer.writerow({
                    "config": cfg,
                    "case_id": r["case_id"],
                    "mean_dice": r["mean_dice"],
                    "wt_dice": r["wt_dice"],
                    "tc_dice": r["tc_dice"],
                    "et_dice": r["et_dice"],
                })

    slice_csv_file = out_dir / "evaluation_slice_results.csv"
    with open(slice_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(slice_records[0].keys()))
        writer.writeheader()
        writer.writerows(slice_records)

    total_time_min = (time.perf_counter() - t_start) / 60.0
    print(f"\n=== Gate 6A Evaluation Completed Successfully ===", flush=True)
    print(f"Total evaluation time: {total_time_min:.2f} minutes", flush=True)
    print(f"Summary JSON saved to: {summary_file}", flush=True)
    print(f"Case metrics CSV saved to: {case_csv_file}", flush=True)
    print(f"Slice records CSV saved to: {slice_csv_file}", flush=True)

    # 9. Auto-generate plots
    try:
        from tools.plot_gate6a_figures import main as generate_figures
        print("\nGenerating figures...", flush=True)
        generate_figures()
        print("Figures generated successfully.", flush=True)
    except Exception as exc:
        print(f"Warning: Figure generation failed: {exc}", flush=True)


if __name__ == "__main__":
    main()
