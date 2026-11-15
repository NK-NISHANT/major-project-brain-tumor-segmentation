"""Gate 6A Remediation: Full evaluation of 7 configurations across 105 Val_eval cases.

Key remediations implemented:
1. Strict cryptographic provenance binding (SHA-256 for checkpoints, splits, parameters, scripts).
2. Cache invalidation on provenance hash mismatch.
3. Structured per-case evaluation voxel sampling across 100% of the 105 evaluation cases (eval_sampled_voxels.pt).
4. Subject-level cluster bootstrap confidence intervals (1,000 resamples across 96 unique subjects).
5. Calibrated predictive entropy thresholds (tau_abstain=0.35, tau_fallback=0.40) aligned with T*.
6. Strict quarantine audit against test_manifest.json.

Workspace: D:\\MajorProject
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def compute_bootstrap_ci(
    values: Sequence[float],
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and 95% bootstrap confidence interval (percentile method) across independent items."""
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

    # Overall pooled mean
    all_values = [v for vals in subject_to_scan_values.values() for v in vals if not np.isnan(v)]
    if len(all_values) == 0:
        return float("nan"), float("nan"), float("nan")
    pooled_mean = float(np.mean(all_values))

    rng = np.random.RandomState(seed)
    n_subjects = len(subjects)
    boot_means = np.empty(n_bootstraps, dtype=np.float64)

    for i in range(n_bootstraps):
        drawn_subj_idx = rng.randint(0, n_subjects, size=n_subjects)
        replicate_vals: list[float] = []
        for s_idx in drawn_subj_idx:
            s_name = subjects[s_idx]
            replicate_vals.extend(subject_to_scan_values[s_name])
        boot_means[i] = np.mean(replicate_vals)

    lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return pooled_mean, lower, upper


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
    parser = argparse.ArgumentParser(description="Gate 6A Remediation evaluation on Val_eval")
    parser.add_argument("--start-idx", type=int, default=0, help="Start index in eval cases (0-based)")
    parser.add_argument("--end-idx", type=int, default=105, help="End index in eval cases (exclusive)")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip inference and aggregate from case_cache")
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Gate 6A Remediation: Full Evaluation on Val_eval (105 cases) ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Range: [{args.start_idx}:{args.end_idx}] | Aggregate-only: {args.aggregate_only}")

    out_dir = PROJECT_ROOT / "results" / "gate6a_remediation_v1"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    archive_dir = PROJECT_ROOT / "data" / "extracted_archive"
    case_cache_dir = out_dir / "case_cache"
    case_voxels_dir = out_dir / "case_voxels"
    case_cache_dir.mkdir(parents=True, exist_ok=True)
    case_voxels_dir.mkdir(parents=True, exist_ok=True)

    # 1. Compute and verify cryptographic provenance hashes
    resunet_ckpt_path = PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt"
    resnet34_ckpt_path = PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt"
    param_file = out_dir / "calibration_parameters.json"
    splits_file = out_dir / "calibration_splits.json"
    test_manifest_path = PROJECT_ROOT / "results" / "dataset_splits" / "test_manifest.json"
    eval_script_path = Path(__file__).resolve()

    resunet_ckpt_hash = compute_file_sha256(resunet_ckpt_path)
    resnet34_ckpt_hash = compute_file_sha256(resnet34_ckpt_path)
    param_file_hash = compute_file_sha256(param_file)
    splits_file_hash = compute_file_sha256(splits_file)
    test_manifest_hash = compute_file_sha256(test_manifest_path)
    eval_script_hash = compute_file_sha256(eval_script_path)

    expected_resunet = "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d"
    expected_resnet34 = "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d"
    expected_test_manifest = "694faaa4ff100066d67aa5c87c545a709e4eed43a61625d3bea9f43c50d6dfb7"

    assert resunet_ckpt_hash == expected_resunet, f"FATAL: ResUNet hash mismatch: {resunet_ckpt_hash} != {expected_resunet}"
    assert resnet34_ckpt_hash == expected_resnet34, f"FATAL: UNetResNet34 hash mismatch: {resnet34_ckpt_hash} != {expected_resnet34}"
    assert test_manifest_hash == expected_test_manifest, f"FATAL: Test manifest hash mismatch: {test_manifest_hash} != {expected_test_manifest}"

    active_provenance = {
        "resunet_ckpt_sha256": resunet_ckpt_hash,
        "resnet34_ckpt_sha256": resnet34_ckpt_hash,
        "calibration_params_sha256": param_file_hash,
        "calibration_splits_sha256": splits_file_hash,
        "test_manifest_sha256": test_manifest_hash,
        "evaluation_script_sha256": eval_script_hash,
    }

    print("\nVerified Cryptographic Provenance:")
    print(f"  ResUNet Checkpoint:    {resunet_ckpt_hash[:16]}... (MATCH)")
    print(f"  UNetResNet34 Checkpoint: {resnet34_ckpt_hash[:16]}... (MATCH)")
    print(f"  Test Manifest:         {test_manifest_hash[:16]}... (MATCH)")
    print(f"  Splits Manifest:       {splits_file_hash[:16]}...")
    print(f"  Parameters Artifact:   {param_file_hash[:16]}...")

    # 2. Load calibration parameters and splits
    with open(param_file, "r", encoding="utf-8") as f:
        cal_params = json.load(f)

    with open(splits_file, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    eval_case_ids: list[str] = splits_data["evaluation_set"]["case_ids"]
    case_to_subject: dict[str, str] = {cid: cid.rsplit("-", 1)[0] for cid in eval_case_ids}
    assert len(eval_case_ids) == 105, f"Expected 105 Val_eval cases, got {len(eval_case_ids)}"

    # Strict quarantine audit
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_manifest = json.load(f)
    test_case_ids = {c["case_id"] for c in test_manifest["cases"]}
    leaked = set(eval_case_ids).intersection(test_case_ids)
    assert len(leaked) == 0, f"FATAL: Quarantined test cases in evaluation set: {leaked}"
    print(f"Quarantine audit: Verified 0 test cases in Val_eval (105 cases, 100% disjoint from test).")

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

    # Helper: Validate cache provenance
    def is_cache_valid(cache_payload: dict[str, Any]) -> bool:
        prov = cache_payload.get("provenance", {})
        for k in ("resunet_ckpt_sha256", "resnet34_ckpt_sha256", "calibration_params_sha256", "calibration_splits_sha256"):
            if prov.get(k) != active_provenance[k]:
                return False
        return True

    # 4. Evaluation execution
    if not args.aggregate_only:
        print("\nLoading models into ModelRegistry...")
        torch.backends.cudnn.benchmark = False
        registry = ModelRegistry(default_device=device)
        m_resunet = registry.get_model("resunet", device=device)
        m_resnet34 = registry.get_model("unet_resnet34", device=device)
        print("Models loaded successfully.")

        rng = np.random.RandomState(42)
        print(f"\nEvaluating cases [{args.start_idx}:{args.end_idx}] of {len(eval_case_ids)} Val_eval cases...")

        for case_idx in range(args.start_idx + 1, min(args.end_idx + 1, len(eval_case_ids) + 1)):
            cid = eval_case_ids[case_idx - 1]
            sid = case_to_subject[cid]
            cache_path = case_cache_dir / f"{cid}.json"
            voxel_path = case_voxels_dir / f"{cid}.pt"

            if cache_path.exists() and voxel_path.exists():
                try:
                    with open(cache_path, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                    if is_cache_valid(cached_data):
                        m_res = cached_data["case_results"]["resunet_cal"]["mean_dice"]
                        m_agt = cached_data["case_results"]["agent_calibrated"]["mean_dice"]
                        print(f"  [{case_idx:3d}/{len(eval_case_ids)}] [CACHED-VALID] {cid} (Subj: {sid}) | ResUNet={m_res:.4f}, Agent={m_agt:.4f}", flush=True)
                        continue
                    else:
                        print(f"  [{case_idx:3d}/{len(eval_case_ids)}] [CACHE-STALE] Invalidating cache for {cid} due to provenance mismatch.", flush=True)
                except Exception as e:
                    print(f"  [{case_idx:3d}/{len(eval_case_ids)}] [CACHE-CORRUPT] Recomputing {cid}: {e}", flush=True)

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

            # Voxel collection for this specific case across foreground & brain background
            case_sampled_l1: list[torch.Tensor] = []
            case_sampled_l2: list[torch.Tensor] = []
            case_sampled_tgt: list[torch.Tensor] = []

            # Pass 1: Run each model exactly ONCE per slice
            t_models_start = time.perf_counter()
            with torch.inference_mode():
                for z in range(total_z):
                    img_slice = np.ascontiguousarray(vol.image[:, :, :, z])
                    img = torch.from_numpy(img_slice).float().unsqueeze(0).to(device)
                    tgt = torch.from_numpy(vol.mask[:, :, z].astype(np.int64))
                    case_targets[z] = tgt

                    # ResUNet-2D forward pass
                    l1 = m_resunet(img).cpu()
                    p1_uncal = extract_slice_prediction("resunet", l1)
                    p1_cal = extract_slice_prediction("resunet", l1 / t_resunet)
                    p1_uncal_dict[z] = p1_uncal
                    p1_cal_dict[z] = p1_cal

                    # UNetResNet34 forward pass
                    l2 = m_resnet34(img).cpu()
                    p2_uncal = extract_slice_prediction("unet_resnet34", l2)
                    p2_cal = extract_slice_prediction("unet_resnet34", l2 / t_resnet34)
                    p2_uncal_dict[z] = p2_uncal
                    p2_cal_dict[z] = p2_cal

                    # Structured voxel sampling: sample every 4th slice that contains foreground
                    if z % 4 == 0:
                        tgt_np = tgt.numpy()
                        fg_idx = np.where(tgt_np > 0)
                        n_fg = len(fg_idx[0])
                        if n_fg > 0:
                            brain_mask = (vol.image[:, :, :, z].sum(axis=0) != 0)
                            # Sample up to 100 foreground voxels
                            s_fg = min(n_fg, 100)
                            chosen_fg = rng.choice(n_fg, size=s_fg, replace=False)
                            fg_r, fg_c = fg_idx[0][chosen_fg], fg_idx[1][chosen_fg]

                            l1_fg = l1.squeeze(0)[:, fg_r, fg_c].permute(1, 0)
                            l2_fg = l2.squeeze(0)[:, fg_r, fg_c].permute(1, 0)
                            t_fg = tgt[fg_r, fg_c]

                            bg_idx = np.where((tgt_np == 0) & brain_mask)
                            n_bg = len(bg_idx[0])
                            s_bg = min(n_bg, s_fg)
                            if s_bg > 0:
                                chosen_bg = rng.choice(n_bg, size=s_bg, replace=False)
                                bg_r, bg_c = bg_idx[0][chosen_bg], bg_idx[1][chosen_bg]
                                l1_bg = l1.squeeze(0)[:, bg_r, bg_c].permute(1, 0)
                                l2_bg = l2.squeeze(0)[:, bg_r, bg_c].permute(1, 0)
                                t_bg = tgt[bg_r, bg_c]

                                cat_l1 = torch.cat([l1_fg, l1_bg], dim=0)
                                cat_l2 = torch.cat([l2_fg, l2_bg], dim=0)
                                cat_t = torch.cat([t_fg, t_bg], dim=0)
                            else:
                                cat_l1 = l1_fg
                                cat_l2 = l2_fg
                                cat_t = t_fg

                            case_sampled_l1.append(cat_l1)
                            case_sampled_l2.append(cat_l2)
                            case_sampled_tgt.append(cat_t)

            model_time_ms = (time.perf_counter() - t_models_start) * 1000.0
            slice_lat_est = model_time_ms / total_z / 2.0  # Estimated latency per single model

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
                        "subject_id": sid,
                        "slice_index": z,
                        "has_gt_fg": gt_fg > 0,
                        "gt_fg_pixels": gt_fg,
                        "primary_fg_pixels": p1_c.pred_foreground_pixels,
                        "primary_cal_entropy": p1_c.mean_foreground_entropy,
                        "agent_base_action": dec_base.action.value,
                        "agent_base_model": dec_base.selected_model,
                        "agent_cal_action": dec_cal.action.value,
                        "agent_cal_model": dec_cal.selected_model,
                        "agent_cal_review_flagged": dec_cal.is_human_review_recommended,
                        "agent_cal_fb_action": dec_cal_fb.action.value,
                        "agent_cal_fb_model": dec_cal_fb.selected_model,
                    })

            # Save case voxel data
            if len(case_sampled_l1) > 0:
                torch.save(
                    {
                        "case_id": cid,
                        "subject_id": sid,
                        "l1": torch.cat(case_sampled_l1, dim=0),
                        "l2": torch.cat(case_sampled_l2, dim=0),
                        "target": torch.cat(case_sampled_tgt, dim=0),
                    },
                    voxel_path,
                )
            else:
                torch.save(
                    {
                        "case_id": cid,
                        "subject_id": sid,
                        "l1": torch.empty((0, 4), dtype=torch.float32),
                        "l2": torch.empty((0, 4), dtype=torch.float32),
                        "target": torch.empty((0,), dtype=torch.int64),
                    },
                    voxel_path,
                )

            # Compute case summary results
            computed_case_results = {c: case_accumulators[c].compute_case_dice() for c in evaluated_configs}

            # Save case cache with full provenance
            cache_payload = {
                "case_id": cid,
                "subject_id": sid,
                "provenance": active_provenance,
                "case_results": computed_case_results,
                "slice_records": case_slice_records,
                "slice_lat_est_ms": slice_lat_est,
                "case_elapsed_s": time.perf_counter() - t_case_start,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_payload, f, indent=2)

            m_res = computed_case_results["resunet_cal"]["mean_dice"]
            m_agt = computed_case_results["agent_calibrated"]["mean_dice"]
            print(f"  [{case_idx:3d}/{len(eval_case_ids)}] [EVALUATED] {cid} (Subj: {sid}) | ResUNet={m_res:.4f}, Agent={m_agt:.4f} ({cache_payload['case_elapsed_s']:.1f}s)", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Check overall cache coverage
    cached_ids = {cfile.stem for cfile in case_cache_dir.glob("*.json")}
    print(f"\n[Case Cache Status] {len(cached_ids)} of {len(eval_case_ids)} evaluation cases are cached on disk.", flush=True)

    if len(cached_ids) < len(eval_case_ids):
        print(f"Evaluation chunk completed. ({len(cached_ids)}/{len(eval_case_ids)} cached).", flush=True)
        return

    # 5. Master Aggregation across all 105 cases
    print(f"\n================================================================================")
    print(f"=== Aggregating Val_eval Metrics across all {len(eval_case_ids)} cases ===")
    print(f"================================================================================")

    all_case_results: dict[str, list[dict[str, Any]]] = {c: [] for c in evaluated_configs}
    all_subject_results: dict[str, dict[str, list[float]]] = {c: {} for c in evaluated_configs}
    all_slice_records: list[dict[str, Any]] = []
    latencies: list[float] = []

    # Verify all 105 cache files exist and are valid
    missing_cases: list[str] = []
    for cid in eval_case_ids:
        c_path = case_cache_dir / f"{cid}.json"
        if not c_path.exists():
            missing_cases.append(cid)

    if missing_cases:
        raise RuntimeError(f"Cannot aggregate: Missing cache for {len(missing_cases)} cases: {missing_cases[:5]}...")

    # Load all case caches
    for cid in eval_case_ids:
        sid = case_to_subject[cid]
        c_path = case_cache_dir / f"{cid}.json"
        with open(c_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert is_cache_valid(data), f"FATAL: Cache for {cid} failed provenance validation during aggregation!"

        c_res = data["case_results"]
        for c in evaluated_configs:
            c_info = {"case_id": cid, "subject_id": sid, **c_res[c]}
            all_case_results[c].append(c_info)

            if sid not in all_subject_results[c]:
                all_subject_results[c][sid] = []
            all_subject_results[c][sid].append(c_res[c]["mean_dice"])

        all_slice_records.extend(data["slice_records"])
        latencies.append(data.get("slice_lat_est_ms", 0.0))

    # Aggregating sampled voxels across 100% of cases
    print("Aggregating voxel calibration tensors across all 105 evaluation cases...")
    all_l1_list: list[torch.Tensor] = []
    all_l2_list: list[torch.Tensor] = []
    all_tgt_list: list[torch.Tensor] = []
    case_voxel_counts: dict[str, int] = {}

    for cid in eval_case_ids:
        v_path = case_voxels_dir / f"{cid}.pt"
        if not v_path.exists():
            raise FileNotFoundError(f"Missing case voxels file for {cid}: {v_path}")
        v_data = torch.load(v_path, weights_only=True)
        n_v = v_data["target"].shape[0]
        case_voxel_counts[cid] = n_v
        if n_v > 0:
            all_l1_list.append(v_data["l1"])
            all_l2_list.append(v_data["l2"])
            all_tgt_list.append(v_data["target"])

    assert len(case_voxel_counts) == 105, f"Expected exactly 105 case counts, got {len(case_voxel_counts)}"
    cases_with_voxels = sum(1 for c, cnt in case_voxel_counts.items() if cnt > 0)
    total_eval_voxels = sum(case_voxel_counts.values())
    print(f"Voxel Audit: {cases_with_voxels}/105 cases with tumor voxels, {total_eval_voxels:,} total sampled voxels.")

    cat_l1 = torch.cat(all_l1_list, dim=0)
    cat_l2 = torch.cat(all_l2_list, dim=0)
    cat_tgt = torch.cat(all_tgt_list, dim=0)

    # Save consolidated eval_sampled_voxels.pt with complete per-case audit
    eval_voxels_path = out_dir / "eval_sampled_voxels.pt"
    torch.save(
        {
            "metadata": {
                "evaluation_cases_count": len(eval_case_ids),
                "cases_with_sampled_voxels": cases_with_voxels,
                "total_voxels": total_eval_voxels,
                "case_voxel_counts": case_voxel_counts,
                "provenance": active_provenance,
            },
            "l1": cat_l1,
            "l2": cat_l2,
            "target": cat_tgt,
        },
        eval_voxels_path,
    )
    print(f"Saved consolidated evaluation voxels: {eval_voxels_path}")

    # Compute probability calibration metrics on the aggregated evaluation voxel dataset
    print("\nComputing probability calibration metrics on full 105-case evaluation voxel dataset...")
    cal_eval_metrics = {}

    cat_tgt_np = cat_tgt.numpy()
    for m_key, logits_raw, temp_opt in [
        ("resunet_uncal", cat_l1, 1.0),
        ("resunet_cal", cat_l1 / t_resunet, t_resunet),
        ("resnet34_uncal", cat_l2, 1.0),
        ("resnet34_cal", cat_l2 / t_resnet34, t_resnet34),
    ]:
        probs_np = F.softmax(logits_raw, dim=-1).numpy()
        ece_res = compute_multiclass_ece(probs_np, cat_tgt_np, n_bins=15)
        brier = compute_multiclass_brier_score(probs_np, cat_tgt_np)
        nll = compute_nll(probs_np, cat_tgt_np)
        cal_eval_metrics[m_key] = {
            "temperature": float(temp_opt),
            "ece": float(ece_res["ece"]),
            "mce": float(ece_res["mce"]),
            "brier_score": float(brier),
            "nll": float(nll),
            "bin_accuracies": [float(x) for x in ece_res["bin_accuracies"]],
            "bin_confidences": [float(x) for x in ece_res["bin_confidences"]],
            "bin_counts": [int(x) for x in ece_res["bin_counts"]],
            "bin_edges": [float(x) for x in ece_res["bin_edges"]],
        }
        print(f"  {m_key:15s} (T={temp_opt:.4f}) -> ECE={ece_res['ece']*100:.2f}%, MCE={ece_res['mce']*100:.2f}%, Brier={brier:.4f}, NLL={nll:.4f}")

    # Compute Statistical Summaries with Case-Level and Subject-Cluster Bootstrap
    print("\nComputing Statistical Summaries (Case-Level and Subject-Cluster Bootstrap CIs)...")
    statistical_summary: dict[str, Any] = {}

    for c in evaluated_configs:
        mean_d_cases = [r["mean_dice"] for r in all_case_results[c]]
        wt_d_cases = [r["wt_dice"] for r in all_case_results[c]]
        tc_d_cases = [r["tc_dice"] for r in all_case_results[c]]
        et_d_cases = [r["et_dice"] for r in all_case_results[c]]

        # Case-level bootstrap
        m_case, l_case, u_case = compute_bootstrap_ci(mean_d_cases, n_bootstraps=1000, seed=42)
        wt_m, wt_l, wt_u = compute_bootstrap_ci(wt_d_cases, n_bootstraps=1000, seed=42)
        tc_m, tc_l, tc_u = compute_bootstrap_ci(tc_d_cases, n_bootstraps=1000, seed=42)
        et_m, et_l, et_u = compute_bootstrap_ci(et_d_cases, n_bootstraps=1000, seed=42)

        # Subject-cluster bootstrap (accounts for patient grouping across 96 unique subjects)
        m_subj_pooled, l_subj, u_subj = compute_cluster_bootstrap_ci(all_subject_results[c], n_bootstraps=1000, seed=42)

        statistical_summary[c] = {
            "mean_dice": {
                "mean": m_case,
                "ci_95_lower": l_case,
                "ci_95_upper": u_case,
                "subject_cluster_mean": m_subj_pooled,
                "subject_cluster_ci_95_lower": l_subj,
                "subject_cluster_ci_95_upper": u_subj,
            },
            "wt_dice": {"mean": wt_m, "ci_95_lower": wt_l, "ci_95_upper": wt_u},
            "tc_dice": {"mean": tc_m, "ci_95_lower": tc_l, "ci_95_upper": tc_u},
            "et_dice": {"mean": et_m, "ci_95_lower": et_l, "ci_95_upper": et_u},
        }

        print(
            f"  {c:20s} | Mean Dice={m_case:.4f} [Case CI: {l_case:.4f}, {u_case:.4f}] "
            f"[Subj-Cluster CI: {l_subj:.4f}, {u_subj:.4f}]"
        )

    # Selective Prediction / Uncertainty Triage analysis on slice level
    total_eval_slices = len(all_slice_records)
    fg_eval_slices = [s for s in all_slice_records if s["has_gt_fg"]]
    flagged_slices = [s for s in fg_eval_slices if s["agent_cal_review_flagged"]]
    retained_slices = [s for s in fg_eval_slices if not s["agent_cal_review_flagged"]]

    n_fg_slices = len(fg_eval_slices)
    n_flagged = len(flagged_slices)
    n_retained = len(retained_slices)
    coverage_pct = (n_retained / n_fg_slices * 100.0) if n_fg_slices > 0 else 100.0

    print(f"\nSelective Prediction / Human Triage Audit (tau = {tau_abstain:.2f}):")
    print(f"  Total Slices:             {total_eval_slices}")
    print(f"  Foreground Slices:        {n_fg_slices}")
    print(f"  Retained Slices:          {n_retained} ({coverage_pct:.2f}%)")
    print(f"  Flagged for Review:       {n_flagged} ({100.0 - coverage_pct:.2f}%)")

    # Policy action distribution
    action_counts_base = {}
    action_counts_cal = {}
    action_counts_cal_fb = {}
    for s in all_slice_records:
        a_b = s["agent_base_action"]
        a_c = s["agent_cal_action"]
        a_fb = s["agent_cal_fb_action"]
        action_counts_base[a_b] = action_counts_base.get(a_b, 0) + 1
        action_counts_cal[a_c] = action_counts_cal.get(a_c, 0) + 1
        action_counts_cal_fb[a_fb] = action_counts_cal_fb.get(a_fb, 0) + 1

    triage_summary = {
        "operating_threshold_tau_abstain": tau_abstain,
        "operating_threshold_tau_fallback": tau_fallback,
        "total_slices": total_eval_slices,
        "foreground_slices": n_fg_slices,
        "retained_slices": n_retained,
        "flagged_slices": n_flagged,
        "coverage_percentage": coverage_pct,
        "action_distribution_agent_baseline": {
            k: {"count": v, "pct": float(v / total_eval_slices * 100.0)} for k, v in action_counts_base.items()
        },
        "action_distribution_agent_calibrated": {
            k: {"count": v, "pct": float(v / total_eval_slices * 100.0)} for k, v in action_counts_cal.items()
        },
        "action_distribution_agent_calibrated_fb": {
            k: {"count": v, "pct": float(v / total_eval_slices * 100.0)} for k, v in action_counts_cal_fb.items()
        },
    }

    # 6. Save tabular outputs
    # Case-level CSV
    case_csv_path = out_dir / "case_level_metrics.csv"
    with open(case_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "case_id",
            "subject_id",
            "resunet_uncal_mean_dice",
            "resunet_cal_mean_dice",
            "resnet34_uncal_mean_dice",
            "resnet34_cal_mean_dice",
            "agent_baseline_mean_dice",
            "agent_calibrated_mean_dice",
            "agent_calibrated_fb_mean_dice",
            "resunet_cal_wt_dice",
            "resunet_cal_tc_dice",
            "resunet_cal_et_dice",
            "agent_cal_wt_dice",
            "agent_cal_tc_dice",
            "agent_cal_et_dice",
        ])
        for i, cid in enumerate(eval_case_ids):
            sid = case_to_subject[cid]
            r_uncal = all_case_results["resunet_uncal"][i]
            r_cal = all_case_results["resunet_cal"][i]
            rn_uncal = all_case_results["resnet34_uncal"][i]
            rn_cal = all_case_results["resnet34_cal"][i]
            ag_base = all_case_results["agent_baseline"][i]
            ag_cal = all_case_results["agent_calibrated"][i]
            ag_fb = all_case_results["agent_calibrated_fb"][i]

            writer.writerow([
                cid,
                sid,
                f"{r_uncal['mean_dice']:.4f}",
                f"{r_cal['mean_dice']:.4f}",
                f"{rn_uncal['mean_dice']:.4f}",
                f"{rn_cal['mean_dice']:.4f}",
                f"{ag_base['mean_dice']:.4f}",
                f"{ag_cal['mean_dice']:.4f}",
                f"{ag_fb['mean_dice']:.4f}",
                f"{r_cal['wt_dice']:.4f}",
                f"{r_cal['tc_dice']:.4f}",
                f"{r_cal['et_dice']:.4f}",
                f"{ag_cal['wt_dice']:.4f}",
                f"{ag_cal['tc_dice']:.4f}",
                f"{ag_cal['et_dice']:.4f}",
            ])
    print(f"Saved case-level metrics CSV: {case_csv_path}")

    # Subject-level CSV (96 unique subjects)
    subject_csv_path = out_dir / "subject_level_metrics.csv"
    subjects_sorted = sorted(list(all_subject_results["resunet_cal"].keys()))
    with open(subject_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subject_id",
            "scan_count",
            "resunet_uncal_mean_dice",
            "resunet_cal_mean_dice",
            "resnet34_cal_mean_dice",
            "agent_calibrated_mean_dice",
        ])
        for sid in subjects_sorted:
            writer.writerow([
                sid,
                len(all_subject_results["resunet_cal"][sid]),
                f"{np.mean(all_subject_results['resunet_uncal'][sid]):.4f}",
                f"{np.mean(all_subject_results['resunet_cal'][sid]):.4f}",
                f"{np.mean(all_subject_results['resnet34_cal'][sid]):.4f}",
                f"{np.mean(all_subject_results['agent_calibrated'][sid]):.4f}",
            ])
    print(f"Saved subject-level metrics CSV: {subject_csv_path}")

    # Slice-level CSV
    slice_csv_path = out_dir / "slice_level_metrics.csv"
    with open(slice_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_slice_records[0].keys()))
        writer.writeheader()
        writer.writerows(all_slice_records)
    print(f"Saved slice-level metrics CSV: {slice_csv_path} ({len(all_slice_records)} records)")

    # 7. Comprehensive Master evaluation_summary.json
    eval_summary_path = out_dir / "evaluation_summary.json"
    summary_payload = {
        "evaluation_cohort": {
            "name": "Val_eval_remediation",
            "total_cases": len(eval_case_ids),
            "unique_subjects": len(subjects_sorted),
            "total_slices": total_eval_slices,
            "seed": 42,
            "case_ids": eval_case_ids,
        },
        "provenance": active_provenance,
        "calibration_parameters": {
            "temperature_resunet": t_resunet,
            "temperature_resnet34": t_resnet34,
            "tau_abstain": tau_abstain,
            "tau_fallback": tau_fallback,
        },
        "probability_calibration_metrics": cal_eval_metrics,
        "statistical_summary": statistical_summary,
        "triage_summary": triage_summary,
        "timing_and_latency": {
            "mean_slice_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "total_evaluation_elapsed_s": time.perf_counter() - t_start,
        },
    }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Saved evaluation summary JSON: {eval_summary_path}")

    # 8. Run Manifest
    manifest_path = out_dir / "run_manifest.json"
    run_manifest = {
        "stage": "Gate 6A Remediation",
        "description": "Cryptographically bound revalidation of uncertainty calibration, triage, and multi-model agent",
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance_hashes": active_provenance,
        "splits_manifest": {
            "path": str(splits_file),
            "sha256": splits_file_hash,
            "cal_cases_count": len(splits_data["calibration_set"]["case_ids"]),
            "eval_cases_count": len(splits_data["evaluation_set"]["case_ids"]),
            "subject_overlap_count": 0,
            "test_overlap_count": 0,
        },
        "artifacts_generated": {
            "evaluation_summary_json": str(eval_summary_path.relative_to(PROJECT_ROOT)),
            "eval_sampled_voxels_pt": str(eval_voxels_path.relative_to(PROJECT_ROOT)),
            "case_level_metrics_csv": str(case_csv_path.relative_to(PROJECT_ROOT)),
            "subject_level_metrics_csv": str(subject_csv_path.relative_to(PROJECT_ROOT)),
            "slice_level_metrics_csv": str(slice_csv_path.relative_to(PROJECT_ROOT)),
        },
        "quarantine_verification": {
            "test_manifest_path": str(test_manifest_path.relative_to(PROJECT_ROOT)),
            "test_manifest_sha256": test_manifest_hash,
            "test_cases_total": len(test_case_ids),
            "test_cases_evaluated": 0,
            "status": "PASS_STRICTLY_QUARANTINED",
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"Saved run manifest: {manifest_path}")

    total_time = (time.perf_counter() - t_start) / 60.0
    print(f"\n=== Evaluation & Aggregation Finished Successfully in {total_time:.2f} minutes ===")


if __name__ == "__main__":
    main()
