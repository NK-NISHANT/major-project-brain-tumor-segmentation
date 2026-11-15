from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.agent.models import CANONICAL_MODEL_NAMES, DEFAULT_CHECKPOINTS, ModelRegistry
from src.agent.runner import InferenceAgent
from src.agent.state import AgentAction, PolicyConfig, SlicePrediction
from src.data.dataset import (
    DEFAULT_RAW_DATA_DIR,
    BraTSGLISliceDataset,
    create_train_val_datasets,
)
from src.evaluation.metrics import _nanmean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "agent_evaluation" / "milestone4"


@dataclass(frozen=True)
class SliceEvaluationResult:
    """Detailed evaluation record for one slice prediction evaluated against ground truth."""

    method: str
    case_id: str
    slice_index: int
    has_foreground: bool
    gt_foreground_pixels: int
    pred_foreground_pixels: int
    mean_dice: float
    mean_iou: float
    wt_dice: float
    tc_dice: float
    et_dice: float
    is_empty_prediction: bool
    is_confirmed_miss: bool
    is_false_positive_slice: bool
    is_true_negative_slice: bool
    latency_ms: float
    # Agent-specific fields (None for standalone models)
    action: str | None = None
    selected_model: str | None = None
    is_fallback: bool | None = None
    is_unreliable_flagged: bool | None = None
    fallback_suppressed_by_context: bool | None = None
    mean_uncertainty: float | None = None
    reasons: str | None = None
    evaluated_neighbors: int | None = None
    unavailable_neighbors: int | None = None
    fg_neighbors: int | None = None
    empty_neighbors: int | None = None


class GlobalMetricAccumulator:
    """Accumulates voxel-level TP, FP, FN across slices to compute exact global micro-averaged metrics.

    Computes:
    - Per-class (0=background, 1=NCR, 2=ED, 3=ET) micro-averaged Dice and IoU
    - Composite clinical regions:
      - Whole Tumor (WT): classes {1, 2, 3}
      - Tumor Core (TC): classes {1, 3}
      - Enhancing Tumor (ET): class {3}
    - Foreground mean (excluding class 0)
    """

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.tp = torch.zeros(num_classes, dtype=torch.float64)
        self.fp = torch.zeros(num_classes, dtype=torch.float64)
        self.fn = torch.zeros(num_classes, dtype=torch.float64)

        # Composite counters
        self.wt_tp = 0.0
        self.wt_fp = 0.0
        self.wt_fn = 0.0

        self.tc_tp = 0.0
        self.tc_fp = 0.0
        self.tc_fn = 0.0

        self.et_tp = 0.0
        self.et_fp = 0.0
        self.et_fn = 0.0

    @torch.no_grad()
    def update(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        pred_cpu = pred_mask.detach().cpu().to(torch.int64)
        target_cpu = target_mask.detach().cpu().to(torch.int64)

        # Per-class accumulation
        for c in range(self.num_classes):
            p_c = pred_cpu == c
            t_c = target_cpu == c
            self.tp[c] += float(torch.logical_and(p_c, t_c).sum().item())
            self.fp[c] += float(torch.logical_and(p_c, ~t_c).sum().item())
            self.fn[c] += float(torch.logical_and(~p_c, t_c).sum().item())

        # Composite: Whole Tumor (WT: classes 1, 2, 3)
        p_wt = pred_cpu > 0
        t_wt = target_cpu > 0
        self.wt_tp += float(torch.logical_and(p_wt, t_wt).sum().item())
        self.wt_fp += float(torch.logical_and(p_wt, ~t_wt).sum().item())
        self.wt_fn += float(torch.logical_and(~p_wt, t_wt).sum().item())

        # Composite: Tumor Core (TC: classes 1, 3)
        p_tc = torch.logical_or(pred_cpu == 1, pred_cpu == 3)
        t_tc = torch.logical_or(target_cpu == 1, target_cpu == 3)
        self.tc_tp += float(torch.logical_and(p_tc, t_tc).sum().item())
        self.tc_fp += float(torch.logical_and(p_tc, ~t_tc).sum().item())
        self.tc_fn += float(torch.logical_and(~p_tc, t_tc).sum().item())

        # Composite: Enhancing Tumor (ET: class 3)
        p_et = pred_cpu == 3
        t_et = target_cpu == 3
        self.et_tp += float(torch.logical_and(p_et, t_et).sum().item())
        self.et_fp += float(torch.logical_and(p_et, ~t_et).sum().item())
        self.et_fn += float(torch.logical_and(~p_et, t_et).sum().item())

    def compute(self) -> dict[str, Any]:
        dice_scores: list[float] = []
        iou_scores: list[float] = []

        for c in range(self.num_classes):
            tp = float(self.tp[c].item())
            fp = float(self.fp[c].item())
            fn = float(self.fn[c].item())

            denom_dice = (2.0 * tp) + fp + fn
            denom_iou = tp + fp + fn

            dice = (2.0 * tp / denom_dice) if denom_dice > 0 else float("nan")
            iou = (tp / denom_iou) if denom_iou > 0 else float("nan")

            dice_scores.append(dice)
            iou_scores.append(iou)

        # Foreground mean (classes 1, 2, 3)
        fg_dice = _nanmean(dice_scores[1:])
        fg_iou = _nanmean(iou_scores[1:])

        def _safe_dice(tp: float, fp: float, fn: float) -> float:
            d = (2.0 * tp) + fp + fn
            return (2.0 * tp / d) if d > 0 else float("nan")

        def _safe_iou(tp: float, fp: float, fn: float) -> float:
            d = tp + fp + fn
            return (tp / d) if d > 0 else float("nan")

        return {
            "global_micro_dice_foreground": fg_dice,
            "global_micro_iou_foreground": fg_iou,
            "global_dice_wt": _safe_dice(self.wt_tp, self.wt_fp, self.wt_fn),
            "global_iou_wt": _safe_iou(self.wt_tp, self.wt_fp, self.wt_fn),
            "global_dice_tc": _safe_dice(self.tc_tp, self.tc_fp, self.tc_fn),
            "global_iou_tc": _safe_iou(self.tc_tp, self.tc_fp, self.tc_fn),
            "global_dice_et": _safe_dice(self.et_tp, self.et_fp, self.et_fn),
            "global_iou_et": _safe_iou(self.et_tp, self.et_fp, self.et_fn),
            "per_class_dice": dice_scores,
            "per_class_iou": iou_scores,
        }


def compute_slice_metrics(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    num_classes: int = 4,
) -> tuple[float, float, float, float, float]:
    """Compute slice-level unweighted macro metrics and composite region Dice scores.

    Returns:
        (mean_fg_dice, mean_fg_iou, wt_dice, tc_dice, et_dice)
    """
    pred_cpu = pred_mask.detach().cpu().to(torch.int64)
    target_cpu = target_mask.detach().cpu().to(torch.int64)

    dice_scores: list[float] = []
    iou_scores: list[float] = []

    for c in range(1, num_classes):
        p_c = pred_cpu == c
        t_c = target_cpu == c
        tp = float(torch.logical_and(p_c, t_c).sum().item())
        fp = float(torch.logical_and(p_c, ~t_c).sum().item())
        fn = float(torch.logical_and(~p_c, t_c).sum().item())

        denom_dice = (2.0 * tp) + fp + fn
        denom_iou = tp + fp + fn

        dice = (2.0 * tp / denom_dice) if denom_dice > 0 else float("nan")
        iou = (tp / denom_iou) if denom_iou > 0 else float("nan")
        dice_scores.append(dice)
        iou_scores.append(iou)

    mean_fg_dice = _nanmean(dice_scores)
    mean_fg_iou = _nanmean(iou_scores)

    # WT Dice
    p_wt = pred_cpu > 0
    t_wt = target_cpu > 0
    wt_tp = float(torch.logical_and(p_wt, t_wt).sum().item())
    wt_fp = float(torch.logical_and(p_wt, ~t_wt).sum().item())
    wt_fn = float(torch.logical_and(~p_wt, t_wt).sum().item())
    wt_denom = (2.0 * wt_tp) + wt_fp + wt_fn
    wt_dice = (2.0 * wt_tp / wt_denom) if wt_denom > 0 else float("nan")

    # TC Dice (classes 1, 3)
    p_tc = torch.logical_or(pred_cpu == 1, pred_cpu == 3)
    t_tc = torch.logical_or(target_cpu == 1, target_cpu == 3)
    tc_tp = float(torch.logical_and(p_tc, t_tc).sum().item())
    tc_fp = float(torch.logical_and(p_tc, ~t_tc).sum().item())
    tc_fn = float(torch.logical_and(~p_tc, t_tc).sum().item())
    tc_denom = (2.0 * tc_tp) + tc_fp + tc_fn
    tc_dice = (2.0 * tc_tp / tc_denom) if tc_denom > 0 else float("nan")

    # ET Dice (class 3)
    p_et = pred_cpu == 3
    t_et = target_cpu == 3
    et_tp = float(torch.logical_and(p_et, t_et).sum().item())
    et_fp = float(torch.logical_and(p_et, ~t_et).sum().item())
    et_fn = float(torch.logical_and(~p_et, t_et).sum().item())
    et_denom = (2.0 * et_tp) + et_fp + et_fn
    et_dice = (2.0 * et_tp / et_denom) if et_denom > 0 else float("nan")

    return mean_fg_dice, mean_fg_iou, wt_dice, tc_dice, et_dice


def evaluate_slice_record(
    method: str,
    case_id: str,
    slice_index: int,
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    latency_ms: float,
    action: str | None = None,
    selected_model: str | None = None,
    is_fallback: bool | None = None,
    is_unreliable_flagged: bool | None = None,
    fallback_suppressed_by_context: bool | None = None,
    mean_uncertainty: float | None = None,
    reasons: str | None = None,
    evaluated_neighbors: int | None = None,
    unavailable_neighbors: int | None = None,
    fg_neighbors: int | None = None,
    empty_neighbors: int | None = None,
) -> SliceEvaluationResult:
    """Evaluate a slice prediction against ground truth and categorize failure modes."""
    gt_fg = int((target_mask > 0).sum().item())
    pred_fg = int((pred_mask > 0).sum().item())

    has_fg = gt_fg > 0
    is_empty_pred = pred_fg == 0
    is_confirmed_miss = has_fg and is_empty_pred
    is_fp_slice = (not has_fg) and (not is_empty_pred)
    is_tn_slice = (not has_fg) and is_empty_pred

    mean_dice, mean_iou, wt_dice, tc_dice, et_dice = compute_slice_metrics(
        pred_mask, target_mask
    )

    return SliceEvaluationResult(
        method=method,
        case_id=case_id,
        slice_index=slice_index,
        has_foreground=has_fg,
        gt_foreground_pixels=gt_fg,
        pred_foreground_pixels=pred_fg,
        mean_dice=mean_dice,
        mean_iou=mean_iou,
        wt_dice=wt_dice,
        tc_dice=tc_dice,
        et_dice=et_dice,
        is_empty_prediction=is_empty_pred,
        is_confirmed_miss=is_confirmed_miss,
        is_false_positive_slice=is_fp_slice,
        is_true_negative_slice=is_tn_slice,
        latency_ms=latency_ms,
        action=action,
        selected_model=selected_model,
        is_fallback=is_fallback,
        is_unreliable_flagged=is_unreliable_flagged,
        fallback_suppressed_by_context=fallback_suppressed_by_context,
        mean_uncertainty=mean_uncertainty,
        reasons=reasons,
        evaluated_neighbors=evaluated_neighbors,
        unavailable_neighbors=unavailable_neighbors,
        fg_neighbors=fg_neighbors,
        empty_neighbors=empty_neighbors,
    )


def summarize_system_evaluation(
    method_name: str,
    results: Sequence[SliceEvaluationResult],
    latencies_ms: Sequence[float],
    model_load_time_seconds: float,
    global_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate slice-level evaluations and global metrics into a comprehensive system summary."""
    total_slices = len(results)
    if total_slices == 0:
        return {"method": method_name, "total_slices": 0}

    fg_results = [r for r in results if r.has_foreground]
    bg_results = [r for r in results if not r.has_foreground]

    fg_dices = [r.mean_dice for r in fg_results if not np.isnan(r.mean_dice)]
    fg_ious = [r.mean_iou for r in fg_results if not np.isnan(r.mean_iou)]
    wt_dices = [r.wt_dice for r in fg_results if not np.isnan(r.wt_dice)]
    tc_dices = [r.tc_dice for r in fg_results if not np.isnan(r.tc_dice)]
    et_dices = [r.et_dice for r in fg_results if not np.isnan(r.et_dice)]

    confirmed_misses = sum(1 for r in fg_results if r.is_confirmed_miss)
    miss_rate = (confirmed_misses / len(fg_results) * 100.0) if fg_results else 0.0

    fp_slices = sum(1 for r in bg_results if r.is_false_positive_slice)
    fp_rate = (fp_slices / len(bg_results) * 100.0) if bg_results else 0.0

    tn_slices = sum(1 for r in bg_results if r.is_true_negative_slice)
    tn_rate = (tn_slices / len(bg_results) * 100.0) if bg_results else 0.0

    empty_predictions = sum(1 for r in results if r.is_empty_prediction)

    # Latency percentiles
    lats = np.array(latencies_ms) if latencies_ms else np.array([0.0])

    summary: dict[str, Any] = {
        "method": method_name,
        "total_slices": total_slices,
        "foreground_slices": len(fg_results),
        "background_slices": len(bg_results),
        # Slice-level macro-averaged metrics
        "slice_macro_mean_dice_foreground": float(np.mean(fg_dices)) if fg_dices else float("nan"),
        "slice_macro_mean_iou_foreground": float(np.mean(fg_ious)) if fg_ious else float("nan"),
        "slice_macro_mean_wt_dice": float(np.mean(wt_dices)) if wt_dices else float("nan"),
        "slice_macro_mean_tc_dice": float(np.mean(tc_dices)) if tc_dices else float("nan"),
        "slice_macro_mean_et_dice": float(np.mean(et_dices)) if et_dices else float("nan"),
        # Global micro-averaged metrics
        "global_micro_dice_foreground": global_metrics.get("global_micro_dice_foreground", float("nan")),
        "global_micro_iou_foreground": global_metrics.get("global_micro_iou_foreground", float("nan")),
        "global_dice_wt": global_metrics.get("global_dice_wt", float("nan")),
        "global_dice_tc": global_metrics.get("global_dice_tc", float("nan")),
        "global_dice_et": global_metrics.get("global_dice_et", float("nan")),
        "per_class_global_dice": global_metrics.get("per_class_dice", []),
        # Error & failure rates
        "confirmed_complete_misses": confirmed_misses,
        "confirmed_miss_rate_percent": miss_rate,
        "false_positive_slices": fp_slices,
        "false_positive_rate_percent": fp_rate,
        "true_negative_slices": tn_slices,
        "true_negative_rate_percent": tn_rate,
        "total_empty_predictions": empty_predictions,
        # Timing & latency profile
        "model_load_time_seconds": model_load_time_seconds,
        "latency_ms": {
            "mean": float(np.mean(lats)),
            "median": float(np.median(lats)),
            "p95": float(np.percentile(lats, 95)),
            "min": float(np.min(lats)),
            "max": float(np.max(lats)),
            "total_inference_time_seconds": float(np.sum(lats) / 1000.0),
        },
    }

    # Agent decision statistics
    agent_results = [r for r in results if r.action is not None]
    if agent_results:
        accepted_count = sum(1 for r in agent_results if r.action == AgentAction.ACCEPT.value)
        fallback_count = sum(1 for r in agent_results if r.action == AgentAction.FALLBACK_SWITCH.value)
        unreliable_count = sum(1 for r in agent_results if r.action == AgentAction.FLAG_UNRELIABLE.value)
        suppressed_count = sum(1 for r in agent_results if r.fallback_suppressed_by_context is True)

        model_counts: dict[str, int] = {}
        for r in agent_results:
            if r.selected_model:
                model_counts[r.selected_model] = model_counts.get(r.selected_model, 0) + 1

        summary["agent_decisions"] = {
            "accepted_count": accepted_count,
            "accepted_percent": (accepted_count / len(agent_results) * 100.0),
            "fallback_count": fallback_count,
            "fallback_percent": (fallback_count / len(agent_results) * 100.0),
            "unreliable_flag_count": unreliable_count,
            "unreliable_flag_percent": (unreliable_count / len(agent_results) * 100.0),
            "fallback_suppressed_by_context_count": suppressed_count,
            "fallback_suppressed_by_context_percent": (suppressed_count / len(agent_results) * 100.0),
            "model_selection_distribution": model_counts,
        }

    return summary


def run_milestone4_evaluation(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    primary_model: str = "unetpp",
    fallback_priority: tuple[str, ...] = ("unet",),
    axial_context_window: int = 2,
    min_neighbor_foreground_slices: int = 1,
    fallback_when_context_unavailable: bool = False,
    high_uncertainty_threshold: float | None = None,
    device_name: str | None = None,
    max_slices: int | None = None,
) -> dict[str, Any]:
    """Run comprehensive 5-system comparative evaluation for Milestone 4."""
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    is_cuda = device.type == "cuda"
    print(f"Milestone 4 Evaluation target device: {device} (CUDA sync: {is_cuda})")

    # Step 1: Load validation split
    print("Loading deterministic validation dataset split...")
    _, val_dataset = create_train_val_datasets(
        raw_dir=raw_dir,
        val_fraction=0.2,
        background_slice_ratio=0.25,
        seed=42,
        case_cache_size=1,
    )

    total_dataset_slices = len(val_dataset)
    num_slices = total_dataset_slices if max_slices is None else min(max_slices, total_dataset_slices)
    print(f"Validation cohort: {val_dataset.case_count} cases, {num_slices}/{total_dataset_slices} slices scheduled")

    # Step 2: Initialize registry and separate model-loading overhead
    print("\nInitializing ModelRegistry and measuring separate model loading times...")
    registry = ModelRegistry(default_device=device)
    model_load_times: dict[str, float] = {}

    for name in ["unet", "unetpp", "attention_unet"]:
        if is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        _ = registry.get_model(name, device=device)
        if is_cuda:
            torch.cuda.synchronize(device)
        load_sec = time.perf_counter() - t0
        model_load_times[name] = load_sec
        print(f"  Model `{name}` loaded to {device} in {load_sec:.3f} s")

    # Step 3: Configure policies and agents
    # System 4: Baseline Agent (Milestone 3: no axial context)
    policy_m3 = PolicyConfig(
        high_uncertainty_threshold=high_uncertainty_threshold,
        enable_empty_prediction_fallback=True,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=False,
    )
    agent_m3 = InferenceAgent(
        registry=registry,
        primary_model=primary_model,
        config=policy_m3,
        device=device,
    )

    # System 5: Context-Aware Agent (Milestone 4: axial spatial continuity)
    policy_m4 = PolicyConfig(
        high_uncertainty_threshold=high_uncertainty_threshold,
        enable_empty_prediction_fallback=True,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=True,
        axial_context_window=axial_context_window,
        min_neighbor_foreground_slices=min_neighbor_foreground_slices,
        fallback_when_context_unavailable=fallback_when_context_unavailable,
    )
    agent_m4 = InferenceAgent(
        registry=registry,
        primary_model=primary_model,
        config=policy_m4,
        device=device,
    )

    methods = ["unet", "unetpp", "attention_unet", "agent_baseline", "agent_context"]
    all_results: dict[str, list[SliceEvaluationResult]] = {m: [] for m in methods}
    latencies: dict[str, list[float]] = {m: [] for m in methods}
    accumulators = {m: GlobalMetricAccumulator(num_classes=4) for m in methods}

    # Group slices by case for rigorous physical continuity and zero-leakage case isolation
    # Map case_id -> list of dataset indices
    case_slice_map: dict[str, list[int]] = {}
    slices_accumulated = 0

    for idx in range(len(val_dataset)):
        if max_slices is not None and slices_accumulated >= max_slices:
            break
        rec = val_dataset.slice_index[idx]
        case_slice_map.setdefault(rec.case_id, []).append(idx)
        slices_accumulated += 1

    print(f"Evaluating across {len(case_slice_map)} cases ({slices_accumulated} slices total)...")
    start_eval = time.perf_counter()
    processed_slices = 0

    # Reset peak memory stats if on CUDA
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for case_id, dataset_indices in case_slice_map.items():
        # Phase A: Precompute primary model (U-Net++) predictions for all slices of THIS CASE.
        # This mirrors a real clinical workflow where the primary model processes the acquired volume.
        # CRITICAL SAFETY: No ground-truth masks are touched during this step!
        case_primary_preds: dict[int, SlicePrediction] = {}
        case_images: dict[int, torch.Tensor] = {}
        case_targets: dict[int, torch.Tensor] = {}
        case_slice_indices: dict[int, int] = {}

        for d_idx in dataset_indices:
            img, tgt = val_dataset[d_idx]
            rec = val_dataset.slice_index[d_idx]
            z = rec.slice_index

            case_images[z] = img
            case_targets[z] = tgt
            case_slice_indices[z] = z

            # Primary model inference (timed with explicit CUDA synchronization)
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred_primary = registry.predict_slice(primary_model, img, device=device)
            if is_cuda:
                torch.cuda.synchronize(device)
            primary_lat = (time.perf_counter() - t0) * 1000.0

            case_primary_preds[z] = pred_primary
            # Record unetpp result
            accumulators[primary_model].update(pred_primary.pred_mask, tgt)
            latencies[primary_model].append(primary_lat)
            all_results[primary_model].append(
                evaluate_slice_record(
                    method=primary_model,
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=pred_primary.pred_mask,
                    target_mask=tgt,
                    latency_ms=primary_lat,
                )
            )

        # Phase B: Evaluate remaining systems on each slice of this case
        for z in sorted(case_primary_preds.keys()):
            img = case_images[z]
            tgt = case_targets[z]

            # 1. Standalone UNet
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred_unet = registry.predict_slice("unet", img, device=device)
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_unet = (time.perf_counter() - t0) * 1000.0

            accumulators["unet"].update(pred_unet.pred_mask, tgt)
            latencies["unet"].append(lat_unet)
            all_results["unet"].append(
                evaluate_slice_record(
                    method="unet",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=pred_unet.pred_mask,
                    target_mask=tgt,
                    latency_ms=lat_unet,
                )
            )

            # 2. Standalone Attention UNet
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred_att = registry.predict_slice("attention_unet", img, device=device)
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_att = (time.perf_counter() - t0) * 1000.0

            accumulators["attention_unet"].update(pred_att.pred_mask, tgt)
            latencies["attention_unet"].append(lat_att)
            all_results["attention_unet"].append(
                evaluate_slice_record(
                    method="attention_unet",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=pred_att.pred_mask,
                    target_mask=tgt,
                    latency_ms=lat_att,
                )
            )

            # 3. Baseline Agent (Milestone 3: no axial context)
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dec_base = agent_m3.run_slice(
                image=img,
                case_id=case_id,
                slice_index=z,
            )
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_base = (time.perf_counter() - t0) * 1000.0

            accumulators["agent_baseline"].update(dec_base.final_mask, tgt)
            latencies["agent_baseline"].append(lat_base)
            all_results["agent_baseline"].append(
                evaluate_slice_record(
                    method="agent_baseline",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=dec_base.final_mask,
                    target_mask=tgt,
                    latency_ms=lat_base,
                    action=dec_base.action.value,
                    selected_model=dec_base.selected_model,
                    is_fallback=(dec_base.action == AgentAction.FALLBACK_SWITCH),
                    is_unreliable_flagged=(dec_base.action == AgentAction.FLAG_UNRELIABLE),
                    fallback_suppressed_by_context=False,
                    mean_uncertainty=dec_base.mean_uncertainty,
                    reasons=";".join(dec_base.reasons),
                )
            )

            # 4. Context-Aware Agent (Milestone 4: axial context)
            # CRITICAL SAFETY: target_mask is NOT passed to run_slice()!
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dec_ctx = agent_m4.run_slice(
                image=img,
                case_id=case_id,
                slice_index=z,
                case_primary_predictions=case_primary_preds,
                case_total_slices=155,
            )
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_ctx = (time.perf_counter() - t0) * 1000.0

            accumulators["agent_context"].update(dec_ctx.final_mask, tgt)
            latencies["agent_context"].append(lat_ctx)

            ctx_summary = dec_ctx.axial_context
            suppressed = bool(dec_ctx.metadata.get("fallback_suppressed_by_context", False))

            all_results["agent_context"].append(
                evaluate_slice_record(
                    method="agent_context",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=dec_ctx.final_mask,
                    target_mask=tgt,
                    latency_ms=lat_ctx,
                    action=dec_ctx.action.value,
                    selected_model=dec_ctx.selected_model,
                    is_fallback=(dec_ctx.action == AgentAction.FALLBACK_SWITCH),
                    is_unreliable_flagged=(dec_ctx.action == AgentAction.FLAG_UNRELIABLE),
                    fallback_suppressed_by_context=suppressed,
                    mean_uncertainty=dec_ctx.mean_uncertainty,
                    reasons=";".join(dec_ctx.reasons),
                    evaluated_neighbors=ctx_summary.evaluated_neighbors_count if ctx_summary else None,
                    unavailable_neighbors=ctx_summary.unavailable_neighbors_count if ctx_summary else None,
                    fg_neighbors=ctx_summary.foreground_neighbors_count if ctx_summary else None,
                    empty_neighbors=ctx_summary.empty_neighbors_count if ctx_summary else None,
                )
            )

            processed_slices += 1
            if processed_slices % 250 == 0 or processed_slices == num_slices:
                elapsed_min = (time.perf_counter() - start_eval) / 60.0
                print(f"  Evaluated {processed_slices}/{num_slices} slices ({elapsed_min:.2f} min elapsed)")

    # Record peak VRAM
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if is_cuda else 0.0
    )

    # Step 4: Aggregate results and compute comparative recovery & suppression statistics
    print("\nComputing aggregated summaries and comparative recovery/suppression statistics...")
    system_summaries: dict[str, Any] = {}
    for m in methods:
        g_metrics = accumulators[m].compute()
        load_time = (
            model_load_times.get(m, 0.0)
            if m in model_load_times
            else (model_load_times.get(primary_model, 0.0) + model_load_times.get("unet", 0.0))
        )
        system_summaries[m] = summarize_system_evaluation(
            method_name=m,
            results=all_results[m],
            latencies_ms=latencies[m],
            model_load_time_seconds=load_time,
            global_metrics=g_metrics,
        )

    # Cross-System Comparative Analysis
    # Compare Primary vs Baseline Agent vs Context-Aware Agent
    res_primary = { (r.case_id, r.slice_index): r for r in all_results[primary_model] }
    res_m3 = { (r.case_id, r.slice_index): r for r in all_results["agent_baseline"] }
    res_m4 = { (r.case_id, r.slice_index): r for r in all_results["agent_context"] }

    # 1. Complete Miss Recovery
    primary_misses = [k for k, r in res_primary.items() if r.is_confirmed_miss]
    m3_recovered = [k for k in primary_misses if not res_m3[k].is_confirmed_miss]
    m4_recovered = [k for k in primary_misses if not res_m4[k].is_confirmed_miss]

    # 2. Healthy Slice False Positives
    primary_fps = [k for k, r in res_primary.items() if r.is_false_positive_slice]
    m3_fps = [k for k, r in res_m3.items() if r.is_false_positive_slice]
    m4_fps = [k for k, r in res_m4.items() if r.is_false_positive_slice]

    # New false positives introduced over primary
    m3_new_fps = [k for k in m3_fps if k not in primary_fps]
    m4_new_fps = [k for k in m4_fps if k not in primary_fps]
    suppressed_fps = [k for k in m3_new_fps if k not in m4_fps]

    comparative_analysis = {
        "primary_confirmed_misses": len(primary_misses),
        "baseline_agent_recovered_misses": len(m3_recovered),
        "baseline_agent_recovery_rate_percent": (len(m3_recovered) / len(primary_misses) * 100.0) if primary_misses else 0.0,
        "context_agent_recovered_misses": len(m4_recovered),
        "context_agent_recovery_rate_percent": (len(m4_recovered) / len(primary_misses) * 100.0) if primary_misses else 0.0,
        "miss_recovery_retention_percent": (len(m4_recovered) / len(m3_recovered) * 100.0) if m3_recovered else 100.0,
        "primary_false_positive_slices": len(primary_fps),
        "baseline_agent_false_positive_slices": len(m3_fps),
        "baseline_agent_introduced_new_fps": len(m3_new_fps),
        "context_agent_false_positive_slices": len(m4_fps),
        "context_agent_introduced_new_fps": len(m4_new_fps),
        "false_positive_suppression_count": len(suppressed_fps),
        "false_positive_reduction_percent": (
            (len(m3_fps) - len(m4_fps)) / max(1, len(m3_fps) - len(primary_fps)) * 100.0
        ) if (len(m3_fps) - len(primary_fps)) > 0 else 0.0,
    }

    full_output = {
        "configuration": {
            "device": str(device),
            "peak_vram_allocated_mb": peak_vram_mb,
            "primary_model": primary_model,
            "fallback_priority": list(fallback_priority),
            "axial_context_window": axial_context_window,
            "min_neighbor_foreground_slices": min_neighbor_foreground_slices,
            "fallback_when_context_unavailable": fallback_when_context_unavailable,
            "high_uncertainty_threshold": high_uncertainty_threshold,
            "total_evaluated_slices": num_slices,
            "total_cohort_slices": total_dataset_slices,
            "validation_case_count": len(case_slice_map),
            "checkpoints": {k: str(v) for k, v in DEFAULT_CHECKPOINTS.items()},
        },
        "systems": system_summaries,
        "comparative_analysis": comparative_analysis,
    }

    # Step 5: Save artifacts strictly to results/agent_evaluation/milestone4/
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "slice_evaluations.csv"
    summary_json_path = output_dir / "evaluation_summary.json"
    report_md_path = output_dir / "report.md"

    # Save CSV
    all_flat_rows = [asdict(r) for m in methods for r in all_results[m]]
    if all_flat_rows:
        fieldnames = list(all_flat_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_flat_rows)
        print(f"Saved slice evaluations to {csv_path}")

    # Save JSON summary
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Saved evaluation summary to {summary_json_path}")

    # Generate Markdown report
    markdown_report = generate_milestone4_markdown_report(full_output)
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"Saved markdown report to {report_md_path}")

    registry.unload()
    return full_output


def generate_milestone4_markdown_report(summary_data: dict[str, Any]) -> str:
    """Format Milestone 4 results into a scientific comparative Markdown report."""
    cfg = summary_data["configuration"]
    systems = summary_data["systems"]
    comp = summary_data["comparative_analysis"]

    report_lines = [
        "# Milestone 4: Axial-Context-Aware Agent Evaluation Report",
        "",
        "## 1. Experimental Setup & Hardware Configuration",
        f"- **Validation Cases Evaluated:** {cfg['validation_case_count']}",
        f"- **Total Slices Evaluated:** {cfg['total_evaluated_slices']}",
        f"- **Evaluation Device:** `{cfg['device']}` (Peak VRAM: {cfg['peak_vram_allocated_mb']:.1f} MB)",
        f"- **Primary Architecture:** `{cfg['primary_model']}`",
        f"- **Fallback Priority:** `{cfg['fallback_priority']}`",
        f"- **Axial Context Window:** $k = \\pm {cfg['axial_context_window']}$ slices (neighborhood size: {2 * cfg['axial_context_window']} slices)",
        f"- **Min Neighbor Support:** $\\ge {cfg['min_neighbor_foreground_slices']}$ slices with primary foreground",
        f"- **Context Unavailable Fallback:** `{cfg['fallback_when_context_unavailable']}`",
        f"- **Uncertainty Threshold:** `{cfg['high_uncertainty_threshold']}`",
        "",
        "## 2. Five-System Benchmark: Segmentation & Reliability Performance",
        "",
        "### Dual-Metric Evaluation Table",
        "| System / Method | Global Micro Dice | Slice Macro Dice | Confirmed Misses ($GT>0, \\hat{Y}=0$) | Miss Rate (%) | False Pos. Slices ($GT=0, \\hat{Y}>0$) | FP Rate (%) | Latency (ms/slice) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    method_display = {
        "unet": "U-Net (Standalone)",
        "unetpp": "U-Net++ (Standalone)",
        "attention_unet": "Attention U-Net (Standalone)",
        "agent_baseline": "Agent M3 (Unconstrained Fallback)",
        "agent_context": "**Agent M4 (Axial-Context Fallback)**",
    }

    for m in ["unet", "unetpp", "attention_unet", "agent_baseline", "agent_context"]:
        s = systems[m]
        d_name = method_display.get(m, m)
        g_dice = s["global_micro_dice_foreground"]
        m_dice = s["slice_macro_mean_dice_foreground"]
        lat = s["latency_ms"]["mean"]

        report_lines.append(
            f"| {d_name} | {g_dice:.4f} | {m_dice:.4f} | "
            f"{s['confirmed_complete_misses']} / {s['foreground_slices']} | {s['confirmed_miss_rate_percent']:.2f}% | "
            f"{s['false_positive_slices']} / {s['background_slices']} | {s['false_positive_rate_percent']:.2f}% | "
            f"{lat:.1f} ms |"
        )

    report_lines.extend([
        "",
        "## 3. Comparative Spatial Continuity Analysis (M3 vs. M4)",
        f"- **Primary Model (`{cfg['primary_model']}`) Complete Misses:** {comp['primary_confirmed_misses']}",
        f"- **Baseline Agent (M3) Recovered Misses:** {comp['baseline_agent_recovered_misses']} ({comp['baseline_agent_recovery_rate_percent']:.2f}%)",
        f"- **Context-Aware Agent (M4) Recovered Misses:** {comp['context_agent_recovered_misses']} ({comp['context_agent_recovery_rate_percent']:.2f}%)",
        f"- **Miss Recovery Retention:** **{comp['miss_recovery_retention_percent']:.1f}%**",
        "",
        f"- **Primary Model False Positive Slices:** {comp['primary_false_positive_slices']}",
        f"- **Baseline Agent (M3) False Positive Slices:** {comp['baseline_agent_false_positive_slices']} (+{comp['baseline_agent_introduced_new_fps']} introduced by fallback)",
        f"- **Context-Aware Agent (M4) False Positive Slices:** {comp['context_agent_false_positive_slices']} (+{comp['context_agent_introduced_new_fps']} introduced)",
        f"- **False Positive Suppression:** **{comp['false_positive_suppression_count']} false-positive slices eliminated** ({comp['false_positive_reduction_percent']:.1f}% suppression of introduced false alarms)",
        "",
        "## 4. Latency and Computational Efficiency Profile",
        "| Architecture / System | Model Load Time (s) | Mean Latency (ms) | Median Latency (ms) | 95th Pct (ms) | Total Inference (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for m in ["unet", "unetpp", "attention_unet", "agent_baseline", "agent_context"]:
        s = systems[m]
        d_name = method_display.get(m, m)
        lat = s["latency_ms"]
        report_lines.append(
            f"| {d_name} | {s['model_load_time_seconds']:.2f} s | {lat['mean']:.1f} ms | {lat['median']:.1f} ms | {lat['p95']:.1f} ms | {lat['total_inference_time_seconds']:.2f} s |"
        )

    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4 Comparative Evaluation Harness")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR, help="Path to raw BraTS dataset")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save evaluation artifacts")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda or cpu)")
    parser.add_argument("--max-slices", type=int, default=None, help="Maximum number of slices to evaluate (e.g. for smoke testing)")
    parser.add_argument("--axial-context-window", type=int, default=2, help="Context radius k for adjacent axial slices")
    parser.add_argument("--min-neighbor-foreground-slices", type=int, default=1, help="Min neighbor slices with primary foreground")
    parser.add_argument("--primary-model", type=str, default="unetpp", help="Primary model name")
    parser.add_argument("--high-uncertainty-threshold", type=float, default=None, help="Optional normalized entropy threshold")
    parser.add_argument(
        "--fallback-when-context-unavailable",
        action="store_true",
        default=False,
        help="Permit fallback when context is unavailable (default: False)",
    )

    args = parser.parse_args()

    run_milestone4_evaluation(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        primary_model=args.primary_model,
        axial_context_window=args.axial_context_window,
        min_neighbor_foreground_slices=args.min_neighbor_foreground_slices,
        fallback_when_context_unavailable=args.fallback_when_context_unavailable,
        high_uncertainty_threshold=args.high_uncertainty_threshold,
        device_name=args.device,
        max_slices=args.max_slices,
    )


if __name__ == "__main__":
    main()
