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
    CaseFiles,
    _load_and_normalize_case,
    discover_valid_cases,
    split_cases,
)
from src.evaluation.metrics import _nanmean

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FULL_VOLUME_OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "agent_evaluation" / "milestone7_full_volume"
)


@dataclass(frozen=True)
class SliceEvaluationResult:
    """Detailed evaluation record for one axial slice prediction against ground truth."""

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
    """Evaluate a discrete segmentation mask against ground truth and categorize failure modes."""
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


def sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float('nan'), float('inf'), float('-inf') to None for RFC 8259 JSON compliance."""
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def run_full_volume_evaluation(
    raw_dir: Path = DEFAULT_RAW_DATA_DIR,
    output_dir: Path = DEFAULT_FULL_VOLUME_OUTPUT_DIR,
    device_name: str | None = None,
    max_cases: int | None = None,
    max_slices_per_case: int | None = None,
    primary_model: str = "unetpp",
    fallback_priority: tuple[str, ...] = ("unet",),
    high_uncertainty_threshold: float | None = None,
    registry: ModelRegistry | None = None,
    cases: Sequence[CaseFiles] | None = None,
) -> dict[str, Any]:
    """Execute complete-volume (155-slice) evaluation across validation cases.

    Evaluates 6 systems under strictly identical preprocessing, weights, and ascending axial slice ordering:
      1. Standalone U-Net
      2. Standalone U-Net++
      3. Standalone Attention U-Net
      4. Agent Baseline (M3: unconstrained fallback)
      5. Agent Option A (Default: k=2, min_neighbor_fg=1, fallback_when_context_unavailable=False)
      6. Agent Option B (Trial: k=3, min_neighbor_fg=1, fallback_when_context_unavailable=False)
    """
    device = torch.device(
        device_name if device_name is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    is_cuda = device.type == "cuda"
    print(f"Full-Volume Evaluation target device: {device} (CUDA sync: {is_cuda})")

    # Step 1: Discover validation cases
    if cases is not None:
        val_cases = list(cases)
        print(f"Using explicitly supplied validation cases: {len(val_cases)} cases")
    else:
        print("Discovering and splitting BraTS 2023 cases...")
        all_cases = discover_valid_cases(raw_dir)
        _, val_cases = split_cases(all_cases, val_fraction=0.2, seed=42)

    if max_cases is not None:
        val_cases = val_cases[:max_cases]
    print(f"Validation cases to evaluate: {len(val_cases)} cases")

    # Step 2: Initialize registry and measure separate model loading times
    print("\nInitializing ModelRegistry and measuring separate model loading times...")
    own_registry = registry is None
    if own_registry:
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
    # System 4: Baseline Agent (Milestone 3: unconstrained)
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

    # System 5: Context-Aware Agent Option A (Milestone 5 Default: k=2, unavail=False)
    policy_option_a = PolicyConfig(
        high_uncertainty_threshold=high_uncertainty_threshold,
        enable_empty_prediction_fallback=True,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
    )
    agent_option_a = InferenceAgent(
        registry=registry,
        primary_model=primary_model,
        config=policy_option_a,
        device=device,
    )

    # System 6: Context-Aware Agent Option B (Milestone 6 Trial: k=3, unavail=False)
    policy_option_b = PolicyConfig(
        high_uncertainty_threshold=high_uncertainty_threshold,
        enable_empty_prediction_fallback=True,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=True,
        axial_context_window=3,
        min_neighbor_foreground_slices=1,
        fallback_when_context_unavailable=False,
    )
    agent_option_b = InferenceAgent(
        registry=registry,
        primary_model=primary_model,
        config=policy_option_b,
        device=device,
    )

    methods = [
        "unet",
        "unetpp",
        "attention_unet",
        "agent_baseline",
        "agent_option_a",
        "agent_option_b",
    ]
    all_results: dict[str, list[SliceEvaluationResult]] = {m: [] for m in methods}
    latencies: dict[str, list[float]] = {m: [] for m in methods}
    accumulators = {m: GlobalMetricAccumulator(num_classes=4) for m in methods}

    total_slices_scheduled = len(val_cases) * 155 if max_slices_per_case is None else len(val_cases) * max_slices_per_case
    print(f"\nBeginning full-volume evaluation ({total_slices_scheduled} slices scheduled across {len(val_cases)} cases)...")

    start_eval = time.perf_counter()
    processed_slices = 0

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    for case_idx, case in enumerate(val_cases):
        case_id = case.case_id
        # Load one case volume into CPU RAM (~152 MB)
        cached_volume = _load_and_normalize_case(case)
        total_z = cached_volume.image.shape[3]  # Exactly 155 for complete BraTS cases
        slices_to_eval = list(range(total_z))
        if max_slices_per_case is not None:
            slices_to_eval = slices_to_eval[:max_slices_per_case]

        # Extract image and target tensors for this case
        case_images: dict[int, torch.Tensor] = {}
        case_targets: dict[int, torch.Tensor] = {}
        case_primary_preds: dict[int, SlicePrediction] = {}

        # Phase A: Complete Primary Prediction Mapping across all 155 slices
        # Evaluated strictly in ascending axial slice order (z = 0, 1, ..., 154)
        for z in slices_to_eval:
            img = torch.from_numpy(cached_volume.image[:, :, :, z])
            tgt = torch.from_numpy(cached_volume.mask[:, :, z].astype(np.int64))

            case_images[z] = img
            case_targets[z] = tgt

            # Primary model inference (timed with explicit CUDA synchronization)
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            pred_primary = registry.predict_slice(primary_model, img, device=device)
            if is_cuda:
                torch.cuda.synchronize(device)
            primary_lat = (time.perf_counter() - t0) * 1000.0

            case_primary_preds[z] = pred_primary
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

        # Phase B: Evaluate remaining systems on each slice in strictly ascending axial order
        for z in slices_to_eval:
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

            # 4. Context-Aware Agent Option A (Default: k=2)
            # CRITICAL SAFETY: target_mask is NOT passed to run_slice()!
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dec_opt_a = agent_option_a.run_slice(
                image=img,
                case_id=case_id,
                slice_index=z,
                case_primary_predictions=case_primary_preds,
                case_total_slices=total_z,
            )
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_opt_a = (time.perf_counter() - t0) * 1000.0

            accumulators["agent_option_a"].update(dec_opt_a.final_mask, tgt)
            latencies["agent_option_a"].append(lat_opt_a)
            ctx_a = dec_opt_a.axial_context
            suppressed_a = bool(dec_opt_a.metadata.get("fallback_suppressed_by_context", False))

            all_results["agent_option_a"].append(
                evaluate_slice_record(
                    method="agent_option_a",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=dec_opt_a.final_mask,
                    target_mask=tgt,
                    latency_ms=lat_opt_a,
                    action=dec_opt_a.action.value,
                    selected_model=dec_opt_a.selected_model,
                    is_fallback=(dec_opt_a.action == AgentAction.FALLBACK_SWITCH),
                    is_unreliable_flagged=(dec_opt_a.action == AgentAction.FLAG_UNRELIABLE),
                    fallback_suppressed_by_context=suppressed_a,
                    mean_uncertainty=dec_opt_a.mean_uncertainty,
                    reasons=";".join(dec_opt_a.reasons),
                    evaluated_neighbors=ctx_a.evaluated_neighbors_count if ctx_a else None,
                    unavailable_neighbors=ctx_a.unavailable_neighbors_count if ctx_a else None,
                    fg_neighbors=ctx_a.foreground_neighbors_count if ctx_a else None,
                    empty_neighbors=ctx_a.empty_neighbors_count if ctx_a else None,
                )
            )

            # 5. Context-Aware Agent Option B (Trial: k=3)
            # CRITICAL SAFETY: target_mask is NOT passed to run_slice()!
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            dec_opt_b = agent_option_b.run_slice(
                image=img,
                case_id=case_id,
                slice_index=z,
                case_primary_predictions=case_primary_preds,
                case_total_slices=total_z,
            )
            if is_cuda:
                torch.cuda.synchronize(device)
            lat_opt_b = (time.perf_counter() - t0) * 1000.0

            accumulators["agent_option_b"].update(dec_opt_b.final_mask, tgt)
            latencies["agent_option_b"].append(lat_opt_b)
            ctx_b = dec_opt_b.axial_context
            suppressed_b = bool(dec_opt_b.metadata.get("fallback_suppressed_by_context", False))

            all_results["agent_option_b"].append(
                evaluate_slice_record(
                    method="agent_option_b",
                    case_id=case_id,
                    slice_index=z,
                    pred_mask=dec_opt_b.final_mask,
                    target_mask=tgt,
                    latency_ms=lat_opt_b,
                    action=dec_opt_b.action.value,
                    selected_model=dec_opt_b.selected_model,
                    is_fallback=(dec_opt_b.action == AgentAction.FALLBACK_SWITCH),
                    is_unreliable_flagged=(dec_opt_b.action == AgentAction.FLAG_UNRELIABLE),
                    fallback_suppressed_by_context=suppressed_b,
                    mean_uncertainty=dec_opt_b.mean_uncertainty,
                    reasons=";".join(dec_opt_b.reasons),
                    evaluated_neighbors=ctx_b.evaluated_neighbors_count if ctx_b else None,
                    unavailable_neighbors=ctx_b.unavailable_neighbors_count if ctx_b else None,
                    fg_neighbors=ctx_b.foreground_neighbors_count if ctx_b else None,
                    empty_neighbors=ctx_b.empty_neighbors_count if ctx_b else None,
                )
            )

            processed_slices += 1

        # Release case volume from CPU RAM
        del cached_volume
        case_images.clear()
        case_targets.clear()
        case_primary_preds.clear()

        elapsed_min = (time.perf_counter() - start_eval) / 60.0
        print(
            f"  [Case {case_idx + 1:2d}/{len(val_cases):2d}] {case_id} evaluated "
            f"({len(slices_to_eval)} slices, {processed_slices}/{total_slices_scheduled} total, {elapsed_min:.2f} min elapsed)"
        )

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

    # Cross-policy comparative analysis
    primary_misses = system_summaries[primary_model]["confirmed_complete_misses"]
    base_misses = system_summaries["agent_baseline"]["confirmed_complete_misses"]
    opt_a_misses = system_summaries["agent_option_a"]["confirmed_complete_misses"]
    opt_b_misses = system_summaries["agent_option_b"]["confirmed_complete_misses"]

    primary_fps = system_summaries[primary_model]["false_positive_slices"]
    base_fps = system_summaries["agent_baseline"]["false_positive_slices"]
    opt_a_fps = system_summaries["agent_option_a"]["false_positive_slices"]
    opt_b_fps = system_summaries["agent_option_b"]["false_positive_slices"]

    comparative_analysis: dict[str, Any] = {
        "primary_confirmed_misses": primary_misses,
        "baseline_agent_recovered_misses": primary_misses - base_misses,
        "option_a_recovered_misses": primary_misses - opt_a_misses,
        "option_b_recovered_misses": primary_misses - opt_b_misses,
        "option_b_additional_misses_over_option_a": opt_a_misses - opt_b_misses,
        "primary_false_positive_slices": primary_fps,
        "baseline_agent_false_positive_slices": base_fps,
        "baseline_agent_introduced_new_fps": base_fps - primary_fps,
        "option_a_false_positive_slices": opt_a_fps,
        "option_a_introduced_new_fps": opt_a_fps - primary_fps,
        "option_a_suppression_vs_m3": base_fps - opt_a_fps,
        "option_b_false_positive_slices": opt_b_fps,
        "option_b_introduced_new_fps": opt_b_fps - primary_fps,
        "option_b_suppression_vs_m3": base_fps - opt_b_fps,
        "option_b_additional_fps_over_option_a": opt_b_fps - opt_a_fps,
    }

    full_output: dict[str, Any] = {
        "evaluation_cohort": "full_volume_complete_155_slices",
        "configuration": {
            "device": str(device),
            "peak_vram_allocated_mb": peak_vram_mb,
            "primary_model": primary_model,
            "fallback_priority": list(fallback_priority),
            "total_evaluated_slices": processed_slices,
            "total_cohort_slices": len(val_cases) * 155,
            "validation_case_count": len(val_cases),
            "slices_per_case": 155 if max_slices_per_case is None else max_slices_per_case,
            "checkpoints": {k: str(v) for k, v in DEFAULT_CHECKPOINTS.items()},
        },
        "systems": system_summaries,
        "comparative_analysis": comparative_analysis,
    }

    # Step 5: Save artifacts strictly to results/agent_evaluation/milestone7_full_volume/
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
        print(f"Saved full-volume slice evaluations to {csv_path}")

    # Save JSON summary (strict RFC 8259 compliant: NaNs serialized as null)
    sanitized_output = sanitize_for_json(full_output)
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(sanitized_output, f, indent=2)
    print(f"Saved full-volume evaluation summary to {summary_json_path}")

    # Generate Markdown report
    markdown_report = generate_full_volume_markdown_report(full_output)
    with report_md_path.open("w", encoding="utf-8") as f:
        f.write(markdown_report)
    print(f"Saved full-volume markdown report to {report_md_path}")

    if own_registry:
        registry.unload()
    return full_output


def generate_full_volume_markdown_report(summary_data: dict[str, Any]) -> str:
    """Format Milestone 7 results into a scientific comparative Markdown report."""
    cfg = summary_data["configuration"]
    systems = summary_data["systems"]
    comp = summary_data["comparative_analysis"]

    report_lines = [
        "# Milestone 7: Full-Volume (155-Slice) Agent Context Evaluation Report",
        "",
        "## 1. Experimental Setup & Hardware Configuration",
        f"- **Evaluation Cohort:** Full-Volume Contiguous (155 axial slices per patient volume)",
        f"- **Validation Cases Evaluated:** {cfg['validation_case_count']}",
        f"- **Total Slices Evaluated:** {cfg['total_evaluated_slices']}",
        f"- **Evaluation Device:** `{cfg['device']}` (Peak VRAM: {cfg['peak_vram_allocated_mb']:.1f} MB)",
        f"- **Primary Architecture:** `{cfg['primary_model']}`",
        f"- **Fallback Priority:** `{cfg['fallback_priority']}`",
        f"- **Evaluated Policies:** Standalone Models, Agent M3, Agent Option A (k=2), Agent Option B (k=3)",
        "",
        "## 2. Six-System Benchmark: Segmentation & Reliability Performance",
        "",
        "### Complete-Volume Evaluation Table",
        "| System / Method | Global Micro Dice | Slice Macro Dice | Confirmed Misses ($GT>0, \\hat{Y}=0$) | Miss Rate (%) | False Pos. Slices ($GT=0, \\hat{Y}>0$) | Healthy Specificity (%) | Latency (ms/slice) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    method_display = {
        "unet": "U-Net (Standalone)",
        "unetpp": "U-Net++ (Standalone)",
        "attention_unet": "Attention U-Net (Standalone)",
        "agent_baseline": "Agent M3 (Unconstrained Fallback)",
        "agent_option_a": "**Agent Option A (k=2, Default)**",
        "agent_option_b": "**Agent Option B (k=3, Trial)**",
    }

    for m in [
        "unet",
        "unetpp",
        "attention_unet",
        "agent_baseline",
        "agent_option_a",
        "agent_option_b",
    ]:
        s = systems[m]
        d_name = method_display.get(m, m)
        g_dice = s["global_micro_dice_foreground"]
        m_dice = s["slice_macro_mean_dice_foreground"]
        lat = s["latency_ms"]["mean"]
        spec = s["true_negative_rate_percent"]

        g_dice_str = f"{g_dice:.4f}" if g_dice is not None and not np.isnan(g_dice) else "N/A"
        m_dice_str = f"{m_dice:.4f}" if m_dice is not None and not np.isnan(m_dice) else "N/A"
        spec_str = f"{spec:.2f}%" if spec is not None and not np.isnan(spec) else "N/A"
        lat_str = f"{lat:.1f} ms" if lat is not None and not np.isnan(lat) else "N/A"

        report_lines.append(
            f"| {d_name} | {g_dice_str} | {m_dice_str} | "
            f"{s['confirmed_complete_misses']} / {s['foreground_slices']} | {s['confirmed_miss_rate_percent']:.2f}% | "
            f"{s['false_positive_slices']} / {s['background_slices']} | {spec_str} | "
            f"{lat_str} |"
        )

    report_lines.extend([
        "",
        "## 3. Spatial Continuity Comparative Analysis (Option A vs. Option B)",
        f"- **Primary Model (`{cfg['primary_model']}`) Complete Misses:** {comp['primary_confirmed_misses']}",
        f"- **Option A (k=2) Recovered Misses:** {comp['option_a_recovered_misses']}",
        f"- **Option B (k=3) Recovered Misses:** {comp['option_b_recovered_misses']} (+{comp['option_b_additional_misses_over_option_a']} additional misses recovered over Option A)",
        "",
        f"- **Primary Model False Positive Slices:** {comp['primary_false_positive_slices']}",
        f"- **Baseline Agent (M3) False Positive Slices:** {comp['baseline_agent_false_positive_slices']} (+{comp['baseline_agent_introduced_new_fps']} introduced)",
        f"- **Option A False Positive Slices:** {comp['option_a_false_positive_slices']} ({comp['option_a_suppression_vs_m3']} false alarms suppressed vs M3)",
        f"- **Option B False Positive Slices:** {comp['option_b_false_positive_slices']} ({comp['option_b_suppression_vs_m3']} false alarms suppressed vs M3)",
        f"- **Option B Additional False Positives over Option A:** +{comp['option_b_additional_fps_over_option_a']} slices",
        "",
        "## 4. Latency and Computational Profile",
        "| Architecture / System | Model Load Time (s) | Mean Latency (ms) | Median Latency (ms) | 95th Pct (ms) | Total Inference (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ])

    for m in [
        "unet",
        "unetpp",
        "attention_unet",
        "agent_baseline",
        "agent_option_a",
        "agent_option_b",
    ]:
        s = systems[m]
        d_name = method_display.get(m, m)
        lat = s["latency_ms"]
        report_lines.append(
            f"| {d_name} | {s['model_load_time_seconds']:.2f} s | {lat['mean']:.1f} ms | {lat['median']:.1f} ms | {lat['p95']:.1f} ms | {lat['total_inference_time_seconds']:.2f} s |"
        )

    return "\n".join(report_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Milestone 7 Full-Volume (155-Slice) Comparative Evaluation Harness"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DATA_DIR,
        help="Path to raw BraTS dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FULL_VOLUME_OUTPUT_DIR,
        help="Directory to save evaluation artifacts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution device (cuda or cpu)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Maximum number of cases to evaluate (e.g. for smoke testing)",
    )
    parser.add_argument(
        "--max-slices-per-case",
        type=int,
        default=None,
        help="Maximum number of slices per case (e.g. for smoke testing)",
    )
    parser.add_argument(
        "--primary-model",
        type=str,
        default="unetpp",
        help="Primary model identifier",
    )
    parser.add_argument(
        "--high-uncertainty-threshold",
        type=float,
        default=None,
        help="Optional normalized entropy threshold",
    )

    args = parser.parse_args()

    run_full_volume_evaluation(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        device_name=args.device,
        max_cases=args.max_cases,
        max_slices_per_case=args.max_slices_per_case,
        primary_model=args.primary_model,
        high_uncertainty_threshold=args.high_uncertainty_threshold,
    )


if __name__ == "__main__":
    main()
