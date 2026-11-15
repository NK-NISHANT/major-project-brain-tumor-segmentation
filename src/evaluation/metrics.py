from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from monai.metrics import compute_hausdorff_distance


@dataclass
class SegmentationMetrics:
    mean_dice: float
    mean_iou: float
    per_class_dice: list[float]
    per_class_iou: list[float]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mean_dice": self.mean_dice,
            "mean_iou": self.mean_iou,
            "per_class_dice": self.per_class_dice,
            "per_class_iou": self.per_class_iou,
        }


@dataclass
class CompartmentHD95:
    """HD95 evaluation record for a single anatomical compartment."""
    raw_hd95: float          # float('nan') if both empty, float('inf') if miss/false alarm, else measured mm
    bounded_hd95: float      # 0.0 if both empty, max_penalty if miss/false alarm, else min(raw, max_penalty)
    status: str              # 'valid', 'both_empty', 'complete_miss', 'false_alarm'
    pred_voxel_count: int
    gt_voxel_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_hd95": self.raw_hd95,
            "bounded_hd95": self.bounded_hd95,
            "status": self.status,
            "pred_voxel_count": self.pred_voxel_count,
            "gt_voxel_count": self.gt_voxel_count,
        }


@dataclass
class VolumeHD95Metrics:
    """3D Volumetric HD95 metrics across BraTS tumor sub-compartments."""
    wt: CompartmentHD95
    tc: CompartmentHD95
    et: CompartmentHD95
    spacing: tuple[float, float, float]
    max_penalty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "wt": self.wt.as_dict(),
            "tc": self.tc.as_dict(),
            "et": self.et.as_dict(),
            "wt_raw_hd95": self.wt.raw_hd95,
            "wt_bounded_hd95": self.wt.bounded_hd95,
            "wt_status": self.wt.status,
            "tc_raw_hd95": self.tc.raw_hd95,
            "tc_bounded_hd95": self.tc.bounded_hd95,
            "tc_status": self.tc.status,
            "et_raw_hd95": self.et.raw_hd95,
            "et_bounded_hd95": self.et.bounded_hd95,
            "et_status": self.et.status,
            "spacing": list(self.spacing),
            "max_penalty": self.max_penalty,
        }


def compute_compartment_hd95(
    pred_mask: np.ndarray | torch.Tensor,
    gt_mask: np.ndarray | torch.Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_penalty: float = 373.12,
    percentile: float = 95.0,
) -> CompartmentHD95:
    """Compute 95th Percentile Hausdorff Distance for a binary 3D compartment with empty-mask handling."""
    if isinstance(pred_mask, torch.Tensor):
        pred_np = pred_mask.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_mask)

    if isinstance(gt_mask, torch.Tensor):
        gt_np = gt_mask.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_mask)

    pred_bool = pred_np > 0
    gt_bool = gt_np > 0

    pred_count = int(np.sum(pred_bool))
    gt_count = int(np.sum(gt_bool))

    # Case 1: Both masks empty (True Negative agreement on absence)
    if pred_count == 0 and gt_count == 0:
        return CompartmentHD95(
            raw_hd95=float("nan"),
            bounded_hd95=0.0,
            status="both_empty",
            pred_voxel_count=0,
            gt_voxel_count=0,
        )

    # Case 2: Prediction empty, Ground Truth present (Complete Miss)
    if pred_count == 0 and gt_count > 0:
        return CompartmentHD95(
            raw_hd95=float("inf"),
            bounded_hd95=float(max_penalty),
            status="complete_miss",
            pred_voxel_count=0,
            gt_voxel_count=gt_count,
        )

    # Case 3: Prediction present, Ground Truth empty (False Alarm)
    if pred_count > 0 and gt_count == 0:
        return CompartmentHD95(
            raw_hd95=float("inf"),
            bounded_hd95=float(max_penalty),
            status="false_alarm",
            pred_voxel_count=pred_count,
            gt_voxel_count=0,
        )

    # Case 4: Both masks non-empty -> compute via MONAI
    pred_t = torch.from_numpy(pred_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    gt_t = torch.from_numpy(gt_bool.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    hd_tensor = compute_hausdorff_distance(
        y_pred=pred_t,
        y=gt_t,
        include_background=False,
        distance_metric="euclidean",
        percentile=percentile,
        directed=False,
        spacing=spacing,
    )
    raw_val = float(hd_tensor.item())
    bounded_val = min(raw_val, float(max_penalty))

    return CompartmentHD95(
        raw_hd95=raw_val,
        bounded_hd95=bounded_val,
        status="valid",
        pred_voxel_count=pred_count,
        gt_voxel_count=gt_count,
    )


def compute_volume_hd95(
    pred_volume: np.ndarray | torch.Tensor,
    gt_volume: np.ndarray | torch.Tensor,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_penalty: float = 373.12,
    percentile: float = 95.0,
) -> VolumeHD95Metrics:
    """Compute 3D Volumetric HD95 for BraTS hierarchical compartments (WT, TC, ET).

    Compartments:
      - Whole Tumor (WT): union of classes {1, 2, 3} (Necrotic Core + Edema + Enhancing Tumor)
      - Tumor Core (TC): union of classes {1, 3} (Necrotic Core + Enhancing Tumor)
      - Enhancing Tumor (ET): class {3}
    """
    if isinstance(pred_volume, torch.Tensor):
        pred_np = pred_volume.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_volume)

    if isinstance(gt_volume, torch.Tensor):
        gt_np = gt_volume.detach().cpu().numpy()
    else:
        gt_np = np.asarray(gt_volume)

    if pred_np.shape != gt_np.shape:
        raise ValueError(f"Shape mismatch: pred {pred_np.shape} vs gt {gt_np.shape}")
    if pred_np.ndim != 3:
        raise ValueError(f"Expected 3D volumes [H, W, D], got ndim={pred_np.ndim}")

    # Hierarchical masks
    wt_pred = np.isin(pred_np, [1, 2, 3])
    wt_gt = np.isin(gt_np, [1, 2, 3])

    tc_pred = np.isin(pred_np, [1, 3])
    tc_gt = np.isin(gt_np, [1, 3])

    et_pred = pred_np == 3
    et_gt = gt_np == 3

    wt_res = compute_compartment_hd95(wt_pred, wt_gt, spacing=spacing, max_penalty=max_penalty, percentile=percentile)
    tc_res = compute_compartment_hd95(tc_pred, tc_gt, spacing=spacing, max_penalty=max_penalty, percentile=percentile)
    et_res = compute_compartment_hd95(et_pred, et_gt, spacing=spacing, max_penalty=max_penalty, percentile=percentile)

    return VolumeHD95Metrics(
        wt=wt_res,
        tc=tc_res,
        et=et_res,
        spacing=spacing,
        max_penalty=max_penalty,
    )



class SegmentationMetricAccumulator:
    """Accumulate Dice and IoU from predicted class IDs over many batches."""

    def __init__(self, num_classes: int = 4, ignore_background: bool = True) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2.")
        self.num_classes = num_classes
        self.ignore_background = ignore_background
        self.true_positive = torch.zeros(num_classes, dtype=torch.float64)
        self.false_positive = torch.zeros(num_classes, dtype=torch.float64)
        self.false_negative = torch.zeros(num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits with shape [N, C, H, W], got {tuple(logits.shape)}.")
        if target.ndim != 3:
            raise ValueError(f"Expected target with shape [N, H, W], got {tuple(target.shape)}.")

        prediction = torch.argmax(logits.detach(), dim=1).to("cpu")
        target_cpu = target.detach().to("cpu")

        for class_id in range(self.num_classes):
            predicted_class = prediction == class_id
            target_class = target_cpu == class_id
            self.true_positive[class_id] += torch.logical_and(predicted_class, target_class).sum()
            self.false_positive[class_id] += torch.logical_and(predicted_class, ~target_class).sum()
            self.false_negative[class_id] += torch.logical_and(~predicted_class, target_class).sum()

    def compute(self) -> SegmentationMetrics:
        dice_scores: list[float] = []
        iou_scores: list[float] = []

        for class_id in range(self.num_classes):
            tp = self.true_positive[class_id]
            fp = self.false_positive[class_id]
            fn = self.false_negative[class_id]

            dice_denominator = (2.0 * tp) + fp + fn
            iou_denominator = tp + fp + fn

            dice = float((2.0 * tp / dice_denominator).item()) if dice_denominator > 0 else float("nan")
            iou = float((tp / iou_denominator).item()) if iou_denominator > 0 else float("nan")
            dice_scores.append(dice)
            iou_scores.append(iou)

        primary_start = 1 if self.ignore_background else 0
        mean_dice = _nanmean(dice_scores[primary_start:])
        mean_iou = _nanmean(iou_scores[primary_start:])
        return SegmentationMetrics(
            mean_dice=mean_dice,
            mean_iou=mean_iou,
            per_class_dice=dice_scores,
            per_class_iou=iou_scores,
        )


def _nanmean(values: list[float]) -> float:
    valid_values = [value for value in values if value == value]
    if not valid_values:
        return float("nan")
    return float(sum(valid_values) / len(valid_values))


__all__ = [
    "SegmentationMetricAccumulator",
    "SegmentationMetrics",
    "CompartmentHD95",
    "VolumeHD95Metrics",
    "compute_compartment_hd95",
    "compute_volume_hd95",
]
