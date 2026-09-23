from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


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


__all__ = ["SegmentationMetricAccumulator", "SegmentationMetrics"]
