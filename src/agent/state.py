from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch


class AgentAction(str, Enum):
    """Categorical action resulting from the policy evaluation of a slice."""

    ACCEPT = "ACCEPT"
    FALLBACK_SWITCH = "FALLBACK_SWITCH"
    FLAG_UNRELIABLE = "FLAG_UNRELIABLE"


@dataclass(frozen=True)
class AxialContextSummary:
    """Summary of primary model predictions across adjacent axial slices for the same case.

    Distinguishes evaluated neighbors from unavailable/missing neighbors.
    """

    evaluated_neighbors_count: int = 0
    unavailable_neighbors_count: int = 0
    foreground_neighbors_count: int = 0
    empty_neighbors_count: int = 0


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration for deterministic slice reliability policy.

    Parameters:
        high_uncertainty_threshold:
            Optional experimental threshold for normalized predictive entropy.
            If None, uncertainty-based flagging is disabled.
            NOTE: This threshold is an experimental exploratory parameter and must
            be calibrated/validated on hold-out data; it is not an established clinical fact.
        enable_empty_prediction_fallback:
            Whether to attempt fallback when the primary model predicts zero foreground voxels.
        fallback_priority:
            Explicit sequence of candidate model names defining deterministic fallback order.
            Candidates are evaluated strictly in this sequence. This order reflects
            user-configured strategy, not intrinsic or clinical superiority.
        num_classes:
            Expected number of segmentation classes (default 4 for BraTS: background + 3 tumor classes).
        enable_axial_context:
            Whether to consult primary model predictions on neighboring axial slices before
            triggering empty-primary fallback. Defaults to False for strict Milestone 3 parity.
        axial_context_window:
            Radius k of adjacent axial slices inspected within the same case ([z - k, z + k]).
        min_neighbor_foreground_slices:
            Minimum number of evaluated adjacent slices with non-empty primary foreground
            required to confirm spatial continuity support for fallback.
        fallback_when_context_unavailable:
            Fallback behavior when adjacent neighbor predictions are unavailable (e.g. isolated slices).
            If True, fallback is permitted when no adjacent slices were evaluated.
            If False (default in Milestone 5 Option A), fallback requires at least one evaluated
            neighbor with non-empty foreground.
    """

    high_uncertainty_threshold: float | None = None
    enable_empty_prediction_fallback: bool = True
    fallback_priority: tuple[str, ...] = ()
    num_classes: int = 4
    enable_axial_context: bool = False
    axial_context_window: int = 2
    min_neighbor_foreground_slices: int = 1
    fallback_when_context_unavailable: bool = False
    # Gate 6A Reliability & Uncertainty Calibration fields
    primary_temperature: float = 1.0
    candidate_temperatures: Mapping[str, float] = field(default_factory=dict)
    enable_uncertainty_fallback: bool = False
    uncertainty_fallback_threshold: float | None = None

    def __post_init__(self) -> None:
        if self.high_uncertainty_threshold is not None:
            if not (0.0 <= self.high_uncertainty_threshold <= 1.0):
                raise ValueError(
                    f"high_uncertainty_threshold must be between 0.0 and 1.0, got {self.high_uncertainty_threshold}"
                )
        if self.uncertainty_fallback_threshold is not None:
            if not (0.0 <= self.uncertainty_fallback_threshold <= 1.0):
                raise ValueError(
                    f"uncertainty_fallback_threshold must be between 0.0 and 1.0, got {self.uncertainty_fallback_threshold}"
                )
        if self.primary_temperature <= 0.0:
            raise ValueError(f"primary_temperature must be > 0, got {self.primary_temperature}")
        for name, t_val in self.candidate_temperatures.items():
            if t_val <= 0.0:
                raise ValueError(f"candidate_temperature for '{name}' must be > 0, got {t_val}")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if self.axial_context_window < 1:
            raise ValueError(f"axial_context_window must be >= 1, got {self.axial_context_window}")
        if self.min_neighbor_foreground_slices < 1:
            raise ValueError(
                f"min_neighbor_foreground_slices must be >= 1, got {self.min_neighbor_foreground_slices}"
            )


@dataclass(frozen=True)
class SlicePrediction:
    """Encapsulates single-model inference results on one 2D slice."""

    model_name: str
    logits: torch.Tensor
    pred_mask: torch.Tensor
    pred_foreground_pixels: int
    mean_foreground_entropy: float | None
    has_empty_foreground: bool


@dataclass(frozen=True)
class AgentState:
    """Complete observation available to the agent for one slice."""

    case_id: str
    slice_index: int
    primary_prediction: SlicePrediction
    candidate_predictions: dict[str, SlicePrediction] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    axial_context: AxialContextSummary | None = None


@dataclass(frozen=True)
class AgentDecision:
    """Actionable decision and routing outcome generated by the policy."""

    action: AgentAction
    selected_model: str
    final_mask: torch.Tensor
    mean_uncertainty: float | None
    has_empty_foreground: bool
    reasons: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    axial_context: AxialContextSummary | None = None
    calibrated_temperature: float = 1.0
    is_human_review_recommended: bool = False
