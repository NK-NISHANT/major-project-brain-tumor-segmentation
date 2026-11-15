from __future__ import annotations

from typing import Mapping, Sequence

import torch

from src.agent.models import CANONICAL_MODEL_NAMES, ModelRegistry, normalize_model_name
from src.agent.policy import evaluate_slice_policy
from src.agent.state import (
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)


class InferenceAgent:
    """Orchestrates slice inference, dynamic on-demand fallback, and deterministic policy evaluation.

    This agent connects the ModelRegistry with the deterministic PolicyConfig:
      1. Runs primary model inference on the provided slice.
      2. If the primary model predicts zero foreground voxels and fallback is enabled,
         it actively queries the registry for candidate predictions in priority order.
         Candidates are queried lazily: execution stops as soon as a candidate produces
         non-empty foreground.
      3. Passes the resulting AgentState into `evaluate_slice_policy` to obtain an
         actionable, auditable AgentDecision.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        primary_model: str = "unetpp",
        config: PolicyConfig | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.registry = registry
        self.primary_model = normalize_model_name(primary_model)
        self.config = config if config is not None else PolicyConfig()
        self.device = torch.device(device) if device is not None else registry.default_device

    def run_slice(
        self,
        image: torch.Tensor,
        case_id: str,
        slice_index: int,
        case_primary_predictions: Mapping[int, SlicePrediction] | None = None,
        case_total_slices: int = 155,
    ) -> AgentDecision:
        """Run complete agent workflow on a single slice.

        Args:
            image: 4-channel MRI slice tensor of shape [1, 4, H, W] or [4, H, W].
            case_id: Identifier of the subject/case.
            slice_index: Axial slice number.
            case_primary_predictions: Optional mapping from slice_index to primary model predictions
                for the SAME case. Used exclusively when enable_axial_context is True.
            case_total_slices: Total axial slices in the 3D volume (default 155 for BraTS).

        Returns:
            AgentDecision detailing chosen action, selected model, final mask, uncertainty,
            and auditable reasons.
        """
        # Step 1: Run primary model inference
        primary_kwargs: dict = {
            "model_name": self.primary_model,
            "image": image,
            "device": self.device,
        }
        if self.config.primary_temperature != 1.0:
            primary_kwargs["temperature"] = self.config.primary_temperature
        primary_prediction = self.registry.predict_slice(**primary_kwargs)

        # Step 2: Compute axial spatial continuity summary if enabled
        axial_context: AxialContextSummary | None = None
        should_query_empty_fallback = (
            primary_prediction.has_empty_foreground
            and self.config.enable_empty_prediction_fallback
        )
        should_query_uncertainty_fallback = (
            self.config.enable_uncertainty_fallback
            and self.config.uncertainty_fallback_threshold is not None
            and primary_prediction.mean_foreground_entropy is not None
            and primary_prediction.mean_foreground_entropy >= self.config.uncertainty_fallback_threshold
        )

        if self.config.enable_axial_context:
            evaluated = 0
            unavailable = 0
            fg_count = 0
            empty_count = 0
            k = self.config.axial_context_window

            for delta in range(-k, k + 1):
                if delta == 0:
                    continue
                neighbor_idx = slice_index + delta
                # Boundary check: out of volume boundaries are ignored
                if 0 <= neighbor_idx < case_total_slices:
                    if (
                        case_primary_predictions is not None
                        and neighbor_idx in case_primary_predictions
                    ):
                        evaluated += 1
                        cand_neighbor = case_primary_predictions[neighbor_idx]
                        if cand_neighbor.has_empty_foreground:
                            empty_count += 1
                        else:
                            fg_count += 1
                    else:
                        unavailable += 1

            axial_context = AxialContextSummary(
                evaluated_neighbors_count=evaluated,
                unavailable_neighbors_count=unavailable,
                foreground_neighbors_count=fg_count,
                empty_neighbors_count=empty_count,
            )

            # Optimization: if context explicitly suppresses fallback, skip querying candidate models
            if should_query_empty_fallback:
                has_fg_support = (
                    axial_context.foreground_neighbors_count
                    >= self.config.min_neighbor_foreground_slices
                )
                all_unavail = (
                    axial_context.evaluated_neighbors_count == 0
                    and axial_context.unavailable_neighbors_count > 0
                )
                allow_fb = has_fg_support or (
                    all_unavail and self.config.fallback_when_context_unavailable
                )
                if not allow_fb:
                    should_query_empty_fallback = False

        candidate_predictions: dict[str, SlicePrediction] = {}
        should_query_fallback = should_query_empty_fallback or should_query_uncertainty_fallback

        # Step 3: On-demand fallback acquisition if primary predicts empty foreground or high uncertainty
        if should_query_fallback:
            # Determine candidate priority order
            if self.config.fallback_priority:
                candidate_sequence: Sequence[str] = [
                    normalize_model_name(name)
                    for name in self.config.fallback_priority
                    if normalize_model_name(name) != self.primary_model
                ]
            else:
                # Default fallback order: remaining canonical models
                candidate_sequence = [
                    name for name in CANONICAL_MODEL_NAMES if name != self.primary_model
                ]

            # Lazily query candidate models until one yields an acceptable prediction
            for cand_name in candidate_sequence:
                cand_temp = self.config.candidate_temperatures.get(cand_name, 1.0)
                cand_kwargs: dict = {
                    "model_name": cand_name,
                    "image": image,
                    "device": self.device,
                }
                if cand_temp != 1.0:
                    cand_kwargs["temperature"] = cand_temp
                cand_pred = self.registry.predict_slice(**cand_kwargs)
                candidate_predictions[cand_name] = cand_pred

                if should_query_empty_fallback and not should_query_uncertainty_fallback:
                    if not cand_pred.has_empty_foreground:
                        # Early stop: viable non-empty candidate found
                        break
                elif should_query_uncertainty_fallback:
                    if (
                        not cand_pred.has_empty_foreground
                        and cand_pred.mean_foreground_entropy is not None
                        and primary_prediction.mean_foreground_entropy is not None
                        and cand_pred.mean_foreground_entropy < primary_prediction.mean_foreground_entropy
                    ):
                        # Early stop: viable candidate with lower uncertainty found
                        break

        # Step 4: Populate agent state
        state = AgentState(
            case_id=case_id,
            slice_index=slice_index,
            primary_prediction=primary_prediction,
            candidate_predictions=candidate_predictions,
            axial_context=axial_context,
        )

        # Step 5: Evaluate deterministic policy
        return evaluate_slice_policy(state=state, config=self.config)
