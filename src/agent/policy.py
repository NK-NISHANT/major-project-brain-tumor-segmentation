from __future__ import annotations

import torch

from src.agent.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.uncertainty.entropy import softmax_entropy


def extract_slice_prediction(
    model_name: str,
    logits: torch.Tensor,
    num_classes: int = 4,
) -> SlicePrediction:
    """Extract mask, pixel statistics, and normalized foreground entropy from raw logits.

    Args:
        model_name: Identifier for the model that generated logits.
        logits: Tensor of shape [1, C, H, W] containing unnormalized class logits.
        num_classes: Expected number of classes. Must match logits.shape[1].

    Returns:
        SlicePrediction with class mask, foreground pixel count, and entropy.

    Raises:
        ValueError: If logits dimensions or class channel count do not match requirements.
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits)}")

    if logits.ndim != 4:
        raise ValueError(
            f"Expected logits tensor of shape [1, C, H, W], got ndim={logits.ndim} (shape={tuple(logits.shape)})"
        )

    if logits.shape[0] != 1:
        raise ValueError(
            f"Expected batch dimension of 1 for slice prediction, got shape={tuple(logits.shape)}"
        )

    if logits.shape[1] != num_classes:
        raise ValueError(
            f"Expected {num_classes} class channels at dim 1, got {logits.shape[1]} (shape={tuple(logits.shape)})"
        )

    if logits.shape[2] <= 0 or logits.shape[3] <= 0:
        raise ValueError(
            f"Spatial dimensions (H, W) must be positive, got ({logits.shape[2]}, {logits.shape[3]})"
        )

    with torch.no_grad():
        pred_mask = torch.argmax(logits, dim=1).squeeze(0).to(torch.uint8)
        fg_mask = pred_mask > 0
        fg_pixels = int(fg_mask.sum().item())
        has_empty_fg = fg_pixels == 0

        if not has_empty_fg:
            entropy_map = softmax_entropy(logits)
            mean_fg_entropy = float(entropy_map[0][fg_mask].mean().item())
        else:
            mean_fg_entropy = None

    return SlicePrediction(
        model_name=model_name,
        logits=logits,
        pred_mask=pred_mask,
        pred_foreground_pixels=fg_pixels,
        mean_foreground_entropy=mean_fg_entropy,
        has_empty_foreground=has_empty_fg,
    )


def evaluate_slice_policy(
    state: AgentState,
    config: PolicyConfig,
) -> AgentDecision:
    """Evaluate deterministic decision rules on an AgentState.

    Rule Order:
    1. Empty Foreground Check:
       Without ground truth, an empty prediction mask is treated as a complete-empty-prediction
       warning rather than a confirmed failure.
       - If fallback is enabled:
           Inspect candidates strictly in `config.fallback_priority` order (or sorted
           candidate names if priority list is empty).
           If an available candidate produces non-empty foreground, trigger FALLBACK_SWITCH.
           Otherwise, FLAG_UNRELIABLE due to unresolvable empty prediction.
       - If fallback is disabled:
           FLAG_UNRELIABLE due to complete-empty-prediction.

    2. Uncertainty Check (if `config.high_uncertainty_threshold` is configured):
       If primary foreground entropy >= threshold:
           FLAG_UNRELIABLE with warning reason.

    3. Nominal Acceptance:
       Primary model prediction satisfies non-empty foreground and acceptable uncertainty.
       Action: ACCEPT.
    """
    primary = state.primary_prediction
    reasons: list[str] = []

    # Condition 1: Primary prediction contains zero foreground voxels
    if primary.has_empty_foreground:
        reasons.append("warning_complete_empty_prediction")

        if config.enable_empty_prediction_fallback:
            # Check axial spatial continuity if enabled
            if config.enable_axial_context and state.axial_context is not None:
                ctx = state.axial_context
                has_fg_support = ctx.foreground_neighbors_count >= config.min_neighbor_foreground_slices
                all_unavailable = ctx.evaluated_neighbors_count == 0 and ctx.unavailable_neighbors_count > 0
                allow_fallback = has_fg_support or (all_unavailable and config.fallback_when_context_unavailable)

                if not allow_fallback:
                    reasons.append("axial_context_suppressed_fallback")
                    reasons.append(f"evaluated_empty_neighbors_{ctx.empty_neighbors_count}")
                    return AgentDecision(
                        action=AgentAction.FLAG_UNRELIABLE,
                        selected_model=primary.model_name,
                        final_mask=primary.pred_mask,
                        mean_uncertainty=None,
                        has_empty_foreground=True,
                        reasons=tuple(reasons),
                        metadata={
                            "fallback_attempted": True,
                            "fallback_suppressed_by_context": True,
                            "original_primary_model": primary.model_name,
                            "primary_model": primary.model_name,
                            "primary_temperature": config.primary_temperature,
                            "primary_has_empty_foreground": True,
                            "primary_uncertainty": None,
                            "axial_context_evaluated": True,
                            "axial_support_confirmed": False,
                            "fallback_invoked": False,
                        },
                        axial_context=ctx,
                        calibrated_temperature=config.primary_temperature,
                        is_human_review_recommended=True,
                    )
                if has_fg_support:
                    reasons.append(f"axial_context_supported_by_{ctx.foreground_neighbors_count}_neighbors")
                elif all_unavailable:
                    reasons.append("axial_context_unavailable_fallback_permitted")

            # Deterministic evaluation order: use fallback_priority first, then any remaining candidates sorted
            candidate_order: list[str] = [
                name for name in config.fallback_priority if name in state.candidate_predictions
            ]
            for name in sorted(state.candidate_predictions.keys()):
                if name not in candidate_order:
                    candidate_order.append(name)

            selected_candidate: SlicePrediction | None = None
            for cand_name in candidate_order:
                cand_pred = state.candidate_predictions[cand_name]
                if not cand_pred.has_empty_foreground:
                    selected_candidate = cand_pred
                    break

            if selected_candidate is not None:
                reasons.append(f"fallback_selected_{selected_candidate.model_name}")
                if (
                    config.high_uncertainty_threshold is not None
                    and selected_candidate.mean_foreground_entropy is not None
                    and selected_candidate.mean_foreground_entropy >= config.high_uncertainty_threshold
                ):
                    reasons.append("candidate_high_uncertainty_warning")

                return AgentDecision(
                    action=AgentAction.FALLBACK_SWITCH,
                    selected_model=selected_candidate.model_name,
                    final_mask=selected_candidate.pred_mask,
                    mean_uncertainty=selected_candidate.mean_foreground_entropy,
                    has_empty_foreground=False,
                    reasons=tuple(reasons),
                    metadata={
                        "original_primary_model": primary.model_name,
                        "primary_model": primary.model_name,
                        "primary_temperature": config.primary_temperature,
                        "primary_has_empty_foreground": True,
                        "primary_uncertainty": None,
                        "axial_context_evaluated": state.axial_context is not None,
                        "axial_support_confirmed": bool(
                            state.axial_context
                            and state.axial_context.foreground_neighbors_count >= config.min_neighbor_foreground_slices
                        ),
                        "fallback_invoked": True,
                        "secondary_model": selected_candidate.model_name,
                        "secondary_temperature": config.candidate_temperatures.get(selected_candidate.model_name, 1.0),
                        "secondary_has_empty_foreground": False,
                        "secondary_uncertainty": selected_candidate.mean_foreground_entropy,
                    },
                    axial_context=state.axial_context,
                    calibrated_temperature=config.candidate_temperatures.get(selected_candidate.model_name, 1.0),
                    is_human_review_recommended=False,
                )

            reasons.append("no_viable_fallback_candidate_with_foreground")

        fallback_meta: dict = {
            "fallback_attempted": config.enable_empty_prediction_fallback,
            "original_primary_model": primary.model_name,
            "primary_model": primary.model_name,
            "primary_temperature": config.primary_temperature,
            "primary_has_empty_foreground": True,
            "primary_uncertainty": None,
            "axial_context_evaluated": state.axial_context is not None,
            "axial_support_confirmed": bool(
                state.axial_context
                and state.axial_context.foreground_neighbors_count >= config.min_neighbor_foreground_slices
            ),
            "fallback_invoked": bool(state.candidate_predictions),
        }
        if state.candidate_predictions:
            first_cand_name = next(iter(state.candidate_predictions))
            first_cand = state.candidate_predictions[first_cand_name]
            fallback_meta["secondary_model"] = first_cand_name
            fallback_meta["secondary_temperature"] = config.candidate_temperatures.get(first_cand_name, 1.0)
            fallback_meta["secondary_has_empty_foreground"] = first_cand.has_empty_foreground
            fallback_meta["secondary_uncertainty"] = first_cand.mean_foreground_entropy

        return AgentDecision(
            action=AgentAction.FLAG_UNRELIABLE,
            selected_model=primary.model_name,
            final_mask=primary.pred_mask,
            mean_uncertainty=None,
            has_empty_foreground=True,
            reasons=tuple(reasons),
            metadata=fallback_meta,
            axial_context=state.axial_context,
            calibrated_temperature=config.primary_temperature,
            is_human_review_recommended=True,
        )

    # Condition 2: Uncertainty evaluation on predicted foreground
    # 2a. Uncertainty-gated fallback (if enabled and candidate has lower uncertainty)
    if config.enable_uncertainty_fallback and config.uncertainty_fallback_threshold is not None:
        if (
            primary.mean_foreground_entropy is not None
            and primary.mean_foreground_entropy >= config.uncertainty_fallback_threshold
        ):
            reasons.append(
                f"uncertainty_{primary.mean_foreground_entropy:.4f}_exceeds_threshold_{config.uncertainty_fallback_threshold:.4f}"
            )
            # Check candidate models in priority order
            candidate_order = [
                name for name in config.fallback_priority if name in state.candidate_predictions
            ]
            for name in sorted(state.candidate_predictions.keys()):
                if name not in candidate_order:
                    candidate_order.append(name)

            best_cand: SlicePrediction | None = None
            for cand_name in candidate_order:
                cand_pred = state.candidate_predictions[cand_name]
                if (
                    not cand_pred.has_empty_foreground
                    and cand_pred.mean_foreground_entropy is not None
                    and cand_pred.mean_foreground_entropy < primary.mean_foreground_entropy
                ):
                    best_cand = cand_pred
                    break

            if best_cand is not None:
                reasons.append(
                    f"uncertainty_fallback_selected_{best_cand.model_name}_{best_cand.mean_foreground_entropy:.4f}"
                )
                return AgentDecision(
                    action=AgentAction.FALLBACK_SWITCH,
                    selected_model=best_cand.model_name,
                    final_mask=best_cand.pred_mask,
                    mean_uncertainty=best_cand.mean_foreground_entropy,
                    has_empty_foreground=False,
                    reasons=tuple(reasons),
                    metadata={
                        "original_primary_model": primary.model_name,
                        "primary_uncertainty": primary.mean_foreground_entropy,
                        "fallback_type": "uncertainty_gated",
                    },
                    axial_context=state.axial_context,
                    calibrated_temperature=config.candidate_temperatures.get(best_cand.model_name, 1.0),
                    is_human_review_recommended=False,
                )
            reasons.append("no_candidate_reduced_uncertainty")

    # 2b. High uncertainty warning / human review flagging
    threshold_to_check = config.high_uncertainty_threshold
    if threshold_to_check is not None:
        if primary.mean_foreground_entropy is not None and primary.mean_foreground_entropy >= threshold_to_check:
            reasons.append(
                f"warning_high_uncertainty_{primary.mean_foreground_entropy:.4f}_exceeds_{threshold_to_check:.4f}"
            )
            return AgentDecision(
                action=AgentAction.FLAG_UNRELIABLE,
                selected_model=primary.model_name,
                final_mask=primary.pred_mask,
                mean_uncertainty=primary.mean_foreground_entropy,
                has_empty_foreground=False,
                reasons=tuple(reasons),
                metadata={
                    "original_primary_model": primary.model_name,
                    "primary_model": primary.model_name,
                    "primary_temperature": config.primary_temperature,
                    "primary_has_empty_foreground": False,
                    "primary_uncertainty": primary.mean_foreground_entropy,
                    "axial_context_evaluated": state.axial_context is not None,
                    "axial_support_confirmed": False,
                    "fallback_invoked": False,
                    "high_uncertainty_flagged": True,
                },
                axial_context=state.axial_context,
                calibrated_temperature=config.primary_temperature,
                is_human_review_recommended=True,
            )

    # Condition 3: Nominal acceptance
    reasons.append("nominal_acceptance")
    return AgentDecision(
        action=AgentAction.ACCEPT,
        selected_model=primary.model_name,
        final_mask=primary.pred_mask,
        mean_uncertainty=primary.mean_foreground_entropy,
        has_empty_foreground=False,
        reasons=tuple(reasons),
        metadata={
            "original_primary_model": primary.model_name,
            "primary_model": primary.model_name,
            "primary_temperature": config.primary_temperature,
            "primary_has_empty_foreground": False,
            "primary_uncertainty": primary.mean_foreground_entropy,
            "axial_context_evaluated": state.axial_context is not None,
            "axial_support_confirmed": False,
            "fallback_invoked": False,
        },
        axial_context=state.axial_context,
        calibrated_temperature=config.primary_temperature,
        is_human_review_recommended=False,
    )
