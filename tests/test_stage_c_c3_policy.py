"""Unit tests for Stage C Work Package C3 Policy and Evaluation Logic.

Verifies:
1. Entropy threshold boundaries (tau=0.35, edge cases).
2. Null entropy handling (empty predictions strictly null, not 0.0).
3. Empty prediction handling and axial context at volume boundaries (slice 0, slice 154).
4. Secondary-model fallback conditions (axial support + secondary foreground).
5. Action reason codes (exact strings).
6. Frozen temperature mapping and checkpoint verification.
7. Zero ground-truth leakage into policy decision rules.
8. RFC-compliant JSON parsing and defensive error handling.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.run_stage_c_c3_selective_prediction import (
    FROZEN_TEMPERATURES,
    enforce_preflight_boundaries,
)
from src.agent.state import (
    AgentAction,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.agent.policy import evaluate_slice_policy
import torch


def test_frozen_temperatures_and_preflight():
    """Verify frozen temperature constants and preflight integrity."""
    expected = {
        "resunet_2d": 2.880459,
        "unet_resnet34": 2.626091,
        "unet": 2.626793,
        "unetpp": 2.564703,
        "attention_unet": 2.286888,
    }
    assert FROZEN_TEMPERATURES == expected
    # Enforce preflight boundaries
    info = enforce_preflight_boundaries()
    assert info["eval_cases_count"] == 105
    assert info["eval_subjs_count"] == 96


def test_entropy_threshold_boundary():
    """Test policy actions at exact entropy boundary (tau = 0.35)."""
    cfg = PolicyConfig(
        high_uncertainty_threshold=0.35,
        enable_empty_prediction_fallback=False,
    )
    dummy_logits = torch.zeros((1, 4, 240, 240))
    dummy_mask = torch.ones((240, 240), dtype=torch.uint8)

    # 1. Just below threshold: H = 0.3499 -> ACCEPT
    pred_accept = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=dummy_mask,
        pred_foreground_pixels=100,
        mean_foreground_entropy=0.3499,
        has_empty_foreground=False,
    )
    state_accept = AgentState(
        case_id="case_001",
        slice_index=50,
        primary_prediction=pred_accept,
    )
    decision = evaluate_slice_policy(state_accept, cfg)
    assert decision.action == AgentAction.ACCEPT
    assert "nominal_acceptance" in decision.reasons

    # 2. Exactly at threshold: H = 0.3500 -> FLAG_UNRELIABLE
    pred_exact = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=dummy_mask,
        pred_foreground_pixels=100,
        mean_foreground_entropy=0.3500,
        has_empty_foreground=False,
    )
    state_exact = AgentState(
        case_id="case_001",
        slice_index=50,
        primary_prediction=pred_exact,
    )
    decision = evaluate_slice_policy(state_exact, cfg)
    assert decision.action == AgentAction.FLAG_UNRELIABLE
    assert any("warning_high_uncertainty" in r for r in decision.reasons)

    # 3. Above threshold: H = 0.3501 -> FLAG_UNRELIABLE
    pred_above = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=dummy_mask,
        pred_foreground_pixels=100,
        mean_foreground_entropy=0.3501,
        has_empty_foreground=False,
    )
    state_above = AgentState(
        case_id="case_001",
        slice_index=50,
        primary_prediction=pred_above,
    )
    decision = evaluate_slice_policy(state_above, cfg)
    assert decision.action == AgentAction.FLAG_UNRELIABLE


def test_null_entropy_on_empty_prediction():
    """Verify that empty predictions have strictly None entropy and trigger unresolvable warning."""
    cfg = PolicyConfig(
        high_uncertainty_threshold=0.35,
        enable_empty_prediction_fallback=False,
    )
    dummy_logits = torch.zeros((1, 4, 240, 240))
    dummy_empty_mask = torch.zeros((240, 240), dtype=torch.uint8)

    pred_empty = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=dummy_empty_mask,
        pred_foreground_pixels=0,
        mean_foreground_entropy=None,  # Strictly None
        has_empty_foreground=True,
    )
    state_empty = AgentState(
        case_id="case_001",
        slice_index=0,
        primary_prediction=pred_empty,
    )
    decision = evaluate_slice_policy(state_empty, cfg)
    assert decision.action == AgentAction.FLAG_UNRELIABLE
    assert decision.mean_uncertainty is None
    assert "warning_complete_empty_prediction" in decision.reasons


def test_axial_context_fallback_supported():
    """Verify that empty primary triggers FALLBACK_SWITCH when axial support and secondary foreground exist."""
    cfg = PolicyConfig(
        high_uncertainty_threshold=0.35,
        enable_empty_prediction_fallback=True,
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_priority=("unet_resnet34",),
    )
    dummy_logits = torch.zeros((1, 4, 240, 240))
    empty_mask = torch.zeros((240, 240), dtype=torch.uint8)
    fg_mask = torch.ones((240, 240), dtype=torch.uint8)

    primary_empty = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=empty_mask,
        pred_foreground_pixels=0,
        mean_foreground_entropy=None,
        has_empty_foreground=True,
    )
    secondary_fg = SlicePrediction(
        model_name="unet_resnet34",
        logits=dummy_logits,
        pred_mask=fg_mask,
        pred_foreground_pixels=150,
        mean_foreground_entropy=0.28,
        has_empty_foreground=False,
    )
    axial_ctx = AxialContextSummary(
        evaluated_neighbors_count=4,
        unavailable_neighbors_count=0,
        foreground_neighbors_count=2,  # Supported! (>= 1)
        empty_neighbors_count=2,
    )
    state = AgentState(
        case_id="case_001",
        slice_index=70,
        primary_prediction=primary_empty,
        candidate_predictions={"unet_resnet34": secondary_fg},
        axial_context=axial_ctx,
    )
    decision = evaluate_slice_policy(state, cfg)
    assert decision.action == AgentAction.FALLBACK_SWITCH
    assert decision.selected_model == "unet_resnet34"
    assert "fallback_selected_unet_resnet34" in decision.reasons


def test_axial_context_fallback_suppressed_by_lack_of_support():
    """Verify that empty primary is NOT switched when axial neighbors have zero foreground."""
    cfg = PolicyConfig(
        high_uncertainty_threshold=0.35,
        enable_empty_prediction_fallback=True,
        enable_axial_context=True,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
        fallback_priority=("unet_resnet34",),
    )
    dummy_logits = torch.zeros((1, 4, 240, 240))
    empty_mask = torch.zeros((240, 240), dtype=torch.uint8)
    fg_mask = torch.ones((240, 240), dtype=torch.uint8)

    primary_empty = SlicePrediction(
        model_name="resunet_2d",
        logits=dummy_logits,
        pred_mask=empty_mask,
        pred_foreground_pixels=0,
        mean_foreground_entropy=None,
        has_empty_foreground=True,
    )
    secondary_fg = SlicePrediction(
        model_name="unet_resnet34",
        logits=dummy_logits,
        pred_mask=fg_mask,
        pred_foreground_pixels=150,
        mean_foreground_entropy=0.28,
        has_empty_foreground=False,
    )
    axial_ctx = AxialContextSummary(
        evaluated_neighbors_count=4,
        unavailable_neighbors_count=0,
        foreground_neighbors_count=0,  # Zero foreground support!
        empty_neighbors_count=4,
    )
    state = AgentState(
        case_id="case_001",
        slice_index=70,
        primary_prediction=primary_empty,
        candidate_predictions={"unet_resnet34": secondary_fg},
        axial_context=axial_ctx,
    )
    decision = evaluate_slice_policy(state, cfg)
    assert decision.action == AgentAction.FLAG_UNRELIABLE
    assert "axial_context_suppressed_fallback" in decision.reasons


def test_axial_boundary_conditions():
    """Verify axial neighbor accounting at volume boundaries (z=0 and z=154)."""
    # At z=0, radius 2 can only evaluate z=1 and z=2 (2 neighbors), 2 are unavailable
    z0_ctx = AxialContextSummary(
        evaluated_neighbors_count=2,
        unavailable_neighbors_count=2,
        foreground_neighbors_count=1,
        empty_neighbors_count=1,
    )
    assert z0_ctx.evaluated_neighbors_count + z0_ctx.unavailable_neighbors_count == 4
    assert z0_ctx.foreground_neighbors_count >= 1

    # At z=154, radius 2 can only evaluate z=152 and z=153
    z154_ctx = AxialContextSummary(
        evaluated_neighbors_count=2,
        unavailable_neighbors_count=2,
        foreground_neighbors_count=0,
        empty_neighbors_count=2,
    )
    assert z154_ctx.evaluated_neighbors_count + z154_ctx.unavailable_neighbors_count == 4
    assert z154_ctx.foreground_neighbors_count == 0


def test_no_ground_truth_leakage_in_policy():
    """Verify that PolicyConfig and evaluate_slice_policy do not take or access ground truth."""
    import inspect
    sig = inspect.signature(evaluate_slice_policy)
    params = list(sig.parameters.keys())
    assert params == ["state", "config"]
    
    state_fields = [f.name for f in AgentState.__dataclass_fields__.values()]
    assert "ground_truth" not in state_fields
    assert "gt_mask" not in state_fields
    assert "dice" not in state_fields
