"""Interactive Clinical Research Showcase Dashboard for Brain Tumor Segmentation & Reliability Agent.

B.Tech CSE Major Project:
Reliability-Aware AI Agent for Brain Tumor Segmentation
Workspace: D:\\MajorProject
Governing Blueprint: MASTER_PROJECT_BLUEPRINT_v2.0.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping

# Ensure project root is on sys.path for robust standalone and sub-process execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch

from src.agent.calibration import (
    DEFAULT_C2_CALIBRATION_JSON,
    STAGE_C2_REFERENCE_TEMPERATURES,
    StageCCalibrationConfig,
    load_stage_c_calibration,
)
from src.agent.models import (
    CANDIDATE_MODEL_NAMES,
    CANONICAL_MODEL_NAMES,
    CheckpointProfile,
    ModelRegistry,
    normalize_model_name,
)
from src.agent.runner import (
    InferenceAgent,
    create_stage_c_agent,
    create_stage_c_policy_config,
)
from src.agent.state import (
    STAGE_C_ALLOWED_MODELS,
    STAGE_C_AXIAL_CONTEXT_RADIUS,
    STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
    STAGE_C_PRIMARY_MODEL,
    STAGE_C_SECONDARY_MODEL,
    STAGE_C_TAU,
    AgentAction,
    AgentDecision,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)
from src.data.dataset import (
    MODALITIES,
    CachedCaseVolume,
    CaseFiles,
    _load_and_normalize_case,
    discover_valid_cases,
)
from src.uncertainty.entropy import softmax_entropy

# -----------------------------------------------------------------------------
# Configuration and Constants
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Canonical MRI Modality channel ordering strictly matching src.data.dataset.MODALITIES
MODALITY_DISPLAY_SPECS: tuple[tuple[str, str, int, str], ...] = (
    ("T1 Native", "t1n", 0, "Anatomical baseline; fluid dark, grey/white matter contrast"),
    ("T1ce Contrast-Enhanced", "t1c", 1, "Post-gadolinium; highlights blood-brain barrier disruption & enhancing core"),
    ("T2-FLAIR", "t2f", 2, "Fluid-attenuated inversion recovery; suppresses free CSF, highlights peritumoral edema"),
    ("T2-Weighted", "t2w", 3, "Water/fluid hyperintense; bright CSF and ventricles"),
)

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic / Non-Enhancing Core (NCR)",
    2: "Peritumoral Edema (ED)",
    3: "Enhancing Tumor (ET)",
}

CLASS_HEX_COLORS = {
    0: "#1a1a1a",
    1: "#e63946",  # Red (NCR)
    2: "#2ca02c",  # Green (ED)
    3: "#1f77b4",  # Blue (ET)
}

CLASS_RGB_FLOAT = {
    0: (0.1, 0.1, 0.1),
    1: (0.90, 0.22, 0.27),
    2: (0.17, 0.63, 0.17),
    3: (0.12, 0.47, 0.71),
}


# -----------------------------------------------------------------------------
# Safety and Input Validation
# -----------------------------------------------------------------------------
def validate_input_slice_tensor(image: Any) -> None:
    """Validate 4-channel MRI slice tensor interface contracts.

    Raises:
        TypeError: If image is not a torch.Tensor.
        ValueError: If tensor shape, dimensions, or channel count violate contracts.
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Input image must be a torch.Tensor, got {type(image)}")
    if image.ndim != 4:
        raise ValueError(
            f"Input image must have ndim=4 with shape [1, 4, H, W], got ndim={image.ndim} (shape={tuple(image.shape)})"
        )
    if image.shape[0] != 1:
        raise ValueError(f"Batch dimension must be exactly 1 for single-slice inference, got {image.shape[0]}")
    if image.shape[1] != 4:
        raise ValueError(
            f"Channel dimension must be exactly 4 (canonical order: t1n, t1c, t2f, t2w), got {image.shape[1]}"
        )
    if image.shape[2] <= 0 or image.shape[3] <= 0:
        raise ValueError(f"Spatial dimensions (H, W) must be positive, got ({image.shape[2]}, {image.shape[3]})")
    if torch.isnan(image).any() or torch.isinf(image).any():
        raise ValueError("Input image tensor contains invalid NaN or Inf numerical values.")


# -----------------------------------------------------------------------------
# Calibration Configuration Loading
# -----------------------------------------------------------------------------
def load_calibrated_policy_config(
    calib_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    tau_threshold: float = STAGE_C_TAU,
    enable_empty_fallback: bool = True,
    enable_axial_context: bool = True,
    axial_context_window: int = STAGE_C_AXIAL_CONTEXT_RADIUS,
    min_neighbor_foreground_slices: int = STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
) -> tuple[PolicyConfig, dict[str, Any]]:
    """Load policy and temperature scaling parameters strictly from verified Stage C2 calibration artifacts.

    Strictly fails closed: if calibration loading or cryptographic verification fails,
    an exception is raised immediately. Hardcoded fallback substitution is prohibited.
    """
    calib_cfg = load_stage_c_calibration(
        calib_path=calib_path,
        manifest_path=manifest_path,
        verify_hashes=True,
    )
    policy_config = create_stage_c_policy_config(
        calibration_config=calib_cfg,
        tau=tau_threshold,
        axial_context_window=axial_context_window,
        min_neighbor_foreground_slices=min_neighbor_foreground_slices,
    )
    metadata = {
        "source": f"Empirical Stage C2 Artifact (SHA-256: {calib_cfg.calibration_json_sha256[:16]}...)",
        "resolved_path": str(calib_path or DEFAULT_C2_CALIBRATION_JSON),
        "primary_temperature": policy_config.primary_temperature,
        "candidate_temperatures": dict(policy_config.candidate_temperatures),
        "calibration_config": calib_cfg,
    }
    return policy_config, metadata


# -----------------------------------------------------------------------------
# Caching Functions & Dataset Discovery
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing Stage C ModelRegistry (15-Epoch)...")
def get_model_registry(
    device_str: str = "cpu",
    profile: CheckpointProfile = CheckpointProfile.STAGE_C_15EP,
) -> ModelRegistry:
    """Instantiate and cache ModelRegistry with CheckpointProfile.STAGE_C_15EP on target execution device."""
    if profile != CheckpointProfile.STAGE_C_15EP:
        raise ValueError(
            f"Stage C demo application strictly requires CheckpointProfile.STAGE_C_15EP, "
            f"got profile '{profile.value}'. Silent fallback to legacy profiles is prohibited."
        )
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    return ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP, default_device=device)


@st.cache_resource(show_spinner="Initializing Stage C Reliability-Aware AI Agent...")
def get_stage_c_agent(device_str: str = "cpu") -> InferenceAgent:
    """Instantiate and cache the authoritative Stage C reliability agent."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    registry = get_model_registry(device_str=device_str, profile=CheckpointProfile.STAGE_C_15EP)
    return create_stage_c_agent(registry=registry, device=device)


@st.cache_data(show_spinner="Loading 105-case validation evaluation cohort...")
def get_validation_cases(
    archive_dir_override: Path | None = None,
    splits_path_override: Path | None = None,
) -> list[CaseFiles]:
    """Retrieve strictly the 105 validation evaluation cases without uncontrolled directory fallback."""
    splits_path = (
        splits_path_override
        if splits_path_override is not None
        else PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_splits.json"
    )
    archive_dir = (
        archive_dir_override
        if archive_dir_override is not None
        else PROJECT_ROOT / "data" / "extracted_archive"
    )

    if not archive_dir.exists():
        raise FileNotFoundError(
            f"Validation archive directory does not exist: {archive_dir}. "
            f"Uncontrolled directory fallback is strictly prohibited to prevent cohort corruption."
        )

    if not splits_path.exists():
        raise FileNotFoundError(
            f"Validation split manifest does not exist at {splits_path}. "
            f"Cannot resolve the 105-case evaluation cohort."
        )

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)
    target_case_ids: list[str] = splits_data.get("evaluation_set", {}).get("case_ids", [])
    if not target_case_ids:
        raise ValueError(f"No evaluation case IDs found in {splits_path}.")

    all_discovered = discover_valid_cases(archive_dir)
    case_map = {c.case_id: c for c in all_discovered}

    matched = [case_map[cid] for cid in target_case_ids if cid in case_map]
    if len(matched) != len(target_case_ids):
        missing = [cid for cid in target_case_ids if cid not in case_map]
        raise FileNotFoundError(
            f"Missing {len(missing)} of {len(target_case_ids)} validation cases in {archive_dir}. "
            f"Example missing: {missing[:5]}. Halting to prevent cohort corruption."
        )

    return matched


@st.cache_data(show_spinner="Loading and normalizing 3D mpMRI case volume...")
def load_case_volume(case_dir_str: str) -> tuple[np.ndarray, np.ndarray]:
    """Load normalized 4-modality image volume and integer segmentation mask."""
    case_path = Path(case_dir_str)
    all_cases = discover_valid_cases(case_path.parent)
    matched = [c for c in all_cases if c.case_id == case_path.name]
    if not matched:
        raise FileNotFoundError(f"Case directory {case_dir_str} could not be resolved.")
    cached: CachedCaseVolume = _load_and_normalize_case(matched[0])
    return cached.image, cached.mask


# -----------------------------------------------------------------------------
# Axial Context & Inference Helper
# -----------------------------------------------------------------------------
def get_axial_context_predictions(
    registry: ModelRegistry,
    primary_model: str,
    image_vol: np.ndarray,
    center_z: int,
    window_k: int = 2,
    total_slices: int = 155,
    device: torch.device | str = "cpu",
    temperature: float = 1.0,
) -> dict[int, SlicePrediction]:
    """Evaluate primary model predictions across adjacent axial slices for context."""
    neighbor_preds: dict[int, SlicePrediction] = {}
    for delta in range(-window_k, window_k + 1):
        if delta == 0:
            continue
        nz = center_z + delta
        if 0 <= nz < total_slices:
            slice_t = torch.from_numpy(image_vol[:, :, :, nz]).unsqueeze(0)
            pred = registry.predict_slice(
                model_name=primary_model,
                image=slice_t,
                device=device,
                temperature=temperature,
            )
            neighbor_preds[nz] = pred
    return neighbor_preds


# -----------------------------------------------------------------------------
# Visualization Helpers
# -----------------------------------------------------------------------------
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert integer mask [H, W] to RGB float array [H, W, 3]."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cls_id, color in CLASS_RGB_FLOAT.items():
        if cls_id == 0:
            continue
        cls_mask = mask == cls_id
        for ch in range(3):
            rgb[:, :, ch][cls_mask] = color[ch]
    return rgb


def overlay_mask_on_grayscale(
    image_gray: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend colored discrete mask over normalized grayscale MRI slice."""
    p_min, p_max = float(np.min(image_gray)), float(np.max(image_gray))
    if p_max > p_min:
        gray_norm = (image_gray - p_min) / (p_max - p_min)
    else:
        gray_norm = np.zeros_like(image_gray)

    rgb_bg = np.stack([gray_norm, gray_norm, gray_norm], axis=-1)
    rgb_fg = colorize_mask(mask)

    fg_pixels = mask > 0
    blended = rgb_bg.copy()
    for ch in range(3):
        blended[:, :, ch][fg_pixels] = (
            (1.0 - alpha) * rgb_bg[:, :, ch][fg_pixels]
            + alpha * rgb_fg[:, :, ch][fg_pixels]
        )
    return np.clip(blended, 0.0, 1.0)


def compute_slice_dice_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Compute exact slice-level Dice scores for clinical composite tumor regions."""
    p_wt = pred > 0
    t_wt = gt > 0
    tp_wt = float(np.logical_and(p_wt, t_wt).sum())
    denom_wt = (2.0 * tp_wt) + float(np.logical_and(p_wt, ~t_wt).sum()) + float(np.logical_and(~p_wt, t_wt).sum())
    wt_dice = (2.0 * tp_wt / denom_wt) if denom_wt > 0 else (1.0 if not t_wt.any() and not p_wt.any() else 0.0)

    p_tc = np.logical_or(pred == 1, pred == 3)
    t_tc = np.logical_or(gt == 1, gt == 3)
    tp_tc = float(np.logical_and(p_tc, t_tc).sum())
    denom_tc = (2.0 * tp_tc) + float(np.logical_and(p_tc, ~t_tc).sum()) + float(np.logical_and(~p_tc, t_tc).sum())
    tc_dice = (2.0 * tp_tc / denom_tc) if denom_tc > 0 else (1.0 if not t_tc.any() and not p_tc.any() else 0.0)

    p_et = pred == 3
    t_et = gt == 3
    tp_et = float(np.logical_and(p_et, t_et).sum())
    denom_et = (2.0 * tp_et) + float(np.logical_and(p_et, ~t_et).sum()) + float(np.logical_and(~p_et, t_et).sum())
    et_dice = (2.0 * tp_et / denom_et) if denom_et > 0 else (1.0 if not t_et.any() and not p_et.any() else 0.0)

    class_dices = []
    for c in [1, 2, 3]:
        p_c = pred == c
        t_c = gt == c
        tp_c = float(np.logical_and(p_c, t_c).sum())
        d_c = (2.0 * tp_c) + float(np.logical_and(p_c, ~t_c).sum()) + float(np.logical_and(~p_c, t_c).sum())
        if d_c > 0:
            class_dices.append(2.0 * tp_c / d_c)
        elif t_c.any():
            class_dices.append(0.0)

    mean_fg = float(np.mean(class_dices)) if class_dices else (0.0 if t_wt.any() else np.nan)
    return {
        "macro_fg_dice": mean_fg,
        "wt_dice": wt_dice,
        "tc_dice": tc_dice,
        "et_dice": et_dice,
    }


# -----------------------------------------------------------------------------
# Non-Tumor Slice Interpretation Notice (Empty-Empty Agreement)
# -----------------------------------------------------------------------------
NON_TUMOR_SLICE_DICE_NOTICE: str = (
    "Both ground-truth and predicted masks are empty on this slice. "
    "Dice = 1.000 reflects empty-empty agreement, not demonstrated tumor segmentation accuracy."
)
# Backward-compatibility alias
HEALTHY_SLICE_DICE_NOTICE: str = NON_TUMOR_SLICE_DICE_NOTICE


def should_display_healthy_slice_notice(pred: np.ndarray, gt: np.ndarray) -> bool:
    """Determine whether the non-tumor slice Dice interpretation notice should be displayed.

    Returns True if and only if BOTH conditions are met:
        1. Ground-truth mask contains zero foreground tumor voxels.
        2. Predicted mask contains zero foreground tumor voxels.
    """
    return bool(not (gt > 0).any() and not (pred > 0).any())


# Semantic alias
should_display_non_tumor_slice_notice = should_display_healthy_slice_notice


def format_healthy_slice_notice_html(
    notice_text: str = NON_TUMOR_SLICE_DICE_NOTICE,
) -> str:
    """Format structured HTML notice for empty-empty Dice agreement on non-tumor slices."""
    return (
        f"<div class='healthy-slice-notice' style='background-color: #1e293b; color: #e2e8f0; "
        f"border-left: 4px solid #38bdf8; border: 1px solid #334155; padding: 10px 14px; "
        f"border-radius: 4px; font-size: 0.91rem; margin-top: 10px; margin-bottom: 12px;'>"
        f"ℹ️ <b>Non-Tumor Slice Metric Notice:</b> {notice_text}"
        f"</div>"
    )


format_non_tumor_slice_notice_html = format_healthy_slice_notice_html


def format_decision_badge_html(
    decision: AgentDecision,
    tau: float = STAGE_C_TAU,
    primary_model: str = "ResUNet-2D",
) -> str:
    """Format structured HTML badge for the agent's categorical action with Stage C3 v1.1.3 audit metadata."""
    action = decision.action
    if action == AgentAction.ACCEPT:
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "N/A"
        return (
            f"<div style='background-color: #2ca02c; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 14px;'>"
            f"✅ <b>ACTION: ACCEPT</b> &nbsp;|&nbsp; Autonomous Retention (H = {h_str} < τ = {tau:.2f}) &nbsp;|&nbsp; "
            f"Selected Model: <code>{decision.selected_model}</code> (T* = {decision.calibrated_temperature:.4f})"
            f"</div>"
        )
    elif action == AgentAction.FALLBACK_SWITCH:
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "N/A"
        has_warning = (
            "candidate_high_uncertainty_warning" in decision.reasons
            or decision.metadata.get("candidate_high_uncertainty_warning", False)
        )
        badge = (
            f"<div style='background-color: #ff7f0e; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 8px;'>"
            f"🔄 <b>ACTION: FALLBACK_SWITCH</b> &nbsp;|&nbsp; Model Substitution (<code>{primary_model}</code> ➔ "
            f"<code>{decision.selected_model}</code>) &nbsp;|&nbsp; Candidate H = {h_str} &nbsp;|&nbsp; "
            f"Axial Context Continuity Confirmed"
            f"</div>"
        )
        if has_warning:
            badge += (
                f"<div style='background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; "
                f"padding: 10px 14px; border-radius: 5px; font-size: 0.93rem; margin-bottom: 14px;'>"
                f"⚠️ <b>Candidate High Uncertainty Notice:</b> Candidate model <code>{decision.selected_model}</code> "
                f"predictive entropy (H = {h_str}) meets or exceeds operating threshold (τ = {tau:.2f}). "
                f"Per Stage C3 v1.1.3 scientific policy, FALLBACK_SWITCH is preserved to prevent false-negative lesion omission, "
                f"but specialist visual confirmation of tumor boundaries is recommended."
                f"</div>"
            )
        else:
            badge += "<div style='margin-bottom: 6px;'></div>"
        return badge
    else:  # AgentAction.FLAG_UNRELIABLE
        reason_desc = (
            "Uncertainty exceeds operating threshold"
            if not decision.has_empty_foreground
            else "Complete empty prediction warning"
        )
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "None (Empty)"
        return (
            f"<div style='background-color: #d62728; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 8px;'>"
            f"⚠️ <b>ACTION: FLAG_UNRELIABLE</b> &nbsp;|&nbsp; Human Specialist Review Required &nbsp;|&nbsp; "
            f"{reason_desc} (H = {h_str}) &nbsp;|&nbsp; Model: <code>{decision.selected_model}</code>"
            f"</div>"
            f"<div style='background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; "
            f"padding: 9px 14px; border-radius: 5px; font-size: 0.90rem; margin-bottom: 14px;'>"
            f"🚩 <b>Clinical Notice:</b> An unreliable prediction indicates statistical prediction uncertainty under the "
            f"reliability agent's policy, NOT a clinically confirmed absence or presence of pathology. Specialist neuroradiologist review is required."
            f"</div>"
        )


# -----------------------------------------------------------------------------
# Main Application Layout
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Reliability-Aware AI Agent Showcase",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            border-left: 5px solid #1f77b4;
            padding: 10px 14px;
            border-radius: 4px;
            margin-bottom: 8px;
        }
        .disclaimer-box {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeeba;
            padding: 10px 16px;
            border-radius: 5px;
            font-size: 0.92rem;
            margin-bottom: 16px;
        }
        .healthy-slice-notice {
            background-color: #1e293b;
            color: #e2e8f0;
            border-left: 4px solid #38bdf8;
            border: 1px solid #334155;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 0.91rem;
            margin-top: 10px;
            margin-bottom: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header and Research Context
    st.title("🧠 Reliability-Aware AI Agent for Brain Tumor Segmentation")
    st.caption(
        "B.Tech CSE Major Project Research Showcase — Primary Champion: ResUNet-2D (15-Epoch Scaled, 8.11M Parameters) "
        "with Calibrated Selective Prediction Triage (T* = 2.880459, τ = 0.35) and UNetResNet34 Axial Fallback on BraTS 2023 GLI"
    )

    # Mandatory Academic / Clinical Transparency Disclaimer
    st.markdown(
        "<div class='disclaimer-box'>"
        "⚠️ <b>Research Demonstration Disclaimer:</b> This software is an academic engineering prototype developed for "
        "evaluating uncertainty calibration and agentic triage policies. It is <b>NOT</b> approved by regulatory bodies (FDA/CDSCO) "
        "and must <b>NEVER</b> be used for clinical diagnosis, patient management, or medical decision-making. "
        "Predicted boundaries reflect predictive uncertainty and mathematical heuristics, not certified diagnostic facts."
        "</div>",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("🛠️ Agent & Model Controls")

    default_dev = "cuda" if torch.cuda.is_available() else "cpu"
    device_selection = st.sidebar.radio(
        "Inference Hardware Device",
        options=["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        index=0,
        help="Executes PyTorch neural network inference on GPU (CUDA) or CPU.",
    )

    # Stage C Multi-Model Reliability Agent Architecture Specification
    st.sidebar.markdown("### 🤖 Stage C Multi-Model Reliability Agent")
    st.sidebar.markdown(
        "- **Primary Champion:** `ResUNet-2D` (15 ep, T* = 2.8805)\n"
        "- **Secondary Fallback:** `UNetResNet34` (15 ep, T* = 2.6261)\n"
        "- **Decision Policy:** `Stage C3 v1.1.3 (Frozen)`\n"
        "- **Entropy Threshold (τ):** `0.35`\n"
        "- **Axial Context Window (k):** `2` (±2 slices, N_min = 1)\n"
        "- **Checkpoint Profile:** `STAGE_C_15EP`"
    )

    # Load 105-case validation evaluation cohort strictly from data/extracted_archive
    try:
        val_cases = get_validation_cases()
    except Exception as e:
        st.error(f"Failed to load validation evaluation cohort: {e}")
        return

    case_options = [c.case_id for c in val_cases]
    selected_case_id = st.sidebar.selectbox(
        "Select Validation Patient",
        options=case_options,
        index=0,
        help="105-case calibration-disjoint internal validation subset (96 unique patient subjects).",
    )

    selected_case = next(c for c in val_cases if c.case_id == selected_case_id)

    # Load Case Volume
    try:
        image_vol, mask_vol = load_case_volume(str(selected_case.case_dir))
    except Exception as e:
        st.error(f"Failed to load case volume: {e}")
        return

    fg_slice_indices = [z for z in range(155) if (mask_vol[:, :, z] > 0).any()]
    tumor_range_str = f"Slices {min(fg_slice_indices)}–{max(fg_slice_indices)}" if fg_slice_indices else "None (Non-Tumor)"
    st.sidebar.markdown(f"**Patient Tumor Profile:** `{tumor_range_str}` ({len(fg_slice_indices)} tumor slices)")

    default_z = fg_slice_indices[len(fg_slice_indices) // 2] if fg_slice_indices else 72
    selected_z = st.sidebar.slider(
        "Axial Slice Index (z)",
        min_value=0,
        max_value=154,
        value=default_z,
        step=1,
        help="Select axial slice index from the 155-slice 3D volume.",
    )

    overlay_alpha = st.sidebar.slider(
        "Mask Overlay Opacity",
        min_value=0.1,
        max_value=0.9,
        value=0.45,
        step=0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎨 Tumor Sub-Regions")
    st.sidebar.markdown(
        f"- <span style='color:{CLASS_HEX_COLORS[1]}; font-weight:bold;'>■</span> **NCR:** Necrotic Core (Class 1)\n"
        f"- <span style='color:{CLASS_HEX_COLORS[2]}; font-weight:bold;'>■</span> **ED:** Peritumoral Edema (Class 2)\n"
        f"- <span style='color:{CLASS_HEX_COLORS[3]}; font-weight:bold;'>■</span> **ET:** Enhancing Tumor (Class 3)",
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Stage C Agent Initialization & Verification Handshake
    # -------------------------------------------------------------------------
    try:
        agent = get_stage_c_agent(device_selection)
    except Exception as e:
        st.error(
            f"❌ **Stage C Reliability Agent Initialization Failed:** {e}\n\n"
            f"Strict checkpoint cryptographic verification (SHA-256) or Stage C2 calibration "
            f"manifest provenance validation failed. Execution is halted immediately to prevent "
            f"unverified model inference or silent fallback to legacy configurations."
        )
        return

    # -------------------------------------------------------------------------
    # Slice Tensor Extraction & Verification
    # -------------------------------------------------------------------------
    slice_img_np = image_vol[:, :, :, selected_z]
    slice_gt_np = mask_vol[:, :, selected_z]
    slice_img_t = torch.from_numpy(slice_img_np).unsqueeze(0).to(agent.device)

    # Explicit input contract validation
    try:
        validate_input_slice_tensor(slice_img_t)
    except Exception as e:
        st.error(f"Input Interface Contract Violation: {e}")
        return

    # -------------------------------------------------------------------------
    # Runtime Inference & Uncertainty Map Generation
    # -------------------------------------------------------------------------
    try:
        # Compute neighboring axial predictions for spatial continuity summary
        neighbor_predictions: dict[int, SlicePrediction] = {}
        if agent.config.enable_axial_context:
            neighbor_predictions = get_axial_context_predictions(
                registry=agent.registry,
                primary_model=agent.primary_model,
                image_vol=image_vol,
                center_z=selected_z,
                window_k=agent.config.axial_context_window,
                total_slices=155,
                device=agent.device,
                temperature=agent.primary_temperature,
            )

        # Execute Autonomous Research Agent
        with st.spinner("Executing Stage C Reliability-Aware AI Agent policy..."):
            decision: AgentDecision = agent.run_slice(
                image=slice_img_t,
                case_id=selected_case.case_id,
                slice_index=selected_z,
                case_primary_predictions=neighbor_predictions,
                case_total_slices=155,
            )

        # Obtain calibrated entropy map for the model selected by the agent
        selected_pred = agent.registry.predict_slice(
            model_name=decision.selected_model,
            image=slice_img_t,
            device=agent.device,
            temperature=decision.calibrated_temperature,
        )
        # Normalized predictive entropy H(p*) in [0, 1]
        entropy_tensor = softmax_entropy(selected_pred.logits)
        entropy_np = entropy_tensor.squeeze(0).cpu().numpy()
        pred_mask_np = decision.final_mask.cpu().numpy()
    except Exception as e:
        st.error(
            f"❌ **Runtime Inference Failed:** {e}\n\n"
            f"An error occurred during neural network inference, axial neighbor context evaluation, "
            f"or calibrated predictive entropy generation. Rendering of the prediction mask and "
            f"agent decision has been aborted to prevent silent model substitution or corrupt outputs."
        )
        return

    # -------------------------------------------------------------------------
    # Section 1: Multi-Parametric MRI Modality Viewer
    # -------------------------------------------------------------------------
    st.subheader("1. Multi-Parametric MRI Modalities (Canonical BraTS Channel Layout)")
    st.caption("Native in-plane resolution: 240 × 240 voxels. Intensity normalized to standard normal distribution.")
    mod_cols = st.columns(4)
    for col, (label, code, ch_idx, desc) in zip(mod_cols, MODALITY_DISPLAY_SPECS):
        with col:
            st.markdown(f"**Channel {ch_idx}: {label} (`{code}`)**")
            ch_data = slice_img_np[ch_idx]
            p_min, p_max = float(np.min(ch_data)), float(np.max(ch_data))
            norm_disp = (ch_data - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(ch_data)
            st.image(norm_disp, clamp=True, use_container_width=True)
            st.caption(desc)

    # -------------------------------------------------------------------------
    # Section 2: Ground Truth vs Model Prediction & Calibrated Uncertainty
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("2. Segmentation Visualizer, Entropy Heatmap & Agent Decision")

    # Display Categorical Action Badge
    decision_html = format_decision_badge_html(
        decision=decision,
        tau=agent.config.high_uncertainty_threshold,
        primary_model="ResUNet-2D",
    )
    st.markdown(decision_html, unsafe_allow_html=True)

    seg_cols = st.columns(3)
    with seg_cols[0]:
        st.markdown("**Ground-Truth (Manual Annotation)**")
        gt_overlay = overlay_mask_on_grayscale(slice_img_np[1], slice_gt_np, alpha=overlay_alpha)
        st.image(gt_overlay, use_container_width=True)
        gt_fg_count = int((slice_gt_np > 0).sum())
        st.caption(f"Lesion Size: **{gt_fg_count:,} voxels** ({'Tumor Present' if gt_fg_count > 0 else 'Non-Tumor Slice'})")

    with seg_cols[1]:
        st.markdown(f"**Agent Output Mask (`{decision.selected_model}`)**")
        p_overlay = overlay_mask_on_grayscale(slice_img_np[1], pred_mask_np, alpha=overlay_alpha)
        st.image(p_overlay, use_container_width=True)
        p_fg_count = int((pred_mask_np > 0).sum())
        model_role = "Primary Champion" if decision.selected_model == agent.primary_model else "Axial Fallback Candidate"
        st.caption(f"Predicted Size: **{p_fg_count:,} voxels** ({model_role}, T* = {decision.calibrated_temperature:.4f})")

    with seg_cols[2]:
        st.markdown("**Calibrated Predictive Entropy ($H \\in [0, 1]$)**")
        fig, ax = plt.subplots(figsize=(4.5, 3.8), dpi=120)
        im = ax.imshow(entropy_np, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Normalized Entropy H(p*)")
        st.pyplot(fig)
        plt.close(fig)
        if decision.has_empty_foreground:
            st.caption("Mean Foreground Uncertainty: **N/A (Empty foreground predicted)**")
        else:
            st.caption(f"Mean Foreground Uncertainty: **{decision.mean_uncertainty:.4f}**")

    # -------------------------------------------------------------------------
    # Section 3: Structured Agent Audit Trail & Decision Ledger
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3. Structured Agent Audit Trail & Policy Ledger")

    card_col1, card_col2 = st.columns(2)

    calib_source = (
        f"Stage C2 c2_calibration_v1 (SHA-256: {agent.calibration_config.calibration_json_sha256[:16]}...)"
        if agent.calibration_config
        else "Authoritative Stage C2 Reference Temperatures"
    )

    with card_col1:
        st.markdown("#### 📋 Action Summary & Execution Parameters")
        st.markdown(
            f"- **Agent Categorical Action:** `{decision.action.value}`\n"
            f"- **Selected Output Model:** `{decision.selected_model}`\n"
            f"- **Primary Champion Model:** `{agent.primary_model}` (ResUNet-2D, T* = {agent.primary_temperature:.4f})\n"
            f"- **Secondary Fallback Model:** `{STAGE_C_SECONDARY_MODEL}` (UNetResNet34, T* = {agent.config.candidate_temperatures.get(STAGE_C_SECONDARY_MODEL, 2.6261):.4f})\n"
            f"- **Selected Calibrated Temperature (T*):** `{decision.calibrated_temperature:.4f}`\n"
            f"- **Operating Threshold (τ):** `{agent.config.high_uncertainty_threshold:.2f}`\n"
            f"- **Mean Foreground Entropy (H):** `{decision.mean_uncertainty:.4f}`"
            if decision.mean_uncertainty is not None
            else f"- **Mean Foreground Entropy (H):** `N/A (Empty Foreground)`\n"
            f"- **Specialist Review Recommended:** `{'YES (Referred)' if decision.is_human_review_recommended else 'NO (Retained)'}\n"
            f"- **Calibration Provenance:** `{calib_source}`"
        )

    with card_col2:
        st.markdown("#### 🧭 Policy Invariants & Auditable Rule Reasons")
        st.markdown("**Evaluated Rule Reasons:**")
        for r in decision.reasons:
            st.markdown(f"- <code>{r}</code>", unsafe_allow_html=True)

        if decision.axial_context is not None:
            ctx = decision.axial_context
            st.markdown(
                f"**Axial Context Window (Radius k={agent.config.axial_context_window}):**\n"
                f"- Evaluated Neighbors: **{ctx.evaluated_neighbors_count}**\n"
                f"- Foreground Neighbors: **{ctx.foreground_neighbors_count}**\n"
                f"- Empty Neighbors: **{ctx.empty_neighbors_count}**\n"
                f"- Unavailable Neighbors: **{ctx.unavailable_neighbors_count}**"
            )

    # -------------------------------------------------------------------------
    # Section 4: Slice-Level Verification Metrics (vs Ground Truth)
    # -------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("4. Slice-Level Evaluation Metrics (vs Manual Annotation)")

    slice_dice = compute_slice_dice_metrics(pred_mask_np, slice_gt_np)
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.metric(label="Whole Tumor (WT) Dice", value=f"{slice_dice['wt_dice']:.3f}")
    with m_col2:
        st.metric(label="Tumor Core (TC) Dice", value=f"{slice_dice['tc_dice']:.3f}")
    with m_col3:
        st.metric(label="Enhancing Tumor (ET) Dice", value=f"{slice_dice['et_dice']:.3f}")
    with m_col4:
        st.metric(
            label="Macro Foreground Dice",
            value=f"{slice_dice['macro_fg_dice']:.3f}" if not np.isnan(slice_dice['macro_fg_dice']) else "N/A (Non-Tumor Slice)",
        )

    # Explanatory notice for empty-empty agreement on healthy slices
    if should_display_healthy_slice_notice(pred_mask_np, slice_gt_np):
        st.markdown(format_healthy_slice_notice_html(), unsafe_allow_html=True)

    # Viva Demonstration Explanation
    if decision.action == AgentAction.ACCEPT:
        st.success(
            f"**Autonomous Retention Rationale:** The primary model output satisfies all reliability criteria. "
            f"Calibrated foreground predictive entropy ({decision.mean_uncertainty:.4f}) is strictly below the operating "
            f"threshold (τ = {agent.config.high_uncertainty_threshold:.2f}). Segmentation exhibits strong confidence gradients and is retained for volumetric measurement."
        )
    elif decision.action == AgentAction.FALLBACK_SWITCH:
        has_cand_warning = (
            "candidate_high_uncertainty_warning" in decision.reasons
            or decision.metadata.get("candidate_high_uncertainty_warning", False)
        )
        warning_note = (
            f" **Note:** Candidate model predictive entropy (H = {decision.mean_uncertainty:.4f}) meets or exceeds operating "
            f"threshold (τ = {agent.config.high_uncertainty_threshold:.2f}). Per Stage C3 v1.1.3 scientific policy, FALLBACK_SWITCH is retained to avoid "
            f"false-negative tumor omission, but specialist visual boundary confirmation is advised."
            if has_cand_warning and decision.mean_uncertainty is not None
            else ""
        )
        st.info(
            f"**Axial Context Fallback Rationale:** The primary model predicted zero foreground voxels on slice z={selected_z}. "
            f"However, neighboring axial slices confirmed spatial tumor continuity ({decision.axial_context.foreground_neighbors_count if decision.axial_context else 0} adjacent foreground slices). "
            f"The agent successfully triggered candidate model `{decision.selected_model}` (T* = {decision.calibrated_temperature:.4f}), which recovered lesion segmentation.{warning_note}"
        )
    else:  # FLAG_UNRELIABLE
        reason_detail = (
            f"Foreground predictive entropy ({decision.mean_uncertainty:.4f}) exceeded operating threshold (τ = {agent.config.high_uncertainty_threshold:.2f}), indicating significant boundary or classification ambiguity."
            if not decision.has_empty_foreground and decision.mean_uncertainty is not None
            else "A complete empty prediction was generated and could not be resolved by axial context fallback (no adjacent foreground tumor continuity or unavailable neighbors)."
        )
        st.warning(
            f"**Specialist Referral Rationale:** The prediction failed reliability criteria. {reason_detail} "
            f"The slice is flagged for neuroradiologist human review to prevent clinical errors.\n\n"
            f"🚩 **Clinical Distinction Notice:** An unreliable flag represents statistical prediction uncertainty under the "
            f"Stage C3 reliability policy, NOT a clinically confirmed diagnosis or absence/presence of pathology."
        )


if __name__ == "__main__":
    main()
