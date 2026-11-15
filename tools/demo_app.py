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

from src.agent.models import (
    CANDIDATE_MODEL_NAMES,
    CANONICAL_MODEL_NAMES,
    ModelRegistry,
    normalize_model_name,
)
from src.agent.runner import InferenceAgent
from src.agent.state import (
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
    config_path: Path | None = None,
    tau_threshold: float = 0.35,
    enable_empty_fallback: bool = True,
    enable_axial_context: bool = True,
    axial_context_window: int = 2,
    min_neighbor_foreground_slices: int = 1,
) -> tuple[PolicyConfig, dict[str, Any]]:
    """Load policy and temperature scaling parameters from empirical calibration artifacts."""
    candidate_paths = [
        config_path,
        PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_parameters.json",
        PROJECT_ROOT / "results" / "gate6a_reliability_calibration" / "calibration_parameters.json",
    ]
    resolved_path: Path | None = None
    for p in candidate_paths:
        if p is not None and p.exists():
            resolved_path = p
            break

    # Verified Gate 6A empirical temperatures (ResUNet-2D and UNetResNet34)
    primary_temp = 2.8807828426361084
    candidate_temps = {"unet_resnet34": 2.6266958713531494}
    fallback_priority = ("unet_resnet34",)
    source_info = "Verified Gate 6A Empirical Temperatures (Default)"

    if resolved_path is not None:
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            frozen = data.get("frozen_policy_configuration", {})
            if "primary_temperature" in frozen:
                primary_temp = float(frozen["primary_temperature"])
            if "candidate_temperatures" in frozen and isinstance(frozen["candidate_temperatures"], dict):
                candidate_temps = {k: float(v) for k, v in frozen["candidate_temperatures"].items()}
            source_info = f"Empirical artifact: {resolved_path.name}"
        except Exception as e:
            source_info = f"Fallback defaults (error reading {resolved_path.name}: {e})"

    policy_config = PolicyConfig(
        high_uncertainty_threshold=tau_threshold,
        enable_empty_prediction_fallback=enable_empty_fallback,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=enable_axial_context,
        axial_context_window=axial_context_window,
        min_neighbor_foreground_slices=min_neighbor_foreground_slices,
        fallback_when_context_unavailable=False,
        primary_temperature=primary_temp,
        candidate_temperatures=candidate_temps,
    )
    metadata = {
        "source": source_info,
        "resolved_path": str(resolved_path) if resolved_path else "In-code calibrated constants",
        "primary_temperature": primary_temp,
        "candidate_temperatures": candidate_temps,
    }
    return policy_config, metadata


def create_policy_config_for_model(
    primary_model: str,
    base_policy_config: PolicyConfig,
    tau_threshold: float,
) -> PolicyConfig:
    """Specialize PolicyConfig temperatures and fallback priorities for the selected primary model."""
    canonical = normalize_model_name(primary_model)

    temp_map = {
        "resunet": 2.8807828426361084,
        "unet_resnet34": 2.6266958713531494,
        "unet": 1.0,
        "unetpp": 1.0,
        "attention_unet": 1.0,
    }
    if base_policy_config.primary_temperature > 0.0:
        temp_map["resunet"] = base_policy_config.primary_temperature
    for k, v in base_policy_config.candidate_temperatures.items():
        temp_map[normalize_model_name(k)] = v

    primary_t = temp_map.get(canonical, 1.0)
    cand_temps = {k: v for k, v in temp_map.items() if k != canonical}

    if canonical == "resunet":
        fallback_priority = ("unet_resnet34", "unetpp", "attention_unet", "unet")
    elif canonical == "unet_resnet34":
        fallback_priority = ("resunet", "unetpp", "attention_unet", "unet")
    else:
        fallback_priority = ("resunet", "unet_resnet34")

    return PolicyConfig(
        high_uncertainty_threshold=tau_threshold,
        enable_empty_prediction_fallback=base_policy_config.enable_empty_prediction_fallback,
        fallback_priority=fallback_priority,
        num_classes=4,
        enable_axial_context=base_policy_config.enable_axial_context,
        axial_context_window=base_policy_config.axial_context_window,
        min_neighbor_foreground_slices=base_policy_config.min_neighbor_foreground_slices,
        fallback_when_context_unavailable=base_policy_config.fallback_when_context_unavailable,
        primary_temperature=primary_t,
        candidate_temperatures=cand_temps,
    )


# -----------------------------------------------------------------------------
# Caching Functions & Dataset Discovery
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Initializing ModelRegistry and scanning checkpoints...")
def get_model_registry(device_str: str) -> ModelRegistry:
    """Instantiate and cache ModelRegistry on target execution device."""
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    return ModelRegistry(default_device=device)


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


def format_decision_badge_html(decision: AgentDecision, tau: float, primary_model: str) -> str:
    """Format structured HTML badge for the agent's categorical action."""
    action = decision.action
    if action == AgentAction.ACCEPT:
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "N/A"
        return (
            f"<div style='background-color: #2ca02c; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 14px;'>"
            f"✅ <b>ACTION: ACCEPT</b> &nbsp;|&nbsp; Autonomous Retention (H = {h_str} ≤ τ = {tau:.2f}) &nbsp;|&nbsp; "
            f"Selected Model: <code>{decision.selected_model}</code> (T* = {decision.calibrated_temperature:.4f})"
            f"</div>"
        )
    elif action == AgentAction.FALLBACK_SWITCH:
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "N/A"
        return (
            f"<div style='background-color: #ff7f0e; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 14px;'>"
            f"🔄 <b>ACTION: FALLBACK_SWITCH</b> &nbsp;|&nbsp; Model Substitution (<code>{primary_model}</code> ➔ "
            f"<code>{decision.selected_model}</code>) &nbsp;|&nbsp; Candidate H = {h_str} &nbsp;|&nbsp; "
            f"Axial Context Continuity Confirmed"
            f"</div>"
        )
    else:  # AgentAction.FLAG_UNRELIABLE
        reason_desc = "Uncertainty exceeds operating threshold" if not decision.has_empty_foreground else "Complete empty prediction warning"
        h_str = f"{decision.mean_uncertainty:.4f}" if decision.mean_uncertainty is not None else "None (Empty)"
        return (
            f"<div style='background-color: #d62728; color: white; padding: 12px 18px; border-radius: 6px; "
            f"font-weight: bold; font-size: 1.1rem; margin-bottom: 14px;'>"
            f"⚠️ <b>ACTION: FLAG_UNRELIABLE</b> &nbsp;|&nbsp; Human Specialist Review Required &nbsp;|&nbsp; "
            f"{reason_desc} (H = {h_str}) &nbsp;|&nbsp; Model: <code>{decision.selected_model}</code>"
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header and Research Context
    st.title("🧠 Reliability-Aware AI Agent for Brain Tumor Segmentation")
    st.caption(
        "B.Tech CSE Major Project Research Showcase — Primary Champion: ResUNet-2D (15-Epoch Scaled, 8.11M Parameters) "
        "with Calibrated Selective Prediction Triage (T* = 2.8808, τ = 0.35) and UNetResNet34 Axial Fallback on BraTS 2023 GLI"
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

    # Model Selection
    model_options = {
        "resunet": "ResUNet-2D (Primary Champion, 15 ep, T*=2.8808)",
        "unet_resnet34": "UNetResNet34 (Comparison/Fallback, 15 ep, T*=2.6267)",
        "unet": "Standard U-Net (5 ep baseline, T=1.0)",
        "attention_unet": "Attention U-Net (5 ep baseline, T=1.0)",
        "unetpp": "U-Net++ (5 ep baseline, T=1.0)",
    }
    selected_model_key = st.sidebar.selectbox(
        "Primary Segmentation Model",
        options=list(model_options.keys()),
        format_func=lambda k: model_options[k],
        index=0,
        help="ResUNet-2D is the project champion architecture with lowest post-calibration ECE.",
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
    tumor_range_str = f"Slices {min(fg_slice_indices)}–{max(fg_slice_indices)}" if fg_slice_indices else "None (Healthy)"
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

    triage_threshold = st.sidebar.slider(
        "Selective Prediction Threshold (τ)",
        min_value=0.10,
        max_value=0.60,
        value=0.35,
        step=0.01,
        help="Slices with mean foreground calibrated entropy H > τ are flagged for specialist referral.",
    )

    enable_context_toggle = st.sidebar.checkbox(
        "Enable Axial Context Verification",
        value=True,
        help="Inspects adjacent axial slices [z - 2, z + 2] to verify spatial lesion continuity before fallback.",
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
    # Policy Configuration & Model Registry Initialization
    # -------------------------------------------------------------------------
    base_policy_cfg, calib_meta = load_calibrated_policy_config(
        tau_threshold=triage_threshold,
        enable_empty_fallback=True,
        enable_axial_context=enable_context_toggle,
        axial_context_window=2,
        min_neighbor_foreground_slices=1,
    )
    policy_config = create_policy_config_for_model(
        primary_model=selected_model_key,
        base_policy_config=base_policy_cfg,
        tau_threshold=triage_threshold,
    )

    registry = get_model_registry(device_selection)
    agent = InferenceAgent(
        registry=registry,
        primary_model=selected_model_key,
        config=policy_config,
        device=registry.default_device,
    )

    # -------------------------------------------------------------------------
    # Slice Tensor Extraction & Verification
    # -------------------------------------------------------------------------
    slice_img_np = image_vol[:, :, :, selected_z]
    slice_gt_np = mask_vol[:, :, selected_z]
    slice_img_t = torch.from_numpy(slice_img_np).unsqueeze(0).to(registry.default_device)

    # Explicit input contract validation
    try:
        validate_input_slice_tensor(slice_img_t)
    except Exception as e:
        st.error(f"Input Interface Contract Violation: {e}")
        return

    # Compute neighboring axial predictions for spatial continuity summary
    neighbor_predictions: dict[int, SlicePrediction] = {}
    if policy_config.enable_axial_context:
        neighbor_predictions = get_axial_context_predictions(
            registry=registry,
            primary_model=selected_model_key,
            image_vol=image_vol,
            center_z=selected_z,
            window_k=policy_config.axial_context_window,
            total_slices=155,
            device=registry.default_device,
            temperature=policy_config.primary_temperature,
        )

    # Execute Autonomous Research Agent
    with st.spinner("Executing Reliability-Aware AI Agent policy..."):
        decision: AgentDecision = agent.run_slice(
            image=slice_img_t,
            case_id=selected_case.case_id,
            slice_index=selected_z,
            case_primary_predictions=neighbor_predictions,
            case_total_slices=155,
        )

    # Obtain calibrated entropy map for the model selected by the agent
    selected_pred = registry.predict_slice(
        model_name=decision.selected_model,
        image=slice_img_t,
        device=registry.default_device,
        temperature=decision.calibrated_temperature,
    )
    # Normalized predictive entropy H(p*) in [0, 1]
    entropy_tensor = softmax_entropy(selected_pred.logits)
    entropy_np = entropy_tensor.squeeze(0).cpu().numpy()
    pred_mask_np = decision.final_mask.cpu().numpy()

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
        tau=triage_threshold,
        primary_model=model_options[selected_model_key],
    )
    st.markdown(decision_html, unsafe_allow_html=True)

    seg_cols = st.columns(3)
    with seg_cols[0]:
        st.markdown("**Ground-Truth (Manual Annotation)**")
        gt_overlay = overlay_mask_on_grayscale(slice_img_np[1], slice_gt_np, alpha=overlay_alpha)
        st.image(gt_overlay, use_container_width=True)
        gt_fg_count = int((slice_gt_np > 0).sum())
        st.caption(f"Lesion Size: **{gt_fg_count:,} voxels** ({'Tumor Present' if gt_fg_count > 0 else 'Healthy Slice'})")

    with seg_cols[1]:
        st.markdown(f"**Agent Output Mask (`{decision.selected_model}`)**")
        p_overlay = overlay_mask_on_grayscale(slice_img_np[1], pred_mask_np, alpha=overlay_alpha)
        st.image(p_overlay, use_container_width=True)
        p_fg_count = int((pred_mask_np > 0).sum())
        model_role = "Primary Champion" if decision.selected_model == selected_model_key else "Axial Fallback Candidate"
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

    with card_col1:
        st.markdown("#### 📋 Action Summary & Execution Parameters")
        st.markdown(
            f"- **Agent Categorical Action:** `{decision.action.value}`\n"
            f"- **Selected Output Model:** `{decision.selected_model}`\n"
            f"- **Configured Primary Model:** `{selected_model_key}`\n"
            f"- **Calibrated Temperature (T*):** `{decision.calibrated_temperature:.4f}`\n"
            f"- **Operating Threshold (τ):** `{triage_threshold:.2f}`\n"
            f"- **Mean Foreground Entropy (H):** `{decision.mean_uncertainty:.4f}`"
            if decision.mean_uncertainty is not None
            else f"- **Mean Foreground Entropy (H):** `N/A (Empty Foreground)`\n"
            f"- **Specialist Review Recommended:** `{'YES (Referred)' if decision.is_human_review_recommended else 'NO (Retained)'}\n"
            f"- **Calibration Provenance:** `{calib_meta['source']}`"
        )

    with card_col2:
        st.markdown("#### 🧭 Policy Invariants & Auditable Rule Reasons")
        st.markdown("**Evaluated Rule Reasons:**")
        for r in decision.reasons:
            st.markdown(f"- <code>{r}</code>", unsafe_allow_html=True)

        if decision.axial_context is not None:
            ctx = decision.axial_context
            st.markdown(
                f"**Axial Context Window (Radius k={policy_config.axial_context_window}):**\n"
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
            value=f"{slice_dice['macro_fg_dice']:.3f}" if not np.isnan(slice_dice['macro_fg_dice']) else "N/A (Healthy)",
        )

    # Viva Demonstration Explanation
    if decision.action == AgentAction.ACCEPT:
        st.success(
            f"**Autonomous Retention Rationale:** The primary model output satisfies all reliability criteria. "
            f"Calibrated foreground predictive entropy ({decision.mean_uncertainty:.4f}) is strictly below the operating "
            f"threshold (τ = {triage_threshold:.2f}). Segmentation exhibits strong confidence gradients and is retained for volumetric measurement."
        )
    elif decision.action == AgentAction.FALLBACK_SWITCH:
        st.info(
            f"**Axial Context Fallback Rationale:** The primary model predicted zero foreground voxels on slice z={selected_z}. "
            f"However, neighboring axial slices confirmed spatial tumor continuity ({decision.axial_context.foreground_neighbors_count if decision.axial_context else 0} adjacent foreground slices). "
            f"The agent successfully triggered candidate model `{decision.selected_model}` (T* = {decision.calibrated_temperature:.4f}), which recovered lesion segmentation."
        )
    else:  # FLAG_UNRELIABLE
        st.warning(
            f"**Specialist Referral Rationale:** The prediction failed reliability criteria. "
            f"{'Foreground predictive entropy exceeded operating threshold, indicating significant boundary or classification ambiguity.' if not decision.has_empty_foreground else 'A complete empty prediction was generated and could not be resolved by axial context fallback.'} "
            f"The slice is flagged for neuroradiologist human review to prevent clinical errors."
        )


if __name__ == "__main__":
    main()
