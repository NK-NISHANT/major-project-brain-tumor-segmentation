"""Inference Agent and Slice Routing Orchestrator.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Scientific Authority: Stage C3 v1.1.3
Batch 1C: Frozen C3 Reliability Agent Integration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.agent.calibration import (
    DEFAULT_C2_CALIBRATION_JSON,
    DEFAULT_C2_RUN_MANIFEST,
    STAGE_C2_REFERENCE_TEMPERATURES,
    StageCCalibrationConfig,
    load_stage_c_calibration,
)
from src.agent.models import (
    CANONICAL_MODEL_NAMES,
    STAGE_C_EXPECTED_HASHES,
    STAGE_C_MODELS,
    CheckpointProfile,
    ModelRegistry,
    normalize_model_name,
)
from src.agent.policy import evaluate_slice_policy
from src.agent.state import (
    STAGE_C_ALLOWED_MODELS,
    STAGE_C_AXIAL_CONTEXT_RADIUS,
    STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
    STAGE_C_PRIMARY_MODEL,
    STAGE_C_SECONDARY_MODEL,
    STAGE_C_TAU,
    AgentDecision,
    AgentState,
    AxialContextSummary,
    PolicyConfig,
    SlicePrediction,
)


def create_stage_c_policy_config(
    calibration_config: StageCCalibrationConfig,
    tau: float = STAGE_C_TAU,
    axial_context_window: int = STAGE_C_AXIAL_CONTEXT_RADIUS,
    min_neighbor_foreground_slices: int = STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
) -> PolicyConfig:
    """Construct a validated PolicyConfig using authoritative Stage C2 temperatures and C3 policy parameters.

    Args:
        calibration_config: Verified immutable StageCCalibrationConfig.
        tau: Predictive entropy decision threshold (authoritative 0.35).
        axial_context_window: Axial spatial radius k (authoritative 2).
        min_neighbor_foreground_slices: Minimum foreground slices in context window (authoritative 1).

    Returns:
        PolicyConfig configured for Stage C3 selective prediction.
    """
    if not isinstance(calibration_config, StageCCalibrationConfig):
        raise TypeError(
            f"calibration_config must be an instance of StageCCalibrationConfig, got {type(calibration_config)}"
        )
    if abs(tau - STAGE_C_TAU) > 1e-7:
        raise ValueError(
            f"Stage C policy requires high_uncertainty_threshold tau={STAGE_C_TAU}, "
            f"got {tau}. Overrides are strictly prohibited."
        )
    if axial_context_window != STAGE_C_AXIAL_CONTEXT_RADIUS:
        raise ValueError(
            f"Stage C policy requires axial_context_window={STAGE_C_AXIAL_CONTEXT_RADIUS}, "
            f"got {axial_context_window}. Overrides are strictly prohibited."
        )
    if min_neighbor_foreground_slices != STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES:
        raise ValueError(
            f"Stage C policy requires min_neighbor_foreground_slices={STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES}, "
            f"got {min_neighbor_foreground_slices}. Overrides are strictly prohibited."
        )

    p_temp = calibration_config.temperature_for(STAGE_C_PRIMARY_MODEL)
    s_temp = calibration_config.temperature_for(STAGE_C_SECONDARY_MODEL)

    return PolicyConfig(
        high_uncertainty_threshold=STAGE_C_TAU,
        enable_empty_prediction_fallback=True,
        fallback_priority=(STAGE_C_SECONDARY_MODEL,),
        num_classes=4,
        enable_axial_context=True,
        axial_context_window=STAGE_C_AXIAL_CONTEXT_RADIUS,
        min_neighbor_foreground_slices=STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
        fallback_when_context_unavailable=False,
        primary_temperature=p_temp,
        candidate_temperatures={STAGE_C_SECONDARY_MODEL: s_temp},
        enable_uncertainty_fallback=False,
        uncertainty_fallback_threshold=None,
    )


def create_stage_c_agent(
    registry: ModelRegistry | None = None,
    calibration_config: StageCCalibrationConfig | None = None,
    tau: float = STAGE_C_TAU,
    axial_context_window: int = STAGE_C_AXIAL_CONTEXT_RADIUS,
    min_neighbor_foreground_slices: int = STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
    device: torch.device | str | None = None,
    calib_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
) -> InferenceAgent:
    """Construct and validate a live Stage C reliability inference agent.

    Enforces:
      - Registry profile CheckpointProfile.STAGE_C_15EP.
      - Authoritative primary model: ResUNet-2D ('resunet').
      - Authoritative secondary model: UNetResNet34 ('unet_resnet34').
      - Strict Stage C2 temperature calibration loading with default hash verification.
      - Calibration-to-checkpoint cryptographic identity handshake.
      - Frozen C3 policy parameters (tau=0.35, k=2).
      - Rejection of invalid configurations or unapproved architectures.
    """
    # 0. Enforce authoritative Stage C policy parameters
    if abs(tau - STAGE_C_TAU) > 1e-7:
        raise ValueError(
            f"Stage C policy requires high_uncertainty_threshold tau={STAGE_C_TAU}, "
            f"got {tau}. Overrides are strictly prohibited."
        )
    if axial_context_window != STAGE_C_AXIAL_CONTEXT_RADIUS:
        raise ValueError(
            f"Stage C policy requires axial_context_window={STAGE_C_AXIAL_CONTEXT_RADIUS}, "
            f"got {axial_context_window}. Overrides are strictly prohibited."
        )
    if min_neighbor_foreground_slices != STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES:
        raise ValueError(
            f"Stage C policy requires min_neighbor_foreground_slices={STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES}, "
            f"got {min_neighbor_foreground_slices}. Overrides are strictly prohibited."
        )

    # 1. Resolve & validate registry
    if registry is None:
        registry = ModelRegistry(
            profile=CheckpointProfile.STAGE_C_15EP,
            default_device=device,
        )
    else:
        if registry.profile != CheckpointProfile.STAGE_C_15EP:
            raise ValueError(
                f"Stage C reliability agent requires CheckpointProfile.STAGE_C_15EP, "
                f"got profile '{registry.profile.value}'. Legacy profiles are strictly prohibited."
            )

    # 2. Resolve & validate calibration configuration
    if calibration_config is None:
        calibration_config = load_stage_c_calibration(
            calib_path=calib_path,
            manifest_path=manifest_path,
            verify_hashes=True,
        )
    else:
        if not isinstance(calibration_config, StageCCalibrationConfig):
            raise TypeError(
                f"calibration_config must be an instance of StageCCalibrationConfig, got {type(calibration_config)}"
            )

    # 3. Cryptographic checkpoint identity handshake
    for model_id in (STAGE_C_PRIMARY_MODEL, STAGE_C_SECONDARY_MODEL):
        calib_hash = calibration_config.checkpoint_hash_for(model_id)
        reg_expected = STAGE_C_EXPECTED_HASHES.get(model_id)
        if reg_expected is not None and calib_hash != reg_expected:
            raise ValueError(
                f"Calibration-to-checkpoint identity mismatch for '{model_id}'!\n"
                f"  Calibration artifact hash: {calib_hash}\n"
                f"  Stage C registry expected hash: {reg_expected}"
            )

    # 4. Construct Stage C policy config
    policy_cfg = create_stage_c_policy_config(
        calibration_config=calibration_config,
        tau=tau,
        axial_context_window=axial_context_window,
        min_neighbor_foreground_slices=min_neighbor_foreground_slices,
    )

    # 5. Return validated agent
    return InferenceAgent(
        registry=registry,
        primary_model=STAGE_C_PRIMARY_MODEL,
        config=policy_cfg,
        device=device,
        calibration_config=calibration_config,
        is_stage_c=True,
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
        calibration_config: StageCCalibrationConfig | None = None,
        is_stage_c: bool = False,
    ) -> None:
        self.registry = registry
        self.primary_model = normalize_model_name(primary_model)
        self.config = config if config is not None else PolicyConfig()
        self.device = torch.device(device) if device is not None else registry.default_device
        self.calibration_config = calibration_config
        self.is_stage_c = is_stage_c

        if self.is_stage_c:
            self._validate_stage_c_invariants()

    def _validate_stage_c_invariants(self) -> None:
        """Validate strict invariants required for Stage C reliability agent."""
        if self.registry.profile != CheckpointProfile.STAGE_C_15EP:
            raise ValueError(
                f"Stage C reliability agent requires CheckpointProfile.STAGE_C_15EP, "
                f"got profile '{self.registry.profile.value}'."
            )
        if self.calibration_config is None:
            raise ValueError(
                "Stage C reliability agent requires a verified StageCCalibrationConfig."
            )
        if not isinstance(self.calibration_config, StageCCalibrationConfig):
            raise TypeError(
                f"Stage C reliability agent requires StageCCalibrationConfig, "
                f"got {type(self.calibration_config)}."
            )

        # 1. Primary Model: ResUNet-2D ('resunet')
        if self.primary_model != STAGE_C_PRIMARY_MODEL:
            raise ValueError(
                f"Stage C reliability agent primary model must be '{STAGE_C_PRIMARY_MODEL}', "
                f"got '{self.primary_model}'."
            )

        # 2. Secondary Model: UNetResNet34 ('unet_resnet34') only
        fb_priority = tuple(normalize_model_name(m) for m in self.config.fallback_priority)
        if fb_priority != (STAGE_C_SECONDARY_MODEL,):
            raise ValueError(
                f"Stage C reliability agent fallback priority must be strictly ('{STAGE_C_SECONDARY_MODEL}',), "
                f"got {fb_priority}."
            )
        for m in fb_priority:
            if m not in STAGE_C_ALLOWED_MODELS:
                raise ValueError(f"Model '{m}' is not permitted in Stage C fallback chain.")

        # 3. Decision Threshold: tau == 0.35
        if (
            self.config.high_uncertainty_threshold is None
            or abs(self.config.high_uncertainty_threshold - STAGE_C_TAU) > 1e-7
        ):
            raise ValueError(
                f"Stage C reliability agent requires high_uncertainty_threshold={STAGE_C_TAU}, "
                f"got {self.config.high_uncertainty_threshold}."
            )

        # 4. Axial Context Window: k == 2
        if self.config.axial_context_window != STAGE_C_AXIAL_CONTEXT_RADIUS:
            raise ValueError(
                f"Stage C reliability agent requires axial_context_window={STAGE_C_AXIAL_CONTEXT_RADIUS}, "
                f"got {self.config.axial_context_window}."
            )

        # 5. Minimum Foreground Neighbours: N_min == 1
        if self.config.min_neighbor_foreground_slices != STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES:
            raise ValueError(
                f"Stage C reliability agent requires min_neighbor_foreground_slices={STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES}, "
                f"got {self.config.min_neighbor_foreground_slices}."
            )

        # 6. Axial Context Enabled
        if not self.config.enable_axial_context:
            raise ValueError(
                "Stage C reliability agent requires enable_axial_context=True."
            )

        # 7. Empty Prediction Fallback Enabled
        if not self.config.enable_empty_prediction_fallback:
            raise ValueError(
                "Stage C reliability agent requires enable_empty_prediction_fallback=True."
            )

        # 8. Fallback when context unavailable disabled (False)
        if self.config.fallback_when_context_unavailable:
            raise ValueError(
                "Stage C reliability agent requires fallback_when_context_unavailable=False."
            )

        # 9. Uncertainty-gated fallback disabled (False)
        if self.config.enable_uncertainty_fallback:
            raise ValueError(
                "Stage C reliability agent does not permit uncertainty-gated fallback (must be False)."
            )

        # 10. Segmentation Classes: 4
        if self.config.num_classes != 4:
            raise ValueError(
                f"Stage C reliability agent requires num_classes=4, got {self.config.num_classes}."
            )

        # 11. Temperature Calibration Verification
        expected_p_temp = self.calibration_config.temperature_for(STAGE_C_PRIMARY_MODEL)
        expected_s_temp = self.calibration_config.temperature_for(STAGE_C_SECONDARY_MODEL)

        if abs(self.config.primary_temperature - expected_p_temp) > 1e-6:
            raise ValueError(
                f"Stage C primary temperature ({self.config.primary_temperature}) does not match "
                f"verified Stage C2 calibration temperature ({expected_p_temp}) for '{STAGE_C_PRIMARY_MODEL}'."
            )

        sec_temp = self.config.candidate_temperatures.get(STAGE_C_SECONDARY_MODEL)
        if sec_temp is None or abs(sec_temp - expected_s_temp) > 1e-6:
            raise ValueError(
                f"Stage C secondary temperature ({sec_temp}) does not match "
                f"verified Stage C2 calibration temperature ({expected_s_temp}) for '{STAGE_C_SECONDARY_MODEL}'."
            )

        if set(self.config.candidate_temperatures.keys()) != {STAGE_C_SECONDARY_MODEL}:
            raise ValueError(
                f"Stage C candidate temperatures must contain exclusively ('{STAGE_C_SECONDARY_MODEL}',), "
                f"got {set(self.config.candidate_temperatures.keys())}."
            )

        ref_p_temp = STAGE_C2_REFERENCE_TEMPERATURES.get(STAGE_C_PRIMARY_MODEL)
        ref_s_temp = STAGE_C2_REFERENCE_TEMPERATURES.get(STAGE_C_SECONDARY_MODEL)
        if ref_p_temp is not None and abs(expected_p_temp - ref_p_temp) > 1e-6:
            raise ValueError(
                f"Calibration config primary temperature ({expected_p_temp}) deviates from "
                f"authoritative Stage C2 reference ({ref_p_temp})."
            )
        if ref_s_temp is not None and abs(expected_s_temp - ref_s_temp) > 1e-6:
            raise ValueError(
                f"Calibration config secondary temperature ({expected_s_temp}) deviates from "
                f"authoritative Stage C2 reference ({ref_s_temp})."
            )

    @classmethod
    def create_stage_c(
        cls,
        registry: ModelRegistry | None = None,
        calibration_config: StageCCalibrationConfig | None = None,
        tau: float = STAGE_C_TAU,
        axial_context_window: int = STAGE_C_AXIAL_CONTEXT_RADIUS,
        min_neighbor_foreground_slices: int = STAGE_C_MIN_NEIGHBOR_FOREGROUND_SLICES,
        device: torch.device | str | None = None,
        calib_path: Path | str | None = None,
        manifest_path: Path | str | None = None,
    ) -> InferenceAgent:
        """Classmethod constructor for Stage C reliability agent."""
        return create_stage_c_agent(
            registry=registry,
            calibration_config=calibration_config,
            tau=tau,
            axial_context_window=axial_context_window,
            min_neighbor_foreground_slices=min_neighbor_foreground_slices,
            device=device,
            calib_path=calib_path,
            manifest_path=manifest_path,
        )

    @property
    def secondary_model(self) -> str | None:
        """Identifier of the secondary model in the fallback chain."""
        return self.config.fallback_priority[0] if self.config.fallback_priority else None

    @property
    def primary_temperature(self) -> float:
        """Calibrated temperature scaling parameter for the primary model."""
        return self.config.primary_temperature

    @property
    def secondary_temperature(self) -> float | None:
        """Calibrated temperature scaling parameter for the secondary model, if configured."""
        if self.secondary_model:
            return self.config.candidate_temperatures.get(self.secondary_model)
        return None

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
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")

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
                # Boundary check: out of volume boundaries count as unavailable
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
