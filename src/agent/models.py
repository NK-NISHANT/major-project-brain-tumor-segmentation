"""Model Registry and Architecture Definitions for Brain Tumor Segmentation.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Batch 1A: Stage C Checkpoint Profile & Provenance Foundation.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Mapping

import torch
from torch import nn

from src.agent.policy import extract_slice_prediction
from src.agent.state import SlicePrediction
from src.models.attention_unet import AttentionUNet2D
from src.models.resunet import ResUNet2D
from src.models.unet import UNet2D
from src.models.unet_resnet34 import UNetResNet34
from src.models.unetpp import UNetPlusPlus2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CheckpointProfile(str, Enum):
    """Explicit model checkpoint profiles.

    Profiles:
        LEGACY_DEFAULT: Mixed legacy default (5-epoch baselines + 15-epoch Gate 5D candidates
            if present on disk). Preserves exact pre-implementation behavior of ModelRegistry().
        GENUINE_5EP: Pure 5-epoch pilot checkpoints for all 5 architectures.
        STAGE_C_15EP: Authoritative Stage C 15-epoch scaled models with cryptographic SHA-256
            verification prior to deserialization.
        HISTORICAL_5EP: Backward compatibility alias for the legacy default.
    """

    LEGACY_DEFAULT = "legacy_default"
    GENUINE_5EP = "genuine_5ep"
    STAGE_C_15EP = "stage_c_15ep"
    HISTORICAL_5EP = "historical_5ep"


def normalize_profile(profile: CheckpointProfile | str | None) -> CheckpointProfile:
    """Normalize input profile identifier to CheckpointProfile enum."""
    if profile is None:
        return CheckpointProfile.LEGACY_DEFAULT
    if isinstance(profile, CheckpointProfile):
        return profile
    if isinstance(profile, str):
        cleaned = profile.strip().lower()
        if cleaned in ("legacy_default", "legacy", "default"):
            return CheckpointProfile.LEGACY_DEFAULT
        if cleaned in ("genuine_5ep", "strict_5ep", "pure_5ep"):
            return CheckpointProfile.GENUINE_5EP
        if cleaned in ("historical_5ep", "historical"):
            return CheckpointProfile.HISTORICAL_5EP
        if cleaned in ("stage_c_15ep", "stage_c", "15ep"):
            return CheckpointProfile.STAGE_C_15EP
        for member in CheckpointProfile:
            if member.value == cleaned or member.name.lower() == cleaned:
                return member
    valid_profiles = [p.value for p in CheckpointProfile]
    raise ValueError(
        f"Unknown checkpoint profile '{profile}'. Expected one of {valid_profiles}."
    )


# Canonical baseline model names and default checkpoint locations verified on disk
CANONICAL_MODEL_NAMES = ("unet", "unetpp", "attention_unet")
CANDIDATE_MODEL_NAMES = ("resunet", "unet_resnet34")

# All approved Stage C models
STAGE_C_MODELS = (
    "resunet",
    "unet_resnet34",
    "unet",
    "unetpp",
    "attention_unet",
)

MODEL_FACTORIES: Mapping[str, type[nn.Module]] = {
    "unet": UNet2D,
    "unetpp": UNetPlusPlus2D,
    "attention_unet": AttentionUNet2D,
    "resunet": ResUNet2D,
    "unet_resnet34": UNetResNet34,
}

# Verified historical baseline checkpoints present on disk (5-epoch pilot)
DEFAULT_CHECKPOINTS: Mapping[str, Path] = {
    "unet": PROJECT_ROOT / "results" / "baseline_unet2d_5ep" / "best_model.pt",
    "unetpp": PROJECT_ROOT / "results" / "unetpp2d_5ep" / "best_model.pt",
    "attention_unet": PROJECT_ROOT / "results" / "attention_unet2d_5ep" / "best_model.pt",
}

# Gate 5D full-scale 15-epoch checkpoints (primary leading model and comparison candidate)
GATE5D_CHECKPOINTS: Mapping[str, Path] = {
    "resunet": PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt",
    "unet_resnet34": PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt",
}

# Candidate checkpoints (defaults to Gate 5D 15-epoch models if present, else 5-epoch screen)
CANDIDATE_CHECKPOINTS: Mapping[str, Path] = {
    "resunet": (
        GATE5D_CHECKPOINTS["resunet"]
        if GATE5D_CHECKPOINTS["resunet"].exists()
        else PROJECT_ROOT / "results" / "resunet2d_5ep" / "best_model.pt"
    ),
    "unet_resnet34": (
        GATE5D_CHECKPOINTS["unet_resnet34"]
        if GATE5D_CHECKPOINTS["unet_resnet34"].exists()
        else PROJECT_ROOT / "results" / "unet_resnet34_5ep" / "best_model.pt"
    ),
}

# Genuinely 5-epoch pilot checkpoints for all 5 models (pure 5-epoch screening profile)
GENUINE_5EP_CHECKPOINTS: Mapping[str, Path] = {
    "unet": PROJECT_ROOT / "results" / "baseline_unet2d_5ep" / "best_model.pt",
    "unetpp": PROJECT_ROOT / "results" / "unetpp2d_5ep" / "best_model.pt",
    "attention_unet": PROJECT_ROOT / "results" / "attention_unet2d_5ep" / "best_model.pt",
    "resunet": PROJECT_ROOT / "results" / "resunet2d_5ep" / "best_model.pt",
    "unet_resnet34": PROJECT_ROOT / "results" / "unet_resnet34_5ep" / "best_model.pt",
}

# Authoritative Stage C 15-epoch scaled model checkpoints verified on disk
STAGE_C_CHECKPOINTS: Mapping[str, Path] = {
    "resunet": PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt",
    "unet_resnet34": PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt",
    "unet": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unet_721_seed42" / "best_model.pt",
    "unetpp": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "best_model.pt",
    "attention_unet": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "attention_unet_721_seed42" / "best_model.pt",
}

# Authoritative Stage C cryptographic SHA-256 identities
STAGE_C_EXPECTED_HASHES: Mapping[str, str] = {
    "resunet": "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d",
    "unet_resnet34": "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d",
    "unet": "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef",
    "unetpp": "fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6",
    "attention_unet": "5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e",
}

# Alias mapping for user convenience
MODEL_ALIASES: Mapping[str, str] = {
    "unet": "unet",
    "u-net": "unet",
    "unet2d": "unet",
    "unetpp": "unetpp",
    "unet++": "unetpp",
    "unetplusplus": "unetpp",
    "unetplusplus2d": "unetpp",
    "attention_unet": "attention_unet",
    "attentionunet": "attention_unet",
    "attention_unet2d": "attention_unet",
    "resunet": "resunet",
    "resunet_2d": "resunet",
    "res-unet": "resunet",
    "resunet2d": "resunet",
    "residual_unet": "resunet",
    "unet_resnet34": "unet_resnet34",
    "resnet34_unet": "unet_resnet34",
    "unet-resnet34": "unet_resnet34",
    "unet_resnet_34": "unet_resnet34",
    "resnet34": "unet_resnet34",
}


def normalize_model_name(name: str) -> str:
    """Normalize input model name to canonical identifier."""
    cleaned = name.strip().lower()
    if cleaned in MODEL_ALIASES:
        return MODEL_ALIASES[cleaned]
    valid_names = tuple(MODEL_FACTORIES.keys())
    raise ValueError(
        f"Unknown model name '{name}'. Expected one of {valid_names} (or valid aliases)."
    )


def compute_stream_sha256(stream: BinaryIO, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 digest of an open binary stream using chunked reads."""
    hasher = hashlib.sha256()
    while chunk := stream.read(chunk_size):
        hasher.update(chunk)
    return hasher.hexdigest()


def compute_file_sha256(path: Path | str, chunk_size: int = 1024 * 1024) -> str:
    """Calculate SHA-256 digest of a file using chunked reads without metadata caching."""
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        raise FileNotFoundError(f"Cannot compute SHA-256: file not found at {p}")
    with open(p, "rb") as f:
        return compute_stream_sha256(f, chunk_size=chunk_size)


def verify_checkpoint_sha256(
    path: Path | str,
    expected_sha256: str | None = None,
    model_name: str = "",
    profile: str = "",
) -> str:
    """Verify file integrity against expected SHA-256 by computing full digest (no metadata shortcut).

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the computed SHA-256 differs from expected_sha256.
    """
    p = Path(path).resolve()
    if not p.exists() or not p.is_file():
        model_info = f" for model '{model_name}'" if model_name else ""
        profile_info = f" under profile '{profile}'" if profile else ""
        raise FileNotFoundError(
            f"Checkpoint file{model_info}{profile_info} not found at {p}."
        )

    actual_sha256 = compute_file_sha256(p)

    if expected_sha256 is not None:
        expected_clean = expected_sha256.strip().lower()
        if actual_sha256 != expected_clean:
            model_info = f" for model '{model_name}'" if model_name else ""
            profile_info = f" under profile '{profile}'" if profile else ""
            raise ValueError(
                f"Checkpoint integrity verification failed{model_info}{profile_info}!\n"
                f"  File Path:        {p}\n"
                f"  Expected SHA-256: {expected_clean}\n"
                f"  Actual SHA-256:   {actual_sha256}"
            )

    return actual_sha256


class ModelRegistry:
    """Lazy-loading model registry and common inference interface supporting profiles.

    Backward-Compatibility Guarantees:
        - Default constructor ModelRegistry() without arguments retains exactly its
          pre-implementation historical checkpoint resolution behavior (LEGACY_DEFAULT).
        - Profile LEGACY_DEFAULT reproduces legacy mixed default mappings (5ep baseline, 15ep Gate 5D candidates).
        - Profile GENUINE_5EP resolves pure 5-epoch screening checkpoints for all 5 models.
        - Profile STAGE_C_15EP requires explicit selection and resolves authoritative
          15-epoch scaled models with cryptographic SHA-256 verification before loading.
        - Profile HISTORICAL_5EP is retained as a compatibility alias for the legacy default.
    """

    def __init__(
        self,
        checkpoint_paths: Mapping[str, Path | str] | None = None,
        default_device: torch.device | str | None = None,
        profile: CheckpointProfile | str | None = None,
    ) -> None:
        if profile is None:
            self.profile = CheckpointProfile.LEGACY_DEFAULT
        else:
            self.profile = normalize_profile(profile)
        self.checkpoint_paths: dict[str, Path] = {}

        if self.profile == CheckpointProfile.STAGE_C_15EP:
            base_mapping = STAGE_C_CHECKPOINTS
            models_to_resolve = STAGE_C_MODELS
        elif self.profile == CheckpointProfile.GENUINE_5EP:
            base_mapping = GENUINE_5EP_CHECKPOINTS
            models_to_resolve = CANONICAL_MODEL_NAMES + CANDIDATE_MODEL_NAMES
        else:
            # LEGACY_DEFAULT / HISTORICAL_5EP: exact preservation of pre-implementation behavior
            base_mapping = {**DEFAULT_CHECKPOINTS, **CANDIDATE_CHECKPOINTS}
            models_to_resolve = CANONICAL_MODEL_NAMES + CANDIDATE_MODEL_NAMES

        for canonical in models_to_resolve:
            default_path = base_mapping[canonical]
            if checkpoint_paths is not None:
                found_path = None
                for key, path_val in checkpoint_paths.items():
                    if normalize_model_name(key) == canonical:
                        found_path = Path(path_val)
                        break
                self.checkpoint_paths[canonical] = found_path if found_path is not None else default_path
            else:
                self.checkpoint_paths[canonical] = default_path

        if default_device is None:
            self.default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.default_device = torch.device(default_device)

        self._loaded_models: dict[str, nn.Module] = {}

    @property
    def loaded_model_names(self) -> tuple[str, ...]:
        """Names of models currently loaded in memory."""
        return tuple(self._loaded_models.keys())

    def verify_checkpoint(self, model_name: str) -> str:
        """Verify and return the SHA-256 digest of a specific model's checkpoint.

        For STAGE_C_15EP, validates against the authoritative expected hash.
        """
        canonical = normalize_model_name(model_name)
        if canonical not in self.checkpoint_paths:
            raise KeyError(
                f"Model '{canonical}' is not configured in profile '{self.profile.value}'."
            )
        expected_hash = (
            STAGE_C_EXPECTED_HASHES.get(canonical)
            if self.profile == CheckpointProfile.STAGE_C_15EP
            else None
        )
        return verify_checkpoint_sha256(
            self.checkpoint_paths[canonical],
            expected_sha256=expected_hash,
            model_name=canonical,
            profile=self.profile.value,
        )

    def verify_all_checkpoints(self) -> dict[str, str]:
        """Verify and return SHA-256 digests for all models in the active profile."""
        digests = {}
        for canonical in self.checkpoint_paths:
            digests[canonical] = self.verify_checkpoint(canonical)
        return digests

    def get_model(
        self,
        model_name: str,
        device: torch.device | str | None = None,
    ) -> nn.Module:
        """Lazily load and return the requested neural network module.

        Args:
            model_name: Name or alias of the model ("unet", "unetpp", "attention_unet", "resunet", "unet_resnet34").
            device: Target execution device. Defaults to `self.default_device`.

        Returns:
            nn.Module placed on target device in eval() mode.

        Raises:
            FileNotFoundError: If the configured checkpoint does not exist on disk.
            ValueError: If model name is unrecognized or checkpoint hash verification fails.
        """
        canonical = normalize_model_name(model_name)
        target_device = torch.device(device) if device is not None else self.default_device

        # If already loaded and on target device, return cached instance
        if canonical in self._loaded_models:
            model = self._loaded_models[canonical]
            first_param = next(model.parameters(), None)
            if first_param is not None and first_param.device != target_device:
                model.to(target_device)
            return model

        if canonical not in self.checkpoint_paths:
            raise KeyError(
                f"Model '{canonical}' is not configured in profile '{self.profile.value}'."
            )

        checkpoint_path = self.checkpoint_paths[canonical]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file for '{canonical}' under profile '{self.profile.value}' not found at {checkpoint_path}."
            )

        model_class = MODEL_FACTORIES[canonical]

        # All existing project models share 4 input channels, 4 output classes, 32 base channels
        if canonical == "unetpp":
            model = model_class(in_channels=4, num_classes=4, base_channels=32, deep_supervision=False)
        elif canonical == "unet_resnet34":
            model = model_class(in_channels=4, num_classes=4, base_channels=32, pretrained=False)
        else:
            model = model_class(in_channels=4, num_classes=4, base_channels=32)

        # For Stage C models:
        # 1. Open checkpoint once in binary mode
        # 2. Calculate SHA-256 using chunked reads from that open file handle
        # 3. Compare against authoritative expected hash
        # 4. On mismatch, raise explicit error (deserialization never occurs)
        # 5. Rewind the same file handle
        # 6. Pass that same verified file handle to torch.load()
        if self.profile == CheckpointProfile.STAGE_C_15EP:
            expected_hash = STAGE_C_EXPECTED_HASHES.get(canonical)
            with open(checkpoint_path, "rb") as f:
                actual_sha256 = compute_stream_sha256(f)
                if expected_hash is not None:
                    expected_clean = expected_hash.strip().lower()
                    if actual_sha256 != expected_clean:
                        raise ValueError(
                            f"Checkpoint integrity verification failed for model '{canonical}' under profile '{self.profile.value}'!\n"
                            f"  File Path:        {checkpoint_path}\n"
                            f"  Expected SHA-256: {expected_clean}\n"
                            f"  Actual SHA-256:   {actual_sha256}"
                        )
                f.seek(0)
                checkpoint = torch.load(f, map_location=target_device, weights_only=False)
        else:
            with open(checkpoint_path, "rb") as f:
                checkpoint = torch.load(f, map_location=target_device, weights_only=False)

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(target_device)
        model.eval()

        self._loaded_models[canonical] = model
        return model

    def unload(self, model_name: str | None = None) -> None:
        """Evict loaded model(s) from memory and release PyTorch CUDA cache.

        Args:
            model_name: Specific model to unload, or None to unload all models.
        """
        if model_name is None:
            self._loaded_models.clear()
        else:
            canonical = normalize_model_name(model_name)
            self._loaded_models.pop(canonical, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def predict_slice(
        self,
        model_name: str,
        image: torch.Tensor,
        device: torch.device | str | None = None,
        temperature: float = 1.0,
    ) -> SlicePrediction:
        """Run slice inference and return a typed SlicePrediction.

        Args:
            model_name: Identifier of the model to execute.
            image: 4-channel MRI slice tensor of shape [1, 4, H, W] or [4, H, W].
            device: Target execution device.
            temperature: Scalar temperature > 0 to scale logits prior to softmax/entropy (default 1.0).

        Returns:
            SlicePrediction containing class mask, pixel counts, and normalized entropy.

        Raises:
            TypeError: If image is not a torch.Tensor.
            ValueError: If input tensor dimensions, batch size, or channel counts are invalid.
        """
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"image must be a torch.Tensor, got {type(image)}")

        # Accept [4, H, W] by unsqueezing to [1, 4, H, W]
        if image.ndim == 3:
            if image.shape[0] != 4:
                raise ValueError(
                    f"Expected 4 modality channels at dim 0, got {image.shape[0]} (shape={tuple(image.shape)})"
                )
            image = image.unsqueeze(0)

        if image.ndim != 4:
            raise ValueError(
                f"Expected 4D image tensor with shape [1, 4, H, W], got ndim={image.ndim} (shape={tuple(image.shape)})"
            )

        if image.shape[0] != 1:
            raise ValueError(
                f"Expected single slice batch with batch_size=1, got shape={tuple(image.shape)}"
            )

        if image.shape[1] != 4:
            raise ValueError(
                f"Expected 4 modality channels at dim 1, got {image.shape[1]} (shape={tuple(image.shape)})"
            )

        if image.shape[2] <= 0 or image.shape[3] <= 0:
            raise ValueError(
                f"Spatial dimensions (H, W) must be positive, got ({image.shape[2]}, {image.shape[3]})"
            )

        canonical = normalize_model_name(model_name)
        target_device = torch.device(device) if device is not None else self.default_device

        model = self.get_model(canonical, target_device)
        image_input = image.to(target_device)

        logits = model(image_input)
        if temperature > 0.0 and temperature != 1.0:
            logits = logits / temperature

        return extract_slice_prediction(
            model_name=canonical,
            logits=logits,
            num_classes=4,
        )
