from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

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

# Canonical model names and default checkpoint locations verified on disk
# Canonical baseline model names and default checkpoint locations verified on disk
CANONICAL_MODEL_NAMES = ("unet", "unetpp", "attention_unet")
CANDIDATE_MODEL_NAMES = ("resunet", "unet_resnet34")

MODEL_FACTORIES: Mapping[str, type[nn.Module]] = {
    "unet": UNet2D,
    "unetpp": UNetPlusPlus2D,
    "attention_unet": AttentionUNet2D,
    "resunet": ResUNet2D,
    "unet_resnet34": UNetResNet34,
}

# Verified historical baseline checkpoints present on disk
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


class ModelRegistry:
    """Lazy-loading model registry and common inference interface for segmentation models.

    Memory Behavior:
        - Checkpoints and models are NOT instantiated or loaded into memory/VRAM at initialization.
        - Models are lazily loaded onto the designated device upon the first `get_model` or
          `predict_slice` call.
        - Loaded models are retained in an internal dictionary (`_loaded_models`) to avoid repeated disk reads.
        - `unload(model_name)` or `unload()` can be called to evict models and release GPU VRAM.
    """

    def __init__(
        self,
        checkpoint_paths: Mapping[str, Path | str] | None = None,
        default_device: torch.device | str | None = None,
    ) -> None:
        self.checkpoint_paths: dict[str, Path] = {}
        for canonical in CANONICAL_MODEL_NAMES:
            default_path = DEFAULT_CHECKPOINTS[canonical]
            if checkpoint_paths is not None:
                # Check if specified by canonical or alias
                found_path = None
                for key, path_val in checkpoint_paths.items():
                    if normalize_model_name(key) == canonical:
                        found_path = Path(path_val)
                        break
                self.checkpoint_paths[canonical] = found_path if found_path is not None else default_path
            else:
                self.checkpoint_paths[canonical] = default_path

        for candidate in CANDIDATE_MODEL_NAMES:
            default_path = CANDIDATE_CHECKPOINTS[candidate]
            if checkpoint_paths is not None:
                found_path = None
                for key, path_val in checkpoint_paths.items():
                    if normalize_model_name(key) == candidate:
                        found_path = Path(path_val)
                        break
                self.checkpoint_paths[candidate] = found_path if found_path is not None else default_path
            else:
                self.checkpoint_paths[candidate] = default_path

        if default_device is None:
            self.default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.default_device = torch.device(default_device)

        self._loaded_models: dict[str, nn.Module] = {}

    @property
    def loaded_model_names(self) -> tuple[str, ...]:
        """Names of models currently loaded in memory."""
        return tuple(self._loaded_models.keys())

    def get_model(
        self,
        model_name: str,
        device: torch.device | str | None = None,
    ) -> nn.Module:
        """Lazily load and return the requested neural network module.

        Args:
            model_name: Name or alias of the model ("unet", "unetpp", "attention_unet").
            device: Target execution device. Defaults to `self.default_device`.

        Returns:
            nn.Module placed on target device in eval() mode.

        Raises:
            FileNotFoundError: If the configured checkpoint does not exist on disk.
            ValueError: If the model name is unrecognized.
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

        checkpoint_path = self.checkpoint_paths[canonical]
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint file for '{canonical}' not found at {checkpoint_path}."
            )

        model_class = MODEL_FACTORIES[canonical]

        # All existing project models share 4 input channels, 4 output classes, 32 base channels
        if canonical == "unetpp":
            model = model_class(in_channels=4, num_classes=4, base_channels=32, deep_supervision=False)
        elif canonical == "unet_resnet34":
            model = model_class(in_channels=4, num_classes=4, base_channels=32, pretrained=False)
        else:
            model = model_class(in_channels=4, num_classes=4, base_channels=32)

        checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=False)
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
