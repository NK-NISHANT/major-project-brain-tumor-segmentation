from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MODALITIES, create_train_val_datasets  # noqa: E402
from src.evaluation.metrics import SegmentationMetricAccumulator  # noqa: E402
from src.models.unet import UNet2D  # noqa: E402


DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "results" / "baseline_unet2d_5ep" / "best_model.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "baseline_unet2d_5ep" / "visualizations"


@dataclass(frozen=True)
class SelectedSlice:
    dataset_index: int
    case_id: str
    axial_slice_index: int
    foreground_pixels: int
    foreground_percent: float
    unique_labels: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize predictions from the trained 2D U-Net baseline.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--background-slice-ratio", type=float, default=0.25)
    parser.add_argument("--case-cache-size", type=int, default=1)
    parser.add_argument("--modality", choices=MODALITIES, default="t1c")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_dataset = create_train_val_datasets(
        val_fraction=args.val_fraction,
        background_slice_ratio=args.background_slice_ratio,
        seed=args.seed,
        case_cache_size=args.case_cache_size,
    )

    model = load_model(args.checkpoint, device)
    selected_slices = select_representative_foreground_slices(val_dataset, args.num_samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "checkpoint": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "num_validation_slices": len(val_dataset),
        "modality": args.modality,
        "label_note": "Segmentation labels are shown only as numeric class IDs 0, 1, 2, 3.",
        "samples": [],
    }

    modality_index = MODALITIES.index(args.modality)
    for display_index, selected in enumerate(selected_slices, start=1):
        image, mask = val_dataset[selected.dataset_index]
        prediction, metrics = run_inference(model, image, mask, device)
        output_path = args.output_dir / (
            f"sample_{display_index:02d}_{selected.case_id}_slice_{selected.axial_slice_index:03d}.png"
        )
        save_visualization(
            output_path=output_path,
            image=image[modality_index].numpy(),
            mask=mask.numpy(),
            prediction=prediction,
            modality=args.modality,
            selected=selected,
            metrics=metrics,
        )

        sample_summary = asdict(selected)
        sample_summary["image_path"] = str(output_path)
        sample_summary["metrics"] = metrics.as_dict()
        summary["samples"].append(sample_summary)
        print(f"Saved visualization: {output_path}")

    summary_path = args.output_dir / "visualization_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(f"Saved summary: {summary_path}")


def load_model(checkpoint_path: Path, device: torch.device) -> UNet2D:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = UNet2D(in_channels=4, num_classes=4, base_channels=32).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_representative_foreground_slices(
    dataset,
    num_samples: int,
) -> list[SelectedSlice]:
    candidates: list[SelectedSlice] = []

    for dataset_index, record in enumerate(dataset.slice_index):
        if not record.has_foreground:
            continue

        _, mask = dataset[dataset_index]
        foreground_pixels = int((mask > 0).sum().item())
        if foreground_pixels == 0:
            continue

        total_pixels = int(mask.numel())
        candidates.append(
            SelectedSlice(
                dataset_index=dataset_index,
                case_id=record.case_id,
                axial_slice_index=record.slice_index,
                foreground_pixels=foreground_pixels,
                foreground_percent=(foreground_pixels / total_pixels) * 100.0,
                unique_labels=[int(value) for value in torch.unique(mask).tolist()],
            )
        )

    if not candidates:
        raise RuntimeError("No foreground validation slices were found.")

    # Deterministic and presentation-friendly: prefer slices with enough visible
    # foreground, while avoiding three adjacent slices from the same case.
    candidates.sort(
        key=lambda sample: (
            -sample.foreground_pixels,
            sample.case_id,
            sample.axial_slice_index,
        )
    )

    selected: list[SelectedSlice] = []
    used_cases: set[str] = set()
    for candidate in candidates:
        if candidate.case_id in used_cases:
            continue
        selected.append(candidate)
        used_cases.add(candidate.case_id)
        if len(selected) == num_samples:
            return selected

    for candidate in candidates:
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == num_samples:
            return selected

    return selected


@torch.inference_mode()
def run_inference(
    model: UNet2D,
    image: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
):
    image_batch = image.unsqueeze(0).to(device=device)
    mask_batch = mask.unsqueeze(0).to(device=device)
    logits = model(image_batch)
    prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    metric_accumulator = SegmentationMetricAccumulator(num_classes=4, ignore_background=True)
    metric_accumulator.update(logits, mask_batch)
    metrics = metric_accumulator.compute()
    return prediction, metrics


def save_visualization(
    output_path: Path,
    image: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
    modality: str,
    selected: SelectedSlice,
    metrics,
) -> None:
    image_display = normalize_for_display(image)
    overlay = build_foreground_overlay(image_display, mask, prediction)

    class_cmap = ListedColormap(["black", "#d62728", "#2ca02c", "#1f77b4"])
    class_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], class_cmap.N)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4.5), dpi=180, constrained_layout=True)
    fig.suptitle(
        (
            f"{selected.case_id} | axial slice {selected.axial_slice_index} | "
            f"foreground {selected.foreground_percent:.2f}% | "
            f"mean Dice {metrics.mean_dice:.3f} | mean IoU {metrics.mean_iou:.3f}"
        ),
        fontsize=10,
    )

    axes[0].imshow(image_display, cmap="gray")
    axes[0].set_title(f"MRI modality: {modality}", fontsize=9)

    mask_plot = axes[1].imshow(mask, cmap=class_cmap, norm=class_norm, interpolation="nearest")
    axes[1].set_title("Ground truth classes", fontsize=9)

    pred_plot = axes[2].imshow(prediction, cmap=class_cmap, norm=class_norm, interpolation="nearest")
    axes[2].set_title("Predicted classes", fontsize=9)

    axes[3].imshow(overlay)
    axes[3].set_title("Foreground overlay", fontsize=9)
    axes[3].text(
        0.02,
        0.98,
        "green=GT\nmagenta=prediction\nwhite=overlap",
        color="white",
        fontsize=7,
        transform=axes[3].transAxes,
        va="top",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )

    for axis in axes:
        axis.axis("off")

    colorbar = fig.colorbar(pred_plot, ax=axes[1:3], fraction=0.035, pad=0.02, ticks=[0, 1, 2, 3])
    colorbar.ax.set_ylabel("Class ID", rotation=270, labelpad=12)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def normalize_for_display(image: np.ndarray) -> np.ndarray:
    finite_image = np.nan_to_num(image.astype(np.float32), copy=False)
    nonzero = finite_image[finite_image != 0]
    if nonzero.size == 0:
        return np.zeros_like(finite_image)

    lower, upper = np.percentile(nonzero, [1, 99])
    if upper <= lower:
        lower = float(nonzero.min())
        upper = float(nonzero.max())
    if upper <= lower:
        return np.zeros_like(finite_image)

    display = np.clip((finite_image - lower) / (upper - lower), 0.0, 1.0)
    return display


def build_foreground_overlay(
    image_display: np.ndarray,
    mask: np.ndarray,
    prediction: np.ndarray,
) -> np.ndarray:
    overlay = np.stack([image_display, image_display, image_display], axis=-1)
    gt_foreground = mask > 0
    pred_foreground = prediction > 0

    overlay[gt_foreground] = (0.0, 0.85, 0.0)
    overlay[pred_foreground] = (0.95, 0.0, 0.95)
    overlay[np.logical_and(gt_foreground, pred_foreground)] = (1.0, 1.0, 1.0)
    return overlay


if __name__ == "__main__":
    main()
