from __future__ import annotations

import csv

import torch

from src.data.dataset import create_train_val_datasets
from src.evaluation.metrics import SegmentationMetricAccumulator
from src.models.unet import UNet2D
from src.models.unetpp import UNetPlusPlus2D
from src.models.attention_unet import AttentionUNet2D
from src.uncertainty.entropy import softmax_entropy


MODELS = {
    "UNet": (
        UNet2D,
        "results/baseline_unet2d_5ep/best_model.pt",
    ),
    "UNetPlusPlus": (
        UNetPlusPlus2D,
        "results/unetpp2d_5ep/best_model.pt",
    ),
    "AttentionUNet": (
        AttentionUNet2D,
        "results/attention_unet2d_5ep/best_model.pt",
    ),
}


OUTPUT_PATH = (
    "results/uncertainty_unet/model_uncertainty_comparison.csv"
)


def load_model(model_class, checkpoint_path, device):
    model = model_class(
        in_channels=4,
        num_classes=4,
        base_channels=32,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    _, val_dataset = create_train_val_datasets(
        val_fraction=0.2,
        background_slice_ratio=0.25,
        seed=42,
        case_cache_size=1,
    )

    print(
        "Validation cases:",
        val_dataset.case_count,
    )

    print(
        "Validation slices:",
        len(val_dataset),
    )

    loaded_models = {}

    for model_name, (model_class, checkpoint_path) in MODELS.items():
        print(f"\nLoading {model_name}...")

        loaded_models[model_name] = load_model(
            model_class,
            checkpoint_path,
            device,
        )

        print(f"{model_name} loaded.")

    fieldnames = [
        "model",
        "case_id",
        "slice_index",
        "has_foreground",
        "gt_foreground_pixels",
        "pred_foreground_pixels",
        "predicted_foreground_uncertainty",
        "mean_dice",
        "mean_iou",
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(len(val_dataset)):

            image, mask = val_dataset[index]
            record = val_dataset.slice_index[index]

            image = image.unsqueeze(0).to(device)
            target = mask.unsqueeze(0).to(device)

            gt_foreground_pixels = int(
                (mask > 0).sum().item()
            )

            for model_name, model in loaded_models.items():

                with torch.no_grad():
                    logits = model(image)

                uncertainty = softmax_entropy(logits)

                prediction = torch.argmax(
                    logits,
                    dim=1,
                )

                predicted_foreground = (
                    prediction[0] > 0
                )

                if predicted_foreground.any():
                    predicted_foreground_uncertainty = (
                        uncertainty[0][predicted_foreground]
                        .mean()
                        .item()
                    )
                else:
                    predicted_foreground_uncertainty = float(
                        "nan"
                    )

                pred_foreground_pixels = int(
                    predicted_foreground.sum().item()
                )

                metrics = SegmentationMetricAccumulator(
                    num_classes=4,
                    ignore_background=True,
                )

                metrics.update(
                    logits,
                    target,
                )

                result = metrics.compute()

                writer.writerow(
                    {
                        "model": model_name,
                        "case_id": record.case_id,
                        "slice_index": record.slice_index,
                        "has_foreground": record.has_foreground,
                        "gt_foreground_pixels":
                            gt_foreground_pixels,
                        "pred_foreground_pixels":
                            pred_foreground_pixels,
                        "predicted_foreground_uncertainty":
                            predicted_foreground_uncertainty,
                        "mean_dice":
                            result.mean_dice,
                        "mean_iou":
                            result.mean_iou,
                    }
                )

            if (
                (index + 1) % 100 == 0
                or index == 0
            ):
                print(
                    f"Processed "
                    f"{index + 1}/{len(val_dataset)} slices"
                )

    print(
        f"\nSaved results to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()