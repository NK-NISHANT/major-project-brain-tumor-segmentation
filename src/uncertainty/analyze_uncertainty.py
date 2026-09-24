from __future__ import annotations
from src.uncertainty.entropy import softmax_entropy
import torch
from src.evaluation.metrics import SegmentationMetricAccumulator
from src.data.dataset import create_train_val_datasets
from src.models.unet import UNet2D
import csv

CHECKPOINT_PATH = "results/baseline_unet2d_5ep/best_model.pt"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    _, val_dataset = create_train_val_datasets(
        val_fraction=0.2,
        background_slice_ratio=0.25,
        seed=42,
        case_cache_size=1,
    )

    print("Validation cases:", val_dataset.case_count)
    print("Validation slices:", len(val_dataset))

    model = UNet2D(
        in_channels=4,
        num_classes=4,
        base_channels=32,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    output_path = "results/uncertainty_unet/slice_uncertainty.csv"

    fieldnames = [
        "case_id",
        "slice_index",
        "has_foreground",
        "gt_foreground_pixels",
        "pred_foreground_pixels",
        "foreground_uncertainty",
        "mean_dice",
        "mean_iou",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(len(val_dataset)):
            image, mask = val_dataset[index]
            record = val_dataset.slice_index[index]

            image = image.unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(image)

            uncertainty = softmax_entropy(logits)

            foreground_mask = mask > 0

            if foreground_mask.any():
                foreground_uncertainty = (
                    uncertainty[0][foreground_mask].mean().item()
                )
            else:
                foreground_uncertainty = float("nan")

            metrics = SegmentationMetricAccumulator(
                num_classes=4,
                ignore_background=True,
            )

            metrics.update(logits, mask.unsqueeze(0))
            result = metrics.compute()

            prediction = torch.argmax(logits, dim=1)

            gt_foreground_pixels = int((mask > 0).sum().item())
            pred_foreground_pixels = int((prediction > 0).sum().item())

            writer.writerow(
                {
                    "case_id": record.case_id,
                    "slice_index": record.slice_index,
                    "has_foreground": record.has_foreground,
                    "gt_foreground_pixels": gt_foreground_pixels,
                    "pred_foreground_pixels": pred_foreground_pixels,
                    "foreground_uncertainty": foreground_uncertainty,
                    "mean_dice": result.mean_dice,
                    "mean_iou": result.mean_iou,
                }
            )

            if (index + 1) % 100 == 0 or index == 0:
                print(
                    f"Processed {index + 1}/{len(val_dataset)} slices"
                )

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()