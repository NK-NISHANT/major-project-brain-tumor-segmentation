from __future__ import annotations

import pandas as pd


UNCERTAINTY_PATH = (
    "results/uncertainty_unet/model_uncertainty_comparison.csv"
)

METRICS_PATH = (
    "results/uncertainty_unet/slice_uncertainty.csv"
)


def main() -> None:
    uncertainty_data = pd.read_csv(
        UNCERTAINTY_PATH
    )

    metrics_data = pd.read_csv(
        METRICS_PATH
    )

    data = uncertainty_data.merge(
        metrics_data[
            [
                "case_id",
                "slice_index",
                "mean_dice",
                "mean_iou",
            ]
        ],
        on=["case_id", "slice_index"],
        how="inner",
    )

    # Only slices containing ground-truth foreground.
    data = data[
        data["has_foreground"] == True
    ].copy()

    print("\nUnified Reliability Summary")
    print("=" * 80)

    for model_name in [
        "UNet",
        "UNetPlusPlus",
        "AttentionUNet",
    ]:
        model_data = data[
            data["model"] == model_name
        ].copy()

        total_slices = len(model_data)

        complete_misses = (
            model_data["pred_foreground_pixels"] == 0
        ).sum()

        non_miss_data = model_data[
            model_data["pred_foreground_pixels"] > 0
        ].copy()

        miss_rate = (
            complete_misses / total_slices * 100
        )

        mean_uncertainty = (
            non_miss_data[
                "predicted_foreground_uncertainty"
            ].mean()
        )

        mean_dice = model_data["mean_dice"].mean()
        mean_iou = model_data["mean_iou"].mean()

        print(f"\n{model_name}")
        print("-" * 50)

        print(
            "Foreground slices:",
            total_slices,
        )

        print(
            "Complete misses:",
            complete_misses,
        )

        print(
            "Complete miss rate:",
            f"{miss_rate:.2f}%",
        )

        print(
            "Mean predictive uncertainty:",
            f"{mean_uncertainty:.4f}",
        )

        print(
            "Mean Dice:",
            f"{mean_dice:.4f}",
        )

        print(
            "Mean IoU:",
            f"{mean_iou:.4f}",
        )


if __name__ == "__main__":
    main()