from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


UNCERTAINTY_PATH = (
    "results/uncertainty_unet/predictive_uncertainty.csv"
)

METRICS_PATH = (
    "results/uncertainty_unet/slice_uncertainty.csv"
)


def main() -> None:
    uncertainty_data = pd.read_csv(UNCERTAINTY_PATH)
    metrics_data = pd.read_csv(METRICS_PATH)

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

    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    uncertainty = data["predicted_foreground_uncertainty"]
    dice = data["mean_dice"]
    iou = data["mean_iou"]

    uncertainty_dice = spearmanr(
        uncertainty,
        dice,
    )

    uncertainty_iou = spearmanr(
        uncertainty,
        iou,
    )

    print("Merged rows:", len(data))

    print("\nPredictive uncertainty vs Dice")
    print(
        "Spearman correlation:",
        uncertainty_dice.statistic,
    )
    print(
        "p-value:",
        uncertainty_dice.pvalue,
    )

    print("\nPredictive uncertainty vs IoU")
    print(
        "Spearman correlation:",
        uncertainty_iou.statistic,
    )
    print(
        "p-value:",
        uncertainty_iou.pvalue,
    )


if __name__ == "__main__":
    main()