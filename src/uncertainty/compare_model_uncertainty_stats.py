from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    print("\nCorrected Model-wise Predictive Uncertainty Analysis")
    print("=" * 75)

    for model_name in [
        "UNet",
        "UNetPlusPlus",
        "AttentionUNet",
    ]:
        model_data = data[
            data["model"] == model_name
        ].copy()

        uncertainty = (
            model_data[
                "predicted_foreground_uncertainty"
            ]
        )

        dice = model_data["mean_dice"]
        iou = model_data["mean_iou"]

        dice_result = spearmanr(
            uncertainty,
            dice,
        )

        iou_result = spearmanr(
            uncertainty,
            iou,
        )

        print(f"\n{model_name}")
        print("-" * 45)

        print(
            "Valid slices:",
            len(model_data),
        )

        print(
            "Uncertainty vs Dice:",
            f"rho={dice_result.statistic:.4f}",
            f"p={dice_result.pvalue:.4e}",
        )

        print(
            "Uncertainty vs IoU:",
            f"rho={iou_result.statistic:.4f}",
            f"p={iou_result.pvalue:.4e}",
        )


if __name__ == "__main__":
    main()