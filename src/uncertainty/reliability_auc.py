from __future__ import annotations

import pandas as pd
from sklearn.metrics import roc_auc_score


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)

DICE_THRESHOLD = 0.50


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    data["poor_segmentation"] = (
        data["mean_dice"] < DICE_THRESHOLD
    ).astype(int)

    print("\nReliability ROC-AUC Analysis")
    print("=" * 70)
    print(
        f"Poor segmentation definition: "
        f"Dice < {DICE_THRESHOLD}"
    )

    for model_name in [
        "UNet",
        "UNetPlusPlus",
        "AttentionUNet",
    ]:
        model_data = data[
            data["model"] == model_name
        ].copy()

        y_true = model_data["poor_segmentation"]

        uncertainty = (
            model_data[
                "predicted_foreground_uncertainty"
            ]
        )

        auc = roc_auc_score(
            y_true,
            uncertainty,
        )

        print(f"\n{model_name}")
        print("-" * 40)
        print(
            "Valid slices:",
            len(model_data),
        )
        print(
            "Poor slices:",
            int(y_true.sum()),
        )
        print(
            "ROC-AUC:",
            f"{auc:.4f}",
        )


if __name__ == "__main__":
    main()