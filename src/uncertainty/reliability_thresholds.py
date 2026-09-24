from __future__ import annotations

import pandas as pd


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)

DICE_THRESHOLD = 0.50

UNCERTAINTY_THRESHOLDS = [
    0.05,
    0.07,
    0.09,
    0.11,
    0.13,
    0.15,
    0.20,
    0.25,
]


def calculate_metrics(
    data: pd.DataFrame,
    uncertainty_threshold: float,
) -> dict[str, float]:

    predicted_poor = (
        data["predicted_foreground_uncertainty"]
        >= uncertainty_threshold
    )

    actual_poor = (
        data["mean_dice"]
        < DICE_THRESHOLD
    )

    tp = int((predicted_poor & actual_poor).sum())
    fp = int((predicted_poor & ~actual_poor).sum())
    tn = int((~predicted_poor & ~actual_poor).sum())
    fn = int((~predicted_poor & actual_poor).sum())

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else float("nan")
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else float("nan")
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else float("nan")
    )

    return {
        "threshold": uncertainty_threshold,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    }


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    # Ground-truth foreground slices only.
    # Also remove cases where the model predicted no
    # foreground because predictive foreground entropy
    # is undefined there.
    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    print("\nReliability Threshold Analysis")
    print("=" * 95)

    print(
        f"Poor segmentation definition: Dice < {DICE_THRESHOLD}"
    )

    for model_name in [
        "UNet",
        "UNetPlusPlus",
        "AttentionUNet",
    ]:

        model_data = data[
            data["model"] == model_name
        ].copy()

        print(f"\n{model_name}")
        print("-" * 95)

        print(
            "Threshold | TP | FP | TN | FN | "
            "Sensitivity | Specificity | Precision"
        )

        for threshold in UNCERTAINTY_THRESHOLDS:

            result = calculate_metrics(
                model_data,
                threshold,
            )

            print(
                f"{result['threshold']:9.2f} | "
                f"{result['TP']:2d} | "
                f"{result['FP']:3d} | "
                f"{result['TN']:3d} | "
                f"{result['FN']:3d} | "
                f"{result['sensitivity']:.4f} | "
                f"{result['specificity']:.4f} | "
                f"{result['precision']:.4f}"
            )


if __name__ == "__main__":
    main()