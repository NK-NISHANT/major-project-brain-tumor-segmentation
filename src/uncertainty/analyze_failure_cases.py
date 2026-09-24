from __future__ import annotations

import pandas as pd


UNCERTAINTY_PATH = (
    "results/uncertainty_unet/model_uncertainty_comparison.csv"
)


def main() -> None:
    data = pd.read_csv(UNCERTAINTY_PATH)

    # Only slices where the ground truth contains foreground.
    foreground_data = data[
        data["has_foreground"] == True
    ].copy()

    print("\nComplete Foreground Miss Analysis")
    print("=" * 70)

    for model_name in [
        "UNet",
        "UNetPlusPlus",
        "AttentionUNet",
    ]:
        model_data = foreground_data[
            foreground_data["model"] == model_name
        ].copy()

        total_foreground = len(model_data)

        complete_misses = (
            model_data["pred_foreground_pixels"] == 0
        ).sum()

        predicted_foreground = (
            model_data["pred_foreground_pixels"] > 0
        ).sum()

        miss_percentage = (
            complete_misses / total_foreground * 100
        )

        print(f"\n{model_name}")
        print("-" * 40)

        print(
            "Foreground slices:",
            total_foreground,
        )

        print(
            "Predicted foreground:",
            predicted_foreground,
        )

        print(
            "Complete foreground misses:",
            complete_misses,
        )

        print(
            "Complete miss percentage:",
            f"{miss_percentage:.2f}%",
        )


if __name__ == "__main__":
    main()