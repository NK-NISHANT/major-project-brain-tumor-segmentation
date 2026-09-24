from __future__ import annotations

import pandas as pd


UNCERTAINTY_PATH = (
    "results/uncertainty_unet/predictive_uncertainty.csv"
)

METRICS_PATH = (
    "results/uncertainty_unet/slice_uncertainty.csv"
)


def main() -> None:
    uncertainty_data = pd.read_csv(UNCERTAINTY_PATH)

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

    # Keep foreground slices with a valid predicted-foreground
    # uncertainty value.
    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    # Divide uncertainty into five equally populated groups.
    data["uncertainty_bin"] = pd.qcut(
        data["predicted_foreground_uncertainty"],
        q=5,
        labels=[
            "Very Low",
            "Low",
            "Medium",
            "High",
            "Very High",
        ],
    )

    summary = (
        data.groupby(
            "uncertainty_bin",
            observed=False,
        )
        .agg(
            slices=("mean_dice", "count"),
            mean_uncertainty=(
                "predicted_foreground_uncertainty",
                "mean",
            ),
            mean_dice=("mean_dice", "mean"),
            mean_iou=("mean_iou", "mean"),
        )
        .reset_index()
    )

    print("\nUncertainty Bin Analysis")
    print("=" * 70)

    print(
        summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()