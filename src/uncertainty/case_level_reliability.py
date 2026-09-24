from __future__ import annotations

import pandas as pd


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)


HIGH_UNCERTAINTY_THRESHOLD = 0.15


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    # We need ground-truth foreground slices for this
    # exploratory reliability analysis.
    data = data[
        data["has_foreground"] == True
    ].copy()

    # Create a flag for complete foreground misses.
    data["complete_miss"] = (
        data["pred_foreground_pixels"] == 0
    )

    # Create a high-uncertainty flag where uncertainty
    # is defined.
    data["high_uncertainty"] = (
        data["predicted_foreground_uncertainty"]
        >= HIGH_UNCERTAINTY_THRESHOLD
    )

    summaries = []

    for (case_id, model), group in data.groupby(
        ["case_id", "model"]
    ):
        valid_uncertainty = group[
            "predicted_foreground_uncertainty"
        ].dropna()

        summary = {
            "case_id": case_id,
            "model": model,
            "foreground_slices": len(group),
            "mean_uncertainty": (
                valid_uncertainty.mean()
            ),
            "median_uncertainty": (
                valid_uncertainty.median()
            ),
            "p90_uncertainty": (
                valid_uncertainty.quantile(0.90)
            ),
            "high_uncertainty_fraction": (
                group["high_uncertainty"].mean()
            ),
            "complete_miss_fraction": (
                group["complete_miss"].mean()
            ),
            "mean_dice": group["mean_dice"].mean(),
            "mean_iou": group["mean_iou"].mean(),
        }

        summaries.append(summary)

    summary_data = pd.DataFrame(summaries)

    print("\nCase-Level Reliability Analysis")
    print("=" * 85)

    print(
        "Cases:",
        summary_data["case_id"].nunique(),
    )

    print(
        "Model-case combinations:",
        len(summary_data),
    )

    print(
        "\nOverall summary by model:"
    )

    model_summary = (
        summary_data
        .groupby("model")
        .agg(
            cases=("case_id", "count"),
            mean_uncertainty=(
                "mean_uncertainty",
                "mean",
            ),
            median_uncertainty=(
                "median_uncertainty",
                "mean",
            ),
            mean_high_uncertainty_fraction=(
                "high_uncertainty_fraction",
                "mean",
            ),
            mean_complete_miss_fraction=(
                "complete_miss_fraction",
                "mean",
            ),
            mean_dice=(
                "mean_dice",
                "mean",
            ),
            mean_iou=(
                "mean_iou",
                "mean",
            ),
        )
        .reset_index()
    )

    print(
        model_summary.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )


if __name__ == "__main__":
    main()