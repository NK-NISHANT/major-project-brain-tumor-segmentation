from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.dataset import discover_valid_cases, split_cases


INPUT_FEATURES = Path(
    "results/input_condition/case_input_features.csv"
)

MODEL_COMPARISON = Path(
    "results/uncertainty_unet/model_uncertainty_comparison.csv"
)

OUTPUT_DIR = Path("results/input_condition")

OUTPUT_CASE_FILE = (
    OUTPUT_DIR / "case_input_reliability.csv"
)

OUTPUT_CORRELATION_FILE = (
    OUTPUT_DIR / "input_reliability_correlations.csv"
)


def main() -> None:
    # ---------------------------------------------------------
    # 1. Load the same cases and split used by the project
    # ---------------------------------------------------------
    cases = discover_valid_cases()

    _, val_cases = split_cases(
        cases,
        val_fraction=0.2,
        seed=42,
    )

    validation_case_ids = {
        case.case_id for case in val_cases
    }

    print(f"Total complete cases: {len(cases)}")
    print(f"Validation cases: {len(validation_case_ids)}")

    # ---------------------------------------------------------
    # 2. Load raw MRI input-condition features
    # ---------------------------------------------------------
    input_df = pd.read_csv(INPUT_FEATURES)

    input_df = input_df[
        input_df["case_id"].isin(validation_case_ids)
    ].copy()

    print(
        f"Input-condition rows after validation filtering: "
        f"{len(input_df)}"
    )

    # ---------------------------------------------------------
    # 3. Load model-specific slice-level reliability results
    # ---------------------------------------------------------
    reliability_df = pd.read_csv(MODEL_COMPARISON)

    reliability_df = reliability_df[
        reliability_df["case_id"].isin(validation_case_ids)
    ].copy()

    # ---------------------------------------------------------
    # 4. Aggregate reliability at case level
    # ---------------------------------------------------------
    case_rows = []

    for case_id, group in reliability_df.groupby("case_id"):

        row = {
            "case_id": case_id,
        }

        for model in sorted(group["model"].unique()):

            model_df = group[
                group["model"] == model
            ].copy()

            valid_uncertainty = model_df[
                model_df["predicted_foreground_uncertainty"].notna()
            ]

            row[
                f"{model}_mean_dice"
            ] = model_df["mean_dice"].mean()

            row[
                f"{model}_mean_iou"
            ] = model_df["mean_iou"].mean()

            row[
                f"{model}_mean_uncertainty"
            ] = valid_uncertainty[
                "predicted_foreground_uncertainty"
            ].mean()

            row[
                f"{model}_high_uncertainty_fraction"
            ] = (
                valid_uncertainty[
                    "predicted_foreground_uncertainty"
                ] >= 0.15
            ).mean()

            row[
                f"{model}_complete_miss_fraction"
            ] = (
                (model_df["gt_foreground_pixels"] > 0)
                & (model_df["pred_foreground_pixels"] == 0)
            ).mean()

        case_rows.append(row)

    reliability_case_df = pd.DataFrame(case_rows)

    # ---------------------------------------------------------
    # 5. Merge input condition + reliability
    # ---------------------------------------------------------
    merged = input_df.merge(
        reliability_case_df,
        on="case_id",
        how="inner",
    )

    merged.to_csv(
        OUTPUT_CASE_FILE,
        index=False,
    )

    print(
        f"Saved case-level dataset: {OUTPUT_CASE_FILE}"
    )
    print(f"Case rows: {len(merged)}")

    # ---------------------------------------------------------
    # 6. Correlation analysis
    # ---------------------------------------------------------
    feature_columns = [
        column
        for column in merged.columns
        if any(
            column.endswith(f"_{feature}")
            for feature in (
                "nonzero_fraction",
                "mean",
                "std",
            )
        )
    ]

    reliability_columns = [
        column
        for column in merged.columns
        if any(
            metric in column
            for metric in (
                "mean_dice",
                "mean_iou",
                "mean_uncertainty",
                "high_uncertainty_fraction",
                "complete_miss_fraction",
            )
        )
    ]

    correlation_rows = []

    for feature in feature_columns:

        for target in reliability_columns:

            valid = merged[
                [feature, target]
            ].dropna()

            if len(valid) < 5:
                continue

            if valid[feature].nunique() < 2:
                continue

            if valid[target].nunique() < 2:
                continue

            rho, p_value = spearmanr(
                valid[feature],
                valid[target],
            )

            correlation_rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "n_cases": len(valid),
                    "spearman_rho": float(rho),
                    "p_value": float(p_value),
                    "abs_rho": float(abs(rho)),
                }
            )

    correlation_df = pd.DataFrame(
        correlation_rows
    ).sort_values(
        "abs_rho",
        ascending=False,
    )

    correlation_df.to_csv(
        OUTPUT_CORRELATION_FILE,
        index=False,
    )

    print(
        f"Saved correlations: "
        f"{OUTPUT_CORRELATION_FILE}"
    )

    print()
    print("Top 15 absolute correlations:")
    print(
        correlation_df.head(15).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()