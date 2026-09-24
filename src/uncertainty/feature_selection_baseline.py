from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)


MODELS = [
    "UNet",
    "UNetPlusPlus",
    "AttentionUNet",
]


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    # We need all three models to have a valid
    # predictive uncertainty value.
    data = data[
        data["has_foreground"] == True
    ].copy()

    data = data[
        data["predicted_foreground_uncertainty"].notna()
    ].copy()

    valid_counts = (
        data.groupby(
            ["case_id", "slice_index"]
        )["model"]
        .nunique()
    )

    complete_cases = valid_counts[
        valid_counts == len(MODELS)
    ].index

    data = data.set_index(
        ["case_id", "slice_index"]
    )

    data = data.loc[
        complete_cases
    ].reset_index()

    # Determine the oracle best model using Dice.
    best_model = (
        data.loc[
            data.groupby(
                ["case_id", "slice_index"]
            )["mean_dice"].idxmax()
        ][
            [
                "case_id",
                "slice_index",
                "model",
            ]
        ]
        .rename(
            columns={
                "model": "best_model"
            }
        )
    )

    data = data.merge(
        best_model,
        on=[
            "case_id",
            "slice_index",
        ],
    )

    data["target"] = (
        data["model"] == data["best_model"]
    ).astype(int)

    # Encode model identity.
    data["model_code"] = (
        data["model"].map(
            {
                "UNet": 0,
                "UNetPlusPlus": 1,
                "AttentionUNet": 2,
            }
        )
    )

    feature_columns = [
        "model_code",
        "predicted_foreground_uncertainty",
        "pred_foreground_pixels",
    ]

    X = data[feature_columns]
    y = data["target"]

    # Random forest baseline.
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    classifier.fit(X, y)

    data["selection_score"] = (
        classifier.predict_proba(X)[:, 1]
    )

    # Select the model with the highest predicted
    # probability of being the Dice-best model.
    selected = (
        data.loc[
            data.groupby(
                ["case_id", "slice_index"]
            )["selection_score"].idxmax()
        ]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    selected_models = selected["model"]
    oracle_models = selected["best_model"]

    accuracy = (
        selected_models == oracle_models
    ).mean()

    print("\nFeature-Based Selection Baseline")
    print("=" * 75)

    print(
        "Comparable slices:",
        len(selected),
    )

    print(
        "Selection agreement:",
        f"{accuracy:.4f}",
    )

    print(
        "Selection agreement percentage:",
        f"{accuracy * 100:.2f}%",
    )

    print("\nSelected model counts:")
    print(
        selected_models.value_counts()
    )

    print("\nOracle model counts:")
    print(
        oracle_models.value_counts()
    )


if __name__ == "__main__":
    main()