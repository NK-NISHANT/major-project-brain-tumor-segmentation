from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


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

    # Keep only slices with ground-truth foreground and
    # a valid predictive uncertainty value.
    data = data[
        (data["has_foreground"] == True)
        & data["predicted_foreground_uncertainty"].notna()
    ].copy()

    # Keep only slices where all three models have
    # a valid uncertainty value.
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

    # Determine the Dice-best model for every slice.
    oracle = (
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
        oracle,
        on=[
            "case_id",
            "slice_index",
        ],
    )

    data["target"] = (
        data["model"] == data["best_model"]
    ).astype(int)

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

    # ---------------------------------------------------------
    # CASE-WISE SPLIT
    # ---------------------------------------------------------

    cases = (
        data[
            ["case_id"]
        ]
        .drop_duplicates()
        .sample(
            frac=1.0,
            random_state=42,
        )
        ["case_id"]
        .tolist()
    )

    split_index = int(
        len(cases) * 0.80
    )

    train_cases = set(
        cases[:split_index]
    )

    test_cases = set(
        cases[split_index:]
    )

    train_data = data[
        data["case_id"].isin(train_cases)
    ].copy()

    test_data = data[
        data["case_id"].isin(test_cases)
    ].copy()

    X_train = train_data[
        feature_columns
    ]

    y_train = train_data["target"]

    X_test = test_data[
        feature_columns
    ]

    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    classifier.fit(
        X_train,
        y_train,
    )

    test_data = test_data.copy()

    test_data["selection_score"] = (
        classifier.predict_proba(X_test)[:, 1]
    )

    selected = (
        test_data.loc[
            test_data.groupby(
                ["case_id", "slice_index"]
            )["selection_score"].idxmax()
        ]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    selected_models = selected["model"]
    oracle_models = selected["best_model"]

    agreement = (
        selected_models == oracle_models
    )

    agreement_rate = agreement.mean()

    print(
        "\nCase-Wise Feature Selection Evaluation"
    )

    print("=" * 75)

    print(
        "Total cases:",
        len(cases),
    )

    print(
        "Training cases:",
        len(train_cases),
    )

    print(
        "Test cases:",
        len(test_cases),
    )

    print(
        "Training slices:",
        len(train_data),
    )

    print(
        "Test slices:",
        len(test_data),
    )

    print(
        "\nUnseen-case selection agreement:",
        f"{agreement_rate:.4f}",
    )

    print(
        "Unseen-case agreement percentage:",
        f"{agreement_rate * 100:.2f}%",
    )

    print(
        "\nSelected model counts:"
    )

    print(
        selected_models.value_counts()
    )

    print(
        "\nOracle model counts:"
    )

    print(
        oracle_models.value_counts()
    )


if __name__ == "__main__":
    main()