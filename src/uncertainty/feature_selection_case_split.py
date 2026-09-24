from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


DATA_PATH = (
    "results/uncertainty_unet/"
    "model_uncertainty_comparison.csv"
)

INPUT_CONDITION_PATH = (
    "results/input_condition/"
    "case_input_reliability.csv"
)

MODELS = [
    "UNet",
    "UNetPlusPlus",
    "AttentionUNet",
]


def main() -> None:
    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    data = pd.read_csv(DATA_PATH)

    input_features = pd.read_csv(
        INPUT_CONDITION_PATH
    )

    input_feature_columns = [
        "t1n_nonzero_fraction",
        "t1n_mean",
        "t1n_std",
        "t1c_nonzero_fraction",
        "t1c_mean",
        "t1c_std",
        "t2f_nonzero_fraction",
        "t2f_mean",
        "t2f_std",
        "t2w_nonzero_fraction",
        "t2w_mean",
        "t2w_std",
    ]

    input_features = input_features[
        ["case_id"] + input_feature_columns
    ]

    # ---------------------------------------------------------
    # FILTER VALID SLICES
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # DETERMINE DICE-BEST MODEL
    # ---------------------------------------------------------

    # For every slice, determine which model achieved
    # the highest actual Dice score.
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

    # Target = 1 if this model is the Dice-best model
    # for the corresponding slice.
    data["target"] = (
        data["model"] == data["best_model"]
    ).astype(int)

    # Convert model identity into a numeric feature.
    data["model_code"] = (
        data["model"].map(
            {
                "UNet": 0,
                "UNetPlusPlus": 1,
                "AttentionUNet": 2,
            }
        )
    )

    # ---------------------------------------------------------
    # ADD CASE-LEVEL MRI CONDITION FEATURES
    # ---------------------------------------------------------

    data = data.merge(
        input_features,
        on="case_id",
        how="inner",
    )

    # ---------------------------------------------------------
    # FEATURE SETS
    # ---------------------------------------------------------

    # Existing baseline:
    # model identity + predictive uncertainty +
    # predicted foreground size.
    baseline_features = [
        "model_code",
        "predicted_foreground_uncertainty",
        "pred_foreground_pixels",
    ]

    # Condition-aware version:
    # baseline features + raw MRI input-condition features.
    condition_features = (
        baseline_features
        + input_feature_columns
    )

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

    # ---------------------------------------------------------
    # PREPARE TRAINING / TEST FEATURES
    # ---------------------------------------------------------

    X_train_baseline = train_data[
        baseline_features
    ]

    X_train_condition = train_data[
        condition_features
    ]

    y_train = train_data["target"]

    X_test_baseline = test_data[
        baseline_features
    ]

    X_test_condition = test_data[
        condition_features
    ]

    # ---------------------------------------------------------
    # TRAIN BASELINE RANDOM FOREST
    # ---------------------------------------------------------

    baseline_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    baseline_classifier.fit(
        X_train_baseline,
        y_train,
    )

    # ---------------------------------------------------------
    # TRAIN CONDITION-AWARE RANDOM FOREST
    # ---------------------------------------------------------

    condition_classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )

    condition_classifier.fit(
        X_train_condition,
        y_train,
    )

    # ---------------------------------------------------------
    # GENERATE TEST SCORES
    # ---------------------------------------------------------

    test_data = test_data.copy()

    test_data["baseline_score"] = (
        baseline_classifier.predict_proba(
            X_test_baseline
        )[:, 1]
    )

    test_data["condition_score"] = (
        condition_classifier.predict_proba(
            X_test_condition
        )[:, 1]
    )

    # ---------------------------------------------------------
    # MODEL SELECTION: BASELINE
    # ---------------------------------------------------------

    baseline_selected = (
        test_data.loc[
            test_data.groupby(
                ["case_id", "slice_index"]
            )["baseline_score"].idxmax()
        ]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    # ---------------------------------------------------------
    # MODEL SELECTION: CONDITION-AWARE
    # ---------------------------------------------------------

    condition_selected = (
        test_data.loc[
            test_data.groupby(
                ["case_id", "slice_index"]
            )["condition_score"].idxmax()
        ]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    # ---------------------------------------------------------
    # ORACLE
    # ---------------------------------------------------------

    oracle_models = (
        baseline_selected["best_model"]
    )

    # ---------------------------------------------------------
    # AGREEMENT
    # ---------------------------------------------------------

    baseline_agreement = (
        baseline_selected["model"].to_numpy()
        == oracle_models.to_numpy()
    ).mean()

    condition_agreement = (
        condition_selected["model"].to_numpy()
        == oracle_models.to_numpy()
    ).mean()

    # ---------------------------------------------------------
    # RESULTS
    # ---------------------------------------------------------

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
        "\nBaseline agreement "
        "(uncertainty + foreground pixels):",
        f"{baseline_agreement:.4f}",
    )

    print(
        "Baseline agreement percentage:",
        f"{baseline_agreement * 100:.2f}%",
    )

    print(
        "\nCondition-aware agreement "
        "(+ MRI condition features):",
        f"{condition_agreement:.4f}",
    )

    print(
        "Condition-aware agreement percentage:",
        f"{condition_agreement * 100:.2f}%",
    )

    print(
        "\nAgreement change:",
        f"{(condition_agreement - baseline_agreement) * 100:+.2f} percentage points",
    )

    print(
        "\nBaseline selected model counts:"
    )

    print(
        baseline_selected["model"].value_counts()
    )

    print(
        "\nCondition-aware selected model counts:"
    )

    print(
        condition_selected["model"].value_counts()
    )

    print(
        "\nOracle model counts:"
    )

    print(
        oracle_models.value_counts()
    )


if __name__ == "__main__":
    main()