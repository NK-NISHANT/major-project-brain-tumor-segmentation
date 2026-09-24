from __future__ import annotations

import pandas as pd


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

    data = data[
        data["has_foreground"] == True
    ].copy()

    # Only keep model predictions where a predicted
    # foreground region exists and therefore predictive
    # foreground uncertainty is defined.
    data = data[
        data["predicted_foreground_uncertainty"].notna()
    ].copy()

    # Keep only slices where all three models have
    # a valid predictive uncertainty value.
    valid_model_counts = (
        data.groupby(
            ["case_id", "slice_index"]
        )["model"]
        .nunique()
    )

    complete_cases = valid_model_counts[
        valid_model_counts == len(MODELS)
    ].index

    data = data.set_index(
        ["case_id", "slice_index"]
    )

    data = data.loc[
        complete_cases
    ].reset_index()

    print("\nUncertainty-Based Model Selection")
    print("=" * 75)

    num_slices = (
        data[
            data["model"] == MODELS[0]
        ][
            ["case_id", "slice_index"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        "Slices with all three models producing "
        "valid foreground uncertainty:",
        num_slices,
    )

    # Select the model with the lowest predictive uncertainty.
    uncertainty_idx = (
        data.groupby(
            ["case_id", "slice_index"]
        )[
            "predicted_foreground_uncertainty"
        ].idxmin()
    )

    uncertainty_selection = (
        data.loc[uncertainty_idx]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    # Select the model with the highest actual Dice.
    # This is an oracle reference because ground truth
    # is required to calculate Dice.
    dice_idx = (
        data.groupby(
            ["case_id", "slice_index"]
        )["mean_dice"].idxmax()
    )

    dice_selection = (
        data.loc[dice_idx]
        .set_index(
            ["case_id", "slice_index"]
        )
    )

    uncertainty_models = (
        uncertainty_selection["model"]
    )

    dice_models = (
        dice_selection["model"]
    )

    agreement = (
        uncertainty_models == dice_models
    )

    agreement_rate = agreement.mean()

    print(
        "\nUncertainty selection vs Dice oracle"
    )

    print(
        "Agreement:",
        int(agreement.sum()),
        "/",
        len(agreement),
    )

    print(
        "Agreement rate:",
        f"{agreement_rate:.4f}",
    )

    print(
        "\nUncertainty-selected model counts:"
    )

    print(
        uncertainty_models.value_counts()
    )

    print(
        "\nDice-oracle model counts:"
    )

    print(
        dice_models.value_counts()
    )

    print(
        "\nSelection confusion table:"
    )

    confusion = pd.crosstab(
        dice_models,
        uncertainty_models,
        rownames=["Dice-best model"],
        colnames=["Uncertainty-selected model"],
    )

    print(confusion)


if __name__ == "__main__":
    main()