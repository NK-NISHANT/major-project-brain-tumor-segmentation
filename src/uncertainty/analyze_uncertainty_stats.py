from __future__ import annotations

import pandas as pd
from scipy.stats import spearmanr


CSV_PATH = "results/uncertainty_unet/slice_uncertainty.csv"


def main() -> None:
    data = pd.read_csv(CSV_PATH)

    foreground_data = data[data["has_foreground"] == True].copy()

    uncertainty = foreground_data["foreground_uncertainty"]
    dice = foreground_data["mean_dice"]
    iou = foreground_data["mean_iou"]

    uncertainty_dice = spearmanr(uncertainty, dice)
    uncertainty_iou = spearmanr(uncertainty, iou)

    print("Total rows:", len(data))
    print("Foreground rows:", len(foreground_data))

    print("\nUncertainty vs Dice")
    print("Spearman correlation:", uncertainty_dice.statistic)
    print("p-value:", uncertainty_dice.pvalue)

    print("\nUncertainty vs IoU")
    print("Spearman correlation:", uncertainty_iou.statistic)
    print("p-value:", uncertainty_iou.pvalue)


if __name__ == "__main__":
    main()