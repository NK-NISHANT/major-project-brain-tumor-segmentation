import json
import os
import sys

sys.path.insert(0, os.path.abspath("."))

import pandas as pd
import torch

experiments = {
    "ResUNet-2D (Stage B)": "results/full_training_15ep/resunet_2d",
    "UNetResNet34 (Stage B)": "results/full_training_15ep/unet_resnet34",
    "U-Net 2D (Stage C)": "results/stage_c/phase1/unet_721_seed42",
    "U-Net++ 2D (Stage C)": "results/stage_c/phase1/unetpp_721_seed42",
    "Attention U-Net 2D (Stage C)": "results/stage_c/phase1/attention_unet_721_seed42",
}

print("=" * 80)
print("FIVE-MODEL DIRECTORY & CONFIG AUDIT")
print("=" * 80)

for name, exp_dir in experiments.items():
    print(f"\n>>> Model: {name} ({exp_dir})")
    if not os.path.exists(exp_dir):
        print("  DIRECTORY NOT FOUND!")
        continue
    files = sorted(os.listdir(exp_dir))
    print(f"  Files: {files}")

    # Config
    cfg_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        print("  Config:")
        for k, v in sorted(cfg.items()):
            print(f"    {k}: {v}")
    else:
        print("  Config: NOT FOUND")

    # Summary
    sum_path = os.path.join(exp_dir, "summary.json")
    if os.path.exists(sum_path):
        with open(sum_path, "r") as f:
            summary = json.load(f)
        print("  Summary:")
        for k, v in summary.items():
            if k != "final_epoch_metrics":
                print(f"    {k}: {v}")
    else:
        print("  Summary: NOT FOUND")

    # History
    hist_path = os.path.join(exp_dir, "history.csv")
    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path)
        print(f"  History columns ({len(df)} rows): {list(df.columns)}")
        print("  Full Epoch Records:")
        # select relevant columns
        cols = [c for c in df.columns if c in [
            'epoch', 'train_loss', 'val_loss', 'mean_dice', 'val_dice', 'mean_iou', 'val_iou',
            'dice_class_0', 'dice_class_1', 'dice_class_2', 'dice_class_3',
            'dice_wt', 'dice_tc', 'dice_et',
            'learning_rate_primary', 'learning_rate', 'lr', 'is_best'
        ]]
        print(df[cols].to_string(index=False))
    else:
        print("  History: NOT FOUND")
