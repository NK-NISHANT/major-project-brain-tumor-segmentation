import json
import os
import pandas as pd

exp_dir = "results/full_training_15ep/unet_resnet34"
print(f"Directory: {exp_dir}")
with open(os.path.join(exp_dir, "config.json")) as f:
    print("Config:", json.dumps(json.load(f), indent=2))
with open(os.path.join(exp_dir, "summary.json")) as f:
    print("Summary:", json.dumps(json.load(f), indent=2))

df = pd.read_csv(os.path.join(exp_dir, "history.csv"))
print(f"\nHistory ({len(df)} rows):")
cols = ['epoch', 'train_loss', 'val_loss', 'mean_dice', 'mean_iou', 'dice_class_0', 'dice_class_1', 'dice_class_2', 'dice_class_3', 'learning_rate_primary', 'learning_rate_encoder', 'is_best']
available_cols = [c for c in cols if c in df.columns]
print(df[available_cols].to_string(index=False))
