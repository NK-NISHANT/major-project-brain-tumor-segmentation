import pandas as pd
import json
from pathlib import Path

models = [
    ('resunet_2d', 'results/full_training_15ep/resunet_2d'),
    ('unet_resnet34', 'results/full_training_15ep/unet_resnet34'),
    ('unet', 'results/stage_c/phase1/unet_721_seed42'),
    ('unetpp', 'results/stage_c/phase1/unetpp_721_seed42'),
    ('attention_unet', 'results/stage_c/phase1/attention_unet_721_seed42'),
]

for name, path_str in models:
    p = Path(path_str)
    hist = pd.read_csv(p / 'history.csv')
    with open(p / 'summary.json') as f:
        summ = json.load(f)
    print(f"================ {name} ================")
    params = summ.get('parameters', 'N/A')
    print(f"Parameters: {params:,}" if isinstance(params, int) else f"Parameters: {params}")
    best_ep = summ.get('best_epoch')
    best_dice = summ.get('best_mean_dice')
    print(f"Best Epoch: {best_ep} | Best Val Mean Dice: {best_dice:.4f}")
    ep15 = hist[hist['epoch'] == 15].iloc[0]
    print(f"Epoch 15: Val Mean Dice: {ep15['mean_dice']:.4f} | Val Loss: {ep15['val_loss']:.4f} | Train Loss: {ep15['train_loss']:.4f}")
    print("Epochs 11-15 Progression:")
    for _, r in hist[hist['epoch'] >= 11].iterrows():
        lr_val = r.get('learning_rate', r.get('learning_rate_primary', 0.0))
        print(f"  Ep {int(r['epoch']):02d}: TrainLoss={r['train_loss']:.4f} | ValLoss={r['val_loss']:.4f} | ValDice={r['mean_dice']:.4f} | ValIoU={r['mean_iou']:.4f} | LR={lr_val:.2e} | Best={r['is_best']}")
