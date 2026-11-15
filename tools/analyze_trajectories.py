import json

with open('tools/extracted_5models_data.json') as f:
    data = json.load(f)

for name, d in data.items():
    hist = d['history']
    print(f"==================== {name} ====================")
    episodes = {r['epoch']: r for r in hist}
    
    # Epoch 11-15 analysis
    print("Epochs 11 to 15:")
    for ep in range(11, 16):
        r = episodes[ep]
        lr = r.get('learning_rate', r.get('learning_rate_primary'))
        print(f"  Ep {ep:02d} | TrainLoss: {r['train_loss']:.4f} | ValLoss: {r['val_loss']:.4f} | ValDice: {r['mean_dice']:.4f} | ValIoU: {r['mean_iou']:.4f} | ET: {r['dice_class_1']:.4f} | ED: {r['dice_class_2']:.4f} | NCR: {r['dice_class_3']:.4f} | LR: {lr:.2e} | is_best: {r['is_best']}")

    # Deltas between 11 and 15
    r11 = episodes[11]
    r14 = episodes[14]
    r15 = episodes[15]
    best_ep = d['summary']['best_epoch']
    r_best = episodes[best_ep]

    print(f"\nDeltas & Trajectories:")
    print(f"  Best Epoch: {best_ep} (Val Dice: {r_best['mean_dice']:.4f}, Val Loss: {r_best['val_loss']:.4f})")
    print(f"  Train Loss: Ep 11 ({r11['train_loss']:.4f}) -> Ep 15 ({r15['train_loss']:.4f}) | Delta: {r15['train_loss'] - r11['train_loss']:+.4f}")
    print(f"  Val Loss:   Ep 11 ({r11['val_loss']:.4f}) -> Ep 15 ({r15['val_loss']:.4f}) | Delta: {r15['val_loss'] - r11['val_loss']:+.4f}")
    min_val_loss_ep = min(hist, key=lambda x: x['val_loss'])
    print(f"  Min Val Loss was at Ep {min_val_loss_ep['epoch']} ({min_val_loss_ep['val_loss']:.4f})")
    print(f"  Val Dice:   Ep 11 ({r11['mean_dice']:.4f}) -> Ep 15 ({r15['mean_dice']:.4f}) | Delta: {r15['mean_dice'] - r11['mean_dice']:+.4f}")
    print(f"  Val Dice:   Ep 14 ({r14['mean_dice']:.4f}) -> Ep 15 ({r15['mean_dice']:.4f}) | Delta: {r15['mean_dice'] - r14['mean_dice']:+.4f}")
    print(f"  Val Dice:   Best ({r_best['mean_dice']:.4f}) vs Ep 15 ({r15['mean_dice']:.4f}) | Delta: {r15['mean_dice'] - r_best['mean_dice']:+.4f}")

    # Check per-class Dice deltas between Ep 11 and 15
    for c_id, c_name in [(1, 'ET'), (2, 'ED'), (3, 'NCR')]:
        k = f'dice_class_{c_id}'
        print(f"  {c_name} Dice: Ep 11 ({r11[k]:.4f}) -> Ep 15 ({r15[k]:.4f}) | Delta: {r15[k] - r11[k]:+.4f} (Ep 14 -> 15: {r15[k] - r14[k]:+.4f})")
