import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
import hashlib
from src.models.unet_resnet34 import UNetResNet34

for fn in ['best_model.pt', 'final_model.pt']:
    p = 'results/full_training_15ep/unet_resnet34/' + fn
    with open(p, 'rb') as f:
        h = hashlib.sha256(f.read()).hexdigest()
    ckpt = torch.load(p, map_location='cpu')
    model = UNetResNet34(in_channels=4, num_classes=4, pretrained=False)
    msg = model.load_state_dict(ckpt['model_state_dict'], strict=True)
    m_dice = ckpt.get('metrics', {}).get('mean_dice')
    print(f"{fn}:")
    print(f"  SHA-256: {h}")
    print(f"  Epoch: {ckpt.get('epoch')}")
    print(f"  Mean Dice: {m_dice:.4f}")
    print(f"  Strict Load: {msg}")
