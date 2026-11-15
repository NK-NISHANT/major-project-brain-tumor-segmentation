import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hashlib
import json
import torch

models_info = [
    {
        'name': 'resunet_2d',
        'dir': 'results/full_training_15ep/resunet_2d',
        'module': 'src.models.resunet',
        'class': 'ResUNet2D',
        'expected_epoch': 13,
    },
    {
        'name': 'unet_resnet34',
        'dir': 'results/full_training_15ep/unet_resnet34',
        'module': 'src.models.unet_resnet34',
        'class': 'UNetResNet34',
        'expected_epoch': 15,
    },
    {
        'name': 'unet',
        'dir': 'results/stage_c/phase1/unet_721_seed42',
        'module': 'src.models.unet',
        'class': 'UNet2D',
        'expected_epoch': 15,
    },
    {
        'name': 'unetpp',
        'dir': 'results/stage_c/phase1/unetpp_721_seed42',
        'module': 'src.models.unetpp',
        'class': 'UNetPlusPlus2D',
        'expected_epoch': 15,
    },
    {
        'name': 'attention_unet',
        'dir': 'results/stage_c/phase1/attention_unet_721_seed42',
        'module': 'src.models.attention_unet',
        'class': 'AttentionUNet2D',
        'expected_epoch': 15,
    },
]

for m in models_info:
    d = Path(m['dir'])
    ckpt_path = d / 'best_model.pt'
    cfg_path = d / 'config.json'
    summ_path = d / 'summary.json'
    
    with open(ckpt_path, 'rb') as f:
        ckpt_sha = hashlib.sha256(f.read()).hexdigest()
    
    with open(cfg_path) as f:
        cfg = json.load(f)
    with open(summ_path) as f:
        summ = json.load(f)
        
    ckpt = torch.load(ckpt_path, map_location='cpu')
    
    print(f"=== {m['name']} ===")
    print(f"Path: {ckpt_path}")
    print(f"SHA-256: {ckpt_sha}")
    print(f"Recorded Best Epoch in ckpt: {ckpt.get('epoch')}, Expected: {m['expected_epoch']}")
    print(f"Recorded Best Epoch in summary: {summ.get('best_epoch')}")
    print(f"Recorded Best Val Mean Dice in summary: {summ.get('best_mean_dice'):.4f}")
    
    # Load model class and verify state_dict
    mod = __import__(m['module'], fromlist=[m['class']])
    cls = getattr(mod, m['class'])
    if m['name'] == 'unet_resnet34':
        model = cls(in_channels=4, num_classes=4, pretrained=False)
    else:
        model = cls(in_channels=4, num_classes=4)
        
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Instantiated parameter count: {param_count:,} (Config params: {cfg.get('parameters'):,})")
    
    # Check load state dict
    res = model.load_state_dict(ckpt['model_state_dict'], strict=True)
    print(f"State dict strict load: missing={len(res.missing_keys)}, unexpected={len(res.unexpected_keys)}")
    del model
    del ckpt
