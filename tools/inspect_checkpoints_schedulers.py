import json
from pathlib import Path
import torch

models = [
    ('resunet_2d', 'results/full_training_15ep/resunet_2d'),
    ('unet_resnet34', 'results/full_training_15ep/unet_resnet34'),
    ('unet', 'results/stage_c/phase1/unet_721_seed42'),
    ('unetpp', 'results/stage_c/phase1/unetpp_721_seed42'),
    ('attention_unet', 'results/stage_c/phase1/attention_unet_721_seed42'),
]

for name, path_str in models:
    p = Path(path_str)
    print(f"=== {name} ===")
    with open(p / 'config.json') as f:
        cfg = json.load(f)
    print('LR/Opt params in config:', {k: cfg[k] for k in ['base_lr', 'min_lr', 'warmup_epochs', 'epochs', 'batch_size', 'weight_decay', 'seed', 'optimizer'] if k in cfg})
    
    ckpt_path = p / 'final_model.pt'
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print('Checkpoint keys:', list(ckpt.keys()))
        if 'scheduler_state_dict' in ckpt:
            sched = ckpt['scheduler_state_dict']
            print('Scheduler state keys:', list(sched.keys()) if isinstance(sched, dict) else type(sched))
            if isinstance(sched, dict):
                for sk in ['last_epoch', '_step_count', 'base_lrs', 'eta_min', 'T_max', '_get_lr_called_within_step']:
                    if sk in sched:
                        print(f"  sched[{sk}] = {sched[sk]}")
        if 'optimizer_state_dict' in ckpt:
            opt = ckpt['optimizer_state_dict']
            if 'param_groups' in opt:
                print('Optimizer param_groups count:', len(opt['param_groups']))
                for i, pg in enumerate(opt['param_groups']):
                    print(f"  group {i} current lr: {pg.get('lr')}, weight_decay: {pg.get('weight_decay')}")
        for extra in ['scaler_state_dict', 'amp_scaler_state_dict', 'rng_state', 'cuda_rng_state']:
            if extra in ckpt:
                print(f"Contains {extra}: Yes")
            else:
                print(f"Contains {extra}: No")
        del ckpt
