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

full_data = {}
for name, p_str in models:
    p = Path(p_str)
    hist = pd.read_csv(p / 'history.csv')
    with open(p / 'summary.json') as f:
        summ = json.load(f)
    with open(p / 'config.json') as f:
        cfg = json.load(f)
    full_data[name] = {
        'config': cfg,
        'summary': summ,
        'history': hist.to_dict(orient='records')
    }

with open('tools/extracted_5models_data.json', 'w') as f:
    json.dump(full_data, f, indent=2)

print("Saved tools/extracted_5models_data.json successfully")
