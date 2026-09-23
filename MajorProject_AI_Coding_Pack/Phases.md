# Phases

## Phase 0 — Environment
Python 3.11.9, dedicated D:\MajorProject\.venv, CUDA-enabled PyTorch, RTX 4070 verified with an actual CUDA computation.

## Phase 1 — Dataset
Download BraTS-GLI training/validation, retain mapping and README, extract safely, inspect cases, verify modalities/masks, detect missing/corrupt data.

## Phase 2 — Clean preprocessing
Load NIfTI, validate dimensions/orientation/metadata, normalize intensity, choose 2D/volume strategy, build reproducible data pipeline.

## Phase 3 — First baseline
Implement 2D U-Net, train on GPU, save checkpoints and curves, generate prediction visualizations, calculate Dice/IoU/precision/recall.

## Phase 4 — Model benchmark
Implement representative U-Net variants and other agreed architectures under controlled conditions. Never fabricate missing results.

## Phase 5 — Experimental analysis
Study selected factors such as epochs, batch size, dataset fraction, augmentation, modality configuration, model complexity, training time, and inference time.

## Phase 6 — Heterogeneous/degraded conditions
Only after the clean baseline is stable. Use mentor-approved degradation/noise types and severity levels. Preserve clean references.

## Phase 7 — Uncertainty
Implement and evaluate a defined uncertainty/reliability mechanism.

## Phase 8 — Agentic selection
Use defined condition/reliability information to select a segmentation strategy/model/modality configuration.

## Phase 9 — Final evaluation
Compare fixed and dynamic strategies using segmentation quality, reliability, robustness, and compute cost.

## Phase 10 — Report and viva
Convert actual evidence into the report, diagrams, tables, limitations, and viva explanations.
