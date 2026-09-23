# PROJECT_SPEC.md

## Project
Uncertainty-Aware Agent for Reliable Brain Tumor Segmentation Under Heterogeneous MRI Conditions

## Current goal
Clean BraTS-GLI segmentation baseline → heterogeneous/degraded conditions → uncertainty → agentic dynamic selection.

## Hard constraints
PyTorch; no Keras; no TensorFlow; no MATLAB; preserve raw data; no fabricated results; no forced AI technologies.

## Environment
Python 3.11.9; PyTorch 2.11.0+cu128; CUDA available; NVIDIA RTX 4070 Laptop GPU; root D:\MajorProject.

## Current dataset
BraTS-GLI training and validation data, mapping spreadsheet, README.

## Current phase
Dataset acquisition / environment setup.

## Immediate next task
Inspect extracted BraTS-GLI data and create a dataset inspector.

## First demo
Clean MRI → preprocessing → 2D U-Net → segmentation → Dice/IoU/precision/recall → visualization.

## Later
Model benchmark → controlled degradation → uncertainty → agentic modality/strategy/model selection.

## AI coding
ChatGPT for architecture/research/review; Codex for scoped implementation; user runs, verifies, understands, and records.
