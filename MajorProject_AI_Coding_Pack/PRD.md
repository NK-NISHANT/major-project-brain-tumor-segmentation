# PRD — Project Requirements Document

## Working Project Title
Uncertainty-Aware Agent for Reliable Brain Tumor Segmentation Under Heterogeneous MRI Conditions

## Mentor Direction
“Uncertainty-aware agent dynamically selects segmentation strategies, modalities, and specialized models for reliable brain tumor segmentation under heterogeneous MRI conditions.”

## Goal
Build a development-backed research project for brain tumor MRI segmentation. First establish clean-image segmentation baselines, then study heterogeneous/degraded conditions, uncertainty/reliability, and finally an agentic selection layer.

## Core capabilities
- Load and validate BraTS-GLI MRI data.
- Preprocess clean MRI data.
- Train/evaluate representative segmentation models.
- Visualize MRI, ground truth, and predictions.
- Calculate segmentation metrics.
- Compare models under controlled conditions.
- Introduce controlled degradation only after clean baselines.
- Estimate uncertainty/reliability.
- Dynamically select an appropriate model/strategy/modality configuration.
- Log experiments for reproducibility.

## Non-goals
- Do not force LLM, RAG, NLP, Generative AI, or MCP into the project without a real use.
- Do not fabricate results.
- Do not claim U-Net or another established architecture as an original invention.
- Do not modify raw dataset files.

## First demonstrable milestone
Clean BraTS-GLI → preprocessing → 2D U-Net → segmentation → Dice/IoU/precision/recall → visualization.
