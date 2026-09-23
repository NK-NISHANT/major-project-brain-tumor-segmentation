# Memory / Project State

## Current state
- Project root: D:\MajorProject
- Python 3.11.9
- PyTorch 2.11.0+cu128
- CUDA available
- NVIDIA GeForce RTX 4070 Laptop GPU
- Actual CUDA computation verified
- BraTS-GLI training/validation downloads in progress

## Current milestone
Dataset acquisition and environment setup.

## Next milestone
Inspect extracted BraTS-GLI structure and build the dataset inspector.

## Immediate next tasks
1. Finish downloads.
2. Install medical-imaging dependencies.
3. Extract safely.
4. Inspect modalities and segmentation masks.
5. Build clean preprocessing.
6. Train first U-Net.

## Important decisions
- Clean-image baseline first.
- Noise/degradation later.
- Start with 2D due practical 8 GB VRAM constraints.
- Preserve raw data.
- Do not force MCP/LLM/RAG/etc.

## AI workflow
ChatGPT: architecture, research framing, experiment design, review, viva.
Codex: scoped implementation/debugging.
User: run, verify, understand, document.
