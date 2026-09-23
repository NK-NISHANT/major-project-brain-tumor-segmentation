# Architecture

## Overall evolution
Clean MRI → validation → preprocessing → model benchmark → controlled heterogeneity/degradation → uncertainty estimation → agentic decision → selected segmentation strategy → reliability report.

## Initial system
Raw BraTS-GLI → dataset inspector → preprocessor → slice/volume dataset → U-Net trainer → evaluator → visualization.

## Final direction
MRI input → condition/quality information → uncertainty estimation → agent → select modality/preprocessing/model → inference → segmentation + uncertainty/reliability report.

## Candidate models
1. U-Net
2. U-Net++
3. Attention U-Net
4. ResUNet / residual U-Net
5. U-Net with ResNet encoder
6. TransUNet
7. 3D U-Net only if data/hardware design permits.

## Stack
Python 3.11, PyTorch + CUDA, MONAI, NiBabel, NumPy, pandas, scikit-learn, matplotlib, tqdm, Git/GitHub, Codex/local AI coding assistance.

## Hard constraints
No Keras, TensorFlow, or MATLAB. Use PyTorch for deep learning. Preserve raw data and make experiments reproducible.
