# Segmentation Model Benchmark Results

## Experimental Setup

Dataset: BraTS 2023 GLI development subset
Train cases: 158
Validation cases: 39
Training slices: 13,074
Validation slices: 3,223

Input: 4 MRI modalities
Input shape: 4 x 240 x 240
Output: 4 classes

Batch size: 2
Learning rate: 1e-3
Optimizer: AdamW
Loss: Dice + Cross Entropy
Epochs: 5
Data augmentation: None
GPU: NVIDIA RTX 4070 Laptop GPU (8 GB)

The comparison is intended as a controlled architecture benchmark. Dataset split, preprocessing, training settings, loss, and evaluation metrics are kept consistent between models.

## U-Net

Parameters: 7,763,428
Best epoch: 5
Best validation mean Dice: 0.7711
Best validation mean IoU: 0.6329
Runtime: approximately 39.9 minutes

## U-Net++

Parameters: 9,160,068
Best epoch: 4
Best validation mean Dice: 0.793724
Best validation mean IoU: 0.658180
Runtime: 67.31 minutes

Per-class results at the best U-Net++ epoch:

| Class | Dice | IoU |
|---|---:|---:|
| 0 | 0.997690 | 0.995391 |
| 1 | 0.784552 | 0.645484 |
| 2 | 0.784979 | 0.646062 |
| 3 | 0.811642 | 0.682994 |

## Current Observation

Under the current 5-epoch experimental setup, U-Net++ achieved higher best validation mean Dice and mean IoU than the U-Net baseline, while using more parameters and requiring longer training time.

This is an experimental observation for the current development split and training configuration. It should not be interpreted as a general claim that U-Net++ is universally superior.

The best validation checkpoint is used rather than automatically using the final epoch because validation performance fluctuated across epochs.

## Next Model

Attention U-Net will be implemented and evaluated using the same experimental setup.
