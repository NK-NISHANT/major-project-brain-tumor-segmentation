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

Random seed: 42

GPU: NVIDIA RTX 4070 Laptop GPU (8 GB)

The comparison is intended as a controlled architecture benchmark. The same
dataset split, input representation, training configuration, loss function,
and evaluation metrics are used across the three models.

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

| Class |     Dice |      IoU |
| ----: | -------: | -------: |
|     0 | 0.997690 | 0.995391 |
|     1 | 0.784552 | 0.645484 |
|     2 | 0.784979 | 0.646062 |
|     3 | 0.811642 | 0.682994 |

## Attention U-Net Benchmark

### Architecture

Attention U-Net extends the U-Net encoder-decoder structure by adding attention
gates to the skip connections. The gates use decoder-side information to
reweight encoder features before they are passed to the decoder.

For this benchmark:

- Input channels: 4
- Output classes: 4
- Base channels: 32
- Deep supervision: Not used
- Parameters: 7,852,160

### Controlled Training Configuration

The Attention U-Net experiment used the same benchmark configuration as the
U-Net and U-Net++ experiments:

- Dataset: BraTS 2023 GLI development subset
- Training cases: 158
- Validation cases: 39
- Training slices: 13,074
- Validation slices: 3,223
- Input: 4-channel 240 x 240 2D slices
- Batch size: 2
- Optimizer: AdamW
- Learning rate: 0.001
- Loss: Dice + Cross Entropy
- Epochs: 5
- Data augmentation: None
- Random seed: 42
- GPU: NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM

### Results

| Epoch | Training Loss | Validation Loss | Mean Dice | Mean IoU |
| ----: | ------------: | --------------: | --------: | -------: |
|     1 |        0.5203 |          0.7450 |    0.6355 |   0.4853 |
|     2 |        0.4389 |          0.5680 |    0.7012 |   0.5549 |
|     3 |        0.3651 |          0.4824 |    0.5118 |   0.4185 |
|     4 |        0.3217 |          0.6172 |    0.6727 |   0.5264 |
|     5 |        0.3106 |          0.5409 |    0.7291 |   0.5886 |

Best epoch: **5**

Best mean Dice: **0.7291**

Best mean IoU: **0.5886**

Runtime: **47.07 minutes**

### Per-Class Results at Best Epoch

| Class |   Dice |    IoU |
| ----: | -----: | -----: |
|     0 | 0.9977 | 0.9954 |
|     1 | 0.5579 | 0.3869 |
|     2 | 0.7720 | 0.6287 |
|     3 | 0.8572 | 0.7501 |

The reported mean Dice and mean IoU exclude the background class and average
the three foreground classes.

### Observation

Under the current controlled five-epoch configuration, Attention U-Net showed
substantial variation across epochs. In particular, Class 1 Dice decreased to
approximately zero at epoch 3 before recovering to 0.5579 by epoch 5.

This observation demonstrates that aggregate segmentation metrics can hide
class-specific instability. The experiment therefore provides a useful
motivation for investigating reliability and uncertainty rather than evaluating
a segmentation model using a single aggregate score alone.

The observed instability is descriptive; its underlying cause is not established
by this experiment alone and would require additional controlled experiments.

---

## Three-Model Benchmark Comparison

| Model           | Parameters | Best Mean Dice | Best Mean IoU | Best Epoch |  Runtime |
| --------------- | ---------: | -------------: | ------------: | ---------: | -------: |
| U-Net           |  7,763,428 |         0.7711 |        0.6329 |          5 | 39.9 min |
| U-Net++         |  9,160,068 |         0.7937 |        0.6582 |          4 | 67.3 min |
| Attention U-Net |  7,852,160 |         0.7291 |        0.5886 |          5 | 47.1 min |

These results are specific to the current development dataset split and
training configuration. They should not be interpreted as universal rankings
of the architectures.

### Benchmark Interpretation

The three architectures provide different architectural mechanisms:

- **U-Net:** standard encoder-decoder with skip connections and serves as the
  baseline.
- **U-Net++:** introduces nested skip pathways intended to reduce the semantic
  gap between encoder and decoder features.
- **Attention U-Net:** introduces attention gates that selectively reweight
  skip features using decoder information.

The experiments show that architectural complexity alone does not guarantee
higher validation metrics under the current training configuration. The
results also show why a reliability-oriented system should consider more than
a single segmentation score, including class-specific performance and
variation across conditions or model choices.

## Current Observation

Under the current five-epoch experimental setup, U-Net++ achieved a higher best
validation mean Dice and mean IoU than the U-Net baseline, while using more
parameters and requiring longer training time.

Attention U-Net achieved a best validation mean Dice of 0.7291 and mean IoU of
0.5886 under the same experimental configuration. Its Class 1 Dice showed
substantial epoch-level variation, including a near-zero value at epoch 3
before recovering by epoch 5.

These are experimental observations for the current development split and
training configuration. They should not be interpreted as general claims that
one architecture is universally superior to another.

The best validation checkpoint is used rather than automatically using the
final epoch because validation performance fluctuated across epochs.

## Next Phase

The three-model architecture benchmark is complete under the current
five-epoch experimental configuration.

The next phase will investigate reliability and uncertainty estimation using
the evaluated segmentation models. The objective is to determine whether
model confidence or uncertainty can provide useful information for identifying
potentially unreliable segmentation predictions and for supporting future
model-selection decisions.

Further experiments will be designed separately from the initial architecture
benchmark so that the benchmark remains reproducible and controlled.
