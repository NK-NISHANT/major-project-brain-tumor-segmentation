# U-Net++ (Nested Dense Skip Pathways 2D Architecture)
## Definitive Model Dossier, Benchmarks, Calibration & Reliability Audit

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation  
**Workspace:** [`D:\MajorProject`](file:///d:/MajorProject)  
**Governing Blueprint:** [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md)  
**Stage C Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)  
**Model Identifier:** `unetpp` (`UNetPlusPlus2D` in [`src/models/unetpp.py`](file:///d:/MajorProject/src/models/unetpp.py))  
**Best Checkpoint:** Epoch 15 (`best_model.pt` in [`results/stage_c/phase1/unetpp_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/))  
**Document Location:** [`documents/UNetPlusPlus_Master_Model_Dossier.md`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.md)  
**Associated PDF Dossier:** [`documents/UNetPlusPlus_Master_Model_Dossier.pdf`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.pdf)  
**Associated Word Master:** [`documents/UNetPlusPlus_Master_Model_Dossier.docx`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.docx)  
**Executive Printout Poster:** [`documents/UNetPlusPlus_Executive_Summary_Printout.pdf`](file:///d:/MajorProject/documents/UNetPlusPlus_Executive_Summary_Printout.pdf) | [`PNG`](file:///d:/MajorProject/documents/UNetPlusPlus_Executive_Summary_Printout.png)  
**Date:** November 18, 2026  
**Final Status:** **`VERIFIED & ARCHIVED`** (All 15 Epochs Complete | Phase 1 & 2 Fully Audited | Zero Errors)

---

## 1. Executive Summary & Core Results

The **U-Net++ (Nested Dense Skip Pathways U-Net)** architecture has successfully completed the entire Stage C experimental evaluation pipeline under the governing Master Project Blueprint v2.0. Evaluated under strict synchronous CUDA execution on the authorized 721-case training cohort and 155-case validation pool, U-Net++ attained a peak selection-exposed validation Mean Dice of **`0.8602`** (Mean IoU: **`0.7568`**) at Epoch 15.

In downstream Stage C Phase 2 evaluation across 105 held-out internal evaluation cases (`Val_eval`, 311,546 voxels), post-hoc temperature scaling ($T^* = 2.5647$) reduced Expected Calibration Error from **11.18%** to **5.51%** (**-50.70% relative reduction**) and Negative Log-Likelihood from **0.8074** to **0.3904** (**-51.64% relative reduction**), while maintaining **100% discrete argmax segmentation invariance** with exactly zero mismatched voxels across nearly one billion evaluated predictions.

![Figure 0: Executive Dashboard Poster](figures/fig_07_unetpp_executive_poster_dashboard.png)

> [!IMPORTANT]
> **Executive Benchmark Verdict:**
> U-Net++ recorded the highest selection-exposed validation Mean Dice (**`0.8602`**) among all three scratch-trained Phase 1 baseline architectures, narrowly exceeding standard U-Net (**`0.8594`**) by +0.0008 (+0.08 pp) and Attention U-Net (**`0.8542`**) by +0.0060 (+0.60 pp). However, this modest performance gain comes with a **2.5× higher VRAM footprint** (1,111.3 MB vs 446.3 MB) and **1.87× runtime overhead** (15.44 hours vs 8.24 hours). In accordance with the project blueprint, this numerical difference does not establish meaningful architectural superiority and must be weighed against its computational cost.

### Core Performance & Reliability Scorecard

| Clinical & Technical Metric | Protocol Specification | Observed Empirical Value | Source Artifact |
| :--- | :--- | :---: | :--- |
| **Model Family** | Nested Dense Skip 2D U-Net | `UNetPlusPlus2D` | [`src/models/unetpp.py`](file:///d:/MajorProject/src/models/unetpp.py) |
| **Trainable Parameters** | Standardized Stage B channels | **9,160,068 (~9.16 M)** | Exact match with Table 4.1 protocol |
| **Training Cohort** | Authorized development set | **721 cases / 667 unique subjects** | [`development_train_manifest.json`](file:///d:/MajorProject/results/dataset_splits/development_train_manifest.json) |
| **Validation Pool** | Internal validation cohort | **155 cases / 143 unique subjects** | [`validation_manifest.json`](file:///d:/MajorProject/results/dataset_splits/validation_manifest.json) |
| **Training Slices** | 25% background slice ratio | **59,020 slices** (47,228 fg + 11,792 bg) | Verified via `CaseSliceBatchSampler` |
| **Best Selected Epoch** | Maximum Validation Mean Dice | **Epoch 15** (Peak observed in budget) | [`summary.json`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/summary.json) |
| **Selection Val Mean Dice** | Internal validation pool | **0.8602** (0.860218) | [`history.csv`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/history.csv) |
| **Selection Val Mean IoU** | Internal validation pool | **0.7568** (0.756770) | [`history.csv`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/history.csv) |
| **Enhancing Tumor (ET) Dice**| Class 3 active neoplastic margin | **0.9047** (IoU: 0.8260) | Peak sub-compartment performance |
| **Peritumoral Edema (ED) Dice**| Class 2 infiltrative zone | **0.8664** (IoU: 0.7642) | High cross-model consistency |
| **Necrotic Core (NCR) Dice** | Class 1 avascular necrotic core | **0.8096** (IoU: 0.6801) | Outperforms Attention U-Net (0.7947) |
| **Background (BG) Dice** | Class 0 normal brain tissue | **0.9986** (IoU: 0.9972) | Delineates tissue void cleanly |
| **Final Training Loss** | Dice + Cross-Entropy ($\lambda=1.0$) | **0.1728** (0.172768) | Lowest train loss among all baselines |
| **Final Validation Loss** | Dice + Cross-Entropy | **0.4280** (0.428033) | Stable convergence |
| **Optimal Temperature ($T^*$)** | Learned on `Val_cal` (50 cases) | **2.564703** (42 L-BFGS iters) | [`calibration_parameters.json`](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json) |
| **Uncalibrated $\rightarrow$ Calibrated ECE**| Held-out `Val_eval` (105 cases) | **11.18% $\rightarrow$ 5.51%** (-50.70% rel.) | [`evaluation_metrics.csv`](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv) |
| **Uncalibrated $\rightarrow$ Calibrated NLL**| Held-out `Val_eval` (105 cases) | **0.8074 $\rightarrow$ 0.3904** (-51.64% rel.) | [`evaluation_metrics.csv`](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv) |
| **Uncalibrated $\rightarrow$ Calibrated Brier**| Held-out `Val_eval` (105 cases) | **0.2332 $\rightarrow$ 0.1983** (-14.98% rel.) | [`evaluation_metrics.csv`](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv) |
| **Held-out Volumetric Dice** | `Val_eval` 3D reconstruction | **0.8549** (WT: 0.8830, TC: 0.8521, ET: 0.8297) | 105 patient volume assemblies |
| **Argmax Prediction Invariance** | Post-hoc scaling discrete check | **EXACTLY 0 MISMATCHES (0.000%)** | Verified across 937,440,000 voxels |
| **Mean Predictive Entropy** | 6,128 non-empty slices | **0.2883 ± 0.0907** (Median: 0.2685) | [`entropy_summary.csv`](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv) |
| **Peak Process VRAM** | NVIDIA RTX 4070 (8 GB VRAM) | **1,111.3 MB** (~1.11 GB) | Well within hardware limits |
| **Cumulative Runtime** | 15 Synchronous CUDA Epochs | **15.44 Hours (926.39 Minutes)** | Zero crashes, zero driver events |
| **Test Quarantine Boundary** | `data/quarantine_test/` | **100% INTACT & SEALED** | Zero test cases exposed |

---

## 2. Architecture & Theoretical Foundation

The U-Net++ architecture (Zhou et al., 2018) redesigns the classic identity skip connections of standard U-Net into a nested, dense convolutional network. In standard U-Net, feature maps from early encoder layers are directly concatenated with decoder feature maps at the same resolution. Because encoder features are semantically coarse (focusing on low-level edges and textures) while decoder features are semantically rich (incorporating high-level contextual abstracts), a large semantic gap exists across identity skip connections.

U-Net++ bridges this semantic gap by routing features through a nested, dense series of intermediate convolutional nodes ($X^{i,j}$):

![Figure 1: U-Net++ Nested Dense Skip Architecture](figures/fig_01_unetpp_nested_architecture.png)

### Formal Mathematical Specification

Let $x^{i,j}$ denote the feature map output by node $X^{i,j}$, where $i \in \{0, 1, 2, 3, 4\}$ denotes the resolution level along the encoder spine, and $j \in \{0, 1, 2, 3, 4 - i\}$ denotes the dense skip pathway index:

$$x^{i,j} = \begin{cases}
\mathcal{C}\left( \text{Input} \right) & i = 0, j = 0 \\
\mathcal{C}\left( \mathcal{D}\left( x^{i-1, 0} \right) \right) & i > 0, j = 0 \\
\mathcal{C}\left( \left[ \left( x^{i, k} \right)_{k=0}^{j-1}, \mathcal{U}\left( x^{i+1, j-1} \right) \right] \right) & j > 0
\end{cases}$$

Where:
- $\mathcal{C}(\cdot)$ denotes a dual 3×3 convolutional block (`ConvBlock`), consisting of two consecutive $3 \times 3$ convolutions, each followed by Batch Normalization and inplace ReLU activation:
  $$\mathcal{C}(z) = \text{ReLU}\left( \text{BN}\left( \text{Conv}_{3\times 3}\left( \text{ReLU}\left( \text{BN}\left( \text{Conv}_{3\times 3}(z) \right) \right) \right) \right) \right)$$
- $\mathcal{D}(\cdot)$ denotes spatial downsampling via $2 \times 2$ max-pooling with stride 2.
- $\mathcal{U}(\cdot)$ denotes spatial upsampling via bilinear interpolation by a factor of 2 (`align_corners=False`).
- $[\cdot]$ denotes channel-wise feature concatenation.

### Structural Parameters by Level

| Node Level | Spatial Grid | Channels | Total Nodes | Dense Input Channel Growth | Parameter Role |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **Level 0** | $240 \times 240$ | 32 | $X^{0,0}, X^{0,1}, X^{0,2}, X^{0,3}, X^{0,4}$ | $4 \rightarrow 32 \rightarrow 64 \rightarrow 96 \rightarrow 128 \rightarrow 160$ | High-resolution spatial delineation |
| **Level 1** | $120 \times 120$ | 64 | $X^{1,0}, X^{1,1}, X^{1,2}, X^{1,3}$ | $32 \rightarrow 64 \rightarrow 128 \rightarrow 192 \rightarrow 256$ | Intermediate boundary context |
| **Level 2** | $60 \times 60$ | 128 | $X^{2,0}, X^{2,1}, X^{2,2}$ | $64 \rightarrow 128 \rightarrow 256 \rightarrow 384$ | Sub-compartment structural representation |
| **Level 3** | $30 \times 30$ | 256 | $X^{3,0}, X^{3,1}$ | $128 \rightarrow 256 \rightarrow 512$ | Coarse tumor envelope semantics |
| **Level 4** | $15 \times 15$ | 512 | $X^{4,0}$ | $256 \rightarrow 512$ | Deepest receptive bottleneck |

**Total Parameter Count:** Exactly **9,160,068** trainable parameters (~9.16 M), requiring 34.9 MB of uncompressed 32-bit floating point storage.

---

## 3. Protocol & Training Trajectory Audit

Training conformed strictly to the locked Stage C protocol:

- **Loss Formulation:** Combined Multiclass Dice and Cross-Entropy with equal weighting:
  $$\mathcal{L}_{\text{total}} = \lambda_{\text{Dice}} \mathcal{L}_{\text{Dice}} + \lambda_{\text{CE}} \mathcal{L}_{\text{CE}}, \quad \lambda_{\text{Dice}} = 1.0, \; \lambda_{\text{CE}} = 1.0$$
  where background (Class 0) is excluded from the Dice component to prioritize tumor foreground classes.
- **Optimizer & Schedule:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$), base learning rate $\eta = 0.0010$, 2-epoch linear warmup, followed by 13-epoch cosine decay to floor $\eta_{\min} = 1.0 \times 10^{-5}$.
- **Hardware Telemetry:** Workstation equipped with NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6), 13th Gen Intel Core i9-13900HX CPU, 16 GB DDR5 RAM, running Windows 11 with synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`).

![Figure 2: 15-Epoch Training Trajectory](figures/fig_02_unetpp_training_trajectory.png)

### Complete Epoch-by-Epoch Execution History (`history.csv`)

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Clip Norm | % Clipped | Active LR | Epoch Duration | Cumulative Runtime | Optimal? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4537 | 0.6158 | 0.7938 | 0.6618 | 0.8592 | 0.8030 | 0.7191 | 0.9978 | 1.69 | 52.0% | $1.00 \times 10^{-4}$ | 61.9 m | 1.03 h | YES |
| **2** | 0.2758 | 0.4585 | 0.7072 | 0.5499 | 0.7525 | 0.7423 | 0.6268 | 0.9976 | 0.86 | 17.8% | $5.50 \times 10^{-4}$ | 61.8 m | 2.06 h | no |
| **3** | 0.2411 | 0.3498 | 0.7990 | 0.6678 | 0.8595 | 0.7931 | 0.7442 | 0.9977 | 0.39 | 5.5% | $1.00 \times 10^{-3}$ | 61.9 m | 3.09 h | YES |
| **4** | 0.2289 | 0.3117 | 0.7895 | 0.6617 | 0.8762 | 0.8319 | 0.6606 | 0.9983 | 0.29 | 3.5% | $9.86 \times 10^{-4}$ | 61.9 m | 4.12 h | no |
| **5** | 0.2151 | 0.3434 | 0.8488 | 0.7389 | 0.8886 | 0.8551 | 0.8027 | 0.9984 | 0.26 | 3.2% | $9.43 \times 10^{-4}$ | 61.9 m | 5.15 h | YES |
| **6** | 0.2074 | 0.2950 | 0.8297 | 0.7123 | 0.8786 | 0.8540 | 0.7565 | 0.9985 | 0.24 | 2.8% | $8.76 \times 10^{-4}$ | 62.0 m | 6.18 h | no |
| **7** | 0.1996 | 0.3364 | 0.8470 | 0.7363 | 0.8971 | 0.8278 | 0.8162 | 0.9982 | 0.24 | 2.4% | $7.86 \times 10^{-4}$ | 61.9 m | 7.21 h | no |
| **8** | 0.1966 | 0.3988 | 0.8246 | 0.7034 | 0.8520 | 0.8527 | 0.7691 | 0.9984 | 0.23 | 2.4% | $6.81 \times 10^{-4}$ | 60.8 m | 8.23 h | no |
| **9** | 0.1893 | 0.3917 | 0.8410 | 0.7270 | 0.8828 | 0.8327 | 0.8076 | 0.9983 | 0.23 | 2.3% | $5.65 \times 10^{-4}$ | 60.0 m | 9.23 h | no |
| **10** | 0.1867 | 0.4557 | 0.8524 | 0.7452 | 0.8973 | 0.8647 | 0.7953 | 0.9985 | 0.23 | 2.2% | $4.45 \times 10^{-4}$ | 59.8 m | 10.22 h | YES |
| **11** | 0.1816 | 0.4158 | 0.8482 | 0.7398 | 0.9001 | 0.8659 | 0.7787 | 0.9986 | 0.22 | 2.2% | $3.29 \times 10^{-4}$ | 61.5 m | 11.25 h | no |
| **12** | 0.1791 | 0.4864 | 0.8376 | 0.7233 | 0.8880 | 0.8491 | 0.7757 | 0.9986 | 0.25 | 2.2% | $2.24 \times 10^{-4}$ | 63.9 m | 12.32 h | no |
| **13** | 0.1766 | 0.5210 | 0.8477 | 0.7388 | 0.9014 | 0.8596 | 0.7821 | 0.9986 | 0.25 | 2.2% | $1.34 \times 10^{-4}$ | 62.7 m | 13.36 h | no |
| **14** | 0.1746 | 0.4382 | 0.8450 | 0.7346 | 0.9014 | 0.8490 | 0.7847 | 0.9984 | 0.26 | 2.2% | $6.67 \times 10^{-5}$ | 62.1 m | 14.40 h | no |
| **15** | **0.1728** | **0.4280** | **0.8602** | **0.7568** | **0.9047** | **0.8664** | **0.8096** | **0.9986** | **0.24** | **2.1%** | **$2.44 \times 10^{-5}$** | 62.4 m | **15.44 h** | **YES [BEST]** |

---

## 4. Tumor Sub-Compartment Accuracy & Clinical Significance

Brain tumor segmentation requires accurate delineation of heterogeneous tissue sub-compartments that vary in intensity, morphology, and clinical implications:

![Figure 3: Sub-Compartment Dice Evolution](figures/fig_03_unetpp_compartment_dice_evolution.png)

1. **Enhancing Tumor (ET / Class 3):** Delineates the actively proliferating, vascularized outer rim of high-grade glioma characterized by blood-brain barrier breakdown (hyperintense on T1ce). U-Net++ attained **`0.9047`** Dice (IoU: 0.8260), demonstrating the highest score among all Phase 1 baselines. This region is critical for surgical debulking and intraoperative guidance.
2. **Peritumoral Edema (ED / Class 2):** Captures the infiltrative fluid margin surrounding the tumor core (hyperintense on T2-FLAIR). U-Net++ achieved **`0.8664`** Dice (IoU: 0.7642), showing strong consistency with standard U-Net (0.8662). This boundary determines radiation therapy planning margins.
3. **Necrotic Core (NCR / Class 1):** Comprises avascular non-enhancing dead tissue and cystic cavity fluid within the mass (hypointense on T1ce). U-Net++ achieved **`0.8096`** Dice (IoU: 0.6801), significantly outperforming Attention U-Net (0.7947).
4. **Composite Tumor Volumes:** Clinical evaluation also measures synthesized anatomical structures:
   - **Whole Tumor (WT):** $\text{WT} = \text{ET} \cup \text{ED} \cup \text{NCR} \rightarrow$ Volumetric Dice on `Val_eval`: **`0.8830`**
   - **Tumor Core (TC):** $\text{TC} = \text{ET} \cup \text{NCR} \rightarrow$ Volumetric Dice on `Val_eval`: **`0.8521`**

---

## 5. Post-Hoc Probability Calibration & Temperature Scaling

Deep convolutional networks trained with unweighted or composite loss functions output uncalibrated logits that produce overconfident softmax probabilities. In Stage C Phase 2 (Work Package C2), post-hoc temperature scaling was executed across all architectures:

$$p(y = c \mid x) = \sigma\left( \frac{z}{T^*} \right)_c = \frac{\exp\left( z_c / T^* \right)}{\sum_{k=0}^3 \exp\left( z_k / T^* \right)}$$

- **Calibration Fitting Cohort (`Val_cal`):** 50 cases / 47 unique subjects (5,020,730 sampled voxels).
- **Evaluation Cohort (`Val_eval`):** 105 cases / 96 unique subjects (311,546 sampled voxels).
- **Optimization:** Single scalar $T > 0$ optimized via PyTorch L-BFGS with strong Wolfe line search ($\eta=0.05$, max 100 iters) minimizing Cross-Entropy NLL.
- **Learned Parameter:** $T^* = 2.564703$ (converged in 42 iterations, 0.62 seconds).

![Figure 4: Reliability Diagram and Calibration Analysis](figures/fig_04_unetpp_calibration_reliability.png)

### Quantitative Calibration Impact on Held-Out `Val_eval`

| Calibration Metric | Uncalibrated ($T=1.00$) | Calibrated ($T^*=2.5647$) | Delta ($\Delta$) | Relative Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **Expected Calibration Error (ECE)** | 11.18% (0.1118) | **5.51%** (0.0551) | -5.67 pp | **-50.70%** |
| **Negative Log-Likelihood (NLL)** | 0.8074 | **0.3904** | -0.4170 | **-51.64%** |
| **Maximum Calibration Error (MCE)** | 39.12% | **26.55%** | -12.57 pp | **-32.13%** |
| **Brier Score** | 0.2332 | **0.1983** | -0.0349 | **-14.98%** |

---

## 6. Predictive Uncertainty & Empty-Prediction Semantics

Normalized four-class Shannon entropy was computed from calibrated probabilities across all 16,275 slices ($105 \text{ cases} \times 155 \text{ slices}$) in `Val_eval`:

$$H(p) = -\frac{1}{\ln 4} \sum_{c=0}^3 p_c \ln p_c \in [0, 1]$$

![Figure 5: Predictive Uncertainty and Slice Accounting](figures/fig_06_unetpp_predictive_entropy_distribution.png)

### Empty-Prediction Semantics & Population Breakdown

- **Total Slices Evaluated:** 16,275 slices (100.0%)
- **Empty Prediction Slices:** **10,147 slices (62.35%)**
  - *Strict Handling Enforcement:* In slices where zero foreground voxels are predicted, mean foreground entropy is assigned Python `None` (JSON `null`). Zero false 0.0 entropy substitutions occurred.
- **Non-Empty Foreground Slices:** **6,128 slices (37.65%)**
  - **Mean Foreground Entropy ($\mu$):** **0.2883 ± 0.0907**
  - **Median Foreground Entropy:** **0.2685**
  - **Interquartile Range [p25, p75]:** **[0.2168, 0.3408]**
  - **Observed Range [Min, Max]:** **[0.1286, 0.7227]**
- **Operating Triage Threshold:** $\tau = 0.35$. Slices with $H < 0.35$ are autonomously accepted; slices with $H \ge 0.35$ are referred to clinical specialists.

### Discrete Argmax Prediction Invariance

Because scalar temperature scaling divides all class logits by a constant $T > 0$, the mathematical argmax order is strictly invariant:

$$\operatorname{argmax}_{c} \left( \frac{z_c}{T} \right) \equiv \operatorname{argmax}_{c} (z_c) \quad \forall T > 0$$

Across all 105 evaluation cases and all 155 slices ($16,275 \text{ slices} \times 240 \times 240 = 937,440,000 \text{ voxels}$), **EXACTLY 0 VOXELS** differed between uncalibrated and calibrated predictions. Prediction invariance is mathematically and empirically 100% verified.

---

## 7. Five-Model Comparative Benchmark Matrix

![Figure 6: Five-Model Comparative Benchmark Matrix](figures/fig_05_five_model_benchmark_matrix.png)

| Evaluation Metric | ResUNet-2D | UNetResNet34 | Standard U-Net | U-Net++ (Nested) | Attention U-Net |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Model Family** | PreAct ResNet U-Net | ResNet-34 Backbone | Classical Identity Skips | Nested Dense Skips | Additive Attention Gates |
| **Trainable Parameters** | 8,114,604 (~8.11 M) | 24,327,524 (~24.33 M) | 7,763,428 (~7.76 M) | **9,160,068 (~9.16 M)** | 7,852,160 (~7.85 M) |
| **Peak Process VRAM** | 446.3 MB | 736.1 MB | 446.3 MB | **1,111.3 MB** | 482.0 MB |
| **Best Training Epoch** | Epoch 13 | Epoch 15 | Epoch 15 | **Epoch 15** | Epoch 15 |
| **Selection Val Mean Dice** | 0.8659 | **0.8674** | 0.8594 | **0.8602** | 0.8542 |
| **Selection Val Mean IoU** | 0.7634 | **0.7658** | 0.7554 | **0.7568** | 0.7482 |
| **Val_eval Volumetric Dice** | **0.8641** | 0.8535 | 0.8574 | **0.8549** | 0.8581 |
| **Whole Tumor (WT) Dice** | **0.9039** | 0.8843 | 0.8900 | **0.8830** | 0.8865 |
| **Tumor Core (TC) Dice** | **0.8597** | 0.8423 | 0.8527 | **0.8521** | 0.8476 |
| **Enhancing Tumor (ET) Dice**| 0.8286 | 0.8339 | 0.8294 | **0.8297** | **0.8400** |
| **Optimal Temperature ($T^*$)** | 2.8805 | 2.6261 | 2.6268 | **2.5647** | **2.2869** |
| **Uncalibrated ECE** | 8.59% | 10.98% | 10.85% | **11.18%** | 10.81% |
| **Calibrated ECE** | **3.41%** | 4.96% | 4.83% | **5.51%** | 5.68% |
| **Relative ECE Reduction** | **-60.32%** | -54.78% | -55.46% | **-50.70%** | -47.47% |
| **Calibrated NLL** | 0.4415 | **0.3655** | 0.3939 | **0.3904** | 0.3689 |
| **Mean Predictive Entropy** | **0.2686** | 0.3169 | 0.2923 | **0.2883** | 0.2775 |

---

## 8. Checkpoint Provenance & Cryptographic Hashes

All artifact files in [`results/stage_c/phase1/unetpp_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/) were verified on disk and cryptographically hashed:

| Artifact File | Size (Bytes) | SHA-256 Checksum | Integrity & Reload Verification |
| :--- | :---: | :--- | :---: |
| **`best_model.pt`** | 110,090,779 | `fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6` | **VERIFIED** (`<All keys matched successfully>`) |
| **`final_model.pt`** | 110,091,243 | `d3c3e16961d4df6c3c6c5ef5434bbe310a637c5cd063baf56950f88db9a17ea0` | **VERIFIED** (`<All keys matched successfully>`) |
| **`history.csv`** | 5,781 | `95775bf441c1792684a6e76e8c8196c9f4e46298531c19b42418b59b0b734ae5` | Exactly 15 epoch rows without gaps |
| **`summary.json`** | 1,257 | `ffd9752e837c6026175bfd385f5734e96ae9ce3fd1e8ebac081ab71cc50e1722` | Strict JSON consistency with history |
| **`config.json`** | 345 | `66da99449907f1cc50da077c1a323e0c44d02d9bb9cb695249eb089d2c1ec059` | Unmodified locked parameter config |

---

## 9. Methodological Limitations & Safety Boundary

1. **Selection-Exposed Internal Validation:** All validation metrics reported in this dossier (including the peak Validation Mean Dice of **`0.8602`**) were computed on the internal 155-case validation pool (`Val_cal` + `Val_eval`). Because model checkpoints were continuously monitored on this pool, these metrics are formally classified as **selection-exposed internal validation metrics**. They do not establish unbiased generalization on external unseen cohorts.
2. **Epoch 15 Checkpoint Peak & Non-Convergence:** Both U-Net++ and standard U-Net achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued or whether extended training would yield further gains.
3. **Single-Seed Statistical Boundary:** Evaluated using a single configured random seed (seed 42). Run-to-run variance cannot be computed without multi-seed replications.
4. **Test Quarantine Preservation:** The quarantined test partition (`data/quarantine_test/`) and test manifest (`results/dataset_splits/test_manifest.json`) were strictly sealed and preserved.
5. **Academic Prototype Status:** U-Net++ is an engineering research prototype. It is NOT certified as a medical device by the US FDA, CE, or CDSCO and is strictly prohibited from autonomous clinical diagnosis.

---

## 10. Deliverables & Resource Index

All deliverables for this U-Net++ dossier package are archived and directly accessible in the workspace:

- **Executive Printout Poster (PDF):** [`documents/UNetPlusPlus_Executive_Summary_Printout.pdf`](file:///d:/MajorProject/documents/UNetPlusPlus_Executive_Summary_Printout.pdf)
- **Executive Printout Poster (High-Res PNG):** [`documents/UNetPlusPlus_Executive_Summary_Printout.png`](file:///d:/MajorProject/documents/UNetPlusPlus_Executive_Summary_Printout.png)
- **Word Master Document:** [`documents/UNetPlusPlus_Master_Model_Dossier.docx`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.docx)
- **Vector PDF Dossier:** [`documents/UNetPlusPlus_Master_Model_Dossier.pdf`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.pdf)
- **Raw Markdown Dossier:** [`documents/UNetPlusPlus_Master_Model_Dossier.md`](file:///d:/MajorProject/documents/UNetPlusPlus_Master_Model_Dossier.md)
- **High-Resolution Figures (300 DPI):**
  - [Figure 1: Full Architecture Diagram](file:///d:/MajorProject/documents/figures/fig_01_unetpp_nested_architecture.png)
  - [Figure 2: 15-Epoch Training Trajectory](file:///d:/MajorProject/documents/figures/fig_02_unetpp_training_trajectory.png)
  - [Figure 3: Sub-Compartment Dice Evolution](file:///d:/MajorProject/documents/figures/fig_03_unetpp_compartment_dice_evolution.png)
  - [Figure 4: Probability Calibration & Reliability](file:///d:/MajorProject/documents/figures/fig_04_unetpp_calibration_reliability.png)
  - [Figure 5: Predictive Uncertainty & Entropy](file:///d:/MajorProject/documents/figures/fig_06_unetpp_predictive_entropy_distribution.png)
  - [Figure 6: 5-Model Comparative Benchmark Matrix](file:///d:/MajorProject/documents/figures/fig_05_five_model_benchmark_matrix.png)
  - [Figure 7: Executive Poster Dashboard](file:///d:/MajorProject/documents/figures/fig_07_unetpp_executive_poster_dashboard.png)
- **Reproduction Scripts:**
  - Asset Generator: [`tools/generate_unetpp_assets.py`](file:///d:/MajorProject/tools/generate_unetpp_assets.py)
  - Dossier Builder: [`tools/build_unetpp_dossier.py`](file:///d:/MajorProject/tools/build_unetpp_dossier.py)
