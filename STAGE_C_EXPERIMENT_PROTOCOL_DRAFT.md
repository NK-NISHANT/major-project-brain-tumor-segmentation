# STAGE C EXPERIMENTAL PROTOCOL DRAFT: SCIENTIFIC DESIGN & DECISION FREEZE

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Target Milestone:** Semester 7–8 Major Project Research Progression
**Predecessor Audits:** G1-A3, G1-C1, G1-D0, G1-D0.1, G1-D0.2, G1-D0.2.1, G1-D0.2.2, G1-P2, G1-P2.1
**Document ID:** `STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md` (Revision G1-P2.2)
**Decision Register:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Execution Status:** **STRICTLY READ-ONLY DESIGN — ZERO TRAINING, INFERENCE, OR RECALIBRATION AUTHORIZED**
**Date:** November 13, 2026
**Auditor / Designer:** Independent Senior ML Research Engineer & Methodology Auditor

---

## 1. Executive Summary & Research Philosophy

The primary objective of Stage C is to transition the project from its historical exploratory phase (Stage A: 5-epoch feasibility pilots on 158 cases) and initial mature baseline scaling (Stage B: 15-epoch training of ResUNet-2D and UNetResNet34 on 721 cases) into a **controlled, reproducible, and scientifically defensible research study**.

### Methodological Classification of Historical Work
- **Stage A Evidence:** The 158-case results obtained in September 2026 are classified strictly as **exploratory historical preliminary evidence**. Because Stage A employed a 5-epoch training duration, an early loss formulation, and an older data loading script, its numerical outputs cannot be directly compared against Stage B or Stage C models. The project explicitly **refuses to claim a formal 158-to-721-to-941 learning curve** without a dedicated, reproducible, matched training protocol across data scales.
- **Stage B Evidence:** The 15-epoch models (`resunet_2d` and `unet_resnet34`) trained in October 2026 represent mature, fully audited baselines with verified checkpoint histories and convergence stability on the 721-case development cohort.

### Stage C Core Research Questions
Stage C rigorously investigates five foundational research questions:
1. **Architectural Comparison (RQ1):** How do distinct skip-connection and residual paradigms (standard skips, nested dense skips, attention gates, and pre-activation residuals) compare when trained under an identical 15-epoch protocol on the same 721-case development cohort?
2. **Probability Calibration (RQ2):** Does post-hoc temperature scaling ($T^*$) independently fitted for each architecture reliably mitigate neural network overconfidence on multi-parametric MRI without altering discriminative segmentation rankings?
3. **Selective Prediction & Reliable Triage (RQ3):** Does an uncertainty-aware agentic policy employing calibrated normalized predictive entropy ($H \in [0, 1]$) and axial spatial continuity ($k=2$) reliably isolate low-fidelity segmentations for human specialist referral?
4. **Input Modality Robustness & Failure Signaling (RQ4):** How do multi-parametric models behave when individual MRI sequences are missing, and does predictive entropy reliably signal severe compartment-level segmentation failure under clinical acquisition defects?
5. **Boundary Quality & Spatial Fidelity (RQ5):** How do spatial boundary distance metrics (95th-percentile Hausdorff Distance, HD95) correlate with volumetric overlap (Dice similarity coefficient) across tumor compartments?

---

## 2. Frozen Cohort Accounting & Epistemological Boundaries

### 2.1 Complete Census and Partition Inventory
Based on the audited 1,251-case census (1,133 unique biological subjects) established under `MASTER_PROJECT_BLUEPRINT_v2.0.md`, all data partitions are frozen as follows:

| Cohort Name | Cases | Unique Subjects | Slices (2D Axial) | Authoritative Interpretation & Usage | Status |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **Development Training** (`Train_dev`) | **721** | **667** | 59,020 (Filtered) | Consumed by existing 15-epoch Stage B models; primary training cohort for Stage C harmonized benchmark. | **FROZEN** |
| **Historical Development Reserve** (`Train_res`) | **220** | **180** | ~18,000 (Filtered) | Unconsumed development training cases; includes 197 cases used during Stage A; candidate for Semester 8 scaling study. **Prohibited from evaluation if models are trained on `Train_full`.** | **FROZEN** |
| **Full Train Allocation** (`Train_full`) | **941** | **847** | 77,020 (Filtered) | Total census train allocation (721 development + 220 historical reserve cases). | **FROZEN** |
| **Validation Calibration** (`Val_cal`) | **50** | **47** | 3,982 (Filtered)<br>7,750 (Raw Volume) | Internal validation calibration subset; **checkpoint-selection-exposed** (belongs to the 155-case validation pool); used for fitting post-hoc calibration scalar ($T^*_m$). **Prohibited from being described as an independent holdout set.** | **FROZEN** |
| **Internal Validation Evaluation** (`Val_eval`) | **105** | **96** | 8,382 (Filtered)<br>16,275 (Raw Volume)<br>6,549 (Foreground) | Internal validation evaluation subset; **checkpoint-selection-exposed**; used for descriptive comparative evaluation. **Prohibited from being described as an independent holdout set.** | **FROZEN** |
| **Total Validation Pool** (`Val_pool`) | **155** | **143** | 12,364 (Filtered)<br>24,025 (Raw Volume) | Combined validation pool used to monitor validation loss and select `best_model.pt` across all epochs. | **FROZEN** |
| **Sealed Test Cohort** (`Test_sealed`) | **155** | **143** | 24,025 (Raw Volume) | Prospective final one-time generalization cohort; strictly sealed and untouched throughout Stage C design. | **SEALED** |
| **Total Verified Census** | **1,251** | **1,133** | 194,000+ (Raw) | Complete ASNR-MICCAI BraTS 2023 Adult Glioma active challenge cohort. | **VERIFIED** |

---

### 2.2 Non-Negotiable Scientific Constraints

#### Constraint A: Absolute Prohibition of Training/Evaluation Overlap
Under no circumstances may models trained on the full 941-case allocation (`Train_full`) be evaluated on the 220 historical reserve cases (`Train_res`). The 220 reserve cases are physically and mathematically subsumed within the 941-case allocation ($721 + 220 = 941$). Evaluating a 941-case model on any of these 220 reserve cases constitutes direct training-set re-evaluation (severe training data leakage). Furthermore, the claim that 168 reserve cases represent an independent, unexposed evaluation cohort is methodologically invalid and is **completely removed** from all protocol specifications.

#### Constraint B: Historical Stage A Exposure Disclosure
The 220 historical reserve cases contain 197 cases (158 training / 39 validation) that were evaluated during the exploratory 5-epoch Stage A pilot in September 2026. Therefore, the reserve cohort must **not** be labeled as "pristine" or "untouched by any prior model". While Stage B training was completely quarantined from these 220 cases, their historical exposure must be candidly documented. Stage A results are strictly classified as preliminary exploratory evidence, and cannot be combined into a formal learning curve without a matched, reproducible re-training run.

#### Constraint C: Mandatory Disclosure of Checkpoint-Selection Exposure for Calibration and Evaluation Cohorts
The checkpoint-selection procedure for the three newly trained Stage C models (U-Net, U-Net++, Attention U-Net) and existing Stage B models is explicitly defined as follows:
- **Selection Metric:** Epoch-end Maximum Validation Mean Dice across all three tumor classes ($\text{Mean Dice} = \frac{\text{Dice}_{WT} + \text{Dice}_{TC} + \text{Dice}_{ET}}{3}$).
- **Monitored Cohort:** To maintain strict methodological parity with Stage B baseline models (`resunet_2d` and `unet_resnet34`), checkpoint selection monitors the full 155-case validation pool (`validation_manifest.json`, 12,364 background-filtered slices), which comprises both `Val_cal` (50 cases) and `Val_eval` (105 cases).
- **Mandatory Disclosure of Dual Overlap:**
  - Because all 105 `Val_eval` evaluation cases are included within the 155-case validation pool, the evaluation cases directly participate in the model-selection loop that saves `best_model.pt`.
  - Crucially, **`Val_cal` (50 cases) is also part of the 155-case checkpoint-selection pool**. Consequently, fitting temperature scaling on `Val_cal` is calibration fitting on a **selection-exposed internal subset**, NOT an independent calibration exercise on unseen data.
- **Strict Prohibition of Independent Holdout Description:**
  - In all Stage C reports, manuscripts, and viva presentations, **neither `Val_cal` nor `Val_eval` may be described, reported, or claimed as an independent held-out test set**.
  - Metrics evaluated on `Val_eval` must be explicitly designated as **"selection-exposed internal validation results"**, subject to model-selection optimization bias.
- **Separation of Calibration-Fit vs Evaluation Metrics:**
  - Calibration-fit metrics computed on `Val_cal` (optimization NLL, fit ECE) measure how effectively temperature scaling fits the selection-exposed calibration subset.
  - Evaluation metrics computed on `Val_eval` (uncalibrated vs calibrated ECE, MCE, Brier score, NLL, and segmentation Dice / HD95) assess probability calibration and segmentation fidelity across the evaluation subset. These metrics must be reported in separate columns/tables and never conflated.
- **Descriptive Status of Risk-Coverage Sweeps:**
  - Risk-coverage sweeps across $\tau \in [0.05, 0.50]$ evaluated on `Val_eval` are strictly **descriptive internal validation analyses** to illustrate selective prediction trade-offs.
  - The protocol strictly **prohibits selecting or optimizing an operating threshold $\tau$ on `Val_eval` and subsequently claiming unbiased or prospective selective prediction performance at that chosen threshold** on the very same data. Any post-hoc tuned threshold evaluated on `Val_eval` would be contaminated by threshold-tuning optimization bias.

#### Constraint D: Cautious Test Cohort Quarantine Status
The test manifest (`results/dataset_splits/test_manifest.json`) was historically parsed during repository inventory and freeze audits in October 2026. However, across all inspected experiment logs, training scripts, and evaluation outputs, **no evidence of quarantined test MRI-volume or mask loading, model evaluation, or metric computation was found**.
- During Stage C protocol design, `data/quarantine_test/` and `results/dataset_splits/test_manifest.json` remain **strictly quarantined, unopened, and uninspected**.
- The 155-case test cohort is reserved exclusively for a formal, pre-registered, single-pass evaluation prior to final capstone degree submission in Semester 8, contingent upon academic supervisor approval.

---

## 3. Stage C Training Strategy Decision & Rationale

### 3.1 Evaluation of Training Scale Alternatives

```
+-------------------------------------------------------------------------------------------------------+
|                                    STAGE C TRAINING TRADE-OFF MATRIX                                  |
+-----------------------------------+-----------------------------------+-------------------------------+
| CRITERION                         | OPTION A: 721-Case Development    | OPTION B: 941-Case Full Train |
+-----------------------------------+-----------------------------------+-------------------------------+
| Scientific Parity with Stage B    | EXACT MATCH (Same cohort & scale) | INCOMPARABLE (1.305x scaling) |
| Calibration Baseline Preservation | PRESERVED (ResUNet T*=2.8808)     | INVALIDATED (Requires re-fit) |
| Training Data Storage Location    | 100% in data/extracted_archive/   | Split across archive/ and raw/|
| GPU Compute Required (3 models)   | ~25.5 hours (~1.06 days)          | N/A (Requires all 5 models)   |
| GPU Compute Required (5 models)   | ~44.5 hours (~1.85 days)          | ~58.0 - 65.0 hours (~2.5 days)|
| Thermal & VRAM Headroom (RTX 4070)| SAFE (8.0h per run, <= 4.5 GB)    | ELEVATED (Thermal throttling) |
| Viva Schedule Compatibility       | HIGH (Fits Sem 7 timeline)        | RISKY (Requires full reset)   |
| Reserve Set Evaluation Validity   | Prohibited if trained on 941      | CATASTROPHIC LEAKAGE          |
+-----------------------------------+-----------------------------------+-------------------------------+
```

### 3.2 Evaluation of Retraining Options
1. **Design A (Targeted Matched Retraining — 3 Models on 721 Cases):**
   - Retrains U-Net, U-Net++, and Attention U-Net on 721 cases for 15 epochs under the exact Stage B optimization schedule.
   - Preserves ResUNet-2D (Epoch 13) and UNetResNet34 (Epoch 15) checkpoints intact.
   - *Advantage:* Immediately closes the 5-model comparison table in ~25.5 GPU hours while maintaining direct comparability with Stage B baseline metrics and preserving historical report documentation.
   - *Calibration Requirement:* Newly trained models do not inherit ResUNet's temperature; each must independently fit its own $T^*_m$ on `Val_cal` (50 cases).
2. **Design B (Fresh Retraining of All 5 Models on 721 Cases):**
   - Retrains all 5 architectures under a single locked runner script with fixed seed 42.
   - *Advantage:* Absolute run-locked parity across all 5 architectures.
   - *Severe Disadvantage:* Stochastically alters model weights, shifting the optimal temperature $T^*$ from 2.8808 and invalidating the empirical calibration figures in Research Edition V4.1.3 (~44.5 GPU hours).

### 3.3 Authoritative Phased Recommendation
We recommend a **Two-Phase Staged Strategy**:
- **Phase 1 (Semester 7 Viva Scope — Design A):** Execute matched retraining of U-Net, U-Net++, and Attention U-Net on the 721-case development cohort (15 epochs, ~25.5 GPU hours). This resolves the historical Stage A vs Stage B data-scale confounding and completes the 5-model architectural benchmark for the Semester 7 progress viva.
- **Phase 2 (Semester 8 Research Expansion — Controlled Data Scaling):** After the progress viva, execute a systematic data-scaling study on the full 941-case allocation across all 5 models (~60 GPU hours). Under Phase 2, any learning curve analysis must use a controlled, matched training protocol across data scales (e.g., 25%, 50%, 75%, 100% of the 941-case pool), rather than loosely comparing historical 5-epoch Stage A runs against 15-epoch Stage B runs.

---

## 4. Frozen Model Portfolio & Hyperparameter Matrix

### 4.1 Primary 2D Architectural Portfolio
The Stage C model portfolio consists strictly of the **five implemented 2D architectures**:

```
+----------------------------------------------------------------------------------------------------+
| 2D MODEL PORTFOLIO (5 ARCHITECTURES)                                                               |
+---+------------------+-----------------------------+--------------+--------------------------------+
| # | Model Identifier | Architectural Paradigm      | Params (M)   | Role in Stage C Study          |
+---+------------------+-----------------------------+--------------+--------------------------------+
| 1 | unet             | Standard 4-level U-Net      | 7.76 M (7,763,428) | Foundational Historical Baseline|
| 2 | unetpp           | Nested Dense Skips (UNet++) | 9.16 M (9,160,068) | Multi-scale Feature Aggregation|
| 3 | attention_unet   | Additive Attention Gates    | 7.85 M (7,852,160) | Spatial Salience Gating        |
| 4 | resunet          | Pre-Activation Residuals    | 8.11 M (8,114,604) | Primary Research Champion      |
| 5 | unet_resnet34    | ResNet-34 ImageNet Encoder  | 24.33 M (24,327,524)| Transfer Learning Fallback     |
+---+------------------+-----------------------------+--------------+--------------------------------+
```

### 4.2 Excluded & Deferred Architectures
1. **SegResNet-3D:**
   - *Status:* Exploratory feasibility pilot only (10 cases, 50 steps, $96 \times 96 \times 96$ crops).
   - *Evidence:* Peak allocated VRAM was **622.3 MB**; reported WT Dice (0.2658) and HD95 (88.87 mm) are **training-case reconstruction metrics** evaluated on `pilot_cases[0]`.
   - *Decision:* **EXCLUDED** from competitive Stage C 2D benchmark tables. Full volumetric 3D training is deferred to Semester 8 (Phase 8.3) on high-memory multi-GPU cloud hardware.
2. **nnU-Net v2:**
   - *Status:* Uninstalled and unexecuted in this repository.
   - *Evidence:* Runtime estimates (~200 GPU hours for 5-fold cross-validation) are theoretical extrapolations.
   - *Decision:* **DEFERRED** to Semester 8 (Phase 8.3).
3. **MedSAM2 / LiteMedSAM:**
   - *Status:* Proposed foundation models; promptable zero-shot inference requires official weights.
   - *Decision:* **DEFERRED** to Semester 8 (Phase 8.4).

---

### 4.3 Reconciled Hyperparameter Matrix

The table below reconciles all hyperparameters against `src/training/train_scaled.py`, `src/training/train.py`, and `results/full_training_15ep/*/config.json`:

| Hyperparameter Dimension | U-Net 2D | U-Net++ 2D | Attention U-Net 2D | ResUNet-2D (Champion) | UNetResNet34 (Reference) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Input Modalities** | 4 channels: `t1n, t1c, t2f, t2w` | 4 channels: `t1n, t1c, t2f, t2w` | 4 channels: `t1n, t1c, t2f, t2w` | 4 channels: `t1n, t1c, t2f, t2w` | 4 channels: `t1n, t1c, t2f, t2w` |
| **Input Spatial Grid** | $240 \times 240$ axial | $240 \times 240$ axial | $240 \times 240$ axial | $240 \times 240$ axial | $240 \times 240$ axial |
| **Base Feature Channels** | 32 | 32 | 32 | 32 | Backbone: 64, 128, 256, 512 |
| **Output Classes** | 4 (BG, NCR, ED, ET) | 4 (BG, NCR, ED, ET) | 4 (BG, NCR, ED, ET) | 4 (BG, NCR, ED, ET) | 4 (BG, NCR, ED, ET) |
| **Deep Supervision** | Disabled | Disabled (`False`) | Disabled | Disabled | Disabled |
| **Weight Initialization** | Kaiming Normal (bias=0) | Kaiming Normal (bias=0) | Kaiming Normal (bias=0) | Kaiming Normal (bias=0) | Encoder: ImageNet; Dec: Kaiming |
| **Pretrained Weights** | None (Scratch) | None (Scratch) | None (Scratch) | None (Scratch) | ResNet-34 ImageNet |
| **Training Cohort** | 721 cases (667 subjects) | 721 cases (667 subjects) | 721 cases (667 subjects) | 721 cases (667 subjects) | 721 cases (667 subjects) |
| **Epoch Budget** | 15 Epochs | 15 Epochs | 15 Epochs | 15 Epochs (Preserved Ep 13) | 15 Epochs (Preserved Ep 15) |
| **Checkpoint Selection** | Max Val Mean Dice on 155-case pool | Max Val Mean Dice on 155-case pool | Max Val Mean Dice on 155-case pool | Max Val Mean Dice on 155-case pool (Ep 13) | Max Val Mean Dice on 155-case pool (Ep 15) |
| **Selection Exposure** | **Yes (`Val_eval` & `Val_cal` overlap)** | **Yes (`Val_eval` & `Val_cal` overlap)** | **Yes (`Val_eval` & `Val_cal` overlap)** | **Yes (`Val_eval` & `Val_cal` overlap)** | **Yes (`Val_eval` & `Val_cal` overlap)** |
| **Batch Sampler** | `CaseSliceBatchSampler` | `CaseSliceBatchSampler` | `CaseSliceBatchSampler` | `CaseSliceBatchSampler` | `CaseSliceBatchSampler` |
| **Physical Batch Size** | 2 slices (within case) | 2 slices (within case) | 2 slices (within case) | 2 slices (within case) | 2 slices (within case) |
| **Gradient Clipping** | 1.0 (L2 norm clipping) | 1.0 (L2 norm clipping) | 1.0 (L2 norm clipping) | 1.0 (L2 norm clipping) | 1.0 (L2 norm clipping) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) | AdamW ($\beta_1=0.9, \beta_2=0.999$) |
| **Weight Decay** | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ | $1 \times 10^{-4}$ |
| **Learning Rate Schedule** | 2-ep Linear Warmup $\rightarrow$ 13-ep Cosine | 2-ep Linear Warmup $\rightarrow$ 13-ep Cosine | 2-ep Linear Warmup $\rightarrow$ 13-ep Cosine | 2-ep Linear Warmup $\rightarrow$ 13-ep Cosine | 2-ep Linear Warmup $\rightarrow$ 13-ep Cosine |
| **Initial / Peak LR** | $1 \times 10^{-4} \rightarrow 1 \times 10^{-3}$ | $1 \times 10^{-4} \rightarrow 1 \times 10^{-3}$ | $1 \times 10^{-4} \rightarrow 1 \times 10^{-3}$ | $1 \times 10^{-4} \rightarrow 1 \times 10^{-3}$ | Enc: $10^{-5} \rightarrow 10^{-4}$; Dec: $10^{-4} \rightarrow 10^{-3}$ |
| **Minimum LR Floor** | Configured $\eta_{\min}=1.0 \times 10^{-5}$ (Observed Ep 15: $2.438 \times 10^{-5}$) | Configured $\eta_{\min}=1.0 \times 10^{-5}$ (Observed Ep 15: $2.438 \times 10^{-5}$) | Configured $\eta_{\min}=1.0 \times 10^{-5}$ (Observed Ep 15: $2.438 \times 10^{-5}$) | Configured $\eta_{\min}=1.0 \times 10^{-5}$ (Observed Ep 15: $2.438 \times 10^{-5}$) | Enc: Conf $\eta_{\min}=1.0 \times 10^{-6}$ (Obs Ep 15: $2.438 \times 10^{-6}$); Dec: Conf $\eta_{\min}=1.0 \times 10^{-5}$ (Obs Ep 15: $2.438 \times 10^{-5}$) |
| **Loss Formulation** | Dice + CE ($\lambda_D=1, \lambda_{CE}=1$) | Dice + CE ($\lambda_D=1, \lambda_{CE}=1$) | Dice + CE ($\lambda_D=1, \lambda_{CE}=1$) | Dice + CE ($\lambda_D=1, \lambda_{CE}=1$) | Dice + CE ($\lambda_D=1, \lambda_{CE}=1$) |
| **Dice Background** | Excluded (`include_bg=False`) | Excluded (`include_bg=False`) | Excluded (`include_bg=False`) | Excluded (`include_bg=False`) | Excluded (`include_bg=False`) |
| **Sampling Ratio** | $25\%$ background slices | $25\%$ background slices | $25\%$ background slices | $25\%$ background slices | $25\%$ background slices |
| **Data Augmentation** | **None** (intensity z-score only) | **None** (intensity z-score only) | **None** (intensity z-score only) | **None** (intensity z-score only) | **None** (intensity z-score only) |
| **Random Seed** | 42 | 42 | 42 | 42 | 42 |
| **Observed / Est. GPU Time**| ~7.5 hours (Estimated) | ~9.5 hours (Estimated) | ~8.5 hours (Estimated) | 8.09 hours (Observed) | 10.80 hours (Observed) |

---

## 5. Experimental Work Packages Breakdown

```
+----------------------------------------------------------------------------------------------------+
| STAGE C EXPERIMENTAL WORK PACKAGES                                                                 |
+----------------------------------------------------------------------------------------------------+
|  [C1] 2D Architecture Benchmark      --> Harmonized 5-model Dice, HD95, latency, memory            |
|  [C2] Calibration & Uncertainty       --> Model-specific T*_m fitting, ECE, Brier, risk-coverage   |
|  [C3] Reliability-Aware AI Agent     --> 3-way triage, axial fallback (k=2), selective prediction  |
|  [C4] Missing-Modality Robustness     --> Controlled C0-C4 study, contrast failure cross-tabulation |
|  [C5] Advanced Modeling Feasibility   --> Scope boundary for 3D/nnU-Net deferral to Semester 8     |
+----------------------------------------------------------------------------------------------------+
```

### 5.1 Work Package C1: Harmonized 2D Architecture Benchmark
- **Primary Objective:** Establish a fair, scientifically rigorous comparison across the 5 architectures on identical validation data without data-scale confounding.
- **Evaluation Cohort:** 105 validation evaluation cases (`Val_eval`: 96 unique subjects, 8,382 background-filtered slices, 16,275 raw axial volume slices, 6,549 ground-truth foreground slices).
- **Epistemological Status:** Formally recorded that metrics evaluated on `Val_eval` are selection-exposed internal validation metrics (not an independent holdout benchmark).
- **Core Metrics:**
  - Volumetric Overlap: Whole Tumor (WT), Tumor Core (TC), and Enhancing Tumor (ET) Dice similarity coefficients.
  - Spatial Boundary Distance: 95th-percentile Hausdorff Distance (HD95) in millimetres (MONAI implementation with Euclidean distance transforms).
  - Computational Complexity: Total trainable parameters, floating-point operations (FLOPs).
  - Inference Latency:
    - *Model-Only Latency:* Measured in milliseconds per slice on GPU (batch size 1, excluding disk I/O).
    - *End-to-End Latency:* Measured in seconds per 3D volume (including NIfTI reading, intensity z-score normalization, slice inference, and argmax mask reconstruction).
- **Hypothesis H1:** Pre-activation residual connections (ResUNet-2D) and nested dense skips (U-Net++) will achieve statistically superior Dice overlap and lower boundary error (HD95) compared to standard U-Net ($p < 0.05$).

---

### 5.2 Work Package C2: Probability Calibration & Model-Specific Uncertainty Analysis
- **Primary Objective:** Quantify and remediate overconfidence in deep segmentation networks across all 5 architectures.
- **Epistemological Classification of `Val_cal`:**
  - Because `Val_cal` (50 cases, 3,982 filtered slices) was part of the 155-case validation pool monitored during training, temperature scaling fitting on `Val_cal` is **calibration fitting on a selection-exposed internal subset**, NOT an independent calibration exercise on unseen data.
  - Describing `Val_cal` as an independent held-out calibration set is strictly prohibited.
- **Distinction Between Calibration-Fit vs Evaluation Metrics:**
  - **Calibration-Fit Metrics on `Val_cal`:** NLL minimization objective value and post-fit ECE evaluated on `Val_cal` measure the mathematical goodness-of-fit of the scalar parameter $T_m^*$.
  - **Evaluation Metrics on `Val_eval`:** Uncalibrated vs calibrated ECE, MCE, Brier score, and NLL evaluated on `Val_eval` measure generalization of probability calibration across the evaluation subset.
  - Both metrics must be reported in separate columns/tables and never conflated.
- **Model-Specific Temperature Scaling Strategy:**
  - Optimal temperature parameters are architecture-specific:
    - Existing ResUNet-2D: $T^* = 2.8808$ (fitted on `Val_cal`).
    - Existing UNetResNet34: $T^* = 2.6267$ (fitted on `Val_cal`).
    - Newly trained models (U-Net, U-Net++, Attention U-Net): Each architecture must **independently fit its own optimal temperature $T_m^*$** on the 50 calibration cases (`Val_cal`, 3,982 slices) by minimizing multi-class cross-entropy Negative Log-Likelihood (NLL) via PyTorch L-BFGS-B optimization.
  - Temperature scaling formulation: $p_{i,m}^* = \text{softmax}(z_{i,m} / T_m^*)$ for logits $z_{i,m}$ from model $m$.
- **Uncertainty Quantification:**
  - Normalized Shannon predictive entropy:
    $$H(p^*) = -\frac{1}{\log(4)} \sum_{c=0}^{3} p_c^* \log\left(\max(p_c^*, 10^{-8})\right) \in [0, 1]$$
  - Slice-level metric: Mean calibrated predictive entropy computed over predicted foreground voxels ($p_i^* > 0$).
  - Empty foreground handling: Slices with zero predicted foreground voxels are assigned `mean_uncertainty = None` (strictly NOT 0.0).
- **Evaluation Metrics:**
  - Expected Calibration Error (ECE): 15 equal-width confidence bins in $[0, 1]$.
  - Maximum Calibration Error (MCE).
  - Brier Score: Multi-class mean squared error over predicted probability vectors.
  - Negative Log-Likelihood (NLL).
- **Hypothesis H2:** Temperature scaling will reduce ECE by $\ge 40\%$ across all architectures without altering class argmax predictions or degrading segmentation Dice scores.

---

### 5.3 Work Package C3: Reliability-Aware AI Agent & Selective Prediction
- **Primary Objective:** Evaluate the autonomous decision policy that triages segmentations between autonomous retention, model fallback, and specialist referral.
- **Decision Engine Actions:**
  1. `ACCEPT`: Calibrated foreground entropy $H < \tau$ on non-empty lesion. Autonomous retention for volumetric measurement.
  2. `FALLBACK_SWITCH`: Primary model predicts zero foreground voxels, but adjacent axial slices in radius $k=2$ confirm spatial tumor continuity ($N_{fg} \ge 1$), and secondary candidate model (`unet_resnet34`) recovers non-empty lesion segmentation.
  3. `FLAG_UNRELIABLE`: Foreground entropy $H \ge \tau$ OR unresolved complete empty prediction (either isolated slice without axial support or candidate model also predicts empty). Flagged for human neuroradiologist review.
- **Authoritative Fallback Rule Clarification:**
  - In the canonical B2 audit, fallback was triggered **strictly by empty primary prediction supported by axial context** (`enable_empty_prediction_fallback = True`, `axial_context_window = 2`, `min_neighbor_foreground_slices = 1`).
  - **Crucial Historical Correction:** `enable_uncertainty_fallback` was `False` and `uncertainty_fallback_threshold` was `None` in the canonical B2 execution. The threshold $\tau_{fb} = 0.40$ was **NOT** part of canonical B2 switching. In Stage C, uncertainty-gated fallback may be explored as a secondary research variant, but must be explicitly documented as an extension beyond canonical B2.
- **Descriptive Status of Risk-Coverage Sweeps:**
  - In Stage C, each architecture will be evaluated across a full risk-coverage threshold sweep $\tau \in [0.05, 0.50]$ (step size 0.05) to characterize its architecture-specific selective prediction trade-off curve.
  - **Methodological Invariant:** Risk-coverage sweeps on `Val_eval` are strictly **descriptive internal validation analyses**. The protocol strictly **prohibits selecting or optimizing an operating threshold $\tau$ on `Val_eval` and subsequently claiming unbiased or prospective performance at that chosen threshold** on the very same cohort. The historical operating point $\tau = 0.35$ established on `Val_cal` serves as the frozen baseline reference point.
- **Authoritative Denominator Accounting (105 Evaluation Cases):**
  - Total Raw Axial Slices: **16,275**
  - Empty Background Slices: **9,726**
  - Authoritative Ground-Truth Foreground Slices: **6,549**
  - Canonical B2 ResUNet-2D Triage Distribution:
    - Retained Slices ($H < 0.35$): **5,505** ($84.06\%$ coverage)
    - Flagged/Referred Slices ($H \ge 0.35$): **1,044** ($15.94\%$ referral rate)
- **Mandatory Reporting Invariant:** The agent's demonstrated empirical contribution is **selective risk reduction and failure referral**, NOT an increase in aggregate segmentation Dice. (In the canonical B2 audit, standalone ResUNet-2D achieved Mean Dice 0.8641 while the agent achieved 0.8640. Retained slices achieved Dice 0.8931 while referred slices had Dice 0.5405).

---

### 5.4 Work Package C4: Controlled Missing-Modality Robustness Study
- **Primary Objective:** Characterize failure modes and agent referral efficacy when specific MRI sequences are missing.
- **Study Conditions (5 Controlled States):**
  - **C0 (Baseline):** Complete 4-modality mpMRI (`t1n, t1c, t2f, t2w`).
  - **C1 (Withhold T1n):** Channel 0 zero-clamped after z-score standardization.
  - **C2 (Withhold T1c):** Channel 1 zero-clamped after z-score standardization (Loss of contrast enhancement).
  - **C3 (Withhold T2f):** Channel 2 zero-clamped after z-score standardization (Loss of edema attenuation). *(Corrected canonical channel index)*
  - **C4 (Withhold T2w):** Channel 3 zero-clamped after z-score standardization (Loss of structural water contrast). *(Corrected canonical channel index)*
- **Specific Failure Mode Under Investigation:**
  - In Stage B, withholding T1c (C2) caused catastrophic collapse of Enhancing Tumor Dice ($0.8286 \rightarrow 0.1684$) and Tumor Core Dice ($0.8597 \rightarrow 0.3132$).
  - However, mean foreground entropy increased only marginally ($0.2408 \rightarrow 0.2571$) and aggregate referral increased only from $12.37\%$ to $14.68\%$.
- **Stage C Protocol Additions:**
  1. *Direct Failure-to-Referral Cross-Tabulation:* Record the exact proportion of severe slice failures ($\text{Dice} < 0.50$) that were successfully referred vs falsely retained across each condition.
  2. *Pre-Inference Tensor Integrity Validator:* Evaluate an upstream validator that detects flatline/missing modality channels prior to neural network inference, demonstrating why input validation must augment uncertainty quantification.

---

### 5.5 Work Package C5: Advanced Architecture Feasibility & Scope Boundaries
- **Boundary Decision:** Full 3D volumetric convolution (SegResNet-3D) and automated pipeline configuration (nnU-Net v2) are **formally deferred to Semester 8 (Phase 8.3)**.
- **Rationale:** Stage C establishes an unassailable 2D benchmark on available hardware (RTX 4070 8GB). Introducing unvalidated 3D models or unexecuted nnU-Net runs risks computational over-extension and methodology confusion.

---

## 6. Resolution of Known Configuration Discrepancies

| Discrepancy Topic | Historical Contradiction | Authoritative Canonical Resolution | Primary Evidence Source |
|---|---|---|---|
| **Axial Context Window Radius** | Textual references in early G1-A3 drafts mentioned radius $r=1$ (3-slice window). | **Canonical parameter is $k=2$** (radius 2, inspecting up to 4 neighbors in $[z-2, z+2]$). | `src/agent/state.py` line 69, `src/agent/runner.py` line 93, `tools/demo_app.py`, November 10 B2 execution log. |
| **Operating Thresholds ($\tau, \tau_{fb}$)** | Earlier drafts claimed $\tau_{fb}=0.40$ was part of the canonical B2 switching policy. | **Canonical B2 fallback is triggered strictly by empty primary prediction with axial support ($k=2$)**. The threshold $\tau=0.35$ is the referral threshold for ResUNet-2D. The parameter $\tau_{fb}=0.40$ was an unexecuted exploratory field in `PolicyConfig` (`enable_uncertainty_fallback=False`). | `tools/reevaluate_missing_modality_policy.py` lines 180–191, `results/gate6a_remediation_v1/calibration_parameters.json`. |
| **Missing Modality Channel Mapping** | `results/missing_modality_robustness/` labeled Channel 2 as T2w and Channel 3 as T2f. | **Channel 2 is T2-FLAIR (`t2f`); Channel 3 is T2-Weighted (`t2w`)**, strictly matching `src.data.dataset.MODALITIES`. | `src/data/dataset.py` line 19, G1-B1.3 audit, G1-D0.2.1 investigation. |
| **Agent Policy Evolution** | Policy described as "future work" in early report drafts. | **Policy v3.0 is fully implemented and tested** in `src/agent/` and integrated into `tools/demo_app.py`. | `src/agent/policy.py`, `tests/test_demo_app_agent.py`, `tests/test_demo_app_end_to_end.py`. |

---

## 7. Statistical Analysis Plan & Reproducibility Standards

### 7.1 Unit of Analysis & Hierarchical Clustering
- **Primary Unit of Analysis:** **Patient / Biological Subject** ($N=96$ in evaluation cohort).
- **Secondary Unit of Analysis:** **Case Volume** ($N=105$ in evaluation cohort).
- **Unit of Observation:** 2D Axial Slice.
  - Raw Volumetric Slices: $N = 16,275$ ($105 \times 155$).
  - Background-Filtered Loader Slices: $N = 8,382$ (`background_slice_ratio=0.25`).
  - Ground-Truth Foreground Slices: $N = 6,549$.
- **Methodological Invariant:** Slices originating from the same patient brain volume are spatially correlated. **Slice-level observations must never be pooled as independent samples for hypothesis testing.** All Dice and HD95 metrics must be aggregated per-patient before statistical comparison.

### 7.2 Hypothesis Testing & Multiple Testing Correction
- **Distribution-Free Testing:** Continuous segmentation metrics (Dice, HD95) exhibit non-Gaussian, skewed distributions with boundary ceiling effects. All paired model comparisons must be evaluated using the two-sided **Paired Wilcoxon Signed-Rank Test**.
- **Rigorous Distinction of Correction Methods:**
  - **Holm–Bonferroni Method (Step-Down):** Controls the **Family-Wise Error Rate (FWER)** in the strong sense, defined as the probability of making at least one Type I error ($P(V \ge 1) \le \alpha$) across the family of tests.
  - **Benjamini–Hochberg Procedure:** Controls the **False Discovery Rate (FDR)**, defined as the expected proportion of false positive discoveries among all rejected hypotheses ($E[Q] \le q$).
- **Designated Multiple Testing Framework:**
  - **Primary Benchmark Decision Method:** **Holm–Bonferroni step-down procedure** at family-wise $\alpha = 0.05$ across the $k = \binom{5}{2} = 10$ pairwise architectural comparisons on patient-level Mean Dice. This provides strong mathematical protection against any false claim of architecture superiority in the core 5-model benchmark.
  - **Secondary / Exploratory Reporting Method:** **Benjamini–Hochberg FDR procedure** at $q = 0.05$ applied across multi-compartment exploratory comparisons (WT, TC, ET, HD95) to balance discovery and false discovery control.
- **Confidence Intervals:** Report 95% Bootstrap Confidence Intervals computed via 1,000 cluster-bootstrap resamples at the subject level.

### 7.3 Reproducibility & Pre-Defined Stopping Criteria
- **Random Seed Lock:** Global random seed fixed at `42` (`torch.manual_seed(42)`, `np.random.seed(42)`, Python `random.seed(42)`).
- **Automated Failure / Early Stopping Criteria:**
  - Loss Explosion / NaN: If training loss or validation loss produces `NaN` or `Inf`, execution halts immediately.
  - Gradient Norm Alert: If $\|\nabla_\theta \mathcal{L}\|_2 > 100.0$ prior to clipping, log gradient warning.
  - VRAM Guard: Peak allocated GPU memory must remain $\le 7.0$ GB on the 8 GB RTX 4070. If allocation exceeds 7.5 GB, training terminates to prevent OS paging.

---

## 8. Computational Resource Budget & Execution Schedule

### 8.1 Local Hardware Profile
- **Workstation:** Acer Predator Helios (Laptop)
- **CPU:** 13th Gen Intel Core i9-13900HX (24 cores, 32 threads)
- **RAM:** 16.0 GB physical DDR5 RAM (1 module, 5600 MHz)
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6 VRAM, 7.9956 GB addressable)
- **Environment:** Windows 11, Python 3.11.9, PyTorch `2.11.0+cu128`, CUDA 12.8

### 8.2 Compute Time Breakdown (Stage C Phase 1)

| Task / Work Package | Target Architecture / Script | GPU Time (Observed / Estimated) | Memory Footprint |
| :--- | :--- | :---: | :---: |
| **C1.1: U-Net 2D Retraining** | Standard U-Net (721 cases, 15 ep) | ~7.5 hours (Estimated) | ~3.8 GB VRAM |
| **C1.2: U-Net++ 2D Retraining** | U-Net++ 2D (721 cases, 15 ep) | ~9.5 hours (Estimated) | ~4.4 GB VRAM |
| **C1.3: Attention U-Net Retraining** | Attention U-Net 2D (721 cases, 15 ep) | ~8.5 hours (Estimated) | ~4.1 GB VRAM |
| **C1.4: ResUNet-2D Baseline** | Epoch 13 Checkpoint | **0.0 hours (Preserved)** | Checkpoint verified |
| **C1.5: UNetResNet34 Baseline** | Epoch 15 Checkpoint | **0.0 hours (Preserved)** | Checkpoint verified |
| **C1–C4: 5-Model Evaluation Suite** | Full 105-case evaluation & triage | ~2.5 hours (Estimated) | ~2.5 GB VRAM |
| **Total Estimated Stage C Compute** | **Phase 1 Retraining & Evaluation** | **~28.0 GPU hours (~1.17 days)** | Within local capacity |

---

## 9. Limitations & Mandatory Disclosures

1. **Selection Exposure of Both Validation Cohorts (`Val_cal` & `Val_eval`):**
   The 50 calibration cases and 105 evaluation cases were both part of the 155-case validation pool that guided epoch checkpointing for Stage B and Stage C models. Temperature fitting on `Val_cal` represents calibration fitting on a selection-exposed internal subset, and metrics evaluated on `Val_eval` reflect internal validation performance with selection exposure. **Neither cohort may be described as an independent held-out test set.**
2. **Descriptive Status of Risk-Coverage Analyses:**
   Risk-coverage curves on `Val_eval` are descriptive internal validation analyses. The protocol strictly prohibits optimizing or selecting an operating threshold $\tau$ on `Val_eval` and subsequently claiming unbiased performance at that chosen threshold.
3. **2D In-Plane Slice Representation:**
   The models operate on 2D axial slices. While axial context ($k=2$) incorporates adjacent spatial continuity into the agent triage rules, the underlying segmentation models do not perform full 3D anisotropic convolutions.
4. **Synthetic Missing-Modality Approximation:**
   Withholding modalities via post-normalization zero-clamping evaluates network resilience to zeroed input channels, but does not simulate non-zero scanner noise or corrupted RF artifacts.
5. **Academic Research Prototype Disclaimer:**
   This system is an academic research demonstration for evaluating uncertainty calibration and selective prediction triage. It is not approved by regulatory bodies (FDA, CDSCO, CE-MDR) and must not be used for clinical decision-making.
6. **Test Quarantine Status:**
   The test manifest was historically parsed during freeze audits in October 2026; however, no evidence of quarantined test MRI-volume or mask loading, evaluation, or metric computation was found in inspected records.

---

## 10. Project Governance & Decision Approval Status

Under the self-governed student academic project framework, experimental decisions are governed by the Student Project Team and Project Owner. The following core decisions from [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv) are formally approved:

1. **Decision DEC-C-01 & DEC-C-03 (Training Scope & 2D Retraining):**
   *Status: APPROVED (2026-11-13) by Student Project Team / Project Owner.*
   Approve Phase 1 (Design A: matched retraining of U-Net, U-Net++, and Attention U-Net on 721 development cases for 15 epochs, ~25.5 GPU hours) to establish architectural parity while preserving Stage B baselines for the Semester 7 progress review.
2. **Decision DEC-C-02 (Architecture Portfolio Scope):**
   *Recommendation:* Formally lock the Stage C comparative benchmark to the 5 implemented 2D architectures, officially deferring SegResNet-3D and nnU-Net v2 to Semester 8 Phase 8.3.
3. **Decision DEC-C-04 (Checkpoint-Selection Procedure & Disclosure):**
   *Status: APPROVED (2026-11-13) by Student Project Team / Project Owner.*
   Approve the epoch-end Max Validation Mean Dice checkpoint selection on the 155-case validation pool for newly trained models, with mandatory disclosure that both `Val_cal` and `Val_eval` are selection-exposed internal validation cohorts.
4. **Decision DEC-C-05 (Pretrained Backbone Policy):**
   *Recommendation:* Formally approve the hybrid initialization policy permitting UNetResNet34 to use ImageNet pretrained encoder weights with mandatory disclosure in all benchmark tables.
5. **Decision DEC-C-07 (Model-Specific Calibration Parameters):**
   *Recommendation:* Approve independent fitting of $T^*_m$ on `Val_cal` (50 cases) as calibration fitting on a selection-exposed internal subset, and descriptive risk-coverage sweeps ($\tau \in [0.05, 0.50]$) on `Val_eval` without post-hoc threshold selection.
6. **Decision DEC-C-09 (Test Cohort Quarantine):**
   *Status: APPROVED (2026-11-13) by Student Project Team / Project Owner.*
   Reaffirm the absolute quarantine of the 155-case sealed test cohort (`test_manifest.json` / `data/quarantine_test/`), keeping it unopened throughout Stage C until the pre-registered Semester 8 capstone defense.

---

## 11. Final Decision Gate Recommendation

# **APPROVED FOR PHASE 1 EXECUTION**

The Stage C experimental protocol has been methodologically corrected, mathematically closed, and aligned with available computational resources. It enforces strict data quarantine boundaries, separates architecture-specific calibration parameters, and establishes an auditable roadmap for Semester 7 progress review and Semester 8 capstone completion. Phase 1 training execution is authorized under self-governed project decisions DEC-C-01, DEC-C-03, and DEC-C-04.
