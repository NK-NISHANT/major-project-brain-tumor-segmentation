# Stage C Phase 1 Comparative Analysis: U-Net, U-Net++, and Attention U-Net

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md)
**Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)
**Decision Register:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Individual Experiment Reports:**
- Standard U-Net 2D: [`G1_P4_UNET_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_UNET_STAGE_C_TRAINING_REPORT.md)
- U-Net++ 2D: [`G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md)
- Attention U-Net 2D: [`G1_P4_ATTENTION_UNET_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_ATTENTION_UNET_STAGE_C_TRAINING_REPORT.md)
**Date:** November 15, 2026
**Document Status:** **FINAL CONSOLIDATED COMPARATIVE ANALYSIS**

---

## 1. Executive Summary & Experiment Overview

Under Stage C Phase 1 of the research project, three distinct 2D convolutional segmentation architectures were trained from scratch on the authorized development cohort under a unified, locked experimental protocol. The overarching objective of Phase 1 was to establish rigorous baseline models on the enlarged 721-case training cohort prior to downstream post-hoc calibration (Work Package C2) and blind test-set evaluation (Work Package C3).

The three architectures exhibit distinct execution histories:
- **Standard U-Net 2D:** Experienced execution interruptions during Epoch 5 and Epoch 6 with CUDA illegal-instruction exceptions and associated NVIDIA `nvlddmkm` driver error events during earlier asynchronous runs. Checkpoint state was recovered from completed epoch boundaries, and training was successfully completed for Epochs 6–15 under synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`). During this final 6.22-hour synchronous resumption window, **zero NVIDIA driver error events** were recorded. The underlying root cause of the earlier asynchronous GPU execution failures remains unresolved.
- **U-Net++ 2D & Attention U-Net 2D:** Both models were trained from scratch and completed their full 15-epoch budgets entirely under synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`) with **zero runtime errors, zero CUDA exceptions, and zero recorded NVIDIA driver events**.

### 1.1 Consolidated Protocol Matrix

| Experimental Dimension | Locked Experimental Protocol | Compliance Finding across All 3 Models |
| :--- | :--- | :---: |
| **Training Cohort** | 721 cases / 667 unique subjects (59,020 slices: 47,228 foreground, 11,792 background) | Verified (`development_train_manifest.json`) |
| **Validation Pool** | 155 cases / 143 unique subjects (12,364 slices: `Val_eval` + `Val_cal`) | Verified (`validation_manifest.json`) |
| **Cohort Disjointness** | Case-level and subject-level $\text{Train} \cap \text{Val} = \emptyset$ | Verified programmatically |
| **Test Quarantine** | Strict quarantine of `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` | **100% Sealed (Zero test cases accessed)** |
| **Input Modalities** | 4-channel co-registered MRI (`t1n`, `t1c`, `t2f`, `t2w`), $240 \times 240$ axial slices | Verified |
| **Output Classes** | 4 classes (0: Background, 1: Necrotic Core, 2: Peritumoral Edema, 3: Enhancing Tumor) | Verified |
| **Initialization** | Scratch training via Kaiming Normal initialization (`bias=0`) | Verified |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$) | Verified |
| **Base Learning Rate** | $1 \times 10^{-3}$ ($0.001000$) | Verified |
| **Learning Rate Schedule** | 2-epoch Linear Warmup (from $0.1 \times \text{base\_lr}$) $\rightarrow$ 13-epoch CosineAnnealingLR ($T_{\max}=13, \eta_{\min}=10^{-5}$) | Verified step schedule |
| **Loss Function** | Multiclass Dice + Cross-Entropy ($\lambda_D=1.0, \lambda_{CE}=1.0$, Dice excludes Background) | Verified |
| **Gradient Clipping** | Maximum L2 gradient norm = 1.0 | Verified |
| **Batch Sampler** | `CaseSliceBatchSampler` (within-case axial slice sampling, batch size = 2) | Verified |
| **Random Seed** | 42 | Verified |
| **Execution Mode** | Synchronous CUDA (`CUDA_LAUNCH_BLOCKING=1`) | Applied to all models; U-Net ran Ep 1–5 async, then Ep 6–15 sync |

---

## 2. Architecture & Trainable Parameter Comparison

The three Phase 1 architectures evaluate distinct structural strategies for multi-scale feature propagation and skip-connection connectivity:

1. **Standard U-Net 2D (`UNet2D`):** Classic encoder-decoder structure with direct identity skip connections linking corresponding encoder and decoder resolution stages.
2. **U-Net++ 2D (`UNetPlusPlus2D`):** Nested, dense skip pathways that connect encoder and decoder stages through intermediate convolutional blocks ($X^{i,j}$), redesigned to reduce the semantic gap between feature maps before fusion. Deep supervision was disabled in accordance with the locked Stage B configuration.
3. **Attention U-Net 2D (`AttentionUNet2D`):** Standard encoder-decoder backbone augmented with additive spatial attention gates at skip junctions. Coarser gating signals ($g$) modulate incoming skip feature maps ($x$) via additive compatibility scoring before concatenation.

### Parameter and Structural Overview

| Model Architecture | Implementation Class | Structural Feature | Total Parameters | Parameter Delta vs U-Net |
| :--- | :--- | :--- | :---: | :---: |
| **Standard U-Net 2D** | `UNet2D` | Direct identity skip connections | **7,763,428** (~7.76 M) | Baseline (0.0%) |
| **U-Net++ 2D** | `UNetPlusPlus2D` | Dense nested skip convolutions ($X^{i,j}$) | **9,160,068** (~9.16 M) | **+1,396,640 (+17.99%)** |
| **Attention U-Net 2D** | `AttentionUNet2D` | Additive spatial attention gates ($W_g, W_x, \psi$) | **7,852,160** (~7.85 M) | **+88,732 (+1.14%)** |

---

## 3. Comparative Quantitative Performance

All models were evaluated on the 155-case validation pool at the end of every epoch. Checkpoint selection identified **Epoch 15** as the optimal checkpoint for all three architectures based on maximum Validation Mean Dice.

> [!IMPORTANT]
> **Selection-Exposure Classification:**
> All metrics reported below were computed on the 155-case validation pool monitored during training. In accordance with [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md), these figures are formally classified as **selection-exposed internal validation metrics**. They do not represent unbiased generalization performance on unseen data or clinical readiness.

### 3.1 Primary Validation Metrics (Epoch 15 Checkpoint)

| Metric | Standard U-Net 2D | U-Net++ 2D | Attention U-Net 2D | Numerical Range ($\Delta_{\max-\min}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Selection-Exposed Val Mean Dice** | 0.8594 (0.859373) | **0.8602** (0.860218) | 0.8542 (0.854212) | 0.0060 (0.60 pp) |
| **Selection-Exposed Val Mean IoU** | 0.7554 (0.755392) | **0.7568** (0.756770) | 0.7482 (0.748189) | 0.0086 (0.86 pp) |
| **Final Validation Loss** | 0.4589 | **0.4280** | 0.5013 | 0.0733 |
| **Final Training Loss** | 0.1815 | **0.1728** | 0.1843 | 0.0115 |

### 3.2 Compartment-Wise Metric Breakdown

| Tumor Sub-Compartment | Standard U-Net 2D | U-Net++ 2D | Attention U-Net 2D | Descriptive Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Enhancing Tumor (ET / Class 3) Dice** | 0.9026 | **0.9047** | 0.9035 | Attn U-Net recorded +0.0009 over U-Net; U-Net++ highest |
| **Enhancing Tumor (ET / Class 3) IoU** | 0.8225 | **0.8260** | 0.8240 | Follows ET Dice progression |
| **Peritumoral Edema (ED / Class 2) Dice** | 0.8662 | **0.8664** | 0.8645 | High numerical parity across all architectures ($\Delta = 0.0019$) |
| **Peritumoral Edema (ED / Class 2) IoU** | 0.7639 | **0.7642** | 0.7613 | High parity across all architectures |
| **Necrotic Core (NCR / Class 1) Dice** | 0.8094 | **0.8096** | 0.7947 | Attn U-Net recorded lower NCR Dice (-0.0147 vs U-Net) |
| **Necrotic Core (NCR / Class 1) IoU** | 0.6798 | **0.6801** | 0.6593 | Follows NCR Dice progression |
| **Background (BG / Class 0) Dice** | 0.9986 | 0.9986 | 0.9986 | Identical across all three models |
| **Background (BG / Class 0) IoU** | 0.9972 | 0.9972 | 0.9973 | Identical across all three models |

---

## 4. Hardware Telemetry & Resource Footprint

All models were trained on an Acer Predator Helios 16 workstation equipped with an NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MiB GDDR6), 13th Gen Intel Core i9-13900HX CPU, and 16 GB DDR5 RAM.

| Telemetry / Resource Dimension | Standard U-Net 2D | U-Net++ 2D | Attention U-Net 2D | Descriptive Comparison |
| :--- | :---: | :---: | :---: | :--- |
| **Trainable Parameters** | 7.76 M (7,763,428) | 9.16 M (9,160,068) | 7.85 M (7,852,160) | Attn U-Net +1.14% vs U-Net; U-Net++ +17.99% |
| **Peak Process VRAM Allocation** | **446.3 MB** | 1,111.3 MB | 489.0 MB | Attn U-Net +9.6% vs U-Net; U-Net++ +149.0% |
| **Total Cumulative Runtime** | **8.24 hours** *(Heterogeneous)* | 15.44 hours *(Homogeneous)* | 10.70 hours *(Homogeneous)* | Not a controlled speed benchmark |
| **Average Epoch Duration (Measured Averages)** | ~37.3 min/epoch *(Sync Ep 6–15 only)* | ~61.8 min/epoch *(Sync Ep 1–15)* | ~42.8 min/epoch *(Sync Ep 1–15)* | Measured averages; not a controlled speed benchmark |
| **GPU Temperature Range** | 58 °C – 78 °C | 62 °C – 78 °C | 55 °C – 74 °C | Stable under active cooling across all runs |
| **Windows Driver Events (`nvlddmkm`)** | Events during async; **0 in final 6.22h sync** | **0 events** (full run) | **0 events** (full run) | Disclosed execution-history distinction |

### Methodological Resource Interpretation
- **Parameter Footprint:** Attention U-Net introduces additive spatial attention gates with an increase of 88,732 parameters (+1.14%) over standard U-Net, whereas U-Net++ adds 1,396,640 parameters (+17.99%) due to intermediate dense convolutions.
- **Process VRAM Footprint:** Attention U-Net recorded a peak process VRAM allocation of 489.0 MB, which is approximately 9.6% higher than U-Net (446.3 MB) and lower than U-Net++ (1,111.3 MB) under these specific experimental conditions.
- **Runtime Comparability & Heterogeneous Execution:**
  The recorded total runtimes are:
  - Standard U-Net: 8.24 hours (494.4 minutes)
  - U-Net++: 15.44 hours (926.39 minutes)
  - Attention U-Net: 10.70 hours (641.70 minutes)
  However, U-Net's execution was heterogeneous: its first five completed epochs ran under asynchronous execution (~24.3 min/epoch), while its final ten epochs were completed under synchronous execution (averaging approximately 37.3 min/epoch and totaling 6.22 hours). In comparison, Attention U-Net averaged approximately 42.8 min/epoch (641.70 minutes across 15 epochs) and U-Net++ averaged approximately 61.8 min/epoch (926.39 minutes across 15 epochs) under fully synchronous CUDA execution. Therefore, these measured averages cannot be presented as a controlled architecture-speed benchmark. The U-Net average of 37.3 minutes/epoch represents only the ten-epoch synchronous resumption window (Epochs 6–15), not all 15 epochs.
- **Disciplinary Boundary:** These measured training figures do not constitute proof of superior deployment efficiency, energy efficiency, inference throughput, or clinical suitability. Operational deployment metrics (latency, batch throughput, energy consumption) require dedicated benchmarking that has not been conducted.

---

## 5. Learning Dynamics & Epoch Interpretation

### 5.1 Training Loss Progression
Across all three models, training loss decreased steadily over the 15-epoch schedule:
- **Epoch 1 Train Loss:** U-Net: 0.4527 | U-Net++: 0.4537 | Attention U-Net: 0.4717
- **Epoch 5 Train Loss:** U-Net: 0.2195 | U-Net++: 0.2151 | Attention U-Net: 0.2193
- **Epoch 10 Train Loss:** U-Net: 0.1963 | U-Net++: 0.1867 | Attention U-Net: 0.1951
- **Epoch 15 Train Loss:** U-Net: 0.1815 | U-Net++: 0.1728 | Attention U-Net: 0.1843

U-Net++ recorded the lowest final training loss (0.1728), standard U-Net achieved 0.1815, and Attention U-Net achieved 0.1843.

### 5.2 Gradient Norms and Clipping
All models demonstrated rapid gradient stabilization following the 2-epoch linear warmup:
- During Epoch 1, gradient clipping was active on 52.0% to 55.9% of batches with pre-clip norms of 1.69 to 1.84.
- By Epoch 4, pre-clip gradient norms decreased to 0.29–0.36, and clipped batches dropped below 5%.
- At Epoch 15, pre-clip gradient norms stabilized at 0.24–0.31, with clipped batches at approximately 2.1%–2.6%.

### 5.3 Epoch Budget & Convergence Interpretation
All three models achieved their highest recorded validation mean Dice at **Epoch 15** within the prescribed 15-epoch training budget:
- Standard U-Net: 0.8594 at Epoch 15
- U-Net++: 0.8602 at Epoch 15
- Attention U-Net: 0.8542 at Epoch 15

> [!NOTE]
> **Scientific Interpretation:**
> All three models achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.

---

## 6. Learning-Rate Schedule Verification & Findings

An empirical audit of the learning-rate schedule implementation in `src/training/train_scaled.py` and the resulting artifact histories was performed to reconcile the recorded minimum learning rates.

### 6.1 Audit Investigation

The training script configures a chained `SequentialLR` scheduler:
```python
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=2
)
cosine_scheduler = PerGroupCosineAnnealingLR(
    optimizer, T_max=13, eta_min=1e-5
)
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[2]
)
```

In the training loop (`train_scaled.py`, lines 614–678):
1. **Pre-Epoch Extraction:** At line 614, `current_lrs = [pg["lr"] for pg in optimizer.param_groups]` records the learning rate **active during the epoch's training updates**.
2. **Epoch Execution:** Training is performed using `current_lrs`.
3. **Post-Epoch Stepping:** At line 641, `scheduler.step()` is invoked at the conclusion of the epoch, advancing the scheduler and updating `optimizer.param_groups` to the learning rate for the subsequent epoch.
4. **History Logging:** At line 678, `row["learning_rate_primary"] = current_lrs[-1]` records the pre-step learning rate in `history.csv`.

### 6.2 Empirical Verification Results

Simulated stepping and inspection of `final_model.pt` confirmed the following step sequence across all three experiments:

| Epoch | Active LR During Training Updates (`current_lrs`) | History Logged LR | Post-Epoch Stepped LR (`next_lrs`) | Schedule State |
| :---: | :---: | :---: | :---: | :--- |
| **1** | $0.00010000$ | $0.00010000$ | $0.00055000$ | Warmup step 1 (0.1x base) |
| **2** | $0.00055000$ | $0.00055000$ | $0.00100000$ | Warmup step 2 (0.55x base) |
| **3** | $0.00100000$ | $0.00100000$ | $0.00098562$ | Full base LR; Cosine step 1 |
| **4** | $0.00098562$ | $0.00098562$ | $0.00094330$ | Cosine step 2 |
| ... | ... | ... | ... | ... |
| **14** | $0.00006670$ | $0.00006670$ | $0.00002438$ | Cosine step 12 |
| **15** | $0.00002438$ | $0.00002438$ | **$0.00001000$** | Cosine step 13 ($T_{\max}=13$, reaches $\eta_{\min}$) |

### 6.3 Checkpoint State Verification

Direct inspection of `final_model.pt` for all three completed models confirms:
- **`optimizer_state_dict["param_groups"][0]["lr"]`:** Exactly **`1.0e-05`** ($0.000010$)
- **`scheduler_state_dict["_schedulers"][1]["_last_lr"]`:** Exactly **`[1.0e-05]`**
- **`scheduler_state_dict["last_epoch"]`:** Exactly **`15`**

### 6.4 Verification Finding: Case A (Logging Convention)
- **Status:** **Case A — Logging Convention Confirmed.**
- **Explanation:** The scheduler followed the approved 2-epoch warmup + 13-epoch cosine protocol exactly. The optimizer parameter group reached the configured minimum learning rate floor of $\eta_{\min} = 1.0 \times 10^{-5}$ upon completion of Epoch 15. The value recorded in `history.csv` for Epoch 15 ($0.00002438$) represents the learning rate active *during* the parameter updates of that epoch. No schedule deviation occurred. Historical artifacts and training code remain untouched.

---

## 7. Methodological Limitations & Scientific Boundaries

To uphold academic integrity and prevent misleading research claims, the following methodological boundaries are formally established:

1. **Selection-Exposed Validation:** Model selection was performed on the 155-case internal validation pool (`Val_eval` + `Val_cal`). The reported metrics reflect performance on the selection-monitored set and are therefore exposed to selection bias.
2. **No Claim of Generalization or Clinical Readiness:** These figures do not represent independent generalization, unbiased test set accuracy, or clinical viability. Generalization can only be established on the quarantined, unexposed test cohort.
3. **Single-Seed Statistical Interpretation & Stochastic Resumption:**
   Because each architecture was evaluated using a single configured random seed (seed 42), run-to-run variability and statistical significance cannot be estimated. Small numerical differences should therefore be interpreted descriptively rather than as robust performance differences. Furthermore, U-Net's checkpoint recovery did not preserve the complete RNG state; the same configured seed does not establish an identical stochastic execution trajectory after resumption.
4. **No Architectural Causality Claims:** The numerical differences observed (+0.0008 Mean Dice for U-Net++, +0.0009 ET Dice for Attention U-Net, -0.0147 NCR Dice for Attention U-Net) cannot be causally attributed to specific architectural mechanisms without controlled ablation studies and variance estimation.
5. **Operational Efficiency & Runtime Comparability Boundaries:** Measured peak process VRAM and training runtimes reflect the training environment and batch sampler parameters; they do not establish inference throughput, memory consumption during batched deployment, or energy efficiency. Moreover, total recorded training runtimes are not directly comparable across models due to U-Net's mixed asynchronous/synchronous execution history.

---

## 8. Balanced Research Conclusions

Evaluating the empirical evidence strictly within these established boundaries:

1. **U-Net++ vs. Standard U-Net:**
   - U-Net++ recorded a marginally higher selection-exposed validation Mean Dice than standard U-Net (**0.8602** vs. **0.8594**, $\Delta = +0.0008$ or 0.08 percentage points).
   - This numerical difference does not establish meaningful architectural superiority. The experiment does not confirm improved spatial refinement or superior generalization.
   - U-Net++ required approximately 18.0% more parameters and recorded 149.0% higher peak process VRAM during training; its total recorded runtime was 15.44 hours (under full synchronous execution) compared to U-Net's 8.24 hours (which was split between asynchronous and synchronous modes, preventing direct controlled speed comparisons).
2. **Attention U-Net vs. Standard U-Net:**
   - Attention U-Net recorded a selection-exposed validation Mean Dice of **0.8542**, compared to **0.8594** for standard U-Net ($\Delta = -0.0052$).
   - On Enhancing Tumor (ET), Attention U-Net recorded an ET Dice of **0.9035**, compared to **0.9026** for standard U-Net ($\Delta = +0.0009$ or 0.09 percentage points). This descriptive difference does not demonstrate improved tumor sensitivity or prove that attention gates focus better on tumor regions.
   - On Necrotic Core (NCR), Attention U-Net recorded **0.7947**, compared to **0.8094** for standard U-Net ($\Delta = -0.0147$). This difference is reported descriptively without attributing it to an unverified architectural mechanism.
   - Attention U-Net required 1.14% more parameters and 9.6% higher peak process VRAM than standard U-Net during training.
3. **Overall Architectural Parity:**
   - Across all three architectures, validation Mean Dice remained tightly clustered between 0.8542 and 0.8602 (a span of only 0.0060).
   - Performance on the dominant tumor compartments—Peritumoral Edema (0.8645–0.8664) and Enhancing Tumor (0.9026–0.9047)—exhibited near-parity across all three model families.

---

## 9. Recommended Next Research Steps

Following the governance blueprint [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md), the recommended subsequent research actions are:

1. **Stage C Work Package C2 (Post-Hoc Calibration):**
   - Conduct post-hoc temperature scaling and selective prediction calibration on the dedicated calibration split (`Val_cal`, 77 cases) for all three retrained architectures:
     - Compute optimal temperature parameters ($T^*_{\text{unet}}$, $T^*_{\text{unetpp}}$, $T^*_{\text{attn\_unet}}$).
     - Evaluate calibration metrics (Expected Calibration Error / ECE, Brier Score, reliability diagrams) on the evaluation split (`Val_eval`, 78 cases).
2. **Uncertainty & Rejection Architecture Integration:**
   - Interface the calibrated models with the uncertainty estimation module (`src/uncertainty/calibration.py`) and agent rejection mechanisms to establish coverage-versus-risk profiles across all three models.
3. **Stage C Work Package C3 (Sealed Test Evaluation):**
   - Only after all model checkpoints, calibration parameters, and decision thresholds are frozen, execute a single, unblinded evaluation pass over the quarantined test cohort (`data/quarantine_test/`, 155 cases) to establish true out-of-sample generalization. Note: The 220 cases belong to the separate historical development reserve and must not be confused with the 155-case sealed test cohort.
