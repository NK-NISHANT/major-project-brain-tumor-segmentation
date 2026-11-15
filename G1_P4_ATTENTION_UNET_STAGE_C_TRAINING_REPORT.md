# G1-P4.4 — Stage C Phase 1 Attention U-Net Training Report (15 Epochs)

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md)
**Stage C Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)
**Decision Register Reference:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Prior Baseline Reports:**
- U-Net Baseline: [`G1_P4_UNET_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_UNET_STAGE_C_TRAINING_REPORT.md)
- U-Net++ Baseline: [`G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md)
**Experiment Output Directory:** [`results/stage_c/phase1/attention_unet_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/attention_unet_721_seed42/)
**Date:** November 15, 2026
**Final Status:** **`COMPLETED FULLY`** (All 15 Epochs Complete | Zero Errors | Best Checkpoint at Epoch 15: Mean Dice 0.8542)

---

## 1. Executive Summary & Core Results

The third Stage C Phase 1 architecture experiment—**Attention U-Net 2D (Spatial Gated U-Net)**—has successfully completed its full 15-epoch training protocol on the authorized 721-case development cohort.

- **Architecture:** Attention U-Net 2D (`AttentionUNet2D`, standard 4-level U-Net backbone augmented with additive spatial attention gates on skip connections).
- **Total Trainable Parameters:** **`7,852,160`** (~7.85 M), exactly matching Table 4.1 & 4.3 of [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md#L135).
- **Training Cohort:** 721 cases / 667 unique subjects (59,020 slices: 47,228 foreground + 11,792 background).
- **Validation Pool:** 155 cases / 143 unique subjects (12,364 slices).
- **Execution Mode:** Synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`), running continuously for 10.70 hours with **zero errors, zero CUDA exceptions, and zero driver events**.
- **Best Model Checkpoint:** **Epoch 15** (`best_model.pt`)
- **Best Selection-Exposed Validation Mean Dice:** **`0.8542`** (0.854212)
- **Best Selection-Exposed Validation Mean IoU:** **`0.7482`** (0.748189)
- **Sub-Compartment Dice Performance (Epoch 15):**
  - **Enhancing Tumor (ET / Class 3):** **`0.9035`** (IoU: 0.8240) — Attention U-Net recorded a marginally higher ET Dice than standard U-Net (0.9026) in this validation comparison.
  - **Peritumoral Edema (ED / Class 2):** **`0.8645`** (IoU: 0.7613)
  - **Necrotic Tumor Core (NCR / Class 1):** **`0.7947`** (IoU: 0.6593)
  - **Background (BG / Class 0):** **`0.9986`** (IoU: 0.9973)
- **Final Model Checkpoint:** **Epoch 15** (`final_model.pt`)
- **Final Training Loss:** **`0.1843`** (0.184298)
- **Total Experiment Runtime:** **10.70 hours (641.70 minutes)**.

---

## 2. Protocol & Architecture Verification

In accordance with Section 3 and Section 4 of [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md) and Decisions `DEC-C-01`, `DEC-C-03`, and `DEC-C-04` in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv):

| Hyperparameter / Dimension | Locked Value | Verification Finding |
| :--- | :--- | :---: |
| **Model Architecture** | `attention_unet` (`AttentionUNet2D`) | Verified (`src/models/attention_unet.py`, `create_model("attention_unet")`) |
| **Trainable Parameters** | 7,852,160 parameters (~7.85 M) | Verified exact match against Table 4.1 & 4.3 |
| **Input Modalities** | 4 channels: `t1n, t1c, t2f, t2w` | Verified (`BraTSGLISliceDataset`) |
| **Spatial Grid** | $240 \times 240$ axial slices | Verified |
| **Output Classes** | 4 (0: BG, 1: NCR, 2: ED, 3: ET) | Verified |
| **Attention Mechanism** | Additive spatial attention gating ($W_g, W_x, \psi$) | Verified in decoder skip junctions |
| **Weight Initialization** | Kaiming Normal (`bias=0`) | Verified scratch initialization |
| **Pretrained Weights** | None (Scratch) | Verified |
| **Training Cohort** | 721 cases / 667 unique subjects | Verified (`development_train_manifest.json`) |
| **Validation Pool** | 155 cases / 143 unique subjects | Verified (`validation_manifest.json`) |
| **Cohort Disjointness** | Case & Subject $\text{Train} \cap \text{Val} = \emptyset$ | Verified (`load_and_verify_cohort_manifests`) |
| **Epoch Budget** | 15 Epochs | Verified 15 full epochs |
| **Batch Sampler** | `CaseSliceBatchSampler` | Verified within-case batching |
| **Physical Batch Size** | 2 slices | Verified |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$) | Verified |
| **Base Learning Rate** | $1 \times 10^{-3}$ ($0.001000$) | Verified |
| **LR Schedule** | 2-epoch Linear Warmup $\rightarrow$ 13-epoch Cosine Decay | Verified step schedule |
| **Configured Cosine Floor** | $\eta_{\min} = 1.0 \times 10^{-5}$ ($0.000010$) | Verified. Note: `history.csv` records the learning rate active *during* training updates ($2.438 \times 10^{-5}$ during Epoch 15); the optimizer parameter group reached the configured floor ($1.0 \times 10^{-5}$) upon the completion of Epoch 15 via `scheduler.step()`, confirmed in `final_model.pt`. |
| **Gradient Clipping** | Maximum L2 norm = 1.0 | Verified |
| **Loss Function** | Multiclass Dice + Cross-Entropy ($\lambda_D=1.0, \lambda_{CE}=1.0$) | Verified (Dice excludes BG) |
| **Background Slice Ratio** | 0.25 (25% background slices) | Verified |
| **Random Seed** | 42 | Verified |
| **Execution Mode** | Synchronous CUDA (`CUDA_LAUNCH_BLOCKING=1`) | Operational safeguard applied |
| **Target Directory** | `results/stage_c/phase1/attention_unet_721_seed42/` | Verified fresh directory |

---

## 3. Preservation of Prior Baseline Experiments

The completed U-Net and U-Net++ experiments in their respective directories were strictly isolated and preserved untouched. Cryptographic SHA-256 hashes confirm zero modifications:

### 3.1 U-Net Baseline Preservation ([`results/stage_c/phase1/unet_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unet_721_seed42/))

| U-Net Artifact File | Size (Bytes) | Verified SHA-256 Hash | Integrity Status |
| :--- | :---: | :--- | :---: |
| `best_model.pt` | 93,281,991 | `37D35BF578868356CAADC7F4A173AF7B82535BB427FDB6CCE8DF121A9BF06CEF` | **UNTOUCHED / INTACT** |
| `final_model.pt` | 93,282,307 | `2C92E32970FF73995BC17078483E7F44CEF038CEC8367C96FF6A372907FD7915` | **UNTOUCHED / INTACT** |
| `history.csv` | 5,852 | `A8624EC117A9E317274FC1B9C2749B0EC05BF115237741B3807504B58AAF94CA` | **UNTOUCHED / INTACT** |
| `summary.json` | 1,253 | `E05BF4E5D4BB878840C3D948B494A8B4B66D6660D49E1BB3F2BCF8F14E475ED9` | **UNTOUCHED / INTACT** |
| `config.json` | 343 | `D25AFCB85F04722815F4093877770DBD48D21DEF5E4D75037767799CC941AE53` | **UNTOUCHED / INTACT** |

### 3.2 U-Net++ Baseline Preservation ([`results/stage_c/phase1/unetpp_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/))

| U-Net++ Artifact File | Size (Bytes) | Verified SHA-256 Hash | Integrity Status |
| :--- | :---: | :--- | :---: |
| `best_model.pt` | 110,090,779 | `FBD237A42BDF12F9C945A1C8CAA953DFB44A1672D3514CC842368FA535F4BFE6` | **UNTOUCHED / INTACT** |
| `final_model.pt` | 110,091,243 | `D3C3E16961D4DF6C3C6C5EF5434BBE310A637C5CD063BAF56950F88DB9A17EA0` | **UNTOUCHED / INTACT** |
| `history.csv` | 5,781 | `95775BF441C1792684A6E76E8C8196C9F4E46298531C19B42418B59B0B734AE5` | **UNTOUCHED / INTACT** |
| `summary.json` | 1,257 | `FFD9752E837C6026175BFD385F5734E96AE9CE3FD1E8EBAC081AB71CC50E1722` | **UNTOUCHED / INTACT** |
| `config.json` | 345 | `66DA99449907F1CC50DA077C1A323E0C44D02D9BB9CB695249EB089D2C1EC059` | **UNTOUCHED / INTACT** |

---

## 4. Complete 15-Epoch Training Trajectory (`history.csv`)

The complete 15-epoch execution history for Attention U-Net is recorded below. All 15 rows exist exactly once without gaps:

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Clip Norm | % Clipped | Learning Rate | Epoch Time | Cumulative Duration | Best Checkpoint? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4717 | 0.6177 | 0.7878 | 0.6547 | 0.8583 | 0.8051 | 0.7000 | 0.9978 | 1.80 | 55.3% | 0.000100 | 45.5 m | 0.76 h | YES |
| **2** | 0.2763 | 0.4193 | 0.7438 | 0.5948 | 0.7640 | 0.7946 | 0.6729 | 0.9977 | 1.08 | 21.6% | 0.000550 | 45.2 m | 1.51 h | NO |
| **3** | 0.2449 | 0.5060 | 0.8061 | 0.6771 | 0.8524 | 0.8105 | 0.7554 | 0.9979 | 0.52 | 7.9% | 0.001000 | 43.3 m | 2.23 h | YES |
| **4** | 0.2285 | 0.3246 | 0.8209 | 0.6993 | 0.8754 | 0.8330 | 0.7544 | 0.9981 | 0.36 | 4.4% | 0.000986 | 42.2 m | 2.94 h | YES |
| **5** | 0.2193 | 0.4774 | 0.8449 | 0.7333 | 0.8874 | 0.8515 | 0.7959 | 0.9983 | 0.34 | 3.6% | 0.000943 | 44.1 m | 3.67 h | YES |
| **6** | 0.2112 | 0.5234 | 0.8325 | 0.7156 | 0.8825 | 0.8432 | 0.7717 | 0.9983 | 0.32 | 2.9% | 0.000876 | 43.5 m | 4.40 h | NO |
| **7** | 0.2065 | 0.4747 | 0.8454 | 0.7340 | 0.8960 | 0.8324 | 0.8076 | 0.9983 | 0.29 | 2.7% | 0.000786 | 44.0 m | 5.13 h | YES |
| **8** | 0.2045 | 0.5085 | 0.8195 | 0.6960 | 0.8450 | 0.8505 | 0.7629 | 0.9985 | 0.27 | 2.5% | 0.000681 | 43.0 m | 5.85 h | NO |
| **9** | 0.1983 | 0.4615 | 0.8425 | 0.7298 | 0.8926 | 0.8340 | 0.8010 | 0.9983 | 0.27 | 2.2% | 0.000565 | 42.8 m | 6.56 h | NO |
| **10** | 0.1951 | 0.5130 | 0.8502 | 0.7417 | 0.8950 | 0.8599 | 0.7958 | 0.9985 | 0.26 | 2.3% | 0.000445 | 41.7 m | 7.26 h | YES |
| **11** | 0.1922 | 0.4895 | 0.8442 | 0.7338 | 0.8964 | 0.8632 | 0.7729 | 0.9986 | 0.30 | 2.3% | 0.000329 | 41.4 m | 7.95 h | NO |
| **12** | 0.1888 | 0.5482 | 0.8473 | 0.7367 | 0.8873 | 0.8536 | 0.8010 | 0.9985 | 0.29 | 2.4% | 0.000224 | 41.3 m | 8.63 h | NO |
| **13** | 0.1860 | 0.5224 | 0.8376 | 0.7244 | 0.8951 | 0.8555 | 0.7622 | 0.9985 | 0.28 | 2.3% | 0.000134 | 41.2 m | 9.32 h | NO |
| **14** | 0.1860 | 0.4912 | 0.8443 | 0.7338 | 0.9016 | 0.8521 | 0.7793 | 0.9985 | 0.30 | 2.4% | 0.000067 | 41.3 m | 10.01 h | NO |
| **15** | **0.1843** | **0.5013** | **0.8542** | **0.7482** | **0.9035** | **0.8645** | **0.7947** | **0.9986** | **0.31** | **2.4%** | **0.000024** | 41.1 m | **10.70 h** | **YES [* BEST]** |

---

## 5. Checkpoint Verification, Reload Compatibility & SHA-256 Hashes

All artifact files in [`results/stage_c/phase1/attention_unet_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/attention_unet_721_seed42/) were verified and cryptographically hashed:

| Artifact File | Size (Bytes) | SHA-256 Hash | Checkpoint State / Reload Status |
| :--- | :---: | :--- | :---: |
| **`best_model.pt`** | 94,417,899 | `5F7D8DAE1FB68E86905D2C36F56446DE8C0A09BABE0CE550230B157AA11E591E` | Selected **Epoch 15** \| Mean Dice = **0.8542** \| **PASS** (`<All keys matched successfully>`) |
| **`final_model.pt`** | 94,418,443 | `A6BF06366CC11D6E2E4DD269C65B34FF87A5C5AACF76924A431A7132CF983BD2` | Completed **Epoch 15** \| Mean Dice = **0.8542** \| **PASS** (`<All keys matched successfully>`) |
| **`history.csv`** | 5,826 | `E7269B9D75957DAC72086F9939B5A8D25FA983D5FB9D8A7C219E8714419E91CE` | Exactly 15 completed epoch rows |
| **`summary.json`** | 1,281 | `2E322A0DA2778C1E082E98F596D38FB90132AA4E82DB17EF0793728232C82D86` | Valid JSON \| Matches `history.csv` and checkpoints |
| **`config.json`** | 353 | `6C85A3CE46F5DF60B9DAF98B2092CE4E9638ACA60E810A1C4ACE3773498279EB` | Unmodified locked parameter configuration |

---

## 6. Hardware Telemetry & Runtime Footprint

- **Workstation:** Acer Predator Helios 16
  - **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6, Ada Lovelace)
  - **CPU:** 13th Gen Intel Core i9-13900HX (24 cores / 32 threads)
  - **System RAM:** 16 GB DDR5 RAM
- **Execution Mode:** Synchronous CUDA (`$env:CUDA_LAUNCH_BLOCKING = "1"`).
- **Process Memory Footprint:** Peak process VRAM reached **489.0 MB** (~0.49 GB), maintaining a remarkably compact memory profile close to standard U-Net (446.3 MB) and well below U-Net++ (1,111.3 MB).
- **GPU Thermal State:** GPU temperature remained controlled between 55 °C and 74 °C under active fan cooling, settling cleanly at 41.0 °C upon completion.
- **Windows Driver Stability:**
  A targeted query of the Windows System Event Log across the entire 10.70-hour execution window confirmed:
  ```text
  nvlddmkm events during Attention U-Net run: 0 (ZERO EVENTS!)
  ```
  Zero CUDA errors, zero illegal instructions, and zero display driver recovery events occurred.

---

## 7. Comparative Tri-Model Benchmark: U-Net vs. U-Net++ vs. Attention U-Net

All three Phase 1 models were trained from scratch on the exact same 721-case development cohort for 15 epochs under the identical Stage B optimization protocol:

| Evaluation Dimension | Standard U-Net 2D (`unet`) | U-Net++ 2D (`unetpp`) | Attention U-Net 2D (`attention_unet`) | Primary Observation |
| :--- | :---: | :---: | :---: | :--- |
| **Architecture Family** | Standard Plain U-Net | Nested Dense Skips | Spatial Attention Gating | Additive gating at skip junctions |
| **Parameter Count** | 7,763,428 (~7.76 M) | 9,160,068 (~9.16 M) | 7,852,160 (~7.85 M) | Only +88,732 params (+1.14%) over U-Net |
| **Peak VRAM Allocation** | 446.3 MB | 1,111.3 MB | 489.0 MB | Compact footprint (only +42.7 MB over U-Net) |
| **Total Cumulative Runtime** | 8.24 hours *(Heterogeneous)* | 15.44 hours *(Homogeneous)* | 10.70 hours *(Homogeneous)* | Not a controlled speed benchmark |
| **Final Training Loss** | 0.1815 | **0.1728** | 0.1843 | U-Net++ lowest (0.1728), U-Net 0.1815, Attn U-Net 0.1843 |
| **Best Selected Epoch** | Epoch 15 | Epoch 15 | Epoch 15 | All models peaked at Epoch 15 |
| **Selection-Exposed Val Mean Dice** | 0.8594 | **0.8602** | 0.8542 | U-Net++ (0.8602), U-Net (0.8594), Attn U-Net (0.8542) |
| **Selection-Exposed Val Mean IoU** | 0.7554 | **0.7568** | 0.7482 | Consistent with Mean Dice ranking |
| **Enhancing Tumor (ET / C3) Dice** | 0.9026 | **0.9047** | **0.9035** | Marginally higher ET Dice than standard U-Net (+0.0009) |
| **Peritumoral Edema (ED / C2) Dice** | 0.8662 | **0.8664** | 0.8645 | High parity across all architectures |
| **Necrotic Core (NCR / C1) Dice** | 0.8094 | **0.8096** | 0.7947 | Attention U-Net recorded lower NCR Dice (0.7947 vs 0.8094/0.8096) |
| **Background (BG / C0) Dice** | 0.9986 | 0.9986 | 0.9986 | Identical background suppression |

### Comparative Interpretation:
1. **Validation Performance & Architectural Differences:**
   - On selection-exposed validation Mean Dice, U-Net++ recorded 0.8602, standard U-Net recorded 0.8594, and Attention U-Net recorded 0.8542. The absolute difference between U-Net++ and U-Net is +0.0008 (0.08 percentage points); this marginal numerical difference does not establish meaningful architectural superiority, improved spatial refinement, or better generalization.
   - On Enhancing Tumor (ET / C3), Attention U-Net recorded a marginally higher ET Dice (0.9035) than standard U-Net (0.9026, absolute difference +0.0009 or 0.09 percentage points), while U-Net++ recorded 0.9047. In accordance with strict scientific reporting standards, this numerical difference does not demonstrate improved tumor sensitivity or prove that attention gates focus better on tumor regions; causality cannot be inferred from this validation comparison.
   - On Necrotic Core (NCR / C1), Attention U-Net recorded 0.7947, compared to 0.8094 for standard U-Net and 0.8096 for U-Net++. This numerical difference is reported descriptively without attributing it to a proven architectural mechanism.
2. **Resource Metrics Interpretation:**
   - Attention U-Net utilizes 7,852,160 parameters (~1.14% more than U-Net's 7,763,428 parameters).
   - Its recorded peak process VRAM was 489.0 MB (~9.6% higher than U-Net's 446.3 MB, and lower than U-Net++'s 1,111.3 MB under these experiment conditions).
   - Its measured runtime was 10.70 hours under full synchronous execution (compared to 15.44 hours for U-Net++ under synchronous execution, and 8.24 hours for U-Net across mixed asynchronous/synchronous execution). These three total runtimes cannot be presented as a controlled architecture-speed benchmark.
   - These measured training metrics do not constitute proof of superior deployment efficiency, energy efficiency, inference speed, or clinical suitability, as those properties have not been established by these training measurements.
3. **Epoch Budget & Convergence:**
   - All three models achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.

---

## 8. Test Quarantine & Git Governance Adherence

1. **Test Quarantine Boundary:**
   - `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` were **NEVER ACCESSED, OPENED, READ, ENUMERATED, OR HASHED**.
   - Zero test cases were exposed.
2. **Git Working Tree State:**
   - Pre-existing tracked modifications in `src/data/dataset.py`, `src/evaluation/metrics.py`, and `src/training/train.py` remain preserved and untouched.
   - Zero Git commits, resets, branch switches, or clean operations were performed.
   - All 14 regression tests in [`tests/test_train_scaled_model_factory.py`](file:///d:/MajorProject/tests/test_train_scaled_model_factory.py) pass cleanly in 1.01s.

---

## 9. Scientific Limitations & Selection-Exposure Disclosure

> [!IMPORTANT]
> **Methodological Disclosure:**
> 1. **Selection-Exposed Internal Validation:** All validation metrics reported in this document (including the best Validation Mean Dice of **`0.8542`**) were computed on the 155-case validation pool (`Val_eval` + `Val_cal`). Because model checkpoints were continuously monitored and selected on this pool, these metrics are formally classified as **selection-exposed internal validation metrics**.
> 2. **Epoch 15 Checkpoint Selection & Non-Convergence:** All three models achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.
> 3. **Single-Seed Statistical Interpretation & Lack of Generalization:** Because each architecture was evaluated using a single configured random seed (seed 42), run-to-run variability and statistical significance cannot be estimated. Small numerical differences should therefore be interpreted descriptively rather than as robust performance differences. In accordance with methodological standards established in [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md), these figures do **NOT** represent independent generalization performance, unbiased test set accuracy, or clinical readiness. Measured training VRAM and parameters do not establish inference speed, energy efficiency, or deployment suitability. No statistical significance testing or multi-seed robustness analysis has been performed, and architectural superiority has not been demonstrated. Independent calibration fitting on `Val_cal` ($T^*_{\text{attn\_unet}}$) and blind evaluation on the quarantined test set will be conducted separately under subsequent Stage C Work Packages.

---

## 10. Final Decision Gate

# **`EXPERIMENT FULLY COMPLETED — ALL 15 EPOCHS VERIFIED`**

> **Final Experiment Summary:**
> The Stage C Phase 1 Attention U-Net 2D training experiment on the authorized 721-case development cohort has successfully completed all 15 epochs. Checkpoint selection identified **Epoch 15** as the optimal checkpoint with a selection-exposed Validation Mean Dice of **`0.8542`** (ET: 0.9035, ED: 0.8645, NCR: 0.7947). Both `best_model.pt` and `final_model.pt` are verified, reloadable, and hashed. All prior U-Net and U-Net++ artifacts remain 100% intact with matching SHA-256 signatures, all 14 regression unit tests pass cleanly, and the sealed test quarantine boundary remains strictly unviolated.
