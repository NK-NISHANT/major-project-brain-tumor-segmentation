# G1-P4.3 — Stage C Phase 1 U-Net++ Training Report (15 Epochs)

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md)
**Stage C Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)
**Decision Register Reference:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Previous Baseline Reference:** [`G1_P4_UNET_STAGE_C_TRAINING_REPORT.md`](file:///d:/MajorProject/G1_P4_UNET_STAGE_C_TRAINING_REPORT.md)
**Experiment Output Directory:** [`results/stage_c/phase1/unetpp_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/)
**Date:** November 14, 2026
**Final Status:** **`COMPLETED FULLY`** (All 15 Epochs Complete | Zero Errors | Best Checkpoint at Epoch 15: Mean Dice 0.8602)

---

## 1. Executive Summary & Core Results

The second Stage C Phase 1 architecture experiment—**U-Net++ (Nested Dense Skip U-Net)**—has successfully completed its full 15-epoch training protocol on the authorized 721-case development cohort.

- **Architecture:** U-Net++ 2D (`UNetPlusPlus2D`, standard 4-level nested dense skip pathways, deep supervision disabled).
- **Total Trainable Parameters:** **`9,160,068`** (~9.16 M), exactly matching Table 4.1 & 4.3 of [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md#L135).
- **Training Cohort:** 721 cases / 667 unique subjects (59,020 slices: 47,228 foreground + 11,792 background).
- **Validation Pool:** 155 cases / 143 unique subjects (12,364 slices).
- **Execution Mode:** Synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`), which ran continuously for 15.44 hours with **zero errors, zero CUDA exceptions, and zero driver events**.
- **Best Model Checkpoint:** **Epoch 15** (`best_model.pt`)
- **Best Selection-Exposed Validation Mean Dice:** **`0.8602`** (0.860218), exceeding the standard U-Net 2D benchmark of **0.8594** by **+0.0008**.
- **Best Selection-Exposed Validation Mean IoU:** **`0.7568`** (0.756770)
- **Sub-Compartment Dice Performance (Epoch 15):**
  - **Enhancing Tumor (ET / Class 3):** **`0.9047`** (IoU: 0.8260)
  - **Peritumoral Edema (ED / Class 2):** **`0.8664`** (IoU: 0.7642)
  - **Necrotic Tumor Core (NCR / Class 1):** **`0.8096`** (IoU: 0.6801)
  - **Background (BG / Class 0):** **`0.9986`** (IoU: 0.9972)
- **Final Model Checkpoint:** **Epoch 15** (`final_model.pt`)
- **Final Training Loss:** **`0.1728`** (0.172768)
- **Total Experiment Runtime:** **15.44 hours (926.39 minutes)**.

---

## 2. Protocol & Architecture Verification

In accordance with Section 3 and Section 4 of [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md) and Decisions `DEC-C-01`, `DEC-C-03`, and `DEC-C-04` in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv):

| Hyperparameter / Dimension | Locked Value | Verification Finding |
| :--- | :--- | :---: |
| **Model Architecture** | `unetpp` (`UNetPlusPlus2D`) | Verified (`src/models/unet.py`, `create_model("unetpp")`) |
| **Trainable Parameters** | 9,160,068 parameters (~9.16 M) | Verified exact match against Table 4.1 & 4.3 |
| **Input Modalities** | 4 channels: `t1n, t1c, t2f, t2w` | Verified (`BraTSGLISliceDataset`) |
| **Spatial Grid** | $240 \times 240$ axial slices | Verified |
| **Output Classes** | 4 (0: BG, 1: NCR, 2: ED, 3: ET) | Verified |
| **Deep Supervision** | Disabled (`False`) | Verified single output head at $X^{0,4}$ |
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
| **Target Directory** | `results/stage_c/phase1/unetpp_721_seed42/` | Verified fresh directory |

---

## 3. Preservation of Prior U-Net Baseline Experiment

The completed U-Net experiment in [`results/stage_c/phase1/unet_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unet_721_seed42/) was strictly isolated and preserved untouched. Cryptographic SHA-256 hashes confirm zero modifications:

| U-Net Artifact File | Size (Bytes) | Verified SHA-256 Hash | Integrity Status |
| :--- | :---: | :--- | :---: |
| `best_model.pt` | 93,281,991 | `37D35BF578868356CAADC7F4A173AF7B82535BB427FDB6CCE8DF121A9BF06CEF` | **UNTOUCHED / INTACT** |
| `final_model.pt` | 93,282,307 | `2C92E32970FF73995BC17078483E7F44CEF038CEC8367C96FF6A372907FD7915` | **UNTOUCHED / INTACT** |
| `history.csv` | 5,852 | `A8624EC117A9E317274FC1B9C2749B0EC05BF115237741B3807504B58AAF94CA` | **UNTOUCHED / INTACT** |
| `summary.json` | 1,253 | `E05BF4E5D4BB878840C3D948B494A8B4B66D6660D49E1BB3F2BCF8F14E475ED9` | **UNTOUCHED / INTACT** |

---

## 4. Complete 15-Epoch Training Trajectory (`history.csv`)

The complete 15-epoch execution history for U-Net++ is recorded below. All 15 rows exist exactly once without gaps:

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Clip Norm | % Clipped | Learning Rate | Epoch Time | Cumulative Duration | Best Checkpoint? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4537 | 0.6158 | 0.7938 | 0.6618 | 0.8592 | 0.8030 | 0.7191 | 0.9978 | 1.69 | 52.0% | 0.000100 | 61.9 m | 1.03 h | YES |
| **2** | 0.2758 | 0.4585 | 0.7072 | 0.5499 | 0.7525 | 0.7423 | 0.6268 | 0.9976 | 0.86 | 17.8% | 0.000550 | 61.8 m | 2.06 h | NO |
| **3** | 0.2411 | 0.3498 | 0.7990 | 0.6678 | 0.8595 | 0.7931 | 0.7442 | 0.9977 | 0.39 | 5.5% | 0.001000 | 61.9 m | 3.09 h | YES |
| **4** | 0.2289 | 0.3117 | 0.7895 | 0.6617 | 0.8762 | 0.8319 | 0.6606 | 0.9983 | 0.29 | 3.5% | 0.000986 | 61.9 m | 4.12 h | NO |
| **5** | 0.2151 | 0.3434 | 0.8488 | 0.7389 | 0.8886 | 0.8551 | 0.8027 | 0.9984 | 0.26 | 3.2% | 0.000943 | 61.9 m | 5.15 h | YES |
| **6** | 0.2074 | 0.2950 | 0.8297 | 0.7123 | 0.8786 | 0.8540 | 0.7565 | 0.9985 | 0.24 | 2.8% | 0.000876 | 62.0 m | 6.18 h | NO |
| **7** | 0.1996 | 0.3364 | 0.8470 | 0.7363 | 0.8971 | 0.8278 | 0.8162 | 0.9982 | 0.24 | 2.4% | 0.000786 | 61.9 m | 7.21 h | NO |
| **8** | 0.1966 | 0.3988 | 0.8246 | 0.7034 | 0.8520 | 0.8527 | 0.7691 | 0.9984 | 0.23 | 2.4% | 0.000681 | 60.8 m | 8.23 h | NO |
| **9** | 0.1893 | 0.3917 | 0.8410 | 0.7270 | 0.8828 | 0.8327 | 0.8076 | 0.9983 | 0.23 | 2.3% | 0.000565 | 60.0 m | 9.23 h | NO |
| **10** | 0.1867 | 0.4557 | 0.8524 | 0.7452 | 0.8973 | 0.8647 | 0.7953 | 0.9985 | 0.23 | 2.2% | 0.000445 | 59.8 m | 10.22 h | YES |
| **11** | 0.1816 | 0.4158 | 0.8482 | 0.7398 | 0.9001 | 0.8659 | 0.7787 | 0.9986 | 0.22 | 2.2% | 0.000329 | 61.5 m | 11.25 h | NO |
| **12** | 0.1791 | 0.4864 | 0.8376 | 0.7233 | 0.8880 | 0.8491 | 0.7757 | 0.9986 | 0.25 | 2.2% | 0.000224 | 63.9 m | 12.32 h | NO |
| **13** | 0.1766 | 0.5210 | 0.8477 | 0.7388 | 0.9014 | 0.8596 | 0.7821 | 0.9986 | 0.25 | 2.2% | 0.000134 | 62.7 m | 13.36 h | NO |
| **14** | 0.1746 | 0.4382 | 0.8450 | 0.7346 | 0.9014 | 0.8490 | 0.7847 | 0.9984 | 0.26 | 2.2% | 0.000067 | 62.1 m | 14.40 h | NO |
| **15** | **0.1728** | **0.4280** | **0.8602** | **0.7568** | **0.9047** | **0.8664** | **0.8096** | **0.9986** | **0.24** | **2.1%** | **0.000024** | 62.4 m | **15.44 h** | **YES [* BEST]** |

---

## 5. Checkpoint Verification, Reload Compatibility & SHA-256 Hashes

All artifact files in [`results/stage_c/phase1/unetpp_721_seed42/`](file:///d:/MajorProject/results/stage_c/phase1/unetpp_721_seed42/) were verified and cryptographically hashed:

| Artifact File | Size (Bytes) | SHA-256 Hash | Checkpoint State / Reload Status |
| :--- | :---: | :--- | :---: |
| **`best_model.pt`** | 110,090,779 | `FBD237A42BDF12F9C945A1C8CAA953DFB44A1672D3514CC842368FA535F4BFE6` | Selected **Epoch 15** | Mean Dice = **0.8602** | **PASS** (`<All keys matched successfully>`) |
| **`final_model.pt`** | 110,091,243 | `D3C3E16961D4DF6C3C6C5EF5434BBE310A637C5CD063BAF56950F88DB9A17EA0` | Completed **Epoch 15** | Mean Dice = **0.8602** | **PASS** (`<All keys matched successfully>`) |
| **`history.csv`** | 5,781 | `95775BF441C1792684A6E76E8C8196C9F4E46298531C19B42418B59B0B734AE5` | Exactly 15 completed epoch rows |
| **`summary.json`** | 1,257 | `FFD9752E837C6026175BFD385F5734E96AE9CE3FD1E8EBAC081AB71CC50E1722` | Valid JSON | Matches `history.csv` and checkpoints |
| **`config.json`** | 345 | `66DA99449907F1CC50DA077C1A323E0C44D02D9BB9CB695249EB089D2C1EC059` | Unmodified locked parameter configuration |

---

## 6. Hardware Telemetry & Runtime Footprint

- **Workstation:** Acer Predator Helios 16
  - **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6, Ada Lovelace)
  - **CPU:** 13th Gen Intel Core i9-13900HX (24 cores / 32 threads)
  - **System RAM:** 16 GB DDR5 RAM
- **Execution Mode:** Synchronous CUDA (`$env:CUDA_LAUNCH_BLOCKING = "1"`).
- **Process Memory Footprint:** Peak process VRAM reached **1,111.3 MB** (~1.11 GB), well within the 8,188 MiB total capacity.
- **GPU Thermal State:** GPU temperature remained controlled between 62 °C and 78 °C under active fan cooling, dropping cleanly to 52.4 °C at completion.
- **Windows Driver Stability:**
  A targeted query of the Windows System Event Log across the entire 15.44-hour execution window confirmed:
  ```text
  nvlddmkm events during U-Net++ run: 0 (ZERO EVENTS!)
  ```
  Zero CUDA errors, zero illegal instructions, and zero display driver recovery events occurred.

---

## 7. Comparative Benchmark: U-Net vs. U-Net++ (Stage C Phase 1)

Both models were trained from scratch on the exact same 721-case development cohort for 15 epochs under the locked Stage B optimization schedule:

| Evaluation Dimension | Standard U-Net 2D (`unet`) | U-Net++ 2D (`unetpp`) | Delta ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **Architecture Family** | Standard 4-level U-Net | Nested Dense Skips (UNet++) | Dense feature aggregation |
| **Parameter Count** | 7,763,428 (~7.76 M) | 9,160,068 (~9.16 M) | +1,396,640 (+18.0%) |
| **Peak VRAM Allocation** | 446.3 MB | 1,111.3 MB | +665.0 MB (+149.0%) |
| **Total Cumulative Runtime** | 8.24 hours *(Heterogeneous)* | 15.44 hours *(Homogeneous)* | Not a controlled speed benchmark |
| **Final Training Loss** | 0.1815 | **0.1728** | **-0.0087** |
| **Best Selected Epoch** | Epoch 15 | Epoch 15 | Both models peaked at Epoch 15 |
| **Selection-Exposed Val Mean Dice** | 0.8594 | **0.8602** | **+0.0008 (+0.08% Dice)** |
| **Selection-Exposed Val Mean IoU** | 0.7554 | **0.7568** | **+0.0014 (+0.14% IoU)** |
| **Enhancing Tumor (ET / C3) Dice** | 0.9026 | **0.9047** | **+0.0021** |
| **Peritumoral Edema (ED / C2) Dice** | 0.8662 | **0.8664** | **+0.0002** |
| **Necrotic Core (NCR / C1) Dice** | 0.8094 | **0.8096** | **+0.0002** |
| **Background (BG / C0) Dice** | 0.9986 | 0.9986 | 0.0000 |

*Comparative Interpretation:*
- **Validation Dice Comparison:** U-Net++ recorded a marginally higher selection-exposed validation Mean Dice than standard U-Net (0.8602 vs 0.8594, absolute difference +0.0008 or 0.08 percentage points). This numerical difference does not establish meaningful architectural superiority. The experiment does not confirm improved spatial refinement or better generalization.
- **Resource Footprint & Runtime Comparability:** U-Net++ recorded a lower final training loss (0.1728 vs 0.1815), while requiring a larger memory footprint (1,111.3 MB vs 446.3 MB peak process VRAM). Total recorded runtime was 15.44 hours for U-Net++ (fully synchronous across all 15 epochs) versus 8.24 hours for U-Net (split between asynchronous and synchronous modes, preventing direct controlled speed comparisons).
- **Epoch Budget & Convergence:** Both models achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.

---

## 8. Test Quarantine & Git Governance Adherence

1. **Test Quarantine Boundary:**
   - `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` were **NEVER ACCESSED, OPENED, READ, ENUMERATED, OR HASHED**.
   - Zero test cases were exposed.
2. **Git Working Tree State:**
   - Pre-existing tracked modifications in `src/data/dataset.py`, `src/evaluation/metrics.py`, and `src/training/train.py` remain preserved and untouched (383 insertions, 31 deletions).
   - Zero Git commits, resets, branch switches, or clean operations were performed.
   - All 14 regression tests in [`tests/test_train_scaled_model_factory.py`](file:///d:/MajorProject/tests/test_train_scaled_model_factory.py) pass cleanly in 1.00s.

---

## 9. Scientific Limitations & Selection-Exposure Disclosure

> [!IMPORTANT]
> **Methodological Disclosure:**
> 1. **Selection-Exposed Internal Validation:** All validation metrics reported in this document (including the best Validation Mean Dice of **`0.8602`**) were computed on the 155-case validation pool (`Val_eval` + `Val_cal`). Because model checkpoints were continuously monitored and selected on this pool, these metrics are formally classified as **selection-exposed internal validation metrics**.
> 2. **Epoch 15 Checkpoint Selection & Non-Convergence:** Both models achieved their highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.
> 3. **Single-Seed Statistical Interpretation & Lack of Generalization:** Because each architecture was evaluated using a single configured random seed (seed 42), run-to-run variability and statistical significance cannot be estimated. Small numerical differences should therefore be interpreted descriptively rather than as robust performance differences. In accordance with methodological standards established in [`MASTER_PROJECT_BLUEPRINT_v2.0.md`](file:///d:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md), these figures do **NOT** represent independent generalization performance, unbiased test set accuracy, or clinical readiness. No statistical significance testing or multi-seed robustness analysis has been performed, and architectural superiority has not been demonstrated. Independent calibration fitting on `Val_cal` ($T^*_{\text{unetpp}}$) will be conducted separately under Stage C Work Package C2.

---

## 10. Final Decision Gate

# **`EXPERIMENT FULLY COMPLETED — ALL 15 EPOCHS VERIFIED`**

> **Final Experiment Summary:**
> The Stage C Phase 1 U-Net++ 2D training experiment on the authorized 721-case development cohort has successfully completed all 15 epochs. Checkpoint selection identified **Epoch 15** as the optimal checkpoint with a selection-exposed Validation Mean Dice of **`0.8602`** (ET: 0.9047, ED: 0.8664, NCR: 0.8096). Both `best_model.pt` and `final_model.pt` are verified, reloadable, and hashed. All 14 test suite regression checks pass cleanly, and the sealed test quarantine boundary remains 100% intact.
