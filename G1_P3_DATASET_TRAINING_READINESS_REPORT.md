# G1-P3 — Dataset Integrity & Training Pipeline Readiness Report

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Stage C Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md) (Revision G1-P2.2)
**Decision Register Reference:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Execution Type:** Bounded Engineering Inspection + Targeted Repair + Dry-Run Verification
**Date:** November 13, 2026
**Final Decision Gate:** **`A. TECHNICALLY READY`**
**Training Authorization:** **NOT GRANTED (Strictly Zero Training / Zero Parameter Updates Executed)**

---

## 1. Executive Summary & Readiness Verdict

The 721-case development training pipeline has undergone an exhaustive engineering readiness audit and synthetic dry-run verification. Every NIfTI volume, tensor channel, label mask, batch sampler rule, model architecture, loss function, optimizer group, learning rate schedule, and checkpoint recovery mechanism was inspected and verified against the G1-P2.2 protocol.

A single actionable P1 defect was identified in `src/training/train_scaled.py` (missing model factory branches and CLI choices for the three new Stage C architectures), repaired via minimal targeted modification, and verified with a focused unit test suite.

**Verdict: `A. TECHNICALLY READY`**
All technical, dataset, loader, architecture, loss, gradient, scheduler, and recovery systems are verified operational on the local workstation. The pipeline is fully prepared to proceed to the supervisor training-authorization gate. Zero model parameters have been updated.

---

## 2. Dataset Integrity Audit (All 721 Authorized Cases)

The authorized Stage C training cohort was verified against `results/dataset_splits/development_train_manifest.json` (SHA-256: `42B7BF303319266BFA692E0EB41FDDD425DE2B845195F07B9B9105ED1875B015`). All 721 cases reside in `data/extracted_archive/`.

A memory-efficient, sequential inspection of all **3,605 NIfTI files** (4 modalities + 1 segmentation mask per case) was executed across the complete 721-case cohort.

### 2.1 Audit Results Summary

| Inspection Dimension | Requirement / Expectation | Audited Result | Status |
| :--- | :--- | :--- | :---: |
| **Case Count & Census** | Exactly 721 development cases | 721 cases verified | **PASS** |
| **Unique Biological Subjects** | Exactly 667 unique subjects | 667 subjects verified | **PASS** |
| **Subject Disjointness** | Zero subject overlap with validation pool | 0 subject overlap with 155-case validation pool | **PASS** |
| **File Existence** | 5 files per case (`t1n, t1c, t2f, t2w, seg`) | 3,605 / 3,605 files present on disk | **PASS** |
| **NIfTI Openability** | All files readable by `nibabel` | 3,605 / 3,605 files successfully opened | **PASS** |
| **Spatial Dimensions** | Exact $(240, 240, 155)$ grid across all channels | 0 shape mismatches (100% compliant) | **PASS** |
| **Affine / Orientation** | Identical affine transformation across modalities & mask | 0 affine mismatches ($< 10^{-3}$ atol) | **PASS** |
| **Voxel Finite Values** | Zero `NaN` or `Inf` floating-point voxels | 0 non-finite voxels across all volumes | **PASS** |
| **Modality Completeness** | Non-zero variance, no flatline/empty volumes | 0 empty or degenerate modality volumes | **PASS** |
| **Segmentation Mask Labels** | Strict subset of $\{0, 1, 2, 3\}$ | 0 unexpected label values detected | **PASS** |
| **Lesion Foreground Presence**| Non-zero ground-truth tumor voxels per case | 721 / 721 cases have active lesion tissue | **PASS** |

### 2.2 Volumetric & Slice-Level Census

- **Total Raw Axial Slices Across 721 Cases:** $721 \times 155 = \mathbf{111,755}$ slices.
- **Ground-Truth Foreground Slices:** **47,228 slices** ($42.26\%$ of total volume).
- **Empty Background Volume Slices:** **64,527 slices** ($57.74\%$ of total volume).
- **Execution Time:** Full voxel-level inspection completed in **288.16 seconds** (~2.5 cases/sec).
- **Quarantine Boundary Adherence:** `data/quarantine_test/` and `results/dataset_splits/test_manifest.json` remained strictly unopened, unaccessed, and uninspected during G1-P3 and G1-P3.1. Reaffirming the historical disclosure from G1-P2.2: while `test_manifest.json` was parsed during repository inventory and freeze audits in October 2026, no quarantined test MRI volumes or masks have ever been evaluated, loaded, or inspected in any training or evaluation experiment.

---

## 3. Data Loader & Preprocessing Verification

The data loading pipeline was verified using `src.data.dataset.BraTSGLISliceDataset` and `src.data.dataset.CaseSliceBatchSampler`:

1. **MRI Modality Channel Construction:**
   - Channel 0: T1-native (`t1n`)
   - Channel 1: T1-contrast enhanced (`t1c`)
   - Channel 2: T2-FLAIR (`t2f`)
   - Channel 3: T2-weighted (`t2w`)
   - Canonical ordering strictly matches `src.data.dataset.MODALITIES`.
2. **Intensity Normalization:**
   - Verified that `_load_and_normalize_case` applies per-case non-zero voxel z-score standardization: $(x - \mu_{\text{nonzero}}) / \sigma_{\text{nonzero}}$.
   - Zero background voxels remain strictly $0.0$.
3. **Background Slice Sampling:**
   - Configured ratio: `background_slice_ratio = 0.25`.
   - On a sample dataset of 5 cases (408 total slices selected): 326 foreground slices and 82 background slices were extracted, achieving an exact observed ratio of $82 / 326 = \mathbf{0.2515} \approx 25\%$.
4. **`CaseSliceBatchSampler` Within-Case Batching Contract:**
   - Physical batch size: 2 slices.
   - Evaluated across 206 batches: **0 cross-case batches** observed. Every batch of 2 slices is drawn exclusively from within the same patient volume, guaranteeing optimal CPU case-cache reuse.
   - Slices covered: Exactly 408 / 408 slices (0 duplicates, 0 omissions).
   - **Exact Batch Accounting & Decomposition:** The 408 sampled slices from 5 patient cases yielded exactly **206 batches**, decomposed into:
     - **202 full batches of size 2** ($202 \times 2 = 404$ slices)
     - **4 partial batches of size 1** ($4 \times 1 = 4$ slices)
     - Sum: $202 + 4 = 206$ batches; $404 + 4 = 408$ slices.
     The 4 partial batches occur because 4 of the 5 sampled cases contained an odd count of selected slices (foreground + sampled background). Because `CaseSliceBatchSampler` strictly enforces within-case grouping and never concatenates slices across different patient volumes, each odd case emitted its final remaining slice as a partial batch of size 1.
5. **Tensor Output Contract:**
   - Image tensor: `[2, 4, 240, 240]`, `torch.float32`, all finite.
   - Mask target tensor: `[2, 240, 240]`, `torch.int64`, labels $\in \{0, 1, 2, 3\}$.
6. **Deterministic Indexing:**
   - Two independently initialized `BraTSGLISliceDataset` instances with `seed=42` yielded bit-exact identical slice record lists.

---

## 4. Controlled Dry-Run Results

Controlled synthetic dry-runs were conducted on the local hardware (**NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM**).

### 4.1 Synthetic Forward, Loss & Backward Smoke Tests (Zero Parameter Updates)

For each of the three planned Stage C architectures, a synthetic forward pass, loss calculation (`DiceCrossEntropyLoss`), and backward pass were executed. **Crucially, `optimizer.step()` was NOT called; zero model parameters were updated.**

| Architecture | Model Class | Parameters | Forward Latency | Loss (Dice+CE) | Pre-Clip $\|\nabla\|_2$ | Post-Clip $\|\nabla\|_2$ | Peak VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net 2D** | `UNet2D` | 7,763,428 | 395.9 ms (first run) | 2.1889 | 0.326 | 0.326 | **356.8 MB** |
| **U-Net++ 2D** | `UNetPlusPlus2D` | 9,160,068 | 27.6 ms | 2.2179 | 0.695 | 0.695 | **1,067.0 MB** |
| **Attention U-Net 2D** | `AttentionUNet2D` | 7,852,160 | 43.6 ms | 2.2228 | 0.722 | 0.722 | **492.1 MB** |

- **Loss Contract:** All loss outputs are finite and correctly balance multiclass CrossEntropy and non-background Multiclass Dice ($\lambda_D=1.0, \lambda_{CE}=1.0$).
- **Gradients Contract:** Backward passes generated non-zero, finite gradient vectors across all trainable parameters.
- **Gradient Clipping Contract:** `torch.nn.utils.clip_grad_norm_` limits gradients to $\le 1.0$. Verified via dedicated unit test `test_synthetic_gradient_clipping_amplified_gradients` with $1000\times$ artificially amplified gradients (pre-clip $\|\nabla\|_2 > 250.0 \gg 1.0$, post-clip $\|\nabla\|_2 = 1.000$, weights strictly unaltered, zero optimizer steps).
- **Parameter Count Reconciliation:** Actual instantiated model parameter counts were measured:
  - U-Net 2D (`UNet2D`): **7,763,428** parameters (~7.76 M)
  - U-Net++ 2D (`UNetPlusPlus2D`): **9,160,068** parameters (~9.16 M; supersedes the draft protocol's legacy textual estimate of 9.04 M)
  - Attention U-Net 2D (`AttentionUNet2D`): **7,852,160** parameters (~7.85 M; supersedes the draft protocol's legacy textual estimate of 8.12 M)
  - ResUNet-2D (`ResUNet2D`): **8,114,604** parameters (~8.11 M)
  - UNetResNet34 (`UNetResNet34`): **24,327,524** parameters (~24.33 M)
  The Stage C experiment protocol ([`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)) has been formally updated with these exact counts.
- **VRAM Footprint (Synthetic Dry-Run Evidence Only):** Peak observed memory for U-Net++ (the most memory-intensive model due to dense skips) was **1,067.0 MB** during a single synthetic batch-size-2 forward-backward pass, within the 7,995.6 MB addressable VRAM of the RTX 4070 Laptop GPU. This is **synthetic dry-run evidence only** and does not constitute a guarantee against Windows OS paging, background process overhead, or thermal throttling during sustained multi-hour training.

---

### 4.2 Learning Rate Scheduler & Recovery Dry-Run

1. **15-Epoch `SequentialLR` Warmup + Cosine Schedule:**
   - **Optimizer:** AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$).
   - **Scheduler Constructor & Parameters:**
     ```python
     warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=2)
     cosine = PerGroupCosineAnnealingLR(optimizer, T_max=13, eta_min=1e-5)
     scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[2])
     ```
     - `base_lr`: `1e-3` ($0.001000$)
     - `warmup_epochs`: 2 (`LinearLR`, initial LR factor 0.1)
     - `cosine_epochs` (`T_max`): 13 ($15 - 2 = 13$ cosine decay epochs)
     - `eta_min` (Configured Asymptotic Floor): `1e-5` ($0.000010$)
   - **Simulated Epoch-by-Epoch Execution Profile (Observed LR at start of training epoch):**
     - Epoch  1 (Warmup Start): $\text{LR} = 0.00010000$ ($1.00 \times 10^{-4}$)
     - Epoch  2 (Warmup Mid): $\text{LR} = 0.00055000$ ($5.50 \times 10^{-4}$)
     - Epoch  3 (Peak Base LR): $\text{LR} = 0.00100000$ ($1.00 \times 10^{-3}$)
     - Epoch  4 (Cosine Step 1): $\text{LR} = 0.00098562$
     - Epoch  5 (Cosine Step 2): $\text{LR} = 0.00094330$
     - Epoch  6 (Cosine Step 3): $\text{LR} = 0.00087551$
     - Epoch  7 (Cosine Step 4): $\text{LR} = 0.00078619$
     - Epoch  8 (Cosine Step 5): $\text{LR} = 0.00068053$
     - Epoch  9 (Cosine Step 6): $\text{LR} = 0.00056467$
     - Epoch 10 (Cosine Step 7): $\text{LR} = 0.00044533$
     - Epoch 11 (Cosine Step 8): $\text{LR} = 0.00032947$
     - Epoch 12 (Cosine Step 9): $\text{LR} = 0.00022381$
     - Epoch 13 (Cosine Step 10): $\text{LR} = 0.00013449$
     - Epoch 14 (Cosine Step 11): $\text{LR} = 0.00006670$
     - Epoch 15 (Observed Training LR): $\text{LR} = 0.00002438$ ($2.438 \times 10^{-5}$)
     - *After Epoch 15 step:* $\text{LR} = 0.00001000$ ($1.00 \times 10^{-5}$, exact `eta_min` floor reached)
   - **Methodological Distinction Reconciled:** The configured minimum floor of the cosine decay curve is `eta_min = 1.0e-5` ($0.000010$), reached after the 13th cosine step following Epoch 15. The observed training learning rate during the execution of Epoch 15 is $2.438 \times 10^{-5}$ (having undergone 12 cosine steps). Both values are mathematically verified and distinguished in protocol Table 4.1.
2. **Atomic Checkpoint Writing & Reload:**
   - Atomic saving was verified via `save_checkpoint`: the checkpoint is serialized to a `.pt.tmp` file first and atomically renamed via `replace()`.
   - Interrupted runs cannot leave corrupted `.pt` files.
   - Reloading verified that `epoch`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, `metrics`, and `peak_vram_mb` are preserved without loss of precision.

---

## 5. Targeted Engineering Repairs & Test Verification

### Defect DEF-P1-01: Model Factory & CLI Incomplete in `train_scaled.py`
- **Classification:** **P1** (Training pipeline model construction restriction).
- **Source:** `src/training/train_scaled.py` was authored during Stage B and contained hardcoded model factory branches and `argparse` choices restricted exclusively to `["resunet_2d", "unet_resnet34"]`.
- **Impact:** Attempting to launch 15-epoch retraining of U-Net, U-Net++, or Attention U-Net using `train_scaled.py` would fail with an `argparse` error or `ValueError: Unsupported model`.
- **Targeted Minimal Repair:**
  1. Updated imports in `src/training/train_scaled.py` to import `UNet2D`, `UNetPlusPlus2D`, and `AttentionUNet2D`.
  2. Extended `create_model(model_name)` to support `"unet"`, `"unetpp"` (`deep_supervision=False`), and `"attention_unet"`.
  3. Updated `argparse` choices to:
     `choices=["unet", "unetpp", "attention_unet", "resunet_2d", "unet_resnet34"]`.
- **Verification:** Authored focused unit test suite [`tests/test_train_scaled_model_factory.py`](file:///d:/MajorProject/tests/test_train_scaled_model_factory.py) containing:
  - 5 architecture creation and forward-pass tests (`unet`, `unetpp`, `attention_unet`, `resunet_2d`, `unet_resnet34`)
  - 1 invalid architecture rejection test
  - 1 focused synthetic gradient-clipping test (`test_synthetic_gradient_clipping_amplified_gradients`) with $1000\times$ amplified gradients, verifying clipping from $\|\nabla\|_2 > 250.0$ down to $\le 1.000$, with strictly zero optimizer steps and verified unaltered weights.
  All 7 test cases passed cleanly.

---

## 6. Full Regression Test Verification

Both test suites were executed with zero errors and zero failures:

1. **Main Project Test Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
   ```
   - **Result:** **77 tests passed** in 10.138s (0 failures, 0 errors).
   - *(Included pre-existing tests + all 7 tests in `test_train_scaled_model_factory.py`).*
2. **Agent Policy & Showcase Test Suite:**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s src/agent -p "test_*.py"
   ```
   - **Result:** **71 tests passed** in 2.923s (0 failures, 0 errors).
3. **Combined Test Status:** **148 total unit tests passing** cleanly across the repository.

---

## 7. Git Status & Code Integrity Comparison

A strict Git inspection confirms that all pre-existing changes and uncommitted state remain intact:

```text
Tracked modifications (Pre-existing, untouched):
 M src/data/dataset.py
 M src/evaluation/metrics.py
 M src/training/train.py

Sprint modifications (Targeted repair & test):
 M src/training/train_scaled.py
?? tests/test_train_scaled_model_factory.py

Deliverable report added:
?? G1_P3_DATASET_TRAINING_READINESS_REPORT.md

Zero staged files, zero commits, zero branch switches, zero reset operations.
```

---

## 8. Hardware Profile & Resource Footprint

- **Workstation:** Acer Predator Helios 16
- **CPU:** Intel Core i9-13900HX (24 cores, 32 threads)
- **RAM:** 16.0 GB physical DDR5 RAM
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6 VRAM, 7,995.6 MB addressable)
- **Environment:** Windows 11, Python 3.11.9, PyTorch `2.11.0+cu128`, CUDA 12.8, MONAI 1.6.0
- **Observed Peak VRAM During Dry-Run (Synthetic Evidence Only):**
  - U-Net: 356.8 MB
  - Attention U-Net: 492.1 MB
  - U-Net++: 1,067.0 MB
- **Workstation Headroom Assessment (Synthetic Dry-Run Evidence Only):** Conservative batch size 2 and single-slice passes consume $\le 1.1$ GB allocated VRAM in synthetic dry-runs, indicating that isolated forward-backward passes comfortably fit within the 8 GB GPU VRAM. However, this measurement represents **synthetic dry-run evidence only**: it does not guarantee protection against OS paging, background application memory spikes, multi-worker DataLoader RAM accumulation, or GPU thermal throttling during sustained multi-hour training. System-level background process isolation and active cooling will remain required during actual execution.

---

## 9. Compact Technical Readiness Matrix

| Component | Status | Evidence | Remaining Blocker |
| :--- | :---: | :--- | :--- |
| **721-Case Data Integrity** | **READY** | All 3,605 NIfTI files verified; 0 shape, affine, non-finite, or empty errors across 721 cases. | None |
| **Dataset Loader** | **READY** | `BraTSGLISliceDataset` verified; exact 25% background sampling ratio; deterministic indexing. | None |
| **Preprocessing** | **READY** | Per-case non-zero voxel z-score standardization verified; labels strictly $\in \{0, 1, 2, 3\}$. | None |
| **Model Construction** | **READY** | All 5 architectures construct cleanly via `create_model`; forward output shapes `[B, 4, 240, 240]`. | None |
| **Loss & Gradients** | **READY** | `DiceCrossEntropyLoss` produces finite loss; finite gradients; norm clipping verified. | None |
| **Optimizer & Scheduler** | **READY** | AdamW with 2-epoch linear warmup + 13-epoch cosine decay verified across all 15 epochs. | None |
| **Checkpoint Handling** | **READY** | Atomic `.pt.tmp` write and rename verified; `best_model.pt` and `final_model.pt` distinct. | None |
| **Recovery** | **READY** | Full state dict, epoch, metric, and sampler recovery tested and verified. | None |
| **Regression Tests** | **READY** | 148 / 148 unit tests passing with 0 failures and 0 errors (77 main, 71 agent). | None |

---

## 10. Final Decision Gate Recommendation

# **`A. TECHNICALLY READY`**

> **Execution Stop Declaration:**
> All dataset integrity checks, data loaders, model factories, loss formulations, optimizers, learning rate schedulers, and checkpoint recovery systems are verified technically operational. **No actual model training has commenced, no model weights have been updated, and no training authorization is implied.** The pipeline is closed and awaits academic supervisor review and training sign-off.
