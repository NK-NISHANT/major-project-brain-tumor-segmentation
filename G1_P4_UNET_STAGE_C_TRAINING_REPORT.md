# G1-P4 — First Stage C Training Run Preparation & Authorization Gate Report: U-Net 2D

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Stage C Protocol Reference:** [`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`](file:///d:/MajorProject/STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md)
**Readiness Evidence Reference:** [`G1_P3_DATASET_TRAINING_READINESS_REPORT.md`](file:///d:/MajorProject/G1_P3_DATASET_TRAINING_READINESS_REPORT.md)
**Decision Register Reference:** [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv)
**Date:** November 13–14, 2026
**Final Decision Gate:** **`COMPLETED FULLY`** (All 15 Epochs Completed Successfully, Best Checkpoint at Epoch 15: Mean Dice 0.8594)
**Training Execution Status:** **EXECUTED UNDER PROJECT-OWNER AUTHORIZATION (15/15 Epochs Complete)**

---

> [!NOTE]
> **Historical Pre-Flight Record Notice (Sections 1–9):**
> Sections 1 through 9 document the historical pre-flight preparation and readiness audit conducted on November 13, 2026 under the initial G1-P4 instruction. The initial mandatory external-supervisor approval requirement noted in Section 1 was subsequently superseded under the self-governed student academic project model. As documented in Section 10.2, internal approval for Decisions `DEC-C-01`, `DEC-C-03`, and `DEC-C-04` was formally approved by the Student Project Team / Project Owner on November 13, 2026. Effective project governance did not require external supervisor approval for routine experiments. Sections 1–9 are preserved verbatim below to maintain complete historical chronology.

## 1. Historical Pre-Flight Record: Mandatory Authorization Gate Assessment (Section 0)

Under Section 0 of the G1-P4 protocol:
> *"Before executing training, establish both:
> 1. Written supervisor approval for the Phase 1 training scope (DEC-C-01 and DEC-C-03) and the checkpoint-selection procedure (DEC-C-04).
> 2. Explicit project-owner authorization to commence this training run.
> Do not interpret `READY FOR SUPERVISOR REVIEW` or `TECHNICALLY READY` as approval.
> If either approval is missing, **DO NOT START TRAINING**. Instead, finish the non-destructive preparation checks, provide the exact launch command and report the outstanding authorization."*

### 1.1 Authorization Status Table

| Authorization Gate Dimension | Required Status | Current Repository Evidence | Gate Finding |
| :--- | :--- | :--- | :---: |
| **Supervisor Approval: DEC-C-01** (721-case training cohort) | Signed approval file, email, or formal reference | Listed in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv) as `RECOMMENDED - PENDING SUPERVISOR REVIEW` | **MISSING / PENDING** |
| **Supervisor Approval: DEC-C-03** (Phase 1 2D Retraining) | Signed approval file, email, or formal reference | Listed in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv) as `RECOMMENDED - PENDING SUPERVISOR REVIEW` | **MISSING / PENDING** |
| **Supervisor Approval: DEC-C-04** (15-ep Checkpoint Selection) | Signed approval file, email, or formal reference | Listed in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv) as `RECOMMENDED - PENDING SUPERVISOR REVIEW` | **MISSING / PENDING** |
| **Project-Owner Training Authorization** | Explicit affirmative (`YES`) | Provided as evaluation template (`[YES / NO]`) without explicit `YES` grant | **NOT GRANTED** |

**Mandatory Gate Action:** In strict compliance with Section 0 and Section 10, **TRAINING HAS NOT COMMENCED**. All technical non-destructive preparation checks were conducted, the exact launch command is formulated, and the run is staged pending explicit written authorization.

---

## 2. Non-Destructive Technical Pre-Execution Checks

All pre-execution verification checks mandated by Section 2 and Section 3 were completed successfully:

### 2.1 Git Status & Source Code Integrity (Check 1)
- Tracked pre-existing uncommitted modifications remain preserved and untouched:
  - `src/data/dataset.py`
  - `src/evaluation/metrics.py`
  - `src/training/train.py`
- Training runner `src/training/train_scaled.py` was inspected:
  - Verified factory resolution: `create_model("unet")` returns an instance of `src.models.unet.UNet2D`.
  - Verified CLI interface: `argparse` choices include `["unet", "unetpp", "attention_unet", "resunet_2d", "unet_resnet34"]`.
- Repository test suite status: **148 unit tests passing** (77 main, 71 agent) with 0 failures and 0 errors.

### 2.2 Model Architecture & Parameter Verification (Checks 2 & 3)
- Target architecture: Standard 4-level U-Net 2D (`UNet2D`).
- Input channels: 4 (`t1n, t1c, t2f, t2w`).
- Output classes: 4 (Background, NCR, ED, ET).
- Base feature channels: 32 (channel progression: 32 $\rightarrow$ 64 $\rightarrow$ 128 $\rightarrow$ 256, bottleneck 512).
- Total trainable parameters: **7,763,428** parameters (~7.76 M).
- Initialization: Kaiming normal (`bias=0`). Pretrained weights: None (scratch).

### 2.3 Partial Batch & Batch Normalization Robustness (Section 3 Contract)
- `UNet2D` employs `nn.BatchNorm2d` across double-convolution blocks.
- In PyTorch, 2D spatial feature maps $[B, C, H, W]$ reduce $B \times H \times W$ values per channel during training.
- Evaluated forward-backward execution on a synthetic singleton batch ($B=1, C=4, H=240, W=240$):
  - Spatial volume per channel: $1 \times 240 \times 240 = \mathbf{57,600}$ values.
  - Forward output shape: `torch.Size([1, 4, 240, 240])` (verified successful).
  - Backward pass: gradient computation completed with finite values (verified successful).
  - `DiceCrossEntropyLoss` forward-backward on $B=1$: produced finite loss ($2.4719$) and successful backpropagation.
- Conclusion: `UNet2D` can safely process partial batches of size 1 emitted by `CaseSliceBatchSampler` without runtime exception or gradient collapse.

### 2.4 Dataset Manifests & Cohort Disjointness (Checks 4 & 5)
- **Training Cohort:** `results/dataset_splits/development_train_manifest.json`
  - Total cases: **721 cases**
  - Unique biological subjects: **667 subjects**
  - Census: 47,228 ground-truth foreground slices, 64,527 background slices (total 111,755 slices).
- **Validation Cohort:** `results/dataset_splits/validation_manifest.json`
  - Total cases: **155 cases**
  - Unique biological subjects: **143 subjects**
- **Disjointness Audit:**
  - Case ID intersection: $\text{Train} \cap \text{Val} = \mathbf{0}$ cases.
  - Biological Subject intersection: $\text{Train} \cap \text{Val} = \mathbf{0}$ subjects.
  - Validation cases do not enter `CaseSliceBatchSampler`.

### 2.5 Output Directory & Checkpoint Protection (Check 6)
- Target experiment directory: `results/stage_c/phase1/unet_721_seed42/`
- Directory status on disk: **Non-existent (`False`)**.
- Parent directory `results/stage_c` status: **Non-existent (`False`)**.
- Conclusion: No pre-existing Stage A, Stage B, or Stage C checkpoints can be accidentally overwritten.

### 2.6 Workstation Disk Space Availability (Check 7)
- Drive `D:` storage capacity:
  - Total size: 634.07 GB
  - Used space: 577.73 GB
  - **Free space:** **56.34 GB**
- Estimated U-Net checkpoint footprint:
  - `best_model.pt`: ~31 MB
  - `final_model.pt`: ~31 MB
  - `history.csv` + `training_log.txt`: < 2 MB
  - Total run storage demand: $\approx 64$ MB ($< 0.12\%$ of available free disk space).

### 2.7 Hardware & Environmental Baseline Telemetry (Check 8)
Recorded at November 13, 2026, 17:05 local time on workstation **Acer Predator Helios 16**:
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6)
  - Driver Version: 572.16 | CUDA Version: 12.8
  - Total VRAM: 8,188 MiB (7,995.6 MB addressable)
  - Baseline Used VRAM: 1,964 MiB (OS / desktop display)
  - Baseline Free VRAM: **5,985 MiB**
  - Baseline GPU Temperature: **41.2 °C**
  - Baseline GPU Utilization: **0 %**
- **CPU:** 13th Gen Intel Core i9-13900HX (24 physical cores, 32 logical threads)
  - Baseline CPU Load: **4 %**
- **System Memory:** 16.0 GB DDR5 RAM
  - Total Visible RAM: 15.73 GB
  - Baseline Free RAM: **6.93 GB**

---

## 3. Pre-Execution Isolation Audit & Quarantine Boundary Resolution

During pre-execution inspection of `src/training/train_scaled.py` (lines 407–420):
- **Initial Observation:** `train_scaled.py` was originally scripted to read `results/dataset_splits/test_manifest.json` at launch to perform an automated set-disjointness assert against the training manifest.
- **Protocol Constraint D & Section 9 Rule:** The Stage C protocol explicitly mandates:
  > *"Never open, parse, enumerate or hash `results/dataset_splits/test_manifest.json`."*
- **Resolution (Implemented in G1-P4.1):** The runner was refactored with `load_and_verify_cohort_manifests`. Live disk access to `test_manifest.json` was completely removed, while training/validation case and subject disjointness safeguards were strengthened. (See Section 8 for complete technical verification).

---

## 4. Locked Launch Specification

When supervisor approval and project-owner authorization are received, the run will execute with the following locked configuration:

| Hyperparameter / Dimension | Locked Value | Verification Status |
| :--- | :--- | :---: |
| **Model Architecture** | `unet` (`UNet2D`, standard 4-level) | Verified (`tests/test_train_scaled_model_factory.py`) |
| **Input Channels** | 4 channels: `t1n, t1c, t2f, t2w` | Verified (`src/data/dataset.py`) |
| **Spatial Grid** | $240 \times 240$ axial slices | Verified |
| **Output Classes** | 4 (0: BG, 1: NCR, 2: ED, 3: ET) | Verified |
| **Training Cases / Subjects** | 721 cases / 667 unique subjects | Verified (`development_train_manifest.json`) |
| **Validation Pool** | 155 cases / 143 unique subjects | Verified (`validation_manifest.json`) |
| **Epoch Budget** | 15 Epochs | Locked |
| **Physical Batch Size** | 2 slices (within-case grouping) | Verified (`CaseSliceBatchSampler`) |
| **Background Slice Ratio** | 0.25 (25% background slices) | Verified |
| **Loss Function** | Multiclass Dice + Cross-Entropy ($\lambda_D=1.0, \lambda_{CE}=1.0$) | Verified (Dice excludes BG) |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$) | Locked |
| **Base Learning Rate** | $1 \times 10^{-3}$ ($0.001000$) | Locked |
| **LR Schedule** | 2-epoch Linear Warmup $\rightarrow$ 13-epoch Cosine Decay | Verified step profile |
| **Configured Cosine Floor** | $\eta_{\min} = 1.0 \times 10^{-5}$ ($0.000010$) | Verified. Note: `history.csv` records the learning rate active *during* training updates ($2.438 \times 10^{-5}$ during Epoch 15); the optimizer parameter group reached the configured floor ($1.0 \times 10^{-5}$) upon the completion of Epoch 15 via `scheduler.step()`, confirmed in `final_model.pt`. |
| **Gradient Clipping** | Maximum L2 norm = 1.0 | Verified (`test_synthetic_gradient_clipping_amplified_gradients`) |
| **Random Seed** | 42 | Locked |
| **Checkpoint Criterion** | Max Validation Mean Dice across 155-case pool | Locked |
| **Checkpoint Exposure** | Formally reported as selection-exposed internal validation | Disclosed |
| **Target Directory** | `results/stage_c/phase1/unet_721_seed42/` | Verified new |

---

## 5. Exact Launch Command

The verified command for initiating this training run once authorization is granted:

```powershell
# Execute from workspace root: D:\MajorProject
.\.venv\Scripts\python.exe src/training/train_scaled.py `
  --model unet `
  --epochs 15 `
  --batch-size 2 `
  --base-lr 0.001 `
  --warmup-epochs 2 `
  --max-grad-norm 1.0 `
  --seed 42 `
  --results-dir results/stage_c/phase1/unet_721_seed42
```

*(Estimated runtime: ~7.5 GPU hours on NVIDIA RTX 4070 Laptop GPU).*

---

## 6. Scientific Limitations & Selection-Exposure Disclosure

1. **Selection Exposure:** Checkpoints selected via maximum validation Mean Dice on the 155-case validation pool (`Val_eval` + `Val_cal`) are selection-exposed. All resulting validation metrics must be cited strictly as **selection-exposed internal validation metrics**, not independent generalization benchmarks.
2. **Calibration Decoupling:** Temperature scaling parameter $T^*$ must not be optimized during training. Post-hoc calibration fitting on `Val_cal` ($T^*_{\text{unet}}$) will be conducted as a separate post-training work package under Work Package C2.
3. **Hardware Vigilance:** Single-batch dry-runs verified a peak allocation of 356.8 MB for U-Net. However, sustained 15-epoch training will require active workstation cooling to prevent thermal throttling.

---

## 7. Initial Pre-Flight Authorization Assessment Summary
At the conclusion of G1-P4 pre-flight checks, all hardware, dataset, model, and configuration verifications were ready. However, actual training remained blocked pending written supervisor sign-off and explicit project-owner authorization. In addition, G1-P4 identified the code-level test quarantine blocker in `train_scaled.py` (runtime access to `test_manifest.json`), which required remediation prior to launch.

---

## 8. G1-P4.1 — Test Quarantine Runtime Dependency Removal

### 8.1 Exact Source Code Change in `src/training/train_scaled.py`
The direct runtime dependency on `results/dataset_splits/test_manifest.json` was removed from `train_candidate`. In its place, a dedicated function `load_and_verify_cohort_manifests` was added to `src/training/train_scaled.py`:

```python
def load_and_verify_cohort_manifests(
    splits_dir: Path,
    expected_train_cases: int = 721,
    expected_train_subjects: int = 667,
    expected_val_cases: int = 155,
    expected_val_subjects: int = 143,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load authorized training and validation manifests and verify strict cohort integrity.

    Verifications performed:
      1. Existence and valid JSON parsing of permitted manifests (development_train_manifest.json, validation_manifest.json).
      2. Key structure validation ('cases' list of dicts with 'case_id' and 'subject_id').
      3. Expected case census (721 train cases, 155 validation cases).
      4. Expected unique biological subject census (667 train subjects, 143 validation subjects).
      5. Strict case-ID disjointness (train_cases & val_cases == empty set).
      6. Strict biological-subject disjointness (train_subjects & val_subjects == empty set).

    Test Quarantine Enforcement:
      This function NEVER opens, reads, parses, enumerates, or hashes test_manifest.json
      or any quarantine path, preserving the pre-registered sealed test quarantine boundary.
    """
    train_manifest_path = splits_dir / "development_train_manifest.json"
    val_manifest_path = splits_dir / "validation_manifest.json"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Missing required development training manifest: {train_manifest_path}")
    if not val_manifest_path.exists():
        raise FileNotFoundError(f"Missing required validation manifest: {val_manifest_path}")

    try:
        with open(train_manifest_path, "r", encoding="utf-8") as f:
            train_manifest = json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed training manifest JSON at {train_manifest_path}: {exc}") from exc

    try:
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            val_manifest = json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed validation manifest JSON at {val_manifest_path}: {exc}") from exc

    if not isinstance(train_manifest, dict) or "cases" not in train_manifest or not isinstance(train_manifest["cases"], list):
        raise ValueError("Training manifest missing valid 'cases' list")
    if not isinstance(val_manifest, dict) or "cases" not in val_manifest or not isinstance(val_manifest["cases"], list):
        raise ValueError("Validation manifest missing valid 'cases' list")

    train_cases_info = train_manifest["cases"]
    val_cases_info = val_manifest["cases"]

    if len(train_cases_info) != expected_train_cases:
        raise ValueError(f"Expected {expected_train_cases} train cases, got {len(train_cases_info)}")
    if len(val_cases_info) != expected_val_cases:
        raise ValueError(f"Expected {expected_val_cases} val cases, got {len(val_cases_info)}")

    train_cids = {c["case_id"] for c in train_cases_info if isinstance(c, dict) and "case_id" in c}
    val_cids = {c["case_id"] for c in val_cases_info if isinstance(c, dict) and "case_id" in c}

    if len(train_cids) != expected_train_cases:
        raise ValueError(f"Malformed training manifest: expected {expected_train_cases} unique case_ids, found {len(train_cids)}")
    if len(val_cids) != expected_val_cases:
        raise ValueError(f"Malformed validation manifest: expected {expected_val_cases} unique case_ids, found {len(val_cids)}")

    overlap_cids = train_cids & val_cids
    if overlap_cids:
        raise ValueError(f"FATAL: Train and Val case ID overlap detected ({len(overlap_cids)} cases): {sorted(list(overlap_cids))[:5]}")

    train_sids = {c["subject_id"] for c in train_cases_info if isinstance(c, dict) and "subject_id" in c}
    val_sids = {c["subject_id"] for c in val_cases_info if isinstance(c, dict) and "subject_id" in c}

    if len(train_sids) != expected_train_subjects:
        raise ValueError(f"Expected {expected_train_subjects} unique train subjects, got {len(train_sids)}")
    if len(val_sids) != expected_val_subjects:
        raise ValueError(f"Expected {expected_val_subjects} unique val subjects, got {len(val_sids)}")

    overlap_sids = train_sids & val_sids
    if overlap_sids:
        raise ValueError(f"FATAL: Biological subject overlap detected between train and val ({len(overlap_sids)} subjects): {sorted(list(overlap_sids))[:5]}")

    return train_cases_info, val_cases_info
```

In `train_candidate`, lines 404–431 were replaced with:
```python
    # --- 1. Load Manifests and Verify Cohort Integrity ---
    print(f"Loading authorized manifests for {model_name}...")
    train_cases_info, val_cases_info = load_and_verify_cohort_manifests(splits_dir)
    print("Cohort integrity verified: Train (721 cases / 667 subjects) and Val (155 cases / 143 subjects) are strictly disjoint.")
```

### 8.2 Focused Regression Test Implementation
In [`tests/test_train_scaled_model_factory.py`](file:///d:/MajorProject/tests/test_train_scaled_model_factory.py), class `TestCohortManifestQuarantine` was added with 7 dedicated test methods:
1. `test_cohort_manifest_loader_does_not_access_test_manifest`: Employs a guarded wrapper on `builtins.open` that raises `PermissionError` immediately if any code attempts to open `test_manifest.json` or any file containing `quarantine`. Proves that `load_and_verify_cohort_manifests` loads the authorized 721 train cases and 155 val cases cleanly without accessing `test_manifest.json`.
2. `test_guarded_open_catches_simulated_quarantine_breach`: Validates that the guard wrapper actively detects and catches attempts to open `test_manifest.json`.
3. `test_cohort_manifest_case_overlap_detected`: Verifies that synthetic overlapping case IDs between train and val trigger an explicit `ValueError` ("case ID overlap detected").
4. `test_cohort_manifest_subject_overlap_detected`: Verifies that synthetic overlapping biological subject IDs trigger an explicit `ValueError` ("Biological subject overlap detected").
5. `test_missing_manifest_raises_file_not_found`: Verifies that missing manifests raise an explicit `FileNotFoundError`.
6. `test_malformed_manifest_raises_value_error`: Verifies that malformed JSON or invalid schema raises an explicit `ValueError`.
7. `test_cli_argument_parsing_and_defaults`: Verifies non-training CLI argument parsing for `--model unet` and all 15-epoch training flags.

### 8.3 Actual Test Execution Outputs
- **Model Factory & Quarantine Test Suite:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest tests/test_train_scaled_model_factory.py
  ```
  *Output:* `Ran 14 tests in 0.814s — OK`
- **Main Project Regression Test Suite:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
  ```
  *Output:* `Ran 84 tests in 7.411s — OK`
- **Agent Policy & Showcase Test Suite:**
  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s src/agent -p "test_*.py"
  ```
  *Output:* `Ran 71 tests in 2.385s — OK`
- **Total Repository Test Census:** **155 tests passing**, 0 failures, 0 errors, 0 skipped.

### 8.4 Confirmation of Safeguards & Zero Quarantine Access
- **Test Manifest Disk Access:** Completely removed. The startup path performs zero open, read, parse, enumerate, or hash calls against `test_manifest.json`.
- **Cohort Integrity:** Strictly enforced on startup. Both case-level disjointness ($721 \cap 155 = 0$) and subject-level disjointness ($667 \cap 143 = 0$) are validated before dataset construction.
- **Model Parameters:** Strictly untouched. Zero training epochs executed, zero optimizer steps called.

### 8.5 Git Status Comparison
```text
Tracked modifications (Pre-existing, untouched):
 M src/data/dataset.py
 M src/evaluation/metrics.py
 M src/training/train.py

Sprint modifications (Targeted quarantine repair & test suite):
 M src/training/train_scaled.py
?? tests/test_train_scaled_model_factory.py

Deliverable reports:
?? G1_P4_UNET_STAGE_C_TRAINING_REPORT.md
```

### 8.6 Remaining Authorization Requirements Prior to Actual Execution
Actual model training execution remains strictly conditioned on:
1. Written supervisor sign-off on decisions `DEC-C-01`, `DEC-C-03`, and `DEC-C-04` in [`STAGE_C_DECISION_REGISTER.csv`](file:///d:/MajorProject/STAGE_C_DECISION_REGISTER.csv).
2. Explicit affirmative project-owner instruction (`YES`).

---

## 9. Final Decision Gate Recommendation

# **`A. LAUNCH PATH TECHNICALLY READY`**

> **Technical Readiness & Standby Declaration:**
> The training runner's runtime dependency on `test_manifest.json` has been **completely removed**. The cohort validation logic relies solely on authorized training and validation manifests, strictly enforcing both case-ID and subject-ID disjointness. All 155 unit tests pass cleanly.
>
> **Zero model training has commenced, no model weights have been updated, and no checkpoints have been created.** The pipeline is in technical launch standby awaiting formal academic supervisor sign-off and explicit project-owner authorization.

---

## 10. Historical Record: Initial G1-P4.2 Authorization Gate Block & Resolution

### 10.1 Historical Gate Block Summary
During the initial G1-P4.2 audit (November 13, 2026, 17:47 IST), execution was temporarily halted because the initial prompt assumed external research supervisor sign-off was required. Decisions `DEC-C-01`, `DEC-C-03`, and `DEC-C-04` were listed as pending external review in `STAGE_C_DECISION_REGISTER.csv`, and project-owner authorization was unresolved.

### 10.2 Governance Correction (G1-P4.2 Correction)
Under the student academic project model, the Student Project Team and Project Owner are the governing authority for routine experimental execution. The decision register was updated to record internal approval:
- `DEC-C-01`: APPROVED (2026-11-13) by Student Project Team / Project Owner
- `DEC-C-03`: APPROVED (2026-11-13) by Student Project Team / Project Owner
- `DEC-C-04`: APPROVED (2026-11-13) by Student Project Team / Project Owner
- Explicit Project-Owner Authorization: Granted affirmatively (`YES`).

---

## 11. G1-P4.2 Execution: Stage C Phase 1 U-Net Training Run

### 11.1 Execution Authorization & Timestamp
- **Governance Authority:** Student Project Team / Project Owner (Decisions DEC-C-01, DEC-C-03, DEC-C-04 Approved)
- **Project-Owner Authorization:** Explicitly granted (`YES`)
- **Execution Start Time:** November 13, 2026, 17:57:46 IST
- **Execution End Time:** November 13, 2026, 19:47:47 IST
- **Total Elapsed Runtime:** 108.8 minutes (~1.81 hours)

### 11.2 Exact Execution Command & Locked Configuration
```powershell
# Executed from D:\MajorProject
.\.venv\Scripts\python.exe src/training/train_scaled.py `
  --model unet `
  --epochs 15 `
  --batch-size 2 `
  --base-lr 0.001 `
  --warmup-epochs 2 `
  --max-grad-norm 1.0 `
  --seed 42 `
  --results-dir results/stage_c/phase1/unet_721_seed42
```

| Dimension | Executed Value |
| :--- | :--- |
| **Model Architecture** | Standard U-Net 2D (`UNet2D`, 4-level) |
| **Trainable Parameters** | 7,763,428 parameters (~7.76 M) |
| **Training Cohort** | 721 cases / 667 unique subjects (59,020 slices: 47,228 foreground + 11,792 background) |
| **Validation Cohort** | 155 cases / 143 unique subjects (12,364 slices) |
| **Input Dimensions** | 4 channels (`t1n, t1c, t2f, t2w`), $240 \times 240$ axial grid |
| **Output Classes** | 4 (0: BG, 1: NCR, 2: ED, 3: ET) |
| **Physical Batch Size** | 2 slices (within-case batching via `CaseSliceBatchSampler`) |
| **Optimizer & Weight Decay** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay $1 \times 10^{-4}$) |
| **Learning Rate Schedule** | 2-epoch Linear Warmup ($0.0001 \rightarrow 0.001$) $\rightarrow$ 13-epoch Cosine Decay ($\eta_{\min}=1.0 \times 10^{-5}$) |
| **Gradient Clipping** | Maximum L2 norm = 1.0 |
| **Target Directory** | `results/stage_c/phase1/unet_721_seed42/` |

### 11.3 Actual Completed Epochs & Training Results
The run completed **4 full epochs** (Epochs 1, 2, 3, 4) and was interrupted during Epoch 5 at batch 17,500 / 29,641. All completed epoch results were flushed to `results/stage_c/phase1/unet_721_seed42/history.csv`:

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Norm | % Clipped | Learning Rate | Epoch Time | Peak VRAM | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4527 | 0.3549 | **0.7977** | 0.6669 | 0.8602 | 0.8061 | 0.7268 | 0.9978 | 1.84 | 55.9% | 0.000100 | 24.08 m | 418.8 MB | **YES [* BEST]** |
| **2** | 0.2605 | 0.4827 | 0.6841 | 0.5223 | 0.6845 | 0.7490 | 0.6188 | 0.9971 | 1.05 | 21.3% | 0.000550 | 24.08 m | 418.9 MB | NO |
| **3** | 0.2438 | 0.3815 | 0.7932 | 0.6590 | 0.8420 | 0.7893 | 0.7483 | 0.9977 | 0.45 | 6.8% | 0.001000 | 24.03 m | 420.3 MB | NO |
| **4** | 0.2309 | 0.3267 | 0.7844 | 0.6552 | 0.8692 | 0.8345 | 0.6496 | 0.9983 | 0.33 | 4.4% | 0.000986 | 24.06 m | 420.3 MB | NO |

- **Training Convergence:** Train loss consistently decreased from $0.4527 \rightarrow 0.2605 \rightarrow 0.2438 \rightarrow 0.2309$.
- **Gradient Stability:** Pre-clip gradient norms decreased from $1.84 \rightarrow 1.05 \rightarrow 0.45 \rightarrow 0.33$, with clipped batches dropping from $55.9\% \rightarrow 21.3\% \rightarrow 6.8\% \rightarrow 4.4\%$.
- **Validation Dice Trajectory:** Val Mean Dice reached **0.7977** at Epoch 1, with stable sub-compartment scores across epochs (ET peaking at $0.8692$ in Ep 4, ED peaking at $0.8345$ in Ep 4, NCR peaking at $0.7483$ in Ep 3).

### 11.4 Best and Final Checkpoint Verification
Both model checkpoints were verified and tested for state-dict reload compatibility:

1. **Best Model Checkpoint (`best_model.pt`):**
   - Path: `results/stage_c/phase1/unet_721_seed42/best_model.pt`
   - Selected Epoch: **Epoch 1**
   - Selected Metric: **Validation Mean Dice = 0.7977** (ET: 0.8602, ED: 0.8061, NCR: 0.7268)
   - File Size: 93,281,287 bytes (~93.3 MB)
   - SHA-256 Hash: `7B5796A8AEA5E7B1C22837A9C1A7293B08C55452E8DC98E5248315ACBD71593A`
   - Reload Test: **PASS** (State dict loads cleanly into newly constructed `UNet2D`).
2. **Final Model Checkpoint (`final_model.pt`):**
   - Path: `results/stage_c/phase1/unet_721_seed42/final_model.pt`
   - Completed Epoch: **Epoch 4**
   - Metrics: Validation Mean Dice = 0.7844 | Train Loss = 0.2309
   - File Size: 93,281,603 bytes (~93.3 MB)
   - SHA-256 Hash: `A527625700BFD5A6AD78690F47BDBEF0CDBF8A23C4D7DA978377C89CE3DA67FB`
   - Reload Test: **PASS** (State dict loads cleanly into newly constructed `UNet2D`).

### 11.5 Hardware Telemetry & Runtime Footprint
- **GPU:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6)
  - Peak GPU Temperature: **58.0 °C** (well within thermal operating boundaries)
  - Peak VRAM Allocation: **420.3 MB** (process allocation; total OS + GPU utilization $\approx 2.36$ GB out of 8.19 GB)
- **Workstation System RAM:** Physical usage remained stable with $> 6.5$ GB free physical memory.
- **CPU:** 13th Gen Intel Core i9-13900HX (stable at low utilization, no bottlenecks).
- **Total Runtime:** **108.8 minutes (~1.81 hours)**.

### 11.6 Technical Interruption Analysis
- **Interruption Event:** At Epoch 5, batch 17,500 / 29,641 (~12.6 minutes into Epoch 5), the execution encountered an asynchronous CUDA kernel exception:
  ```text
  torch.AcceleratorError: CUDA error: an illegal instruction was encountered
  ```
  reported at line 278 of `src/training/train_scaled.py` during `torch.isfinite(loss)` synchronization.
- **Protocol Compliance:** In strict adherence to Section 5 and Section 8 (*"If training fails: Preserve all valid outputs and logs. Record the exact failure and last completed epoch. Do not automatically restart from scratch or overwrite checkpoints. Do not change hyperparameters to force completion"*), execution was stopped safely, preserving all artifacts, checkpoints, logs, and history records.
- **Recoverability:** The training runner's atomic checkpointing ensures that `final_model.pt` at Epoch 4 is fully recoverable via `--resume`.

### 11.7 Scientific Limitations & Selection-Exposure Disclosure
- **Selection Exposure:** Validation Dice scores were evaluated on the 155-case validation pool (`Val_eval` + `Val_cal`). In accordance with methodological standards, these metrics are formally classified as **selection-exposed internal validation metrics**, not independent generalization performance.
- **Test Quarantine:** The sealed 155-case test cohort (`results/dataset_splits/test_manifest.json` and `data/quarantine_test/`) remained **100% UNTOUCHED** throughout execution. Zero test data was accessed, read, parsed, or evaluated.

### 11.8 Git Working Tree State
```text
Tracked modifications (Pre-existing, untouched):
 M src/data/dataset.py
 M src/evaluation/metrics.py
 M src/training/train.py

Sprint modifications:
 M STAGE_C_DECISION_REGISTER.csv
 M STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md
 M src/training/train_scaled.py
?? tests/test_train_scaled_model_factory.py
?? results/stage_c/phase1/unet_721_seed42/

Deliverable report:
?? G1_P4_UNET_STAGE_C_TRAINING_REPORT.md
```
`git diff --stat` confirms that all pre-existing tracked modifications in `dataset.py`, `metrics.py`, and `train.py` remain untouched (383 insertions, 31 deletions). Zero Git commits, resets, or branch switches were performed.

---

## 12. G1-P4.2A — CUDA Interruption Diagnosis and Recovery Assessment

### 12.1 Execution State & Process Verification
- **Current Process Status:** Terminated. Non-invasive process scan (`Get-Process python*`) confirms zero active python training processes.
- **Process Exit Code:** `1` (clean Python unhandled exception termination; no OS-level hard crash or memory dump).
- **Last Fully Completed Epoch:** **Epoch 4** (completed at 19:34 IST, validation metrics evaluated, checkpoint saved).
- **Interrupted Epoch & Batch:** **Epoch 5**, batch **17,500 / 29,641** (~12.6 minutes into Epoch 5; ~59.0% through the epoch's training batches).
- **Failure Operation:** Asynchronous GPU kernel execution failure detected on host during `torch.isfinite(loss)` synchronization in `run_train_epoch` (line 278 of `src/training/train_scaled.py`).

### 12.2 Complete Traceback Evidence
```text
Traceback (most recent call last):
  File "D:\MajorProject\src\training\train_scaled.py", line 782, in <module>
    main()
  File "D:\MajorProject\src\training\train_scaled.py", line 767, in main
    train_candidate(
  File "D:\MajorProject\src\training\train_scaled.py", line 618, in train_candidate
    train_loss, pre_norm, post_norm, clipped_count = run_train_epoch(
                                                     ^^^^^^^^^^^^^^^^
  File "D:\MajorProject\src\training\train_scaled.py", line 278, in run_train_epoch
    if not torch.isfinite(loss):
torch.AcceleratorError: CUDA error: an illegal instruction was encountered
Search for `cudaErrorIllegalInstruction' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

### 12.3 Windows Kernel & Display Driver Forensic Evidence
A targeted query of the Windows System Event Log (`Get-WinEvent`) at the exact timestamp of failure (`13-11-2026 19:47:46`) revealed kernel-level driver events from `nvlddmkm`:
- **Event ID 13 (`nvlddmkm`, Error):**
  `\Device\Video3; Graphics SM Global Exception on (GPC 2, TPC 5, SM 1): Multiple Warp Errors`
- **Event ID 13 (`nvlddmkm`, Error):**
  `\Device\Video3; Graphics Exception: ESR 0x516fb0=0xd 0x516fb4=0x24 0x516fa8=0xf81eb60 0x516fac=0x1174`
- **Event ID 153 (`nvlddmkm`, Error):**
  `\Device\Video3; Error occurred on GPUID: 100`

*Distinction regarding TDR:* Event ID 153 records a GPU-side error event on GPUID 100. However, absent corroborating display driver reset events (such as Desktop Window Manager TDR Event ID 4101), this event alone does not definitively prove a WDDM Timeout Detection and Recovery (TDR). The direct evidence firmly establishes an unresolved GPU-side CUDA/driver execution failure originating in the Streaming Multiprocessor (SM).

### 12.4 Failure Cause Analysis: Evidence vs. Hypotheses
- **Direct Empirical Evidence:**
  1. The failure was an internal streaming multiprocessor exception (`Multiple Warp Errors` on GPC 2, TPC 5, SM 1) logged by the NVIDIA kernel driver `nvlddmkm`.
  2. The failure was an unresolved GPU-side CUDA/driver execution failure, rather than an algorithmic loss divergence or CPU exception:
     - **Logged Batch Snapshots:** In Epoch 5, all 7 logged batch progress snapshots (printed every 2,500 batches) exhibited normal finite loss and gradient values: batch 2,500 (loss 0.4587, pre-norm 0.33), batch 5,000 (loss 0.1056, pre-norm 0.18), batch 7,500 (loss 0.2434, pre-norm 0.28), batch 10,000 (loss 0.0351, pre-norm 0.05), batch 12,500 (loss 0.0982, pre-norm 0.08), batch 15,000 (loss 0.0247, pre-norm 0.09), batch 17,500 (loss 0.1239, pre-norm 0.22).
     - **Epoch Averages:** Completed Epochs 1–4 recorded stable average training losses of 0.4527, 0.2605, 0.2438, and 0.2309 with pre-clip gradient norms of 1.84, 1.05, 0.45, and 0.33.
     - **Clipping Statistics:** The percentage of clipped batches declined steadily from 55.9% in Epoch 1 to 21.3% in Epoch 2, 6.8% in Epoch 3, and 4.4% in Epoch 4.
     - **Host Validation:** Every intermediate step before batch 17,500 passed the host-side `torch.isfinite(loss)` check without raising.
  3. The failure was **NOT** caused by out-of-memory: process VRAM allocation was stable at 420.3 MB (well below the 8,188 MiB capacity).
  4. The failure was **NOT** caused by GPU overheating: peak GPU temperature during training was 58.0 °C (well below thermal throttling limits of ~85 °C).
  5. The GPU returned to 100% operational, healthy, and responsive status immediately after process exit (`nvidia-smi` reported 48.6 °C, 0% utilization, 6,033 MiB free).
- **Hypothesis (Root Cause Mechanism):**
  The exact root cause is characterized as an unresolved GPU-side CUDA/driver execution failure. In a shared desktop workstation environment where the GPU simultaneously services OS display compositing and high-throughput CUDA training, a low-level driver or SM warp execution anomaly occurred at batch 17,500.

### 12.5 Artifact Preservation & Checkpoint Integrity Audit
All artifacts in `results/stage_c/phase1/unet_721_seed42/` were preserved intact:

| Artifact File | Size | Verification Finding | SHA-256 Hash |
| :--- | :---: | :--- | :--- |
| **`best_model.pt`** | 93,281,287 B | Intact, represents Epoch 1 (Mean Dice 0.7977). State dict reloads into `UNet2D` with zero parameter mismatches. | `7B5796A8AEA5E7B1C22837A9C1A7293B08C55452E8DC98E5248315ACBD71593A` |
| **`final_model.pt`** | 93,281,603 B | Intact, represents Epoch 4 (Mean Dice 0.7844, Train Loss 0.2309). State dict reloads into `UNet2D` with zero parameter mismatches. | `A527625700BFD5A6AD78690F47BDBEF0CDBF8A23C4D7DA978377C89CE3DA67FB` |
| **`history.csv`** | 1,717 B | Contains complete, uncorrupted records for Epochs 1, 2, 3, and 4. | `EA1A839B1D58F4D0358DA8A1DD57C4F9959CAC5365F5C7F4BCE1A27309116DF9` |
| **`train.log`** | 5,726 B | Complete progress telemetry log through Epoch 5 batch 17,500. | N/A |
| **`config.json`** | 343 B | Complete executed hyperparameter matrix. | N/A |

Zero checkpoints were deleted, overwritten, or corrupted.

### 12.6 Recovery State Completeness Assessment
An inspection of `final_model.pt` and the `--resume` mechanism in `src/training/train_scaled.py` was conducted against the required state components:

1. **Model Weights:** **PRESENT & VERIFIED** (`model_state_dict` covers all 64 parameter tensors of `UNet2D`).
2. **Optimizer State:** **PRESENT & VERIFIED** (`optimizer_state_dict` contains all 64 momentum state entries with `exp_avg`, `exp_avg_sq`, and `step`).
3. **Scheduler State:** **PRESENT & VERIFIED** (`scheduler_state_dict` contains `last_epoch=4`, `_last_lr=[0.000943300732698339]`, milestones, and step counters for both warmup and cosine sub-schedulers).
4. **Completed Epoch Number:** **PRESENT & VERIFIED** (`epoch=4`). Resumption begins at **Epoch 5, batch 1**.
5. **Interrupted Epoch Execution Plan:** Resuming from the Epoch 4 checkpoint means that training restarts at the beginning of Epoch 5 (batch 1). Consequently, the first 17,500 batches of the interrupted Epoch 5 will be repeated. The training loss and weights will proceed from the intact Epoch 4 boundary.
6. **Continuity vs Mathematical Equivalence:**
   - Model weights, AdamW first/second momentum vectors, and learning rate scheduler states are strictly preserved through the completed Epoch 4 boundary.
   - However, this is **not** claimed to be a "bit-for-bit identical continuation" or "mathematically equivalent" to an uninterrupted run: intermediate weight updates applied between batch 1 and batch 17,500 in the failed process are discarded with the process crash, and non-deterministic CUDA floating-point reductions and thread scheduling prevent bitwise identical arithmetic trajectories.
7. **Sampler State:** **PRESENT & VERIFIED** (`train_sampler.epoch = 4`). In `CaseSliceBatchSampler`, `__iter__` initializes `rng = random.Random(self.seed + self.epoch)` (`seed = 42 + 4 = 46`).
8. **AMP Scaler State:** Not applicable (run executed in standard FP32 precision).

### 12.7 Resumability Determination
Resumption via `train_scaled.py --resume` provides genuine epoch-boundary state restoration:
- It preserves model, optimizer, and scheduler continuity strictly through the Epoch 4 checkpoint.
- Resumption begins at **Epoch 5, batch 1**, repeating the first 17,500 batches of the interrupted Epoch 5 from the Epoch 4 weights.
- It will **NOT** restart the optimizer or learning-rate schedule.
- It will load the exact AdamW momentum buffers at the end of Epoch 4.
- It will resume the cosine learning rate schedule at Epoch 5 ($\text{LR} = 0.0009433$) and step through Epoch 15.
- It will preserve the exact 15-epoch experiment budget and reload historical metrics from `history.csv` without duplicate records.
- The best checkpoint (`best_model.pt` from Epoch 1, Mean Dice 0.7977) will remain the active comparison baseline.

### 12.8 Test-Quarantine Compliance
- `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` were **NOT ACCESSED, OPENED, READ, ENUMERATED, OR HASHED** during G1-P4.2 or G1-P4.2A.
- The test quarantine boundary remains 100% intact.

### 12.9 Diagnostic Gate Finding
Execution diagnostics confirmed that the Epoch 4 boundary was fully intact and resumable. Resumption from Epoch 5 batch 1 was approved and authorized under G1-P4.2B.

---

## 13. G1-P4.2B — Resumption Execution, Epoch 5 Completion, and Epoch 6 Diagnosis

### 13.1 Pre-Resumption Sanity Verification & Snapshot Preservation
Prior to launching resumption under G1-P4.2B:
1. **Artifact Preservation Snapshot (Epoch 4):**
   A dedicated recovery directory [`results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch4/`](file:///d:/MajorProject/results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch4/) was created. The four key artifacts were mirrored and verified bit-for-bit against the active files:
   - `best_model.pt`: `7B5796A8AEA5E7B1C22837A9C1A7293B08C55452E8DC98E5248315ACBD71593A`
   - `final_model.pt`: `A527625700BFD5A6AD78690F47BDBEF0CDBF8A23C4D7DA978377C89CE3DA67FB`
   - `history.csv`: `EA1A839B1D58F4D0358DA8A1DD57C4F9959CAC5365F5C7F4BCE1A27309116DF9`
   - `config.json`: `D25AFCB85F04722815F4093877770DBD48D21DEF5E4D75037767799CC941AE53`
2. **Bounded CUDA Sanity Check:**
   100 forward/backward passes were executed on a disposable `UNet2D` instance with synthetic tensors $(2, 4, 240, 240)$ using identical loss, optimizer, and gradient clipping settings:
   - Total runtime: 2.89 s (28.9 ms/iteration).
   - Clipped iterations: 0/100.
   - Peak VRAM: 418.8 MB.
   - GPU Temperature: 40.28 °C $\rightarrow$ 51.04 °C.
   - Result: 100% finite loss and gradients, 0 CUDA exceptions, clean teardown.

### 13.2 Resumption Execution & Launch Telemetry
- **Command:**
  ```powershell
  .\.venv\Scripts\python.exe src/training/train_scaled.py `
    --model unet `
    --epochs 15 `
    --batch-size 2 `
    --base-lr 0.001 `
    --warmup-epochs 2 `
    --max-grad-norm 1.0 `
    --seed 42 `
    --resume `
    --results-dir results/stage_c/phase1/unet_721_seed42
  ```
- **Launch Timestamp:** November 13, 2026, 20:30:11 IST
- **Task ID:** `task-11909`
- **State Restoration:** Checkpoint at Epoch 4 loaded cleanly; 4 prior history records loaded; prior duration initialized to 96.26 min; training initiated at **Epoch 5, batch 1** with learning rate $0.0009433$.

### 13.3 Epoch 5 Training & Validation Results
Epoch 5 ran to complete fulfillment (all 29,641 training batches and 6,211 validation batches):
- **Epoch Duration:** 25.18 minutes
- **Train Loss:** **0.2195** (consistent reduction from 0.4527 $\rightarrow$ 0.2605 $\rightarrow$ 0.2438 $\rightarrow$ 0.2309 $\rightarrow$ 0.2195)
- **Validation Loss:** 0.4320
- **Validation Mean Dice:** **`0.8425`** (**NEW BEST MODEL**, marked `[* BEST]`, exceeding Epoch 1 baseline of 0.7977 by **+0.0448**)
- **Validation Sub-Compartment Dice:**
  - **ET (Class 3):** **0.8858** (+0.0166 over Ep 4)
  - **ED (Class 2):** **0.8566** (+0.0221 over Ep 4)
  - **NCR (Class 1):** **0.7851** (+0.0368 over Ep 3 best)
  - **BG (Class 0):** **0.9984**
- **Validation Mean IoU:** **0.7302** (ET: 0.7950, ED: 0.7492, NCR: 0.6463, BG: 0.9968)
- **Gradient Dynamics:** Pre-clip gradient norm **0.30**; clipped batches down to **3.6%** (1,053 batches out of 29,641).
- **Peak Process VRAM:** 446.0 MB.
- **Checkpoint Updates:** Both `best_model.pt` and `final_model.pt` were updated on disk at 20:56 IST with the new Epoch 5 weights and metrics. `history.csv` was cleanly appended with row 5.

### 13.4 Completed History Census (Epochs 1–5)

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Norm | % Clipped | Learning Rate | Epoch Time | Cumulative Time | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4527 | 0.3549 | 0.7977 | 0.6669 | 0.8602 | 0.8061 | 0.7268 | 0.9978 | 1.84 | 55.9% | 0.000100 | 24.08 m | 24.08 m | YES |
| **2** | 0.2605 | 0.4827 | 0.6841 | 0.5223 | 0.6845 | 0.7490 | 0.6188 | 0.9971 | 1.05 | 21.3% | 0.000550 | 24.08 m | 48.17 m | NO |
| **3** | 0.2438 | 0.3815 | 0.7932 | 0.6590 | 0.8420 | 0.7893 | 0.7483 | 0.9977 | 0.45 | 6.8% | 0.001000 | 24.03 m | 72.20 m | NO |
| **4** | 0.2309 | 0.3267 | 0.7844 | 0.6552 | 0.8692 | 0.8345 | 0.6496 | 0.9983 | 0.33 | 4.4% | 0.000986 | 24.06 m | 96.26 m | NO |
| **5** | **0.2195** | 0.4320 | **0.8425** | **0.7302** | **0.8858** | **0.8566** | **0.7851** | 0.9984 | **0.30** | **3.6%** | 0.000943 | 25.18 m | 121.45 m | **YES [* BEST]** |

### 13.5 Epoch 6 Interruption & Hardware Forensic Diagnosis
- **Event:** Epoch 6 initiated at 20:56 IST with $\text{LR} = 0.000876$. Training proceeded through batches 2,500, 5,000, 7,500, 10,000, 12,500, 15,000, 17,500, 20,000, and 22,500 with all logged batch losses between $0.05$ and $0.57$ and pre-norms between $0.00$ and $0.54$.
- At approximately batch 23,000 / 29,641 (~17.8 minutes into Epoch 6; 21:14:33 IST), training halted with:
  ```text
  torch.AcceleratorError: CUDA error: an illegal instruction was encountered
  ```
- **Windows System Event Log Forensic:**
  Querying the Windows System Event Log at 21:14:33 IST revealed a cascade of 93 kernel driver events from `nvlddmkm`:
  ```text
  Event ID 13 (nvlddmkm):
  \Device\Video3; Graphics SM Warp Exception on (GPC x, TPC y, SM z): Out Of Range Register
  \Device\Video3; Graphics SM Global Exception on (GPC x, TPC y, SM z): Multiple Warp Errors
  \Device\Video3; Graphics Exception: ESR 0x5...
  ```
  followed by Event ID 153 (`\Device\Video3; Error occurred on GPUID: 100`).
- **Comprehensive Event Log Comparison:**
  A forensic comparison of both failure timestamps (`19:47:46` with 111 events, and `21:14:33` with 93 events) revealed that in both instances, exceptions were logged across virtually all GPCs (GPC 0–3), TPCs (0–5), and SMs (0–1). The appearance of `(GPC 2, TPC 5, SM 1)` in preliminary diagnostic snippets was an artifact of tailing the log during enumeration, rather than an isolated single-core fault. Between the two failures, exactly zero GPU or driver errors were recorded, and Epoch 5 completed with 100% stability. Physical hardware memory checks (`nvidia-smi -q`) confirm zero remapped rows and zero hardware thermal throttling. Therefore, no physical silicon defect is claimed.

### 13.6 Preservation of Epoch 5 Artifacts & Checkpoints
Immediately following the interruption, the completed Epoch 5 state was verified and preserved:
1. **Active Checkpoints in `results/stage_c/phase1/unet_721_seed42/`:**
   - `best_model.pt`: 93,281,991 bytes | SHA-256: `54875719F85F155AF883FCB8F00CB752E56930EA5E9FF03B22501B47D29B01C2` | Epoch 5 | Mean Dice: **0.8425**
   - `final_model.pt`: 93,282,307 bytes | SHA-256: `624401417E16606503061448347A542B0B576C5FDAF777D79228C8F644DED244` | Epoch 5 | Mean Dice: 0.8425
   - `history.csv`: 2,092 bytes | SHA-256: `1DC4D0E99597C2D007A56806AB39B2C8684D7A07543AE70B8281E3E137FDC03A` | 5 complete epoch rows
2. **Recovery Snapshot Directory (Epoch 5):**
   Created at [`results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch5/`](file:///d:/MajorProject/results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch5/).
   All 4 mirrored files verified bit-for-bit against active checkpoints (all hashes match `True`).
3. **Recovery State Summary:**
   The training pipeline possesses full state continuity through the completed Epoch 5 boundary (`epoch=5`, `best_mean_dice=0.8425`, `prior_duration_min=121.45`, `train_sampler.epoch=5`).
   Resumption is ready to proceed from **Epoch 6, batch 1** to Epoch 15 via `--resume`.

### 13.7 Test Quarantine Integrity
Throughout G1-P4.2, G1-P4.2A, and G1-P4.2B:
- `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` were **NEVER ACCESSED, OPENED, READ, ENUMERATED, OR HASHED**.
- Zero test data was touched. The sealed test quarantine boundary remains 100% intact.

---

## 14. G1-P4.2C — Targeted GPU Stability Diagnosis & Decision Gate

### 14.1 State & Checkpoint Preservation Verification
1. **Process Scan:** Confirmed 0 active python training processes running on the workstation.
2. **Epoch 5 Checkpoint Integrity:** Both `final_model.pt` and `best_model.pt` strictly represent the completed Epoch 5 with best Validation Mean Dice = **`0.8425`** (ET: 0.8858, ED: 0.8566, NCR: 0.7851).
3. **Recovery Snapshot Verification:** Verified all 4 files in [`results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch5/`](file:///d:/MajorProject/results/stage_c/phase1/unet_721_seed42/recovery_snapshot_epoch5/) against active files:
   - `best_model.pt`: `54875719F85F155AF883FCB8F00CB752E56930EA5E9FF03B22501B47D29B01C2` (Match: True)
   - `final_model.pt`: `624401417E16606503061448347A542B0B576C5FDAF777D79228C8F644DED244` (Match: True)
   - `history.csv`: `1DC4D0E99597C2D007A56806AB39B2C8684D7A07543AE70B8281E3E137FDC03A` (Match: True)
   - `config.json`: `D25AFCB85F04722815F4093877770DBD48D21DEF5E4D75037767799CC941AE53` (Match: True)
   Zero checkpoints or history records were deleted or overwritten.

### 14.2 Detailed Hardware & Driver Investigation (`nvidia-smi -q`)
A comprehensive hardware query via `nvidia-smi -q` and Windows System Event Log analysis confirmed:
- **Driver / CUDA Stack:** KMD Version 621.14 (CUDA 13.5 runtime, PyTorch CUDA 12.8 compatibility layer), VBIOS 95.06.18.00.25.
- **Hardware Memory Health:**
  - `Remapped Rows: Correctable Error = 0, Uncorrectable Error = 0, Pending = No`.
  - Zero single-bit or double-bit memory page retirements.
- **Clock Throttling & Thermal Readings:**
  - `SW Power Cap: Not Active` (power draw 4.8 W idle, ~72 W under load; power ceiling 130 W).
  - `HW Slowdown: Not Active` (HW Thermal Slowdown = 0 µs, HW Power Brake Slowdown = 0 µs).
  - GPU Target Temperature: 87 °C; current temperature 39.6 °C idle, 78.4 °C under sustained load.
- **Bus & PCIe Interface:** PCIe Generation 4, Link Width 8x (Max 8x), Operating cleanly.
- **Driver Recovery Events:** Event ID 153 occurred concurrently with Event ID 13 exception bursts. Zero Event ID 4101 display driver recovery events occurred. No errors occurred during the intermediate 25.18-minute Epoch 5 training run.
- **Silicon Integrity Determination:** Because the entire SM array logged exceptions simultaneously upon process crash, hardware memory contains 0 remapped rows, and thermal metrics remain well below ceilings, there is **no corroborating evidence of physical hardware silicon defect**.

### 14.3 20-Minute Isolated CUDA Stability Test Execution
An isolated stress test was executed using a fresh disposable `UNet2D` instance (7,763,428 parameters) on synthetic tensors $(2, 4, 240, 240)$ in standard FP32 precision with `DiceCrossEntropyLoss`, AdamW optimizer, and $1.0$ gradient clipping:
- **Diagnostic Configuration:** Executed with `CUDA_LAUNCH_BLOCKING=1` and per-iteration CUDA synchronization.
- **Duration:** Exactly **1,200.02 seconds (20.00 minutes)**.
- **Total Iterations Completed:** **42,593 iterations** (~1.44× the volume of an entire 29,641-batch training epoch).
- **Throughput:** Stable at **35.49 iterations/second (28.2 ms/iteration)**.
- **Numerical Stability:**
  - Average Loss: **2.1339** (completely finite and stable across all 42,593 iterations).
  - Average Gradient Norm: **0.006** (0.00% clipped iterations).
- **Resource Footprint:**
  - Peak PyTorch VRAM: **418.8 MB** (process allocation; total device VRAM ~2.6 GB out of 8.2 GB).
  - Peak GPU Temperature: **78.9 °C** (stabilized at 78.4 °C, well below the 87 °C target).
  - Power Consumption: ~71–73 W (ceiling 130 W).
  - GPU Utilization: ~75%.
- **Error Count:** **0 errors encountered (`None (Clean Run)`)**. Zero illegal instructions, zero SM warp exceptions, zero `nvlddmkm` Event Log entries during the entire 20-minute run.
- **Zero State Leakage:** No checkpoints saved; zero interaction with the real experiment directory or optimizer state.

### 14.4 Diagnostic Evaluation & Recommendation

# **`STABILITY TEST PASSED (UNDER SPECIFIED CONDITIONS) — CONTROLLED RESUME CAN BE CONSIDERED`**

> **Diagnostic Finding & Limitations:**
> The 20-minute isolated CUDA stability test passed under its specified conditions (42,593 iterations completed with zero errors, finite gradients, stable ~78 °C thermals, and zero recorded driver events using synthetic tensors in synchronous mode). However, passing this bounded 20-minute synthetic workload does not establish universal or long-term GPU stability across prolonged multi-hour end-to-end training.
>
> The precise root cause of the earlier U-Net CUDA illegal-instruction failures and driver exceptions remains unresolved. A potential working explanation—that asynchronous kernel queue congestion or micro-timing conflicts between desktop window management and background DataLoader host-device transfers under default non-blocking execution (`CUDA_LAUNCH_BLOCKING=0`) may have triggered the faults—is an **unconfirmed hypothesis**, not an established fact. No causal connection with CPU thermals or background software is established.
>
> **Recommendation:**
> Under synchronous CUDA execution (`CUDA_LAUNCH_BLOCKING=1`), resume U-Net training from the intact Epoch 5 checkpoint (`epoch=5`, best Mean Dice = **0.8425**), starting at **Epoch 6, batch 1**. Active cooling should remain engaged. Zero test quarantine data was accessed.
>
> **Historical outcome clarification:** This was the contemporaneous recommendation before resumption. It was subsequently executed successfully, as documented in Section 15, and the U-Net experiment completed all 15 epochs. The original recommendation is retained for historical chronology.

---

## 15. G1-P4.2D — Resumption, Full 15-Epoch Completion, Final Metrics, and Scientific Disclosure

### 15.1 Resumption Execution under Synchronous Execution
Following the successful 20-minute isolated CUDA stability diagnosis (G1-P4.2C), controlled resumption was authorized and initiated under `$env:CUDA_LAUNCH_BLOCKING = "1"`:
- **Resumption Command:**
  ```powershell
  $env:CUDA_LAUNCH_BLOCKING = "1"
  .\.venv\Scripts\python.exe src/training/train_scaled.py `
    --model unet `
    --epochs 15 `
    --batch-size 2 `
    --base-lr 0.001 `
    --warmup-epochs 2 `
    --max-grad-norm 1.0 `
    --seed 42 `
    --resume `
    --results-dir results/stage_c/phase1/unet_721_seed42
  ```
- **Startup Verification:**
  - Checkpoint loaded at Epoch 5 (`final_model.pt`).
  - Five prior history records restored cleanly.
  - Prior duration initialized to 121.45 minutes.
  - Best Mean Dice baseline restored as **`0.8425`** (Epoch 5).
  - Resumption initiated at **Epoch 6, batch 1** with learning rate `0.000876`.
- **Execution Stability:**
  Under synchronous execution (`CUDA_LAUNCH_BLOCKING=1`), Epochs 6 through 15 ran continuously for over 6.2 hours with **zero errors, zero illegal instructions, and zero nvlddmkm Event Log exceptions**.

### 15.2 Complete 15-Epoch Metric Trajectory (`history.csv`)

| Epoch | Train Loss | Val Loss | Val Mean Dice | Val Mean IoU | Val ET (C3) | Val ED (C2) | Val NCR (C1) | Val BG (C0) | Pre-Norm | % Clipped | Learning Rate | Epoch Time | Cumulative Duration | Best? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 0.4527 | 0.3549 | 0.7977 | 0.6669 | 0.8602 | 0.8061 | 0.7268 | 0.9978 | 1.84 | 55.9% | 0.000100 | 24.1 m | 24.1 m | YES |
| **2** | 0.2605 | 0.4827 | 0.6841 | 0.5223 | 0.6845 | 0.7490 | 0.6188 | 0.9971 | 1.05 | 21.3% | 0.000550 | 24.1 m | 48.2 m | NO |
| **3** | 0.2438 | 0.3815 | 0.7932 | 0.6590 | 0.8420 | 0.7893 | 0.7483 | 0.9977 | 0.45 | 6.8% | 0.001000 | 24.0 m | 72.2 m | NO |
| **4** | 0.2309 | 0.3267 | 0.7844 | 0.6552 | 0.8692 | 0.8345 | 0.6496 | 0.9983 | 0.33 | 4.4% | 0.000986 | 24.1 m | 96.3 m | NO |
| **5** | 0.2195 | 0.4320 | 0.8425 | 0.7302 | 0.8858 | 0.8566 | 0.7851 | 0.9984 | 0.30 | 3.6% | 0.000943 | 25.2 m | 121.5 m | YES |
| **6** | 0.2133 | 0.4767 | 0.8369 | 0.7215 | 0.8819 | 0.8437 | 0.7851 | 0.9982 | 0.28 | 3.1% | 0.000876 | 32.2 m | 153.7 m | NO |
| **7** | 0.2074 | 0.4408 | 0.8481 | 0.7380 | 0.8966 | 0.8363 | 0.8114 | 0.9983 | 0.29 | 2.8% | 0.000786 | 33.3 m | 187.0 m | YES |
| **8** | 0.2061 | 0.5589 | 0.8125 | 0.6870 | 0.8421 | 0.8525 | 0.7428 | 0.9984 | 0.28 | 2.9% | 0.000681 | 36.7 m | 223.7 m | NO |
| **9** | 0.1991 | 0.5135 | 0.8409 | 0.7269 | 0.8850 | 0.8317 | 0.8059 | 0.9983 | 0.25 | 2.4% | 0.000565 | 39.3 m | 263.0 m | NO |
| **10** | 0.1963 | 0.5184 | 0.8526 | 0.7451 | 0.8951 | 0.8625 | 0.8002 | 0.9986 | 0.26 | 2.5% | 0.000445 | 39.9 m | 302.9 m | YES |
| **11** | 0.1914 | 0.4911 | 0.8506 | 0.7427 | 0.8972 | 0.8659 | 0.7885 | 0.9986 | 0.28 | 2.8% | 0.000329 | 39.7 m | 342.6 m | NO |
| **12** | 0.1878 | 0.5224 | 0.8384 | 0.7241 | 0.8890 | 0.8431 | 0.7830 | 0.9985 | 0.28 | 2.7% | 0.000224 | 39.9 m | 382.5 m | NO |
| **13** | 0.1852 | 0.5381 | 0.8399 | 0.7282 | 0.8993 | 0.8583 | 0.7622 | 0.9986 | 0.28 | 2.7% | 0.000134 | 37.5 m | 420.0 m | NO |
| **14** | 0.1832 | 0.4966 | 0.8481 | 0.7391 | 0.9024 | 0.8542 | 0.7877 | 0.9985 | 0.28 | 2.6% | 0.000067 | 37.1 m | 457.1 m | NO |
| **15** | **0.1815** | **0.4589** | **0.8594** | **0.7554** | **0.9026** | **0.8662** | **0.8094** | **0.9986** | **0.29** | **2.6%** | **0.000024** | 37.3 m | **494.4 m (8.24 h)** | **YES [* BEST]** |

### 15.3 Checkpoint Selection & Sub-Compartment Breakdown
- **Selected Best Checkpoint:** **Epoch 15** (`best_model.pt`)
- **Selection Criterion:** Maximum Validation Mean Dice across the 155-case internal validation pool.
- **Best Validation Mean Dice:** **`0.8594`** (0.859373)
- **Best Validation Mean IoU:** **`0.7554`** (0.755392)
- **Sub-Compartment Performance Breakdown:**
  - **Enhancing Tumor (ET / Class 3):** **`0.9026`** (IoU: 0.8225)
  - **Peritumoral Edema (ED / Class 2):** **`0.8662`** (IoU: 0.7639)
  - **Necrotic Tumor Core (NCR / Class 1):** **`0.8094`** (IoU: 0.6798)
  - **Background (BG / Class 0):** **`0.9986`** (IoU: 0.9972)
- **Final Model Checkpoint:** **Epoch 15** (`final_model.pt`). Because Epoch 15 was the best overall epoch, `best_model.pt` and `final_model.pt` both contain the final Epoch 15 weights and optimizer state.

### 15.4 Checkpoint Reload Compatibility & Cryptographic Hashes

All checkpoints and execution artifacts were verified for reloadability and hashed via SHA-256:

| Artifact File | Size (Bytes) | SHA-256 Hash | Reload Test Finding |
| :--- | :---: | :--- | :---: |
| **`best_model.pt`** | 93,281,991 | `37D35BF578868356CAADC7F4A173AF7B82535BB427FDB6CCE8DF121A9BF06CEF` | **PASS** (`<All keys matched successfully>`) |
| **`final_model.pt`** | 93,282,307 | `2C92E32970FF73995BC17078483E7F44CEF038CEC8367C96FF6A372907FD7915` | **PASS** (`<All keys matched successfully>`) |
| **`history.csv`** | 5,852 | `A8624EC117A9E317274FC1B9C2749B0EC05BF115237741B3807504B58AAF94CA` | Exactly 15 unique epoch records |
| **`summary.json`** | 1,253 | `E05BF4E5D4BB878840C3D948B494A8B4B66D6660D49E1BB3F2BCF8F14E475ED9` | Valid JSON, matches `history.csv` |
| **`config.json`** | 343 | `D25AFCB85F04722815F4093877770DBD48D21DEF5E4D75037767799CC941AE53` | Unmodified locked hyperparameter matrix |

### 15.5 Hardware Telemetry & Execution Footprint
- **Total Experiment Runtime:** **8.24 hours (494.4 minutes)** cumulative.
- **Synchronous Resumption Runtime:** 6.22 hours for Epochs 6 through 15 (~37.3 min/epoch under `CUDA_LAUNCH_BLOCKING=1`).
- **Peak Process VRAM Allocation:** **446.3 MB** (device VRAM remained stable at ~2.4 GB out of 8.2 GB).
- **GPU Thermal State:** Peaked at 78.4 °C under sustained compute; dropped cleanly to 44.1 °C at run termination.
- **Driver Stability:** **0 nvlddmkm driver events** recorded in Windows System Event Log across the entire 6.22-hour resumed run.

### 15.6 Test-Quarantine Compliance
- Throughout the entire 15-epoch training lifecycle, `results/dataset_splits/test_manifest.json` and `data/quarantine_test/` were **NEVER ACCESSED, OPENED, READ, ENUMERATED, OR HASHED**.
- Zero test data was touched. The sealed test quarantine boundary remains 100% intact.

### 15.7 Scientific Limitations & Selection-Exposure Disclosure
1. **Selection-Exposed Internal Validation:** All validation metrics reported in this section (including the best Validation Mean Dice of **`0.8594`**) were evaluated on the 155-case validation pool (`Val_eval` + `Val_cal`). Because model checkpoints were monitored and selected based on maximum validation performance on this pool, these metrics are formally classified as **selection-exposed internal validation metrics**.
2. **Epoch 15 Checkpoint Selection & Non-Convergence:** The model achieved its highest recorded validation mean Dice at Epoch 15 within the prescribed 15-epoch budget. This establishes only the maximum observed score within the evaluated interval. It does not establish whether training had plateaued, whether overfitting occurred at intermediate epochs, or whether additional training would improve performance.
3. **Single-Seed Statistical Interpretation & Stochastic Resumption:** Because the architecture was evaluated using a single configured random seed (seed 42), run-to-run variability and statistical significance cannot be estimated. Small numerical differences should therefore be interpreted descriptively rather than as robust performance differences. Furthermore, checkpoint recovery did not preserve the complete RNG state; the same configured seed does not establish an identical stochastic execution trajectory after resumption.
4. **Runtime Comparability & Heterogeneous Execution:** The total recorded runtime of 8.24 hours was heterogeneous: Epochs 1–5 ran under asynchronous execution (~24.3 min/epoch), while Epochs 6–15 were completed under synchronous execution (6.22 hours, averaging ~37.3 min/epoch). This cumulative runtime cannot be presented as a controlled architecture-speed benchmark against fully synchronous runs.
5. **Decoupled Calibration:** No temperature scaling or uncertainty threshold calibration was fitted during this training run. Post-hoc calibration on `Val_cal` ($T^*_{\text{unet}}$) remains a separate post-training work package under Stage C Work Package C2.

---

## 16. Final Experiment Decision Gate

# **`EXPERIMENT FULLY COMPLETED — ALL 15 EPOCHS VERIFIED`**

> **Final Experiment Summary:**
> The Stage C Phase 1 U-Net 2D training experiment on the authorized 721-case development cohort has successfully completed all 15 epochs. Checkpoint selection identified **Epoch 15** as the optimal checkpoint with a selection-exposed Validation Mean Dice of **`0.8594`** (ET: 0.9026, ED: 0.8662, NCR: 0.8094). Both `best_model.pt` and `final_model.pt` are verified, reloadable, and hashed. All 14 test suite regression checks pass cleanly, and the sealed test quarantine boundary remains 100% intact.
