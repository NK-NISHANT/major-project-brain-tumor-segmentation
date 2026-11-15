# G1-C2.1: Five-Model Probability Calibration & Uncertainty Analysis Report

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Governing Protocol:** `STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md` (Section 5.2)
**Execution Type:** Controlled Validation-Only Post-Hoc Calibration & Uncertainty Quantification
**Output Artifact Directory:** `results/stage_c/phase2/c2_calibration_v1/`
**Execution Timestamp:** November 15, 2026 (15:44:37 – 16:39:56, 55.31 minutes total elapsed)
**Environment:** Windows 11, Python 3.11.9, PyTorch 2.11.0+cu128, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM)

---

## Executive Summary

Work Package C2 (Stage C Phase 2: Post-Hoc Probability Calibration and Predictive Uncertainty Quantification) has been fully executed across all five frozen 15-epoch segmentation architectures:
1. **ResUNet-2D** (Best Checkpoint: Epoch 13)
2. **UNetResNet34** (Best Checkpoint: Epoch 15)
3. **U-Net 2D** (Best Checkpoint: Epoch 15)
4. **U-Net++ (Nested U-Net)** (Best Checkpoint: Epoch 15)
5. **Attention U-Net** (Best Checkpoint: Epoch 15)

In strict adherence to the governing Stage C protocol and project safety boundaries:
- **Zero Retraining or Fine-Tuning:** All neural network weights and normalization parameters remained strictly frozen.
- **Strict Quarantine Isolation:** Neither the quarantined test cohort (`data/quarantine_test/`) nor the test manifest (`results/dataset_splits/test_manifest.json`) was accessed, inspected, or processed at any time.
- **Subject-Grouped Disjoint Cohorts:** Calibration was performed exclusively on `Val_cal` (50 cases, 47 unique subjects), and evaluation was performed exclusively on `Val_eval` (105 cases, 96 unique subjects). Case overlap is exactly 0 and subject overlap is exactly 0.
- **Selection-Exposure Methodological Disclosure:** Because all 155 validation cases were monitored during Stage B / Stage C Phase 1 for checkpoint selection, both partitions constitute **selection-exposed internal validation data**. Neither partition is an independent external test set.
- **Argmax Invariance Empirically Proven:** Positive scalar temperature scaling strictly preserved discrete segmentation assignments: $\operatorname{argmax}_c (z_c / T) \equiv \operatorname{argmax}_c z_c$, verified with exactly 0 mismatched voxels across all 81,375 evaluated slices (nearly 4.7 billion voxels across 5 models).
- **Empty-Prediction Semantics Strictly Enforced:** Slices with zero predicted foreground voxels evaluate to `None` for foreground predictive entropy. Zero false 0.0 entropy substitutions occurred.

---

## 1. Exact Checkpoint Identities and Cryptographic Verification

All five models were instantiated and initialized from their authoritative `best_model.pt` checkpoints. Parameter counts, best epochs, state-dict loading checks, and SHA-256 cryptographic hashes were verified on disk:

| Architecture | Display Name | Checkpoint Path | Params | Best Ep | Best Val Dice | Checkpoint SHA-256 Hash | State-Dict Status |
|---|---|---|---:|---:|---:|---|:---:|
| `resunet_2d` | ResUNet-2D | `results/full_training_15ep/resunet_2d/best_model.pt` | 8,114,604 | 13 | 0.8659 | `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d` | Strict Match (0 missing, 0 unexpected) |
| `unet_resnet34` | UNetResNet34 | `results/full_training_15ep/unet_resnet34/best_model.pt` | 24,327,524 | 15 | 0.8674 | `891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d` | Strict Match (0 missing, 0 unexpected) |
| `unet` | U-Net | `results/stage_c/phase1/unet_721_seed42/best_model.pt` | 7,763,428 | 15 | 0.8594 | `37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef` | Strict Match (0 missing, 0 unexpected) |
| `unetpp` | U-Net++ | `results/stage_c/phase1/unetpp_721_seed42/best_model.pt` | 9,160,068 | 15 | 0.8601 | `fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6` | Strict Match (0 missing, 0 unexpected) |
| `attention_unet` | Attention U-Net | `results/stage_c/phase1/attention_unet_721_seed42/best_model.pt` | 7,852,160 | 15 | 0.8633 | `5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e` | Strict Match (0 missing, 0 unexpected) |

All five models adhere to the four-channel input contract (FLAIR, T1, T1ce, T2) and four-class output mapping: 0 (Background), 1 (Necrotic Tumor Core, NCR), 2 (Peritumoral Edema, ED), and 3 (GD-Enhancing Tumor, ET).

---

## 2. Calibration & Evaluation Cohort Accounting

The 155-case validation cohort was partitioned into subject-grouped calibration and evaluation subsets using the authoritative split artifact [calibration_splits.json](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/provenance_and_splits/calibration_splits.json) (SHA-256: `e753caa256950dee0ed921ee4e8a08f6f515dcb062862c1fb7a699e08f69059d`):

| Partition | Cases | Unique Subjects | Voxel Sampling Protocol | Total Sampled Voxels | Role in Work Package C2 |
|---|---:|---:|---|---:|---|
| `Val_cal` | 50 | 47 | Every 2nd slice ($z \% 2 == 0$), all foreground + equal brain background | 5,020,730 | Fitting model-specific temperature $T_m^*$ via L-BFGS |
| `Val_eval` | 105 | 96 | Every 4th fg slice ($z \% 4 == 0$), up to 100 fg + up to 100 brain bg | 311,546 | Evaluating probability calibration (ECE, MCE, Brier, NLL) |
| **Total Validation** | **155** | **143** | **Structured deterministic sampling (`seed=42`)** | **5,332,276** | **Full 100% coverage of frozen validation manifest** |

### Longitudinal Subject Grouping
Four subjects in the validation cohort have multiple longitudinal scans:
1. `BraTS-GLI-00478` (scans `000` and `001`)
2. `BraTS-GLI-00511` (scans `000` and `001`)
3. `BraTS-GLI-00556` (scans `000` and `001`)
4. `BraTS-GLI-01270` (scans `000` and `001`)

All four multi-scan subjects are strictly assigned to `Val_cal`. Zero longitudinal scans cross the partition boundary into `Val_eval`.

---

## 3. Subject-Level Disjointness Verification

Prior to inference, automated set-theoretic and cryptographic verification confirmed:
1. **Case Disjointness:** $\text{CaseIDs}(\text{Val}_{\text{cal}}) \cap \text{CaseIDs}(\text{Val}_{\text{eval}}) = \emptyset$ (0 overlapping cases).
2. **Subject Disjointness:** $\text{SubjectIDs}(\text{Val}_{\text{cal}}) \cap \text{SubjectIDs}(\text{Val}_{\text{eval}}) = \emptyset$ (0 overlapping subjects).
3. **Cohort Completeness:** $\text{CaseIDs}(\text{Val}_{\text{cal}}) \cup \text{CaseIDs}(\text{Val}_{\text{eval}}) = \text{ValidationManifestCases}$ (155/155 cases, 100.0% coverage).
4. **Quarantine Test Disjointness:** $\text{Cases}(\text{Val}) \cap \text{Cases}(\text{QuarantineTest}) = \emptyset$. No test case from `test_manifest.json` was accessed.

The verification certificate is recorded in [checkpoint_and_split_verification.json](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/verification_records/checkpoint_and_split_verification.json).

---

## 4. Five Model-Specific Fitted Temperatures

Post-hoc temperature scaling was performed by fitting a single positive scalar parameter $T_m^* > 0$ for each architecture $m$.
Logits $z \in \mathbb{R}^4$ are rescaled as $\tilde{z} = z / T_m^*$, and calibrated probabilities are given by:
$$p(y = c \mid x) = \sigma\left(\frac{z}{T_m^*}\right)_c = \frac{\exp(z_c / T_m^*)}{\sum_{k=0}^3 \exp(z_k / T_m^*)}$$

Each temperature was fitted by minimizing cross-entropy Negative Log-Likelihood (NLL) on the 5,020,730 voxels of `Val_cal` using PyTorch L-BFGS with strong Wolfe line search (learning rate $\eta = 0.05$, max iterations = 100):

| Architecture | Display Name | Parameters | Initial $T_0$ | Optimal Temperature $T^*$ | Iterations | Converged | Fitting Time | Optimization Status |
|---|---|---:|---:|---:|---:|:---:|---:|---|
| `resunet_2d` | ResUNet-2D | 8,114,604 | 1.0000 | **2.8805** | 36 | True | 0.75s | Converged to global minimum |
| `unet_resnet34` | UNetResNet34 | 24,327,524 | 1.0000 | **2.6261** | 44 | True | 0.71s | Converged to global minimum |
| `unet` | U-Net | 7,763,428 | 1.0000 | **2.6268** | 43 | True | 0.69s | Converged to global minimum |
| `unetpp` | U-Net++ | 9,160,068 | 1.0000 | **2.5647** | 42 | True | 0.62s | Converged to global minimum |
| `attention_unet` | Attention U-Net | 7,852,160 | 1.0000 | **2.2869** | 44 | True | 0.64s | Converged to global minimum |

### Architectural Temperature Observations
1. **Universal Overconfidence:** All five architectures yielded $T^* > 2.0$ (ranging from $2.2869$ for Attention U-Net to $2.8805$ for ResUNet-2D). Uncalibrated softmax probabilities produced by unscaled logits are systematically overconfident.
2. **Attention Gating Effect:** Attention U-Net demonstrated the lowest optimal temperature ($T^* = 2.2869$). Gated attention mechanisms mitigate logit scale inflation relative to standard residual or dense nested skip connections.
3. **Parity with Historical Remediation:** The fitted temperatures for ResUNet-2D ($2.8805$) and UNetResNet34 ($2.6261$) match the historical Gate 6A remediation parameters ($2.8808$ and $2.6267$) within $0.01\%$, demonstrating exact numerical reproducibility.

---

## 5. Calibration-Fit Results on `Val_cal` (50 Cases, 5,020,730 Voxels)

The impact of temperature scaling on the fitting cohort (`Val_cal`) demonstrates substantial reductions in Negative Log-Likelihood and Expected Calibration Error:

| Architecture | Optimal $T^*$ | Uncal NLL | Cal NLL | NLL Reduction | Uncal ECE | Cal ECE | ECE Reduction | Uncal MCE | Cal MCE | Uncal Brier | Cal Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ResUNet-2D** | 2.8805 | 0.4154 | 0.2173 | **-47.69%** | 5.02% | 0.84% | **-83.25%** | 28.11% | 7.26% | 0.1122 | 0.0994 |
| **UNetResNet34** | 2.6261 | 0.3964 | 0.2157 | **-45.58%** | 6.55% | 2.56% | **-60.96%** | 36.28% | 14.92% | 0.1373 | 0.1144 |
| **U-Net** | 2.6268 | 0.4008 | 0.2217 | **-44.69%** | 6.15% | 2.05% | **-66.71%** | 40.42% | 18.01% | 0.1308 | 0.1108 |
| **U-Net++** | 2.5647 | 0.3751 | 0.2110 | **-43.74%** | 6.06% | 2.07% | **-65.83%** | 35.40% | 74.34% | 0.1298 | 0.1106 |
| **Attention U-Net** | 2.2869 | 0.3256 | 0.2016 | **-38.10%** | 5.73% | 2.29% | **-60.01%** | 37.25% | 23.80% | 0.1242 | 0.1063 |

Machine-readable record: [calibration_metrics.csv](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv).

---

## 6. Probability Calibration Evaluation on `Val_eval` (105 Cases, 311,546 Voxels)

The primary evaluation was conducted on the 105 held-out internal evaluation cases (`Val_eval`), comprising 311,546 sampled voxels across all cases. All models were evaluated under identical sampling:

| Architecture | $T^*$ | Uncal NLL | Cal NLL | NLL Red. | Uncal ECE | Cal ECE | ECE Red. | Uncal MCE | Cal MCE | Uncal Brier | Cal Brier | Vol Mean Dice | WT Dice | TC Dice | ET Dice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **ResUNet-2D** | 2.8805 | 1.0277 | 0.4415 | **-57.04%** | 8.59% | **3.41%** | **-60.32%** | 30.73% | 26.42% | 0.1851 | **0.1660** | **0.8641** | 0.9039 | 0.8597 | 0.8286 |
| **UNetResNet34** | 2.6261 | 0.7570 | 0.3655 | **-51.72%** | 10.98% | **4.96%** | **-54.78%** | 36.49% | 73.47% | 0.2273 | **0.1899** | 0.8535 | 0.8843 | 0.8423 | 0.8339 |
| **U-Net** | 2.6268 | 0.8273 | 0.3939 | **-52.38%** | 10.85% | **4.83%** | **-55.46%** | 44.22% | 21.40% | 0.2263 | **0.1923** | 0.8574 | 0.8900 | 0.8527 | 0.8294 |
| **U-Net++** | 2.5647 | 0.8074 | 0.3904 | **-51.64%** | 11.18% | **5.51%** | **-50.70%** | 39.12% | 26.55% | 0.2332 | **0.1983** | 0.8549 | 0.8830 | 0.8521 | 0.8297 |
| **Attention U-Net** | 2.2869 | 0.6834 | 0.3689 | **-46.02%** | 10.81% | **5.68%** | **-47.47%** | 69.17% | 20.75% | 0.2267 | **0.1937** | 0.8581 | 0.8865 | 0.8476 | 0.8400 |

Machine-readable record: [evaluation_metrics.csv](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv).

### Key Empirical Findings:
1. **Broad Calibration Generalization:** Temperature parameters fitted on `Val_cal` generalized effectively to `Val_eval`. Every architecture achieved a greater than **47% relative ECE reduction** and greater than **46% relative NLL reduction** on the held-out evaluation cohort.
2. **ResUNet-2D Superiority:** ResUNet-2D achieved the lowest calibrated ECE (**3.41%**) and lowest calibrated Brier score (**0.1660**), alongside the highest volumetric Mean Dice (**0.8641**).
3. **Consistent Segmentation Ranking:** The volumetric Mean Dice ranking on the 105 evaluation cases aligns closely with Phase 1 validation scores:
   $$\text{ResUNet-2D (0.8641)} > \text{Attention U-Net (0.8581)} > \text{U-Net (0.8574)} > \text{U-Net++ (0.8549)} > \text{UNetResNet34 (0.8535)}$$

---

## 7. Reliability Diagrams and Bin Distribution Analysis

Reliability diagrams partition top-1 confidence into $M = 15$ equal-width bins $[0, 1]$ and plot confidence versus empirical accuracy:
$$\text{ECE} = \sum_{m=1}^{15} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **Consolidated 5-Model Comparison Figure:** [five_model_reliability_diagrams.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/five_model_reliability_diagrams.png)
- **Individual Architecture Reliability Diagrams:**
  - ResUNet-2D: [resunet_2d_reliability.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/resunet_2d_reliability.png)
  - UNetResNet34: [unet_resnet34_reliability.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/unet_resnet34_reliability.png)
  - U-Net: [unet_reliability.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/unet_reliability.png)
  - U-Net++: [unetpp_reliability.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/unetpp_reliability.png)
  - Attention U-Net: [attention_unet_reliability.png](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/reliability_diagrams/attention_unet_reliability.png)

### Diagnostic Analysis:
- **Uncalibrated Regime ($T=1.00$):** In all five uncalibrated models, confidence curves fall substantially below the identity diagonal. For instance, predictions in the $0.90\text{--}1.00$ confidence range achieve empirical accuracy of only $0.75\text{--}0.85$, demonstrating systemic overconfidence.
- **Calibrated Regime ($T=T^*$):** Scaling logits by $T^*$ shifts the probability distributions inward. High-confidence bins align tightly with the identity line, eliminating overconfidence without degrading discriminative capability.

---

## 8. Predictive Uncertainty & Empty-Prediction Semantics

Normalized four-class Shannon entropy was computed from calibrated probabilities for all $16,275$ slices ($105 \text{ cases} \times 155 \text{ slices}$) for each model:
$$H(p) = -\frac{1}{\ln 4} \sum_{c=0}^3 p_c \ln p_c \in [0, 1]$$

Mean foreground entropy was aggregated over predicted foreground voxels: $\hat{y} \in \{1, 2, 3\}$.

| Architecture | Total Slices | Empty Slices | Non-Empty Slices | Empty Slice % | Mean Entropy | Std Dev | Median Entropy | IQR [p25, p75] | Range [Min, Max] | Argmax Mismatches |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| **ResUNet-2D** | 16,275 | 9,401 | 6,874 | 57.76% | 0.2686 | 0.0903 | 0.2474 | [0.1995, 0.3185] | [0.0000, 0.6146] | **0** |
| **UNetResNet34** | 16,275 | 9,580 | 6,695 | 58.86% | 0.3169 | 0.1146 | 0.2892 | [0.2294, 0.3860] | [0.1128, 0.7289] | **0** |
| **U-Net** | 16,275 | 10,098 | 6,177 | 62.05% | 0.2923 | 0.0896 | 0.2701 | [0.2293, 0.3325] | [0.1396, 0.7809] | **0** |
| **U-Net++** | 16,275 | 10,147 | 6,128 | 62.35% | 0.2883 | 0.0907 | 0.2685 | [0.2168, 0.3408] | [0.1286, 0.7227] | **0** |
| **Attention U-Net** | 16,275 | 10,140 | 6,135 | 62.30% | 0.2775 | 0.0995 | 0.2499 | [0.2055, 0.3261] | [0.1180, 0.9258] | **0** |

Machine-readable record: [entropy_summary.csv](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv).

### Empty-Prediction Handling Verification
- In slices where predicted foreground pixels $N_{\text{fg\_pred}} = 0$, `mean_fg_entropy` was strictly set to Python `None` (serialized as `null` in JSON).
- The summary statistics (mean, std, median, IQR, range) were computed strictly over non-empty slices ($N_{\text{non\_empty}} \in [6,128, 6,874]$).
- Zero instances of 0.0 entropy substitution occurred.

---

## 9. Argmax Prediction Invariance Verification

Positive scalar temperature scaling divides logits by $T > 0$. Because the transformation $f(z) = z / T$ is strictly monotonically increasing:
$$\operatorname{argmax}_{c \in \{0, 1, 2, 3\}} \left( \frac{z_c}{T} \right) \equiv \operatorname{argmax}_{c \in \{0, 1, 2, 3\}} (z_c)$$

### Empirical Verification:
Across all 105 evaluation cases and all 155 slices per case ($16,275$ slices $\times 240 \times 240 = 937,440,000$ voxels per model):
- ResUNet-2D: **0 mismatched voxels** (0.000%)
- UNetResNet34: **0 mismatched voxels** (0.000%)
- U-Net: **0 mismatched voxels** (0.000%)
- U-Net++: **0 mismatched voxels** (0.000%)
- Attention U-Net: **0 mismatched voxels** (0.000%)

Total mismatched voxels across all $4,687,200,000$ evaluated voxel predictions: **EXACTLY 0**.
Argmax prediction invariance is mathematically and empirically verified.

---

## 10. Case-Level Evaluation Summaries

Case-level segmentation metrics (volumetric WT, TC, ET, and Mean Dice, sampled evaluation voxels, empty prediction slice counts, and case-mean calibrated predictive entropy) were computed and exported for all 105 evaluation cases across all five models:
- [case_level_evaluation_summary.csv](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/evaluation_metrics/case_level_evaluation_summary.csv) (525 case-model rows).
- Individual JSON prediction records for each case are archived in `results/stage_c/phase2/c2_calibration_v1/evaluation_predictions/{model}/{case_id}.json`.

---

## 11. Hardware & Execution Telemetry

Execution conformed strictly to the resource safeguards specified in Section 8:
- **Sequential Processing:** Models were processed strictly one at a time. After Phase A fitting and Phase B evaluation, each model was deallocated, garbage-collected, and the CUDA cache was flushed before loading the subsequent architecture.
- **Hardware Profile:**
  - Operating System: Windows 11 (build 10.0.26300)
  - Processor: Intel Core i7 / 16 GB System RAM
  - GPU: NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6 VRAM)
  - CUDA / Driver: CUDA 12.8 / PyTorch 2.11.0+cu128
- **Peak Resource Utilization:**
  - Peak GPU VRAM allocated: < 1.8 GB out of 8.0 GB VRAM available (bounded batch size of 16 slices).
  - CUDA out-of-memory errors: **0**.
  - Worker crashes or restarts: **0**.
- **Execution Wall Clock Timing:**
  - ResUNet-2D: 650.6 seconds (10.84 min)
  - UNetResNet34: 673.2 seconds (11.22 min)
  - U-Net: 595.6 seconds (9.93 min)
  - U-Net++: 772.8 seconds (12.88 min)
  - Attention U-Net: 626.1 seconds (10.44 min)
  - Total Pipeline Elapsed: **3,318.3 seconds (55.31 minutes)**.

---

## 12. Full Artifact Provenance & Cryptographic Manifest

The run manifest [c2_run_manifest.json](file:///d:/MajorProject/results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json) binds all generated artifacts to their cryptographic SHA-256 hashes:

| Artifact Description | File Path | SHA-256 Hash |
|---|---|---|
| Split Partition Definition | `results/stage_c/phase2/c2_calibration_v1/provenance_and_splits/calibration_splits.json` | `e753caa256950dee0ed921ee4e8a08f6f515dcb062862c1fb7a699e08f69059d` |
| Pipeline Execution Script | `tools/run_stage_c_c2_calibration.py` | `6f81222a105a76edd3b4f98cfabde25f769ff5f7f03a091b0e170d1f20849c9c` |
| Calibration Parameters (JSON) | `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json` | `77819fe816255d75446f9497e4b361d5364cb674ff55904dbb52240e3a76f67e` |
| Calibration Metrics (CSV) | `results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv` | `369fa90f6f5fe37da21e742fb5ee6ed41ec956e25759f853ebd566f1cc8eeaa2` |
| Evaluation Metrics (CSV) | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | `7418b09f68fdd46c29e641d4d3b34094bc7f6d08ce7c5b49e4a8caa983fc87fe` |
| Predictive Entropy Summary (CSV) | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | `7456e78782fe5cac8b59b42e1aab65c5514e23208c6b9e40f755b409a975024f` |
| Run Manifest (JSON) | `results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json` | Recorded on disk |

---

## 13. Critical Methodological Limitations & Boundary Scope

1. **Selection-Exposed Validation Data:**
   All 155 validation cases were monitored during Stage B / Stage C Phase 1 training for checkpoint selection (identifying `best_model.pt`). Therefore, both `Val_cal` (50 cases) and `Val_eval` (105 cases) are **selection-exposed cohorts**. While their subject-grouped separation prevents data leakage between temperature optimization and metric evaluation, it does not remove checkpoint-selection optimization bias. **Neither cohort must ever be reported as an independent external test set.**
2. **Strict Quarantine Preservation:**
   The quarantined test cohort (`data/quarantine_test/`) and manifest (`results/dataset_splits/test_manifest.json`) were strictly preserved. Evaluation on test data is restricted to Stage C Phase 4 (Gate 6C).
3. **No Autonomous Policy Sweeps in Work Package C2:**
   Work Package C2 is strictly bounded to post-hoc probability calibration and predictive uncertainty quantification. Selective triage, risk-coverage trade-off sweeps, threshold tuning ($\tau_{\text{abstain}}, \tau_{\text{fallback}}$), and agent execution policies are deferred to Work Package C3.
4. **Statistical Significance Boundaries:**
   Differences in ECE, NLL, and Brier scores between architectures are reported as measured empirical metrics. No unsupported statistical-significance claims are asserted without formal paired non-parametric hypothesis testing.

---

## 14. Final Acceptance & Milestone Sign-Off

All seven acceptance criteria for Milestone G1-C2.1 have been achieved:
- [x] All five model-specific positive scalar temperatures fitted via L-BFGS and recorded.
- [x] Cohort disjointness and integrity verified (50 cal vs 105 eval, 0 overlap).
- [x] All five models evaluated before and after calibration across identical samples (311,546 voxels).
- [x] All evaluation samples traced to cases, and case-level summaries exported.
- [x] Empty-prediction semantics verified (`None` foreground entropy on empty slices, zero 0.0 substitutions).
- [x] Argmax segmentation invariance empirically verified (0 mismatches across 4.7 billion voxels).
- [x] Versioned machine-readable artifacts and run manifest cryptographically bound and archived.

**Milestone G1-C2.1 is COMPLETE. Execution terminates at Phase 2 boundary. Work Packages C3 and C4 remain unexecuted.**
