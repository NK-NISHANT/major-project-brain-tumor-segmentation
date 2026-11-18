# Final Master Source of Truth (v1.2.4 Canonical Reference Specification)

**Document ID:** `AUDIT-RESUNET2D-P1.1.4-04`  
**Audit Phase:** Phase 1.1.4 — Final Register Scope, Slice Classification & Statistical Disclosure  
**Date:** November 2026  
**Status:** **AUTHORITATIVE, MATHEMATICALLY SEALED & FINAL SOURCE OF TRUTH**  
**Workspace:** `D:\MajorProject`

---

## 1. Document Authority & Classification Framework

This document represents the **definitive, corrected single source of truth** for all parameters, file paths, cryptographic digests, training milestones, segmentation metrics, calibration parameters, and Stage C3 selective prediction counts for **ResUNet-2D**.

It incorporates all verified findings across Phase 1.1, Phase 1.1.2, Phase 1.1.3, and Phase 1.1.4:
- Cryptographic checkpoint identity, exact parameter counts (8,114,604), 1-indexed epoch notation (Epoch 13 of 15), and native spatial dimensions ($240 \times 240$).
- Optimal Stage C2 temperature calibration parameters ($T^* = 2.880459$ for primary ResUNet-2D; $T^* = 2.626091$ for secondary UNetResNet34).
- Deterministic Stage C3 selective prediction accounting across all 16,275 validation slices (5,619 ACCEPT, 147 FALLBACK_SWITCH, 10,509 FLAG_UNRELIABLE).
- Canonical decomposition of the 9,451 flagged ground-truth-background slices into 8,264 context-suppressed TN, 704 secondary-empty TN (total TN: 8,968), and 483 high-entropy FP.
- Rigorous decomposition of the 1,058 flagged GT-positive predictions, distinguishing foreground presence detection (772 positive detections flagged as uncertain) from detection false negatives (116 primary misses, 170 dual misses).
- Precise statistical disclosure of the 3D volumetric estimand: **Case-weighted mean performance with subject-clustered bootstrap confidence intervals**.
- Complete sanitization of clinical claims, adhering strictly to retrospective research simulation and Software-as-a-Medical-Device (SaMD) communication guardrails.

### Classification Taxonomy:
- **`VERIFIED`**: Mathematically and programmatically cross-validated against raw repository code, files, and serialized state dictionaries with zero discrepancy.
- **`VERIFIED WITH DISCLOSURE`**: Empirically validated, accompanied by mandatory methodological, sampling, or cohort exposure disclosures.
- **`CONFLICTING (RESOLVED)`**: A metric where prior drafts contained conflicting numbers, now definitively resolved to the true value, with erroneous values permanently expunged.
- **`EXPUNGED`**: An unverified string placeholder, hallucination, or unsupported claim from prior drafts that is permanently declared invalid.

---

## 2. Checkpoint Identity, Architecture & Hardware Specifications

| Category / Property | Canonical Value | Verification Source Artifact / File | Audit Classification |
| :--- | :--- | :--- | :---: |
| **Model Identity** | `resunet_2d` (ResUNet-2D) | [src/models/resunet.py](file:///D:/MajorProject/src/models/resunet.py) | **`VERIFIED`** |
| **Production Role** | Primary Champion / Stage C Reliability Agent | [src/agent/runner.py](file:///D:/MajorProject/src/agent/runner.py) | **`VERIFIED`** |
| **Canonical Checkpoint Path** | `results/full_training_15ep/resunet_2d/best_model.pt` | [best_model.pt](file:///D:/MajorProject/results/full_training_15ep/resunet_2d/best_model.pt) | **`VERIFIED`** |
| **Exact Checkpoint File Size** | `97,486,845 bytes` (92.97 MiB / 97.49 MB) | Windows Filesystem Attribute | **`VERIFIED`** |
| **SHA-256 Cryptographic Hash** | `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d` | Computed over full file bytes | **`VERIFIED`** |
| **Saved Training Epoch** | **Epoch 13 of 15** (1-indexed human-readable) | `checkpoint['epoch']` in `best_model.pt` | **`CONFLICTING (RESOLVED)`** |
| — *Resolved Epoch Index* | Training loop: `range(1, 16)`; Epoch 13 is the 13th epoch | `src/training/train_scaled.py` line 612 | **`VERIFIED`** |
| — *Expunged Epoch Claim* | "0-indexed: 14th epoch" | Erroneous draft assumption | **`EXPUNGED`** |
| **Input Spatial Dimensions** | **$240 \times 240$** axial slice ($4 \times 240 \times 240$) | `src/data/dataset.py` line 415 | **`CONFLICTING (RESOLVED)`** |
| — *Expunged Dimension Claim* | "$128 \times 128$" (confused with 128ch at level 2) | Erroneous draft assumption | **`EXPUNGED`** |
| **Input Modalities** | 4 channels: T1, T1CE, T2, FLAIR | Preprocessing pipeline | **`VERIFIED`** |
| **Output Classes** | 4: Background (0), NCR/NET (1), ED (2), ET (3) | Model classification head | **`VERIFIED`** |
| **Base Channels** | 32 channels (`base_channels = 32`) | `ResUNet2D` constructor | **`VERIFIED`** |
| **Normalization Layers** | Group Normalization (`num_groups = 8`) | Structural code in `resunet.py` | **`VERIFIED`** |
| **Activation Functions** | Rectified Linear Unit (`ReLU`) | Structural code in `resunet.py` | **`VERIFIED`** |
| **Total Parameter Count** | **`8,114,604`** | PyTorch tensor enumeration | **`CONFLICTING (RESOLVED)`** |
| — *Trainable Parameters* | **`8,114,604`** (100.0%) | `p.requires_grad == True` on all 91 tensors | **`VERIFIED`** |
| — *Non-Trainable Parameters* | **`0`** | Tensor enumeration | **`VERIFIED`** |
| — *Persistent Buffers* | **`0`** (`named_buffers()` is empty) | PyTorch buffer inspection | **`VERIFIED`** |
| — *Exact Component Sum* | **`8,114,604`** (sum of all 14 named submodules) | `AUDIT-RESUNET2D-P1.1-03` | **`VERIFIED`** |
| — *Expunged Component Sum* | `8,244,100` | Table error in v1.2 audit | **`EXPUNGED`** |
| — *Expunged Legacy Count* | `8,111,892` | Hallucinated draft placeholder | **`EXPUNGED`** |
| **Memory Footprint (Weights)** | **`32.46 MB`** (30.95 MiB / 32,458,416 bytes) | $8,114,604 \times 4\text{ bytes}$ | **`VERIFIED`** |
| **Full Checkpoint Payload** | Weights (32.46 MB) + Adam States (64.92 MB) $\approx$ 97.4 MB | PyTorch checkpoint state dict | **`VERIFIED`** |
| **Deprecated Checkpoint** | `weights/stage_c_15ep/resunet_best.pt` | Does not exist on disk | **`EXPUNGED`** |
| **Deprecated Checkpoint SHA** | `1d85be47...` | Hallucinated string placeholder | **`EXPUNGED`** |

---

## 3. Training Protocol & Milestone Evidence

| Training Property | Canonical Value | Source Artifact Path / Reference | Audit Classification |
| :--- | :--- | :--- | :---: |
| **Optimizer** | Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$) | `results/full_training_15ep/resunet_2d/config.json` | **`VERIFIED`** |
| **Base Learning Rate** | **`0.001`** ($1 \times 10^{-3}$) | `config.json` (`learning_rate: 0.001`) | **`CONFLICTING (RESOLVED)`** |
| — *Expunged Learning Rate* | `1e-4` | Erroneous draft text | **`EXPUNGED`** |
| **Weight Decay** | `0.0001` ($1 \times 10^{-4}$) | `config.json` (`weight_decay: 0.0001`) | **`VERIFIED`** |
| **Warmup Protocol** | Linear warmup over 2 epochs ($10^{-6} \rightarrow 0.001$) | `config.json` (`warmup_epochs: 2`) | **`VERIFIED`** |
| **Learning Rate Schedule** | Cosine Annealing over 13 epochs ($\eta_{\min} = 10^{-6}$) | `config.json` (`lr_scheduler: "cosine"`) | **`VERIFIED`** |
| **Loss Function** | Compound: $0.5 \cdot \mathcal{L}_{\text{Dice}} + 0.5 \cdot \mathcal{L}_{\text{CE}}$ | `config.json` (`loss: "compound"`) | **`VERIFIED`** |
| **Batch Size** | **`2`** (2D axial slices per iteration) | `config.json` (`batch_size: 2`) | **`CONFLICTING (RESOLVED)`** |
| — *Expunged Batch Size* | `16` | Erroneous draft text | **`EXPUNGED`** |
| **Total Training Epochs** | **`15`** | `config.json` (`epochs: 15`) | **`VERIFIED`** |
| **Total Training Duration** | **`8.0851 hours`** (29,106.36 seconds) | `results/full_training_15ep/resunet_2d/summary.json`| **`VERIFIED`** |
| **Best Model Metric (Ep 13)** | Foreground Macro Dice = **`0.867045`** | `summary.json` (`best_mean_dice`) | **`VERIFIED`** |
| — *Ep 13 Class Dice (NCR)* | `0.820963` | `checkpoint['metrics']` | **`VERIFIED`** |
| — *Ep 13 Class Dice (ED)* | `0.876465` | `checkpoint['metrics']` | **`VERIFIED`** |
| — *Ep 13 Class Dice (ET)* | `0.903708` | `checkpoint['metrics']` | **`VERIFIED`** |
| — *Ep 13 Background Dice* | `0.998738` | `checkpoint['metrics']` | **`VERIFIED`** |
| — *Ep 13 4-Class Mean Dice* | `0.899968` | Arithmetic mean across 4 classes | **`VERIFIED`** |
| — *Ep 13 Validation Loss* | `0.207431` | `checkpoint['best_val_loss']` | **`VERIFIED`** |
| **Lowest Val Loss (Ep 14)** | Validation Loss = **`0.206310`** (Dice: `0.861537`) | `summary.json` detailed history | **`VERIFIED`** |
| **Final Epoch (Ep 15)** | Val Loss = `0.208839`, Train Loss = `0.176589` | `summary.json` (`final_epoch_metrics`) | **`VERIFIED`** |

---

## 4. 3D Volumetric Segmentation Performance & Statistical Disclosure

*Evaluated on `Val_eval` ($N=105$ cases, $N=96$ unique subject clusters) via Non-Parametric Clustered Percentile Bootstrap ($B=1,000$, `seed=42`).*  
*Statistical Estimand Disclosure: All reported figures represent **case-weighted mean performance with subject-clustered bootstrap confidence intervals**. Point estimates reflect arithmetic means across all 105 cases. Bootstrap replicates draw 96 subject clusters with replacement and retain all associated cases.*

| Metric Dimension | Point Estimate (Case-Weighted Mean) | Median | Sample Std | Approved 95% Clustered Bootstrap CI | Source Artifact Path / Reference | Audit Classification |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **Whole Tumor (WT) Dice** | **`0.9039`** (`0.903862`) | — | — | **`[0.8765, 0.9241]`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| **Tumor Core (TC) Dice** | **`0.8597`** (`0.859712`) | — | — | **`[0.8076, 0.8986]`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| **Enhancing Tumor (ET) Dice**| **`0.8286`** (`0.828616`) | — | — | **`[0.7794, 0.8684]`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| **Volumetric Macro Dice** | **`0.8641`** (`0.864063`) | — | — | **`[0.8265, 0.8930]`** | Case macro bootstrap (`case_hd95_metrics.csv`)| **`VERIFIED WITH DISCLOSURE`** |
| **WT Bounded HD95 (mm)** | **`15.29 mm`** (`15.2928`) | `4.12 mm` | `22.28 mm` | **`[11.38, 20.05] mm`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| **TC Bounded HD95 (mm)** | **`8.98 mm`** (`8.9793`) | `3.16 mm` | `16.26 mm` | **`[6.08, 12.74] mm`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| **ET Bounded HD95 (mm)** | **`9.92 mm`** (`9.9242`) | `2.00 mm` | `38.16 mm` | **`[4.71, 18.66] mm`** | `results/hd95_evaluation/hd95_summary.json` | **`VERIFIED WITH DISCLOSURE`** |
| — *Mandatory Exposure Disclosure* | `Val_eval` is a selection-exposed internal retrospective validation cohort (exposed to checkpoint selection and policy ablation). External generalizability is not claimed. Formal TRIPOD/STARD compliance claims are expunged. |

---

## 5. Stage C Phase 2 Temperature Calibration Evidence

| Calibration Dimension | Primary Champion: ResUNet-2D | Secondary Fallback: UNetResNet34 | Verification Source Artifact | Audit Classification |
| :--- | :---: | :---: | :--- | :---: |
| **Architecture Name** | `resunet_2d` | `unet_resnet34` | `src/agent/models.py` | **`VERIFIED`** |
| **Parameters** | **`8,114,604`** | **`24,327,524`** | `calibration_parameters.json` | **`VERIFIED`** |
| — *Expunged Secondary Count* | — | `24,435,588` | Legacy blueprint placeholder | **`EXPUNGED`** |
| **Checkpoint SHA-256** | `81ef568df7dc47e3...` | `891d869a2369c395...` | `STAGE_C_EXPECTED_HASHES` | **`VERIFIED`** |
| **Fitted Optimal Temp ($T^*$)**| **`2.8804590702056885`** | **`2.6260907649993896`** | `calibration_parameters.json` | **`CONFLICTING (RESOLVED)`** |
| — *Expunged Secondary Temp* | — | `1.839846` | Erroneous v1.2 audit text | **`EXPUNGED`** |
| **Fitting Iterations (L-BFGS)**| **36 iterations** | **28 iterations** | `calibration_parameters.json` | **`VERIFIED`** |
| **Argmax Prediction Invariance**| **100% Invariant (0 shifts)** | **100% Invariant (0 shifts)** | Mathematical proof and empirical trace | **`VERIFIED`** |
| **Fitting Cohort (`Val_cal`)** | 50 cases, 47 subjects, 5,020,730 sampled voxels | `c2_run_manifest.json` | **`VERIFIED WITH DISCLOSURE`** |
| — *Uncalibrated ECE* | `0.050168` (5.02%) | `0.043612` (4.36%) | `calibration_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Calibrated ECE* | **`0.008402` (0.84%)** ($-83.25\%$) | **`0.008914` (0.89%)** ($-79.56\%$) | `calibration_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Uncalibrated NLL* | `0.415425` | `0.457812` | `calibration_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Calibrated NLL* | **`0.217311`** ($-47.69\%$) | **`0.201564`** ($-55.97\%$) | `calibration_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| **Evaluation Cohort (`Val_eval`)**| 105 cases, 96 subjects, 311,546 sampled voxels | `evaluation_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Uncalibrated ECE* | `0.085896` (8.59%) | `0.076812` (7.68%) | `evaluation_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Calibrated ECE* | **`0.034080` (3.41%)** ($-60.32\%$) | **`0.022915` (2.29%)** ($-70.17\%$) | `evaluation_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Uncalibrated NLL* | `1.027677` | `0.984120` | `evaluation_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Calibrated NLL* | **`0.441455`** ($-57.04\%$) | **`0.412890`** ($-58.04\%$) | `evaluation_metrics.csv` | **`VERIFIED WITH DISCLOSURE`** |
| — *Mandatory Sampling Disclosure* | Evaluated on stratified foreground/boundary voxel samples. Dense whole-volume evaluation would dilute metrics with non-brain air voxels. |

---

## 6. Stage C3 Action Accounting & Canonical Slice-Level Classification

*Evaluated across all 16,275 axial slices in `Val_eval` ($105$ cases $\times$ $155$ slices). Sourced from `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_action_trace.csv`.*

| Action Category / Subcategory | Total Slices | % Total (N=16,275) | Ground-Truth Positive ($GT > 0$) | Ground-Truth Background ($GT = 0$) | Exact Metric Value & Source Model Description | Audit Classification |
| :--- | :---: | :---: | :---: | :---: | :--- | :---: |
| **`ACCEPT`** | **`5,619`** | **34.53%** | **`5,437`** | **`182`** | **`0.785461`** (ResUNet primary Dice on accepted GT+) | **`VERIFIED`** |
| **`FALLBACK_SWITCH`** | **`147`** | **0.90%** | **`54`** | **`93`** | **`0.261235`** (UNetResNet34 selected Dice on fallback GT+) | **`VERIFIED WITH DISCLOSURE`** |
| — *Positive Overlap ($\text{Dice} > 0$)* | `37` | 0.23% | `37` | 0 | `0.381268` (UNetResNet34 selected Dice) | **`VERIFIED`** |
| — *Zero Overlap ($\text{Dice} = 0$)* | `17` | 0.10% | `17` | 0 | `0.000000` (UNetResNet34 selected Dice) | **`VERIFIED`** |
| **`FLAG_UNRELIABLE`** | **`10,509`** | **64.57%** | **`1,058`** | **`9,451`** | **`0.389146`** (ResUNet primary audit Dice on deferred GT+) | **`CONFLICTING (RESOLVED)`** |
| — *C1: Context-Suppressed TN* | **`8,264`** | 50.78% | 0 | **`8,264`** | `0.000000` (Primary empty; axial context suppressed) | **`CONFLICTING (RESOLVED)`** |
| — *C2: Secondary-Empty TN* | **`704`** | 4.33% | 0 | **`704`** | `0.000000` (Primary empty; secondary also empty) | **`CONFLICTING (RESOLVED)`** |
| — *C3: High-Entropy FP* | **`483`** | 2.97% | 0 | **`483`** | Spurious false alarms intercepted by entropy gate | **`CONFLICTING (RESOLVED)`** |
| — *C4: High-Entropy GT+ Flagged* | **`772`** | 4.74% | **`772`** | 0 | `0.531201` (Positive detection flagged as uncertain) | **`CONFLICTING (RESOLVED)`** |
| — *C5: Context-Suppressed Primary FN* | **`116`** | 0.71% | **`116`** | 0 | `0.000000` (Primary detection false negative) | **`CONFLICTING (RESOLVED)`** |
| — *C6: Secondary-Also-Empty Dual FN* | **`170`** | 1.04% | **`170`** | 0 | `0.000000` (Dual model detection false negative) | **`CONFLICTING (RESOLVED)`** |
| **Total Global Slices** | **`16,275`** | **100.00%** | **`6,549`** | **`9,726`** | — | **`VERIFIED`** |

### Definitive Canonical Reconciliation:
1. **Ground-Truth-Background Flagged Slices ($9,451$ slices):**
   - True Negatives ($P=0, GT=0$): Exactly **`8,968` slices** ($8,264$ context-suppressed + $704$ secondary-empty).
   - False Positives ($P>0, H \ge 0.35, GT=0$): Exactly **`483` slices** (spurious false alarms intercepted).
   - Total: $8,968 + 483 = \mathbf{9,451}$ slices.
2. **Ground-Truth-Positive Flagged Slices ($1,058$ slices):**
   - Positive Foreground Detections Flagged as Uncertain ($P>0, H \ge 0.35, GT>0$): Exactly **`772` slices** (mean uncalibrated primary Dice: `0.5312`).
   - Detection False Negatives ($P=0, GT>0$): Exactly **`286` slices** ($116$ context-suppressed misses + $170$ dual-model misses).
   - Total: $772 + 286 = \mathbf{1,058}$ slices.
3. **Register Scope:** Full 10,509-slice accounting is housed in [complete_flagged_slice_classification_register.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/complete_flagged_slice_classification_register.csv). The 1,473-row historical extract is preserved in [historical_discrepancy_extract_1473.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/historical_discrepancy_extract_1473.csv).

---

## 7. Risk-Coverage & Severe Failure Trade-Off (Denominator = 6,549 GT+ Slices)

| Performance Dimension | Metric / Count | Coverage (%) | Formula / Provenance | Audit Classification |
| :--- | :---: | :---: | :--- | :---: |
| **GT-Positive Slices Retained** | **`5,491` slices** | **83.84%** | $5,437 \;(\text{ACCEPT}) + 54 \;(\text{FALLBACK\_SWITCH})$ | **`VERIFIED`** |
| **GT-Positive Slices Flagged** | **`1,058` slices** | **16.16%** | $772 \;(\text{High } H) + 116 \;(\text{Context}) + 170 \;(\text{Sec Empty})$ | **`VERIFIED`** |
| **Combined Retained Mean Dice** | **`0.780306`** (0.7803) | 83.84% | $\frac{5,437 \times 0.785461 + 54 \times 0.261235}{5,491}$ | **`VERIFIED`** |
| **Flagged GT+ Primary Mean Dice** | **`0.389146`** (0.3891) | 16.16% | Mean primary Dice across 1,058 flagged GT+ slices (retrospective audit) | **`CONFLICTING (RESOLVED)`** |
| **Unfiltered Primary Baseline Failures**| **`1,343` slices** (**20.51%**) | 100.00% | Primary model on all 6,549 GT+ slices ($\text{Dice} < 0.50$) | **`VERIFIED`** |
| **Paired Baseline Failures (Same Subset)**| **`753` slices** (**13.71%**) | 83.84% | Primary model on same 5,491 retained slices ($\text{Dice} < 0.50$) | **`VERIFIED`** |
| **Stage C3 Selected Failures (Retained)** | **`742` slices** (**13.51%**) | 83.84% | Selected predictions on 5,491 retained slices ($\text{Dice} < 0.50$) | **`VERIFIED`** |
| **Observed Failure Difference** | **`601` fewer failures** | — | $1,343 - 742 = 601$ fewer failures entering autonomous retained pool | **`CONFLICTING (RESOLVED)`** |
| — *Mechanistic Reduction (Triage)* | 590 primary failures routed to human review workload | — | $1,343 - 753 = 590$ | **`VERIFIED`** |
| — *Mechanistic Reduction (Rescue)* | 11 failures rescued $\ge 0.50$ by secondary model | — | $753 - 742 = 11$ | **`VERIFIED`** |
| — *Mandatory Triage Disclosure* | This represents a workflow triage trade-off: 590 failures were routed to a simulated review-routing workload, not eliminated. |

---

## 8. Simulation & Workload Governance Statements

1. **Retrospective Research Simulation:**  
   All reported Stage C3 selective prediction results derive from **offline retrospective policy evaluation** on stored prediction traces. No clinical deployment occurred.
2. **Review-Routing Workload Interpretation:**  
   The 10,509 flagged slices represent a **simulated review-routing workload** for designated human reviewers in a human-in-the-loop triage architecture. It is not claimed that clinical radiologists inspected these slices during this research.
3. **Reproducibility & Provenance Standard:**  
   *Checkpoint paths and cryptographic SHA-256 digests match the recorded Stage C execution metadata, and the stored action trace reconciles mathematically. No fresh inference replay was performed during this audit.*
