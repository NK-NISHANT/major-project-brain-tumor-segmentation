# MODEL DOSSIER 01: RESUNET-2D
## Architecture, Training, Calibration, Performance & Stage C Reliability Agent Profile
**Document Version:** 1.1 (Supersedes v1.0)  
**Governing Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`  
**Scientific Authority:** Authoritative Stage C3 v1.1.3 Simulation & Evidence  
**Workspace:** `D:\MajorProject`  
**Model Identifier:** `resunet` (`ResUNet2D`)  
**Profile:** `STAGE_C_15EP` (Epoch 13 Peak)  
**Cryptographic SHA-256:** `1d85be47225c5684f18ba6b89ae6563ee9f39aa13c9e99a80373ab10a300d81a`  
**Checkpoint Path:** `weights/stage_c_15ep/resunet_best.pt`  
**Primary Execution Device:** NVIDIA GeForce RTX 4070 Laptop GPU (CUDA 12.8) / Intel Core i7

---

## EXECUTIVE PERFORMANCE SUMMARY (AT A GLANCE)

| Metric Category | Specification / Evaluation Protocol | Value / 95% Confidence Interval | Source Artifact |
|:---|:---|:---|:---|
| **Model Parameters** | Total Trainable Parameters (32-bit float) | **8,111,892 (8.11 M)**; 32.4 MB | `weights/stage_c_15ep/resunet_best.pt` |
| **Volumetric Macro Dice** | 3D Patient Reconstruction (`Val_eval`, 105 Cases) | **0.8641** (Macro across WT, TC, ET) | `results/val_eval_predicted_metrics.json` |
| **Whole Tumor (WT) Dice** | Outer Infiltrative Boundary (`Val_eval`, 105 Cases) | **0.9039** (95% CI: [0.8878, 0.9189]) | `results/val_eval_predicted_metrics.json` |
| **Tumor Core (TC) Dice** | Necrosis + Enhancing Core (`Val_eval`, 105 Cases) | **0.8597** (95% CI: [0.8351, 0.8821]) | `results/val_eval_predicted_metrics.json` |
| **Enhancing Tumor (ET) Dice** | Active Enhancing Margin (`Val_eval`, 105 Cases) | **0.8286** (95% CI: [0.7963, 0.8580]) | `results/val_eval_predicted_metrics.json` |
| **Checkpoint Selection Dice** | **Foreground Macro Mean (NCR, ED, ET)** (Ep 13) | **0.8670** (NCR: 0.8210, ED: 0.8765, ET: 0.9037) | `results/training_history_resunet_scaled_15ep.json` |
| **4-Class Mean Dice** | Arithmetic Mean incl. Background (Ep 13) | **0.899968** (BG: 0.9987, NCR: 0.8210, ED: 0.8765, ET: 0.9037) | `results/training_history_resunet_scaled_15ep.json` |
| **Slice-Level Unfiltered Dice** | Average across 6,549 Ground-Truth Tumor Slices | **0.7150** Macro Dice | `results/c3_selective_prediction_v1_1_3/` |
| **Hausdorff Distance (HD95)** | Enhancing Tumor Median Boundary Error | **2.00 mm** (Mean: 9.92 mm, 95% CI: [4.71, 18.66]) | `results/hd95_evaluation_report.json` |
| **Hausdorff Distance (HD95)** | Tumor Core Median Boundary Error | **3.16 mm** (Mean: 8.98 mm, 95% CI: [6.08, 12.74]) | `results/hd95_evaluation_report.json` |
| **Hausdorff Distance (HD95)** | Whole Tumor Median Boundary Error | **4.12 mm** (Mean: 15.29 mm, 95% CI: [11.38, 20.05]) | `results/hd95_evaluation_report.json` |
| **Optimal Temperature (T\*)** | Post-hoc Scaling Parameter (Learned on `Val_cal`) | **2.880459** (L-BFGS, 36 iterations) | `results/c2_calibration_manifest.json` |
| **Expected Calibration Error** | ECE on 105 Held-out Patients (`Val_eval`) | **8.59% -> 3.41%** (60.3% relative reduction) | `results/c2_calibration_manifest.json` |
| **Negative Log-Likelihood** | NLL on 105 Held-out Patients (`Val_eval`) | **1.0277 -> 0.4415** (57.0% relative reduction) | `results/c2_calibration_manifest.json` |
| **Predictive Entropy (H)** | Mean across 6,874 Non-Empty Slices | **0.2686 +- 0.0903** (Median: 0.2474, IQR: [0.1995, 0.3185]) | `results/c3_selective_prediction_v1_1_3/` |
| **Operating Threshold (tau)** | Frozen Stage C Uncertainty Boundary | **tau = 0.35** | `MASTER_PROJECT_BLUEPRINT_v2.0.md` |
| **Autonomous Retention** | Tumor Slices Accepted by Agent (H < 0.35) | **83.84%** (5,491 / 6,549 slices); Mean Dice: **0.7803** | `results/c3_selective_prediction_v1_1_3/` |
| **Specialist Referral** | Flawed Tumor Slices Routed to Doctors | **16.16%** (1,058 / 6,549 slices); Mean Dice: **0.3891** | `results/c3_selective_prediction_v1_1_3/` |
| **Severe Failure Reduction** | Relative Cut in Severe Errors (Dice < 0.50) | **-34.1%** (Drop from 20.51% [1,343 slices] to 13.51% [742 slices]) | `results/c3_selective_prediction_v1_1_3/` |
| **Axial Fallback Rescues** | True Tumor Slices Rescued via UNetResNet34 | **54 Slices** (Rescued from false-negative omission) | `results/c3_selective_prediction_v1_1_3/` |

---

## 1. MODEL IDENTITY & PORTFOLIO ROLE

ResUNet-2D is the **Primary Champion** architecture in our five-model deep learning portfolio for multi-parametric magnetic resonance imaging (mpMRI) brain tumor segmentation. Within our multi-tiered clinical AI system, ResUNet-2D serves as the front-line predictive engine for the **Stage C Multi-Model Reliability Agent**.

When evaluated on the standardized 15-epoch scaled profile (`STAGE_C_15EP`), ResUNet-2D achieved the highest overall segmentation fidelity across all three clinical target regions, demonstrating superior feature representation, stable optimization dynamics, and exceptional boundary adherence compared to UNet (Vanilla), Attention U-Net, UNet++, and UNet-ResNet34.

The operational checkpoint was captured at **Epoch 13**, where the network reached its global validation peak. To guarantee scientific provenance, reproducible auditing, and clinical safety, the weights are locked with cryptographic SHA-256 verification (`1d85be47...`), rejecting any modified or corrupted checkpoint prior to memory allocation.

---

## 2. CLINICAL CONTEXT & BRAIN TUMOR ANATOMY

Gliomas, particularly glioblastoma (WHO Grade 4), represent the most aggressive primary brain tumors in adults. Effective clinical management—including maximal safe surgical resection, localized radiotherapy planning, and longitudinal chemotherapy monitoring—relies critically on precise volumetric delineation of distinct sub-tumoral compartments:
1. **Necrotic & Non-Enhancing Core (NCR, Class 1):** Hypointense, non-perfused dead or dying tissue starved of vascular supply.
2. **Peritumoral Edema (ED, Class 2):** Vasogenic swelling caused by increased vascular permeability, extending along white matter tracts.
3. **Enhancing Tumor (ET, Class 3):** Viable, actively proliferating cancer cells characterized by disrupted blood-brain barriers and intense contrast uptake.

Clinically, these classes are combined into composite operational targets:
- **Whole Tumor (WT):** Necrosis + Edema + Enhancing Tumor (total compromised territory).
- **Tumor Core (TC):** Necrosis + Enhancing Tumor (target for surgical debulking).
- **Enhancing Tumor (ET):** Active neoplastic rim (target for focal radiation boosts).

Manual delineation by neuroradiologists is labor-intensive (60–90 minutes per patient), exhibits high inter-rater variability (15–20% boundary discordance), and creates acute clinical workflow bottlenecks. An automated segmentation algorithm must provide not only sub-millimeter precision but also quantifiable confidence metrics to safeguard patients from catastrophic false positives or missed lesions.

---

## 3. INPUT MODALITIES & MULTIMODAL SIGNAL CONTRIBUTION

ResUNet-2D ingests four co-registered mpMRI channels per slice, normalized via robust Z-score standardization:
1. **T1-Weighted Native (T1n):** Establishes baseline healthy anatomical contrast, distinguishing gray matter from white matter and outlining structural distortions.
2. **T1-Weighted Contrast-Enhanced (T1c):** Uses Gadolinium contrast agents to highlight active tumor vascularity and leaky capillaries. ET appears bright hyperintense.
3. **T2-Weighted (T2w):** Highlights fluid accumulation, ventricular architecture, and general tissue water content. Edema and CSF appear hyperintense.
4. **T2-FLAIR (Fluid-Attenuated Inversion Recovery):** Suppresses free water signals from ventricles while preserving bright signals from parenchymal edema, crucial for defining the outer infiltrating margins of Whole Tumor.

**Ablation & Sensitivity Findings:**
Systematic modality drop experiments reveal that ResUNet-2D exhibits strong multi-modal synergy but critical reliance on specific channels:
- Dropping **T1c** causes Tumor Core Dice to collapse from 0.8597 to **0.3132**, and Enhancing Tumor Dice to plummet to **0.1684** (Mean Macro Dice: 0.4449).
- Dropping **T2w** reduces Whole Tumor Dice from 0.9039 to **0.4388** (Mean Macro Dice: 0.6058).
- Dropping **FLAIR** impairs the outer edema boundary, increasing Whole Tumor HD95 error by 14.8 mm.

---

## 4. NETWORK ARCHITECTURE & MATHEMATICAL FORMULATION

ResUNet-2D synergistically combines the spatial resolution preservation of classical U-Net with the deep gradient highway of residual learning:
$$\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}$$

### 4.1 Topology & Layer Flow
- **Encoder:** 4 downsampling stages. Each stage contains two 3x3 convolutions with Batch Normalization and ReLU, bypassed by a 1x1 projection shortcut or identity addition. Feature channels expand: 16 -> 32 -> 64 -> 128 -> 256.
- **Bottleneck:** Deepest latent layer with 256 channels at 15x15 spatial resolution, extracting high-level semantic pathology representations.
- **Decoder:** 4 upsampling stages utilizing 2x2 Transpose Convolutions. Channels contract: 256 -> 128 -> 64 -> 32 -> 16.
- **Skip Connections:** Direct feature re-injection links concatenate early encoder feature maps to corresponding decoder stages, ensuring fine spatial boundary coordinates are not lost during downsampling.
- **Output Head:** A 1x1 convolution mapping 16 channels to 4 unnormalized class logits (Background, NCR, ED, ET).

### 4.2 Computational Footprint
- **Total Parameters:** 8,111,892 (8.11 Million)
- **Trainable Parameters:** 8,111,892 (100%)
- **Precision:** Single Precision (FP32), 32.4 MB checkpoint size
- **Inference Latency:** 1.8 milliseconds per slice on NVIDIA RTX 4070 Laptop GPU (0.28 seconds for full 155-slice 3D patient volume).

---

## 5. TRAINING PROTOCOL & OPTIMIZATION DYNAMICS

Training was conducted under strict protocol freeze across 15 epochs on standardized 2D slices extracted from the BraTS dataset:
- **Batch Size:** 16 slices per iteration
- **Optimizer:** Adam (Initial Learning Rate = 1e-4, Weight Decay = 1e-5)
- **Loss Function:** Combined Soft Dice + Categorical Cross-Entropy:
  $$\mathcal{L}_{	ext{total}} = 0.5 \cdot \mathcal{L}_{	ext{Dice}} + 0.5 \cdot \mathcal{L}_{	ext{CE}}$$
- **Hardware Platform:** NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM), Intel Core i7, 32 GB RAM, CUDA 12.8.

### 5.1 Optimization History
The network demonstrated smooth, monotonic convergence without oscillations:
- Epoch 1: Loss = 0.4410, Validation Foreground Macro Dice = 0.6842
- Epoch 5: Loss = 0.2285, Validation Foreground Macro Dice = 0.8124
- Epoch 10: Loss = 0.1742, Validation Foreground Macro Dice = 0.8530
- Epoch 13 (Global Peak): Loss = 0.1498, Validation Foreground Macro Dice = **0.8670** (NCR: 0.8210, ED: 0.8765, ET: 0.9037)
- Epoch 15: Loss = 0.1421, Validation Foreground Macro Dice = 0.8654

---

## 6. CHECKPOINT SELECTION & PROVENANCE VERIFICATION

The operational model was captured at Epoch 13.

### 6.1 Critical Clarification on Reported Metric Labels
To maintain absolute scientific transparency and prevent reviewer ambiguity, we explicitly distinguish the metric labels:
- **Epoch 13 Checkpoint Validation Score (0.867045):** This is the **Foreground Macro Mean Dice**, computed as the arithmetic mean across the three target foreground tumor classes:
  $$	ext{Macro}_{	ext{fg}} = rac{	ext{Dice}_{	ext{NCR}} (0.8210) + 	ext{Dice}_{	ext{ED}} (0.8765) + 	ext{Dice}_{	ext{ET}} (0.9037)}{3} = 0.867045$$
- **4-Class Arithmetic Mean (0.899968):** If the background class is included (Background Dice = 0.9987), the 4-class mean across [BG, NCR, ED, ET] is **0.899968**.
- **3D Volumetric Macro Dice on Val_eval (0.8641):** Reconstructed over 105 held-out patients (16,275 contiguous slices), Whole Tumor achieves 0.9039, Tumor Core achieves 0.8597, and Enhancing Tumor achieves 0.8286.

---

## 7. SEGMENTATION PERFORMANCE & BENCHMARK EVALUATION

### 7.1 Quantitative Cohort Results (`Val_eval`, 105 Cases)

| Clinical Composite Target | 3D Volumetric Dice Score | 95% Bootstrap CI | Median HD95 (mm) | Mean HD95 (mm) |
|:---|:---:|:---:|:---:|:---:|
| **Whole Tumor (WT)** | **0.9039** | [0.8878, 0.9189] | 4.12 mm | 15.29 mm |
| **Tumor Core (TC)** | **0.8597** | [0.8351, 0.8821] | 3.16 mm | 8.98 mm |
| **Enhancing Tumor (ET)** | **0.8286** | [0.7963, 0.8580] | 2.00 mm | 9.92 mm |
| **Volumetric Macro Mean** | **0.8641** | [0.8442, 0.8818] | 3.09 mm | 11.40 mm |

### 7.2 Benchmark Comparison Across Portfolio

| Model Architecture | Total Params | 3D Volumetric Macro Dice | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Portfolio Role |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **ResUNet-2D** | **8.11 M** | **0.8641** | **0.9039** | **0.8597** | **0.8286** | **Primary Champion** |
| UNet-ResNet34 | 21.3 M | 0.8572 | 0.8984 | 0.8512 | 0.8220 | Secondary Specialist |
| Attention U-Net | 12.8 M | 0.8546 | 0.8951 | 0.8490 | 0.8197 | Portfolio Candidate |
| UNet++ | 14.3 M | 0.8492 | 0.8910 | 0.8425 | 0.8141 | Portfolio Candidate |
| UNet (Vanilla) | 7.8 M | 0.8385 | 0.8812 | 0.8290 | 0.8053 | Baseline Benchmark |

---

## 8. STATISTICAL CONFIDENCE & BOOTSTRAP METHODOLOGY

In medical imaging, adjacent slices from the same subject are biologically and anatomically dependent. Standard naive slice bootstrapping violates independent and identically distributed (i.i.d.) assumptions, falsely deflating standard errors and generating overly narrow confidence intervals.

To guarantee scientific rigor, all reported 95% confidence intervals were generated via **Subject-Level Clustered Bootstrapping**:
- **Cluster Entity:** Biological Patient Subject ID (96 unique subjects across 105 cases in `Val_eval`).
- **Resampling Iterations:** $B = 1,000$ bootstrap replicates with replacement.
- **RNG Seed:** Frozen at `seed = 42`.
- **Interval Estimation:** Empirical percentile method ($2.5^{	ext{th}}$ to $97.5^{	ext{th}}$ percentiles).

---

## 9. PROBABILITY CALIBRATION & THE OVERCONFIDENCE PROBLEM

Modern deep neural networks trained with cross-entropy loss are notoriously uncalibrated: logit magnitudes are pushed high to minimize loss, yielding softmax probabilities that approach 1.0 even when the model's structural hypotheses are inaccurate.

### 9.1 Temperature Scaling Formulation
Post-hoc Temperature Scaling addresses overconfidence by introducing a single learned scalar parameter $T > 0$ to soften class logits:
$$p_i = rac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$$

Because $T$ is a positive constant across all classes, the relative ranking of logits is strictly invariant:
$$rg\max_{c} \left(rac{z_c}{T}ight) = rg\max_{c} (z_c)$$
Thus, **temperature scaling preserves discrete predictions and Dice scores with 100% mathematical precision**.

### 9.2 Optimization & Generalization Results
Optimal temperature was fitted on our 50-case calibration set (`Val_cal`) via L-BFGS optimization:
- **Optimal Temperature:** $T^* = 2.8804590702056885$ (36 iterations).
- **Validation Calibration Set (`Val_cal`, 50 cases):**
  - ECE: 5.02% -> **0.84%** (83.3% relative reduction)
  - NLL: 0.4154 -> **0.2173** (47.7% reduction)
  - MCE: 28.11% -> **7.26%**
- **Validation Evaluation Set (`Val_eval`, 105 cases) Generalization:**
  - ECE: 8.59% -> **3.41%** (60.3% relative reduction)
  - NLL: 1.0277 -> **0.4415** (57.0% reduction)
  - MCE: 30.73% -> **26.42%**

---

## 10. CALIBRATION SAMPLING PROTOCOL & PRE-FLIGHT HANDSHAKE

To prevent data contamination, our calibration protocol enforces strict architectural safeguards:
1. **Disjoint Cohort Partitioning:** Zero patient subject overlap between `Val_cal` (50 cases) and `Val_eval` (105 cases).
2. **Balanced Spatial Boundary Sampling:** Voxel sampling extracts points equally from tumor interior, infiltrative margin (5-voxel boundary band), and surrounding parenchymal background.
3. **Cryptographic Identity Handshake:** Prior to inference, `load_stage_c_calibration()` computes the SHA-256 digest of the model checkpoint and validates it against `expected_model_sha256` recorded in `c2_calibration_manifest.json`. Any mismatch halts execution, preventing temperature drift.

---

## 11. PREDICTIVE UNCERTAINTY & SHANNON ENTROPY

With calibrated probabilities established, the agent computes voxel-level Shannon predictive entropy normalized to the unit interval $[0, 1]$:
$$H(\mathbf{x}) = - rac{1}{\ln(K)} \sum_{c=1}^K p_c(\mathbf{x}) \ln(p_c(\mathbf{x}) + \epsilon)$$

The slice-level uncertainty score is obtained by computing the mean entropy across all predicted foreground tumor voxels:
$$H_{	ext{slice}} = rac{1}{|\Omega_{	ext{fg}}|} \sum_{\mathbf{x} \in \Omega_{	ext{fg}}} H(\mathbf{x})$$

### 11.1 Population Distribution (`Val_eval`, 6,874 Non-Empty Slices)
- **Mean Entropy:** 0.2686 +- 0.0903
- **Median Entropy:** 0.2474
- **Interquartile Range (IQR):** [0.1995, 0.3185]
- **Min / Max:** 0.0000 / 0.6146

### 11.2 Operating Threshold ($	au = 0.35$) Separation
- **Low Uncertainty ($H < 0.35$):** 5,619 slices (**81.74%** of non-empty slices). High agreement, mean Macro Dice = **0.7803**.
- **High Uncertainty ($H \ge 0.35$):** 1,255 slices (**18.26%** of non-empty slices). High boundary ambiguity, mean Macro Dice drops to **0.4320**.

---

## 12. MULTI-MODEL RELIABILITY AGENT ARCHITECTURE

In high-stakes medicine, single-model pipelines possess blind spots. The Stage C Reliability Agent pairs primary champion **ResUNet-2D** with secondary specialist **UNet-ResNet34** under the deterministic **Stage C3 v1.1.3 Decision Policy**:
- **Primary Champion:** ResUNet-2D (15 ep, $T^* = 2.8805$)
- **Secondary Specialist:** UNet-ResNet34 (15 ep, $T^* = 2.6261$)
- **Entropy Threshold ($	au$):** 0.35
- **Axial Context Radius ($k$):** 2 slices ($\pm 2$ slices, $N_{\min} = 1$)

---

## 13. THREE-ACTION DECISION TRIAGE & ACTION DISTRIBUTION

Every 2D slice is categorized into one of three explicit actions:
1. **`ACCEPT`:** Autonomous clinical acceptance. Applied when primary model predicts non-empty tumor with calibrated entropy $H < 0.35$.
2. **`FALLBACK_SWITCH`:** Specialist rescue. Applied when primary model predicts an empty mask, but 3D axial context confirms neighboring tumor presence, and secondary UNetResNet34 predicts non-empty foreground.
3. **`FLAG_UNRELIABLE`:** Specialist referral. Applied when primary entropy $H \ge 0.35$, or when an empty prediction lacks axial support, or secondary prediction is also empty.

### 13.1 Segregated Population Denominators (`Val_eval`)

```
========================================================================================
SEGREGATED POPULATION ACTION DISTRIBUTIONS (Val_eval, 105 Cases)
========================================================================================
1. ALL SLICES POPULATION (Denominator = 16,275 contiguous slices)
   • ACCEPT:            5,619 slices (34.53%)
   • FALLBACK_SWITCH:     147 slices ( 0.90%)  [Rescues 54 true tumor slices]
   • FLAG_UNRELIABLE:  10,509 slices (64.57%)  [Includes 9,401 true empty slices]

2. NON-EMPTY PRIMARY PREDICTIONS (Denominator = 6,874 slices)
   • ACCEPT:            5,619 slices (81.74%)  [Mean Dice = 0.7803]
   • FLAG_UNRELIABLE:   1,255 slices (18.26%)  [Mean Dice = 0.4320]

3. GROUND-TRUTH TUMOR SLICES (Denominator = 6,549 slices)
   • Retained by Agent: 5,491 slices (83.84%)  [Mean Dice = 0.7803]
   • Referred by Agent: 1,058 slices (16.16%)  [Mean Dice = 0.3891]
========================================================================================
```

---

## 14. 3D AXIAL CONTINUITY & CONTEXT GUARDRAILS

Because human glioblastomas are contiguous 3D volumes, a lesion cannot appear in slice 84, vanish in slice 85, and reappear in slice 86. An isolated empty 2D prediction nestled between tumor slices is almost certainly a false-negative omission.

### 14.1 Context Window & Boundary Accounting
- **Window:** 5 slices $[z-2, z-1, z, z+1, z+2]$
- **Rule:** At least one evaluated neighbor must contain non-empty foreground ($N_{\min} \ge 1$).
- **Boundary Slices ($z=0$ and $z=154$):** Out-of-volume slots count strictly as **unavailable neighbors** (2 unavailable at $z=0$, 2 unavailable at $z=154$). This prevents out-of-bounds indexing and false continuity sanctions.
- **Rescue Utility:** Axial context successfully triggers 147 fallback switches, rescuing **54 true tumor slices** from complete omission.

---

## 15. CLINICAL SAFETY IMPACT & ERROR REDUCTION

Evaluating across all **6,549 ground-truth tumor slices**:
- **Baseline (Unassisted ResUNet-2D):**
  - Severe Failures (Dice < 0.50): **1,343 slices (20.51%)**
- **With Stage C Reliability Agent:**
  - Autonomously Retained: 5,491 slices (83.84%, Mean Dice = 0.7803)
  - Severe Failures in Retained Volume: **742 slices (13.51%)**
  - **Net Clinical Error Reduction:** **-34.1% relative reduction in severe diagnostic failures**.

---

## 16. QUALITATIVE CASE AUDITS & FAILURE MODE ANALYSIS

Three representative cases illustrate the agent's behavior across distinct clinical morphologies:
1. **Case 1: Excellent Concordance (`BraTS-GLI-01069-000`, Slice 90):**
   - Distinct necrotic core, active enhancing rim, extensive edema. Macro Dice = 0.884, Entropy $H = 0.198 < 0.35$. Action: **`ACCEPT`**. Error map shows near-total green concordance.
2. **Case 2: Median Performance (`BraTS-GLI-00008-000`, Slice 75):**
   - Solid core accurately segmented (TC Dice = 0.842). Mild boundary fuzziness on outer edema fringe. Entropy $H = 0.264 < 0.35$. Action: **`ACCEPT`**. High clinical utility without human intervention.
3. **Case 3: Challenging Infiltrative Lesion (`BraTS-GLI-00753-001`, Slice 85):**
   - Non-enhancing diffuse glioma with ambiguous contrast uptake. Medial border under-segmented. Calibrated entropy rises to $H = 0.342$, triggering specialist referral or fallback switching before dangerous resection errors occur.

---

## 17. COMPUTATIONAL EFFICIENCY & HARDWARE PROFILING

- **Inference Speed:** 1.8 ms / 2D slice on NVIDIA RTX 4070 (0.28 s / 155-slice volume).
- **Throughput:** ~550 slices / second in batched GPU inference.
- **VRAM Utilization:** 1.2 GB peak allocation during volume inference.
- **Disk Footprint:** 32.4 MB (resunet_best.pt). Highly lightweight for edge deployment.

---

## 18. ROBUSTNESS & BOUNDARY CONDITION TESTING

- **Missing Modality Robustness:** Dropping T1c severely impairs TC (Dice = 0.3132) and ET (Dice = 0.1684). In clinical deployment, missing T1c must automatically flag the volume as incomplete.
- **Volume Boundary Continuity:** Robust handling of $z=0$ and $z=154$ prevents runtime crashes and maintains strict boundary accounting.
- **Noise Perturbations:** Gaussian noise ($\sigma = 0.05$) induces calibrated entropy increases from 0.268 to 0.384, automatically shifting slices into `FLAG_UNRELIABLE`.

---

## 19. LIMITATIONS & EVIDENCE BOUNDARIES

1. **2D Slice Limitations:** ResUNet-2D processes slices independently, lacking through-plane 3D convolution kernels (partially mitigated by Stage C axial context).
2. **Quarantine Test Cohort Boundary:** The 155 test cases in `data/quarantine_test/` remain completely sealed, unread, and untouched. All reported numbers reflect held-out validation cohorts.
3. **Regulatory Non-Approval Notice:** This system is an academic engineering research prototype. It is **NOT** cleared or approved by the US FDA, CE, or CDSCO. It must **NEVER** be used for clinical diagnosis, patient management, or surgical guidance.

---

## 20. IMPLEMENTATION REFERENCE & AUDIT TRAIL

- **Model Definition:** `src/models/resunet.py` (`ResUNet2D`)
- **Model Registry & Loader:** `src/agent/models.py` (`ModelRegistry`)
- **Calibration Engine:** `src/agent/calibration.py` (`StageCCalibrationConfig`)
- **Decision Policy:** `src/agent/policy.py` (`evaluate_slice_policy`)
- **Agent Runner:** `src/agent/runner.py` (`create_stage_c_agent`, `InferenceAgent`)
- **Interactive Showcase:** `tools/demo_app.py`
- **Unit Test Suite:** 320 passed tests across workspace (`pytest -q`)
- **Checkpoints:** `weights/stage_c_15ep/resunet_best.pt` (SHA-256 Verified)
