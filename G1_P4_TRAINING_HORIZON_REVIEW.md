# Five-Model Training-Horizon Review (15 vs 20 Epochs)
## Stage C Phase 1 Empirical Horizon Audit & Strategic Horizon Determination

**Document Version:** 1.2.0 (Corrected per G1-P4.11)
**Date:** November 15, 2026
**Workspace:** `D:\MajorProject`
**Governing Blueprint:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Protocol References:** `STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`, `STAGE_C_DECISION_REGISTER.csv`
**Investigation Mode:** STRICTLY READ-ONLY INVESTIGATION AND AUDIT
**Authorization Scope:** Existing records and source code inspection only. Zero model execution, training, inference, calibration, or test-cohort access permitted.

---

## A. Executive Decision

### 1. Authoritative Recommendation: OPTION A — Retain 15-Epoch Checkpoints and Proceed to Work Package C2 (Probability Calibration & Model-Specific Uncertainty Analysis)

Based on an empirical audit of all 15-epoch training curves, checkpoint state dictionaries, optimizer parameter groups, and scheduler configurations across all five segmentation models (`resunet_2d`, `unet_resnet34`, `unet`, `unetpp`, and `attention_unet`), the recommendation is:

> **RECOMMENDATION: OPTION A**
> **Preserve the five existing 15-epoch `best_model.pt` checkpoints without modification and proceed directly to Work Package C2: Probability Calibration & Model-Specific Uncertainty Analysis.**
> Under the governing protocol (`STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md` Section 5.2), Work Package C2 encompasses:
> 1. Fitting architecture-specific optimal temperature scaling parameters $T_m^*$ on the 50 calibration cases (`Val_cal`, 3,982 slices) by minimizing multi-class cross-entropy Negative Log-Likelihood (NLL).
> 2. Evaluating uncalibrated versus calibrated Expected Calibration Error (ECE, 15 equal-width bins), Maximum Calibration Error (MCE), Brier score, and NLL on the 105 internal evaluation cases (`Val_eval`).
> 3. Quantifying slice-level normalized Shannon predictive entropy across predicted foreground regions.
>
> Downstream activities—specifically the reliability-aware triage agent, selective prediction policies (`ACCEPT`, `FALLBACK_SWITCH`, `FLAG_UNRELIABLE`), risk-coverage sweeps, and failure referral—belong strictly to **Work Package C3: Reliability-Aware AI Agent & Selective Prediction** (Section 5.3) and are not combined into C2.
> Do not continue training existing checkpoints (Option B), and do not retrain all models from scratch for 20 epochs (Option C).

---

### 2. Empirical Evidence and Core Reasoning

This recommendation is grounded in the following empirical, architectural, and methodological observations:

#### 1. ResUNet-2D Peaked Prior to Epoch 15
- The baseline architecture **ResUNet-2D** achieved its highest recorded validation mean Dice score at **Epoch 13 (`0.8670`)** with a validation loss of `0.2073`.
- In the subsequent epochs, its validation score showed a modest decline: Epoch 14 achieved `0.8661` (validation loss `0.2063`), and Epoch 15 recorded **`0.8631`** (validation loss `0.2088`, a difference of $-0.0039$ Dice points from the Epoch 13 peak).
- ResUNet-2D's training loss delta between Epochs 14 and 15 diminished to $-0.0015$ (`0.1781` $\rightarrow$ `0.1766`).
- **Interpretation:** ResUNet-2D achieved its best recorded validation performance before Epoch 15 and showed signs of plateauing on the validation cohort. While 15 epochs of single-seed observations cannot definitively establish convergence or rule out further changes under different training budgets, these observations are consistent with diminishing returns.

#### 2. Training Loss Progression versus Validation Loss Behavior Across Architectures
- While four of the models (`unet_resnet34`, `unet`, `unetpp`, and `attention_unet`) achieved their highest recorded validation mean Dice at Epoch 15, inspection of their validation loss trajectories indicates that validation loss remained elevated, fluctuating, or above earlier minima:
  - **`unet_resnet34`:** Minimum validation loss occurred at **Epoch 4 (`0.2861`)**. At Epoch 15, validation loss was **`0.5166`** ($+80.5\%$ above the minimum), while training loss continued to decline from `0.2021` (Epoch 4) to `0.1349` (Epoch 15). Validation mean Dice changed from `0.8537` at Epoch 7 to `0.8554` at Epoch 15, a difference of $+0.0017$ across eight training epochs.
  - **`unet`:** Minimum validation loss occurred at **Epoch 4 (`0.3267`)**. At Epoch 15, validation loss was `0.4589` ($+40.5\%$ above the minimum), while training loss reached `0.1815`.
  - **`unetpp`:** Minimum validation loss occurred at **Epoch 6 (`0.2950`)**. At Epoch 15, validation loss was `0.4280` ($+45.1\%$ above the minimum, and higher than Epoch 11's `0.4158`), while training loss reached `0.1728`.
  - **`attention_unet`:** Minimum validation loss occurred at **Epoch 4 (`0.3246`)**. At Epoch 15, validation loss was `0.5013` ($+54.4\%$ above the minimum, and higher than Epoch 9's `0.4880` and Epoch 11's `0.4895`), while training loss reached `0.1843`.
- **Methodological Context:** In all models, training loss continued to decrease while validation loss remained above its early-epoch minima or exhibited plateauing. These divergent patterns are consistent with diminishing returns and possible overfitting. However, 15 epochs of single-seed observations cannot definitively establish convergence or forecast the exact effect of training beyond Epoch 15.

#### 3. Granular Class-Wise Contributions to Final-Epoch Changes
- An inspection of class-wise validation Dice changes between Epochs 14 and 15 reveals:
  - **U-Net:** ET change was $+0.0216$ (`0.7877` $\rightarrow$ `0.8094`), ED change was $+0.0120$ (`0.8542` $\rightarrow$ `0.8662`), and NCR change was $+0.0002$ (`0.9024` $\rightarrow$ `0.9026`). Mean validation Dice increased from `0.8481` to `0.8594`.
  - **U-Net++:** ET change was $+0.0249$ (`0.7847` $\rightarrow$ `0.8096`), ED change was $+0.0173$ (`0.8490` $\rightarrow$ `0.8664`), and NCR change was $+0.0033$ (`0.9014` $\rightarrow$ `0.9047`). Mean validation Dice increased from `0.8450` to `0.8602`.
  - **Attention U-Net:** ET change was $+0.0154$ (`0.7793` $\rightarrow$ `0.7947`), ED change was $+0.0124$ (`0.8521` $\rightarrow$ `0.8645`), and NCR change was $+0.0019$ (`0.9016` $\rightarrow$ `0.9035`). Mean validation Dice increased from `0.8443` to `0.8542`.
- **Descriptive Finding:** In all three Phase 1 architectures, ET was the largest contributor to the observed class-wise improvements between Epochs 14 and 15, while ED also improved meaningfully and NCR changed relatively little.
- **Methodological Note:** A single training seed and one validation cohort cannot establish whether these class-wise shifts reflect boundary adjustments, slice-level variance, or optimization momentum. Furthermore, while validation selection exposure (selecting `best_model.pt` via $\operatorname{argmax}$ over 15 evaluation points) is an inherent property of the checkpoint-selection protocol, it does not prove that the Epoch 15 gains are spurious.

#### 4. Balanced Treatment of the U-Net Final-Epoch Trajectory
- Between Epochs 14 and 15, the Standard U-Net exhibited a clear recorded improvement:
  - Mean validation Dice increased from `0.8481` to `0.8594` ($+0.0113$).
  - Validation loss decreased from `0.4966` to `0.4589` ($-0.0377$).
  - ET, ED, and NCR all improved, with ET demonstrating the largest numerical increase ($+0.0216$).
- This represents a genuine observed improvement in the final recorded epoch of that run. However, earlier in the same run, validation loss had fluctuated (reaching `0.5381` at Epoch 13) and remained well above the Epoch 4 minimum (`0.3267`).
- While this final-epoch rebound is a verified empirical feature of the trajectory, it does not demonstrate that Epochs 16–20 would yield continued gains; conversely, the earlier fluctuations mean the run cannot be described as continuous validation degradation. Retaining the Epoch 15 checkpoint captures this recorded peak without speculating on unobserved epochs.

#### 5. Methodological and Technical Constraints on Continuation (Option B)
- **Schedule Configuration:** All five models were trained using a composite learning-rate schedule (`SequentialLR`) configured for exactly 15 epochs: a 2-epoch linear warmup followed by a 13-epoch cosine decay (`T_max = 13`), ending at minimum learning-rate floors ($\eta_{\min} = 1 \times 10^{-5}$ uniformly, and $1 \times 10^{-6}$ for the ResNet34 encoder).
- **Continuation Constraints:** Simply incrementing the epoch count and resuming the saved scheduler past its configured $T_{\max}$ is not an authorized protocol. If the underlying `CosineAnnealingLR` advances past $T_{\max}$, there is a structural risk of cyclic learning-rate resurgence (increasing LR back towards $1 \times 10^{-3}$), unless specifically clamped or reconfigured. Alternatively, clamping the rate at $1 \times 10^{-5}$ would mean training for five epochs at an ad-hoc static floor.
- While a deliberate continuation schedule could technically be designed (e.g., adding an explicit secondary annealing schedule), that would represent a separately specified continuation protocol rather than an uninterrupted continuation of the original schedule.
- **State Preservation:** Inspection of `final_model.pt` across all five models confirms that checkpoints store `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`, and scalar metrics, but do **not** store CPU/CUDA random number generator states (`torch.get_rng_state()`, `torch.cuda.get_rng_state()`) or DataLoader shuffle states. Resuming from Epoch 15 therefore introduces non-deterministic sampling variations.

#### 6. Computational Resource Investment and Research Program Scope
- **Computational Investment:** Retraining all five models from scratch for 20 epochs (Option C) would require an estimated **~72.5 GPU hours** (~3 days of continuous compute on the local NVIDIA GeForce RTX 4070 Laptop GPU), compared to the ~53.3 GPU hours already invested across the 15-epoch runs.
- **Scope Alignment:** The primary objective of the research program is to evaluate **predictive uncertainty and post-hoc probability calibration (Work Package C2)**, followed by **reliability-aware selective prediction and autonomous triage governance (Work Package C3)**. The five models currently provide an established, diversified ensemble of backbones with validation mean Dice scores between **0.8542 and 0.8670**.
- Whether additional training epochs would meaningfully alter calibration quality or probability reliability cannot be determined from training and validation loss curves alone; these properties must be measured empirically during C2.

---

## B. Five-Model Comparison Table

The following table summarizes the empirical training metrics extracted directly from the authoritative training histories (`history.csv`) and run summaries (`summary.json`) for each model. All validation metrics represent **selection-exposed internal validation metrics** computed on the frozen 155-case validation cohort (`val_cases: 155`, `val_slices: 12364`).

| Model | Authoritative Experiment Directory | Best Epoch | Best Val Mean Dice | Ep 15 Val Mean Dice | Ep 11–15 Val Dice Trend | Training-Loss Trend (Ep 11 $\rightarrow$ 15) | Validation-Loss Trend (Ep 11 $\rightarrow$ 15) | Min Val Loss (Epoch) | Assessment |
|---|---|---:|---:|---:|---|---|---|---|---|
| **ResUNet-2D** | `results/full_training_15ep/resunet_2d/` | **13** | **0.8670** | 0.8631 | Best at Ep 13; modest decline through Ep 14 (0.8661) & Ep 15 (0.8631) | Decreased slowly: $0.1895 \rightarrow 0.1766$ ($\Delta = -0.0129$) | Flattened / slight uptick: $0.2150 \rightarrow 0.2088$ ($\Delta = -0.0061$) | **0.2063** (Ep 14) | Peaked at Ep 13; validation performance plateaued through Ep 15 with diminishing training-loss reductions. Consistent with diminishing returns. |
| **UNetResNet34** | `results/full_training_15ep/unet_resnet34/` | **15** | **0.8554** | 0.8554 | Oscillated around plateau: 0.8494 $\rightarrow$ 0.8454 $\rightarrow$ 0.8472 $\rightarrow$ 0.8477 $\rightarrow$ 0.8554 (+0.0017 over Ep 7) | Decreased monotonically: $0.1505 \rightarrow 0.1349$ ($\Delta = -0.0155$) | Elevated / oscillating: $0.4948 \rightarrow 0.5166$ ($\Delta = +0.0218$) | **0.2861** (Ep 4) | Highest score at Ep 15, but validation loss is $+80.5\%$ above Ep 4 minimum while training loss fell. Consistent with diminishing returns and possible overfitting. |
| **U-Net** | `results/stage_c/phase1/unet_721_seed42/` | **15** | **0.8594** | 0.8594 | Dipped at Ep 12–13, rebounded at Ep 15: 0.8506 $\rightarrow$ 0.8384 $\rightarrow$ 0.8399 $\rightarrow$ 0.8481 $\rightarrow$ 0.8594 (+0.0065 over Ep 10) | Rate of decrease flattened: $0.1914 \rightarrow 0.1815$ ($\Delta = -0.0099$) | Fluctuated above minimum: $0.4911 \rightarrow 0.4589$ ($\Delta = -0.0322$) | **0.3267** (Ep 4) | Rebounded in final epoch across all classes and validation loss, but validation loss remains above Ep 4 minimum. Observations cannot forecast post-Ep 15 behavior. |
| **U-Net++** | `results/stage_c/phase1/unetpp_721_seed42/` | **15** | **0.8602** | 0.8602 | Dipped at Ep 12, rebounded at Ep 15: 0.8482 $\rightarrow$ 0.8376 $\rightarrow$ 0.8477 $\rightarrow$ 0.8450 $\rightarrow$ 0.8602 (+0.0053 over Ep 10) | Rate of decrease flattened: $0.1816 \rightarrow 0.1728$ ($\Delta = -0.0088$) | Rose from Ep 11: $0.4158 \rightarrow 0.4280$ ($\Delta = +0.0122$) | **0.2950** (Ep 6) | Highest score at Ep 15, driven primarily by ET and ED gains. Validation loss ended higher than Ep 10 and 11, and above Ep 6 minimum. |
| **Attention U-Net** | `results/stage_c/phase1/attention_unet_721_seed42/` | **15** | **0.8542** | 0.8542 | Oscillated around plateau: 0.8442 $\rightarrow$ 0.8473 $\rightarrow$ 0.8376 $\rightarrow$ 0.8443 $\rightarrow$ 0.8542 (+0.0043 over Ep 9) | Rate of decrease flattened: $0.1922 \rightarrow 0.1843$ ($\Delta = -0.0079$) | Remained above minimum: $0.4895 \rightarrow 0.5013$ ($\Delta = +0.0118$) | **0.3246** (Ep 4) | Modest gain over Ep 9 peak (+0.0043). Training loss rate of decrease slowed markedly, and validation loss remained $+54.4\%$ above minimum. |

---

### Additional Detailed Architecture and Computational Resource Comparison

| Metric / Attribute | ResUNet-2D | UNetResNet34 | Standard U-Net | U-Net++ | Attention U-Net |
|---|---:|---:|---:|---:|---:|
| **Model Parameters** | 8,114,604 | 24,327,524 | 7,763,428 | 9,160,068 | 7,852,160 |
| **Relative Parameter Size** | $1.05\times$ | $3.13\times$ | $1.00\times$ (Ref) | $1.18\times$ | $1.01\times$ |
| **Total Recorded Runtime** | 8.09 hours (485.4 min) | 10.80 hours (648.1 min) | 8.24 hours (494.4 min)* | 15.44 hours (926.3 min) | 10.70 hours (642.0 min) |
| **Average Epoch Duration** | ~32.4 min/epoch | ~43.2 min/epoch | ~37.3 min/epoch (sync)** | ~61.8 min/epoch | ~42.8 min/epoch |
| **Peak VRAM Allocated** | 436.5 MB | 781.0 MB | 442.2 MB | 623.1 MB | 458.7 MB |
| **Epoch 15 Train Loss** | 0.1766 | 0.1349 | 0.1815 | 0.1728 | 0.1843 |
| **Epoch 15 Val Loss** | 0.2088 | 0.5166 | 0.4589 | 0.4280 | 0.5013 |
| **Epoch 15 Val Mean IoU** | 0.7613 | 0.7497 | 0.7554 | 0.7568 | 0.7482 |
| **Epoch 15 Class 1 (ET) Dice** | 0.8097 | 0.7993 | 0.8094 | 0.8096 | 0.7947 |
| **Epoch 15 Class 2 (ED) Dice** | 0.8739 | 0.8648 | 0.8662 | 0.8664 | 0.8645 |
| **Epoch 15 Class 3 (NCR) Dice**| 0.9057 | 0.9020 | 0.9026 | 0.9047 | 0.9035 |
| **Epoch 15 Class 0 (BG) Dice** | 0.9987 | 0.9986 | 0.9986 | 0.9986 | 0.9986 |

*\*Note on U-Net Total Runtime:* The recorded 8.24 hours reflects mixed asynchronous execution (Epochs 1–5 before interruption, ~2.02 hours) and synchronous execution (Epochs 6–15 after resumption, 6.22 hours).
*\*\*Note on U-Net Average Epoch Duration:* The 37.3 min/epoch average is computed strictly on the synchronous resumption phase (6.22 hours / 10 epochs = 37.31 min).

---

## C. Individual Model Assessments

### 1. ResUNet-2D (Stage B Baseline)
- **Authoritative Files:**
  - Config: `results/full_training_15ep/resunet_2d/config.json`
  - History: `results/full_training_15ep/resunet_2d/history.csv`
  - Summary: `results/full_training_15ep/resunet_2d/summary.json`
  - Checkpoint: `results/full_training_15ep/resunet_2d/best_model.pt` (Epoch 13)
- **Observed Learning-Curve Dynamics:**
  - **Epochs 1–10:** Steady reduction in loss. Training loss decreased from `0.3067` (Epoch 1) to `0.1983` (Epoch 10). Validation loss decreased from `0.2974` to `0.2173`. Validation mean Dice improved from `0.8078` to `0.8541`.
  - **Epochs 11–15 Trajectory:**
    - Ep 11: Train Loss `0.1895` | Val Loss `0.2150` | Val Mean Dice `0.8569` | ET `0.8013` | ED `0.8687` | NCR `0.9007`
    - Ep 12: Train Loss `0.1833` | Val Loss `0.2194` | Val Mean Dice `0.8571` | ET `0.8139` | ED `0.8600` | NCR `0.8973`
    - **Ep 13 (Best):** Train Loss `0.1803` | Val Loss `0.2073` | Val Mean Dice **`0.8670`** | ET **`0.8210`** | ED **`0.8765`** | NCR `0.9037`
    - Ep 14: Train Loss `0.1781` | Val Loss `0.2063` (Min Val Loss) | Val Mean Dice `0.8661` | ET `0.8174` | ED `0.8745` | NCR **`0.9065`**
    - Ep 15: Train Loss `0.1766` | Val Loss `0.2088` | Val Mean Dice `0.8631` | ET `0.8097` | ED `0.8739` | NCR `0.9057`
  - **Assessment:** ResUNet-2D achieved its highest recorded validation mean Dice at **Epoch 13** (`0.8670`). Performance in Epochs 14 and 15 showed slight reductions in mean Dice and a flat validation loss trajectory. Training loss reductions between Epochs 14 and 15 diminished to $-0.0015$.
- **Learning-Rate Progression:**
  - Configured with base LR `1e-3`, 2-epoch linear warmup, and 13-epoch cosine decay to $\eta_{\min} = 1 \times 10^{-5}$.
  - Active LR at Epoch 15 start was `2.44e-5`; stepped to `1.0e-5` post-epoch.
- **Checkpoint Completeness & Feasibility:**
  - `best_model.pt` contains: `epoch=13`, `metrics` (mean Dice `0.8670`), `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`.
  - Missing state: CPU/CUDA RNG states, DataLoader sampler state.
  - Continuation would face schedule completion constraints and unrecorded RNG states, on a trajectory that already showed modest validation performance declines.

---

### 2. UNetResNet34 (Stage B Pretrained Backbone)
- **Authoritative Files:**
  - Config: `results/full_training_15ep/unet_resnet34/config.json`
  - History: `results/full_training_15ep/unet_resnet34/history.csv`
  - Summary: `results/full_training_15ep/unet_resnet34/summary.json`
  - Checkpoint: `results/full_training_15ep/unet_resnet34/best_model.pt` (Epoch 15)
- **Observed Learning-Curve Dynamics:**
  - **Epochs 1–10:** Rapid initial optimization. Validation loss reached its minimum at **Epoch 4 (`0.2861`)**. By Epoch 7, validation mean Dice reached `0.8537` with validation loss `0.4086`.
  - **Epochs 11–15 Trajectory:**
    - Ep 11: Train Loss `0.1505` | Val Loss `0.4948` | Val Mean Dice `0.8494` | ET `0.7800` | ED `0.8672` | NCR `0.9011`
    - Ep 12: Train Loss `0.1449` | Val Loss `0.5412` | Val Mean Dice `0.8454` | ET `0.7917` | ED `0.8579` | NCR `0.8864`
    - Ep 13: Train Loss `0.1399` | Val Loss `0.5303` | Val Mean Dice `0.8472` | ET `0.7877` | ED `0.8576` | NCR `0.8961`
    - Ep 14: Train Loss `0.1364` | Val Loss `0.5041` | Val Mean Dice `0.8477` | ET `0.7953` | ED `0.8517` | NCR `0.8962`
    - **Ep 15 (Best):** Train Loss `0.1349` | Val Loss `0.5166` | Val Mean Dice **`0.8554`** | ET `0.7993` | ED `0.8648` | NCR `0.9020`
  - **Assessment:** While training loss decreased from `0.2021` (Epoch 4) to `0.1349` (Epoch 15), validation loss rose from `0.2861` to `0.5166` ($+80.5\%$). Validation mean Dice changed from `0.8537` (Epoch 7) to `0.8554` (Epoch 15), a difference of $+0.0017$ across eight training epochs. This pattern is consistent with diminishing returns and possible overfitting on the validation set.
- **Learning-Rate Progression:**
  - Differential LR parameter groups: Group 0 (encoder base `1e-4`, floor `1e-6`); Group 1 (decoder base `1e-3`, floor `1e-5`).
  - Active primary LR at Epoch 15 start was `2.44e-5`; stepped to `1.0e-5` / `1.0e-6` post-epoch.
- **Checkpoint Completeness & Feasibility:**
  - Missing state: CPU/CUDA RNG states, DataLoader sampler state.
  - Continuation would require deliberate schedule re-engineering and would operate in a regime where validation loss was substantially higher than in earlier epochs.

---

### 3. Standard U-Net (Stage C Phase 1)
- **Authoritative Files:**
  - Config: `results/stage_c/phase1/unet_721_seed42/config.json`
  - History: `results/stage_c/phase1/unet_721_seed42/history.csv`
  - Summary: `results/stage_c/phase1/unet_721_seed42/summary.json`
  - Checkpoint: `results/stage_c/phase1/unet_721_seed42/best_model.pt` (Epoch 15)
- **Observed Learning-Curve Dynamics:**
  - **Epochs 1–10:** Training loss decreased from `0.4287` to `0.1963`. Validation loss reached its minimum at **Epoch 4 (`0.3267`)**, subsequently oscillating between `0.45` and `0.54`. Validation mean Dice reached `0.8529` at Epoch 10.
  - **Epochs 11–15 Trajectory:**
    - Ep 11: Train Loss `0.1914` | Val Loss `0.4911` | Val Mean Dice `0.8506` | ET `0.7885` | ED `0.8659` | NCR `0.8972`
    - Ep 12: Train Loss `0.1878` | Val Loss `0.5224` | Val Mean Dice `0.8384` | ET `0.7830` | ED `0.8431` | NCR `0.8890`
    - Ep 13: Train Loss `0.1852` | Val Loss `0.5381` | Val Mean Dice `0.8399` | ET `0.7622` | ED `0.8583` | NCR `0.8993`
    - Ep 14: Train Loss `0.1832` | Val Loss `0.4966` | Val Mean Dice `0.8481` | ET `0.7877` | ED `0.8542` | NCR `0.9024`
    - **Ep 15 (Best):** Train Loss `0.1815` | Val Loss `0.4589` | Val Mean Dice **`0.8594`** | ET **`0.8094`** | ED `0.8662` | NCR `0.9026`
  - **Assessment:** U-Net exhibited an observed rebound between Epochs 14 and 15: mean validation Dice increased from `0.8481` to `0.8594` ($+0.0113$), validation loss improved from `0.4966` to `0.4589`, and all three tumor compartments improved (ET $+0.0216$, ED $+0.0120$, NCR $+0.0002$). While this represents an empirical improvement at the final recorded epoch, validation loss remained above the Epoch 4 minimum (`0.3267`), and the training loss rate of decrease slowed to $-0.0017$/epoch. These observations document a final-epoch gain without establishing whether extending training would produce further improvements.
- **Learning-Rate Progression:**
  - 2-epoch linear warmup, 13-epoch cosine decay ($T_{\max}=13$, floor `1e-5`).
  - Active LR at Epoch 15 start was `2.44e-5`; stepped to `1.0e-5` at completion.
- **Checkpoint Completeness & Feasibility:**
  - `final_model.pt` contains: `epoch=15`, `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`.
  - Missing state: CPU/CUDA RNG states, DataLoader sampler state. (Note: U-Net also experienced an asynchronous interruption at Epoch 5; see Section F).
  - Resumption would require redesigning the learning-rate schedule to avoid running at an unconfigured or static rate.

---

### 4. U-Net++ (Nested U-Net, Stage C Phase 1)
- **Authoritative Files:**
  - Config: `results/stage_c/phase1/unetpp_721_seed42/config.json`
  - History: `results/stage_c/phase1/unetpp_721_seed42/history.csv`
  - Summary: `results/stage_c/phase1/unetpp_721_seed42/summary.json`
  - Checkpoint: `results/stage_c/phase1/unetpp_721_seed42/best_model.pt` (Epoch 15)
- **Observed Learning-Curve Dynamics:**
  - **Epochs 1–10:** Training loss decreased from `0.4035` to `0.1852`. Validation loss reached its minimum at **Epoch 6 (`0.2950`)**. Validation mean Dice reached `0.8549` at Epoch 10.
  - **Epochs 11–15 Trajectory:**
    - Ep 11: Train Loss `0.1816` | Val Loss `0.4158` | Val Mean Dice `0.8482` | ET `0.7787` | ED `0.8659` | NCR `0.9001`
    - Ep 12: Train Loss `0.1791` | Val Loss `0.4864` | Val Mean Dice `0.8376` | ET `0.7757` | ED `0.8491` | NCR `0.8880`
    - Ep 13: Train Loss `0.1766` | Val Loss `0.5210` | Val Mean Dice `0.8477` | ET `0.7821` | ED `0.8596` | NCR `0.9014`
    - Ep 14: Train Loss `0.1746` | Val Loss `0.4382` | Val Mean Dice `0.8450` | ET `0.7847` | ED `0.8490` | NCR `0.9014`
    - **Ep 15 (Best):** Train Loss `0.1728` | Val Loss `0.4280` | Val Mean Dice **`0.8602`** | ET **`0.8096`** | ED `0.8664` | NCR `0.9047`
  - **Assessment:** Validation loss reached `0.4158` at Epoch 11, rose to `0.5210` at Epoch 13, and finished at `0.4280` at Epoch 15. The final validation loss is higher than at Epoch 11 (`0.4158`) and Epoch 10 (`0.4214`), and $+45.1\%$ above the Epoch 6 minimum (`0.2950`). Between Epochs 14 and 15, validation Dice improved $+0.0152$, driven by gains in ET ($+0.0249$) and ED ($+0.0173$), with NCR changing $+0.0033$.
- **Learning-Rate Progression:**
  - 2-epoch warmup, 13-epoch cosine decay ($T_{\max}=13$, floor `1e-5`). Active LR at Epoch 15 start was `2.44e-5`; stepped to `1.0e-5` at completion.
- **Checkpoint Completeness & Feasibility:**
  - Missing state: CPU/CUDA RNG states, DataLoader sampler state.
  - U-Net++ required the longest per-epoch runtime (~61.8 min/epoch). Extending training would entail substantial additional compute while operating under schedule exhaustion.

---

### 5. Attention U-Net (Stage C Phase 1)
- **Authoritative Files:**
  - Config: `results/stage_c/phase1/attention_unet_721_seed42/config.json`
  - History: `results/stage_c/phase1/attention_unet_721_seed42/history.csv`
  - Summary: `results/stage_c/phase1/attention_unet_721_seed42/summary.json`
  - Checkpoint: `results/stage_c/phase1/attention_unet_721_seed42/best_model.pt` (Epoch 15)
- **Observed Learning-Curve Dynamics:**
  - **Epochs 1–10:** Training loss decreased from `0.4219` to `0.1963`. Validation loss reached its minimum at **Epoch 4 (`0.3246`)**. Validation mean Dice reached `0.8499` at Epoch 9.
  - **Epochs 11–15 Trajectory:**
    - Ep 11: Train Loss `0.1922` | Val Loss `0.4895` | Val Mean Dice `0.8442` | ET `0.7729` | ED `0.8632` | NCR `0.8964`
    - Ep 12: Train Loss `0.1888` | Val Loss `0.5482` | Val Mean Dice `0.8473` | ET `0.8010` | ED `0.8536` | NCR `0.8873`
    - Ep 13: Train Loss `0.1860` | Val Loss `0.5224` | Val Mean Dice `0.8376` | ET `0.7622` | ED `0.8555` | NCR `0.8951`
    - Ep 14: Train Loss `0.1860` | Val Loss `0.4912` | Val Mean Dice `0.8443` | ET `0.7793` | ED `0.8521` | NCR `0.9016`
    - **Ep 15 (Best):** Train Loss `0.1843` | Val Loss `0.5013` | Val Mean Dice **`0.8542`** | ET `0.7947` | ED `0.8645` | NCR `0.9035`
  - **Assessment:** Training loss showed little change between Epochs 13 and 14 (`0.1860` $\rightarrow$ `0.1860`), decreasing by $-0.0017$ at Epoch 15 (`0.1843`). Validation loss at Epoch 15 (`0.5013`) was higher than at Epoch 9 (`0.4880`), Epoch 11 (`0.4895`), and Epoch 14 (`0.4912`), remaining $+54.4\%$ above the Epoch 4 minimum (`0.3246`). The net gain in validation Dice between Epoch 9 and Epoch 15 was $+0.0043$.
- **Learning-Rate Progression:**
  - 2-epoch warmup, 13-epoch cosine decay ($T_{\max}=13$, floor `1e-5`). Active LR at Epoch 15 start was `2.44e-5`; stepped to `1.0e-5` at completion.
- **Checkpoint Completeness & Feasibility:**
  - Missing state: CPU/CUDA RNG states, DataLoader sampler state.
  - Slower training-loss decreases and elevated validation loss suggest diminishing returns under the current optimization schedule.

---

## D. Strategy Comparison

| Evaluation Dimension | Option A: Retain 15-Epoch Checkpoints & Proceed to C2 (Recommended) | Option B: Continue Existing Checkpoints from Ep 15 to 20 | Option C: Retrain All 5 Models from Scratch (20 Epochs) |
|---|---|---|---|
| **Protocol Consistency** | **High.** Preserves the approved 15-epoch protocol across all five architectures. Maintains a consistent training horizon without post-hoc schedule modifications. | **Low.** The original schedule was configured for 15 epochs. Resuming past Epoch 15 requires either ad-hoc rate freezing at the floor or an unverified schedule extension. | **High.** Scientifically consistent if newly configured schedules ($T_{\max}=18$, warmup 2) are trained uniformly from initialization across all models. |
| **Reproducibility & State Integrity** | **Documented & Intact.** All existing checkpoints, training histories, configurations, and summaries are preserved and hash-verified. Exact stochastic re-runs remain bounded by absent RNG states and the historical U-Net resumption. | **Limited.** Checkpoints do not record CPU/CUDA RNG states or DataLoader sampler iterators, precluding exact deterministic continuation from Epoch 15. | **Conditionally High (Requires Explicit Environment Control).** Reproducibility from scratch cannot be assumed solely from fixing seed 42. It would require an explicitly controlled and recorded execution environment, hardware and library version parity, initialization tracking, CPU/CUDA RNG state logging, DataLoader worker seeding, deterministic algorithm enforcement (e.g. cuDNN determinism), and strict manifest parity. Without all of these factors formally guaranteed, exact deterministic reproduction cannot be claimed. |
| **Cross-Model Comparability** | **High.** All five models share an identical 15-epoch training budget, 2-epoch warmup, 13-epoch cosine decay, and identical evaluation checkpoints. | **Low.** Merges a standard 15-epoch cosine schedule with an ad-hoc 5-epoch continuation, introducing cross-model schedule discrepancies. | **High.** Provides cross-model comparability across a 20-epoch budget, but creates a divergence against prior 15-epoch literature benchmarks. |
| **Estimated GPU Compute** | **Zero Additional Compute (0.0 hours).** Preserves current GPU resource budget for C2 calibration. | **~18.1 GPU Hours (Estimated).** Calculated as 5 epochs $\times$ 3.625 hours/epoch across all 5 models based on measured average epoch runtimes.* | **~72.5 GPU Hours (Estimated).** Calculated as 20 epochs $\times$ 3.625 hours/epoch across all 5 models from scratch.* |
| **Relevance to Stage C Objectives** | **High.** Enables direct transition to Work Package C2 (Probability Calibration & Model-Specific Uncertainty Analysis) and subsequently Work Package C3 (Reliability-Aware AI Agent & Selective Prediction). Baseline calibration and uncertainty metrics must be measured empirically. | **Uncertain.** Does not resolve whether calibration metrics improve with additional epochs, while introducing schedule artifacts. | **Uncertain.** Additional training epochs may alter baseline logit distributions, but whether this improves post-hoc calibration must be tested empirically. |
| **Hardware & Operational Considerations** | **Minimal.** No additional compute load; avoids extending local GPU runs. | **Moderate.** Involves ~18.1 hours of additional compute across five separate resumption setups. | **Substantial.** Involves ~72.5 hours of continuous compute on local hardware, requiring extended execution monitoring. |

*\*Derivation of Computational Estimates:*
- Sum of representative per-epoch durations across the five models: ResUNet-2D (32.4 min) + UNetResNet34 (43.2 min) + Standard U-Net (37.3 min, synchronous portion) + U-Net++ (61.8 min) + Attention U-Net (42.8 min) = 217.5 min = **3.625 hours per full 5-model epoch**.
- **Option B (Continuation):** 5 epochs $\times$ 3.625 hours = **~18.1 estimated GPU hours**.
- **Option C (Scratch Retraining):** 20 epochs $\times$ 3.625 hours = **~72.5 estimated GPU hours**.
- These values are projections based on measured epoch averages and assume continuous execution without thermal throttling, driver resets, or system interruptions.

---

## E. Proposed Next-Step Protocol (Contingency Specification)

> **MANDATORY NOTICE:** Option A is adopted. The following protocol is a **contingency specification only** defining the procedural requirements that would apply *if and only if* project leadership or an external supervisory committee specifically mandates a 20-epoch training horizon. **DO NOT EXECUTE.**

### 1. Distinction Between Potential 20-Epoch Approaches:

If a 20-epoch training horizon were formally mandated, two distinct technical approaches exist:

- **Approach 1: Standardized 20-Epoch Comparative Experiment (Retraining from Scratch)**
  To preserve rigorous scientific comparability across all five architectures from initialization, all models should be trained from scratch under a newly specified, uniform 20-epoch protocol. This approach avoids mixing different optimization trajectories and eliminates checkpoint resumption artifacts.
  - Schedulers must be re-parameterized from initialization with `warmup_epochs = 2` and `T_max = 18` (total epochs 20), such that the cosine decay progresses smoothly across all 18 post-warmup epochs to $\eta_{\min} = 1 \times 10^{-5}$ at Epoch 20.
  - Requires an estimated **~72.5 GPU hours** across all five architectures.

- **Approach 2: Separately Designed Continuation Experiment (Extending Existing Checkpoints)**
  Technically, existing 15-epoch checkpoints could be extended for an additional five epochs under a newly specified and documented continuation schedule (e.g., an explicit secondary annealing phase, a warm-restarted cosine cycle, or a constant fine-tuning rate at the floor).
  - While technically feasible, this cannot be represented as an uninterrupted 20-epoch comparative baseline; it constitutes a distinct experimental protocol with its own methodological characteristics, including non-continuous optimizer state trajectories and absent RNG states from earlier epochs.
  - Requires an estimated **~18.1 GPU hours** across all five architectures.

### 2. Requirements That Must Be Frozen Prior to Any Potential 20-Epoch Execution:
1. **Protocol and Schedule Formalization:**
   - The selected approach (scratch retraining vs. continuation) must be formally specified, including the exact mathematical formulation of the learning-rate schedule, before launching any run.
2. **Cohort Manifest and Data Pipeline Freeze:**
   - Lock `development_train_manifest.json` (721 cases, 59,020 slices; SHA-256 `42b7bf30...`) and `validation_manifest.json` (155 cases, 12,364 slices; SHA-256 `20accad8...`).
   - Maintain physical quarantine of `test_manifest.json` and `data/quarantine_test/`.
3. **Hyperparameter Parity and Reproducibility Controls:**
   - Batch size ($2$), base learning rate ($1 \times 10^{-3}$ with differential $1 \times 10^{-4}$ for the ResNet34 encoder), weight decay ($1 \times 10^{-4}$), gradient clipping (`max_norm = 1.0`), and random seed ($42$) must remain identical.
   - If deterministic reproducibility is required, execution configurations must explicitly enforce cuDNN determinism (`torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic = True`), worker-level DataLoader seed fixing, and RNG state logging.
4. **Validation Monitoring:**
   - Validation loss and class-wise Dice metrics must continue to be evaluated at every epoch to monitor generalization behavior.

---

## F. Methodological Limitations and Evidence Boundaries

1. **State Preservation Limitations in Existing Checkpoints:**
   - None of the five model checkpoints archive `torch.get_rng_state()`, `torch.cuda.get_rng_state()`, or DataLoader iterator states. As a result, resuming an existing checkpoint does not reproduce the exact mini-batch shuffle sequence that an uninterrupted run would have produced.
2. **Distinction Between Artifact Integrity and Stochastic Reproducibility:**
   - Checkpoint hashes, configuration files, and training logs are fully preserved and verified. However, exact bit-for-bit numerical re-execution from scratch or resumption is subject to non-deterministic CUDA operations and unrecorded sampler states.
3. **Historical Execution Interruption in U-Net Run:**
   - As documented in `G1_P4_UNET_STAGE_C_TRAINING_REPORT.md`, the Standard U-Net experiment experienced an execution interruption at Epoch 5, batch 17,500, and was subsequently resumed from Epoch 5, batch 1 (re-executing the first 17,500 batches of that epoch). This non-continuous event is an established historical characteristic of the U-Net artifact.
4. **Validation Selection Exposure:**
   - All validation metrics reported across all five models represent **selection-exposed internal validation metrics**, as `best_model.pt` was selected via $\operatorname{argmax}$ over the validation set. They do not constitute unbiased generalization estimates on held-out test data.
5. **Single-Seed Evaluation:**
   - All five models were trained using a single random seed (`seed = 42`). Single-seed observations cannot quantify the variance of validation metrics across different random initializations or mini-batch orderings.
6. **Strict Test Cohort Quarantine During This Investigation:**
   - During this G1-P4.9 / G1-P4.10 / G1-P4.11 horizon review, the sealed test cohort (`data/quarantine_test/`, 155 cases) and its manifest (`results/dataset_splits/test_manifest.json`) were not accessed. No test image or label data was loaded, no models were evaluated on test cases, and no test metrics were computed or reported. Historical project audits that previously verified manifest presence or hashes are documented separately in earlier project records.

---

## G. Audit Verification, Inspected Files, and Final Closeout

### 1. Verification of Existing Artifact Integrity
All existing checkpoint files, histories, and configurations were verified via SHA-256 hash checks. Zero existing experiment files were modified, overwritten, or renamed during this audit.

### 2. Exact Files Inspected During Audit:
- **ResUNet-2D:**
  - `results/full_training_15ep/resunet_2d/config.json`
  - `results/full_training_15ep/resunet_2d/summary.json`
  - `results/full_training_15ep/resunet_2d/history.csv`
  - `results/full_training_15ep/resunet_2d/best_model.pt`
  - `results/full_training_15ep/resunet_2d/final_model.pt`
- **UNetResNet34:**
  - `results/full_training_15ep/unet_resnet34/config.json`
  - `results/full_training_15ep/unet_resnet34/summary.json`
  - `results/full_training_15ep/unet_resnet34/history.csv`
  - `results/full_training_15ep/unet_resnet34/best_model.pt`
  - `results/full_training_15ep/unet_resnet34/final_model.pt`
- **Standard U-Net:**
  - `results/stage_c/phase1/unet_721_seed42/config.json`
  - `results/stage_c/phase1/unet_721_seed42/summary.json`
  - `results/stage_c/phase1/unet_721_seed42/history.csv`
  - `results/stage_c/phase1/unet_721_seed42/best_model.pt`
  - `results/stage_c/phase1/unet_721_seed42/final_model.pt`
- **U-Net++:**
  - `results/stage_c/phase1/unetpp_721_seed42/config.json`
  - `results/stage_c/phase1/unetpp_721_seed42/summary.json`
  - `results/stage_c/phase1/unetpp_721_seed42/history.csv`
  - `results/stage_c/phase1/unetpp_721_seed42/best_model.pt`
  - `results/stage_c/phase1/unetpp_721_seed42/final_model.pt`
- **Attention U-Net:**
  - `results/stage_c/phase1/attention_unet_721_seed42/config.json`
  - `results/stage_c/phase1/attention_unet_721_seed42/summary.json`
  - `results/stage_c/phase1/attention_unet_721_seed42/history.csv`
  - `results/stage_c/phase1/attention_unet_721_seed42/best_model.pt`
  - `results/stage_c/phase1/attention_unet_721_seed42/final_model.pt`
- **Source Code & Project Records:**
  - `src/training/train_scaled.py`
  - `src/training/run_overnight_unet_resnet34.py`
  - `src/training/train.py`
  - `MASTER_PROJECT_BLUEPRINT_v2.0.md`
  - `STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md`
  - `STAGE_C_DECISION_REGISTER.csv`
  - `G1_P4_UNET_STAGE_C_TRAINING_REPORT.md`
  - `G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md`
  - `G1_P4_ATTENTION_UNET_STAGE_C_TRAINING_REPORT.md`
  - `G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md`

### 3. Report Modified:
- [G1_P4_TRAINING_HORIZON_REVIEW.md](file:///D:/MajorProject/G1_P4_TRAINING_HORIZON_REVIEW.md)

### 4. Git Status Reporting:
- Working directory clean of any tracked repository modifications beyond previously existing development work. No commits, staging, branch switches, or resets performed.

---
**END OF REPORT — EXECUTION HALTED PER PROTOCOL STOP CONDITION.**
