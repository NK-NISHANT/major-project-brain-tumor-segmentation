# Reliability-Aware AI Agent for Brain Tumor Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1%2Bcu128-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11.9-3776AB.svg?logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-325%20Passed-brightgreen.svg)]()
[![Blueprint](https://img.shields.io/badge/Blueprint-v2.0%20Canonical-blue.svg)](MASTER_PROJECT_BLUEPRINT_v2.0.md)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

> **Full Project Title:** *Reliability-Aware AI Agent for Brain Tumor Segmentation Using Multimodal MRI, 2D CNNs, and Selective Uncertainty Calibration*
> **Institutional Affiliation:** Department of Computer Science & Engineering, Jaypee Institute of Information Technology (JIIT), Noida
> **Project Team:** Nishant Kumar, Mohammad Kaif, Tariq (B.Tech CSE 2023–2027)
> **Domain:** Artificial Intelligence / Machine Learning in Healthcare
> **Canonical Authority:** [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md)

---

## 1. Executive Summary

Standard deep neural networks applied to medical image segmentation produce uncalibrated probability estimates—frequently exhibiting dangerous overconfidence on ambiguous tumor margins, corrupted slices, or out-of-distribution scan conditions.

This project develops an **end-to-end reliability-aware AI agent system** that couples a portfolio of five deep 2D convolutional architectures with **rigorous post-hoc probability calibration** and **selective prediction policies**. Rather than forcing a static segmentation on every input, the system:
1. Ingests 4-channel multimodal MRI volumes (T1n, T1c, T2w, T2f).
2. Performs multi-class segmentation across glioblastoma sub-compartments (Whole Tumor, Tumor Core, Enhancing Tumor).
3. Produces well-calibrated voxel-level probabilities via learned temperature scaling (reducing Expected Calibration Error by 46%–60%).
4. Computes predictive entropy and multi-axial consistency metrics to determine whether the segmentation is clinically trustworthy or should be flagged for expert review.

```text
Multimodal MRI Input (T1n, T1c, T2w, T2f)
                  ↓
       [Input Integrity Gate]
                  ↓
       [5-Model Deep Architecture Portfolio]
       (ResUNet, UNetResNet34, UNet, UNet++, AttnUNet)
                  ↓
       [Temperature Scaling Calibration]  -->  46%-60% ECE Reduction (Argmax Invariant)
                  ↓
       [Reliability & Uncertainty Evaluation]
       (Predictive Entropy, Multi-Axial Consistency, Slice Variance)
                  ↓
       [Agentic Decision Policy]
                  ↓
  ┌───────────────────────┬────────────────────────┐
  ↓                       ↓                        ↓
[Autonomous Accept]  [Model Switch / Fallback]  [Clinical Escalation]
```

---

## 2. Five-Model Architecture Portfolio & Calibration Baseline

All models are trained for 15 epochs on subject-disjoint BraTS 2023 GLI partitions under identical optimizer, learning rate schedule, and loss formulations (Dice + Cross-Entropy). Post-hoc temperature scaling is optimized on a dedicated 50-case calibration cohort (`Val-Cal`, 5,020,730 foreground voxels) and evaluated on a separate 105-case validation evaluation cohort (`Val-Eval`). *(Note: `Val-Eval` is a validation evaluation partition, strictly distinct from the quarantined test set).*

| Model Architecture | Parameters | Optimal $T$ | Calibrated ECE (Val-Eval) | ECE Reduction | Mean Volumetric Dice | Whole Tumor (WT) | Tumor Core (TC) | Enhancing Tumor (ET) | Test Set Status |
|---|---|---|---|---|---|---|---|---|---|
| **ResUNet-2D** | 8,114,604 | 2.8805 | **0.0341** | **60.32%** | **0.8641** | **0.9039** | **0.8597** | 0.8286 | `PENDING_AUTH` |
| **UNetResNet34** | 24,327,524 | 2.6261 | 0.0496 | 54.78% | 0.8535 | 0.8843 | 0.8423 | 0.8339 | `PENDING_AUTH` |
| **U-Net (Classical)** | 7,763,428 | 2.6268 | 0.0483 | 55.46% | 0.8574 | 0.8900 | 0.8527 | 0.8294 | `PENDING_AUTH` |
| **U-Net++** | 9,160,068 | 2.5647 | 0.0551 | 50.70% | 0.8549 | 0.8830 | 0.8521 | 0.8297 | `PENDING_AUTH` |
| **Attention U-Net** | 7,852,160 | 2.2869 | 0.0568 | 47.47% | 0.8581 | 0.8865 | 0.8476 | **0.8400** | `PENDING_AUTH` |

*Note: All test-cohort evaluations remain strictly locked under double-blind quarantine until formal Gate 6B authorization.*

---

## 3. Repository Governance & Companion Records

Per Section 22 of the canonical blueprint, the project maintains seven synchronized companion records at the root:

| Companion Record | Primary Function |
|---|---|
| [PROJECT_STATUS.md](file:///D:/MajorProject/PROJECT_STATUS.md) | Living status, completed milestone matrix, active branch state, test quarantine status |
| [MODEL_AND_EXPERIMENT_REGISTRY.csv](file:///D:/MajorProject/MODEL_AND_EXPERIMENT_REGISTRY.csv) | 36-column verified structured register of all model candidates, optimal temperatures, HD95, and provenance paths |
| [ARCHITECTURE_AND_RESEARCH_DECISIONS.md](file:///D:/MajorProject/ARCHITECTURE_AND_RESEARCH_DECISIONS.md) | Narrative companion to `STAGE_C_DECISION_REGISTER.csv` detailing rationale and math |
| [EVIDENCE_AND_ARTIFACT_INDEX.md](file:///D:/MajorProject/EVIDENCE_AND_ARTIFACT_INDEX.md) | Cryptographic map of reports, dossiers, figures, split manifests, and SHA-256 hashes |
| [PROJECT_CHANGELOG.md](file:///D:/MajorProject/PROJECT_CHANGELOG.md) | Dated chronological record of scope, research baseline, code, and document changes |
| [README.md](file:///D:/MajorProject/README.md) | Setup, test execution, demo instructions, architecture overview |
| [REPOSITORY_INVENTORY_READONLY.md](file:///D:/MajorProject/REPOSITORY_INVENTORY_READONLY.md) | Read-only tree, environment, directory footprint, and ResUNet-2D 110-file census |

The authoritative structured decision register is [STAGE_C_DECISION_REGISTER.csv](file:///D:/MajorProject/STAGE_C_DECISION_REGISTER.csv).

---

## 4. Environment & Installation

### Prerequisites
- **Operating System:** Windows 10/11 or Linux
- **Python:** Version 3.11.x (tested on 3.11.9)
- **CUDA Device:** NVIDIA GPU with $\ge 8$ GB VRAM (tested on NVIDIA GeForce RTX 4070 Laptop GPU)
- **Virtual Environment:** Recommended `.venv`

### Setup Instructions
```powershell
# Clone the repository
git clone https://github.com/NK-NISHANT/major-project-brain-tumor-segmentation.git
cd major-project-brain-tumor-segmentation

# Activate existing project virtual environment
.\.venv\Scripts\Activate.ps1

# Verify Python and PyTorch GPU availability
python -c "import torch; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 5. Verification & Test Suite

The repository contains an automated test suite covering model architectures, data loaders, HD95/Dice metrics, temperature calibration routines, selective prediction policies, and the Streamlit demo application.

Execute the full regression suite:
```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
**Passing Baseline:** Exactly **325 passed, 8 warnings**.

---

## 6. Interactive Clinical Demonstration App

The project includes an interactive Streamlit application allowing users to inspect multimodal MRI slices, execute real-time model inference, view calibrated probability maps, inspect predictive entropy heatmaps, and test selective prediction thresholds.

Launch the demo application:
```powershell
.\.venv\Scripts\python.exe -m streamlit run tools/demo_app.py
```

The web interface will launch locally at `http://localhost:8501`.

---

## 7. Model Dossiers & Reporting Tools

Model dossiers and high-resolution publication figures are generated using decoupled, single-model tools in `tools/`:
- **ResUNet-2D Dossier:** `.\.venv\Scripts\python.exe tools/build_resunet_dossier.py`
- **U-Net Dossier & Figures:** `.\.venv\Scripts\python.exe tools/build_unet_dossier_docx_pdf.py`
- **U-Net++ Dossier & Figures:** `.\.venv\Scripts\python.exe tools/build_unetpp_dossier.py`
- **U-Net++ Asset Generator:** `.\.venv\Scripts\python.exe tools/generate_unetpp_assets.py`

Under **Option A Binary Tracking Policy**, all Markdown dossiers, Python scripts, CSV tables, and PNG/SVG publication figures in `reports/model_profiles/` are tracked in version control, while compiled Word and PDF binaries are maintained locally.

---

## 8. Academic & Medical Disclaimers

> [!CAUTION]
> **Not for Diagnostic or Clinical Use:** This software and all accompanying models, dossiers, and predictions are developed solely for academic research and engineering demonstration as part of a B.Tech Major Project at JIIT Noida. They have **not** been cleared or approved by any medical regulatory authority (such as CDSCO, FDA, or EMA). Under no circumstances should these segmentation outputs or reliability scores be used for patient diagnosis, surgical planning, or direct clinical decision-making.
