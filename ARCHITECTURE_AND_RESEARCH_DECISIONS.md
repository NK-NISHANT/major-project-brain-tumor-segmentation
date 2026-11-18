# Architecture & Research Decisions: Narrative Companion

- **Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
- **Document Status:** Authoritative Narrative Companion
- **Structured Source of Truth:** [STAGE_C_DECISION_REGISTER.csv](file:///D:/MajorProject/STAGE_C_DECISION_REGISTER.csv) (13.1 KB, root)
- **Canonical Blueprint:** [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md) (Section 22)
- **Date:** 2026-11-19

---

## 1. Relationship to Structured Decision Register

This document provides the detailed technical rationale, theoretical justification, empirical evidence, and risk analysis for the architectural and methodological choices in this project. The authoritative, machine-readable register of record remains [STAGE_C_DECISION_REGISTER.csv](file:///D:/MajorProject/STAGE_C_DECISION_REGISTER.csv). This companion is maintained in lockstep with the CSV register and the canonical project blueprint.

```text
STAGE_C_DECISION_REGISTER.csv (Structured Record: IDs, Dates, Owners, Status)
       ↕ Lockstep Parity
ARCHITECTURE_AND_RESEARCH_DECISIONS.md (Narrative Companion: Rationale, Math, Evidence, Alternatives)
```

---

## 2. Core Architectural & Methodological Decisions

### Decision ARD-01: Multimodal 4-Channel Input Representation & Integrity Gate
- **Decision:** Standardize input to 4 core MRI modalities: T1-native (T1n), T1-contrast enhanced (T1c), T2-weighted (T2w), and T2-FLAIR (T2f) at shape $(4, 240, 240)$ per axial slice. Implement an upstream input-integrity gate that checks modality ordering, voxel value sanity, and missing channels prior to model inference.
- **Rationale:** Glioblastoma sub-compartments exhibit complementary radiological signatures across modalities: necrotic core and active tumor enhance on T1c; peritumoral edema is prominently hyperintense on T2-FLAIR; gross structural abnormalities are visible on T1n and T2w.
- **Alternatives Considered:** 
  1. *Single-modality or dual-modality inputs:* Rejected due to severe clinical under-segmentation of non-enhancing core and edema.
  2. *Latent multimodal fusion networks:* Rejected for initial phase due to high parameter complexity and difficulty in debugging modality failure modes.
- **Empirical Evidence:** Multimodal input achieves baseline mean Dice > 0.85 across all models, whereas missing-modality degradation studies demonstrate performance drops of 15%–40% if T1c or T2-FLAIR is omitted.
- **Owner & Date:** AI/ML Team | Stage C Inception (2026-11-05).

---

### Decision ARD-02: 4-Class Tumor Compartmentalization & Hierarchical Aggregation
- **Decision:** Train models on 4-class mutually exclusive voxel labels: Background (0), Necrotic/Non-enhancing Core (NCR/NET, 1), Peritumoral Edema (ED, 2), and Enhancing Tumor (ET, 3). For clinical evaluation, aggregate probabilities into the standard BraTS hierarchical sub-regions:
  - **Whole Tumor (WT):** Classes $1 + 2 + 3$ (Core + Edema + Enhancing)
  - **Tumor Core (TC):** Classes $1 + 3$ (Necrotic Core + Enhancing)
  - **Enhancing Tumor (ET):** Class $3$
- **Rationale:** Direct 4-class multi-class softmax training prevents label conflicts during backpropagation, while post-processing hierarchical evaluation aligns with neurosurgical resection planning (gross total resection targets WT/TC; biopsy targets ET).
- **Alternatives Considered:** 
  1. *3 independent binary sigmoid networks:* Rejected due to higher training cost (3x training passes) and probability inconsistencies where sub-compartments exceed whole tumor predictions.
- **Empirical Evidence:** Hierarchical validation Dice consistently exceeds 0.85 (WT: 0.88–0.90, TC: 0.84–0.86, ET: 0.82–0.84) across the 5-model portfolio.
- **Owner & Date:** AI/ML Team | Stage C Inception (2026-11-05).

---

### Decision ARD-03: Five-Model Deep Learning Portfolio
- **Decision:** Establish a benchmark portfolio of five distinct 2D convolutional architectures rather than relying on a single architecture:
  1. **ResUNet-2D (8,114,604 params):** Residual blocks in both encoder and decoder paths.
  2. **UNetResNet34 (24,327,524 params):** Heavyweight ResNet-34 backbone encoder with standard decoder.
  3. **U-Net (7,763,428 params):** Classical symmetric contracting/expanding path baseline.
  4. **U-Net++ (9,160,068 params):** Dense, nested skip connections reducing semantic gap between feature maps.
  5. **Attention U-Net (7,852,160 params):** Additive spatial attention gates filtering irrelevant background activations.
- **Rationale:** Inductive bias diversity is essential for reliability-aware agentic systems. A single architecture cannot provide meaningful disagreement or ensemble consensus signals. Comparing residual learning, dense nesting, heavy backbones, and attention gates provides a rich feature space for uncertainty analysis.
- **Alternatives Considered:** 
  1. *Only 1 or 2 architectures:* Rejected because failure modes would be model-specific rather than generalizable.
  2. *3D volumetric networks (e.g. 3D U-Net, Swin UNETR):* Deferred to later phases due to GPU memory constraints (RTX 4070 Laptop GPU 8GB VRAM) during multi-model calibration and multi-axial sampling.
- **Empirical Evidence:** Detailed in [G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md](file:///D:/MajorProject/G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md) and [G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md](file:///D:/MajorProject/G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md).
- **Owner & Date:** AI/ML Team | 2026-11-08.

---

### Decision ARD-04: Subject-Disjoint Dataset Partitioning
- **Decision:** Enforce strict subject-level isolation across all dataset splits. Scans from the same patient across longitudinal timepoints must never be split across training, calibration, validation, or test cohorts.
- **Rationale:** Intra-subject anatomical similarity between repeated scans causes severe data leakage, resulting in inflated performance metrics that fail to generalize to novel clinical subjects.
- **Partitioning Breakdown:**
  - *Training Cohort:* 104 cases (98 subjects)
  - *Calibration Cohort (Val-Cal):* 50 cases (47 subjects)
  - *Validation Evaluation Cohort (Val-Eval):* 105 cases (96 subjects) — used for comparative benchmark evaluations; explicitly distinct from the held-out test set.
  - *Held-Out Test Cohort:* 44 cases (44 subjects) — sealed under double-blind quarantine.
- **Provenance Manifest:** `results/dataset_splits/final_split_manifest.json` (SHA-256: `01a32c7fdc96644458f338d4e9ad1cf5cb52f7547926210f274cbba1db98a004`).
- **Owner & Date:** AI/ML Team | 2026-11-10.

---

### Decision ARD-05: Post-Hoc Probability Calibration via Temperature Scaling
- **Decision:** Implement post-hoc temperature scaling ($T > 0$) applied to the model logit vector $z(x)$:
  $$\hat{p}_k(x; T) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^K \exp(z_j(x) / T)}$$
  The optimal temperature $T^*$ is learned by minimizing Negative Log-Likelihood (NLL) using L-BFGS on a dedicated 50-case calibration cohort (`Val-Cal`, 5,020,730 foreground voxels), with model weights strictly frozen.
- **Mathematical & Clinical Rationale:**
  1. **Argmax Invariance:** Because scaling all logits by a single positive scalar $T$ does not change their relative ordering:
     $$\arg\max_k \left(\frac{z_k}{T}\right) = \arg\max_k z_k \quad \forall T > 0$$
     Therefore, **temperature scaling preserves segmentation masks, volumetric contours, and Dice scores with 100% mathematical precision**.
  2. **ECE Reduction:** Uncalibrated deep neural networks are notoriously overconfident (ECE 8.6%–11.2%). Temperature scaling aligns predicted confidence with true empirical frequency.
- **Empirical Results (Measured on Validation Evaluation Cohort `Val-Eval`):**
  - ResUNet-2D ($T = 2.8805$): ECE reduced from $0.0859$ to $0.0341$ (**60.32% reduction**).
  - UNetResNet34 ($T = 2.6261$): ECE reduced from $0.1098$ to $0.0496$ (**54.78% reduction**).
  - U-Net ($T = 2.6268$): ECE reduced from $0.1085$ to $0.0483$ (**55.46% reduction**).
  - U-Net++ ($T = 2.5647$): ECE reduced from $0.1118$ to $0.0551$ (**50.70% reduction**).
  - Attention U-Net ($T = 2.2869$): ECE reduced from $0.1081$ to $0.0568$ (**47.47% reduction**).
- **Owner & Date:** AI/ML Team | Stage C Phase 2 (2026-11-15).

---

### Decision ARD-06: Strict Test Cohort Quarantine
- **Decision:** Permanently lock and isolate the 44-case held-out test cohort manifest (`results/dataset_splits/test_manifest.json`, SHA-256: `694faaa4b09ec4d173c2ea661fa567a5f6e62551a37c4ec140ff4985eeaeaa0a`). Prohibit all test-set inference until explicit Gate 6B authorization.
- **Rationale:** Strict double-blind separation prevents metric hacking, inadvertent test-leakage, and subtle hyperparameter overfitting.
- **Implementation Rule:** In all registries, tables, and reports, test-set metrics must remain explicitly marked **`PENDING_AUTHORIZATION`** or **`N/A`**. Any fabricated or premature test metrics constitute an immediate governance violation.
- **Owner & Date:** Project Lead & Supervisor | 2026-11-15.

---

### Decision ARD-07: Option A Lightweight Binary Tracking Policy & Strict Extension Scrutiny
- **Decision:** Adopt Option A for repository version control:
  - **Explicitly Approved Eligible Categories:** Source code (`.py`), Markdown documentation (`.md`), publication figures (`.png`, `.svg`), and structured evidence tables (`.csv`).
  - **Explicitly Approved Local-Only Binary Categories:** Compiled Word documents (`reports/**/*.docx`), PDFs (`reports/**/*.pdf`), and PowerPoint decks (`reports/**/*.pptx`). All local binary files are strictly preserved on disk; zero deletions.
  - **Review Required / Extension Exceptions:** All other extensions (e.g. `.ps1`, `.json`, `.yaml`, `.txt`) must be explicitly justified and approved by the project owner rather than silently included. Specifically, PowerShell scripts (such as `resunet_2d/source/generate_explainer_deck.ps1`) are marked `REVIEW_REQUIRED` pending owner decision.
- **Rationale:** Compiled Word, PDF, and PowerPoint files total > 28 MB and undergo frequent binary regeneration, which causes git database bloat without providing readable diffs. Tracking high-resolution PNG/SVG figures and Markdown reports provides full visual and technical provenance while keeping the Git history lightweight.
- **Owner & Date:** Repository Governance Engineer | Stage 4B (2026-11-19).

---

### Decision ARD-08: Retention of Root Gate 1 Milestone Reports (Stage 3A Cancellation)
- **Decision:** Permanently cancel Stage 3A (proposed relocation of the eight `G1_*` milestone reports to subdirectories). Retain all eight files at the repository root.
- **Rationale:** The `G1_*` reports serve as formal gate audit deliverables referenced by institutional evaluation committees and supervisors. Moving them disrupts external review links and violates Section 15.2 principles against cosmetic restructuring.
- **Owner & Date:** Project Owner | 2026-11-18.

---

### Decision ARD-09: Decoupled Dossier Generation Architecture (Stage 3B)
- **Decision:** Refactor monolithic dossier building scripts into decoupled, single-model generation tools (`tools/build_resunet_dossier.py`, `tools/build_unet_dossier_docx_pdf.py`, `tools/build_unetpp_dossier.py`, `tools/generate_unetpp_assets.py`) that emit directly into `reports/model_profiles/<model_name>/`. Archive three unique historical U-Net++ assets into `reports/model_profiles/unetpp/.../historical/`.
- **Rationale:** Decoupled tools allow independent maintenance and testing of model dossiers without unintended cross-model side-effects. Historical assets are preserved with exact SHA-256 provenance while active scripts regenerate fresh artifacts.
- **Owner & Date:** Senior Lead Developer | Stage 3B (2026-11-18).

---

### Decision ARD-10: Narrowed Results Directory Tracking Boundary (Stage 4B.1 Remediation)
- **Decision:** Remove the blanket `/results/` ignore rule from `.gitignore`. Replace it with narrowly scoped local-only rules for confirmed model checkpoints, large weight files, and intermediate cache dumps:
  - Ignore: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.safetensors`
  - Ignore local cache directories: `results/checkpoints/`, `results/models/`, `results/**/case_voxels/`, `results/**/recovery_snapshot_*/`
  - Preserve as trackable: legitimate scientific research evidence in `results/`, including run manifests (`c2_run_manifest.json`), evaluation metrics (`evaluation_metrics.csv`), calibration parameters (`calibration_parameters.json`), split manifests (`final_split_manifest.json`, `test_manifest.json`), HD95 summaries (`hd95_summary.json`), and training histories (`history.csv`, `config.json`).
- **Rationale:** A blanket ignore on `results/` silently excludes reproducible scientific evidence and machine-readable experiment records from version control. Narrowing the ignore boundary to neural net weights (> 5 GB) preserves full experiment provenance while protecting Git from binary bloat.
- **Owner & Date:** Senior Lead Developer & Governance Engineer | Stage 4B.1 (2026-11-19).
