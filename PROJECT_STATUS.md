# Project Status: Reliability-Aware AI Agent for Brain Tumor Segmentation

- **Project Title:** Reliability-Aware AI Agent for Brain Tumor Segmentation Using Multimodal MRI, 2D CNNs, and Selective Uncertainty Calibration
- **Institutional Affiliation:** Jaypee Institute of Information Technology (JIIT), Noida — Department of Computer Science & Engineering
- **Investigators:** Nishant Kumar, Mohammad Kaif, Tariq (B.Tech CSE 2023–2027)
- **Repository URL:** `https://github.com/NK-NISHANT/major-project-brain-tumor-segmentation`
- **Canonical Blueprint:** [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md) (Sections 15.2, 22, 23)
- **Current Governance Stage:** **Stage 4B.1 — Governance Remediation & Verification**
- **Date:** 2026-11-19
- **Overall Project Health:** **HEALTHY / CONTROLLED**

---

## 1. Executive Summary & Research Baseline

The project develops an uncertainty-aware, reliability-guided AI agent for multimodal brain tumor segmentation under heterogeneous MRI conditions. Moving beyond standard static segmentation model benchmarks, this system integrates:
1. A **5-model deep learning portfolio** spanning distinct inductive biases (ResUNet-2D, UNetResNet34, U-Net, U-Net++, Attention U-Net).
2. **Post-hoc probability calibration** via temperature scaling on a dedicated 50-case calibration cohort (`Val-Cal`), reducing Expected Calibration Error (ECE) across all architectures by 46.0%–60.3% while strictly preserving argmax segmentation accuracy.
3. An **agentic selective prediction policy** operating on predictive entropy, slice-level uncertainty variance, and multi-axial consistency to detect failure modes and request expert clinician review or fallback strategy switching.

---

## 2. Milestone Progress & Governance Tracker

| Milestone / Stage | Scope & Description | Status | Evidence / Artifact |
|---|---|---|---|
| **Phase A** | Dataset exploration, baseline U-Net, BraTS 2023 GLI audit | **CLOSED** | [README.md](file:///D:/MajorProject/README.md), `data/metadata/` |
| **Phase B** | Uncertainty estimation & failure case characterization | **CLOSED** | `src/uncertainty/`, `results/baseline_unet2d/` |
| **Stage C Phase 1** | 5-Model portfolio training (15 epochs, Dice+CE loss) | **CLOSED** | [G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md](file:///D:/MajorProject/G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md) |
| **Stage C Phase 2 (C2)** | Five-model probability calibration & uncertainty benchmarking | **CLOSED** | [G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md](file:///D:/MajorProject/G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md) |
| **Stage C Phase 3 (C3)** | Selective prediction, abstention policies & Streamlit demo | **CLOSED** | `tools/demo_app.py`, `tests/test_demo_app_*.py` |
| **Restructuring Stage 1** | Raw BraTS ZIP relocation to safe archive | **CLOSED** | `data/archives/brats2023/` (14.8 GB archived) |
| **Restructuring Stage 2** | Submissions folder organization & asset preservation | **CLOSED** | `submissions/` (13 cataloged, 12 retained assets) |
| **Restructuring Stage 3A** | Proposed relocation of Gate 1 milestone reports | **CANCELLED** | All 8 `G1_*` reports permanently retained at root |
| **Restructuring Stage 3B** | Decoupled dossier generation & U-Net++ archive rescue | **CLOSED** | `reports/model_profiles/` (14 figures, 325 tests passed) |
| **Governance Stage 4A** | Read-only repository & Git audit | **CLOSED** | `STAGE_4A_REPOSITORY_GOVERNANCE_AUDIT.md` |
| **Governance Stage 4B** | Seven root companion records & Option A Git ignore | **CLOSED (CONDITIONAL PASS)** | Addressed in Stage 4B.1 |
| **Governance Stage 4B.1** | Results boundary remediation, Option A review, provenance audit | **COMPLETED & VERIFIED** | `.gitignore` refined, registry verified, 325 tests passed |
| **Governance Stage 4C** | Baseline GitHub synchronization | **HOLD (PENDING OWNER CONFIRMATION)** | Staged curated commits, push to `origin/main` |
| **Gate 6B** | Final held-out test cohort locked evaluation | **LOCKED / SEALED** | `results/dataset_splits/test_manifest.json` |

---

## 3. Working Tree & Git State

- **Current Branch:** `main`
- **Local HEAD Commit:** `136b5b9` (*"Checkpoint: preserve current project progress"*, Nov 15, 2026).
- **Remote HEAD Commit (`origin/main`):** `64c5ff5` (*"Merge remote cleanup commits"*, Oct 23, 2026). Note: Remote-tracking reference is stale; no remote operations permitted in Stage 4B/4B.1.
- **Working Tree Cleanliness:** 
  - Staged changes: **0 files** (`git diff --cached` is strictly empty).
  - Modified tracked files: 11 files (`.gitignore`, `README.md` [updated from historical roadmap], `src/agent/models.py`, `src/agent/policy.py`, `src/agent/runner.py`, `src/agent/state.py`, `src/agent/test_axial_context.py`, `src/agent/test_full_volume.py`, `tests/test_demo_app_agent.py`, `tests/test_demo_app_end_to_end.py`, `tools/demo_app.py`).
  - Active untracked companion records: 6 files (`PROJECT_STATUS.md`, `MODEL_AND_EXPERIMENT_REGISTRY.csv`, `ARCHITECTURE_AND_RESEARCH_DECISIONS.md`, `EVIDENCE_AND_ARTIFACT_INDEX.md`, `PROJECT_CHANGELOG.md`, `REPOSITORY_INVENTORY_READONLY.md`).
  - Active untracked directories containing trackable research evidence: `reports/`, `results/`, `tools/`, `tests/`.
  - Ignored directories: `documents/` (22 legacy files preserved locally), `data/archives/` (14.8 GB), `.venv/`, `scratch/`.
- **Merge/Rebase In Progress:** None (Zero conflicts, zero interrupted merges).

---

## 4. Dataset Cohort & Quarantine Invariants

| Cohort | Case Count | Subject Count | Slice / Voxel Scope | Purpose & Scientific Definition | Access Status |
|---|---|---|---|---|---|
| **Training Cohort** | 104 cases | 98 subjects | ~13,000 axial slices | Model parameter optimization (15 epochs) | Training closed |
| **Calibration Cohort (Val-Cal)** | 50 cases | 47 subjects | 5,020,730 foreground voxels | Post-hoc temperature optimization (NLL minimization) | Calibration closed |
| **Evaluation Cohort (Val-Eval)** | 105 cases | 96 subjects | 311,546 evaluated voxels | **Validation Evaluation Cohort**: Used for comparative calibration and segmentation benchmarking. *(NOT an independent test set).* | Benchmark frozen |
| **Quarantined Test Cohort** | 44 cases | 44 subjects | Sealed volumes | **Held-Out Test Set**: Preserved for final one-time Gate 6B evaluation. | **LOCKED / SEALED** |

> [!IMPORTANT]
> **Test Cohort Quarantine Invariant:** The test cohort defined in `results/dataset_splits/test_manifest.json` (SHA-256: `694faaa4b09ec4d173c2ea661fa567a5f6e62551a37c4ec140ff4985eeaeaa0a`) and `results/dataset_splits/final_split_manifest.json` (SHA-256: `01a32c7fdc96644458f338d4e9ad1cf5cb52f7547926210f274cbba1db98a004`) has **NEVER BEEN EVALUATED**. No inference scripts have accessed these cases. In all project tables, registries, and reports, test-set metrics remain strictly **`PENDING_AUTHORIZATION`** or **`N/A`**.

---

## 5. Active Test & Regression Gate Baseline

- **Test Framework:** `pytest 8.3.4` (invoked via `.\.venv\Scripts\python.exe -m pytest -q`)
- **Regression Suite Passing Baseline:** **325 passed, 8 warnings in 22.73s**
- **Test Invariants:**
  - Zero test regressions across model factories, calibration routines, selective prediction policies, and Streamlit demo components.
  - All warnings are non-fatal upstream deprecations (`torch.jit.interface`, MONAI `get_mask_edges`, LR scheduler step ordering in test fixtures).
