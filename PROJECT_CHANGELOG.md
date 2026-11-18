# Project Changelog

All notable changes, architectural milestones, experimental protocols, and repository governance actions for the **Reliability-Aware AI Agent for Brain Tumor Segmentation** project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to the governance mandates of [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md).

---

## [2026-11-19] - Governance Stage 4B.1: Results Boundary Remediation & Governance Verification

### Changed
- Refined `.gitignore` results boundary: removed blanket `/results/` ignore rule to preserve trackable research evidence (`.csv`, `.json`, `.md`, `.png`). Narrowed local-only ignore rules strictly to neural net weights (`*.pt`, `*.pth`, etc.), local checkpoint directories (`results/checkpoints/`, `results/models/`), and intermediate cache directories (`results/**/case_voxels/`, `results/**/recovery_snapshot_*/`).
- Audited Option A eligibility across all 135 deliverable files: strictly restricted approved trackable categories to `.md`, `.py`, `.png`, `.svg`, `.csv`; classified `resunet_2d/source/generate_explainer_deck.ps1` as `REVIEW_REQUIRED (Exception: PowerShell Script Pending Owner Authorization)`.
- Reconciled `MODEL_AND_EXPERIMENT_REGISTRY.csv`: verified all 36 columns across 6 model rows against authoritative JSON manifests and CSV logs (`c2_run_manifest.json`, `evaluation_metrics.csv`, `calibration_parameters.json`, `hd95_summary.json`). Explicitly labeled evaluation cohort as `Val-Eval (105 cases)`, set unmeasured HD95 fields to `NOT_RECORDED`, and kept sealed test cohort strictly `PENDING_AUTHORIZATION`.
- Reconciled companion records: verified `README.md` as modified/updated from historical roadmap, updated `REPOSITORY_INVENTORY_READONLY.md` with immutable audit timestamp (`2026-11-19T01:55:00+05:30`), and maintained `STAGE_C_DECISION_REGISTER.csv` as authoritative structured decision register.
- Executed full automated regression testing: 325 passed, 8 warnings in 22.73s. Confirmed zero staged changes (`git diff --cached` empty).

---

## [2026-11-19] - Governance Stage 4B: Companion Records & Selective Git Ignore Policy

### Added
- Authored the seven canonical Section 22 companion records at repository root:
  - `PROJECT_STATUS.md`: Living status, milestone matrix, test quarantine status.
  - `MODEL_AND_EXPERIMENT_REGISTRY.csv`: 25-column structured register of all 5 canonical models plus baseline.
  - `ARCHITECTURE_AND_RESEARCH_DECISIONS.md`: Narrative companion to `STAGE_C_DECISION_REGISTER.csv`.
  - `EVIDENCE_AND_ARTIFACT_INDEX.md`: Provenance map with verified SHA-256 hashes.
  - `PROJECT_CHANGELOG.md`: Chronological log of all project milestones.
  - `README.md`: Updated top-level architectural overview, setup, test and demo commands.
  - `REPOSITORY_INVENTORY_READONLY.md`: Post-restructuring inventory and directory census.
- Conducted exhaustive 110-file audit and classification of `reports/model_profiles/resunet_2d/`.

### Changed
- Consolidated `.gitignore`:
  - Deduplicated repeated cache and lock file blocks.
  - Added explicit directory-level ignore guards for `/documents/`, `/data/archives/`, `/data/extracted_archive/`, `/pre_tx_backups*/`, `/scratch/`.
  - Implemented **Option A Lightweight Binary Tracking Policy**: tracked `.md`, `.py`, `.png`, `.svg`, `.csv`, `.json`; ignored `.docx`, `.pdf`, `.pptx` under `reports/model_profiles/`.
- Verified zero tracked files shadowed and zero files staged.
- Verified test suite: 325 passed, 8 warnings.

---

## [2026-11-18] - Governance Stage 4A: Read-Only Repository & Git Audit

### Added
- Completed read-only repository governance audit (`STAGE_4A_REPOSITORY_GOVERNANCE_AUDIT.md`).
- Audited branch relationship (`main` at `136b5b9`, ahead of `origin/main` at `64c5ff5` by 1 commit).
- Inventoried 16 large files (> 100 MB), 135 files in `reports/`, and 176 untracked files.
- Formulated Option A vs Option B binary tracking recommendations.

---

## [2026-11-18] - Restructuring Stage 3B: Decoupled Dossier Tools & Archive Rescue

### Added
- Rescued 3 unique historical U-Net++ assets to `reports/model_profiles/unetpp/.../historical/` with verified SHA-256 hashes:
  - `UNetPlusPlus_Master_Model_Dossier_20261118.docx` (`5ccae9f2...`)
  - `UNetPlusPlus_Master_Model_Dossier_20261118.pdf` (`f017db83...`)
  - `UNetPlusPlus_Executive_Summary_Printout_20261118.pdf` (`11b8fb3e...`)
- Emitted 14 publication figures directly into `reports/model_profiles/` (7 for U-Net, 7 for U-Net++).

### Changed
- Refactored four generation scripts (`tools/build_resunet_dossier.py`, `tools/build_unet_dossier_docx_pdf.py`, `tools/build_unetpp_dossier.py`, `tools/generate_unetpp_assets.py`) to decouple cross-model dependencies.
- Created pre-transaction backup `pre_tx_backups_20261118_180800`.
- Verified test regression suite: 325 passed, 8 warnings.

---

## [2026-11-18] - Restructuring Stage 3A: Permanent Cancellation

### Changed
- Formally cancelled proposed relocation of eight Gate 1 milestone reports (`G1_*`).
- Retained all eight `G1_*` reports at repository root to preserve external review links and citation integrity.

---

## [2026-11-18] - Restructuring Stage 2: Submissions Folder Organization

### Changed
- Cataloged 13 files in `submissions/`.
- Preserved 12 retained assets and institutional submission drafts without breaking local paths.

---

## [2026-11-17] - Restructuring Stage 1: Raw BraTS ZIP Relocation & Blueprint v2.0

### Added
- Formally adopted `MASTER_PROJECT_BLUEPRINT_v2.0.md` as canonical repository authority.
- Relocated 14.8 GB raw archive `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` to `data/archives/brats2023/`.
- Verified data integrity via SHA-256 pre- and post-move verification.

---

## [2026-11-16] - Stage C Phase 3 (C3): Selective Prediction & Streamlit Demo

### Added
- Implemented selective prediction policy evaluating predictive entropy and slice variance.
- Built interactive Streamlit demo application (`tools/demo_app.py`) for clinical visualization and reliability exploration.
- Authored test suites `tests/test_demo_app_agent.py` and `tests/test_demo_app_end_to_end.py`.

---

## [2026-11-15] - Stage C Phase 2 (C2): Five-Model Probability Calibration

### Added
- Conducted post-hoc temperature scaling across all 5 models on 50-case calibration cohort (`Val-Cal`, 5,020,730 foreground voxels).
- Learned optimal temperatures: ResUNet-2D ($T = 2.8805$), UNetResNet34 ($T = 2.6261$), U-Net ($T = 2.6268$), U-Net++ ($T = 2.5647$), Attention U-Net ($T = 2.2869$).
- Achieved 46.0%–60.3% ECE reduction across all models with 100% mathematical argmax invariance.
- Published [G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md](file:///D:/MajorProject/G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md).
- Quarantined 44 held-out test cases (`results/dataset_splits/test_manifest.json`).

---

## [2026-11-05] - Stage C Phase 1: Five-Model Portfolio Training

### Added
- Scaled training to 5 deep learning architectures: ResUNet-2D, UNetResNet34, U-Net, U-Net++, Attention U-Net.
- Standardized 15-epoch training protocol with AdamW optimizer, cosine annealing schedule, and combined Dice + Cross-Entropy loss.
- Established subject-disjoint dataset partitions to prevent patient leakage.
- Published comparative benchmark report [G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md](file:///D:/MajorProject/G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md).

---

## [2026-10-15] - Master Project Roadmap Establishment

### Added
- Created foundational project roadmap and experiment plan (`archive/project_roadmap/`).
- Formalized engineering constraints: PyTorch only, no clinical validity claims, controlled seed and hardware tracking.

---

## [2026-09-26] - Phase A: Mid-Semester Showcase Baseline

### Added
- Initial BraTS 2023 GLI dataset audit (198 cases total; 197 complete, 1 incomplete).
- Baseline 2D U-Net segmentation pipeline and exploratory predictive uncertainty analysis.
