# Evidence & Artifact Index: Authoritative Provenance Map

- **Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
- **Canonical Blueprint:** [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md) (Section 22)
- **Status:** Active Authoritative Index
- **Date:** 2026-11-19

---

## 1. Overview & Verification Protocol

This index serves as the central directory of verified scientific artifacts, formal milestone reports, model dossiers, split manifests, calibration parameters, and historical archives. All cryptographic hashes are computed using SHA-256 over raw file bytes.

---

## 2. Root Milestone Deliverables (Gate 1 Reports)

Per the Stage 3A cancellation decision, all eight Gate 1 milestone reports remain permanently positioned at the repository root.

| Artifact File | Size | SHA-256 Provenance Hash | Purpose / Coverage |
|---|---|---|---|
| [G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md](file:///D:/MajorProject/G1_C2_FIVE_MODEL_CALIBRATION_REPORT.md) | 22.9 KB | `6819a477f6e5abc1c3d57a83cf0dd26ae684d391d075a7a3936faae9931ea41c` | Stage C Phase 2 five-model probability calibration & ECE reduction |
| [G1_P1_1_DEMONSTRATION_VERIFICATION_CLOSEOUT.md](file:///D:/MajorProject/G1_P1_1_DEMONSTRATION_VERIFICATION_CLOSEOUT.md) | 11.2 KB | `c305497cd951b2fde1b8abc2dc5084bc6749fb411c016f09e9c0b4c1a8973572` | Streamlit demo application & slice-level agent verification closeout |
| [G1_P3_DATASET_TRAINING_READINESS_REPORT.md](file:///D:/MajorProject/G1_P3_DATASET_TRAINING_READINESS_REPORT.md) | 18.6 KB | `70db3e4cf034b0729740fda62250a4f2cf3df159c23534ebdece53364d3f0c1c` | BraTS 2023 GLI training readiness, voxel distributions, data checks |
| [G1_P4_ATTENTION_UNET_STAGE_C_TRAINING_REPORT.md](file:///D:/MajorProject/G1_P4_ATTENTION_UNET_STAGE_C_TRAINING_REPORT.md) | 20.4 KB | `7d47831a28dcf519b1e395102be24636e88603b50de93b03109a86c26f19c538` | Attention U-Net 15-epoch Stage C training trajectory & loss metrics |
| [G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md](file:///D:/MajorProject/G1_P4_PHASE1_COMPARATIVE_ANALYSIS.md) | 22.2 KB | `92856e234a7d52a2e41657a7b64a05021ad109dc2ccb20528488c5dcadda80cd` | Comparative benchmark analysis of 5 architectures post Phase 1 |
| [G1_P4_TRAINING_HORIZON_REVIEW.md](file:///D:/MajorProject/G1_P4_TRAINING_HORIZON_REVIEW.md) | 39.8 KB | `98ec35285dc19d519e3085360dbf1ea29a6b428933cce3b2f83d36f8fdcc201e` | 15-epoch training horizon justification & convergence review |
| [G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md](file:///D:/MajorProject/G1_P4_UNETPP_STAGE_C_TRAINING_REPORT.md) | 16.9 KB | `8f756426a387a141c2f5bfff8354ad05a1cdb027edb8017bf7b398b695ef9b0b` | U-Net++ 15-epoch training performance, skip connection analysis |
| [G1_P4_UNET_STAGE_C_TRAINING_REPORT.md](file:///D:/MajorProject/G1_P4_UNET_STAGE_C_TRAINING_REPORT.md) | 63.0 KB | `d9d88b0a4bce590659a13980f5e379b0dd90df3761f304e1f0d763e050dc15e9` | Classical U-Net 15-epoch baseline comprehensive training audit |

---

## 3. Dataset Splits & Quarantined Test Cohort Manifests

| Manifest Path | Case Count | SHA-256 Hash | Quarantine Status |
|---|---|---|---|
| `results/dataset_splits/test_manifest.json` | 44 | `694faaa4ff100066d67aa5c87c545a709e4eed43a61625d3bea9f43c50d6dfb7` | **LOCKED / SEALED (Gate 6B)** |
| `results/dataset_splits/final_split_manifest.json` | 198 | `01a32c7f9541fee00f479bb8b9d88435eff023cc6d7fd3dcf8e4fa29f0788904` | **LOCKED / SEALED (Gate 6B)** |

> [!WARNING]
> Both split manifests are cryptographic invariants. Zero modifications or test-set inferences are permitted prior to Gate 6B.

---

## 4. Stage C Calibration & Evaluation Artifacts

- **Run Manifest:** `results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json`  
  *SHA-256:* `e56617faf06a605e6c51bde3b117ba0e02f438019ef48b2947edd1b92c758460`
- **Evaluation Metrics (CSV):** `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv`  
  *SHA-256:* `7418b09f68fdd46c29e641d4d3b34094bc7f6d08ce7c5b49e4a8caa983fc87fe`
- **Calibration Parameters (JSON):** `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json`  
  *SHA-256:* `77819fe816255d75446f9497e4b361d5364cb674ff55904dbb52240e3a76f67e`
- **Entropy Summary (CSV):** `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv`  
  *SHA-256:* `7456e78782fe5cac8b59b42e1aab65c5514e23208c6b9e40f755b409a975024f`

---

## 5. Curated Model Profiles & Visual Deliverables (`reports/model_profiles/`)

Tracked under Option A (Markdown, Python scripts, PNG/SVG figures, CSV data):

### 5.1 ResUNet-2D Profile (`reports/model_profiles/resunet_2d/`)
- Total files: 110 (104 trackable under Option A; 6 local-only binary deliverables preserved).
- Key markdown dossiers:
  - `reports/model_profiles/resunet_2d/ResUNet_2D_Model_Dossier_v1.0.md`
  - `reports/model_profiles/resunet_2d/dossier/ResUNet_2D_Model_Dossier_v1.1.md`
  - `reports/model_profiles/resunet_2d/one_page/ResUNet_2D_One_Page_Overview.md`
- Figures: 56 publication-grade figures in `assets/` (e.g. `fig1_dice_by_region.png`, `fig2_entropy_distribution.png`, `fig4_calibration_curve.png`).

### 5.2 U-Net Profile (`reports/model_profiles/unet_2d/`)
- Total files: 10 (7 publication figures, 1 markdown dossier, 2 local-only binaries).
- Canonical dossier: `reports/model_profiles/unet_2d/UNet_2D_Master_Model_Dossier.md`
- Active figures in `reports/model_profiles/unet_2d/assets/`:
  - `fig_01_unet_classical_architecture.png`
  - `fig_02_unet_training_trajectory.png`
  - `fig_03_unet_compartment_dice_evolution.png`
  - `fig_04_unet_calibration_reliability.png`
  - `fig_05_five_model_benchmark_matrix.png`
  - `fig_06_unet_predictive_entropy_distribution.png`
  - `fig_07_unet_executive_poster_dashboard.png`

### 5.3 U-Net++ Profile (`reports/model_profiles/unetpp/`)
- Total files: 15 (7 active publication figures, 1 active markdown dossier, 3 active binaries/PNG, 1 one-page PNG, and 3 rescued historical assets).
- Canonical dossier: `reports/model_profiles/unetpp/UNetPlusPlus_Master_Model_Dossier.md`
- Active figures in `reports/model_profiles/unetpp/assets/`:
  - `fig_01_unetpp_nested_architecture.png`
  - `fig_02_unetpp_training_trajectory.png`
  - `fig_03_unetpp_compartment_dice_evolution.png`
  - `fig_04_unetpp_calibration_reliability.png`
  - `fig_05_five_model_benchmark_matrix.png`
  - `fig_06_unetpp_predictive_entropy_distribution.png`
  - `fig_07_unetpp_executive_poster_dashboard.png`

---

## 6. Rescued Historical Assets (Stage 3B)

Preserved with byte-for-byte fidelity in dedicated historical directories:

| Historical Asset Path | Size | SHA-256 Provenance Hash |
|---|---|---|
| `reports/model_profiles/unetpp/dossier/historical/UNetPlusPlus_Master_Model_Dossier_20261118.docx` | 4,685,873 bytes | `5ccae9f20e6c1d80d89c6952aa18c4d2adf2c83bb1dd1950cedcaabdd391761a` |
| `reports/model_profiles/unetpp/dossier/historical/UNetPlusPlus_Master_Model_Dossier_20261118.pdf` | 1,509,582 bytes | `f017db83bdd766eac21b1cd11c7d5c5794109093712bf33edd72f7d6790ef5a4` |
| `reports/model_profiles/unetpp/one_page/historical/UNetPlusPlus_Executive_Summary_Printout_20261118.pdf` | 56,742 bytes | `11b8fb3ea02fde2ab7d02e7c8149f5e577082acefadfc86ca61f388b1d0a2da7` |

---

## 7. Archived Data & Submission Materials

- **BraTS 2023 Raw ZIP Archive:** `data/archives/brats2023/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` (14.8 GB, safely relocated in Stage 1).
- **Submissions Folder:** `submissions/` (13 cataloged files, 12 retained assets preserved in Stage 2).
- **Legacy Documents Backup:** `documents/` (22 legacy files preserved as local rollback source).
