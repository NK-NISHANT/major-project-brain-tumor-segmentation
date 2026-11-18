# Repository Inventory (Read-Only Audit Record)

- **Document:** `REPOSITORY_INVENTORY_READONLY.md`
- **Canonical Blueprint Authority:** [MASTER_PROJECT_BLUEPRINT_v2.0.md](file:///D:/MajorProject/MASTER_PROJECT_BLUEPRINT_v2.0.md) (Sections 15.2, 22)
- **Governance Stage:** Stage 4B.1 (Governance Remediation & Verification)
- **Audit Snapshot Timestamp:** `2026-11-19T01:55:00+05:30`
- **Audit Mode:** Read-Only Physical Inventory & Point-in-Time Disk Census

> [!NOTE]
> **Point-in-Time Snapshot Disclosure:** This document records an immutable, read-only audit snapshot of the repository topology, file inventory, and disk allocations as measured on **2026-11-19T01:55:00+05:30**. It represents a verified historical audit baseline, not a dynamically mutating live dashboard.

---

## 1. Executive Summary & Audit Mandate

In strict accordance with Section 22 of `MASTER_PROJECT_BLUEPRINT_v2.0.md`, this inventory provides a complete audit of directory boundaries, storage footprint, Git tracking eligibility, environment dependencies, and curated model deliverables across `D:\MajorProject`.

Zero destructive operations, file moves, or Git staging actions (`git add`) were performed during this audit.

---

## 2. Git Boundary & Remote Tracking Baseline (Audit Snapshot)

| Metric | Measured Baseline Value | Governance Significance |
|---|---|---|
| **Active Branch** | `main` | Primary release branch |
| **Local HEAD Commit** | `136b5b9` (*"Checkpoint: preserve current project progress"*, Nov 15, 2026) | Base commit prior to restructuring stages |
| **Remote Origin URL** | `https://github.com/NK-NISHANT/major-project-brain-tumor-segmentation.git` | Authoritative remote GitHub repository |
| **Remote Tracking State** | Stale tracking ref at `64c5ff5` (Oct 23, 2026). Upstream branch unconfigured. | **No remote fetch, branch update, or push authorized** in Stage 4B/4B.1. Remote sync deferred to Stage 4C. |
| **Staged Changes** | **0 files** (`git diff --cached` is strictly empty) | Satisfies Stage 4B.1 zero-stage constraint |
| **Modified Tracked Files** | 11 files (`.gitignore`, `README.md` [updated], `src/agent/*`, `tests/*`, `tools/demo_app.py`) | Baseline tracked edits |
| **Merge/Rebase In Progress** | **None** (zero merge conflicts, zero operational locks) | Repository is in stable state |

---

## 3. Storage Footprint & Directory Census

| Directory Path | Approximate Disk Size | File Census | Category / Governance Role |
|---|---|---|---|
| `data/archives/` | ~14.8 GB | 1 archive | **Relocated Raw Archives (Stage 1)**: `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip` (14.8 GB) safely archived; ignored in Git. |
| `results/` | ~5.17 GB | 1,921 files | **Research Evidence & Model Checkpoints**: Checkpoints (142 `.pt` files, 5.04 GB) strictly ignored via `*.pt`; research evidence (147 `.csv`, 1,371 `.json`, 175 `.md`, 36 `.png`) preserved as **trackable**. |
| `.venv/` | ~4.3 GB | > 15,000 files | **Python Virtual Environment**: Python 3.11 runtime, PyTorch with CUDA, MONAI; ignored in Git. |
| `data/processed/` | ~1.8 GB | ~16,297 files | **Preprocessed Dataset**: Pre-sliced axial arrays; ignored in Git. |
| `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData/` | ~1.2 GB | ~990 files | **Unpacked Raw Scans**: Multimodal NIfTI volumes (T1n, T1c, T2w, T2f, seg); ignored in Git. |
| `Report_Materials/` | ~140 MB | 4 files | **Source Reference Materials**: Institutional reports, large reference files; ignored in Git. |
| `reports/` | ~60.3 MB | 135 files | **Curated Model Profiles**: Structured dossiers, publication figures, CSV evidence; governed under **Option A**. |
| `documents/` | ~35 MB | 22 files | **Deprecated Migration Source**: Preserved locally as rollback safety net; explicitly ignored in Git. |
| `archive/` | ~15 MB | 20 files | **Historical Roadmaps & Legacy Packs**: Historical project roadmaps and planning logs. |
| `docs/` | ~12 MB | ~25 files | **Architecture Documentation**: Blueprints, reference guides, and generation logs. |
| `submissions/` | ~8 MB | 13 files | **Institutional Submissions (Stage 2)**: Preserved submission deliverables; ignored in Git. |
| `src/` | ~1.5 MB | ~35 files | **Core Application Code**: Data loaders, 5 model definitions, uncertainty metrics, agent logic. Tracked in Git. |
| `tools/` | ~0.8 MB | ~30 files | **Utility & Generator Scripts**: Dossier generators, calibration orchestrators, demo application. Tracked in Git. |
| `tests/` | ~0.4 MB | ~12 files | **Automated Regression Suite**: 325 passing tests. Tracked in Git. |

---

## 4. Curated Model Deliverables Census (`reports/model_profiles/`)

The curated deliverable tree contains exactly **135 files** across three model profile hierarchies:
1. `reports/model_profiles/resunet_2d/`: **110 files** (56 visual assets, 34 audit artifacts, 8 deliverables, 8 source scripts, 4 root docs).
2. `reports/model_profiles/unet_2d/`: **10 files** (7 publication figures, 1 master dossier, 2 local-only binaries).
3. `reports/model_profiles/unetpp/`: **15 files** (7 publication figures, 1 master dossier, 3 active binaries, 1 one-page PNG, 3 rescued historical assets).

### 4.1 Strict Option A Eligibility vs. Staging Recommendations
A clear distinction is maintained between **eligibility** (file format permitted by policy) and **staging recommendation** (curated subset proposed for inclusion in Git):

- **Option A Approved Eligible Categories:** Markdown (`.md`), Python (`.py`), PNG images (`.png`), SVG vectors (`.svg`), and CSV tables (`.csv`).
- **Option A Approved Local-Only Binary Categories:** Word (`.docx`), PDF (`.pdf`), and PowerPoint (`.pptx`).
- **Review Required / Extension Exceptions:** Non-standard formats (e.g. `.ps1`, `.json`, `.yaml`, `.txt`) must be explicitly justified and approved by the owner rather than silently included.

Across all 135 files in `reports/model_profiles/`:
- **Eligible under Option A:** **120 files**
- **Local-Only Binaries (Ignored):** **14 files** (totaling ~28.3 MB across ResUNet, U-Net, and U-Net++)
- **Review Required Exceptions:** **1 file** (`reports/model_profiles/resunet_2d/source/generate_explainer_deck.ps1`, 31.5 KB) — held pending explicit owner authorization.

---

## 5. Exhaustive File-by-File Audit: `reports/model_profiles/resunet_2d/` (110 Files)

| Relative Path | Ext | Size (Bytes) | Role / Taxonomy | Option A Eligibility | Staging Recommendation |
|---|---|---|---|---|---|
| `assets/agent_flow_diagram.png` | .png | 530,322 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/architecture_diagram.png` | .png | 498,177 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/calibration_comparison.png` | .png | 343,663 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/class_dice_curves.png` | .png | 372,609 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/entropy_distribution.png` | .png | 277,986 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_01_what_is_resunet.png` | .png | 218,008 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_02_why_brain_tumors_are_hard.png` | .png | 226,067 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_03_four_mri_modalities_explained.png` | .png | 156,209 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_04_three_tumor_regions_explained.png` | .png | 233,258 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_05_encoder_decoder_concept.png` | .png | 222,513 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_06_residual_connections_explained.png` | .png | 205,218 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_07_what_dice_score_means.png` | .png | 218,753 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_08_what_hd95_means.png` | .png | 228,826 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_09_why_ai_gets_overconfident.png` | .png | 182,251 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_10_temperature_scaling_visualized.png` | .png | 186,950 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_11_entropy_as_traffic_light.png` | .png | 199,686 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_12_axial_context_sandwich.png` | .png | 201,225 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_13_three_action_triage_decision.png` | .png | 183,530 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_14_why_fallback_saves_empty_predictions.png` | .png | 191,297 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_15_clinical_safety_story.png` | .png | 165,830 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/non_technical/simple_16_performance_snapshot_dashboard.png` | .png | 158,298 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/qualitative_examples/qual_01_val_eval_excellent_case.png` | .png | 693,987 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/qualitative_examples/qual_02_val_eval_median_case.png` | .png | 544,270 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/qualitative_examples/qual_03_val_eval_challenging_case.png` | .png | 549,028 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_accept.png` | .png | 35,201 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_brain_mri.png` | .png | 43,580 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_calibration.png` | .png | 35,752 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_fallback.png` | .png | 29,119 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_flag.png` | .png | 22,242 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/reusable_icons/icon_resunet_badge.png` | .png | 31,228 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/01_resunet_full_architecture.png` | .png | 665,983 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/02_residual_block_explained.png` | .png | 357,570 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/03_input_to_output_pipeline.png` | .png | 283,154 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/04_four_channel_mri_explainer.png` | .png | 357,660 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/05_training_validation_loss.png` | .png | 324,622 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/06_classwise_dice_progression.png` | .png | 372,786 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/07_foreground_macro_dice_progression.png` | .png | 307,179 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/08_checkpoint_selection_explainer.png` | .png | 355,063 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/09_segmentation_performance_summary.png` | .png | 266,235 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/10_reliability_diagram_val_cal.png` | .png | 318,076 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/11_reliability_diagram_val_eval.png` | .png | 322,583 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/12_calibration_metrics_comparison.png` | .png | 264,974 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/13_temperature_scaling_explainer.png` | .png | 288,482 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/14_calibration_sampling_protocol.png` | .png | 317,802 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/15_predictive_entropy_distribution.png` | .png | 290,269 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/16_entropy_threshold_explainer.png` | .png | 328,417 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/17_entropy_population_breakdown.png` | .png | 308,631 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/18_full_c3_decision_flow.png` | .png | 625,168 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/19_primary_secondary_model_interaction.png` | .png | 326,816 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/20_agent_action_distribution.png` | .png | 272,107 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/21_empty_prediction_fallback_explainer.png` | .png | 290,668 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/22_context_window_explainer.png` | .png | 339,267 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/23_model_checkpoint_provenance.png` | .png | 297,948 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/24_evaluation_cohort_summary.png` | .png | 290,104 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/technical/25_model_limitations_and_evidence_boundaries.png` | .png | 295,256 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `assets/training_curves.png` | .png | 378,546 | Visualization Figure | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit/CHANGELOG.md` | .md | 5,183 | Audit Artifact (audit) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit/metric_consistency_audit.md` | .md | 6,466 | Audit Artifact (audit) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit/source_audit.md` | .md | 15,419 | Audit Artifact (audit) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit/visual_qa_report.md` | .md | 4,743 | Audit Artifact (audit) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/architecture_training_reconciliation.md` | .md | 11,206 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/c3_action_accounting_reconciliation.md` | .md | 11,706 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/calibration_evidence_reconciliation.md` | .md | 9,191 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/checkpoint_identity_reconciliation.md` | .md | 9,451 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/claim_and_disclosure_audit.md` | .md | 7,231 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/confidence_interval_reconciliation.md` | .md | 7,355 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/MASTER_SOURCE_OF_TRUTH.md` | .md | 11,902 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2/PHASE_1_FINAL_VERDICT.md` | .md | 7,298 | Audit Artifact (audit_v1_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/c3_risk_coverage_interpretation.md` | .md | 9,361 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/CORRECTED_MASTER_SOURCE_OF_TRUTH.md` | .md | 15,728 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/epoch_and_input_dimension_reconciliation.md` | .md | 6,612 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/PHASE_1_1_DISCREPANCY_REGISTER.md` | .md | 6,902 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/resunet_parameter_breakdown_reconciliation.md` | .md | 12,116 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/secondary_model_calibration_reconciliation.md` | .md | 7,607 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_1/validation_exposure_and_ci_disclosure.md` | .md | 9,238 | Audit Artifact (audit_v1_2_1) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_2/FINAL_C3_CLASSIFICATION_CORRECTION.md` | .md | 9,044 | Audit Artifact (audit_v1_2_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_2/FINAL_SOURCE_OF_TRUTH_v1_2_2.md` | .md | 17,600 | Audit Artifact (audit_v1_2_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_2/PHASE_1_1_2_FINAL_VERDICT.md` | .md | 6,707 | Audit Artifact (audit_v1_2_2) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_3/bootstrap_implementation_verification.md` | .md | 8,866 | Audit Artifact (audit_v1_2_3) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_3/FINAL_SOURCE_OF_TRUTH_v1_2_3.md` | .md | 18,277 | Audit Artifact (audit_v1_2_3) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_3/PHASE_1_1_3_FINAL_VERDICT.md` | .md | 9,436 | Audit Artifact (audit_v1_2_3) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_3/raw_trace_partition_reconciliation.md` | .md | 14,434 | Audit Artifact (audit_v1_2_3) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_3/slice_level_discrepancy_register.csv` | .csv | 429,776 | Audit Artifact (audit_v1_2_3) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/BOOTSTRAP_ESTIMAND_DISCLOSURE.md` | .md | 5,872 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/complete_flagged_slice_classification_register.csv` | .csv | 2,862,620 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/FINAL_REGISTER_SCOPE_AUDIT.md` | .md | 6,505 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/FINAL_SLICE_CLASSIFICATION_CLARIFICATION.md` | .md | 9,044 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/FINAL_SOURCE_OF_TRUTH_v1_2_4.md` | .md | 20,067 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/historical_discrepancy_extract_1473.csv` | .csv | 393,493 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `audit_v1_2_4/PHASE_1_1_4_FINAL_VERDICT.md` | .md | 7,077 | Audit Artifact (audit_v1_2_4) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `dossier/ResUNet_2D_Model_Dossier_v1.1.docx` | .docx | 3,951,264 | Deliverable Document (dossier) | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `dossier/ResUNet_2D_Model_Dossier_v1.1.md` | .md | 24,372 | Deliverable Document (dossier) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `dossier/ResUNet_2D_Model_Dossier_v1.1.pdf` | .pdf | 1,053,470 | Deliverable Document (dossier) | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `one_page/ResUNet_2D_One_Page_Overview.pdf` | .pdf | 72,709 | Deliverable Document (one_page) | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `one_page/ResUNet_2D_One_Page_Overview.png` | .png | 1,375,646 | Deliverable Document (one_page) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `one_page/ResUNet_2D_One_Page_Overview.svg` | .svg | 536,048 | Deliverable Document (one_page) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `presentation/Presenter_Notes.md` | .md | 19,037 | Deliverable Document (presentation) | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `presentation/ResUNet_2D_Explainer_Deck.pptx` | .pptx | 2,888,108 | Deliverable Document (presentation) | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `README.md` | .md | 10,322 | Root Profile Script/Doc | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `ResUNet_2D_Model_Dossier_v1.0.docx` | .docx | 2,240,289 | Root Profile Script/Doc | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `ResUNet_2D_Model_Dossier_v1.0.md` | .md | 27,692 | Root Profile Script/Doc | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `ResUNet_2D_Model_Dossier_v1.0.pdf` | .pdf | 988,665 | Root Profile Script/Doc | Local-Only Binary (Ignored) | **Exclude (Keep Local Only)** |
| `source/build_resunet_dossier_v1_1.py` | .py | 51,924 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/generate_explainer_deck.ps1` | .ps1 | 31,466 | Compilation Source Script | REVIEW_REQUIRED (Exception: .PS1 Pending Owner Authorization) | **Hold for Owner Review** |
| `source/generate_non_technical_assets.py` | .py | 52,517 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/generate_one_page_overview.py` | .py | 21,645 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/generate_qualitative_assets.py` | .py | 15,775 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/generate_reusable_icons.py` | .py | 3,693 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/generate_technical_assets.py` | .py | 66,033 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |
| `source/README.md` | .md | 4,069 | Compilation Source Script | Option A Eligible | **Staging Recommended (Curated Baseline)** |

---

## 6. Results Directory Tracking Boundary (`results/`)

Following the Stage 4B.1 governance review, the blanket `/results/` ignore rule was removed to ensure legitimate research evidence remains versionable. The refined policy enforces:

1. **Ignored Local-Only Checkpoints & Weights:**
   - Global weight patterns: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.safetensors` (excludes 5.04 GB of neural net weights).
   - Local checkpoint directories: `results/checkpoints/`, `results/models/`.
   - Voxel dumps and cache snapshots: `results/**/case_voxels/`, `results/**/recovery_snapshot_*/`.
2. **Trackable Research Evidence (Preserved for Stage 4C Curation):**
   - Run manifests: `results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json`.
   - Calibration metrics & parameters: `evaluation_metrics.csv`, `calibration_parameters.json`, `entropy_summary.csv`.
   - HD95 evaluations: `results/hd95_evaluation/hd95_summary.json`.
   - Quarantined split manifests: `results/dataset_splits/test_manifest.json` and `final_split_manifest.json`.
   - Training histories and configs: `results/full_training_15ep/*/config.json`, `summary.json`.

---

## 7. Runtime & Virtual Environment Specifications

- **Operating System:** Windows 11 Enterprise (build 10.0.26300)
- **Python Runtime:** Python 3.11.9 (64-bit)
- **PyTorch Version:** PyTorch 2.11.0+cu128
- **CUDA Device:** NVIDIA GeForce RTX 4070 Laptop GPU (8,188 MB VRAM)
- **Core Environment Packages:**
  - `monai==1.5.0` (Medical Open Network for AI)
  - `nibabel==5.3.2` (Neuroimaging data I/O)
  - `torch==2.11.0+cu128` (Deep learning framework)
  - `scikit-learn==1.6.1` (Calibration, metrics, ROC/PR curves)
  - `scipy==1.15.2` (Scientific computing & optimization)
  - `streamlit==1.42.2` (Interactive web application framework)
  - `pytest==8.3.4` (Automated testing framework)

---

## 8. Preserved Governance Invariants

1. **Quarantined Test Cohort:** Manifests `results/dataset_splits/test_manifest.json` and `final_split_manifest.json` are cryptographically verified and sealed. Zero test-set inferences executed.
2. **Root Milestone Deliverables:** All eight `G1_*` milestone reports stay at the repository root.
3. **Rollback Source Integrity:** `documents/` preserved intact with 22 files on local disk.
4. **Historical Provenance:** Rescued U-Net++ assets preserved in dedicated historical directories with exact SHA-256 matches.
5. **Decision Register Authority:** [STAGE_C_DECISION_REGISTER.csv](file:///D:/MajorProject/STAGE_C_DECISION_REGISTER.csv) remains the authoritative structured decision register.
