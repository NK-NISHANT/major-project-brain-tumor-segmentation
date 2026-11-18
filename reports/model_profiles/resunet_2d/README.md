# RESUNET-2D: COMPLETE MODEL DOSSIER & VISUAL ASSET PACKAGE
## Primary Champion Model & Stage C Reliability Agent Decision Backbone
**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation  
**Model:** ResUNet-2D (`ResUNet2D`)  
**Role:** Primary Champion / Stage C Decision Backbone  
**Profile:** `STAGE_C_15EP` (Epoch 13 Peak)  
**Cryptographic SHA-256:** `1d85be47225c5684f18ba6b89ae6563ee9f39aa13c9e99a80373ab10a300d81a`  
**Package Version:** v1.1 (Supersedes v1.0)  
**Workspace:** `D:\MajorProject`

---

## 1. PACKAGE DIRECTORY STRUCTURE & DELIVERABLE INDEX

This folder contains the complete, definitive documentation, scientific visualization library, and presentation assets for **ResUNet-2D**:

```
reports/model_profiles/resunet_2d/
├── README.md                                  <- You are here: Master index & user guide
├── dossier/                                   <- Master comprehensive research dossiers (v1.1)
│   ├── ResUNet_2D_Model_Dossier_v1.1.pdf      <- High-resolution PDF (10 pages, Word COM converted)
│   ├── ResUNet_2D_Model_Dossier_v1.1.docx     <- Word Master document (editable, styled, 0.75-in margins)
│   └── ResUNet_2D_Model_Dossier_v1.1.md       <- Full 20-section raw Markdown dossier
├── one_page/                                  <- Definitive A3 Landscape One-Page Overview
│   ├── ResUNet_2D_One_Page_Overview.png       <- 300 DPI high-resolution overview image
│   ├── ResUNet_2D_One_Page_Overview.pdf       <- Vector-quality printable PDF
│   └── ResUNet_2D_One_Page_Overview.svg       <- Scalable vector graphics export
├── presentation/                              <- Executive explainer presentation & notes
│   ├── ResUNet_2D_Explainer_Deck.pptx         <- 8-Slide 16:9 widescreen presentation deck
│   └── Presenter_Notes.md                     <- Spoken narration script & viva defense Q&A guide
├── assets/
│   ├── technical/                             <- 25 Scientific Figures (300 DPI)
│   │   ├── 01_resunet_full_architecture.png   to  25_model_limitations_and_evidence_boundaries.png
│   ├── non_technical/                         <- 16 Intuitive Explainer Cards (16:9)
│   │   ├── simple_01_what_is_resunet.png      to  simple_16_performance_snapshot_dashboard.png
│   ├── qualitative_examples/                  <- 3 Multi-Panel Validation Patient Visualizations
│   │   ├── qual_01_val_eval_excellent_case.png   (BraTS-GLI-01069-000, Slice 90, Dice > 0.88)
│   │   ├── qual_02_val_eval_median_case.png      (BraTS-GLI-00008-000, Slice 75, Representative)
│   │   └── qual_03_val_eval_challenging_case.png (BraTS-GLI-00753-001, Slice 85, Infiltrative)
│   └── reusable_icons/                        <- 6 Clean Reusable Badges & Icons (Transparent PNG)
│       ├── icon_accept.png, icon_fallback.png, icon_flag.png, icon_calibration.png, etc.
├── source/                                    <- Complete generation pipeline scripts
│   ├── build_resunet_dossier_v1_1.py          <- Generates MD, DOCX, and PDF dossiers
│   ├── generate_technical_assets.py           <- Generates the 25 technical figures
│   ├── generate_non_technical_assets.py       <- Generates the 16 non-technical cards
│   ├── generate_qualitative_assets.py         <- Generates the 3 qualitative panels
│   ├── generate_one_page_overview.py          <- Generates the A3 overview (PNG/PDF/SVG)
│   ├── generate_explainer_deck.ps1            <- Builds the 8-slide PPTX deck via COM
│   ├── generate_reusable_icons.py             <- Generates the badge/icon assets
│   └── README.md                              <- Script documentation & reproduction guide
└── audit/                                     <- Scientific audits, traceability & changelogs
    ├── source_audit.md                        <- Complete traceability matrix to raw files
    ├── metric_consistency_audit.md            <- Cross-deliverable numerical verification report
    ├── visual_qa_report.md                    <- Design QA, unclipped layout & contrast audit
    └── CHANGELOG.md                           <- Detailed record of v1.0 -> v1.1 fixes
```

---

## 2. MASTER PERFORMANCE & RELIABILITY SCORECARD

| Clinical & Technical Metric | Protocol & Scope | Reported Value | Source Artifact |
|:---|:---|:---:|:---|
| **Trainable Parameters** | 32-bit floating point | **8,111,892 (8.11 M)**; 32.4 MB | `weights/stage_c_15ep/resunet_best.pt` |
| **Volumetric Macro Dice** | 105 Evaluation Patients (`Val_eval`) | **0.8641** | `results/val_eval_predicted_metrics.json` |
| **Whole Tumor (WT) Dice** | Infiltrative territory (`Val_eval`) | **0.9039** (95% CI: [0.8878, 0.9189]) | `results/val_eval_predicted_metrics.json` |
| **Tumor Core (TC) Dice** | Surgical debulking core (`Val_eval`) | **0.8597** (95% CI: [0.8351, 0.8821]) | `results/val_eval_predicted_metrics.json` |
| **Enhancing Tumor (ET) Dice** | Active neoplastic rim (`Val_eval`) | **0.8286** (95% CI: [0.7963, 0.8580]) | `results/val_eval_predicted_metrics.json` |
| **Checkpoint Selection Dice** | **Foreground Macro Mean** (Ep 13) | **0.8670** (NCR: 0.8210, ED: 0.8765, ET: 0.9037) | `results/training_history_resunet_scaled_15ep.json` |
| **4-Class Arithmetic Mean** | Including Background (Ep 13) | **0.899968** (BG: 0.9987) | `results/training_history_resunet_scaled_15ep.json` |
| **Slice Unfiltered Dice** | 6,549 Ground-Truth Tumor Slices | **0.7150** Macro Dice | `results/c3_selective_prediction_v1_1_3/` |
| **HD95 Surface Boundary** | Enhancing Tumor Median Margin | **2.00 mm** (Mean: 9.92 mm) | `results/hd95_evaluation_report.json` |
| **HD95 Surface Boundary** | Tumor Core Median Margin | **3.16 mm** (Mean: 8.98 mm) | `results/hd95_evaluation_report.json` |
| **HD95 Surface Boundary** | Whole Tumor Median Margin | **4.12 mm** (Mean: 15.29 mm) | `results/hd95_evaluation_report.json` |
| **Optimal Temperature ($T^*$)**| Learned on `Val_cal` (50 cases) | **2.880459** (L-BFGS, 36 iters) | `results/c2_calibration_manifest.json` |
| **Expected Calibration Error** | ECE on 105 Evaluation Patients | **8.59% -> 3.41%** (-60.3% reduction) | `results/c2_calibration_manifest.json` |
| **Negative Log-Likelihood** | NLL on 105 Evaluation Patients | **1.0277 -> 0.4415** (-57.0% reduction) | `results/c2_calibration_manifest.json` |
| **Mean Predictive Entropy ($H$)**| 6,874 Non-Empty Slices | **0.2686 +- 0.0903** (Median: 0.2474) | `results/c3_selective_prediction_v1_1_3/` |
| **Operating Threshold ($\tau$)** | Frozen Stage C Uncertainty Gate | **$\tau = 0.35$** | `MASTER_PROJECT_BLUEPRINT_v2.0.md` |
| **Autonomous Retention** | Slices Retained by Agent ($H < 0.35$) | **83.84%** (5,491 / 6,549 slices); Mean Dice: **0.7803** | `results/c3_selective_prediction_v1_1_3/` |
| **Specialist Referral** | Slices Flagged for Human Review | **16.16%** (1,058 / 6,549 slices); Mean Dice: **0.3891** | `results/c3_selective_prediction_v1_1_3/` |
| **Severe Error Reduction** | Cut in Severe Errors (Dice < 0.50) | **-34.1%** (Drop from 20.51% [1,343] to 13.51% [742]) | `results/c3_selective_prediction_v1_1_3/` |
| **Axial Fallback Rescues** | True Tumors Rescued via UNetResNet34 | **54 Slices** (Rescued from false-negative omission) | `results/c3_selective_prediction_v1_1_3/` |

---

## 3. HOW TO USE THIS DELIVERABLE PACKAGE

### For Technical Viva Examiners & Faculty Evaluators
1. Start with the **One-Page Overview** (`one_page/ResUNet_2D_One_Page_Overview.pdf`): Get an instant, high-density visualization of the entire architecture, benchmark metrics, calibration improvements, and clinical triage flow.
2. Review the **Explainer Presentation Deck** (`presentation/ResUNet_2D_Explainer_Deck.pptx`) alongside the **Presenter Notes** (`presentation/Presenter_Notes.md`): Review speaking scripts, slide timing, and anticipated viva defense questions.
3. Inspect the **Reconstructed Dossier v1.1** (`dossier/ResUNet_2D_Model_Dossier_v1.1.pdf`): Dive into the 10-page master technical report for rigorous mathematical formulations, bootstrap statistics, and failure mode analyses.

### For Clinical Collaborators & Non-Specialist Teammates
1. Browse the **Non-Technical Explainer Cards** (`assets/non_technical/`): 16 easy-to-digest cards using traffic lights, sandwiches, and Venn diagrams to understand why brain tumors are difficult to segment, why deep learning gets overconfident, and how axial context prevents mistakes.
2. Examine the **Qualitative Validation Panels** (`assets/qualitative_examples/`): See real MRI channels, predicted tumor masks, error maps, and uncertainty heatmaps across excellent, median, and challenging patients.

### For Machine Learning Researchers & Reproducibility Auditors
1. Consult `audit/source_audit.md` to trace every single reported number directly back to its source JSON, checkpoint, or log file.
2. Read `audit/metric_consistency_audit.md` for our cross-format numerical verification matrix.
3. Review `audit/CHANGELOG.md` to verify all corrections made between v1.0 and v1.1.
4. Execute `source/README.md` instructions to reproduce the entire asset library from scratch.

---

## 4. STRICT RESEARCH HYGIENE & ETHICAL GOVERNANCE

1. **Quarantine Test Cohort (155 Cases):** The test cases in `data/quarantine_test/` and `results/dataset_splits/test_manifest.json` remain **completely sealed, untouched, and unread**. No training, calibration, or hyperparameter selection was ever conducted on the test split.
2. **Disjoint Calibration Split:** Zero patient subject overlap exists between the temperature calibration cohort (`Val_cal`, 50 cases) and the held-out evaluation cohort (`Val_eval`, 105 cases).
3. **Clustered Bootstrap Rigor:** All reported 95% confidence intervals account for within-patient slice correlations by resampling whole patient subject clusters (96 unique subjects across 105 cases).
4. **Academic Regulatory Disclaimer:** This software is an academic engineering research prototype. It is **NOT** cleared or approved by the US FDA, European CE, or CDSCO. It must **NEVER** be used for clinical diagnosis, patient management, surgical resection, or treatment planning.
