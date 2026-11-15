# G1-B1 — Research Edition V4.1.1 Generation & Verification Log

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Generated Document:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx`
**Artwork Directory:** `tmp/report_artwork_research_v4_1_1/`
**PDF Export:** `tmp/v4_1_1_word.pdf`
**Timestamp:** 2026-11-12

---

## 1. Summary of Changes Applied in Edition V4.1.1

1. **Historical Edition Preservation:**
   - Byte-for-byte immutability preserved across all historical editions:
     - Original Historical Minor Project Report: `Final_Report_Brain_Tumor_Segmentation.docx`
     - Research Edition V1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
     - Research Edition V2: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx`
     - Research Edition V3: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx`
     - Research Edition V4: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx`
     - Research Edition V4.1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx`
   - Added `PROTECTED_V4_1_REPORT` invariant in `tools/build_final_report.py` preventing accidental overwrite of V4.1.

2. **Figure 7.1 Visual Redesign & Readability Optimization:**
   - Rebuilt Figure 7.1 from the ground up on a balanced 100-unit horizontal canvas utilizing a clean 4-stage pipeline layout:
     - **Stage 1 (Input & Integrity):** Displays canonical 4-channel MRI input (`Ch 0: T1n`, `Ch 1: T1c`, `Ch 2: T2f`, `Ch 3: T2w`), input shape `[1, 4, H, W]`, batch size 1, and positive spatial dimensions. Includes a dedicated dashed `INPUT REJECTION` exception box illustrating pre-inference halts on malformed tensors (raising `TypeError` or `ValueError`), clarifying that validation halts are execution exceptions rather than agent triage decisions. No unverified claims of semantic modality detection.
     - **Stage 2 (Primary & Calibration):** Primary ResUNet-2D inference (Epoch 13), post-hoc temperature scaling ($T^* = 2.8808$), calibrated normalized Shannon entropy estimation $H(p)$, and axial context guard (radius $k=2$).
     - **Stage 3 (Triage Engine):** Deterministic routing logic evaluating foreground presence, calibrated entropy threshold ($H < 0.35$), axial context support, and dual-model fallback mechanism. Explicitly notes that candidate switching on entropy alone is disabled in canonical B2.
     - **Stage 4 (Triage Actions):**
       - `ACTION 1: ACCEPT` ($H < 0.35$ & non-empty, 5,437 slices, 83.02%).
       - `ACTION 2: FALLBACK SWITCH` (empty primary + confirmed context support, query UNetResNet34 $T^*=2.63$, retain non-empty candidate, 54 slices, 0.82%).
       - `ACTION 3: FLAG / REFER` ($H \ge 0.35$ or unresolvable empty, simulated referral, 1,058 slices, 16.16%).
   - Expanded horizontal connector spacing between stages from 4 units to 8 units. Automated bounding-box collision detection (`get_window_extent()`) verified **0 overlapping text pairs** across all 31 rendered text elements.

3. **Section 1.5 Contribution Summary Correction:**
   - Removed universal claim: *"proving that post-hoc predictive entropy alone cannot detect missing contrast data"*.
   - Replaced with the bounded empirical finding:
     *"Demonstrates a specific limitation of the tested entropy-based policy under synthetic post-normalization T1c zero-clamping, where severe enhancing tumor and tumor core degradation (ET Dice collapses to 0.1684) was not accompanied by a statistically significant increase in aggregate referral (p_adj = 1.0000), motivating explicit upstream modality-presence and sequence-identity validation."*
   - Harmonized wording across Abstract, Section 1.5, and Chapter 7.

---

## 2. File Verification & Exact SHA-256 Digests

| Artifact | File Path | SHA-256 Digest | Size (bytes) | Status |
|---|---|---|---:|:---:|
| **Original Minor Report** | `Final_Report_Brain_Tumor_Segmentation.docx` | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | 4,301,505 | **PRESERVED** |
| **Research Edition V1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | 3,112,658 | **PRESERVED** |
| **Research Edition V2** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | 1,905,004 | **PRESERVED** |
| **Research Edition V3** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx` | `2d85fc7cd92ae80371ec31584b36535a0d4772a7be6018e5a4d00b6f15ebb90b` | 1,928,076 | **PRESERVED** |
| **Research Edition V4** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx` | `bb8f02746bbc55dd38ac86bcb1b36b3c878b8284a6919a3496d6530855d709bb` | 1,968,938 | **PRESERVED** |
| **Research Edition V4.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx` | `3d9e09e4417adb0df826cfce8dcf6c1161e5131d7927b197849cece8237a65ee` | 1,992,079 | **PRESERVED** |
| **Research Edition V4.1.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx` | `58f04ddc7a6fa3b85905c7e4f0f35be8e078acff4b0a4213abb87ca85cd7bab5` | 1,996,480 | **CREATED** |
| **Report Builder** | `tools/build_final_report.py` | `7f1e27373d39cb7332971a52043e685877fe5637668d8014777ca391de70a722` | 159,657 | **UPDATED** |
| **Word COM Export** | `tmp/v4_1_1_word.pdf` | `869091171e640a25c60afe04722d3f7c6793a77ff5485fcdfda05c27158a3a43` | 1,054,946 | **VERIFIED** |

---

## 3. Structural Package Validation

- **OpenXML Package Integrity:** Valid `.docx` zip archive containing `word/document.xml`, `[Content_Types].xml`, and `word/_rels/document.xml.rels`.
- **Embedded Media:** Exactly 12 images in `word/media/`:
  - `image1.png` (JIIT Logo, title page)
  - `image2.png` (Figure 4.1: Four MRI Modalities)
  - `image3.png` (Figure 4.2: Five-Model Architecture Portfolio)
  - `image4.png` (Figure 4.3: Sample Segmentation Visualizations)
  - `image5.png` (Figure 6.1: Pilot Dice & IoU Bar Chart)
  - `image6.png` (Figure 6.2: Class-Wise Dice Comparison)
  - `image7.png` (Figure 6.3: Training/Validation Loss Curves)
  - `image8.png` (Figure 6.4: Validation Dice Progression)
  - `image9.png` (Figure 6.5A: Entropy Threshold Distribution)
  - `image10.png` (Figure 6.5B: Quality across Entropy Quintiles)
  - `image11.png` (Figure 6.6: Foreground Miss Rate Progression)
  - `image12.png` (Figure 7.1: Redesigned Agent Architecture & Triage Pipeline)
- **Table Formatting Invariants:**
  - Total tables: Exactly 21 tables (TOC, LOT, LOF, 2 metadata tables, 16 numbered content tables).
  - Total rows evaluated: 224 rows.
  - `w:cantSplit` enforcement: 224 / 224 rows (100.0%).
  - `w:tblHeader` enforcement: 20 / 20 content tables (100.0%).
- **Placeholder Audit:** 0 occurrences of `TODO`, `TBD`, `FIXME`, `XXX`, `???`, or `[placeholder]`.

---

## 4. Heading Alignment & Zero-Drift Pagination

Word COM automation (`Word.Application`) confirmed an exact total of **39 pages**. All 44 document headings matched V4 and V4.1 page-for-page with **zero drift**:

| Heading Index | Heading Title | V4 Page | V4.1.1 Page | Drift |
|:---:|---|:---:|:---:|:---:|
| H01 | CHAPTER 1: INTRODUCTION | 1 | 1 | 0 |
| H02 | 1.1 Clinical and Technical Context | 1 | 1 | 0 |
| H03 | 1.2 Limitations of Segmentation Dice as a Sole Reliability Measure | 1 | 1 | 0 |
| H04 | 1.3 Predictive Uncertainty and Reliability-Aware Decision-Making | 2 | 2 | 0 |
| H05 | 1.4 Research Motivation, Objectives, and Scope | 2 | 2 | 0 |
| H06 | 1.5 Summary of Project Contributions | 3 | 3 | 0 |
| H07 | 1.6 Report Organization | 4 | 4 | 0 |
| H08 | CHAPTER 2: BACKGROUND STUDY AND RELATED WORK | 5 | 5 | 0 |
| H09 | 2.1 Multi-modal MRI and Glioma Subregions | 5 | 5 | 0 |
| H10 | 2.2 Deep Learning Architectures for Medical Image Segmentation | 5 | 5 | 0 |
| H11 | 2.3 Evaluation, Overlap, and Distance Metrics | 6 | 6 | 0 |
| H12 | 2.4 Predictive Uncertainty Quantification in Medical Imaging | 6 | 6 | 0 |
| H13 | 2.5 Probability Calibration in Neural Networks | 7 | 7 | 0 |
| H14 | 2.6 Selective Prediction, Triage, and Reliability-Aware AI | 7 | 7 | 0 |
| H15 | CHAPTER 3: DATASET CENSUS AND PARTITIONING METHODOLOGY | 8 | 8 | 0 |
| H16 | 3.1 BraTS 2023 GLI Dataset Overview | 8 | 8 | 0 |
| H17 | 3.2 Subject-Level Dataset Census and Split Accounting | 8 | 8 | 0 |
| H18 | 3.3 Historical 197-Case Pilot Split vs Mature Cohort | 9 | 9 | 0 |
| H19 | 3.4 Data Preprocessing and Tensor Normalization | 9 | 9 | 0 |
| H20 | CHAPTER 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE | 11 | 11 | 0 |
| H21 | 4.1 Five-Model Architecture Portfolio | 11 | 11 | 0 |
| H22 | 4.2 Structural Details of Stage A Architectures | 12 | 12 | 0 |
| H23 | 4.3 Structural Details of Stage B Architectures | 13 | 13 | 0 |
| H24 | 4.4 Training Dynamics and Loss Optimization Framework | 13 | 13 | 0 |
| H25 | 4.5 Software Architecture and Repository Design | 14 | 14 | 0 |
| H26 | CHAPTER 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS | 15 | 15 | 0 |
| H27 | 5.1 Two-Stage Experimental Progression | 15 | 15 | 0 |
| H28 | 5.2 Stage A: Historical Five-Epoch Pilot Benchmark | 15 | 15 | 0 |
| H29 | 5.3 Stage B: Mature 15-Epoch Research Baselines | 17 | 17 | 0 |
| H30 | 5.4 Checkpoint-Selection Bias and Methodological Boundaries | 18 | 18 | 0 |
| H31 | CHAPTER 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE | 19 | 19 | 0 |
| H32 | 6.1 Predictive Uncertainty and Segmentation Quality Correlation | 19 | 19 | 0 |
| H33 | 6.2 Separation of Uncertainty Binning and Slice Distribution | 20 | 20 | 0 |
| H34 | 6.3 Post-Hoc Probability Calibration and Temperature Scaling | 21 | 21 | 0 |
| H35 | 6.4 Implemented Deterministic Reliability-Aware Agent | 22 | 22 | 0 |
| H36 | CHAPTER 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS | 23 | 23 | 0 |
| H37 | 7.1 Missing-Modality Robustness Stress Study | 23 | 23 | 0 |
| H38 | 7.2 Methodological Boundaries and Comprehensive Limitations | 24 | 24 | 0 |
| H39 | 7.3 Exploratory 3D / Foundation Model Pilot Studies | 25 | 25 | 0 |
| H40 | 7.4 Future Research Directions and Conclusion | 25 | 25 | 0 |
| H41 | REFERENCES | 27 | 27 | 0 |
| H42 | APPENDIX A: FULL DATASET SPLIT MANIFEST AND CENSUS AUDIT | 28 | 28 | 0 |
| H43 | APPENDIX B: REPRODUCIBILITY RECORD AND HYPERPARAMETER SPECIFICATIONS | 28 | 28 | 0 |
| H44 | APPENDIX C: RESEARCH GOVERNANCE, VERIFICATION CHECKLIST, AND INTEGRITY DECLARATION | 29 | 29 | 0 |

- **Table of Contents (TOC):** Perfectly matches actual body page numbers.
- **List of Tables (LOT):** All 16 tables match actual body page numbers.
- **List of Figures (LOF):** All 11 figures match actual body page numbers.
- **PDF Text Searchability:** 100% searchable via PyMuPDF (`fitz`), with all key terminology, metrics, statistical findings, and figure captions located on expected pages.

---

## 5. Environment & System Provenance

- **Operating System:** Windows 11
- **Working Directory:** `D:\MajorProject`
- **Git Branch:** `main`
- **Git HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904`
- **Test Cohort Status:** Quarantined test cohort (`data/quarantine_test/`) and test manifest (`test_manifest.json`) remain strictly untouched and non-existent on the open development path.
- **Office / PDF Automation Engine:** Microsoft Word COM (`Word.Application`) is installed and serves as the authoritative layout, pagination, and PDF generation engine. LibreOffice (`soffice`) is not installed on the host system.
