# G1-B1 — Research Edition V4.1.2 Generation & Verification Log

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Generated Document:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx`
**Artwork Directory:** `tmp/report_artwork_research_v4_1_2/`
**PDF Export:** `tmp/v4_1_2_word.pdf`
**Timestamp:** 2026-11-12

---

## 1. Summary of Changes Applied in Edition V4.1.2

1. **Historical Edition Preservation:**
   - Byte-for-byte immutability preserved across all historical editions:
     - Original Historical Minor Project Report: `Final_Report_Brain_Tumor_Segmentation.docx`
     - Research Edition V1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
     - Research Edition V2: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx`
     - Research Edition V3: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx`
     - Research Edition V4: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx`
     - Research Edition V4.1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx`
     - Research Edition V4.1.1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx`
   - Added `PROTECTED_V4_1_1_REPORT` invariant in `tools/build_final_report.py` preventing accidental overwrite of V4.1.1.

2. **Figure 7.1 Redesign for Actual Report-Size Readability:**
   - **Removed Bottom Banner:** Removed the oversized horizontal architectural-invariant banner from inside the graphic canvas to eliminate horizontal clipping and visual crowding.
   - **Moved Invariant Explanation to Caption:** The invariant description is fully articulated in the Figure 7.1 caption.
   - **Removed "Upstream data issue":** Cleaned the `INPUT REJECTION` box in Column 1 to avoid semantic confusion.
   - **Shortened Text & Enlarged Font Sizes:** Bullet text across all boxes was tightened, and effective font sizes were increased (titles to 9.6–9.8 pt bold, body text to 8.0–8.2 pt, column headers to 9.2 pt, figure title to 13.0 pt).
   - **Automated Layout Validation:** Verified **0 overlapping text pairs** across all 31 text elements, and confirmed **100% text containment** within all box boundaries and within the $[0, 100] \times [0, 100]$ canvas coordinate space.
   - **Report Sizing:** Inserted at width = 6.0 inches (3.14 inches height) in Word, rendering crisply and fitting comfortably on Page 26.
   - **Preserved Routing & Actions:**
     - `ACTION 1: ACCEPT` ($H < 0.35$ & non-empty): 5,437 slices (83.02%).
     - `ACTION 2: FALLBACK SWITCH` (empty primary + context $k=2$): 54 slices (0.82%).
     - `ACTION 3: FLAG / REFER` ($H \ge 0.35$ or unresolved empty): 1,058 slices (16.16%).
     - Total evaluated foreground cohort: 6,549 slices.

3. **Tensor-Integrity Terminology Standardization:**
   - In Section 6.4 (Stage 1), clarified that upstream data loading is responsible for assembling inputs in the expected canonical channel order (`Ch 0: T1n`, `Ch 1: T1c`, `Ch 2: T2f`, `Ch 3: T2w`).
   - Specified that the pre-inference validator checks the tensor interface, dimensionality, batch and channel counts ($[1, 4, H, W]$), and positive spatial dimensions ($H, W > 0$).
   - Explicitly disclosed that the validator does not independently identify semantic modality content or detect every all-zero placeholder. Malformed tensors raise `TypeError` or `ValueError` exceptions and halt execution prior to inference (an interface exception, not an agent triage decision).

4. **Calibration Terminology Correction:**
   - Renamed "Calibration Generalization" in Section 6.3 to "Internal Evaluation Assessment".
   - Replaced "To verify calibration generalization" with wording describing ECE measured on the internal `Val_eval` subset across $M = 15$ probability bins.
   - Retained the checkpoint-selection-bias limitation explicitly.
   - Preserved all established ECE values (0.0502 \u2192 0.0084 on calibration; 0.0861 \u2192 0.0344 on evaluation for ResUNet-2D) and fitted temperatures ($T^* = 2.8808$ for ResUNet-2D, $T^* = 2.6267$ for UNetResNet34).

5. **Bounded Missing-T1c Conclusion:**
   - In Section 7.1, replaced universal claims of entropy being "blind to subtle internal subregion collapse" with:
     > *"These empirical findings confirm that the empty-prediction guard provides robust protection against extensive tissue loss (C3); however, under the tested C2 condition (synthetic post-normalization T1c zero-clamping), the entropy-based policy did not reliably flag the observed ET and TC degradation."*
   - Preserved explicit synthetic post-normalization zero-clamping context, $p_{\text{adj}} = 1.0000$ (C2), and $p_{\text{adj}} = 0.00048$ (C4).
   - In Table C.1, updated row entry to "Missing-Modality Limitation" disclosing that under the tested C2 condition, the entropy policy did not reliably flag ET/TC degradation.

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
| **Research Edition V4.1.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx` | `58f04ddc7a6fa3b85905c7e4f0f35be8e078acff4b0a4213abb87ca85cd7bab5` | 1,996,480 | **PRESERVED** |
| **Research Edition V4.1.2** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx` | `6f1496ed45ccb398b48262bc8f853406c69a40b295b9f36fae27f641bf7cec8f` | 2,047,929 | **CREATED** |
| **Report Builder** | `tools/build_final_report.py` | `85dbe548f124889c4065a79b8a4ff3cd4a51109a2df565575c65370c5a9492d6` | 160,312 | **UPDATED** |
| **Word COM Export** | `tmp/v4_1_2_word.pdf` | `87301b76f440e4a576039de086fcf8357b147f16cfa681f4c1c146993c3b97da` | 1,149,267 | **VERIFIED** |

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

Word COM automation (`Word.Application`) confirmed an exact total of **39 pages**. All 44 document headings matched V4.1.1 and V4 page-for-page with **zero drift**:

| Heading Index | Heading Title | V4.1.1 Page | V4.1.2 Page | Drift |
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
- **PDF Text Searchability:** 100% searchable via PyMuPDF (`fitz`), with all key terminology, bounded statistical findings ($p_{\text{adj}} = 1.0000$, $p_{\text{adj}} = 0.00048$), and captions located on expected pages.

---

## 5. Environment & System Provenance

- **Operating System:** Windows 11
- **Working Directory:** `D:\MajorProject`
- **Git Branch:** `main`
- **Git HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904`
- **Git Working Tree Status:** 3 tracked modifications (`src/data/dataset.py`, `src/evaluation/metrics.py`, `src/training/train.py`) and 519 untracked entries preserved from the prior status snapshot. The working tree is explicitly not claimed to be clean.
- **Test Cohort Status:** Quarantined test cohort (`data/quarantine_test/`) and test manifest (`test_manifest.json`) remain strictly untouched and unaccessed.
- **Office / PDF Automation Engine:** Microsoft Word COM (`Word.Application`) is installed and functions as the authoritative layout, pagination, and PDF generation engine. LibreOffice (`soffice`) is not installed on the host.
