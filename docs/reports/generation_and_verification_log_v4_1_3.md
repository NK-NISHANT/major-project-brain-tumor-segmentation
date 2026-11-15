# G1-B1 — Research Edition V4.1.3 Generation & Verification Log

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Generated Document:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_3.docx`
**Artwork Directory:** `tmp/report_artwork_research_v4_1_3/`
**PDF Export:** `tmp/v4_1_3_word.pdf`
**Timestamp:** 2026-11-12

---

## 1. Summary of Changes Applied in Edition V4.1.3

1. **Historical Edition Preservation:**
   - Byte-for-byte immutability strictly preserved across all eight historical editions:
     - Original Historical Minor Project Report: `Final_Report_Brain_Tumor_Segmentation.docx`
     - Research Edition V1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
     - Research Edition V2: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx`
     - Research Edition V3: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx`
     - Research Edition V4: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx`
     - Research Edition V4.1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx`
     - Research Edition V4.1.1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx`
     - Research Edition V4.1.2: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx`
   - Added `PROTECTED_V4_1_2_REPORT` invariant check to `tools/build_final_report.py`, prohibiting accidental overwrite of V4.1.2.
   - Updated `DEFAULT_OUT` to `Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_3.docx` and `DEFAULT_ARTWORK_DIR` to `tmp/report_artwork_research_v4_1_3`.

2. **CRITICAL — Verified C3/C4 Channel-Index Mapping:**
   - Verified canonical channel ordering in `src/data/dataset.py` (`MODALITIES = ("t1n", "t1c", "t2f", "t2w")`):
     - Channel 0: T1n (native T1)
     - Channel 1: T1c (contrast-enhanced T1)
     - Channel 2: T2f (T2 Fluid-Attenuated Inversion Recovery)
     - Channel 3: T2w (T2-weighted)
   - Corrected Table 7.1 rows:
     - Condition C3: Withholding T2w is mapped to Channel 3 (`Omitted T2w (channel 3)`).
     - Condition C4: Withholding T2f is mapped to Channel 2 (`Omitted T2f (channel 2)`).
   - Retained all empirical Dice, referral percentages, and p-values exactly as measured without modification.

3. **Training Configuration Consistency:**
   - Revised Section 4.4 to explicitly distinguish Stage A and Stage B training settings:
     - Stage A (historical 5-epoch pilot benchmark): mini-batch size of 2 axial slices, weight decay of $1 \times 10^{-4}$ on the 197-case pilot split.
     - Stage B (mature 15-epoch research baselines): mini-batch size of 16 axial slices, cosine learning-rate decay, gradient clipping norm 1.0 on 721 training cases.
   - Section 4.4 is now 100% harmonized with Section 5.2, Section 5.3, and Appendix B Table B.1.

4. **Missing-Modality Objective:**
   - Updated Section 1.4 bullet to include all four tested MRI sequence ablations:
     - `"Missing-Modality Stress Testing: Systematically evaluate model robustness and uncertainty responsiveness under missing-channel stress conditions across all four MRI inputs (ablating T1n, T1c, T2w, and T2f), discovering failure modes and uncertainty blind spots."`
   - Harmonized Section 1.4 with Table 7.1 and Section 7.1.

5. **3D Pilot & Model-Comparison Language:**
   - Section 7.3: Described SegResNet-3D accurately as an exploratory patch-based 3D feasibility pilot (`MONAI SegResNet`, 1,176,852 parameters, 10 training cases, 5 epochs, 50 steps, $96 \times 96 \times 96$ sub-volume crops, batch size 1).
   - Recorded exact execution metrics from `results/segresnet3d/segresnet3d_pilot_summary.json`: compound loss decreased from 2.0956 to 1.7366, peak training VRAM 622.3 MB, exploratory reconstruction WT Dice of 0.2658.
   - Explicitly clarified that this pilot was not full-volume training and did not reach convergence or benchmark parity with the mature 2D baselines.
   - Section 7.2: Updated limitations bullet to state SegResNet-3D was a bounded patch-based 5-epoch feasibility pilot ($96 \times 96 \times 96$ crops).
   - Table 7.2: Replaced unsupported claim that 3D CNNs are "Current state-of-the-art for clinical volumetric contouring" with a bounded research-oriented description: `"Established research standard for volumetric segmentation benchmarks"`.

6. **Citation & Claim Corrections:**
   - Section 2.2 and Reference [11]: Accurately attributed Zhang, Liu, and Wang (2018) to its actual domain—remote sensing road feature delineation—and explained how its introduction of residual units into U-Net motivated and inspired residual U-Net architectures in biomedical image segmentation.
   - Section 1.1: Removed unverified ~80% exact prevalence percentage, replacing it with verified epidemiological wording: `"accounting for the vast majority of primary malignant brain tumors"`.
   - Zero fabricated references or unverified citations.

7. **Academic Front Matter Governance:**
   - Title Page: Retained Course Name ("Major Project-II"), Course Code ("15B19CI791"), and Semester ("4th Year 8th Semester"). Flagged for student confirmation against official institutional registration.
   - Acknowledgement: Refined compute platform statement to acknowledge local dedicated infrastructure (NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM) while expressing appreciation to the department and institute for academic facilities.
   - Preserved signature placeholders and dates without inventing approvals or signatures.

8. **Pagination Correction (Table Header Orphans Eliminated):**
   - Added `keep_with_next = True` to all paragraphs of table header cells in `add_table`.
   - In Word OpenXML layout, this ensures `<w:keepNext/>` is applied, preventing Word from splitting a table immediately after its header row.
   - Eliminated the orphaned Table 3.1 header from the bottom of printed page 8 (PDF page 18): Table 3.1 now starts cleanly at the top of printed page 9 (PDF page 19) with its header and data rows together.
   - Eliminated the orphaned Table 7.2 header from the bottom of printed page 24 (PDF page 34): Table 7.2 now starts cleanly at the top of printed page 25 (PDF page 35) with its header and data rows together.
   - Preserved readable font sizes, column widths, and margins throughout.

---

## 2. File Verification & Cryptographic Provenance

### Output Artifacts (Edition V4.1.3)
| Artifact | Absolute File Path | File Size (Bytes) | SHA-256 Digest |
|---|---|---|---|
| **DOCX** | `D:\MajorProject\docs\reports\Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_3.docx` | 2,048,297 | `216af823d9e188392f7d2a915fa0acff610b297da8662cd61724491fc58df20a` |
| **PDF** | `D:\MajorProject\tmp\v4_1_3_word.pdf` | 1,148,771 | `08b4d019a97dc1d8bb834ef4af9b9e627447cce8dc3498e024ec6c594647e2e6` |
| **Page Log** | `D:\MajorProject\tmp\v4_1_3_pages.json` | 6,188 | `ec015ff17b9b1e779a11ef065983777d0663dbb45db0adbb7ef945fe5966ae8f` |

### Byte-for-Byte Immutability of Protected Editions
| Edition | File Path | File Size (Bytes) | SHA-256 Digest | Status |
|---|---|---|---|---|
| **Original Minor** | `Final_Report_Brain_Tumor_Segmentation.docx` | 4,301,505 | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | **VERIFIED INTACT** |
| **Research V1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` | 3,112,658 | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | **VERIFIED INTACT** |
| **Research V2** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` | 1,905,004 | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | **VERIFIED INTACT** |
| **Research V3** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx` | 1,928,076 | `2d85fc7cd92ae80371ec31584b36535a0d4772a7be6018e5a4d00b6f15ebb90b` | **VERIFIED INTACT** |
| **Research V4** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx` | 1,968,938 | `bb8f02746bbc55dd38ac86bcb1b36b3c878b8284a6919a3496d6530855d709bb` | **VERIFIED INTACT** |
| **Research V4.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx` | 1,992,079 | `3d9e09e4417adb0df826cfce8dcf6c1161e5131d7927b197849cece8237a65ee` | **VERIFIED INTACT** |
| **Research V4.1.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx` | 1,996,480 | `58f04ddc7a6fa3b85905c7e4f0f35be8e078acff4b0a4213abb87ca85cd7bab5` | **VERIFIED INTACT** |
| **Research V4.1.2** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx` | 2,047,929 | `6f1496ed45ccb398b48262bc8f853406c69a40b295b9f36fae27f641bf7cec8f` | **VERIFIED INTACT** |

---

## 3. OpenXML Package Integrity & Structural Metrics

- **Total Tables in Document:** 21 tables
- **Total Rows across All Tables:** 224 rows
- **Rows with `w:cantSplit` Attribute:** 224 rows (100% coverage across all table rows)
- **Rows with `w:tblHeader` Attribute:** 20 rows (all multi-row data tables repeat header row on page breaks)
- **Header Cells with `w:keepNext` Paragraph Format:** 79 paragraphs (100% header row coverage)
- **Total Embedded Media / Figures:** 12 images (`word/media/image1.png` through `image12.png`)
- **Unresolved Placeholders / Template Markers:** 0 placeholders found

---

## 4. Word COM PDF Rendering & Pagination Verification

- **Total Rendered Pages:** Exactly 39 pages (10 preliminary pages `i`–`x`, 29 main content pages `1`–`29`).
- **Heading Alignment & Page Drift:** 44 headings audited across all 39 pages. Zero drift relative to V4.1.2.
- **Table 3.1 Orphan Elimination:**
  - Printed page 8 (PDF page 18) ends with: `"To guarantee complete methodological integrity, our audit established that the 1,251 cases in the project repository belong to 1,133 unique biological subjects. We instituted strict subject-level grouping to partition the census into completely subject-disjoint cohorts. Table 3.1 details the full project dataset accounting."`
  - Printed page 9 (PDF page 19) starts with Table 3.1 header: `Partition | Cases | Subjects | Axial Slices | Operational Status / Usage`.
  - Header row is NOT stranded alone on page 8.
- **Table 7.2 Orphan Elimination:**
  - Printed page 24 (PDF page 34) ends with: `"Non-Clinical Academic Prototype: The software contains no DICOM parser, PACS integration, or prospective clinical validation, and does not make diagnostic or treatment recommendations."`
  - Printed page 25 (PDF page 35) starts with Table 7.2 header: `Segmentation Paradigm | Core Strengths | Primary Technical Limitations | Suitability for Clinical Workflow`.
  - Header row is NOT stranded alone on page 24.
- **List of Tables (LOT) Alignment:** 16 tables verified. 100% alignment between LOT listed page numbers and physical PDF rendered pages.
- **List of Figures (LOF) Alignment:** 11 figures verified. 100% alignment between LOF listed page numbers and physical PDF rendered pages.

---

## 5. Text Searchability & Term Verification in Rendered PDF

| Verified Text Expression | Target Section / Context | Rendered Page | Search Status |
|---|---|---|---|
| `Omitted T2w (channel 3)` | Table 7.1 Row C3 | Printed p. 23 (PDF p. 33) | **VERIFIED 100%** |
| `Omitted T2f (channel 2)` | Table 7.1 Row C4 | Printed p. 23 (PDF p. 33) | **VERIFIED 100%** |
| `mini-batch size of 2` | Section 4.4 Stage A Settings | Printed p. 13 (PDF p. 23) | **VERIFIED 100%** |
| `mini-batch size of 16` | Section 4.4 Stage B Settings | Printed p. 13 (PDF p. 23) | **VERIFIED 100%** |
| `ablating T1n, T1c, T2w, and T2f` | Section 1.4 Research Objectives | Printed p. 3 (PDF p. 13) | **VERIFIED 100%** |
| `SegResNet-3D Feasibility Pilot` | Section 7.3 Exploratory Studies | Printed p. 25 (PDF p. 35) | **VERIFIED 100%** |
| `96 x 96 x 96` | Section 7.2 & 7.3 Patch Crops | Printed p. 24 & 25 | **VERIFIED 100%** |
| `0.2658` | Section 7.3 Reconstruction WT Dice | Printed p. 25 (PDF p. 35) | **VERIFIED 100%** |
| `Established research standard for volumetric segmentation benchmarks` | Table 7.2 Row 2 Paradigm | Printed p. 25 (PDF p. 35) | **VERIFIED 100%** |
| `remote sensing feature delineation` | Section 2.2 Zhang et al. Attribution | Printed p. 6 (PDF p. 16) | **VERIFIED 100%** |
| `accounting for the vast majority of primary malignant brain tumors` | Section 1.1 Clinical Background | Printed p. 1 (PDF p. 11) | **VERIFIED 100%** |

---

## 6. Strict Research Boundaries & Repository State

- **Quarantined Test Cohort:** `results/dataset_splits/test_manifest.json` (47,665 bytes, LastWriteTime 2026-10-27 12:38:30) was strictly preserved unopened and uninspected.
- **Model Checkpoints:** Frozen baseline checkpoints for ResUNet-2D (Epoch 13) and UNetResNet34 (Epoch 15) remain unmodified.
- **No Model Training:** No model retraining, fine-tuning, or inference was executed during report generation.
- **Git Repository State:**
  - **Branch:** `main`
  - **HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904` ("Merge remote cleanup commits")
  - **Tracked Modifications (3 files):**
    - `M src/data/dataset.py`
    - `M src/evaluation/metrics.py`
    - `M src/training/train.py`
  - **Untracked Directory Snapshot:** Untracked entries (`Report_Materials/`, `docs/reports/`, `results/`, `tmp/`, `tools/`) remain preserved.
