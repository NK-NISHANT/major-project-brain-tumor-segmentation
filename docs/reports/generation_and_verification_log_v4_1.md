# G1-B1 — Research Edition V4.1 Generation & Verification Log

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Generated Document:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx`
**Artwork Directory:** `tmp/report_artwork_research_v4_1/`
**PDF Export:** `tmp/v4_1_word.pdf`
**Timestamp:** 2026-11-12

---

## 1. Summary of Changes Applied in Edition V4.1

1. **Historical Edition Preservation:**
   - Byte-for-byte immutability preserved across all historical editions:
     - Original Historical Minor Project Report: `Final_Report_Brain_Tumor_Segmentation.docx`
     - Research Edition V1: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
     - Research Edition V2: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx`
     - Research Edition V3: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx`
     - Research Edition V4: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx`
   - Added `PROTECTED_V4_REPORT` invariant in `tools/build_final_report.py` preventing overwrite of V4.

2. **Input-Integrity Terminology Standardization:**
   - Standardized on precise terminology: **input tensor-integrity validation** across Abstract, Chapters 1, 3, 6, 7, Table 4.2, and Figure 7.1 caption.
   - Accurately described the checks performed by the validator (`src/agent/models.py:241-270`):
     - Input is a valid `torch.Tensor`.
     - Expected four-dimensional tensor shape `[1, 4, H, W]`.
     - Batch dimension equals one.
     - Four input channels matching canonical sequence order.
     - Positive spatial dimensions ($H, W > 0$).
   - Explicitly clarified that invalid inputs raise `TypeError` or `ValueError` exceptions and halt execution prior to inference, rather than generating an `AgentDecision` referral action.
   - Disclosed that semantic modality presence and correct sequence identification remain strictly an upstream data-loading responsibility; synthetic zero-clamping does not validate the integrity gate.

3. **Canonical Channel Ordering Alignment:**
   - Explicitly mapped the canonical 4-channel input layout across Figure 7.1, Chapter 3, and Chapter 6:
     - **Channel 0:** T1n (native T1)
     - **Channel 1:** T1c (contrast-enhanced T1)
     - **Channel 2:** T2f (T2 FLAIR)
     - **Channel 3:** T2w (T2-weighted)

4. **Agent Policy Narrative Alignment:**
   - Documented the authoritative B2 triage policy:
     - **ACCEPT:** Calibrated entropy $H < 0.35$ with non-empty foreground (5,437 slices, 83.02%).
     - **FALLBACK_SWITCH:** Primary ResUNet-2D produces empty foreground despite confirmed spatial axial-context support (radius $k = 2$), triggering evaluation of candidate UNetResNet34 ($T^* = 2.6267$) where non-empty output is retained (54 slices, 0.82%).
     - **FLAG_UNRELIABLE / SIMULATED REFERRAL:** Calibrated entropy $H \ge 0.35$ or unresolvable empty-primary prediction (1,058 slices, 16.16%).
   - Total evaluated foreground census: 6,549 slices across 105 evaluation cases (5,437 + 54 + 1,058 = 6,549 slices).
   - Removed legacy claims that moderate entropy triggers fallback switching.
   - Clarified that invalid tensor interfaces cause execution halts/exceptions, not referral actions.

5. **Training Precision Reconciliation:**
   - Reconciled Section 4.4 and Table 4.2 against Appendix B and the actual codebase (`src/training/train.py`, `train_scaled.py`).
   - Removed inaccurate references to mixed-precision floating-point arithmetic (`torch.cuda.amp.autocast`).
   - Standardized on: "Full-precision FP32 training, validation loop, checkpointing and JSON history logging."

6. **Bounded Missing-T1c Conclusions:**
   - Replaced broad statements with the empirically bounded finding:
     *"Under the tested synthetic post-normalization T1c zero-clamping condition, severe ET and TC degradation was not accompanied by a statistically significant increase in aggregate referral (pooled referral shifted only to 17.44%, $p_{\text{adj}} = 1.0000$). This demonstrates a specific limitation of the tested entropy-based policy and motivates explicit modality-presence and identity validation upstream prior to tensor assembly. The proportion of severe compartment-level failures referred remains unmeasured, as existing records report only aggregate foreground referral."*
   - Preserved C2 $p_{\text{adj}} = 1.0000$ and C4 $p_{\text{adj}} = 0.00048$.

7. **Shannon Entropy Citation Correction:**
   - Corrected citation in Section 1.3 from `[6]` (Guo et al., calibration) to `[9]` (Shannon 1948, information theory).

8. **Figure 7.1 Architecture & Triage Diagram Enhancements:**
   - Re-rendered Figure 7.1 at 300 DPI with explicit channel mapping (`Ch 0: T1n`, `Ch 1: T1c`, `Ch 2: T2f`, `Ch 3: T2w`).
   - Created a visually distinct, dashed `INPUT REJECTION` exception box illustrating pre-inference halts on malformed tensors (separate from triage decisions).
   - Repositioned decision route arrows with clean non-overlapping trajectories to Action 1 (Accept), Action 2 (Fallback Switch), and Action 3 (Flag / Refer).

---

## 2. File Verification & Exact SHA-256 Digests

| Artifact | File Path | SHA-256 Digest | Size (bytes) | Status |
|---|---|---|---:|:---:|
| **Original Minor Report** | `Final_Report_Brain_Tumor_Segmentation.docx` | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | 4,301,505 | **PRESERVED** |
| **Research Edition V1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | 3,112,658 | **PRESERVED** |
| **Research Edition V2** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | 1,905,004 | **PRESERVED** |
| **Research Edition V3** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx` | `2d85fc7cd92ae80371ec31584b36535a0d4772a7be6018e5a4d00b6f15ebb90b` | 1,928,076 | **PRESERVED** |
| **Research Edition V4** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx` | `bb8f02746bbc55dd38ac86bcb1b36b3c878b8284a6919a3496d6530855d709bb` | 1,968,938 | **PRESERVED** |
| **Research Edition V4.1** | `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx` | `3d9e09e4417adb0df826cfce8dcf6c1161e5131d7927b197849cece8237a65ee` | 1,992,079 | **CREATED** |
| **Report Builder** | `tools/build_final_report.py` | `5b0ec24492d3034f833f0a17652d8b81dca73d119bfcd0eee46fa6501936faf7` | 158,951 | **UPDATED** |
| **Word COM Export** | `tmp/v4_1_word.pdf` | `ba4f69b384d45ef09c33049046e56c1e80025ce04a9e56bfa5a12e2a5e82568d` | 1,162,052 | **VERIFIED** |

---

## 3. Structural Package Validation

- **OpenXML Package Integrity:** Valid `.docx` zip archive containing `word/document.xml`, `[Content_Types].xml`, `word/_rels/document.xml.rels`.
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
  - `image11.png` (Figure 6.6: Complete Foreground Miss Rates)
  - `image12.png` (Figure 7.1: Implemented AI Agent Architecture & Triage Pipeline)
- **Table Count:** Exactly 21 tables (1 title-page roster table + 20 numbered content tables).
- **Row-Splitting Protection:** 224 / 224 rows (100.0%) have `<w:cantSplit/>` enforced.
- **Table Header Repeat:** 20 / 20 content tables (100.0%) have `<w:tblHeader/>` enforced on row 0.
- **Placeholder Audit:** 0 occurrences of `TODO`, `TBD`, `FIXME`, `XXX`, `???`, or `[placeholder]`.

---

## 4. Pagination & Heading Zero-Drift Audit

Exported via Word COM (`Word.Application` SaveAs PDF, format 17).
**Total Page Count:** Exactly **39 pages**.
**Searchability:** 100% searchable text with valid font mappings.

### Heading Alignment (44 / 44 Headings Match Page-for-Page with V4)

| Heading Text | Style | V4 Page | V4.1 Page | Drift |
|---|---|:---:|:---:|:---:|
| `CHAPTER 1: INTRODUCTION` | Heading 1 | 1 | 1 | 0 |
| `1.1 Clinical and Technical Context` | Heading 2 | 1 | 1 | 0 |
| `1.2 Limitations of Segmentation Dice as a Sole Reliability Measure` | Heading 2 | 1 | 1 | 0 |
| `1.3 Predictive Uncertainty and Reliability-Aware Decision-Making` | Heading 2 | 2 | 2 | 0 |
| `1.4 Research Motivation, Objectives, and Scope` | Heading 2 | 2 | 2 | 0 |
| `1.5 Summary of Project Contributions` | Heading 2 | 3 | 3 | 0 |
| `1.6 Report Organization` | Heading 2 | 4 | 4 | 0 |
| `CHAPTER 2: BACKGROUND STUDY AND RELATED WORK` | Heading 1 | 5 | 5 | 0 |
| `2.1 Multi-modal MRI and Glioma Subregions` | Heading 2 | 5 | 5 | 0 |
| `2.2 Deep Learning Architectures for Medical Image Segmentation` | Heading 2 | 5 | 5 | 0 |
| `2.3 Evaluation, Overlap, and Distance Metrics` | Heading 2 | 6 | 6 | 0 |
| `2.4 Predictive Uncertainty Quantification in Medical Imaging` | Heading 2 | 6 | 6 | 0 |
| `2.5 Probability Calibration in Neural Networks` | Heading 2 | 7 | 7 | 0 |
| `2.6 Selective Prediction, Triage, and Reliability-Aware AI` | Heading 2 | 7 | 7 | 0 |
| `CHAPTER 3: DATASET CENSUS AND PARTITIONING METHODOLOGY` | Heading 1 | 8 | 8 | 0 |
| `3.1 BraTS 2023 GLI Dataset Overview` | Heading 2 | 8 | 8 | 0 |
| `3.2 Subject-Level Dataset Census and Split Accounting` | Heading 2 | 8 | 8 | 0 |
| `3.3 Historical 197-Case Pilot Split vs Mature Cohort` | Heading 2 | 9 | 9 | 0 |
| `3.4 Data Preprocessing and Tensor Normalization` | Heading 2 | 9 | 9 | 0 |
| `CHAPTER 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE` | Heading 1 | 11 | 11 | 0 |
| `4.1 Five-Model Architecture Portfolio` | Heading 2 | 11 | 11 | 0 |
| `4.2 Structural Details of Stage A Architectures` | Heading 2 | 12 | 12 | 0 |
| `4.3 Structural Details of Stage B Architectures` | Heading 2 | 13 | 13 | 0 |
| `4.4 Training Dynamics and Loss Optimization Framework` | Heading 2 | 13 | 13 | 0 |
| `4.5 Software Architecture and Repository Design` | Heading 2 | 14 | 14 | 0 |
| `CHAPTER 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS` | Heading 1 | 15 | 15 | 0 |
| `5.1 Two-Stage Experimental Progression` | Heading 2 | 15 | 15 | 0 |
| `5.2 Stage A: Historical Five-Epoch Pilot Benchmark` | Heading 2 | 15 | 15 | 0 |
| `5.3 Stage B: Mature 15-Epoch Research Baselines` | Heading 2 | 17 | 17 | 0 |
| `5.4 Checkpoint-Selection Bias and Methodological Boundaries` | Heading 2 | 18 | 18 | 0 |
| `CHAPTER 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE` | Heading 1 | 19 | 19 | 0 |
| `6.1 Predictive Uncertainty and Segmentation Quality Correlation` | Heading 2 | 19 | 19 | 0 |
| `6.2 Separation of Uncertainty Binning and Slice Distribution` | Heading 2 | 20 | 20 | 0 |
| `6.3 Post-Hoc Probability Calibration and Temperature Scaling` | Heading 2 | 21 | 21 | 0 |
| `6.4 Implemented Deterministic Reliability-Aware Agent` | Heading 2 | 22 | 22 | 0 |
| `CHAPTER 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS` | Heading 1 | 23 | 23 | 0 |
| `7.1 Missing-Modality Robustness Stress Study` | Heading 2 | 23 | 23 | 0 |
| `7.2 Methodological Boundaries and Comprehensive Limitations` | Heading 2 | 24 | 24 | 0 |
| `7.3 Exploratory 3D / Foundation Model Pilot Studies` | Heading 2 | 25 | 25 | 0 |
| `7.4 Future Research Directions and Conclusion` | Heading 2 | 25 | 25 | 0 |
| `REFERENCES` | Heading 1 | 27 | 27 | 0 |
| `APPENDIX A: FULL DATASET SPLIT MANIFEST AND CENSUS AUDIT` | Heading 1 | 28 | 28 | 0 |
| `APPENDIX B: REPRODUCIBILITY RECORD AND HYPERPARAMETER SPECIFICATIONS` | Heading 1 | 28 | 28 | 0 |
| `APPENDIX C: RESEARCH GOVERNANCE, VERIFICATION CHECKLIST, AND INTEGRITY DECLARATION` | Heading 1 | 29 | 29 | 0 |

---

## 5. Git Status & Provenance Record

- **Branch:** `main`
- **HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904`
- **Quarantine Invariant:** The sealed test cohort (`data/quarantine_test/` and `test_manifest.json`) remains completely uninspected, unmounted, and untouched.
- **Integrity Status:** All audit evidence requirements, architectural diagrams, model parameter accountings, ECE figures, and triage narratives strictly conform to `MASTER_PROJECT_BLUEPRINT_v2.0.md` and the final `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`.
