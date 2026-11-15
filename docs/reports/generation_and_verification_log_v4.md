# Generation and Verification Log — Research Edition V4

**Document Title:** Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent
**Edition:** Research Edition V4 (Final Sign-Off Patch: Actual Code Behavior, Appendix A Slice Accounting, and Claim Governance)
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Final Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Target Document Path:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx`
**Artwork Directory:** `tmp/report_artwork_research_v4/`
**Authoritative Builder Script:** `tools/build_final_report.py`
**Log Timestamp:** November 12, 2026

---

## 1. Executive Summary & Verification Outcomes

The Research Edition V4 generation implements the final, narrowly scoped corrections authorized under the G1-B1 milestone. All modifications were restricted to `tools/build_final_report.py` and the newly generated V4 document. Research Edition V1, V2, V3, and the original Minor Project report remain byte-for-byte identical.

### Summary of Key Compliance Metrics
| Item / Metric | Verified Value in V4 | Governance Target | Compliance Status |
|---|---|---|:---:|
| **Total Document Pages** | **39 pages** (1 cover, 9 prelims i–ix, 29 body pages 1–29) | Exactly 39 pages, 0 blank pages | **PASSED** |
| **Total Body Chapters** | 7 Chapters + References + 3 Appendices | Complete research portfolio | **PASSED** |
| **Total Tables in DOCX** | 21 tables (5 prelim/cover + 16 numbered report tables) | 21 tables | **PASSED** |
| **Total Embedded Media Parts** | 12 images (1 institutional logo + 11 research figures) | 12 parts | **PASSED** |
| **`w:cantSplit` Table Row Coverage** | 224 / 224 rows (100.0% coverage across all tables) | 100% coverage | **PASSED** |
| **`w:tblHeader` Table Header Coverage** | 20 / 20 content tables (100% of tables with headers) | 100% coverage | **PASSED** |
| **Placeholder Tokens (`TODO`, `TBD`, etc.)** | Exactly 0 found across entire document | Exactly 0 | **PASSED** |
| **Table of Contents (TOC) Drift** | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| **List of Tables (LOT) Drift** | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| **List of Figures (LOF) Drift** | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| **Body Page 22 Layout** | Section 6.4 + Table 6.3 + Caption + Note together | Complete section on Page 22 | **PASSED** |
| **Body Page 23 Layout** | Chapter 7 clean top-of-page opening | No isolated text / full page | **PASSED** |
| **Body Page 26 Layout** | Figure 7.1 + Complete Caption + Conclusion | Complete layout on Page 26 | **PASSED** |
| **Body Page 28 Layout** | Appendix A + Table A.1 + Note + Appendix B + Table B.1 | Clean layout on Page 28 | **PASSED** |
| **Body Page 29 Layout** | Heading, intro, Table C.1, and caption together on Page 29 | Zero orphan headings | **PASSED** |
| **Minor Project Baseline Preservation** | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| **Research Edition V1 Preservation** | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| **Research Edition V2 Preservation** | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| **Research Edition V3 Preservation** | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| **Sealed Test Cohort Quarantine** | Untouched; zero access, opening, or evaluation | Sealed quarantine | **PASSED** |

---

## 2. File Integrity & Baseline Preservation

All historical documents, raw datasets, pre-rendered charts, and model checkpoints remained strictly read-only throughout the patch:

| Artifact / File Path | File Size | SHA-256 Digest | Status |
|---|---:|---|:---:|
| `Final_Report_Brain_Tumor_Segmentation.docx` (Minor Project Baseline) | 4,301,505 bytes | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` (V1 Baseline) | 3,112,658 bytes | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` (V2 Baseline) | 1,905,004 bytes | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx` (V3 Baseline) | 1,928,076 bytes | `2d85fc7cd92ae80371ec31584b36535a0d4772a7be6018e5a4d00b6f15ebb90b` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx` (Target V4 Report) | 1,968,938 bytes | `bb8f02746bbc55dd38ac86bcb1b36b3c878b8284a6919a3496d6530855d709bb` | **GENERATED** |
| `tools/build_final_report.py` (Authoritative Builder Script) | 156,505 bytes | `f85a3759881ada1a2b2c3e11cc2225f135d8a099268146cd227f0e69244b0ed2` | **VERIFIED** |

---

## 3. Figure 7.1 Architecture Diagram and Caption Alignment

The visual diagram for Figure 7.1 (`tmp/report_artwork_research_v4/agent_architecture.png`) and its DOCX caption were updated to match the production codebase:

1. **Input Rejection / Exception Path:** Added an isolated failure path branching from the Tensor Integrity Gate to an `INPUT REJECTION` container. Corrupted shapes, invalid dimensions, or non-tensor inputs halt execution by raising `TypeError` or `ValueError` exceptions before model execution. This path does **not** connect to or produce an `AgentDecision` referral action.
2. **Action 1 Labeling (ACCEPT):** Labeled strictly as $H < 0.35$ and non-empty foreground, accepting the primary ResUNet-2D prediction (5,437 slices, 83.02%).
3. **Action 2 Labeling (FALLBACK SWITCH):** Labeled as empty primary prediction with confirmed neighboring axial-context support ($k=2$, `foreground_neighbors_count >= 1`), substituting candidate UNetResNet34 (54 slices, 0.82%).
4. **Action 3 Labeling (FLAG / REFER):** Labeled as $H \ge 0.35$ or an unresolvable empty-primary prediction, generating a simulated referral flag for review (1,058 slices, 16.16%).
5. **Elimination of Intermediate Entropy Band:** The exploratory fallback interval ($0.35 < H \le 0.40$) and any implication of entropy-based candidate switching in canonical B2 were completely removed from diagram, caption, and text.
6. **Action Mutual Exclusivity Census:** Preserved exact three-way census across all 105 evaluation cases:
   - Direct Primary Acceptance: 5,437 slices (83.02%)
   - Fallback Retention: 54 slices (0.82%)
   - Simulated Referral: 1,058 slices (16.16%)
   - Total Evaluated Foreground Cohort: **6,549 slices (100.00%)**

---

## 4. Appendix A Dataset Split Census & Slice Accounting

In compliance with quarantine governance, the unverified `~100,000+ slices` estimate was removed from Table A.1 and Table 3.1:

1. **Documented Partitions:**
   - Train Partition: 941 cases, 847 subjects, 77,020 axial slices.
   - Validation Pool: 155 cases, 143 subjects, 12,364 axial slices.
   - Verified Development Population Total: **1,096 cases, 990 subjects, 89,384 axial slices**.
   - Sealed Test Cohort: 155 cases, 143 subjects, Quarantined (unopened, uninspected slice count).
2. **Explicit Full Census Declaration:** Table A.1 and Table 3.1 explicitly state:
   > *"Total slice count not reported because the sealed test cohort's slice count remains uninspected under strict quarantine."*
3. **Quarantine Invariant:** Zero test files or test manifests were accessed, preserving blind evaluation integrity.

---

## 5. Scope, Claim Boundaries, and Terminology Corrections

1. **Simulated Referral Terminology:** Replaced occurrences of "expert inspection" and "mandatory radiologist review" with "simulated referral for review" or "flagged for review" across Chapters 1, 2, 6, 7, and Conclusion.
2. **Integrity Gate vs Synthetic Ablation Distinction:** Chapter 7 explicitly distinguishes between:
   - Upstream tensor-integrity validation implemented in the input pipeline (`src/agent/models.py`).
   - Synthetic channel zero-clamping experiments evaluating post-normalization network response under zeroed arrays.
   - Emphasized that the synthetic ablation stress study does **not** validate the modality integrity gate itself.
3. **Preservation of Resolved Statistical Findings:**
   - Condition C4 (Missing T2-FLAIR) Bonferroni-adjusted p-value: **$p_{\text{adj}} = 0.0004847664 \approx 0.00048$** (two-sided paired Wilcoxon test across $N=96$ paired subjects, $W = 793.5$, $p_{\text{raw}} = 1.21 \times 10^{-4}$, $k=4$ comparisons vs C0).
   - Condition C2 (Missing T1c) non-significance preserved: $p_{\text{adj}} = 1.0000$ (pooled referral shifts only from 16.16% to 17.44%, exposing the contrast uncertainty blind spot).
   - Condition C3 (Missing T2w) empty-guard interception preserved: pooled referral surges to 62.65% ($p_{\text{adj}} = 7.12 \times 10^{-17}$).

---

## 6. Document Layout and Table of Contents Alignment Map

Word COM layout verification confirmed 100% zero-drift alignment across all 52 TOC, 16 LOT, and 11 LOF entries:

| Content Item / Section | Heading Level | Recorded Page | Word COM Page | Alignment Status |
|---|:---:|:---:|:---:|:---:|
| **Prefatory Matter** | | | | |
| Acknowledgement | Prelim | i | i (Page 1) | **EXACT MATCH** |
| Declaration | Prelim | ii | ii (Page 2) | **EXACT MATCH** |
| Certificate | Prelim | iii | iii (Page 3) | **EXACT MATCH** |
| Abstract | Prelim | iv | iv (Page 4) | **EXACT MATCH** |
| Table of Contents | Prelim | vi | vi (Page 6) | **EXACT MATCH** |
| List of Tables | Prelim | vii | vii (Page 7) | **EXACT MATCH** |
| List of Figures | Prelim | viii | viii (Page 8) | **EXACT MATCH** |
| List of Abbreviations | Prelim | ix | ix (Page 9) | **EXACT MATCH** |
| **Report Body** | | | | |
| Chapter 1: INTRODUCTION | Heading 1 | 1 | 1 | **EXACT MATCH** |
| 1.1 Clinical and Technical Context | Heading 2 | 1 | 1 | **EXACT MATCH** |
| 1.2 Limitations of Segmentation Dice as a Sole Reliability Measure | Heading 2 | 1 | 1 | **EXACT MATCH** |
| 1.3 Predictive Uncertainty and Reliability-Aware Decision-Making | Heading 2 | 2 | 2 | **EXACT MATCH** |
| 1.4 Research Motivation, Objectives, and Scope | Heading 2 | 2 | 2 | **EXACT MATCH** |
| 1.5 Summary of Project Contributions | Heading 2 | 3 | 3 | **EXACT MATCH** |
| 1.6 Report Organization | Heading 2 | 4 | 4 | **EXACT MATCH** |
| Chapter 2: BACKGROUND STUDY AND RELATED WORK | Heading 1 | 5 | 5 | **EXACT MATCH** |
| 2.1 Multi-modal MRI and Glioma Subregions | Heading 2 | 5 | 5 | **EXACT MATCH** |
| 2.2 Deep Learning Architectures for Medical Image Segmentation | Heading 2 | 5 | 5 | **EXACT MATCH** |
| 2.3 Evaluation, Overlap, and Distance Metrics | Heading 2 | 6 | 6 | **EXACT MATCH** |
| 2.4 Predictive Uncertainty Quantification in Medical Imaging | Heading 2 | 6 | 6 | **EXACT MATCH** |
| 2.5 Probability Calibration in Neural Networks | Heading 2 | 7 | 7 | **EXACT MATCH** |
| 2.6 Selective Prediction, Triage, and Reliability-Aware AI | Heading 2 | 7 | 7 | **EXACT MATCH** |
| Chapter 3: DATASET CENSUS AND PARTITIONING METHODOLOGY | Heading 1 | 8 | 8 | **EXACT MATCH** |
| 3.1 BraTS 2023 GLI Dataset Overview | Heading 2 | 8 | 8 | **EXACT MATCH** |
| 3.2 Subject-Level Dataset Census and Split Accounting | Heading 2 | 8 | 8 | **EXACT MATCH** |
| 3.3 Historical 197-Case Pilot Split vs Mature Cohort | Heading 2 | 9 | 9 | **EXACT MATCH** |
| 3.4 Data Preprocessing and Tensor Normalization | Heading 2 | 9 | 9 | **EXACT MATCH** |
| Chapter 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE | Heading 1 | 11 | 11 | **EXACT MATCH** |
| 4.1 Five-Model Architecture Portfolio | Heading 2 | 11 | 11 | **EXACT MATCH** |
| 4.2 Structural Details of Stage A Architectures | Heading 2 | 12 | 12 | **EXACT MATCH** |
| 4.3 Structural Details of Stage B Architectures | Heading 2 | 13 | 13 | **EXACT MATCH** |
| 4.4 Training Dynamics and Loss Optimization Framework | Heading 2 | 13 | 13 | **EXACT MATCH** |
| 4.5 Software Architecture and Repository Design | Heading 2 | 14 | 14 | **EXACT MATCH** |
| Chapter 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS | Heading 1 | 15 | 15 | **EXACT MATCH** |
| 5.1 Two-Stage Experimental Progression | Heading 2 | 15 | 15 | **EXACT MATCH** |
| 5.2 Stage A: Historical Five-Epoch Pilot Benchmark | Heading 2 | 15 | 15 | **EXACT MATCH** |
| 5.3 Stage B: Mature 15-Epoch Research Baselines | Heading 2 | 17 | 17 | **EXACT MATCH** |
| 5.4 Checkpoint-Selection Bias and Methodological Boundaries | Heading 2 | 18 | 18 | **EXACT MATCH** |
| Chapter 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE | Heading 1 | 19 | 19 | **EXACT MATCH** |
| 6.1 Predictive Uncertainty and Segmentation Quality Correlation | Heading 2 | 19 | 19 | **EXACT MATCH** |
| 6.2 Separation of Uncertainty Binning and Slice Distribution | Heading 2 | 20 | 20 | **EXACT MATCH** |
| 6.3 Post-Hoc Probability Calibration and Temperature Scaling | Heading 2 | 21 | 21 | **EXACT MATCH** |
| 6.4 Implemented Deterministic Reliability-Aware Agent | Heading 2 | 22 | 22 | **EXACT MATCH** |
| Chapter 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS | Heading 1 | 23 | 23 | **EXACT MATCH** |
| 7.1 Missing-Modality Robustness Stress Study | Heading 2 | 23 | 23 | **EXACT MATCH** |
| 7.2 Methodological Boundaries and Comprehensive Limitations | Heading 2 | 24 | 24 | **EXACT MATCH** |
| 7.3 Exploratory 3D / Foundation Model Pilot Studies | Heading 2 | 25 | 25 | **EXACT MATCH** |
| 7.4 Future Research Directions and Conclusion | Heading 2 | 25 | 25 | **EXACT MATCH** |
| References | Heading 1 | 27 | 27 | **EXACT MATCH** |
| Appendix A: Full Dataset Split Manifest and Census Audit | Heading 1 | 28 | 28 | **EXACT MATCH** |
| Appendix B: Reproducibility Record and Hyperparameter Specifications | Heading 1 | 28 | 28 | **EXACT MATCH** |
| Appendix C: Research Governance, Verification Checklist, and Integrity Declaration | Heading 1 | 29 | 29 | **EXACT MATCH** |

---

## 7. Git Repository State

- **Current HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904`
- **Active Branch:** `main`
- **Modified Tracked Files:**
  - `src/data/dataset.py`
  - `src/evaluation/metrics.py`
  - `src/training/train.py`
- *Note:* In accordance with repository governance, all report builder scripts (`tools/build_final_report.py`) and generated DOCX editions in `docs/reports/` remain untracked artifacts, preserving the immutable git tree.
