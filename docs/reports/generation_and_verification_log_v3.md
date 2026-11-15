# Generation and Verification Log — Research Edition V3

**Document Title:** Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent
**Edition:** Research Edition V3 (Final Scientific, Visual, and Document Consistency Corrections)
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Target Document Path:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx`
**Artwork Directory:** `tmp/report_artwork_research_v3/`
**Authoritative Builder Script:** `tools/build_final_report.py`
**Log Timestamp:** November 12, 2026

---

## 1. Executive Summary & Verification Outcomes

The Research Edition V3 generation represents the final, authoritative synthesis of all scientific findings, architectural baselines, post-hoc calibration metrics, deterministic agent triage accounting, and missing-modality stress analyses for the major project.

Every modification was implemented strictly within `tools/build_final_report.py`. All visual assets, tables, layout boundaries, and page numbers were verified through automated scripts, OpenXML package inspection, Word COM layout extraction, and PDF rendering.

### Summary of Key Compliance Metrics
| Item / Metric | Verified Value in V3 | Governance Target | Compliance Status |
|---|---|---|:---:|
| Total Document Pages | **39 pages** (1 cover, 9 prelims i–ix, 29 body pages 1–29) | Exactly 39 pages, 0 blank pages | **PASSED** |
| Total Body Chapters | 7 Chapters + References + 3 Appendices | Complete research portfolio | **PASSED** |
| Total Tables in DOCX | 21 tables (5 prelim/cover + 16 numbered report tables) | 21 tables | **PASSED** |
| Total Embedded Media Parts | 12 images (1 institutional logo + 11 research figures) | 12 parts | **PASSED** |
| `w:cantSplit` Table Row Coverage | 222 / 222 rows (100.0% coverage across all tables) | 100% coverage | **PASSED** |
| `w:tblHeader` Table Header Coverage | 20 / 20 content tables (100% of tables with headers) | 100% coverage | **PASSED** |
| Placeholder Tokens (`TODO`, `TBD`, etc.) | Exactly 0 found across entire document | Exactly 0 | **PASSED** |
| Table of Contents (TOC) Drift | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| List of Tables (LOT) Drift | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| List of Figures (LOF) Drift | Exactly 0 page mismatches (100% match with Word COM) | 0 drift | **PASSED** |
| Body Page 23 Layout | Chapter 7 start; isolated paragraph **ELIMINATED** | No isolated paragraph | **PASSED** |
| Appendix C Layout | Heading, intro, Table C.1, and caption together on Page 29 | Zero orphan headings | **PASSED** |
| Minor Project Baseline Preservation | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| Research Edition V1 Preservation | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| Research Edition V2 Preservation | Byte-for-byte identical (SHA-256 verified) | Immutable | **PASSED** |
| Sealed Test Cohort Quarantine | Untouched; zero access, opening, or evaluation | Sealed quarantine | **PASSED** |

---

## 2. File Integrity & Baseline Preservation

In strict accordance with the project governance invariants, the historical Minor Project report, Research Edition V1, Research Edition V2, all 18 files in `Report_Materials/`, and all 9 production charts in `tmp/report_figures/` remained read-only and untouched.

| Artifact / File Path | File Size | SHA-256 Digest | Status |
|---|---:|---|:---:|
| `Final_Report_Brain_Tumor_Segmentation.docx` (Minor Project Baseline) | 4,301,505 bytes | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` (V1 Baseline) | 3,112,658 bytes | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` (V2 Baseline) | 1,905,004 bytes | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx` (Target V3 Report) | 1,928,076 bytes | `2d85fc7cd92ae80371ec31584b36535a0d4772a7be6018e5a4d00b6f15ebb90b` | **GENERATED** |
| `tools/build_final_report.py` (Authoritative Builder Script) | 148,842 bytes | `06c8b939fa22a492984180860ae7e7ca4bc0d9e802ee61ec3fdb6b30689b940e` | **VERIFIED** |

### Test Cohort Quarantine Invariant
The sealed test cohort (`test_manifest.json` and associated test MRI volumes) remained completely sealed throughout the generation and verification process. Zero test files were accessed, opened, inspected, or evaluated.

---

## 3. Forensic Reconciliation of Word COM Lock-File Deletion

During the V2 build cycle, an ephemeral file `D:\MajorProject\~$nal_Report_Brain_Tumor_Segmentation.docx` was removed during workspace cleanup. A complete forensic audit was conducted to establish its origin, purpose, and impact:

1. **Origin and Nature:** The file was an owner lock file created automatically by Microsoft Word (`WINWORD.EXE`, PID 12212) when `Final_Report_Brain_Tumor_Segmentation.docx` was opened via Word COM automation for page-number extraction and PDF export.
2. **File Characteristics:** Microsoft Word generates hidden temporary files with the `~$` prefix in the same directory as the target document to store the current user's login identity and lock status. These files contain no document content.
3. **Protected Document Integrity:** The underlying protected file `Final_Report_Brain_Tumor_Segmentation.docx` remained completely untouched and byte-for-byte identical, confirmed by its exact SHA-256 digest `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5`.
4. **Resolution and Safety Protocol:** Word COM processes are now explicitly terminated via `$word.Quit()` followed by process cleanup. No protected documents were affected, and no further lock files exist in the repository.

---

## 4. Authoritative Analysis B2 Missing-Modality Robustness (Table 7.1)

Chapter 7 incorporates the authoritative Analysis B2 findings from `results/final_research_baseline/MISSING_MODALITY_POLICY_REEVALUATION.md`. This experiment was conducted across all 105 internal evaluation cases ($N = 6,549$ foreground slices) under five synthetic post-normalization zero-clamping conditions:

| Input Condition | Description | Mean Dice | WT Dice | TC Dice | ET Dice | Pooled Referral Rate | Subject-Avg Referral Rate | Statistical Significance (vs C0) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **C0** | Full 4-Channel Baseline (T1n, T1c, T2w, T2f) | **0.8641** | 0.9039 | 0.8597 | 0.8286 | **16.16%** (1,058 / 6,549) | **17.24%** | Baseline Reference |
| **C1** | Missing Native T1 (T1n zero-clamped) | **0.8517** | 0.8939 | 0.8494 | 0.8118 | **18.16%** (1,189 / 6,549) | **19.04%** | Non-significant ($p_{\text{adj}} = 1.0000$) |
| **C2** | Missing Contrast T1c (T1c zero-clamped) | **0.4449** | 0.8532 | 0.3132 | **0.1684** | **17.44%** (1,142 / 6,549) | **18.55%** | Non-significant ($p_{\text{adj}} = 1.0000$) |
| **C3** | Missing T2-weighted (T2w zero-clamped) | **0.6058** | 0.4388 | 0.6901 | 0.6885 | **62.65%** (4,103 / 6,549) | **63.34%** | Highly significant ($p_{\text{adj}} = 7.12 \times 10^{-17}$) |
| **C4** | Missing FLAIR (T2f zero-clamped) | **0.8394** | 0.8748 | 0.8342 | 0.8091 | **19.61%** (1,284 / 6,549) | **20.70%** | Non-significant ($p_{\text{adj}} = 1.0000$) |

### Crucial Scientific Findings & Methodological Disclosures
1. **The C2 Contrast-Blind Spot:** Under missing contrast-enhanced T1c (C2), Enhancing Tumor (ET) Dice collapses precipitously to 0.1684 and Tumor Core (TC) to 0.3132. However, Whole Tumor (WT) Dice remains high (0.8532) because edema is clearly visible on T2w and T2f. Because the model outputs non-empty, plausible-looking contours with moderate confidence, post-hoc predictive entropy fails to signal failure: the pooled referral rate shifts only from 16.16% to 17.44% ($p_{\text{adj}} = 1.0000$). This proves that **uncertainty alone cannot detect missing contrast data**, establishing the necessity of an upstream modality integrity gate.
2. **Analysis A vs Analysis B2 Contrast:** In historical Analysis A, a software defect caused empty predictions to be silently dropped, creating the misleading impression of high Dice under missing channels. Under authoritative Analysis B2, empty predictions are strictly penalized, and the empty-prediction guard intercepts 2,621 empty masks under C3, driving referral to 62.65%.
3. **Synthetic Protocol Bounding:** This experiment is explicitly disclosed as a synthetic post-normalization channel zero-clamping stress test, rather than a clinical multi-center missing-sequence trial.

---

## 5. Authoritative Reliability Agent Triage Action Accounting (Table 6.3)

Table 6.3 details the mutually exclusive action distribution logged slice-by-slice across all 105 internal evaluation cases ($N = 6,549$ foreground slices):

| Agent Triage Action Category | Slice Count | Percentage of Slices | Empirical Outcome / Action Rationale |
|---|---:|---:|---|
| **Autonomous Retain (Direct ResUNet-2D Accept)** | 5,437 slices | 83.02% | Calibrated foreground entropy $H \le 0.35$ with non-empty foreground; accepted directly from primary ResUNet-2D. |
| **Autonomous Retain (Fallback Switch to UNetResNet34)** | 54 slices | 0.82% | Primary ResUNet-2D predicted zero foreground voxels on slices supported by neighboring axial context ($k=2$); candidate substituted autonomously. |
| **Simulated Referral (Flagged for Review)** | 1,058 slices | 16.16% | Calibrated foreground entropy $H > 0.35$ or unresolvable empty-primary prediction; autonomous output withheld and flagged for review. |
| **Total Evaluated Foreground Cohort** | **6,549 slices** | **100.00%** | Complete census of foreground slices across all 105 evaluation cases ($5,437 + 54 + 1,058 = 6,549$ slices). |

### Key Clarifications on Triage Accounting
- **Mutual Exclusivity:** Slices receiving a fallback switch are autonomously retained without review (`is_human_review_recommended = False`). The 54 fallback switches are a subset of the **5,491 total retained slices** ($5,437 + 54 = 5,491$, 83.84%).
- **Denominator Census:** The three categories sum exactly to 6,549 slices ($5,437 + 54 + 1,058 = 6,549$, 100.00%).
- **Pipeline Overlap Utility:** Aggregate pipeline Mean Dice across all 6,549 slices is 0.8640, compared to 0.8641 for standalone ResUNet-2D. The utility of the agent is **not raw overlap expansion**, but rather risk containment and selective prediction that safely segregates reliable segmentations from uncertain cases.

---

## 6. Figure 7.1 Architecture Diagram Verification

The visual generator for Figure 7.1 (`tmp/report_artwork_research_v3/agent_architecture.png`) was completely overhauled to align with the audited codebase:

1. **Repaired Gate-Failure Connector:** The red curved connector originating from the Modality Integrity Gate failure branch now routes cleanly and directly into the Flag/Refer Action Box, eliminating the visual gap present in earlier drafts.
2. **Authoritative Primary Baseline Metric:** Replaced the unverified historical claim ("0.884 WT") with the documented validation Mean Dice of **0.8670 at Epoch 13** achieved by the primary ResUNet-2D baseline.
3. **Canonical Input Labeling:** Replaced the phrase "4 coils" with the correct clinical and technical terminology: **4 channels** with canonical BraTS sequence labels: **T1n, T1c, T2w, and T2f**.
4. **Action Box Clarifications:** Replaced ambiguous language ("switch or combine") with the actual deterministic policy:
   - **Accept Box:** Autonomous retain when $H \le 0.35$ and foreground is non-empty.
   - **Fallback Switch Box:** Autonomous substitution of candidate UNetResNet34 when primary predicts empty foreground supported by neighboring axial context ($k=2$).
   - **Flag / Refer Box:** Simulated referral when calibrated entropy exceeds threshold ($H > 0.35$) or an unresolvable empty-primary failure occurs.

---

## 7. Clinical Language, Scope, and Citation Governance

All sections of the report were systematically audited to ensure defensible academic boundaries and remove unverified clinical claims:

1. **Academic Prototype Status:** The system is explicitly described throughout the Abstract, Chapter 1, Chapter 6, Chapter 7, and Conclusion as an **academic research prototype**. Claims of clinical clearance, prospective safety, or actual neuroradiologist interaction have been removed.
2. **Citation Accuracy (Section 1.1):** Removed the unverified claims that manual segmentation takes "45–90 minutes per scan" and achieves "0.75–0.85 Dice". In their place, the foundational BraTS benchmark by Menze et al. [1] is cited accurately regarding manual annotation burden and subregion inter-observer variability.
3. **Statistical Dependence Disclosure (Section 6.1):** Added an explicit methodological limitation disclosing that the Spearman rank correlations reported between uncertainty and segmentation quality ($N = 3,223$ slices) treat pooled slices as independent observations, inflating effective degrees of freedom due to intra-subject anatomical spatial autocorrelation.
4. **Bounded Scope:** All empirical findings are bounded to the 39-case pilot validation cohort and the 105-case internal evaluation pool.

---

## 8. Appendix B Reproducibility Checklist & Hyperparameter Specifications

Appendix B (Table B.1) details the exact runtime environment and training specifications:

| Parameter / Component | Recorded Specification |
|---|---|
| **Software Repository** | `github.com/NK-NISHANT/major-project-brain-tumor-segmentation` (audited research branch) |
| **Python & DL Framework** | Python 3.11.9 (`.venv`); PyTorch 2.11.0+cu128; MONAI 1.6.0 |
| **Hardware Compute Platform** | NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM); 13th Gen Intel(R) Core(TM) i9-13900HX CPU |
| **Training Execution Precision** | Full single precision (FP32, `torch.float32`); no automatic mixed precision (AMP) |
| **Stage A Pilot Hyperparameters** | 5 epochs; batch size 2; AdamW optimizer ($lr = 1 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$); seed = 42 |
| **Stage B Baseline Hyperparameters** | 15 epochs; batch size 16; AdamW optimizer ($lr = 1 \times 10^{-3}$, cosine annealing); gradient clip norm = 1.0; seed = 42 |
| **Compound Loss Formulation** | $\mathcal{L}_{\text{total}} = 1.0 \times \mathcal{L}_{\text{Dice}} + 1.0 \times \mathcal{L}_{\text{CE}}$ (Dice $\epsilon = 1 \times 10^{-5}$) |
| **Primary Baseline Checkpoint** | `results/full_training_15ep/resunet_2d/best_model.pt` (Epoch 13, Val Mean Dice 0.8670) |
| **Reference Baseline Checkpoint** | `results/full_training_15ep/unet_resnet34/best_model.pt` (Epoch 15, Val Mean Dice 0.8554) |
| **Post-Hoc Temperatures ($T^*$)** | ResUNet-2D: $T^* = 2.880783$; UNetResNet34: $T^* = 2.626696$ (fitted on 50-case calibration pool) |
| **Reliability Agent Operating Point** | Selective prediction threshold $\tau = 0.35$, empty-prediction guard enabled, axial context window $k = 2$ |

---

## 9. Layout Optimization & Pagination Alignment Map

### Elimination of Isolated Paragraph on Body Page 23
In earlier builds, Table 6.3 was placed near the bottom of Body Page 22, causing its caption and the Aggregate Segmentation Overlap Note to spill onto Body Page 23, leaving Body Page 23 as an isolated 8-line page before Chapter 7.

This layout defect was permanently resolved:
1. Compacted Figure 6.6 width to 2.4 inches and tightened calibration discussion on Page 21.
2. Section 6.3 now completely finishes at the bottom of Page 21.
3. Section 6.4 begins at the top of Page 22 and contains the entire agent description, 5 decision stages, Canonical B2 results, Table 6.3, Table 6.3 caption, and Aggregate note on Page 22.
4. Chapter 7 starts cleanly at the top of Page 23. Body Page 23 is now a full, rich chapter-opening page.
5. In Appendix C, the heading, intro, Table C.1, and caption sit together on Page 29.

### Full Document Page Alignment Map (0 Page Drift)
| Section / Item Title | Recorded Page in V3 Table | Word COM Rendered Page | Alignment Status |
|---|:---:|:---:|:---:|
| **Preliminary Pages** | | | |
| Acknowledgement | i | i (Page 1) | **EXACT MATCH** |
| Declaration | ii | ii (Page 2) | **EXACT MATCH** |
| Certificate | iii | iii (Page 3) | **EXACT MATCH** |
| Abstract | iv | iv (Page 4) | **EXACT MATCH** |
| Table of Contents | vi | vi (Page 6) | **EXACT MATCH** |
| List of Tables | vii | vii (Page 7) | **EXACT MATCH** |
| List of Figures | viii | viii (Page 8) | **EXACT MATCH** |
| List of Abbreviations | ix | ix (Page 9) | **EXACT MATCH** |
| **Chapter 1: INTRODUCTION** | **1** | **1** | **EXACT MATCH** |
| 1.1 Clinical and Technical Context | 1 | 1 | **EXACT MATCH** |
| 1.2 Limitations of Segmentation Dice as a Sole Reliability Measure | 1 | 1 | **EXACT MATCH** |
| 1.3 Predictive Uncertainty and Reliability-Aware Decision-Making | 2 | 2 | **EXACT MATCH** |
| 1.4 Research Motivation, Objectives, and Scope | 2 | 2 | **EXACT MATCH** |
| 1.5 Summary of Project Contributions | 3 | 3 | **EXACT MATCH** |
| 1.6 Report Organization | 4 | 4 | **EXACT MATCH** |
| **Chapter 2: BACKGROUND STUDY AND RELATED WORK** | **5** | **5** | **EXACT MATCH** |
| 2.1 Multi-modal MRI and Glioma Subregions | 5 | 5 | **EXACT MATCH** |
| 2.2 Deep Learning Architectures for Medical Image Segmentation | 5 | 5 | **EXACT MATCH** |
| 2.3 Evaluation, Overlap, and Distance Metrics | 6 | 6 | **EXACT MATCH** |
| Table 2.1 | 6 | 6 | **EXACT MATCH** |
| 2.4 Predictive Uncertainty Quantification in Medical Imaging | 6 | 6 | **EXACT MATCH** |
| 2.5 Probability Calibration in Neural Networks | 7 | 7 | **EXACT MATCH** |
| 2.6 Selective Prediction, Triage, and Reliability-Aware AI | 7 | 7 | **EXACT MATCH** |
| **Chapter 3: DATASET CENSUS AND PARTITIONING METHODOLOGY** | **8** | **8** | **EXACT MATCH** |
| 3.1 BraTS 2023 GLI Dataset Overview | 8 | 8 | **EXACT MATCH** |
| 3.2 Subject-Level Dataset Census and Split Accounting | 8 | 8 | **EXACT MATCH** |
| Table 3.1 | 9 | 9 | **EXACT MATCH** |
| 3.3 Historical 197-Case Pilot Split vs Mature Cohort | 9 | 9 | **EXACT MATCH** |
| Table 3.2 | 9 | 9 | **EXACT MATCH** |
| 3.4 Data Preprocessing and Tensor Normalization | 9 | 9 | **EXACT MATCH** |
| **Chapter 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE** | **11** | **11** | **EXACT MATCH** |
| 4.1 Five-Model Architecture Portfolio | 11 | 11 | **EXACT MATCH** |
| Table 4.1 | 11 | 11 | **EXACT MATCH** |
| Figure 4.1 | 11 | 11 | **EXACT MATCH** |
| Figure 4.2 | 12 | 12 | **EXACT MATCH** |
| Figure 4.3 | 12 | 12 | **EXACT MATCH** |
| 4.2 Structural Details of Stage A Architectures | 12 | 12 | **EXACT MATCH** |
| 4.3 Structural Details of Stage B Architectures | 13 | 13 | **EXACT MATCH** |
| 4.4 Training Dynamics and Loss Optimization Framework | 13 | 13 | **EXACT MATCH** |
| 4.5 Software Architecture and Repository Design | 14 | 14 | **EXACT MATCH** |
| Table 4.2 | 14 | 14 | **EXACT MATCH** |
| **Chapter 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS** | **15** | **15** | **EXACT MATCH** |
| 5.1 Two-Stage Experimental Progression | 15 | 15 | **EXACT MATCH** |
| 5.2 Stage A: Historical Five-Epoch Pilot Benchmark | 15 | 15 | **EXACT MATCH** |
| Table 5.1 | 15 | 15 | **EXACT MATCH** |
| Table 5.2 | 15 | 15 | **EXACT MATCH** |
| Figure 6.1 | 16 | 16 | **EXACT MATCH** |
| Figure 6.2 | 16 | 16 | **EXACT MATCH** |
| Figure 6.3 | 17 | 17 | **EXACT MATCH** |
| Figure 6.4 | 17 | 17 | **EXACT MATCH** |
| 5.3 Stage B: Mature 15-Epoch Research Baselines | 17 | 17 | **EXACT MATCH** |
| Table 5.3 | 18 | 18 | **EXACT MATCH** |
| 5.4 Checkpoint-Selection Bias and Methodological Boundaries | 18 | 18 | **EXACT MATCH** |
| **Chapter 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE** | **19** | **19** | **EXACT MATCH** |
| 6.1 Predictive Uncertainty and Segmentation Quality Correlation | 19 | 19 | **EXACT MATCH** |
| Table 6.1 | 19 | 19 | **EXACT MATCH** |
| 6.2 Separation of Uncertainty Binning and Slice Distribution | 20 | 20 | **EXACT MATCH** |
| Figure 6.5A | 20 | 20 | **EXACT MATCH** |
| Figure 6.5B | 20 | 20 | **EXACT MATCH** |
| Figure 6.6 | 21 | 21 | **EXACT MATCH** |
| 6.3 Post-Hoc Probability Calibration and Temperature Scaling | 21 | 21 | **EXACT MATCH** |
| Table 6.2 | 21 | 21 | **EXACT MATCH** |
| 6.4 Implemented Deterministic Reliability-Aware Agent | 22 | 22 | **EXACT MATCH** |
| Table 6.3 | 22 | 22 | **EXACT MATCH** |
| **Chapter 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS** | **23** | **23** | **EXACT MATCH** |
| 7.1 Missing-Modality Robustness Stress Study | 23 | 23 | **EXACT MATCH** |
| Table 7.1 | 23 | 23 | **EXACT MATCH** |
| 7.2 Methodological Boundaries and Comprehensive Limitations | 24 | 24 | **EXACT MATCH** |
| Table 7.2 | 25 | 25 | **EXACT MATCH** |
| 7.3 Exploratory 3D / Foundation Model Pilot Studies | 25 | 25 | **EXACT MATCH** |
| 7.4 Future Research Directions and Conclusion | 25 | 25 | **EXACT MATCH** |
| Figure 7.1 | 26 | 26 | **EXACT MATCH** |
| **REFERENCES** | **27** | **27** | **EXACT MATCH** |
| **APPENDIX A: FULL DATASET SPLIT MANIFEST AND CENSUS AUDIT** | **28** | **28** | **EXACT MATCH** |
| Table A.1 | 28 | 28 | **EXACT MATCH** |
| **APPENDIX B: REPRODUCIBILITY RECORD AND HYPERPARAMETER SPECIFICATIONS** | **28** | **28** | **EXACT MATCH** |
| Table B.1 | 28 | 28 | **EXACT MATCH** |
| **APPENDIX C: RESEARCH GOVERNANCE, VERIFICATION CHECKLIST, AND INTEGRITY DECLARATION** | **29** | **29** | **EXACT MATCH** |
| Table C.1 | 29 | 29 | **EXACT MATCH** |

---

## 10. Summary of Unresolved Methodological Limitations

In strict compliance with academic transparency and the scientific accuracy audit, the following empirical limitations are formally noted:

1. **Sealed Test Cohort Quarantine:** The 155-case sealed test cohort was intentionally preserved under strict read-only quarantine. All reported evaluation metrics represent internal validation rather than blinded prospective testing.
2. **Checkpoint-Selection Bias:** The 105-case internal evaluation pool was monitored alongside the 50-case calibration subset during epoch selection. Consequently, reported metrics on this cohort reflect internal validation performance.
3. **Synthetic Missing-Modality Protocol:** The missing-modality experiment represents a controlled post-normalization channel zero-clamping stress test rather than real-world missing-sequence clinical acquisitions.
4. **2D Axial Formulation:** All models operate on 2D axial slices, lacking full 3D volumetric spatial context along the coronal and sagittal planes.
5. **C2 Contrast Blind Spot:** Under missing T1c MRI, enhancing tumor Dice collapses to 0.1684 with only a modest shift in predictive entropy (16.16% to 17.44%, $p_{\text{adj}} = 1.0000$). This blind spot requires an upstream modality integrity gate for safe operation.
6. **Slice Autocorrelation in Correlation Analysis:** The Spearman rank correlations ($N = 3,223$ slices) treat slices from the same subject as independent observations, inflating effective degrees of freedom.

---

## 11. Final Verification Sign-Off

Research Edition V3 has achieved 100% compliance across all architectural, visual, layout, and scientific criteria. The DOCX package is fully validated, completely free of placeholder tokens, and ready for official academic submission.
