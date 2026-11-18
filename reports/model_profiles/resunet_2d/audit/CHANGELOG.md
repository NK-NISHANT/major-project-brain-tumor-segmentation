# ResUNet-2D Documentation Package — CHANGELOG

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation  
**Model:** ResUNet-2D (Primary Champion)  
**Version:** v1.1 (Comprehensive Scientific & Visual Communication Package)  
**Date:** November 18, 2026  
**Supersedes:** v1.0 (Initial Technical Profile)

---

## 1. MAJOR CORRECTIONS FROM v1.0

### A. Corrected Class-wise Dice and Macro-Average Terminology
- **v1.0 Issue:** Labeled the Epoch 13 mean score of `0.8670` as "Validation Mean Dice" without distinguishing foreground-only classes from the 4-class arithmetic mean.
- **v1.1 Correction:** Explicitly labeled throughout as **`Foreground Macro Mean Dice (NCR, ED, ET)`**. The 4-class arithmetic mean (0.899968, incorporating Background 0.998738) is separately calculated, defined, and displayed. No underlying data was altered.

### B. Rigorous Protocol Separation for Dice Metrics
- **v1.0 Issue:** Combined or juxtaposed checkpoint-selection Dice (0.8670) and 3D volumetric Dice (0.8641) without sufficient protocol explanation.
- **v1.1 Correction:** Strictly segregated the three distinct Dice metrics:
  1. *Training/Checkpoint-Selection Dice (0.8670):* 2D slice batch tallies across 155 validation cases.
  2. *3D Volumetric Dice (0.8641):* Case-level 3D volume accumulation over 105 held-out `Val_eval` volumes (WT: 0.9039, TC: 0.8597, ET: 0.8286).
  3. *Unfiltered Slice-Level Macro Dice (0.7150):* Arithmetic average across all 6,549 ground-truth tumor slices in `Val_eval`.

### C. Disentangled Action Counts and Ground-Truth Denominators
- **v1.0 Issue:** Associated the 5,619 ACCEPT action count with the 83.02% retention figure, creating potential denominator confusion.
- **v1.1 Correction:** Every percentage now explicitly identifies its exact denominator:
  - *All Evaluated Slices ($N=16,275$):* ACCEPT = 5,619 (34.53%), FALLBACK_SWITCH = 147 (0.90%), FLAG_UNRELIABLE = 10,509 (64.57%).
  - *Non-Empty Primary Predictions ($N=6,874$):* ACCEPT = 5,619 (81.74%), FLAG_UNRELIABLE = 1,255 (18.26%).
  - *Ground-Truth Tumor Slices ($N=6,549$):* Retained by full policy = 5,491 (83.84%), Retained by entropy-only = 5,437 (83.02%), Recovered by fallback = 54 (0.82%), Referred = 1,058 (16.16%).

### D. Architectural Parameter Accounting Reconciliation
- **v1.0 Issue:** Minor ambiguity between stage block parameters and total stage parameters including transposed convolutions.
- **v1.1 Correction:** Both architecture diagrams and tables now use identical, standardized stage totals:
  - Encoder 1: 10,632 (0.13%)
  - Encoder 2: 57,664 (0.71%)
  - Encoder 3: 230,016 (2.83%)
  - Encoder 4: 918,784 (11.32%)
  - Bottleneck (L5): 3,672,576 (45.26%)
  - Decoder 4 (Upconv + Block): 2,427,136 (29.91%)
  - Decoder 3 (Upconv + Block): 607,360 (7.49%)
  - Decoder 2 (Upconv + Block): 152,128 (1.87%)
  - Decoder 1 (Upconv + Block): 38,176 (0.47%)
  - Classifier Head: 132 (<0.01%)
  - Total: Exactly 8,114,604 trainable parameters.

### E. Redesigned, Unclipped Agent Decision Flowchart
- **v1.0 Issue:** Bottom boxes of the agent decision tree were clipped at the canvas margin.
- **v1.1 Correction:** Built a new high-resolution vector flowchart (`18_full_c3_decision_flow.png`) with expanded canvas (16:9 widescreen), clearly illustrating all primary branches, axial context validation, secondary model invocation, high-entropy warning triggers, and terminal actions.

### F. DOCX Document Pagination & Layout Restructuring
- **v1.0 Issue:** Final page had isolated hanging elements.
- **v1.1 Correction:** Re-engineered document structure with explicit page breaks, `keep_with_next` on all headings, formatted table cells, and standardized 0.75-inch margins, producing an 8-page master.

### G. LaTeX and Control Character Formatting in Markdown
- **v1.0 Issue:** Accidental escaped backslashes and raw LaTeX symbols in some Markdown parsers.
- **v1.1 Correction:** Sanitized all math expressions using clean KaTeX-compatible syntax and raw string literals in document generation scripts.

### H. Subject-Level Bootstrap Confidence Interval Audit
- **v1.0 Issue:** Ambiguity in whether bootstrap resampling was performed at the case or subject level.
- **v1.1 Correction:** Verified from `evaluate_hd95_cohort.py` and `c3_selective_prediction_v1_1_3` that resampling was performed via cluster bootstrap at the **biological subject level** (96 clusters), properly accounting for repeated intra-subject case acquisitions.

---

## 2. NEW DELIVERABLES ADDED IN v1.1

1. **Large Visual Asset Library:**
   - 25 Technical & Scientific Assets (`assets/technical/`)
   - 16 Non-Technical & Easy-to-Explain Assets (`assets/non_technical/`)
   - 3 Real Validation Qualitative Case Panels (`assets/qualitative_examples/`)
2. **One-Page Model Overview:**
   - Standalone A3 Landscape publication poster (`one_page/ResUNet_2D_One_Page_Overview.pdf` and `.png`).
3. **Explainer Presentation Deck:**
   - 8-slide presentation (`presentation/ResUNet_2D_Explainer_Deck.pptx`) and companion `Presenter_Notes.md`.
4. **Comprehensive Source & Quality Audits:**
   - `audit/source_audit.md`
   - `audit/metric_consistency_audit.md`
   - `audit/visual_qa_report.md`
