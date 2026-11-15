# Generation and Verification Log — Research Edition V2

**Document Title:** Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent
**Edition:** Research Edition V2 (Architectural Baselines, Calibration, and Reliability Analysis)
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** Reconciled `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Target Document Path:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx`
**Artwork Directory:** `tmp/report_artwork_research_v2/`
**Build Script:** `tools/build_final_report.py`
**Log Timestamp:** November 12, 2026

---

## 1. Executive Summary & Verification Outcomes

The Research Edition V2 report generation was executed using the refactored, authoritative report builder `tools/build_final_report.py`. All visual, layout, pagination, and scientific integrity requirements were verified through automated inspection, Word COM layout extraction, and PyMuPDF document rendering.

### Summary of Key Metrics
| Item / Metric | Verified Value | Compliance Status |
|---|---|:---:|
| Total Document Pages | **39 pages** (1 cover, 9 prelims i–ix, 29 body pages 1–29) | **PASSED** |
| Total Body Chapters | 7 Chapters + References + 3 Appendices | **PASSED** |
| Total Tables in DOCX | 21 tables (5 prelim/cover + 16 numbered report tables) | **PASSED** |
| Total Embedded Media Parts | 12 images (1 institutional logo + 11 research figures) | **PASSED** |
| `w:cantSplit` Table Row Coverage | 221 / 221 rows (100% coverage, 0 missing rows) | **PASSED** |
| `w:tblHeader` Table Header Coverage | 20 / 20 content tables (100% of tables with headers) | **PASSED** |
| Placeholder Tokens (`TODO`, `TBD`, etc.) | Exactly 0 found across entire document | **PASSED** |
| Table of Contents (TOC) Page Drift | Exactly 0 page mismatches (100% match with Word COM) | **PASSED** |
| List of Tables (LOT) Page Drift | Exactly 0 page mismatches (100% match with Word COM) | **PASSED** |
| List of Figures (LOF) Page Drift | Exactly 0 page mismatches (100% match with Word COM) | **PASSED** |
| Protected Original Report Preservation | Byte-for-byte identical (SHA-256 verified) | **PASSED** |
| Protected Research Edition V1 Preservation | Byte-for-byte identical (SHA-256 verified) | **PASSED** |
| Sealed Test Cohort Quarantine | Untouched; zero access, opening, or evaluation | **PASSED** |

---

## 2. File Integrity & Baseline Preservation

The generation strictly adhered to baseline preservation invariants. The historical Minor Project report, the Research Edition V1 report, all 18 files in `Report_Materials/`, and all 9 production charts in `tmp/report_figures/` remained untouched and read-only.

| Artifact / File Path | File Size | SHA-256 Digest | Status |
|---|---:|---|:---:|
| `Final_Report_Brain_Tumor_Segmentation.docx` (Minor Project Baseline) | 4,301,505 bytes | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx` (V1 Baseline) | 3,112,658 bytes | `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3` | **PRESERVED** |
| `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` (Target V2 Report) | 1,905,004 bytes | `4c0da7044488cd7826ddd904f1d487fed9644c416271467d1d76bae0f45332db` | **GENERATED** |
| `tools/build_final_report.py` (Authoritative Builder Script) | 149,576 bytes | `5b170191f5036bd6c62d94cbea5aaaa3d6b0ed624dacca6f08223afd963da5c3` | **VERIFIED** |

### Test Cohort Quarantine Invariant
The sealed test cohort (`test_manifest.json` and associated test MRI volumes) remained completely sealed throughout the generation and verification process. Zero test files were accessed, opened, inspected, or evaluated.

---

## 3. Parameter Accounting Reconciliation (Table 4.1)

Table 4.1 of the report explicitly presents the authoritative parameter counts derived from runtime model instantiation and state-dictionary inspections, distinguishing between trainable weights and non-trainable batch normalization buffers:

| Model Architecture | Encoder Backbone | Trainable Parameters | Non-Trainable Buffers | Total Tensor Elements | Research Role |
|---|---|---:|---:|---:|---|
| **U-Net** | 4-level DoubleConv | 7,763,428 | 5,906 | 7,769,334 | Historical pilot (Stage A) |
| **U-Net++** | Nested dense skips | 9,160,068 | 7,326 | 9,167,394 | Historical pilot (Stage A) |
| **Attention U-Net** | Additive attention | 7,852,160 | 6,886 | 7,859,046 | Historical pilot (Stage A) |
| **ResUNet-2D** | Pre-activation residual | 8,114,604 | 0 | 8,114,604 | Primary Baseline (Stage B) |
| **UNetResNet34** | Pretrained ResNet-34 | 24,327,524 | 18,988 | 24,346,512 | Reference Baseline (Stage B) |

### Contextual Distinctions
- **ResUNet-2D (8,114,604 parameters):** Uses Group Normalization (`num_groups=8` or `num_groups=16`), which stores scale ($\gamma$) and shift ($\beta$) as trainable parameters and maintains zero running mean/variance buffers (0 non-trainable buffers).
- **UNetResNet34 (24,346,512 tensor elements):** Utilizes `torchvision.models.resnet34` with standard Batch Normalization layers in the encoder and decoder, contributing 18,988 non-trainable running mean and variance buffer elements alongside 24,327,524 trainable weights.
- **Stage A Models (U-Net, U-Net++, Attention U-Net):** Also employ standard Batch Normalization layers, containing between 5,906 and 7,326 buffer elements.

---

## 4. Visual Assets & Artwork Verification

Six fresh artwork assets were generated in `tmp/report_artwork_research_v2/` and embedded into the DOCX OpenXML package:

| Artwork File Name | Dimensions | File Size | Description & Layout Integrity Verification |
|---|:---:|---:|---|
| `agent_architecture.png` (Figure 7.1) | 3300 × 1860 | 552,209 bytes | Standalone high-resolution architecture diagram. Correctly depicts 4-channel input, Modality Integrity Gate, ResUNet-2D primary baseline, temperature calibration ($T^* = 2.88$), empty prediction guard, and three-way triage ($\le 0.35$, $0.35 < H \le 0.40$, $> 0.40$). Future conceptual LLM/PACS modules are clearly demarcated in a dashed grey container. |
| `entropy_quintile_quality.png` (Figure 6.5B) | 2550 × 1350 | 236,170 bytes | Corrected dual-axis plot. Secondary axis $y$-limit expanded to `(0, 0.40)` with legend centered at top (`loc="upper center", ncol=3`). Zero text collision with Q5 entropy label (0.2875). |
| `entropy_threshold_distribution.png` (Figure 6.5A) | 2400 × 1260 | 132,714 bytes | Histogram showing slice distribution across rounded entropy thresholds for all 3,223 pilot validation slices. |
| `model_portfolio.png` (Figure 4.2) | 2700 × 1350 | 134,765 bytes | High-resolution portfolio diagram illustrating the five 2D architectures and parameter complexity. |
| `mri_modalities.png` (Figure 4.1) | 1020 × 295 | 360,108 bytes | Multi-parametric MRI modalities (FLAIR, T1, T1c, T2) and corresponding glioma subregions (WT, TC, ET). |
| `jiit_logo.png` | 236 × 275 | 83,687 bytes | Official institutional insignia rendered on preliminary cover page. |

### Complete DOCX Embedded Media Parts (12 parts total)
1. `word/media/image1.png`: Institutional logo (`jiit_logo.png`)
2. `word/media/image2.png`: Figure 4.1 (`mri_modalities.png`)
3. `word/media/image3.png`: Figure 4.2 (`model_portfolio.png`)
4. `word/media/image4.png`: Figure 4.3 (Sample qualitative segmentation output)
5. `word/media/image5.png`: Figure 6.1 (Pilot validation Dice and IoU comparison)
6. `word/media/image6.png`: Figure 6.2 (Class-wise Dice comparison across pilot models)
7. `word/media/image7.png`: Figure 6.3 (Training and validation loss progression)
8. `word/media/image8.png`: Figure 6.4 (Validation mean foreground Dice trend)
9. `word/media/image9.png`: Figure 6.5A (`entropy_threshold_distribution.png`)
10. `word/media/image10.png`: Figure 6.5B (`entropy_quintile_quality.png`)
11. `word/media/image11.png`: Figure 6.6 (Complete foreground miss rate vs entropy)
12. `word/media/image12.png`: Figure 7.1 (`agent_architecture.png`)

---

## 5. Layout & Document Pagination Mapping

Pagination was extracted directly from Microsoft Word COM (`wdActiveEndAdjustedPageNumber`) and validated against PDF rendering (`tmp/v2_word.pdf`). All entries in the Table of Contents, List of Tables, and List of Figures match the actual layout with 100% precision.

### Front Matter Pagination (Roman Numerals)
| Front Matter Section | Printed Page | Listed in TOC | Alignment Status |
|---|:---:|:---:|:---:|
| Cover Page | (Uncounted) | — | Aligned |
| Acknowledgement | i | i | **MATCH** |
| Declaration | ii | ii | **MATCH** |
| Certificate | iii | iii | **MATCH** |
| Abstract | iv | iv | **MATCH** |
| Table of Contents | vi | vi | **MATCH** |
| List of Tables | vii | vii | **MATCH** |
| List of Figures | viii | viii | **MATCH** |
| List of Abbreviations | ix | ix | **MATCH** |

### Body Chapters & Headings Pagination (Arabic Numerals)
| Chapter / Section Title | Listed Page | Word COM Actual | PyMuPDF Actual | Alignment Status |
|---|:---:|:---:|:---:|:---:|
| **Chapter 1: INTRODUCTION** | **1** | **1** | **1** | **MATCH** |
| 1.1 Clinical and Technical Context | 1 | 1 | 1 | **MATCH** |
| 1.2 Limitations of Segmentation Dice as a Sole Reliability Measure | 1 | 1 | 1 | **MATCH** |
| 1.3 Predictive Uncertainty and Reliability-Aware Decision-Making | 2 | 2 | 2 | **MATCH** |
| 1.4 Research Motivation, Objectives, and Scope | 2 | 2 | 2 | **MATCH** |
| 1.5 Summary of Project Contributions | 3 | 3 | 3 | **MATCH** |
| 1.6 Report Organization | 4 | 4 | 4 | **MATCH** |
| **Chapter 2: BACKGROUND STUDY AND RELATED WORK** | **5** | **5** | **5** | **MATCH** |
| 2.1 Multi-modal MRI and Glioma Subregions | 5 | 5 | 5 | **MATCH** |
| 2.2 Deep Learning Architectures for Medical Image Segmentation | 5 | 5 | 5 | **MATCH** |
| 2.3 Evaluation, Overlap, and Distance Metrics | 6 | 6 | 6 | **MATCH** |
| 2.4 Predictive Uncertainty Quantification in Medical Imaging | 6 | 6 | 6 | **MATCH** |
| 2.5 Probability Calibration in Neural Networks | 7 | 7 | 7 | **MATCH** |
| 2.6 Selective Prediction, Triage, and Reliability-Aware AI | 7 | 7 | 7 | **MATCH** |
| **Chapter 3: DATASET CENSUS AND PARTITIONING METHODOLOGY** | **8** | **8** | **8** | **MATCH** |
| 3.1 BraTS 2023 GLI Dataset Overview | 8 | 8 | 8 | **MATCH** |
| 3.2 Subject-Level Dataset Census and Split Accounting | 8 | 8 | 8 | **MATCH** |
| 3.3 Historical 197-Case Pilot Split vs Mature Cohort | 9 | 9 | 9 | **MATCH** |
| 3.4 Data Preprocessing and Tensor Normalization | 9 | 9 | 9 | **MATCH** |
| **Chapter 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE** | **11** | **11** | **11** | **MATCH** |
| 4.1 Five-Model Architecture Portfolio | 11 | 11 | 11 | **MATCH** |
| 4.2 Structural Details of Stage A Architectures | 12 | 12 | 12 | **MATCH** |
| 4.3 Structural Details of Stage B Architectures | 13 | 13 | 13 | **MATCH** |
| 4.4 Training Dynamics and Loss Optimization Framework | 13 | 13 | 13 | **MATCH** |
| 4.5 Software Architecture and Repository Design | 14 | 14 | 14 | **MATCH** |
| **Chapter 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS** | **15** | **15** | **15** | **MATCH** |
| 5.1 Two-Stage Experimental Progression | 15 | 15 | 15 | **MATCH** |
| 5.2 Stage A: Historical Five-Epoch Pilot Benchmark | 15 | 15 | 15 | **MATCH** |
| 5.3 Stage B: Mature 15-Epoch Research Baselines | 17 | 17 | 17 | **MATCH** |
| 5.4 Checkpoint-Selection Bias and Methodological Boundaries | 18 | 18 | 18 | **MATCH** |
| **Chapter 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE** | **19** | **19** | **19** | **MATCH** |
| 6.1 Predictive Uncertainty and Segmentation Quality Correlation | 19 | 19 | 19 | **MATCH** |
| 6.2 Separation of Uncertainty Binning and Slice Distribution | 19 | 19 | 19 | **MATCH** |
| 6.3 Post-Hoc Probability Calibration and Temperature Scaling | 21 | 21 | 21 | **MATCH** |
| 6.4 Implemented Deterministic Reliability-Aware Agent | 22 | 22 | 22 | **MATCH** |
| **Chapter 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS** | **24** | **24** | **24** | **MATCH** |
| 7.1 Missing-Modality Robustness Stress Study | 24 | 24 | 24 | **MATCH** |
| 7.2 Methodological Boundaries and Comprehensive Limitations | 24 | 24 | 24 | **MATCH** |
| 7.3 Exploratory 3D / Foundation Model Pilot Studies | 25 | 25 | 25 | **MATCH** |
| 7.4 Future Research Directions and Conclusion | 26 | 26 | 26 | **MATCH** |
| **References** | **27** | **27** | **27** | **MATCH** |
| **Appendix A: Full Dataset Split Manifest and Census Audit** | **28** | **28** | **28** | **MATCH** |
| **Appendix B: Reproducibility Record and Hyperparameter Specifications** | **28** | **28** | **28** | **MATCH** |
| **Appendix C: Research Governance, Verification Checklist, and Integrity Declaration** | **28** | **28** | **28** | **MATCH** |

### List of Tables (LOT) Alignment Verification
| Table Number | Table Title | Listed Page | Actual Page | Alignment Status |
|---|---|:---:|:---:|:---:|
| Table 2.1 | Evaluation, overlap, distance, and uncertainty measures used in this report | 6 | 6 | **MATCH** |
| Table 3.1 | Full subject-level dataset census and partition allocation for BraTS 2023 GLI | 9 | 9 | **MATCH** |
| Table 3.2 | Historical 197-case pilot development split vs mature 1,251-case cohort partitioning | 9 | 9 | **MATCH** |
| Table 4.1 | Five-model 2D segmentation architecture portfolio and verified runtime parameter counts | 11 | 11 | **MATCH** |
| Table 4.2 | Core implementation modules and functional responsibilities across the codebase | 14 | 14 | **MATCH** |
| Table 5.1 | Stage A historical five-epoch pilot benchmark on the 39-case validation cohort | 15 | 15 | **MATCH** |
| Table 5.2 | Stage A foreground class-wise Dice at model-specific best validation epochs | 15 | 15 | **MATCH** |
| Table 5.3 | Stage B mature 15-epoch baseline comparison on the 105-case internal evaluation pool | 18 | 18 | **MATCH** |
| Table 6.1 | Historical pilot uncertainty and segmentation quality correlation analysis | 19 | 19 | **MATCH** |
| Table 6.2 | Post-hoc temperature scaling and ECE reduction across calibration and evaluation cohorts | 21 | 21 | **MATCH** |
| Table 6.3 | Triage and referral action distribution of the deterministic reliability-aware agent | 22 | 22 | **MATCH** |
| Table 7.1 | Systematic missing-modality stress test across four input conditions on 105 evaluation cases | 24 | 24 | **MATCH** |
| Table 7.2 | Qualitative comparison of deep learning brain tumor segmentation paradigms | 25 | 25 | **MATCH** |
| Table A.1 | Subject-level dataset split manifest summary | 28 | 28 | **MATCH** |
| Table B.1 | Reproducibility checklist and hyperparameter specifications | 28 | 28 | **MATCH** |
| Table C.1 | Research governance, claim boundary, and methodological integrity checklist | 29 | 29 | **MATCH** |

### List of Figures (LOF) Alignment Verification
| Figure Number | Figure Title | Listed Page | Actual Page | Alignment Status |
|---|---|:---:|:---:|:---:|
| Figure 4.1 | Multi-modal MRI input channels and segmentation subregions with modality mapping | 11 | 11 | **MATCH** |
| Figure 4.2 | Five-model 2D segmentation architecture portfolio and parameter complexity | 12 | 12 | **MATCH** |
| Figure 4.3 | Representative qualitative segmentation output from historical 5-epoch U-Net pilot | 12 | 12 | **MATCH** |
| Figure 6.1 | Best validation Dice and IoU scores across the historical three-model pilot | 16 | 16 | **MATCH** |
| Figure 6.2 | Foreground Dice comparison by encoded tumor label at model-specific best epochs | 16 | 16 | **MATCH** |
| Figure 6.3 | Training and validation loss recorded at each epoch for historical pilot models | 17 | 17 | **MATCH** |
| Figure 6.4 | Validation mean foreground Dice across the five pilot training epochs | 17 | 17 | **MATCH** |
| Figure 6.5A | Pilot validation slice distribution across rounded predictive entropy thresholds (N = 3,223) | 20 | 20 | **MATCH** |
| Figure 6.5B | Segmentation quality across equal-frequency predictive entropy quintiles (N = 2,460) | 20 | 20 | **MATCH** |
| Figure 6.6 | Complete foreground miss rate among validation slices containing tumor pixels | 21 | 21 | **MATCH** |
| Figure 7.1 | Architecture of the implemented deterministic reliability-aware AI agent and triage pipeline | 26 | 26 | **MATCH** |

---

## 6. Layout Defect Remediation Details

### Remediation of Table 4.2 Stranded Row Defect
- **Prior Issue:** Table 4.2 previously split across pages 13 and 14, with the section heading, introductory paragraph, and first six rows on page 13, leaving a single stranded row and table caption isolated at the top of page 14.
- **Resolution Applied:** Inserted an explicit page break immediately preceding Section 4.5.
- **Verified Layout:** Section 4.5 heading, introductory text, Table 4.2 (all 7 content rows), and the table caption are positioned cohesively on Body Page 14 with zero stranded elements.

### Remediation of Table 6.3 and Captions
- **Prior Issue:** Risk of table caption becoming detached from the table across page transitions.
- **Resolution Applied:** Removed extraneous paragraph spacers separating tables from captions and applied `w:cantSplit` to all 4 triage rows.
- **Verified Layout:** Table 6.3 and its caption sit together on Body Page 22.

---

## 7. Preliminary Pages & Academic Governance Verification

The preliminary pages were structured and populated according to formal academic specifications:

1. **Acknowledgement (Page i):**
   - Mentions research supervisor Dr. Ruchika Bala, Assistant Professor (Senior Grade).
   - Accurately cites compute infrastructure: dedicated local compute featuring an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM) alongside departmental workstation facilities. Replaced former references to "institutional high-performance GPU workstations".
   - Acknowledges BraTS 2023 organizers and open-source scientific software communities.

2. **Declaration (Page ii):**
   - States project title, student candidates (Nishant Kumar, Mohammad Tariq, Mohammad Kaif), supervisor, and institution.
   - Formal dating: "Place: Noida", "Date: November 2026".

3. **Certificate (Page iii):**
   - Formal supervision certificate by Dr. Ruchika Bala.
   - Includes signature line `(Signature of Supervisor)` and `Date: ________________________`, left intentionally blank for manual endorsement by authorized faculty.

4. **Abstract (Pages iv–v):**
   - Openly discloses checkpoint-selection bias on the 155-case validation pool.
   - Qualifies probability calibration as measured on the internal $Val_{eval}$ subset.
   - Characterizes missing-T1c vulnerability accurately (failure to trigger automated referral despite collapse in enhancing tumor segmentation).
   - Avoids clinical overclaims; frames agent actions as triage, selective prediction, and simulated referral flags.

---

## 8. Scientific & Citation Accuracy Verification

All citations were checked against the bibliography:

- **Shannon [9]:** C. E. Shannon (1948), cited for foundational information entropy.
- **Guo et al. [6]:** C. Guo et al. (2017), cited for neural network miscalibration and post-hoc temperature scaling.
- **Gal & Ghahramani [7]:** Y. Gal and Z. Ghahramani (2016), cited as the theoretical foundation for Monte Carlo Dropout.
- **Kendall & Gal [10]:** A. Kendall and Y. Gal (2017), cited for distinguishing epistemic from aleatoric uncertainty.
- **Geifman & El-Yaniv [13]:** Y. Geifman and R. El-Yaniv (2017), cited for the formal theory of selective classification and risk-coverage triage.
- **He et al. [8]:** K. He et al. (2016), cited for deep residual learning and residual blocks.
- **Menze et al. [1]:** B. H. Menze et al. (2015), cited for the BraTS benchmark, manual segmentation variability (0.74–0.85 Dice), and 45–90 minute annotation times.
- **Cardoso et al. [12]:** MONAI framework.
- **Paszke et al. [14]:** PyTorch deep learning library.

---

## 9. Conclusion & Acceptance

The Research Edition V2 report generation has been completed in compliance with all governance, visual, and scientific criteria:

1. `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx` is generated, syntactically verified, and structurally aligned.
2. Pagination across TOC, LOT, and LOF matches physical document layout with 100% precision across all 39 pages.
3. Protected baselines remain verified byte-for-byte.
4. Clean repository discipline is preserved with zero git branch/commit alterations.
