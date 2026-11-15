# Phase G1-B1: Research Edition Generation and Verification Log

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation
**Workspace:** `D:\MajorProject`
**Governance Authority:** `MASTER_PROJECT_BLUEPRINT_v2.0.md`
**Scientific Evidence Authority:** `G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md`
**Phase:** G1-B1 (Full Research Report Upgrade — Research Edition Content Reconstruction & DOCX Generation)
**Execution Timestamp:** 2026-11-12T02:44:00+05:30
**Generated Document:** `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
**Dedicated Artwork Directory:** `tmp/report_artwork_research/`
**Authoritative Builder Script:** `tools/build_final_report.py`

---

## 1. Executive Summary

Phase G1-B1 successfully transformed the historical Minor Project report into a comprehensive, scientifically defensible academic research report: the **Research Edition**.

The new document represents the true maturity of the project:
1. **Full Architectural Portfolio:** Covers all five implemented 2D architectures (U-Net, U-Net++, Attention U-Net, ResUNet-2D, UNetResNet34) with verified PyTorch runtime parameter counts.
2. **Two-Stage Experimental Progression:** Formally separates Stage A (historical 5-epoch pilot benchmark on 39 validation cases) from Stage B (mature 15-epoch research baselines trained on 721 cases / 59,020 axial slices).
3. **Audited Dataset Accounting:** Incorporates the full subject-level census of 1,251 cases across 1,133 distinct subjects, clearly distinguishing active training (721 cases), reserve training (220 cases), validation pool (155 cases: 50 calibration, 105 evaluation), and the sealed test cohort (155 cases).
4. **Transparent Limitation Disclosures:** Explicitly discloses checkpoint-selection bias on the 105-case evaluation pool (Val_eval) in the body of Chapter 5, and documents the sealed test cohort as strictly quarantined.
5. **Probability Calibration & Triage:** Reports post-hoc temperature scaling fitted on 50 calibration cases ($T^* = 2.8808$ for ResUNet, $2.6267$ for UNetResNet34) reducing ECE by 60% on evaluation data (0.0861 to 0.0344). Details the deterministic reliability agent (83.84% retained, 16.16% referred, 54 fallback switches).
6. **Failure Mode Discovery:** Exposes the missing-T1c vulnerability (severe ET/TC collapse with modest entropy and referral changes), proving the necessity of an upstream pre-inference modality gate.
7. **Resolution of Prior Presentation Flaws:** Separated the conflated historical Figure 6.5 into two mathematically consistent visuals: Figure 6.5A (threshold-based slice counts on 3,223 slices) and Figure 6.5B (equal-frequency quintiles on 2,460 foreground slices).

---

## 2. Baseline Hash Verification & Asset Preservation

All 29 protected baseline assets were verified before implementation and re-verified byte-for-byte identical post-generation:

| Protected Asset Path | Verified Status | SHA-256 Checksum |
|---|---|---|
| `Final_Report_Brain_Tumor_Segmentation.docx` | **PASS (UNTOUCHED)** | `534470d399c2e194bedba15721266e3c70f59b1f8151ce1df6c9c24fc08dffa5` |
| `results/baseline_unet2d_5ep/visualizations/sample_01_BraTS-GLI-01416-000_slice_090.png` | **PASS (UNTOUCHED)** | `60601bc2a83f32f5f4f300ea447e9acd69fc5a74f3b02f19c2363361658b0370` |
| `Report_Materials/Brain Tumor Segmentation Benchmark Infographic.png` | **PASS (UNTOUCHED)** | `86d37109f281a2c42e4b42fa1c2a911d7e1be1ce03b600c11bd9bec34fe0b298` |
| `Report_Materials/Brain Tumor Segmentation Conclusions and Future Work.png` | **PASS (UNTOUCHED)** | `044f91afa565757819cafac760e21fae988a559c4d32184693165018fb9deff3` |
| `Report_Materials/Brain Tumor Segmentation Project Overview.png` | **PASS (UNTOUCHED)** | `dbf025fbd1483ecf540ef36ec37980450c1a5afe95a5caa29e126a4480116868` |
| `Report_Materials/Brain Tumor Segmentation Results Infographic.png` | **PASS (UNTOUCHED)** | `fe494cf48a9d163e4d30f91dc5ac9c1ee70b18e751bde29ef56d9b65a933b5e2` |
| `Report_Materials/Final_Minor_Project-2_report_and_PPT_format_&_instructions_Odd2025[2].docx` | **PASS (UNTOUCHED)** | `b864ee4064cc8368d733115e67936fddc08e86c378cb14a5951025d103e14710` |
| `Report_Materials/MINOR_REPORT_NISHANT_KAIF_TARIQ.docx` | **PASS (UNTOUCHED)** | `942b29932be12dae63bf90f846ac6d5826eb9f3b7cb2859912cbd6f13593f967` |
| `Report_Materials/MajorProject_Part1_Project_Handoff_Current_State.pdf` | **PASS (UNTOUCHED)** | `cfff61321fc3e1cf39e9e98cb8d57745b2d1ff1cdc7e47a8bb601fa29b63cfad` |
| `Report_Materials/MajorProject_Part2_Midterm_to_Final_Roadmap.pdf` | **PASS (UNTOUCHED)** | `22009c09fc383188192fbb444a3e5fc5cd4491afe7f839793f37e8e0ad801e7e` |
| `Report_Materials/Reliability-Aware Brain Tumor AI Agent.png` | **PASS (UNTOUCHED)** | `489821dba1ea4872553c1fe956396dc67b7b3ea3e530e8c40b4b95224cb55715` |
| `Report_Materials/Uncertainty Analysis for Brain Tumor Segmentation.png` | **PASS (UNTOUCHED)** | `1e4f3a71af488928fbf722d5268793f42aabb9d3f51977072e5b20efedc07b21` |
| `Report_Materials/all details.pdf` | **PASS (UNTOUCHED)** | `d985766d577be5fa223fc5562a4a7add15f838169cc84de5fbb2708ae21c5a7a` |
| `Report_Materials/best IoU scores.png` | **PASS (UNTOUCHED)** | `67fb9f2bba30d38f1b7a58f3376cab36a54fb653526f42123e4f75751a8cbd87` |
| `Report_Materials/best dice scores.png` | **PASS (UNTOUCHED)** | `13149ad9f69361f0e8b7fe9d4da004083673a689201bb5c3ee9be424d02378e6` |
| `Report_Materials/loss vs epochs.png` | **PASS (UNTOUCHED)** | `530175840a8ba17b3488b294e827cbdf43a28f4272df5304a1ada776ed72c747` |
| `Report_Materials/major project eval 1.docx` | **PASS (UNTOUCHED)** | `5dd786fc5fe7bfef0d416a5c883c39dc659b5764cc26c3d49d4560ec2fb079c8` |
| `Report_Materials/mean dice scores vs epochs.png` | **PASS (UNTOUCHED)** | `71fd052ae3574551f554329480f88bd946cbe423835e30e51206708c7c825f7c` |
| `Report_Materials/miss rate comparison.png` | **PASS (UNTOUCHED)** | `de01616a7140bcc33b00386ee521e5c440f1ae88805c0eaf684ca0f4aa30b138` |
| `Report_Materials/per class dice score.png` | **PASS (UNTOUCHED)** | `67ade54f8771ed7981b607193b3bb0f947cfff92d83a8b2d8070726d18ec3b27` |
| `tmp/report_figures/jiit_logo.png` | **PASS (UNTOUCHED)** | `898aee516b1f00baca3cc6320cad94663ca60ed5dd845cea9574a88395ff61b6` |
| `tmp/report_figures/loss_curves.png` | **PASS (UNTOUCHED)** | `91e5f65d872c371930fd976ac0c3912e065fa257831be5ef63b357a3ba3b5f1d` |
| `tmp/report_figures/mean_dice_trend.png` | **PASS (UNTOUCHED)** | `b7b10deecb23ba9c4ef0fa63ca5f0407b673c9af933e60152b3fd209a84d5255` |
| `tmp/report_figures/miss_rate.png` | **PASS (UNTOUCHED)** | `c15349644a13bf5f25ee748e6aebed9d10cfdeffac4a464c1a0a1fb3ee66cad6` |
| `tmp/report_figures/model_architectures.png` | **PASS (UNTOUCHED)** | `ce05c72db049635c8c5245427f6ca0606211496080fc2801ee3e9b8f32f5c909` |
| `tmp/report_figures/model_metrics.png` | **PASS (UNTOUCHED)** | `0af9ac1374154a2b7bfd5c8d6fc29fbb0b199adddfcd7d26003d293ceb048229` |
| `tmp/report_figures/mri_modalities.png` | **PASS (UNTOUCHED)** | `938902c5aa705ceb5899d5200926ad7873c40552a41704beb7b86a5530023109` |
| `tmp/report_figures/per_class_dice.png` | **PASS (UNTOUCHED)** | `0891a7dc0bc5dc4b396959a2e16ae8fbedbdc9d322bb0fa2c2a93d8434c20c63` |
| `tmp/report_figures/uncertainty_analysis_panels.png` | **PASS (UNTOUCHED)** | `b179f78e159b86e8bc3a6c77537a0886905a2bbda5737fc034fbbd228b366ef6` |

**Verification Status:** 29 / 29 protected baseline assets verified byte-for-byte identical.
**Original Report DOCX Preservation:** CONFIRMED BYTE-FOR-BYTE IDENTICAL.
**Quarantined Test Cohort Preservation:** STRICT READ-ONLY INTEGRITY MAINTAINED; UNOPENED.

---

## 3. Generated Deliverables & Hashes

1. **New Research Edition DOCX:**
   Path: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
   Size: `3,112,362 bytes` (~3.1 MB)
   SHA-256: `4102349254fe57141bdb0903291eb56f9be7b2daa896e7cae1af6d295b4cc5d3`

2. **Dedicated Research Edition Artwork Assets:**
   Directory: `tmp/report_artwork_research/`
   - `jiit_logo.png`: 83,687 bytes (SHA-256: `898aee516b1f00baca3cc6320cad94663ca60ed5dd845cea9574a88395ff61b6`)
   - `mri_modalities.png`: 360,108 bytes (SHA-256: `938902c5aa705ceb5899d5200926ad7873c40552a41704beb7b86a5530023109`)
   - `model_portfolio.png`: 134,765 bytes (SHA-256: `3b8d4bb9ce722ebc3258c734003d6d5eb351ca1a494d6eef39281a4b162c938b`)
   - `entropy_threshold_distribution.png`: 132,714 bytes (SHA-256: `3896561e1b1239857d9d06b5278c7304e286cf25712e0a29486c9d7494cb2fd6`)
   - `entropy_quintile_quality.png`: 240,286 bytes (SHA-256: `996f7c10b2e81fa498a18fe11c21087cf283fce58e376ea02e7740e557b44766`)
   - `agent_architecture.png`: 1,732,843 bytes (SHA-256: `489821dba1ea4872553c1fe956396dc67b7b3ea3e530e8c40b4b95224cb55715`)

3. **Updated Authorized Report Builder:**
   Path: `tools/build_final_report.py`
   Size: `100,560 bytes`
   SHA-256: `c6af824ff89fb0ac3ccd3842aafc1097a6d31db697ff8857596409822f99c850`

---

## 4. Document Structure & Package Integrity Metrics

- **OpenXML Validation:** Valid zip archive containing `word/document.xml`, `[Content_Types].xml`, `word/_rels/document.xml.rels`.
- **Package Files Count:** 51 internal parts.
- **Embedded Media Parts:** Exactly 12 embedded PNG images (1 cover logo + 11 technical figures: Figures 4.1, 4.2, 4.3, 6.1, 6.2, 6.3, 6.4, 6.5A, 6.5B, 6.6, 7.1). All verified valid and decodable via PIL.
- **Document Sections:** 11 sections (Cover page with border, Preliminary pages in Roman lower case, Chapters 1–7 in Arabic numerals with first-page header/footer suppressed, References, Appendices).
- **Paragraphs Count:** 269 paragraphs.
- **Tables Count:** 21 tables total (roster, TOC, List of Tables, List of Figures, List of Abbreviations, plus 16 numbered technical tables).
- **Word Count:** ~11,161 words.
- **Placeholder Audit:** 0 occurrences of `TODO`, `FIXME`, `TBD`, `PLACEHOLDER`, `XXX`, or unformatted text.
- **Sequential Numbering:** Figures and tables strictly sequential from 2.1 to C.1 and 4.1 to 7.1.

---

## 5. Source-to-Claim Evidence Map

| Claim / Numerical Result in Report | Value Reported in Research Edition | Authoritative Evidence Source | Status |
|---|---|---|---|
| Full Dataset Census | 1,251 cases / 1,133 subjects | `data/splits/split_manifest.json` / Gate 4 audit | VERIFIED |
| Active 15-Epoch Training Cohort | 721 cases / 667 subjects / 59,020 slices | `results/baseline_resunet2d_15ep/training_history.json` | VERIFIED |
| Historical Development Reserve | 220 cases / 180 subjects / 18,000 slices | `data/splits/split_manifest.json` audit | VERIFIED |
| Validation Calibration Cohort ($Val_{cal}$) | 50 cases / 47 subjects / 3,982 slices | `data/splits/split_manifest.json` | VERIFIED |
| Validation Evaluation Cohort ($Val_{eval}$) | 105 cases / 96 subjects / 8,382 slices | `data/splits/split_manifest.json` | VERIFIED |
| Sealed Test Cohort ($Test_{sealed}$) | 155 cases / 143 subjects / Quarantined | `data/splits/split_manifest.json` / Split audit | VERIFIED |
| Model Channels & Input Order | `("t1n", "t1c", "t2f", "t2w")` | `src/data/dataset.py` | VERIFIED |
| U-Net Trainable Parameters | 7,763,428 | `src/models/unet.py` live runtime PyTorch audit | VERIFIED |
| U-Net++ Trainable Parameters | 9,160,068 | `src/models/unetplusplus.py` live runtime audit | VERIFIED |
| Attention U-Net Trainable Parameters | 7,852,160 | `src/models/attention_unet.py` live runtime audit | VERIFIED |
| ResUNet-2D Trainable Parameters | 8,114,604 | `src/models/resunet.py` live runtime audit | VERIFIED |
| UNetResNet34 Trainable Parameters | 24,327,524 (18,988 buffers; 24,346,512 total) | `src/models/unet_resnet34.py` live runtime audit | VERIFIED |
| Stage A U-Net Best Dice / IoU | 0.7711 / 0.6329 (Epoch 5) | `results/baseline_unet2d_5ep/training_summary.json` | VERIFIED |
| Stage A U-Net++ Best Dice / IoU | 0.7937 / 0.6582 (Epoch 4; final ep 5 IoU: 0.630749) | `results/baseline_unetplusplus_5ep/training_history.json` | VERIFIED |
| Stage A Attention U-Net Best Dice / IoU | 0.7291 / 0.5886 (Epoch 5) | `results/baseline_attention_unet_5ep/training_summary.json` | VERIFIED |
| Stage B ResUNet Best Validation Epoch | Epoch 13 (Mean Dice: 0.8670 on 155 val pool) | `results/baseline_resunet2d_15ep/training_history.json` | VERIFIED |
| Stage B UNetResNet34 Best Validation Epoch | Epoch 15 (Mean Dice: 0.8554 on 155 val pool) | `results/baseline_unet_resnet34_15ep/training_history.json` | VERIFIED |
| Stage B ResUNet Evaluation Mean Dice [95% CI] | 0.8641 [0.827, 0.893] | Gate 6A / Gate 6 remediation evaluation logs | VERIFIED |
| Stage B ResUNet Hierarchical Dice | WT: 0.9039, TC: 0.8597, ET: 0.8286 | Gate 6A evaluation log (`eval_val_eval.json`) | VERIFIED |
| Stage B ResUNet Median HD95 | WT: 4.12 mm, TC: 3.16 mm, ET: 2.00 mm | `results/baseline_resunet2d_15ep/hd95_evaluation.json` | VERIFIED |
| Stage B UNetResNet34 Evaluation Mean Dice | 0.8535 [0.814, 0.889] | Gate 6A evaluation log | VERIFIED |
| Stage B UNetResNet34 Hierarchical Dice | WT: 0.8843, TC: 0.8423, ET: 0.8339 | Gate 6A evaluation log | VERIFIED |
| Stage B UNetResNet34 Median HD95 | WT: 5.00 mm, TC: 2.64 mm, ET: 1.73 mm | `results/baseline_unet_resnet34_15ep/hd95_evaluation.json` | VERIFIED |
| Fitted Temperature Scaling (50 cases) | ResUNet: $T^* = 2.880783$; UNetResNet34: $T^* = 2.626696$ | Gate 6A calibration log | VERIFIED |
| Calibration ECE Reduction (50 cases) | ResUNet: $0.0502 \to 0.0084$; UNetResNet34: $0.0655 \to 0.0093$ | Gate 6A calibration log | VERIFIED |
| Evaluation ECE Reduction (105 cases) | ResUNet: $0.0861 \to 0.0344$; UNetResNet34: $0.1098 \to 0.0496$ | Gate 6A remediation evaluation log | VERIFIED |
| Pilot Uncertainty Correlation (U-Net) | Spearman $\rho = -0.6542$ (Dice), $\rho = -0.6864$ (IoU) | `results/baseline_unet2d_5ep/uncertainty_summary.json` | VERIFIED |
| Pilot Low-Quality Detection ROC-AUC | 0.8059 (Dice < 0.50) | `results/baseline_unet2d_5ep/uncertainty_summary.json` | VERIFIED |
| Figure 6.5A Threshold Slice Distribution | 820, 713, 648, 537, 505 (N = 3,223 pilot slices) | Pilot validation slice audit / Gate 6 remediation | VERIFIED |
| Figure 6.5B Quintile Quality Degradation | Q1: 0.8217 $\to$ Q5: 0.3008 (N = 2,460 foreground slices) | Pilot validation foreground quintile log | VERIFIED |
| Agent Retention / Referral (Nov 10 B2) | Retained: 5,491 (83.84%), Referred: 1,058 (16.16%) | Canonical Nov 10 B2 evaluation log | VERIFIED |
| Agent Fallback Switches | 54 fallback switches (0.82%) to UNetResNet34 | Canonical Nov 10 B2 evaluation log | VERIFIED |
| Agent Aggregate Dice | 0.8640 (vs standalone ResUNet-2D 0.8641) | Canonical Nov 10 B2 evaluation log | VERIFIED |
| Missing T1c Stress Impact | ET Dice collapses to 0.1684, TC to 0.3132; referral is 14.68% | Missing-modality stress study log | VERIFIED |

---

## 6. Unresolved Source or Citation Questions

1. **Intermediate Draft Parameter Counts:** The historical draft figures `7,759,524` (U-Net), `9,161,220` (U-Net++), and `7,848,519` (Attention U-Net) that appeared in early informal notes remain unresolved regarding their historical origin. Live PyTorch runtime inspection confirmed the true counts as `7,763,428`, `9,160,068`, and `7,852,160`. The intermediate figures are formally marked unresolved and superseded.
2. **Missing-T1c Severe Failure Sensitivity:** While aggregate foreground referral under missing T1c is 14.68%, the exact proportion of individual severely degraded cases ($Dice < 0.50$) that triggered referral versus silent failure cannot be determined without a patient-level cross-tabulation. The report explicitly adopts the bounded empirical interpretation without asserting unverified case-level sensitivity.
3. **Word Dynamic Field Refresh:** The Table of Contents, List of Tables, and List of Figures contain accurate initial page numbers based on the standard Word layout engine, but Microsoft Word users should perform a standard `F9` field refresh upon opening the document to account for local printer driver pagination variations.

---

## 7. Git Verification & Working Tree State

- **Branch:** `main`
- **HEAD Commit:** `64c5ff5f86e631df22a73180ba5512b618749904`
- **Authorized Source File Modified:** Exactly one file: `tools/build_final_report.py`.
- **New Deliverables Confined to Authorized Paths:**
  - DOCX: `docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx`
  - Artwork: `tmp/report_artwork_research/`
  - Verification Log: `docs/reports/generation_and_verification_log.md`
- **Git Operations:** Zero commits, zero staging, zero branch or remote modifications.
