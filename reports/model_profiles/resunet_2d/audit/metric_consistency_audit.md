# METRIC CONSISTENCY & CROSS-DELIVERABLE AUDIT REPORT: ResUNet-2D
**Model:** ResUNet-2D (`ResUNet2D`)  
**Profile:** `STAGE_C_15EP` (Epoch 13 Peak)  
**Audit Date:** 2026-11-18  
**Scope:** Verification of numerical consistency across all 6 deliverable formats:
1. Reconstructed Dossier v1.1 (`.md`, `.docx`, `.pdf`)
2. One-Page Model Overview (`.png`, `.pdf`, `.svg`)
3. Presentation Deck (`.pptx`) and Presenter Notes (`.md`)
4. 25 Technical Scientific Assets (`assets/technical/`)
5. 16 Non-Technical Explainer Cards (`assets/non_technical/`)
6. 3 Qualitative Example Panels (`assets/qualitative_examples/`)

---

## 1. CROSS-DELIVERABLE METRIC MATRIX

The following table cross-references all core metrics against every deliverable format to confirm 100% numerical concordance and zero data drift:

| Metric Category & Exact Definition | Canonical Truth | Dossier v1.1 (MD/DOCX/PDF) | One-Page Overview | Presentation Deck & Notes | Technical Figures | Non-Technical Cards | Audit Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Total Trainable Parameters** | **8,111,892 (8.11M)** | 8,111,892 (8.11M) | 8.11M (8,111,892) | 8.11M (8,111,892) | 8.11M (Fig 01, 03) | 8.11M (Card 01, 16) | **PASS (100%)** |
| **Model Checkpoint Size (Disk)** | **32.4 MB (FP32)** | 32.4 MB | 32.4 MB | 32.4 MB | 32.4 MB (Fig 23) | 32.4 MB (Card 16) | **PASS (100%)** |
| **Volumetric Macro Dice (Val_eval)** | **0.8641** | 0.8641 | 0.8641 | 0.8641 | 0.8641 (Fig 09) | 0.8641 (Card 07, 16) | **PASS (100%)** |
| **Whole Tumor (WT) Dice** | **0.9039** [0.8878, 0.9189] | 0.9039 [0.8878, 0.9189] | 0.9039 [0.8878, 0.9189] | 0.9039 [0.8878, 0.9189] | 0.9039 (Fig 09) | 0.9039 (Card 04, 16) | **PASS (100%)** |
| **Tumor Core (TC) Dice** | **0.8597** [0.8351, 0.8821] | 0.8597 [0.8351, 0.8821] | 0.8597 [0.8351, 0.8821] | 0.8597 [0.8351, 0.8821] | 0.8597 (Fig 09) | 0.8597 (Card 04, 16) | **PASS (100%)** |
| **Enhancing Tumor (ET) Dice** | **0.8286** [0.7963, 0.8580] | 0.8286 [0.7963, 0.8580] | 0.8286 [0.7963, 0.8580] | 0.8286 [0.7963, 0.8580] | 0.8286 (Fig 09) | 0.8286 (Card 04, 16) | **PASS (100%)** |
| **Checkpoint Foreground Macro Dice** | **0.8670** (NCR, ED, ET) | 0.8670 (NCR, ED, ET) | 0.8670 (NCR, ED, ET) | 0.8670 (NCR, ED, ET) | 0.8670 (Fig 07, 08) | 0.8670 (Card 01, 16) | **PASS (100%)** |
| **Checkpoint 4-Class Mean Dice** | **0.899968** (incl. BG) | 0.899968 (incl. BG) | 0.899968 (incl. BG) | 0.899968 (incl. BG) | 0.899968 (Fig 08) | Disclosed in text | **PASS (100%)** |
| **Slice-Level Unfiltered Dice** | **0.7150** (6,549 slices) | 0.7150 (6,549 slices) | 0.7150 (6,549 slices) | 0.7150 (6,549 slices) | 0.7150 (Fig 17) | Disclosed in text | **PASS (100%)** |
| **Enhancing Tumor HD95 (Median)** | **2.00 mm** (Mean: 9.92) | 2.00 mm (Mean: 9.92) | 2.00 mm (Mean: 9.92) | 2.00 mm (Mean: 9.92) | 2.00 mm (Fig 09) | 2.00 mm (Card 08) | **PASS (100%)** |
| **Tumor Core HD95 (Median)** | **3.16 mm** (Mean: 8.98) | 3.16 mm (Mean: 8.98) | 3.16 mm (Mean: 8.98) | 3.16 mm (Mean: 8.98) | 3.16 mm (Fig 09) | 3.16 mm (Card 08) | **PASS (100%)** |
| **Whole Tumor HD95 (Median)** | **4.12 mm** (Mean: 15.29) | 4.12 mm (Mean: 15.29) | 4.12 mm (Mean: 15.29) | 4.12 mm (Mean: 15.29) | 4.12 mm (Fig 09) | 4.12 mm (Card 08) | **PASS (100%)** |
| **Optimal Temperature (T\*)** | **2.880459** | 2.880459 (2.8805) | 2.880459 (2.8805) | 2.880459 (2.8805) | 2.8805 (Fig 10-13) | 2.8805 (Card 10, 16) | **PASS (100%)** |
| **Expected Calibration Error (ECE)** | **8.59% -> 3.41%** | 8.59% -> 3.41% | 8.59% -> 3.41% | 8.59% -> 3.41% | 8.59% -> 3.41% (Fig 12)| 8.59% -> 3.41% (Card 10)| **PASS (100%)** |
| **Negative Log-Likelihood (NLL)** | **1.0277 -> 0.4415** | 1.0277 -> 0.4415 | 1.0277 -> 0.4415 | 1.0277 -> 0.4415 | 1.0277 -> 0.4415 (Fig 12)| 1.0277 -> 0.4415 (Card 10)| **PASS (100%)** |
| **Mean Predictive Entropy (H)** | **0.2686 +- 0.0903** | 0.2686 +- 0.0903 | 0.2686 +- 0.0903 | 0.2686 +- 0.0903 | 0.2686 (Fig 15) | 0.2686 (Card 11) | **PASS (100%)** |
| **Entropy Operating Threshold (tau)** | **0.35** | 0.35 | 0.35 | 0.35 | 0.35 (Fig 15-18) | 0.35 (Card 11, 16) | **PASS (100%)** |
| **Autonomous Retention Volume** | **83.84%** (5,491 slices) | 83.84% (5,491 / 6,549) | 83.84% (5,491 slices) | 83.84% (5,491 slices) | 83.84% (Fig 20) | 83.8% (Card 15) | **PASS (100%)** |
| **Referred Flawed Volume** | **16.16%** (1,058 slices) | 16.16% (1,058 / 6,549) | 16.16% (1,058 slices) | 16.16% (1,058 slices) | 16.16% (Fig 20) | 16.2% (Card 15) | **PASS (100%)** |
| **Severe Error Reduction (<0.50)** | **-34.1%** (20.5% -> 13.5%) | -34.1% (20.5% -> 13.5%) | -34.1% (20.5% -> 13.5%) | -34.1% (20.5% -> 13.5%) | -34.1% (Fig 20) | -34.1% (Card 16) | **PASS (100%)** |
| **Axial Fallback Rescued Slices** | **54 True Tumor Slices** | 54 True Tumor Slices | 54 True Tumor Slices | 54 True Tumor Slices | 54 Slices (Fig 21) | 54 Slices (Card 14) | **PASS (100%)** |

---

## 2. POPULATION DENOMINATOR AUDIT

To eliminate reviewer confusion, all percentages across all assets were audited for explicit denominator identification:
1. **Total Contiguous Slices in `Val_eval`:** Exactly **16,275 slices** ($105 \text{ cases} \times 155 \text{ slices}$).
   - `ACCEPT`: 5,619 slices = **34.53%** of all slices.
   - `FALLBACK_SWITCH`: 147 slices = **0.90%** of all slices.
   - `FLAG_UNRELIABLE`: 10,509 slices = **64.57%** of all slices (including 9,401 true-empty background slices).
2. **Non-Empty Primary Predictions:** Exactly **6,874 slices**.
   - `ACCEPT`: 5,619 slices = **81.74%** of non-empty predictions.
   - `FLAG_UNRELIABLE`: 1,255 slices = **18.26%** of non-empty predictions.
3. **Ground-Truth Tumor Slices:** Exactly **6,549 slices**.
   - Autonomously retained by full Stage C policy: 5,491 slices = **83.84%** (Mean Dice = 0.7803).
   - Referred to clinicians: 1,058 slices = **16.16%** (Mean Dice = 0.3891).

---

## 3. CHECKPOINT INTEGRITY & CRYPTOGRAPHIC VERIFICATION

- Checkpoint File: `weights/stage_c_15ep/resunet_best.pt`
- File Size: `32,448,512 bytes` (32.4 MB)
- Expected SHA-256: `1d85be47225c5684f18ba6b89ae6563ee9f39aa13c9e99a80373ab10a300d81a`
- Live Computed SHA-256: `1d85be47225c5684f18ba6b89ae6563ee9f39aa13c9e99a80373ab10a300d81a`
- **Result:** **CRYPTOGRAPHIC MATCH CONFIRMED (100%)**.

---

## 4. AUDIT SIGN-OFF

All reported figures, tables, text spans, presentations, and visual graphics across the ResUNet-2D portfolio have been verified for complete metric consistency. Zero discrepancies or ungrounded statistics exist.
