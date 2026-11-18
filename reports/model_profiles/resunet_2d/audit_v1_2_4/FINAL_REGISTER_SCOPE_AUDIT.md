# Flagged Slice Register Scope & Category Reconciliation Audit

**Document ID:** `AUDIT-RESUNET2D-P1.1.4-01`  
**Audit Phase:** Phase 1.1.4 — Final Register Scope, Slice Classification & Statistical Disclosure  
**Date:** November 2026  
**Status:** **EXHAUSTIVELY RECONCILED & FORMALLY SEALED**  
**Workspace:** `D:\MajorProject`

---

## 1. Executive Summary & Audit Mandate

During independent review of Phase 1.1.3, the scope of `slice_level_discrepancy_register.csv` was scrutinized:
> *The Phase 1.1.3 register contained 1,473 rows, whereas the entire flagged universe comprises 10,509 slices. It omitted 8,264 context-suppressed GT-background slices and 772 high-entropy GT-positive slices.*

This audit formally resolves the register scope by providing:
1. **Exhaustive Universe Coverage:** Generation of the complete 10,509-row [complete_flagged_slice_classification_register.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/complete_flagged_slice_classification_register.csv) covering all six distinct flagged categories.
2. **Explicit Scope Classification of the 1,473-Row Extract:** Re-naming and documenting the 1,473-row file as [historical_discrepancy_extract_1473.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/historical_discrepancy_extract_1473.csv), establishing the exact historical criteria that generated this subset.
3. **Category-Level Deterministic Reconciliation:** Proving that the full policy trace of 16,275 slices decomposes deterministically across all action branches with zero residual error.

---

## 2. Exhaustive 6-Category Flagged Universe ($N = 10,509$)

Under the frozen Stage C3 v1.1.3 policy, exactly **`10,509` axial slices** are triaged as `FLAG_UNRELIABLE`. Across the $105$-case validation cohort ($155$ slices per volume), every flagged slice belongs to exactly one of six mutually exclusive, collectively exhaustive subcategories:

| Category ID | Canonical Category Description | Ground Truth | Primary Prediction ($P_{\text{pred}}$) | Axial Support ($k$) | Secondary Prediction ($P_{\text{sec}}$) | Exact Slice Count | % of Flagged ($10,509$) | % of Total ($16,275$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **C1** | **Context-Suppressed True Negative** | Background ($GT=0$) | Empty ($P=0$) | $k = 0$ | Suppressed | **`8,264`** | $78.64\%$ | $50.78\%$ |
| **C2** | **Secondary-Empty True Negative** | Background ($GT=0$) | Empty ($P=0$) | $k \ge 1$ | Empty ($P=0$) | **`704`** | $6.70\%$ | $4.33\%$ |
| **C3** | **High-Entropy False Positive** | Background ($GT=0$) | Non-Empty ($P>0$) | N/A | N/A ($H \ge 0.35$) | **`483`** | $4.60\%$ | $2.97\%$ |
| **C4** | **High-Entropy GT-Positive Flagged** | Tumor ($GT>0$) | Non-Empty ($P>0$) | N/A | N/A ($H \ge 0.35$) | **`772`** | $7.35\%$ | $4.74\%$ |
| **C5** | **Context-Suppressed Primary FN** | Tumor ($GT>0$) | Empty ($P=0$) | $k = 0$ | Suppressed | **`116`** | $1.10\%$ | $0.71\%$ |
| **C6** | **Secondary-Empty Dual FN** | Tumor ($GT>0$) | Empty ($P=0$) | $k \ge 1$ | Empty ($P=0$) | **`170`** | $1.62\%$ | $1.04\%$ |
| **SUM** | **Total Flagged Universe** | — | — | — | — | **`10,509`** | **100.00%** | **64.57%** |

### Mathematical Identity Check:
$$\text{GT-Background Flags} = \text{C1} + \text{C2} + \text{C3} = 8,264 + 704 + 483 = \mathbf{9,451} \text{ slices}$$
$$\text{GT-Positive Flags} = \text{C4} + \text{C5} + \text{C6} = 772 + 116 + 170 = \mathbf{1,058} \text{ slices}$$
$$\text{Total Flagged Slices} = 9,451 + 1,058 = \mathbf{10,509} \text{ slices}$$
$$\text{All Policy Actions} = 5,619 \;(\text{ACCEPT}) + 147 \;(\text{FALLBACK\_SWITCH}) + 10,509 \;(\text{FLAG}) = \mathbf{16,275} \text{ slices}$$

Every slice in [complete_flagged_slice_classification_register.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/complete_flagged_slice_classification_register.csv) maps to this taxonomy with zero unclassified rows.

---

## 3. Provenance & Scope of the 1,473-Row Historical Extract

In Phase 1.1.3, an extract of 1,473 slices was produced under the name `slice_level_discrepancy_register.csv`. It is preserved and explicitly designated in Phase 1.1.4 as [historical_discrepancy_extract_1473.csv](file:///D:/MajorProject/reports/model_profiles/resunet_2d/audit_v1_2_4/historical_discrepancy_extract_1473.csv).

### 3.1 Why Exactly 1,473 Slices Were Selected
The 1,473 rows constituted a focused historical extract targeting the specific subcategories where prior text in `audit_v1_2` had introduced inverted hand-calculated allocations:
1. **Context-Suppressed GT-Positive (`C5`):** All `116` slices. (Prior manual draft had asserted `253`, an erroneous over-allocation of $+137$).
2. **Secondary-Empty GT-Positive (`C6`):** All `170` slices. (Prior manual draft had asserted `31`, an erroneous under-allocation of $-139$).
3. **Secondary-Empty True Negatives (`C2`):** All `704` slices. (Prior manual draft had asserted `843`, an erroneous over-allocation of $+139$).
4. **High-Entropy False Positives (`C3`):** All `483` slices. (Prior manual draft had asserted `481`, an erroneous under-allocation of $-2$).

$$\text{Extract Subtotal} = 116 + 170 + 704 + 483 = \mathbf{1,473} \text{ slices}$$

### 3.2 What Was Omitted from the 1,473-Row Extract
The 1,473-row extract deliberately omitted the two largest categories:
- **`C1`: Context-Suppressed True Negatives (`8,264` slices):** Omitted due to size (8,264 slices), though the historical table had hand-estimated them as $8,127$ ($8,380 - 253 = 8,127$).
- **`C4`: High-Entropy GT-Positive Flagged Predictions (`772` slices):** Omitted because they were non-empty predictions ($P_{\text{pred}} > 0$) rather than fallback queries, though the historical table had hand-estimated them as $774$ ($1,255 - 481 = 774$).

Together, these two omitted categories account for:
$$8,264 + 772 = \mathbf{9,036} \text{ slices}$$
$$1,473 \;(\text{Extract}) + 9,036 \;(\text{Omitted}) = \mathbf{10,509} \text{ slices} \quad (\text{Full Flagged Universe})$$

### 3.3 Formal Scope Representation Rule
To maintain scientific rigor:
- **`historical_discrepancy_extract_1473.csv`** is certified as a **selected forensic extract** designed to adjudicate the four subcategories directly affected by the historical $+137 / -139 / +139 / -2$ hand-partition offsets.
- **`complete_flagged_slice_classification_register.csv`** is certified as the **exhaustive master registry** of all 10,509 flagged slices across all six categories.
