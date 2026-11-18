# Statistical Estimand Disclosure: Case-Weighted Mean with Subject-Clustered Bootstrap

**Document ID:** `AUDIT-RESUNET2D-P1.1.4-03`  
**Audit Phase:** Phase 1.1.4 — Final Register Scope, Slice Classification & Statistical Disclosure  
**Date:** November 2026  
**Status:** **STATISTICALLY RIGOROUS & FORMALLY SEALED**  
**Workspace:** `D:\MajorProject`

---

## 1. Executive Summary & Statistical Estimand Definition

In statistical reporting for clinical machine learning, precision regarding the target estimand and resampling mechanics is paramount. 

Following independent review of Phase 1.1.3, this document provides the explicit, unambiguous statistical disclosure governing all reported 3D volumetric metrics and confidence intervals for **ResUNet-2D**:

> **Authoritative Estimand Definition:**  
> All reported segmentation performance metrics represent **case-weighted mean performance with subject-clustered bootstrap confidence intervals** ($N_{\text{cases}} = 105$, $N_{\text{clusters}} = 96$, $B = 1,000$ replications, `seed = 42`).  
> **They must not be described as equal-weighted subject means.**

---

## 2. Mathematical Definition of the Estimand

### 2.1 The Two Competing Estimands in Clustered Data
In the `Val_eval` cohort ($105$ cases, $96$ unique patient subjects), $87$ subjects have exactly $1$ case ($m_c = 1$), while $9$ subjects have longitudinal multi-scan acquisitions with $2$ cases ($m_c = 2$).

1. **Equal-Weighted Subject Mean ($\theta_{\text{subject}}$):**  
   First averages observations within each subject, then averages across all subjects:
   $$\bar{x}_c = \frac{1}{m_c} \sum_{k=1}^{m_c} x_{c,k}, \quad \theta_{\text{subject}} = \frac{1}{N_{\text{subjects}}} \sum_{c=1}^{N_{\text{subjects}}} \bar{x}_c$$
2. **Case-Weighted Mean ($\theta_{\text{case}}$):**  
   Averages across all volume cases directly:
   $$\theta_{\text{case}} = \frac{1}{N_{\text{cases}}} \sum_{i=1}^{N_{\text{cases}}} x_i = \frac{1}{\sum_{c} m_c} \sum_{c=1}^{N_{\text{subjects}}} \sum_{k=1}^{m_c} x_{c,k}$$

### 2.2 Proof of Implementation in `tools/evaluate_hd95_cohort.py`
The project evaluation script [tools/evaluate_hd95_cohort.py](file:///D:/MajorProject/tools/evaluate_hd95_cohort.py#L53-L79) implements `cluster_bootstrap_mean_ci`:

```python
    mean_val = float(df["val"].mean())
    rng = np.random.default_rng(seed)
    boot_means = []

    for _ in range(n_bootstraps):
        sampled_c = rng.choice(clusters, size=n_clusters, replace=True)
        boot_vals = np.concatenate([cluster_map[c] for c in sampled_c])
        boot_means.append(float(np.mean(boot_vals)))
```

1. **Sample Point Estimate:** `df["val"].mean()` calculates the simple arithmetic average across all $105$ cases. It is therefore a **case-weighted mean**.
2. **Bootstrap Replicate Average:** In each replicate, $96$ clusters are drawn with replacement (`sampled_c`). `cluster_map[c]` retrieves all case values for drawn subject $c$. `np.mean(boot_vals)` averages across all drawn cases. A subject with $2$ cases contributes $2$ values to `boot_vals`; a subject with $1$ case contributes $1$.
3. **Conclusion:** Both the sample point estimate and every bootstrap replication evaluate the **case-weighted mean**. Clustering is accounted for during resampling by preserving intra-subject correlation, but the estimand itself weights each 3D MRI volume equally.

---

## 3. Approved Volumetric Metrics & Preserved 95% Confidence Intervals

In strict compliance with project preservation rules, all approved numerical values are preserved intact without modification:

| Metric Dimension | Point Estimate (Case-Weighted Mean) | Median | Sample Std | Approved 95% Clustered Bootstrap CI | Resampling Specification |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Whole Tumor (WT) Dice** | **`0.9039`** (`0.903862`) | — | — | **`[0.8765, 0.9241]`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |
| **Tumor Core (TC) Dice** | **`0.8597`** (`0.859712`) | — | — | **`[0.8076, 0.8986]`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |
| **Enhancing Tumor (ET) Dice**| **`0.8286`** (`0.828616`) | — | — | **`[0.7794, 0.8684]`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |
| **Macro Volumetric Dice** | **`0.8641`** (`0.864063`) | — | — | **`[0.8265, 0.8930]`** | Clustered Macro Bootstrap on case mean |
| **WT Bounded HD95 (mm)** | **`15.29 mm`** (`15.2928`) | `4.12 mm` | `22.28 mm` | **`[11.38, 20.05] mm`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |
| **TC Bounded HD95 (mm)** | **`8.98 mm`** (`8.9793`) | `3.16 mm` | `16.26 mm` | **`[6.08, 12.74] mm`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |
| **ET Bounded HD95 (mm)** | **`9.92 mm`** (`9.9242`) | `2.00 mm` | `38.16 mm` | **`[4.71, 18.66] mm`** | Clustered by `subject_id` ($B=1000$, `seed=42`) |

*Note: Distance metrics are bounded by the maximum volume diagonal penalty ($373.12$ mm). Median values reflect high boundary accuracy ($2.00$ to $4.12$ mm).*

---

## 4. Scope Limitations & Expunged Guideline Claims

1. **Expungement of Regulatory Claims:**  
   All claims that this statistical evaluation complies with formal clinical reporting standards (e.g. TRIPOD-AI, STARD-AI) are **permanently expunged**. The methodology is a standard academic non-parametric cluster bootstrap.
2. **Selection Exposure of `Val_eval`:**  
   The evaluation cohort (`Val_eval`, $105$ cases) is an internal retrospective validation cohort. While it is strictly disjoint from temperature scaling fitting (`Val_cal`), it was exposed to model checkpoint selection (Epoch 13) and Stage C3 policy threshold tuning ($\tau = 0.35$).
3. **No Prospective Generalizability Claimed:**  
   Confidence intervals quantify internal sampling variation within the retrospective BraTS dataset split. Prospective clinical generalizability across external clinical institutions is not claimed.
