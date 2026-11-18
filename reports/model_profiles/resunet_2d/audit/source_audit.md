# ResUNet-2D Evidence Source Audit & Traceability Matrix

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation  
**Model:** ResUNet-2D (Primary Champion)  
**Profile:** `STAGE_C_15EP`  
**Best Checkpoint Epoch:** Epoch 13  
**Audit Date:** November 18, 2026  
**Auditor:** Senior AI/ML Research Communicator & Evidence Auditor  
**Scope:** Formal verification of all metrics, configurations, and artifacts reported in the ResUNet-2D Model Dossier v1.1, Visual Asset Library, One-Page Overview, and Presentation Deck.

---

## 1. EVIDENCE AUDIT MATRIX

| Reported Item | Primary Source File Path | Verified Numerical Value | Cohort / Protocol | Evidence Status |
|---|---|---:|---|---|
| **Total Trainable Parameters** | `src/models/resunet.py`, `results/full_training_15ep/resunet_2d/config.json` | 8,114,604 | Full architecture parameter summation (100% active gradients) | Verified from source |
| **Best Checkpoint SHA-256** | `results/full_training_15ep/resunet_2d/best_model.pt` | `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d` | Binary SHA-256 digest of physical model weights | Verified from source |
| **Best Checkpoint File Size** | `results/full_training_15ep/resunet_2d/best_model.pt` | 97,486,845 bytes (~92.97 MB) | Physical disk size | Verified from source |
| **Final Checkpoint SHA-256** | `results/full_training_15ep/resunet_2d/final_model.pt` | `3c6e265f9b6093963ce346468d5dc1d40e878cf335184a4d0a309f145331103f` | Binary SHA-256 digest of Epoch 15 weights | Verified from source |
| **Final Checkpoint File Size** | `results/full_training_15ep/resunet_2d/final_model.pt` | 97,488,175 bytes (~92.97 MB) | Physical disk size | Verified from source |
| **Training Run Duration** | `results/full_training_15ep/resunet_2d/summary.json` | 8.0851 hours (485.10 min) | Cumulative 15-epoch training execution | Verified from source |
| **Peak VRAM Consumption** | `results/full_training_15ep/resunet_2d/summary.json` | 538.61 MB | CUDA memory tracking during training | Verified from source |
| **Best Checkpoint Epoch** | `results/full_training_15ep/resunet_2d/summary.json`, `history.csv` | Epoch 13 | Highest validation foreground macro mean Dice | Verified from source |
| **Epoch 13 Foreground Macro Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.867045 (reported as 0.8670) | Mean of NCR (0.8210), ED (0.8765), ET (0.9037) | Verified from source |
| **Epoch 13 Class 0 (BG) Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.998738 | Validation slice background segmentation | Verified from source |
| **Epoch 13 Class 1 (NCR) Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.820963 | Validation slice necrotic core segmentation | Verified from source |
| **Epoch 13 Class 2 (ED) Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.876465 | Validation slice peritumoral edema segmentation | Verified from source |
| **Epoch 13 Class 3 (ET) Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.903708 | Validation slice enhancing tumor segmentation | Verified from source |
| **Epoch 13 4-Class Arithmetic Mean** | Derived from `results/full_training_15ep/resunet_2d/history.csv` | 0.899968 | Arithmetic mean across all 4 classes including BG | Derived from verified data |
| **Epoch 14 Validation Loss** | `results/full_training_15ep/resunet_2d/history.csv` | 0.206310 (reported as 0.2063) | Global minimum validation loss | Verified from source |
| **Epoch 15 Validation Loss** | `results/full_training_15ep/resunet_2d/history.csv` | 0.208839 (reported as 0.2088) | Final epoch validation loss | Verified from source |
| **Epoch 15 Validation Mean Dice** | `results/full_training_15ep/resunet_2d/history.csv` | 0.863087 (reported as 0.8631) | Final epoch validation foreground macro Dice | Verified from source |
| **Epoch 11–15 Training Loss Range** | `results/full_training_15ep/resunet_2d/history.csv` | 0.189466 to 0.176589 | Monotonic loss progression across late epochs | Verified from source |
| **Optimal Calibrated Temperature ($T^*$)** | `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json` | 2.8804590702056885 | L-BFGS fit on `Val_cal` (50 cases, 36 iterations) | Verified from source |
| **Val_cal Case & Voxel Count** | `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json` | 50 cases, 47 subjects, 5,020,730 voxels | Stratified foreground/boundary calibration sample | Verified from source |
| **Val_cal Uncalibrated ECE** | `results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv` | 0.050168 (5.02%) | Pre-calibration baseline ($T=1.0$) on `Val_cal` | Verified from source |
| **Val_cal Calibrated ECE** | `results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv` | 0.008402 (0.84%) | Post-calibration ($T^*=2.8805$); −83.25% relative | Verified from source |
| **Val_cal Uncalibrated NLL** | `results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv` | 0.415425 | Pre-calibration cross-entropy on `Val_cal` | Verified from source |
| **Val_cal Calibrated NLL** | `results/stage_c/phase2/c2_calibration_v1/calibration_metrics.csv` | 0.217311 | Post-calibration cross-entropy; −47.69% relative | Verified from source |
| **Val_eval Case & Voxel Count** | `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json` | 105 cases, 96 subjects, 311,546 voxels | Held-out evaluation sample | Verified from source |
| **Val_eval Uncalibrated ECE** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.085896 (8.59%) | Pre-calibration baseline ($T=1.0$) on `Val_eval` | Verified from source |
| **Val_eval Calibrated ECE** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.034080 (3.41%) | Post-calibration ($T^*=2.8805$); −60.32% relative | Verified from source |
| **Val_eval Uncalibrated NLL** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 1.027677 | Pre-calibration cross-entropy on `Val_eval` | Verified from source |
| **Val_eval Calibrated NLL** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.441455 | Post-calibration cross-entropy; −57.04% relative | Verified from source |
| **Val_eval Uncalibrated MCE** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.307274 (30.73%) | Maximum Calibration Error ($T=1.0$) on `Val_eval` | Verified from source |
| **Val_eval Calibrated MCE** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.264217 (26.42%) | Maximum Calibration Error ($T^*=2.8805$) | Verified from source |
| **Val_eval Uncalibrated Brier** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.185057 | Quadratic scoring rule ($T=1.0$) on `Val_eval` | Verified from source |
| **Val_eval Calibrated Brier** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 0.165966 | Quadratic scoring rule ($T^*=2.8805$) | Verified from source |
| **Argmax Invariance Mismatches** | `results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json` | 0 (100% invariant) | Zero mask changes across all 16,275 slices | Verified from source |
| **Val_eval Volumetric Mean Dice** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv`, `results/hd95_evaluation/hd95_summary.json` | 0.864063 (reported as 0.8641) | 3D Volumetric Dice over 105 cases; 95% CI [0.8265, 0.8930] | Verified from source |
| **Val_eval WT Volumetric Dice** | `results/hd95_evaluation/hd95_summary.json` | 0.903862 (reported as 0.9039) | Whole Tumor 3D Dice; 95% CI [0.8765, 0.9241] | Verified from source |
| **Val_eval TC Volumetric Dice** | `results/hd95_evaluation/hd95_summary.json` | 0.859712 (reported as 0.8597) | Tumor Core 3D Dice; 95% CI [0.8076, 0.8986] | Verified from source |
| **Val_eval ET Volumetric Dice** | `results/hd95_evaluation/hd95_summary.json` | 0.828616 (reported as 0.8286) | Enhancing Tumor 3D Dice; 95% CI [0.7794, 0.8684] | Verified from source |
| **Val_eval WT Bounded HD95** | `results/hd95_evaluation/hd95_summary.json` | 15.29 mm (Median: 4.12 mm) | Bounded 95th percentile Hausdorff dist; 95% CI [11.38, 20.05] mm | Verified from source |
| **Val_eval TC Bounded HD95** | `results/hd95_evaluation/hd95_summary.json` | 8.98 mm (Median: 3.16 mm) | Bounded 95th percentile Hausdorff dist; 95% CI [6.08, 12.74] mm | Verified from source |
| **Val_eval ET Bounded HD95** | `results/hd95_evaluation/hd95_summary.json` | 9.92 mm (Median: 2.00 mm) | Bounded 95th percentile Hausdorff dist; 95% CI [4.71, 18.66] mm | Verified from source |
| **Val_eval Total Slice Count** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 16,275 slices | 105 cases $\times$ 155 slices per volume | Verified from source |
| **Empty Prediction Slices** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 9,401 slices (57.76% of 16,275) | 0 predicted foreground voxels; entropy undefined | Verified from source |
| **Non-Empty Prediction Slices** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 6,874 slices (42.24% of 16,275) | $\ge 1$ predicted foreground voxels | Verified from source |
| **Mean Foreground Entropy** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 0.2686 (Std: 0.0903) | Mean normalized Shannon entropy over 6,874 slices | Verified from source |
| **Median Foreground Entropy** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 0.2474 | Median normalized Shannon entropy | Verified from source |
| **IQR Foreground Entropy** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | [0.1995, 0.3185] (IQR = 0.1190) | 25th and 75th percentiles of slice entropy | Verified from source |
| **Min / Max Foreground Entropy** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 0.0000 / 0.6146 | Extreme values across 6,874 non-empty slices | Verified from source |
| **Non-Empty Slices with $H \ge 0.35$** | Derived from slice JSONs in `results/.../resunet_2d/` | 1,255 slices (18.26% of 6,874) | Non-empty slices triggering high-entropy triage | Derived from verified data |
| **Non-Empty Slices with $H < 0.35$** | Derived from slice JSONs in `results/.../resunet_2d/` | 5,619 slices (81.74% of 6,874) | Non-empty slices satisfying acceptance threshold | Derived from verified data |
| **Total Evaluation Action: ACCEPT** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 5,619 slices (34.53% of 16,275) | Total autonomous ACCEPT decisions across all slices | Verified from source |
| **Total Action: FALLBACK_SWITCH** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 147 slices (0.90% of 16,275) | Empty primary replaced by secondary model | Verified from source |
| **Total Action: FLAG_UNRELIABLE** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 10,509 slices (64.57% of 16,275) | 1,255 high-entropy + 9,254 unrescued empty slices | Verified from source |
| **Ground-Truth Positive Slices** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 6,549 slices (40.24% of 16,275) | True tumor-bearing slices in `Val_eval` | Verified from source |
| **GT-Positive Retained: Full Policy** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 5,491 slices (83.84% of 6,549) | 5,437 primary ACCEPT + 54 FALLBACK_SWITCH | Verified from source |
| **GT-Positive Retained: Entropy-Only** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 5,437 slices (83.02% of 6,549) | Primary ACCEPT without fallback recovery | Verified from source |
| **GT-Positive Fallback Recovery** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 54 slices (0.82% of 6,549) | Tumor slices rescued by UNetResNet34 | Verified from source |
| **Unfiltered Primary Tumor Dice** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 0.7150 | Mean slice Dice across all 6,549 tumor slices | Verified from source |
| **Retained Tumor Slice Dice** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 0.7803 | Mean slice Dice across 5,491 retained slices (+0.0653) | Verified from source |
| **Referred Tumor Slice Dice** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 0.3891 | Mean slice Dice across 1,058 referred tumor slices | Verified from source |
| **Unfiltered Severe Failures (<0.5)** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 1,343 slices (20.51% of 6,549) | Severe failure baseline | Verified from source |
| **Retained Severe Failures (<0.5)** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 742 slices (13.51% of 5,491) | −34.1% relative reduction in severe errors | Verified from source |
| **Missing Modality: Drop T1n** | `results/missing_modality_robustness/modality_summary.json` | Mean Dice: 0.8517 (WT: 0.8939, TC: 0.8494, ET: 0.8118) | Withhold T1n channel | Verified from source |
| **Missing Modality: Drop T1c** | `results/missing_modality_robustness/modality_summary.json` | Mean Dice: 0.4449 (WT: 0.8532, TC: 0.3132, ET: 0.1684) | Withhold T1c channel (catastrophic core failure) | Verified from source |
| **Missing Modality: Drop T2w** | `results/missing_modality_robustness/modality_summary.json` | Mean Dice: 0.6058 (WT: 0.4388, TC: 0.6901, ET: 0.6885) | Withhold T2w channel (catastrophic WT failure) | Verified from source |
| **Missing Modality: Drop T2-FLAIR** | `results/missing_modality_robustness/modality_summary.json` | Mean Dice: 0.8394 (WT: 0.8748, TC: 0.8342, ET: 0.8091) | Withhold T2-FLAIR channel | Verified from source |
| **Bootstrap CI Resampling Unit** | `tools/evaluate_hd95_cohort.py:53`, `results/.../cluster_bootstrap_confidence_intervals.csv` | Biological Subject (96 clusters) | Cluster-bootstrap (1,000 resamples, seed 42) | Verified from source |

---

## 2. AUDIT SUMMARY & LABELS DISCLOSURE

1. **Denominators Strictly Segregated:**
   - Evaluated Slices ($N = 16,275$): All 2D axial slices in `Val_eval` (105 cases $\times$ 155 slices).
   - Non-Empty Predictions ($N = 6,874$): Slices where ResUNet-2D predicted $\ge 1$ tumor voxel ($42.24\%$).
   - Ground-Truth Tumor Slices ($N = 6,549$): Slices containing true anatomical tumor foreground ($40.24\%$).
   - **Crucial Distinction:** The 5,619 ACCEPT actions represent **34.53% of all 16,275 slices** and **81.74% of 6,874 non-empty predictions**. The 83.02% figure refers strictly to the **ground-truth tumor slice retention (5,437 / 6,549)** under entropy gating. These metrics are strictly segregated in v1.1.
2. **Foreground Macro Mean Labeling:**
   - The value **0.8670** (0.867045) is verified as the macro average across the three tumor foreground classes (NCR: 0.8210, ED: 0.8765, ET: 0.9037). It is correctly labeled **`Foreground Macro Mean Dice (NCR, ED, ET)`**. The 4-class mean (including Background 0.9987) is calculated as **0.899968** and explicitly separated.
3. **Resampling Unit Clarification:**
   - Both `evaluate_hd95_cohort.py` and `c3_selective_prediction_v1_1_3` explicitly resample at the **biological subject level** (96 clusters), properly accounting for intra-subject slice correlation across repeated longitudinal cases.
