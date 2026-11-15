# Master Project Roadmap: Brain Tumor Segmentation & Reliability-Aware AI Agent

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation Using Multimodal MRI, 2D CNNs, and Selective Uncertainty Calibration
**Root Document:** `D:\MajorProject\archive\project_roadmap\MASTER_ROADMAP.md`
**Master PDF Blueprint:** `../../docs/MASTER_PROJECT_BLUEPRINT_AND_ROADMAP.pdf`
**Affiliation:** Jaypee Institute of Information Technology (JIIT), Noida — Department of CSE & IT
**Course & Academic Year:** B.Tech Major Project-II (`15B19CI791`, 8th Semester, 2026–2027)
**Student Team:** Nishant Kumar (9923103009), Mohammad Tariq (9923103011), Mohammad Kaif (9923103012)
**Project Lead & Supervisor:** Dr. Ruchika Bala, Assistant Professor (Senior Grade)
**Reconciliation & Master Freeze Date:** November 2026
**Project Scorecard:** **40.3 / 50.0 (80.6% • Grade A-)**

---

## 1. Overall Goal & Strategic Framing

The goal of this project is to build, rigorously evaluate, and demonstrate a **Reliability-Aware Brain Tumor Segmentation System** using multi-parametric MRI (mpMRI) scans from the BraTS 2023 Adult Glioma (GLI) dataset.

Rather than chasing marginal aggregate Dice improvements through opaque black-box ensembles or proposing impractical automated model-switching heuristics that degrade performance, the project solves two foundational clinical challenges:
1. **The Overconfidence & Miscalibration Problem:** Deep neural networks produce raw softmax probabilities that are dangerously uncalibrated—predicting ambiguous or missing tumor boundaries with 95%+ confidence despite catastrophic spatial overlap ($Dice < 0.50$).
2. **The Safe Clinical Deployment Imperative:** Slices with high predictive uncertainty must be systematically isolated and deferred for expert radiologist review, rather than forced through automated pipelines.

### Our Proven Three-Pillar Architecture:
1. **Champion Segmentor (ResUNet-2D):** 8.11M parameters, pre-activation residual units, Group Normalization ($G=8$), 538 MB peak VRAM. Achieves superior validation Mean Dice (**0.8670**) and out-of-sample evaluation Mean Dice (**0.8641**), outperforming heavy transfer learning backbones (UNetResNet34, 24.3M params, Dice 0.8535) which suffer from severe validation loss inflation (0.5166).
2. **Post-Hoc Probability Calibration:** Parameter-efficient Temperature Scaling ($T^* = 2.8808$) optimized via L-BFGS, reducing Expected Calibration Error (ECE) by **60.1%** ($8.61\% \rightarrow 3.44\%$) across 311,546 voxels while strictly preserving argmax spatial predictions ($Dice = 0.8641$ invariant).
3. **Selective Prediction & Radiological Triage Agent:** At threshold $\tau_{\text{abstain}} = 0.35$, the agent autonomously accepts **84.06%** of high-confidence foreground slices (Mean Dice **0.8303**), while flagging the **15.94%** lowest-quality, most degraded slices (Mean Dice **0.4994**) for human radiologist review, operating at real-time inference latency (18.23 ms per slice).

---

## 2. Definitive Model Portfolio (Why 5 Models is Optimal)

The project established and benchmarked **5 distinct model architectures**:
- **Baseline 1:** `UNet2D` (7.76M params) $\rightarrow$ 5-epoch baseline (Dice 0.7711).
- **Baseline 2:** `UNetPlusPlus2D` (9.16M params) $\rightarrow$ 5-epoch dense pathways & deep supervision (Dice 0.7937).
- **Baseline 3:** `AttentionUNet2D` (7.85M params) $\rightarrow$ 5-epoch additive attention gating (Dice 0.7291).
- **Champion Model:** `ResUNet2D` (**8,114,604 params**) $\rightarrow$ 15-epoch full training (Best Val Dice **0.8670**, Eval Dice **0.8641**).
- **Comparative Backbone:** `UNetResNet34` (24,327,524 params) $\rightarrow$ 15-epoch transfer learning (Best Val Dice 0.8554, overfitted).

*Formally Scoped-Out Architectures:*
- `TransUNet`: Excluded due to GPU memory constraints (exceeds 8 GB VRAM with transformer self-attention).
- `3D U-Net`: Excluded due to 16+ GB VRAM requirement for 3D volumetric convolutions.
- `MedSAM`: Evaluated and rejected as primary model (uncalibrated, prompt-dependent, 360M params).

---

## 3. Verified Master Execution Roadmap

```
[Phase 0: Ideation & Supervisor Mandate] (COMPLETED)
                   ▲
[Phase 1: Dataset Census & 3-Way Patient-Level Split] (COMPLETED - 1,251 Cases)
                   ▲
[Phase 2: 5-Epoch Baseline Benchmarking on 197-Case Split] (COMPLETED - 3 Models)
                   ▲
[Phase 3: Predictive Uncertainty & The Model Selection Paradox] (COMPLETED - rho=-0.6542, 48.35%)
                   ▲
[Phase 4: Full-Scale 15-Epoch Training on 721 Cases] (COMPLETED - ResUNet vs UNetResNet34)
                   ▲
[Phase 5: Temperature Calibration & Gate 6A Remediation] (COMPLETED - ECE -60.1%, Health Score 80.6%)
                   ▲
[Phase 6: Clinical PACS/Triage Dashboard Update (tools/demo_app.py)] (IMMEDIATE NEXT STEP)
                   ▲
[Phase 7: Gate 6B Quarantined Test Cohort Final Benchmark] (QUARANTINED - 155 Cases, Single Pass)
                   ▲
[Phase 8: Comprehensive Dissertation Rewrite & Viva Defense Deck] (FINAL ACADEMIC SUBMISSION)
```

---

## 4. Key Artifacts on Disk

- **Master PDF Blueprint (15 Pages):** `../../docs/MASTER_PROJECT_BLUEPRINT_AND_ROADMAP.pdf`
- **Champion Checkpoint:** `results/full_training_15ep/resunet_2d/best_model.pt` (SHA-256: `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d`)
- **Gate 6A Remediation Results:** `results/gate6a_remediation_v1/evaluation_summary.json`
- **Quarantined Test Manifest:** `results/dataset_splits/test_manifest.json` (SHA-256: `694faaa4ff100066d67aa5c87c545a709e4eed43a61625d3bea9f43c50d6dfb7`)
- **Project Health Audit Scorecard:** `results/project_health_audit/project_scorecard.json` (40.3 / 50.0)
- **Clinical Showcase Prototype:** `tools/demo_app.py`
