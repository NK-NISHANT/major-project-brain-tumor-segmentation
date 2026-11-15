# Project Progress Checklist & Milestone Tracker (Authoritative Master)

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation Using Multimodal MRI, 2D CNNs, and Selective Uncertainty Calibration
**Root Document:** `D:\MajorProject\archive\project_roadmap\PROGRESS_CHECKLIST.md`
**Parent Blueprint:** `../../docs/MASTER_PROJECT_BLUEPRINT_AND_ROADMAP.pdf`
**Last Updated & Reconciled:** November 2026
**Audited Score:** **40.3 / 50.0 (80.6% • Grade A-)**

---

## 1. Master Task Tracker

| Task ID | Phase | Priority | Task Title | Status | Verification Evidence / Reference |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **AUDIT-01** | Phase A | HIGH | Audit `data/raw/` census (197 complete, 1 excluded on disk) | **COMPLETE** | 197 complete cases verified. |
| **AUDIT-02** | Phase A | HIGH | Verify 158/39 split disjointness ($158 \cap 39 = \emptyset$) | **COMPLETE** | Disjointness mathematically proven. |
| **AUDIT-03** | Phase A | HIGH | Census 1,251 cases in archive; verify headers | **COMPLETE** | 1,251 cases verified across 1,133 unique patients. |
| **AUDIT-04** | Phase A | HIGH | Establish experiment ledger & compute budgets | **COMPLETE** | Hardware and memory budgets frozen. |
| **ARCH-01** | Phase B | HIGH | Implement ResUNet-2D in `src/models/resunet.py` | **COMPLETE** | 8.11M params, PreAct residual units, GroupNorm ($G=8$). |
| **ARCH-02** | Phase B | MEDIUM | Implement U-Net with ResNet-34 encoder | **COMPLETE** | 24.33M params, pretrained torchvision backbone. |
| **ARCH-03** | Phase B | HIGH | Scoping analysis of TransUNet | **SCOPED OUT** | Infeasible due to 8 GB laptop VRAM ceiling. |
| **ARCH-04** | Phase B | HIGH | 3D U-Net VRAM feasibility assessment | **SCOPED OUT** | Infeasible due to 16+ GB VRAM requirement. |
| **ARCH-05** | Phase B | HIGH | Run 5-epoch pilot runs on 197-case split | **COMPLETE** | U-Net (0.7711), U-Net++ (0.7937), AttnUNet (0.7291). |
| **UNCERT-01**| Phase C | HIGH | Quantify predictive entropy correlation with Dice | **COMPLETE** | Spearman $\rho = -0.6542$, ROC-AUC 0.8059. |
| **UNCERT-02**| Phase C | HIGH | Test model-selection via uncertainty paradox | **COMPLETE** | Discovered 48.35% agreement, proving agent necessity. |
| **SPLIT-01** | Phase D | CRITICAL | Ingest and audit 1,251 cases for subject-level split | **COMPLETE** | 1,133 unique patients mapped. |
| **SPLIT-02** | Phase D | CRITICAL | Generate 3-way subject-level split (Train / Val / Test) | **COMPLETE** | Train=847 subj (941 cases), Val=143 subj (155 cases), Test=143 subj (155 cases). |
| **SPLIT-03** | Phase D | CRITICAL | Prove zero patient leakage across Train, Val, Test | **COMPLETE** | Disjointness verified ($Train \cap Val \cap Test = \emptyset$). |
| **SPLIT-04** | Phase D | CRITICAL | Quarantine pristine test set (155 cases) | **COMPLETE** | SHA-256: `694faaa4...` locked. Zero unauthorized access. |
| **TRAIN-01** | Phase E | HIGH | Resolve PyTorch `SequentialLR` scheduler bug | **COMPLETE** | Implemented custom `PerGroupCosineAnnealingLR`. |
| **TRAIN-02** | Phase E | HIGH | 15-epoch training of ResUNet-2D on 721 cases | **COMPLETE** | Best Val Mean Dice: **0.8670** (Epoch 13), Val Loss 0.2088. |
| **TRAIN-03** | Phase E | HIGH | 15-epoch training of UNetResNet34 on 721 cases | **COMPLETE** | Best Val Mean Dice: **0.8554** (Epoch 15), Val Loss 0.5166 (overfit). |
| **CALIB-01** | Phase F | HIGH | Fit Temperature Scaling on 50 calibration cases | **COMPLETE** | $T^*_{\text{ResUNet}} = 2.8808$, $T^*_{\text{UNetResNet34}} = 2.6267$ via L-BFGS. |
| **CALIB-02** | Phase F | HIGH | Evaluate probability calibration on 105 eval cases | **COMPLETE** | ECE reduced by 60.1% ($8.61\% \rightarrow 3.44\%$) on 311,546 voxels. |
| **AGENT-01** | Phase F | HIGH | Implement selective prediction triage policy ($\tau=0.35$) | **COMPLETE** | 84.06% coverage, flags 15.94% bad slices (Dice 0.4994). |
| **AUDIT-05** | Phase G | CRITICAL | Remediate 9 Gate 6A methodological findings | **COMPLETE** | Zero leakage, balanced sampling in `results/gate6a_remediation_v1/`. |
| **AUDIT-06** | Phase G | HIGH | Comprehensive Independent Project Health Audit | **COMPLETE** | Final Score: **40.3 / 50.0 (80.6%)**, Grade **A-**. |
| **UI-01**    | Phase H | HIGH | Upgrade Streamlit clinical PACS dashboard | **PENDING** | Wire to ResUNet checkpoint, calibrated $T^*$, live triage flags. |
| **TEST-01**  | Phase I | CRITICAL | Gate 6B Locked Single-Pass Test Cohort Benchmark | **QUARANTINED** | Pending user authorization (155 cases, Dice, IoU, HD95). |
| **DOC-01**   | Phase J | HIGH | Rewrite Dissertation (`Final_Report_Brain_Tumor_Segmentation.docx`) | **PENDING** | 55–65 pages adhering to JIIT rubric; 7 standard chapters. |
| **VIVA-01**  | Phase J | HIGH | Construct 18-Slide Viva Defense Presentation Deck | **PENDING** | Structured slide deck matching "5-minute whiteboard test". |
| **VIVA-02**  | Phase J | HIGH | Finalize Examiner Defense Q&A Sheet | **PENDING** | Defenses for 2D vs 3D, overparameterization, triage framing. |

---

## 2. Review Gates Status

- [x] **Gate 1: Dataset Census & Baseline Freeze** — 1,251 cases verified; 197-case pilot documented.
- [x] **Gate 2: Candidate Model Pilots** — ResUNet-2D and UNetResNet34 5-epoch baselines completed.
- [x] **Gate 3: Model Portfolio Selection** — ResUNet-2D confirmed leading candidate; TransUNet/3D U-Net scoped out.
- [x] **Gate 4: Patient-Level Partitioning Sign-off** — 3-way split frozen; 155 test cases quarantined.
- [x] **Gate 5: Full-Scale Training Resource Approval** — Gate 5D orchestrator launched and completed.
- [x] **Gate 6: Full-Scale Checkpoints & Convergence Sign-off** — ResUNet-2D (Dice 0.8670) and UNetResNet34 (Dice 0.8554) verified.
- [x] **Gate 7 (6A): Temperature Calibration & Triage Sign-off** — ECE reduced by 60.1%; selective triage verified.
- [x] **Gate 7 Remediation: Independent Audit Sign-off** — All 9 issues remediated; Health Audit: 40.3/50 (80.6%).
- [ ] **Gate 8: Clinical Showcase Dashboard Sign-off** (`tools/demo_app.py` update)
- [ ] **Gate 9 (6B): Quarantined Test Set Final Evaluation Sign-off** (155 cases single pass)
- [ ] **Gate 10: Final Dissertation & Viva Defense Readiness Sign-off**
