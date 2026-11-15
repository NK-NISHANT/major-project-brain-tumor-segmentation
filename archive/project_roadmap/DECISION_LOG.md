# Architectural and Experimental Decision Log

**Project:** Brain Tumor Segmentation Using Uncertainty and Towards a Reliability-Aware AI Agent
**Root Document:** `D:\MajorProject\archive\project_roadmap\DECISION_LOG.md`
**Parent Roadmap:** `../../docs/historical/MASTER_PROJECT_ROADMAP.md`
**Tracking System:** DEC-01 to DEC-14
**Last Updated:** October 26, 2026 (Targeted Planning Corrections Applied)

---

## 1. Historical & Established Baseline Decisions

### DEC-01: 2D Axial Slices vs. Native 3D Volumetric Models
- **Decision:** Train 2D convolutional neural networks on individual axial slices ($240 \times 240 \times 4$) rather than native 3D patches ($240 \times 240 \times 155$).
- **Date:** September 2026 (Initial architecture setup)
- **Rationale:** The development machine is equipped with an NVIDIA RTX 4070 Laptop GPU with 8 GB of VRAM. Native 3D volumes require sub-volume patch cropping (e.g., $96 \times 96 \times 96$) and complex sliding-window reassembly, risking GPU out-of-memory errors and severely slowing iteration. 2D training executes in batch size 16 (or 8) with full in-plane spatial resolution, converging in 40–67 minutes per architecture.
- **Trade-off:** 2D models discard inter-slice z-axis anatomical context during individual forward passes.
- **Remediation:** Compensated post-hoc via the agent's axial-context spatial continuity policy ($z \pm k$).
- **Status:** **PERMANENT / KEPT.**

### DEC-02: 197 Extracted Cases vs. Immediate Full Archive Extraction
- **Decision:** Extract a development subset of 198 folders from the 12.27 GiB training zip file to establish baseline methodology.
- **Date:** September 2026 (Part 1 Handoff) & Reconciled October 2026
- **Rationale:** Preserved disk space and prevented long, unverified training loops while debugging pipeline mechanics, normalization, and evaluation metrics. Exactly 197 cases were complete on disk. `BraTS-GLI-00646-001` was incomplete on disk (missing `t1c` and `t2w` due to interrupted extraction), but confirmed intact in the root ZIP archive. `BraTS-GLI-01438-000` is complete on disk and archive, and is an active case in the historical 158-case training split.
- **Status:** **SUPERSEDED BY DEC-07.** The 158/39 split is frozen as the historical development benchmark. Full-scale training on all eligible cases in the archive is mandatory under Phase E, following Phase A census.

### DEC-03: Softmax Entropy as Primary Uncertainty Metric
- **Decision:** Use normalized Shannon entropy of softmax output probabilities ($-\sum p_c \log p_c / \log 4$) as the voxel-wise predictive uncertainty measure.
- **Date:** September 2026
- **Rationale:** Computable from a single deterministic forward pass without requiring expensive Monte Carlo dropout iterations or deep ensemble training.
- **Limitation Discovered:** When a model predicts zero foreground voxels, foreground entropy is undefined (`None`).
- **Status:** **REFINED IN DEC-12.** Foreground entropy remains valid for non-empty slices; whole-brain non-zero voxel entropy ($H_{\text{brain}}$) is introduced to detect ambiguous near-threshold boundary lesions.

### DEC-04: Spatial Continuity Fallback Gating over Pure Entropy Fallback
- **Decision:** Condition fallback switching primarily on primary model empty-prediction status combined with axial neighbor foreground support ($z \pm k$).
- **Date:** October 2026 (Milestones 4–5)
- **Rationale:** M3 unconstrained fallback (switching to U-Net whenever U-Net++ predicted empty) recovered 213 misses, but generated 762 false positive slices on healthy tissue (specificity dropped to 67.90%). In M1/M2 offline tests, lowest uncertainty alone agreed with the oracle best model only 48.35% of the time. Requiring at least 1 adjacent slice with foreground tumor verified spatial continuity and suppressed 635 false alarms.
- **Status:** **VERIFIED & KEPT.**

### DEC-05: Option A ($k=2$, Default) vs. Option B ($k=1$, Aggressive)
- **Decision:** Establish Option A ($k=2$, `fallback_when_context_unavailable=False`) as the canonical project default, retaining Option B ($k=1$) as an exploratory sensitivity trial.
- **Date:** October 2026 (Milestone 7C/7D)
- **Rationale:** Milestone 7 full-volume evaluation revealed that widening the window to $k=1$ (Option B) yielded 27 additional boundary recoveries across 19 patients, but introduced 51 additional false alarm slices across 21 patients. This established a 1 : 1.89 penalty ratio. For standard automated screening, false alarms on healthy brain tissue are disruptive.
- **Status:** **VERIFIED & KEPT.**

### DEC-06: Scoping Out LLM and MCP Tools from Core Segmentation Routing
- **Decision:** Exclude Large Language Models (LLMs) and Model Context Protocol (MCP) tool-calling from real-time segmentation decision loops.
- **Date:** September 2026 (Project Pack) & Reaffirmed October 2026
- **Rationale:** Medical image segmentation requires deterministic pixel-level mask transformations and mathematical guarantees. LLMs introduce non-deterministic hallucinations, latency overhead (>1,000 ms vs. 14.9 ms for CNN inference), and unverified medical reasoning.
- **Status:** **PERMANENT / KEPT.**

---

## 2. Reconciled Decisions (October 26, 2026)

### DEC-07: Dataset Scaling Strategy & Mandatory Full-Data Training
- **Decision:** Mandate training selected final models on all verified eligible cases in the final training partition (~875 cases `[ESTIMATE]`, drawn from the 1,251 cases in `ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip`).
- **Date:** October 26, 2026
- **Rationale:** Scaling training from the 158-case pilot to the full available dataset is a core deliverable. Drive D has 75.15 GB free disk space (`[MEASURED]`). Extracting remaining cases is projected to take ~11.3 GB `[ESTIMATE]`, leaving >61 GB free `[ESTIMATE]`. Full training is projected at ~23.8 GPU-hours `[ESTIMATE]` across 5 models, executed sequentially under Stage 4.
- **Status:** **MANDATORY (Phase E, blocked until Gates 1–5 pass).**

### DEC-08: Architectural Expansion Selection
- **Decision:** Implement and benchmark ResUNet-2D, U-Net with ResNet-34 encoder, TransUNet (CNN-Transformer hybrid), and conduct a documented 3D U-Net feasibility assessment.
- **Date:** October 26, 2026
- **Rationale:** Broadens the model comparison beyond basic CNNs. ResUNet stabilizes gradient flow for boundary sensitivity; TransUNet evaluates self-attention for global spatial context; 3D U-Net feasibility rigorously documents VRAM limits on the 8 GB RTX 4070.
- **Status:** **MANDATORY (Phase B).**

### DEC-09: Demonstration Framework Selection (Full-Stack Web App)
- **Decision:** Mandate the development of a decoupled full-stack clinical demonstration web application (React 18 + Vite + TypeScript frontend, FastAPI backend, PyTorch inference serving). Preserve the existing Streamlit app (`tools/demo_app.py`) as the Phase 3 research prototype.
- **Date:** October 26, 2026
- **Rationale:** A decoupled web architecture provides professional medical UI responsiveness, smooth slice scrubbing, independent model/agent updates, and meets final demonstration requirements.
- **Status:** **MANDATORY (Phase H).**

### DEC-10: Programmatic Thesis Compilation Workflow
- **Decision:** Overhaul `tools/build_final_report.py` to automatically compile `Final_Report_Brain_Tumor_Segmentation.docx`, integrating final benchmark tables, statistical tests, and publication figures directly from source artifacts into a complete academic dissertation conforming strictly to JIIT guidelines without placeholder stubs.
- **Date:** October 26, 2026
- **Rationale:** Eliminates manual copy-paste errors, guarantees numerical synchronization with evaluation JSON files, and ensures 100% compliance with B.Tech dissertation guidelines.
- **Status:** **MANDATORY (Phase J).**

### DEC-11: 3-Way Leakage-Safe Patient-Level Partitioning Protocol
- **Decision:** Partition the eligible archive cases into a 3-way split: Training (~70%, ~875 cases `[ESTIMATE]`), Validation (~15%, ~188 cases `[ESTIMATE]`), and Held-Out Pristine Test (~15%, ~188 cases `[ESTIMATE]`) using seed 42.
- **Date:** October 26, 2026
- **Rationale:** The historical 39 validation cases were repeatedly used for hyperparameter selection and agent tuning and cannot be relabeled as a pristine test set. A truly quarantined test set guarantees zero-leakage, publication-grade evaluation.
- **Status:** **MANDATORY (Phase D).**

### DEC-12: Multi-Signal Reliability Agent Formulation
- **Decision:** Expand agent routing beyond the 2-model context-based fallback prototype to evaluate multiple signals: primary empty prediction, whole-brain background ambiguity ($H_{\text{brain}}$), spatial axial continuity ($z \pm k$), candidate model disagreement, and axial slice position index.
- **Date:** October 26, 2026
- **Rationale:** Addresses the limitation where foreground entropy is undefined for empty predictions, enabling uncertainty-guided boundary recovery without elevating false alarms.
- **Status:** **MANDATORY (Phase F).**

### DEC-13: Conditional Mobile Application Feasibility Assessment
- **Decision:** Subject mobile application development to a formal feasibility review gate after the full-stack web application is operational.
- **Date:** October 26, 2026
- **Rationale:** A mobile app should only be implemented if the examination rubric explicitly demands it or native functionality is justified. If the React web UI is responsive across mobile viewports, building a redundant native app wastes engineering resources.
- **Status:** **CONDITIONAL GATE (Phase I / Gate 11).**

### DEC-14: Mandatory Staged-Development Policy & Governance Protocol
- **Decision:** Enforce a strict 5-stage scaling policy, 12 explicit review gates, single-writer file ownership, zero concurrent GPU training jobs, and independent verification before gate sign-offs.
- **Date:** October 26, 2026
- **Rationale:** Protects laptop hardware from thermal throttling, prevents premature compute expenditure on unverified models, and ensures zero data leakage.
- **Status:** **MANDATORY OPERATING POLICY.**

### DEC-15: Pilot Subject-Disjoint Isolation Policy
- **Decision:** Mandate exclusion of longitudinal companion scans (`-000` vs `-001` vs `-002`) belonging to validation subjects from training (`exclude_val_subject_companions=True`).
- **Date:** October 26, 2026
- **Rationale:** Prevents patient-level leakage discovered in historical random splits (where 10 companion scans in train contaminated 25.64% of validation cases). Guarantees true subject-disjoint out-of-distribution evaluation for all subsequent pilots.
- **Status:** **MANDATORY POLICY.**

### DEC-16: Gate 2 Candidate Model Pilot Verdicts & Stabilization Prerequisites
- **Decision:** Formally approve ResUNet-2D (Full GO) and UNetResNet34 (Conditional GO). UNetResNet34 sets the project peak Mean Dice benchmark at 0.8388 (NCR: 0.8280) and ResUNet sets the efficiency benchmark (6.50 min/ep, 0.7954 Dice, ET: 0.8697). Full-scale training of UNetResNet34 is conditioned on gradient clipping ($\le 1.0$) and learning rate warmup/decay to prevent transient class collapse.
- **Date:** October 26, 2026
- **Rationale:** Supported by independent audit and empirical evidence. Eliminates slow/sub-par architectures (UNet++, Attention U-Net) from primary candidate selection.
- **Status:** **VERIFIED & ADOPTED (Gate 2 Passed).**
