---
title: "Master Project Blueprint v2.0"
subtitle: "Reliability-Aware Brain Tumor Segmentation Framework"
author: "B.Tech CSE Project Team — JIIT Noida"
date: "10 November 2026"
lang: en-US
---

# MASTER PROJECT BLUEPRINT v2.0

**Brain Tumor Segmentation — A Reliability-Aware Framework Using Multi-Parametric MRI, Predictive Uncertainty and Spatial Context**

> **Purpose:** The single operational reference for the research question, evidence-backed project state, model portfolio, reliability agent, data governance, software, environment, Git workflow, remaining milestones, and dissertation/viva delivery.

| Document control | Value |
|---|---|
| Version | **2.0 — Evidence-reconciled operating blueprint** |
| Effective date | 10 November 2026 |
| Workspace | `D:\MajorProject` |
| Team | Three B.Tech CSE students, JIIT Noida |
| Project domain | Medical image computing / AI / machine learning |
| Authority | Replaces V1.1 as the project operating roadmap once placed in the repository; not a substitute for supervisor approval or raw evidence |
| Research baseline | November 10, 2026 corrected research outputs; several provenance checks remain open |
| Primary environment | `D:\MajorProject\.venv\Scripts\python.exe` |
| Git state | User reports the remote connection was disconnected; actual `.git`, branch and remote state are **not yet audited** |
| Final test evaluation | **Blocked** pending explicit authorization after model, policy, software and evaluation freeze |

**Important:** This document was prepared from the supplied V1.1 roadmap and the November 10, 2026 research baseline files. It is not a live inspection of `D:\MajorProject`. File existence, Git status, dependencies, code behavior and artifact hashes must still be checked locally during the read-only repository audit.

## Contents

1. Executive project charter
2. Authority, evidence rules and document hierarchy
3. Current project state — 10 November 2026
4. Dataset and cohort accounting
5. Data integrity and preprocessing protocol
6. Model portfolio and training evidence
7. Metrics, evaluation and statistics
8. Predictive uncertainty and calibration
9. Reliability-aware agent architecture
10. Missing-modality robustness study
11. Inter-model prediction disagreement
12. Application and software delivery
13. Environment, virtual environment and reproducibility
14. Git and repository preservation
15. Repository structure and cleanup policy
16. Model and agent extension roadmap
17. Project execution roadmap and gates
18. Dissertation, presentation and viva requirements
19. Risk register and control decisions
20. Troubleshooting workflow
21. Working protocol for ChatGPT, Antigravity and coding agents
22. Supporting records and expected root-level companion files
23. Immediate next actions
24. Source register and evidence map
25. Final operating statement

---

## 1. Executive project charter

### 1.1 Project in one sentence

Build and evaluate a reproducible brain-tumor MRI segmentation pipeline that produces candidate masks, measures prediction reliability using calibrated uncertainty and complementary signals, and either accepts a candidate output, uses a justified fallback, or flags the result for human review.

### 1.2 Recommended working title

**Development and Evaluation of a Reliability-Aware Brain Tumor Segmentation Framework Using Multi-Parametric MRI**

Optional subtitle: *Predictive uncertainty, spatial context and explainable model selection for research decision support.*

The final title remains subject to supervisor confirmation and the official JIIT synopsis/rubric. Avoid putting “autonomous clinical agent” or “clinical-grade” in the title.

### 1.3 Problem statement

Multi-parametric MRI glioma segmentation is challenging because tumor size, shape, location, tissue appearance and scan conditions vary. A segmentation model can produce a plausible-looking mask while missing a clinically relevant compartment. A single aggregate Dice score does not indicate when a particular prediction may be unreliable. This project studies segmentation quality together with measurable reliability signals and explicit accept/fallback/review behavior.

### 1.4 Aim

Develop and evaluate a reproducible multi-modal MRI segmentation framework and a transparent reliability-aware decision component. The system should make its decisions traceable and be able to abstain or recommend review when its signals indicate an unreliable output.

### 1.5 Research questions

1. **RQ1 — Segmentation:** Under a clearly described patient/subject-level protocol, how do the selected segmentation architectures compare on WT, TC and ET quality, complete misses, false alarms, boundaries and compute cost?
2. **RQ2 — Predictive reliability:** How strongly do predictive uncertainty and calibration measures relate to segmentation quality and identify low-quality outputs?
3. **RQ3 — Agent decisions:** What does a deterministic multi-signal policy add beyond a fixed primary model, a single-signal entropy rule and a simple fallback baseline? Does it improve segmentation, selective triage, or both?
4. **RQ4 — Input robustness:** How do model quality and agent decisions change when MRI modalities are omitted or input quality is degraded? Which failures are detected, and which remain silent?

### 1.6 What counts as the project contribution

- A validated four-channel data and preprocessing pipeline with explicit split governance.
- A documented comparison of five historical architecture families, with a deeper full-training comparison of the strongest two available checkpoints.
- A calibration and uncertainty analysis with defined limitations.
- A deterministic reliability policy combining uncertainty, empty-prediction handling, axial context and a secondary-model fallback.
- A controlled missing-modality stress study that includes negative findings and demonstrates where entropy-only reliability checks fail.
- An explainable local research demo and reproducible dissertation evidence.

**Not the contribution:** merely increasing the number of models, wrapping the model in an LLM, claiming better Dice because ECE improved, or presenting an academic prototype as a clinical product.

---

## 2. Authority, evidence rules and document hierarchy

### 2.1 Source-of-truth hierarchy

When records conflict, use this order and record the resolution in the decision log:

1. Supervisor-approved synopsis, written supervisor instructions and official JIIT assessment rubric.
2. Raw, reproducible experiment evidence: manifests, configs, code revision, logs, checkpoints, hashes, per-case/per-subject metrics and test output.
3. Corrected November 10 baseline artifacts in `results/final_research_baseline/`, including the machine-readable JSON/CSVs and the reports that explain them.
4. This `MASTER_PROJECT_BLUEPRINT_v2.0.md` for scope, operating rules, current decisions and project sequence.
5. Supporting `PROJECT_STATUS.md`, model/experiment registry, decision log, evidence index and changelog once their contents have been locally audited.
6. Older roadmaps, screenshots, presentations, chat transcripts and superseded reports as historical context only.

A report saying that a task completed is not by itself proof that the current repository contains the expected artifact. A file name is not proof of its contents. A passing test count is not proof of test coverage. A validation score is not clinical validation.

### 2.2 Status vocabulary

| Status | Meaning | Permitted wording |
|---|---|---|
| **VERIFIED** | Directly checked against the named source artifact or repeatable command in the current audit | “Verified in [artifact/run]” |
| **REPORTED** | Stated by an execution report but not independently rechecked in the current repository | “Reported by the November 10 run” |
| **CONDITIONAL** | Evidence exists but a methodological/provenance limitation remains | “Provisional / conditional finding” |
| **PLANNED** | Intended future work only | “Planned; not implemented” |
| **PROPOSED** | Candidate idea awaiting a decision | “Proposed candidate” |
| **BLOCKED** | Must not proceed until a named approval or dependency is met | “Blocked pending…” |
| **REJECTED** | Considered and not executed or not accepted | “Preflight rejected / not evaluated” |
| **EXPLORATORY** | Useful analysis, but threshold/protocol was post-hoc or not independently validated | “Exploratory association” |

### 2.3 Non-negotiable operating rules

- Never access, extract, load, infer on, tune against or debug with held-out test images or labels before explicit authorization for the locked final test.
- Never overwrite historical checkpoints, manifests, result folders, audit outputs or thesis versions. Write new versioned outputs.
- Never claim an experiment is completed because a script was created or launched; require exit code, logs and output coverage.
- Never add an architecture, dataset, framework, app stack or agent framework without a research reason, resource estimate and approval.
- Use one GPU-heavy job at a time on the RTX 4070 Laptop GPU.
- Inspect diffs and run relevant tests after code changes. Team members must understand and be able to explain the code in a viva.
- Do not modify `.venv` or Git metadata during documentation/repository-inventory work.
- No destructive cleanup or bulk file moves without a reviewed mapping and explicit user approval.

> **Anti-drift rule:** Every new task must improve segmentation evidence, reliability/uncertainty evidence, agent decision quality, robustness, reproducibility or the demonstrable research product. Otherwise it is optional and should be dropped when time or compute becomes constrained.

---

## 3. Current project state — 10 November 2026

This is the evidence-based working status from the supplied reports. It is not a claim that a local repository audit has already been completed.

| Area | Current documented state | Operating status |
|---|---|---|
| Research scope | Multi-parametric GLI MRI segmentation with reliability-aware triage | Established |
| Dataset | Reported 1,251 cases / 1,133 subjects; edition label conflicts between supplied records | Conditional — verify archive provenance |
| Main partitions | 721 development-train; 155 validation; 155 held-out test; 220 historical cases | Reported — manifest audit still required |
| Calibration/evaluation split | 50 calibration cases / 47 subjects; 105 evaluation cases / 96 subjects; zero subject overlap reported | Reported remediation; preserve checkpoint-selection caveat |
| Main architecture portfolio | Five architecture families, eleven reported training runs | Established historical portfolio; not eleven architectures |
| Deeply trained candidates | ResUNet-2D and UNetResNet34, 15-epoch checkpoints | Reported and hash-documented |
| Primary model | ResUNet-2D | Current benchmark champion |
| Secondary model | UNetResNet34 | Secondary comparator / fallback candidate |
| Calibration | Temperature scaling reported for both current checkpoints | Conditional on the exact calibration artifact and ECE definition |
| Gate 6A | Remediation and full-policy re-evaluation reported complete | Conditional scientific baseline; preserve run distinctions |
| Missing-modality study | C0–C4; entropy-only and full Gate 6A policies compared | Evidence-backed validation stress study |
| Inter-model disagreement | Post-hoc 10% threshold, 94/11 case split | Exploratory only |
| SegResNet-3D | 5 epochs, 10 development cases, no checkpoint persisted | Feasibility pilot only |
| Promptable decoder | Custom simulation; no official LiteMedSAM weights loaded | Simulation only, not a pretrained model benchmark |
| nnU-Net v2 | Planner/preprocessing not executed | Preflight rejected; not trained/evaluated |
| MedSAM2 | No evaluation evidence supplied | Not evaluated |
| Additional 2–3 model candidates | Team reports candidates are nearly finalized, but names/configs/evidence are not in the supplied source pack | Proposed; identify before admission |
| Application | Streamlit headless startup and HTTP 200 health endpoint reported | Startup verified; browser interaction not verified |
| Tests | 109 passing tests reported: 38 in `tests/`, 71 in `src/agent/` | Reported; rerun only after scoped audit/approval |
| Held-out test | Manifest digest unchanged; no test evaluation evidenced in records reviewed | Quarantined; Gate 6B unauthorized |
| Git | Remote connection reportedly disconnected | Actual `.git`, branch and remote state unknown; audit read-only |
| Virtual environment | `.venv` is the active project environment | Preserve exactly; do not recreate or mutate |

### 3.1 What is frozen and what is still open

**Frozen as the current reference baseline:** the two 15-epoch checkpoints, the reported 105-case validation-subset comparison, the corrected missing-modality study outputs and the test-cohort quarantine rule. Preserve these artifacts.

**Not frozen as final project scope:** admission of additional models, final agent policy after any new models, upstream modality-presence validation, full application design, final 941-case training eligibility and the locked held-out test.

Do not silently replace the current primary/secondary results with future model results. Keep a versioned baseline and report new extensions separately before any portfolio decision.

---

## 4. Dataset and cohort accounting

### 4.1 Modalities, labels and derived regions

The documented input order is:

1. `T1n` — native T1.
2. `T1c` — contrast-enhanced T1.
3. `T2w` — T2-weighted.
4. `T2f` — T2-FLAIR.

The project records describe segmentation labels as background `0`, NCR `1`, edema `2`, enhancing tumor `3`. The derived evaluation regions are:

- **WT (Whole Tumor):** NCR + edema + ET.
- **TC (Tumor Core):** NCR + ET.
- **ET (Enhancing Tumor):** ET only.

Confirm the official archive label mapping and local loader implementation before final thesis publication. Never infer channel order from file sorting.

### 4.2 Dataset edition discrepancy — must be resolved locally

V1.1 names the source **BraTS 2023 Adult Glioma (GLI)**, while the supplied `FINAL_EVIDENCE_AND_LIMITATIONS.md` calls it **BraTS 2024 GLI**. The current supplied documents do not conclusively settle this discrepancy. Until archive metadata, original download/source record and manifest provenance are inspected, use the neutral term **“BraTS Adult Glioma (GLI) archive”** in working material. Do not silently choose 2023 or 2024 in the final dissertation.

### 4.3 Reported cohort ledger

| Cohort | Cases | Unique subjects | Role / restriction |
|---|---:|---:|---|
| Development training | 721 | 667 | Used for the reported 15-epoch model runs |
| Validation pool | 155 | 143 | Later divided into calibration and internal evaluation subsets |
| Calibration subset | 50 | 47 | Used for temperature fitting; subject-disjoint from 105-case evaluation subset |
| Internal evaluation subset | 105 | 96 | Used for reported 105-case validation-subset results |
| Held-out test | 155 | 143 | Sealed; no access until explicit Gate 6B approval |
| Historical development pool | 220 | 180 | Previously exposed; may only enter final training after eligibility review and model/policy freeze |
| Reported archive total | 1,251 | 1,133 | 1,031 previously unseen + 220 historical cases |
| Proposed final training | 941 | 847 | 721 + 220; proposed only, not the training set used for current 15-epoch checkpoints |

The reported arithmetic is 721 + 155 + 155 = 1,031 previously unseen cases, and 1,031 + 220 = 1,251 total cases. Subject counts must be confirmed from the parent-subject ledger; case counts alone do not establish independence.

### 4.4 Manifest identities reported in the project history

| Manifest | SHA-256 | Status |
|---|---|---|
| Development training (721 cases) | `42b7bf303319266bfa692e0eb41fddd425de2b845195f07b9b9105ed1875b015` | Reported historical digest; recheck locally |
| Validation (155 cases) | `20accad8a9c2717f50db4a582c749bf50f1316ae03e9616cf589d7dbf44c852c` | Reported historical digest; recheck locally |
| Proposed final training (941 cases) | `f1171290d692fab54c6d2e353cd73ff5d53c0055ee3ff7a36663a15fe9be8409` | Proposed manifest; do not imply it has been used to train current checkpoints |
| Held-out test (155 cases) | `694faaa4ff100066d67aa5c87c545a709e4eed43a61625d3bea9f43c50d6dfb7` | Reported test manifest digest; preserve and do not access test data |

These digests must be recalculated against the exact local files during the read-only audit. A matching manifest hash establishes file identity, not proof that the images were never accessed historically.

### 4.5 Calibration/evaluation dependence caveat

The remediation reports 50 calibration cases / 47 subjects and 105 evaluation cases / 96 subjects with zero subject overlap. However, best checkpoints were selected using the full 155-case validation cohort before it was divided into calibration and evaluation subsets. Therefore, the 105-case evaluation subset is **disjoint from calibration fitting at subject level, but not fully independent of earlier checkpoint selection**. This distinction must appear in the dissertation methods and limitations.

### 4.6 Test quarantine rule

Use this exact qualified statement:

> “The held-out test cohort remains quarantined, its manifest hash is unchanged, and no test evaluation is evidenced in the documented execution records reviewed.”

Do not say “100% proven untouched” or call the 155 cases 155 subjects. The reported test set is 155 cases from 143 unique subjects. Gate 6B is not authorized by this blueprint.

---

## 5. Data integrity and preprocessing protocol

Keep the existing pipeline and its versioned configuration unchanged while auditing. The following requirements are the standard for any future modification or new candidate model.

| Area | Required rule / evidence |
|---|---|
| File validation | Check that all required modalities and segmentation files exist, are readable and belong to the same case |
| Geometry | Validate shape, affine, orientation, spacing, dimensionality and modality/mask alignment |
| Labels | Validate allowed label IDs and derive WT/TC/ET in one tested function |
| Intensity | Record normalization method and handle NaN/Inf, all-zero volumes and abnormal ranges explicitly |
| Resampling | Use continuous interpolation for MRI intensities and nearest-neighbour for discrete segmentation labels |
| Channel order | Enforce `[T1n, T1c, T2w, T2f]` explicitly |
| 2D extraction | Document axial ordering, slice indexing, tumour-positive slices, empty-slice behavior and volume aggregation |
| Cache identity | Bind cache keys to source case, preprocessing version, manifest, model/checkpoint and relevant code/config hashes |
| Visual QA | Retain representative modality/mask overlays and sanity checks for permitted development/validation cases |

Never silently drop empty slices from evaluation. Empty ground-truth slices and empty predictions must have explicitly defined metric and agent behavior.

---

## 6. Model portfolio and training evidence

### 6.1 Five established architecture families

| Architecture | Trainable parameters | Documented role | Evidence/status |
|---|---:|---|---|
| U-Net | 7,763,428 | Baseline encoder-decoder | Historical 5-epoch baseline |
| U-Net++ | 9,160,068 | Nested/dense skip-pathway comparison | Historical 5-epoch baseline |
| Attention U-Net | 7,852,160 | Attention-gated skip-feature comparison | Historical 5-epoch baseline |
| ResUNet-2D | 8,114,604 | Residual 2D segmentation model | 5-epoch pilot and 15-epoch extended run; current primary champion |
| UNetResNet34 | 24,327,524 trainable; +18,988 non-trainable BatchNorm buffers | Transfer-learning encoder comparison | 5-epoch pilot/ablations and 15-epoch extended run; current secondary model |

UNetResNet34 has 24,346,512 tensor elements when the 18,988 non-trainable buffers are included. Do not report the historical erroneous trainable-parameter figure of 24,435,556.

### 6.2 Historical baseline metrics — not a single fair ranking

The earlier 158-train / 39-validation comparison reported:

| Architecture | Mean Dice | Mean IoU |
|---|---:|---:|
| U-Net | 0.7711 | 0.6329 |
| U-Net++ | 0.7937 | 0.6582 |
| Attention U-Net | 0.7291 | 0.5886 |

These are historical results from a protocol with longitudinal subject-overlap concerns. Keep them as historical baselines; do not merge them into a ranking against later 15-epoch results.

### 6.3 Extended 15-epoch runs

| Model | Training cases / subjects | Best epoch | Reported best mean Dice on 155-case validation pool | Runtime |
|---|---:|---:|---:|---:|
| ResUNet-2D | 721 / 667 | 13 | approximately 0.8670 | approximately 8.09 h |
| UNetResNet34 | 721 / 667 | 15 | approximately 0.8554 | approximately 10.80 h |

The selected checkpoints were later re-evaluated on the 105-case internal evaluation subset. This subset result is reported separately in Section 7. The 5-epoch and 15-epoch scores are not evidence of an inherently superior architecture unless split, compute budget, training protocol and evaluation are matched.

### 6.4 Frozen checkpoint provenance reported on November 10

| Model | Checkpoint | SHA-256 | File size |
|---|---|---|---:|
| ResUNet-2D, epoch 13 | `results/full_training_15ep/resunet_2d/best_model.pt` | `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d` | 97,486,845 bytes |
| UNetResNet34, epoch 15 | `results/full_training_15ep/unet_resnet34/best_model.pt` | `891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d` | 292,222,285 bytes |

Recalculate both hashes against the live files before any future model integration or final submission. Do not overwrite these checkpoints.

### 6.5 Current 105-case internal evaluation comparison

The following values are from the supplied reconciled model-comparison report and refer to 105 cases / 96 subjects, not the full 155-case selection cohort.

| Metric | ResUNet-2D | UNetResNet34 | Interpretation |
|---|---:|---:|---|
| Overall mean Dice | **0.8641** [95% CI 0.827–0.893] | 0.8535 [0.814–0.889] | ResUNet higher by 0.0106 |
| WT Dice | **0.9039** [0.877–0.924] | 0.8843 [0.855–0.909] | ResUNet higher |
| TC Dice | **0.8597** [0.808–0.899] | 0.8423 [0.788–0.883] | ResUNet higher |
| ET Dice | 0.8286 [0.779–0.868] | **0.8339** [0.783–0.873] | UNetResNet34 slightly higher |
| WT median HD95 | **4.12 mm** | 5.00 mm | ResUNet lower |
| TC median HD95 | 3.16 mm | **2.64 mm** | UNetResNet34 lower on valid non-empty cases |
| ET median HD95 | 2.00 mm | **1.73 mm** | UNetResNet34 lower on valid non-empty cases |
| Complete misses WT / TC / ET | **0 / 0 / 0** | 0 / 2 / 1 | ResUNet had no complete compartment misses in this cohort |
| False alarms WT / TC / ET | 0 / 0 / 1 | **0 / 0 / 0** | UNetResNet34 had no complete false alarms in this cohort |

HD95 uses a bounded empty-mask penalty of 373.12 mm for a one-empty comparison; both-empty is 0 mm. Report valid-case summaries, bounded summaries, complete misses and false alarms separately. Never mix raw valid-case means with bounded means without a label. No “sub-millimeter boundary accuracy” claim is supported; the medians range from 1.73 mm to 5.00 mm.

### 6.6 Primary/secondary decision

**ResUNet-2D is the current primary benchmark model.** It leads overall mean Dice, WT and TC Dice, and had zero complete WT/TC/ET misses in the 105-case evaluation. **UNetResNet34 is the secondary comparator and fallback candidate.** It is slightly stronger on ET Dice and has lower median HD95 for valid TC/ET cases, but has two complete TC misses and one ET miss in this evaluation cohort.

This is a research selection on the available validation evidence, not a clinical model approval. The current agent did not improve aggregate Dice over standalone ResUNet in the nominal B2 re-evaluation.

### 6.7 Additional 2–3 model candidates — admission is pending

The team reports that 2–3 additional models are nearly finalized. Their exact names, code locations, configurations and evidence were not present in the supplied V1.1 + November 10 final baseline package. Therefore they are **not yet part of the approved portfolio in this blueprint**.

For each candidate, create a registry entry with:

- Exact architecture and implementation module.
- Why it is needed; the research question it answers.
- Whether it offers a genuinely different inductive bias or operating trade-off.
- Status: proposed, implemented, trained, checkpointed, evaluated, calibrated, agent-integrated.
- Training cohort/subject manifest, epochs, seed, configuration, command, logs and environment.
- Checkpoint path, SHA-256 and parameter count.
- Comparable WT/TC/ET Dice, HD95, misses/false alarms, calibration, latency and peak memory where applicable.
- Known limitations and a decision: admit, retain as exploratory, defer or reject.

**Admission rule:** Do not add a model just to reach 7 or 8. It must be feasible on the 8 GB GPU, have a clear purpose, and be evaluated on a protocol that allows an honest comparison. No candidate may access the sealed test set.

### 6.8 Exploratory architecture work — do not mislabel

| Experiment | Correct status | What may be claimed |
|---|---|---|
| SegResNet-3D | 5-epoch, 10-development-case, 50-step feasibility pilot; checkpoint not persisted | MONAI 3D pipeline feasibility on the target GPU; WT Dice 0.2658 and HD95 88.87 mm are underfit pilot metrics, not a competitive benchmark |
| Promptable decoder / “LiteMedSAM” exploration | Custom promptable-decoder simulation; official pretrained LiteMedSAM weights were not loaded | Exploratory peak-tumor axial-slice WT simulation only; not a LiteMedSAM benchmark |
| Genuine nnU-Net v2 | Preflight rejected; official planning/preprocessing was never executed | No nnU-Net model was trained or evaluated; resource estimates were theoretical |
| MedSAM2 | Not evaluated in supplied evidence | No performance or feasibility claim |

The prompt simulation used GT-informed peak-tumor slices, so it is not a full-volume or fully automatic comparison. Report that restriction prominently if included in the dissertation.

---

## 7. Metrics, evaluation and statistics

### 7.1 Segmentation and failure metrics

Report at minimum:

- WT, TC and ET Dice separately.
- Macro foreground / compartment mean Dice, with its aggregation definition.
- Case-level and subject-level results separately when longitudinal scans exist.
- Complete misses when ground truth contains tumor but the prediction is empty.
- False alarms on empty ground-truth compartments/slices.
- HD95 with explicit empty-mask handling, valid-case and bounded summaries.
- Compute cost: parameters, inference latency, peak allocated memory and runtime conditions.

### 7.2 Calibration and reliability metrics

Use calibration metrics such as ECE and NLL only with the exact bins, class/voxel mask, sampling, temperature and cohort definition recorded. A temperature-scaled softmax can improve calibration without changing argmax segmentation. Therefore calibration improvements do not automatically imply Dice improvement.

**Metric reconciliation hold:** the latest supplied comparison reports post-calibration ECE of 0.0163 for ResUNet and 0.0194 for UNetResNet34, while earlier research notes cited a different ResUNet calibration value (0.0344). Confirm the exact calibration artifact, mask/voxel sampling and ECE binning before making the ECE number a dissertation headline. The values are reported in the latest document but remain subject to provenance reconciliation.

### 7.3 Statistical rules

- Treat parent subject—not slice or voxel—as the independent unit for inferential comparisons where repeated scans exist.
- Use paired subject-level tests where appropriate; report test statistic, sample count and raw/adjusted p-values.
- Correct for multiple comparisons when specified (the missing-modality summary uses Bonferroni adjustment).
- Use subject-cluster bootstrap confidence intervals, not slice-wise resampling that treats correlated slices as independent patients.
- For every percentage, identify the numerator, denominator and unit: slices, foreground slices, cases, subjects or voxels.
- Keep calibration fitting separate from evaluation and never tune a threshold on the test set.

### 7.4 Required metric disclosure

Every result table/figure must state the cohort, case count, unique subject count, aggregation method, unit, missing/empty-mask policy, checkpoint identity and confidence-interval method. Do not combine case means with subject means in one unlabeled “mean Dice” column.

---

## 8. Predictive uncertainty and calibration

### 8.1 What is implemented/reported

- The project computes normalized predictive entropy from four-class softmax probabilities.
- Temperature scaling is reported for the current candidate checkpoints.
- ResUNet-2D calibration temperature: **T* = 2.880783**.
- UNetResNet34 calibration temperature: **T* = 2.626696**.
- Calibration used the 50-case / 47-subject calibration subset, according to the remediation report.
- The current foreground-slice threshold is **tau = 0.35** for the ResUNet primary policy; exact model-specific use must follow the policy configuration and calibration artifact.

### 8.2 Interpretation

Entropy is a useful uncertainty signal, not a truth detector. It can correlate with low-quality predictions in validation data, but low entropy does not prove correctness. It is undefined or misleading for empty foreground predictions unless the policy handles those cases explicitly.

### 8.3 Historical uncertainty finding

An earlier U-Net analysis reported an approximate Spearman correlation of -0.6542 between predictive entropy and Dice and ROC-AUC of approximately 0.8059 for identifying slices with Dice below 0.50. Treat these as historical exploratory validation findings, not universal thresholds or clinical operating characteristics.

### 8.4 Current limitation

The missing-T1c experiment demonstrates a critical blind spot: the network can continue producing confident, non-empty Whole Tumor predictions while missing much of the enhancing tumor. Entropy alone cannot reliably detect that the input contrast channel is absent. A deterministic input-integrity/modality-presence gate is therefore a required upstream design component.

---

## 9. Reliability-aware agent architecture

### 9.1 Current conceptual flow

`MRI INPUT -> INPUT VALIDATION -> PRIMARY MODEL -> RELIABILITY SIGNALS -> POLICY -> ACCEPT / FALLBACK / REVIEW / ABSTAIN`

The current documented research implementation is a deterministic, explainable policy. It is not an LLM agent and does not require LangChain, RAG or MCP in the pixel-level segmentation loop.

### 9.2 Current main components

| Component | Responsibility | Status / limitation |
|---|---|---|
| Input/data validation | Validate modality presence, shapes, basic quality and expected channel order | Upstream modality-presence gate is required; implementation status must be checked and treated as planned until verified |
| Primary predictor | ResUNet-2D segmentation and probabilities | Current primary checkpoint |
| Secondary predictor | UNetResNet34 fallback candidate | Model is available as secondary checkpoint; fallback decision should be justified by policy evidence |
| Calibration | Apply model-specific temperature scaling | Reported for current checkpoints |
| Entropy signal | Quantify predictive distribution concentration | Does not detect every missing-input failure |
| Empty-prediction guard | Prevent a zero-foreground output from being silently assigned zero entropy and accepted | Present in full Gate 6A policy re-evaluation |
| Axial context | Use neighbouring slices to support or suppress fallback/review decisions | Must retain the exact context radius and rules in run metadata |
| Model disagreement | Compare predictions from candidate architectures | Exploratory signal, not epistemic uncertainty |
| Policy/action trace | Emit accept, fallback, review or abstain with reason | Deterministic and auditable |

### 9.3 Nominal C0 Gate 6A re-evaluation

The new full-policy B2 run and the earlier Gate 6A remediation run are distinct execution records and must remain distinct.

| Evaluation record | Retained foreground slices | Referred foreground slices | Coverage / referral |
|---|---:|---:|---|
| Earlier Gate 6A remediation record | 5,505 / 6,549 | 1,044 / 6,549 | 84.06% retained / 15.94% referred |
| New B2 single-pass re-evaluation, C0 | 5,491 / 6,549 | 1,058 / 6,549 | 83.84% retained / 16.16% referred |

The new B2 run reports 54 fallback switches among foreground slices. Its mean agent compartment Dice is 0.863979 versus 0.864063 for standalone ResUNet-2D. This does **not** establish a segmentation Dice improvement. The earlier 84.06% figure must not be presented as the new B2 result.

### 9.4 Required action trace

For every action, retain case/slice ID, model and checkpoint hash, input validation flags, calibrated entropy definition/value, empty-prediction state, axial-context evidence, fallback model/action, thresholds, final mask source, reason code, latency and error status. The runtime agent must not access ground truth; oracle comparisons are offline analyses only.

### 9.5 Minimum safe policy behavior

- Missing required modality or malformed channel stack: stop inference and report an input-integrity error; do not attempt to “guess” the missing channel.
- Empty predicted foreground: explicit high-risk branch; do not assign synthetic entropy 0 and accept.
- High calibrated uncertainty: recommend review or abstain according to frozen policy.
- Fallback: use only a verified, compatible model; record why and what was changed.
- Unavailable neighbour/context: use a specified edge-of-volume rule, not an implicit default.
- Model loading/inference failure: return a controlled error and preserve diagnostic logs.

The exact modality-presence gate and its test coverage are an implementation task until the repository audit proves they are already complete.

---

## 10. Missing-modality robustness study

### 10.1 Protocol

The re-evaluation covers 105 validation cases, 96 subjects, 16,275 axial slices and 6,549 ground-truth foreground slices. It evaluates five conditions using the same reported checkpoints and calibration parameters:

- C0: all four modalities.
- C1: omit T1n.
- C2: omit T1c (contrast).
- C3: omit T2w.
- C4: omit T2f (FLAIR).

Two policy families are distinguished:

- **Analysis A:** simplified entropy-only ablation. Empty predicted foreground was assigned entropy 0.0 and could be incorrectly retained.
- **Analysis B1:** full Gate 6A standalone ResUNet policy, including the empty-prediction guard, without the secondary model agent fallback.
- **Analysis B2:** full Gate 6A multi-model agent policy, including the guard and UNetResNet34 fallback.

The operationally relevant agent comparison is B2. Analysis A is a flawed/limited ablation and must not be reported as the deployed policy.

### 10.2 ResUNet segmentation under channel omission

| Condition | Omitted modality | Case-mean compartment Dice | Subject-mean compartment Dice | WT Dice | TC Dice | ET Dice |
|---|---|---:|---:|---:|---:|---:|
| C0 | None | **0.8641** | **0.8652** | 0.9039 | 0.8597 | 0.8286 |
| C1 | T1n | 0.8517 | 0.8534 | 0.8939 | 0.8494 | 0.8118 |
| C2 | T1c | **0.4449** | **0.4534** | 0.8532 | 0.3132 | **0.1684** |
| C3 | T2w | **0.6058** | **0.6081** | **0.4388** | 0.6901 | 0.6885 |
| C4 | T2f | 0.8394 | 0.8439 | 0.8748 | 0.8342 | 0.8091 |

### 10.3 B2 full-agent foreground-slice referral behavior

| Condition | B2 pooled referral | Pooled numerator / denominator | B2 subject-averaged referral | Empty predicted foreground slices | Referral shift vs C0, Bonferroni p |
|---|---:|---:|---:|---:|---:|
| C0 | 16.16% | 1,058 / 6,549 | 17.24% | 340 | Reference |
| C1 — omit T1n | 18.16% | 1,189 / 6,549 | 19.04% | 276 | 0.0213 |
| C2 — omit T1c | 17.44% | 1,142 / 6,549 | 18.55% | 319 | 1.0000 (not significant) |
| C3 — omit T2w | 62.65% | 4,103 / 6,549 | 63.34% | 2,621 | 7.12e-17 |
| C4 — omit T2f | 19.61% | 1,284 / 6,549 | 20.70% | 464 | 0.00048 |

Pooled and subject-averaged referral values answer different questions. For C2, 17.44% is the pooled foreground-slice rate; 18.55% is the mean of subject-level referral rates. Neither is a percentage of cases or patients.

### 10.4 Critical scientific interpretation

**C2 — missing T1c / contrast:** ET Dice falls from 0.8286 to 0.1684 and TC Dice from 0.8597 to 0.3132. Yet B2 referral is 17.44% pooled, with subject-averaged referral 18.55%; the paired subject-level referral shift is not significant after Bonferroni adjustment (p = 1.0000). The agent does not reliably detect this severe contrast-related failure through entropy/empty-mask signals alone. Do not describe “82% of cases accepted”: the relevant measure is slice-level retention, not case-level acceptance.

**C3 — missing T2w:** Empty predicted foreground rises to 2,621 foreground slices. The flawed Analysis A can retain empty predictions because it assigns entropy 0.0. Under B2, referral rises to 62.65% pooled; the study reports accepted-slice Dice 0.5307 versus rejected-slice Dice 0.1535. The empty-prediction guard therefore addresses this failure mode in this experiment, but this does not prove universal robustness.

**C1 and C4:** The full policy shows statistically significant referral shifts versus C0, but a referral shift by itself is not proof that the agent selected better masks or that a clinical workload benefit exists.

### 10.5 Required design consequence

A deterministic modality-presence/input-integrity gate must run before segmentation inference. The system must verify that all required channels are present and correctly assigned. If a modality is missing, the system should stop or enter a separately validated missing-modality model path. It must not silently run a four-channel model with an absent or fabricated channel and then rely on predictive entropy to detect the problem.

---

## 11. Inter-model prediction disagreement — exploratory only

Two-model sampled voxel disagreement is the fraction of sampled spatial voxels where the argmax class from ResUNet-2D and UNetResNet34 differs. The reported sampling used structured foreground/boundary and brain-background samples, approximately 1,500–2,500 voxels per case.

It is **inter-model prediction disagreement**, not Bayesian epistemic uncertainty. The 10% case threshold was selected post-hoc after inspecting the 105-case cohort.

| Exploratory tier | Cases | Cohort share | Mean Dice | Status |
|---|---:|---:|---:|---|
| Tier 1: disagreement below 10% | 94 | 89.5% | 0.8836 | Lower-disagreement group |
| Tier 2: primary overconfidence | 0 | 0.0% | N/A | Empty tier |
| Tier 3: disagreement at/above 10% | 11 | 10.5% | 0.6968 | Higher-disagreement group |
| Tier 4: catastrophic discordance | 0 | 0.0% | N/A | Empty tier |

Reported subject-level associations (N = 96): disagreement vs true voxel error rate, Spearman rho = 0.6637 (p = 1.71e-13); disagreement vs volumetric mean Dice, rho = -0.4971 (p = 2.57e-7).

Interpretation: disagreement is associated with difficult predictions in this validation analysis. Because the threshold is post-hoc and two tiers are empty, it is not an independently validated decision policy. Do not claim clinical workload reduction, prospective referral value or guaranteed interception of catastrophic cases.

---

## 12. Application and software delivery

### 12.1 Current state

`tools/demo_app.py` is the documented Streamlit prototype. A controlled headless startup using `.venv` reportedly returned HTTP 200 `ok` from `http://localhost:8501/_stcore/health`. This verifies process startup and health-endpoint readiness only. Manual browser interaction, every UI control and complete end-to-end demo behavior have not been verified in the supplied report.

For the current B.Tech project, a stable local Streamlit demonstration may be sufficient unless the supervisor rubric specifically requires a separate frontend/backend stack. Do not rebuild the UI in React/FastAPI merely because it sounds more production-like.

### 12.2 Optional full-stack direction

If justified by the rubric and time, the proposed stack is React + Vite + TypeScript UI, FastAPI backend and frozen PyTorch inference pipeline. The supplied repository screenshots did not show a visible `web_app/` or `src/api/` implementation. Treat this as **planned**, not implemented, until a local audit finds code and integration evidence.

### 12.3 Must-have demonstration journey

1. Select an approved validation/demo case or authorised input.
2. Validate modality presence, channel order, shape and dimensions.
3. View T1n/T1c/T2w/T2f MRI slices with synchronized navigation.
4. Display primary mask, optional fallback mask, final agent output and uncertainty visualization.
5. Explain the action (accept/fallback/review/abstain) with signals and reasons.
6. Display WT/TC/ET metrics only where permitted ground truth is available; otherwise label as prediction-only.
7. Display checkpoint/model identity, latency, device, warnings and errors.
8. Provide offline launch instructions and a recorded fallback demo.

Keep all MRI data local by default. Do not send medical images to external model APIs. Label the application as an academic research prototype, not a diagnostic device.

---

## 13. Environment, virtual environment and reproducibility

### 13.1 Reported primary environment

| Item | Reported value |
|---|---|
| Operating system | Windows 11 |
| Python | 3.11.9 |
| PyTorch | 2.11.0+cu128 |
| CUDA runtime shown by PyTorch | 12.8 |
| MONAI | 1.6.0 |
| GPU | NVIDIA GeForce RTX 4070 Laptop GPU, 8 GB VRAM |
| Project interpreter | `D:\MajorProject\.venv\Scripts\python.exe` |

Confirm current versions with the project interpreter during the read-only audit; do not assume global Python is equivalent.

### 13.2 Virtual environment preservation policy

**The existing `.venv` must be preserved exactly.** Do not delete, move, rename, recreate, upgrade, downgrade or modify it during repository cleanup or blueprint work. Do not run package installation commands unless a separate task explicitly authorizes a specific dependency change. Do not copy `.venv` into Git.

Capture a read-only environment snapshot when authorized:

- `D:\MajorProject\.venv\Scripts\python.exe --version`
- `D:\MajorProject\.venv\Scripts\python.exe -m pip freeze`
- PyTorch/CUDA/MONAI/Streamlit versions and CUDA availability.
- `nvidia-smi` output where available.

Store the snapshot as a versioned audit artifact. It is evidence of the environment at capture time, not a command to recreate or alter it.

### 13.3 Dependency policy

Track a reviewed dependency specification or lock file, not the virtual environment directory. Before regenerating `requirements.txt`, compare it with the actual `.venv` and inspect existing pins. Do not blindly replace pinned dependencies with a fresh `pip freeze` output.

Keep PyTorch as the DL framework. Do not introduce TensorFlow, Keras or MATLAB into the core pipeline.

---

## 14. Git and repository preservation

### 14.1 Current known state

The user reports that the Git link/remote was disconnected from the project folder and that it will be reconnected later for pushing updates. This document does **not** know whether `.git` remains intact, which branch is checked out, whether there are uncommitted changes, or what remote URL was used. Those facts must be inspected locally without mutation.

### 14.2 Git rules for all coding agents

- Preserve `.git` and all local commit history if present.
- Do not run `git init`, clone over the folder, delete `.git`, reset, clean, checkout, rebase, force push, add/remove remotes or push unless separately authorized.
- First inspect `git status --short --branch`, recent commit history and `git remote -v` read-only. Redact any credentials/tokens from reported URLs.
- Never “fix” a disconnected remote by guessing a URL. Ask the owner for the correct remote when the user is ready to reconnect.
- Before any future write/reorganization, confirm working-tree state and create a user-approved backup or checkpoint commit where appropriate.
- Inspect staged and unstaged changes before committing. No generated result is “committed” unless a commit hash is verified.

### 14.3 Git content policy

Do not blindly track MRI volumes, ZIP archives, extracted datasets, caches, temporary outputs or large model checkpoints in ordinary Git. Decide separately whether a specific checkpoint belongs in Git LFS or a controlled artifact store. Keep manifest IDs, hashes, configurations and reproducibility metadata in the repository where appropriate.

The `.gitignore` should normally exclude `.venv/`, data that is not licensed/appropriate for source control, cache directories, temporary files and machine-specific runtime artifacts. Audit the actual ignore rules before changing them.

### 14.4 Safe future reconnect sequence

1. Read-only inspect `.git`, branch, status, remotes and latest commits.
2. Confirm correct repository URL/owner with the user.
3. Verify local work and decide how it should be preserved.
4. Add or repair remote only with explicit approval.
5. Review tracked/untracked large files and secrets before staging.
6. Create a focused commit with a clear message.
7. Push only after the intended branch and remote are confirmed.

---

## 15. Repository structure and cleanup policy

V1.1 contained a screenshot-derived directory map, not a full source audit. Treat the following as a **provisional observed structure** until a read-only inventory is completed:

- `.git/`, `.venv/`
- `data/` with `raw/`, `extracted_archive/`, metadata and processed data
- `src/` with `agent/`, `data/`, `evaluation/`, `models/`, `training/`, `uncertainty/`, `visualization/`
- `configs/`, `tools/`, `tests/`, `experiments/`, `results/`
- `project_roadmap/`, `docs/`, `Report_Materials/`, `MajorProject_AI_Coding_Pack/`, `tmp/`
- root documentation, `README.md`, dependency files and historical reports

### 15.1 Four-way classification

| Classification | Meaning | Allowed action now |
|---|---|---|
| KEEP | Active source, required configuration, data provenance, checkpoint or evidence | Preserve |
| ARCHIVE CANDIDATE | Superseded but historically important | Identify only; do not move yet |
| REVIEW | Unclear ownership, duplicate or dependency | Inspect and ask for decision |
| REMOVE CANDIDATE | Apparently obsolete and proven unused | List only; no deletion without explicit approval |

### 15.2 Cleanup sequence

A. Read-only inventory and dependency/path audit.
B. Produce a current-to-proposed path map with risk and dependency checks.
C. User reviews and approves specific moves/archive actions.
D. Make one small change batch at a time; update imports, configs and paths; run tests; verify hashes and app launch.
E. Publish a final repository map and changelog.

**No cleanup operation is authorized by this blueprint.** Preserve `.venv`, `.git`, raw data, split manifests, checkpoints, caches required for reproducibility, historical logs and audit artifacts. A tidy folder is not worth breaking imports, run paths or provenance.

---

## 16. Model and agent extension roadmap

### 16.1 Candidate admission gates

Any new model must pass the following sequence:

1. **Research rationale:** name the gap it addresses and state a falsifiable hypothesis.
2. **Feasibility:** confirm supported input shape, dependencies, GPU/VRAM, training/inference time and environment compatibility before a full run.
3. **Implementation audit:** code review, tests, parameter count and reproducible config.
4. **Controlled evaluation:** use permitted development/validation data, clearly identified subject manifest and comparable metrics.
5. **Checkpoint provenance:** record selected checkpoint, hash, logs, code/config version and best epoch.
6. **Reliability assessment:** determine whether calibration is required and whether the model's uncertainty is comparable.
7. **Agent integration:** only after standalone evaluation; define what signal/action the model contributes and test fallback failures.
8. **Portfolio decision:** admit, defer, keep exploratory or reject. Preserve the old baseline regardless.

### 16.2 Agent extension priorities

1. Verify or implement deterministic modality-presence and input-integrity checks before inference.
2. Keep the empty-prediction guard explicit and unit-tested.
3. Reconfirm model-specific calibration and threshold provenance whenever checkpoints change.
4. Test axial context and fallback on healthy-slice false alarms as well as recovered misses.
5. Keep inter-model disagreement exploratory until a threshold is selected on an appropriate calibration set and evaluated on a separate cohort.
6. Record the final policy version and action traces; do not call the policy adaptive or learned unless it actually is.

### 16.3 Do not expand scope casually

No LLM/RAG/MCP orchestration in pixel-level inference. No new dataset unless a clear cross-dataset generalization question, permissions and compute plan are approved. No full 3D or foundation-model rework without a measured reason and feasible compute budget.

---

## 17. Project execution roadmap and gates

| Phase / gate | Workstream | Required output | Exit condition |
|---|---|---|---|
| **G0 — Baseline and repository audit** | Corrected evidence review, local repo/Git/venv inventory, open discrepancy list | Blueprint V2.0, status/registry and read-only audit | Major unknowns visible; no destructive changes |
| **G1 — Data/provenance freeze** | Dataset edition, subject ledger, manifests, label/channel checks, preprocessing | Verified data census, overlap report, hashes and config | Zero subject overlap across intended partitions; archive edition resolved |
| **G2 — Model portfolio freeze** | Existing five families and proposed new candidates | Candidate registry, comparable benchmark plan and selection decision | Primary/secondary plus optional exploratory models justified by evidence |
| **G3 — Calibration/reliability freeze** | Temperature scaling, thresholds, entropy definitions, empty-prediction policy and missing-modality gate | Reproducible calibration + policy artifacts | Exact source/checkpoint/cohort linkage and honest risk/coverage evidence |
| **G4 — Agent evaluation** | Fixed primary vs fallback vs review/abstain vs disagreement analysis | Subject-level comparisons, action traces, failure tests | Measured triage value or explicit triage-only conclusion; no forced Dice-gain claim |
| **G5 — Final training (conditional)** | Decide whether 220 historical cases are eligible after model/policy freeze | Final checkpoint(s), config, logs, hash, environment and manifest | Approved 941-case eligibility or a documented alternative; recalibrate new weights |
| **G6 — Application acceptance** | Streamlit or justified full-stack UI | Stable local demo, offline fallback, integration evidence | End-to-end validation/demo-case journey passes |
| **G7 — Locked test / Gate 6B** | One final held-out test run | Locked manifest, checkpoint/policy/code hashes, per-case outputs and audit | Explicit authorization first; full coverage; no test tuning |
| **G8 — Dissertation, presentation and viva** | Thesis, results tables, figures, code/runbook and defense | Final academic package and rehearsed demo | Every claim traces to evidence; limitations clearly defended |

The current next operational task is **G0 read-only repository, Git and environment audit**, after this blueprint is placed in the root. Do not jump to G5/G7 or start new GPU runs before the audit and candidate decisions.

### 17.1 Gate decision vocabulary

Each gate is **PASS**, **CONDITIONAL PASS** or **BLOCKED**. Record reviewer, date, evidence paths, exact command/run ID, unresolved issues and next authorized action. A coding agent cannot self-approve a scientific gate solely by writing “PASS” in a report.

---

## 18. Dissertation, presentation and viva requirements

### 18.1 Dissertation structure (adapt to official JIIT template)

1. Introduction, clinical/research context and problem statement.
2. Literature review: glioma segmentation, multi-parametric MRI, uncertainty, calibration and selective prediction.
3. Dataset, cohort provenance, split strategy, modalities, labels and preprocessing.
4. Model architectures, training protocols and checkpoint selection.
5. Evaluation metrics and statistical methods.
6. Calibration, reliability signals and deterministic agent design.
7. Controlled missing-modality study and exploratory inter-model disagreement.
8. Software/demo implementation and reproducibility.
9. Results, failure analysis, limitations and discussion.
10. Conclusion and bounded future work.

### 18.2 Claims that must not be made

- “The agent improves Dice” unless a specific, paired, properly evaluated agent output actually improves segmentation quality. The current nominal B2 mean Dice is essentially the same as standalone ResUNet.
- “The system detects all missing modalities.” The missing-T1c result shows a serious blind spot.
- “The 105-case evaluation is completely independent.” The checkpoints were selected on the earlier full validation pool.
- “The 155 test subjects were untouched with absolute proof.” Use 155 test cases / 143 subjects and the qualified quarantine statement.
- “LiteMedSAM was evaluated.” Official pretrained LiteMedSAM weights were not loaded.
- “SegResNet-3D is competitive.” It is an underfit feasibility pilot.
- “nnU-Net v2 was trained.” The official planner was not executed.
- “Inter-model disagreement is epistemic uncertainty” or the post-hoc 10% threshold is a validated clinical policy.
- “Sub-millimeter boundary accuracy.” The reported HD95 medians do not support it.
- “Clinical validation,” “diagnostic-grade,” “regulatory approved,” or “reduces radiologist workload.” No prospective clinical validation exists.

### 18.3 Viva readiness

Every team member should be able to explain:

- Dataset channels, labels, subject split and test quarantine.
- The difference between case count and subject count.
- The five architecture families and why the final pair was selected.
- How Dice/WT/TC/ET and HD95 are calculated, including empty masks.
- Predictive entropy and temperature scaling, and why calibration may not change the mask.
- The difference between entropy, inter-model disagreement and true error.
- Why missing T1c remains difficult for the current policy.
- What the empty-prediction guard fixes and what it does not fix.
- Why the agent is a deterministic research policy rather than an LLM.
- The Streamlit demo status, limitations and failure handling.
- The exact test authorization and quarantine rule.

Prepare a concise live demo and a local offline fallback video/screenshots. Do not rely on an external API or network connection.

---

## 19. Risk register and control decisions

| Risk | Severity | Control |
|---|---|---|
| Dataset edition/provenance unresolved | High | Verify archive metadata, source record and manifests before thesis naming |
| Longitudinal subject leakage | Critical | Parent-subject grouping and overlap audit across all splits |
| Checkpoint selection on full validation cohort | High | Disclose dependence; reserve test for independent locked evaluation |
| ECE value/protocol discrepancy | Medium–High | Reconcile exact artifact, mask, bins and code before final claim |
| Missing T1c silently degrades ET | Critical | Upstream modality-presence gate; stop inference on missing required channel |
| Empty predictions accepted by entropy-only ablation | High | Preserve explicit empty-mask guard and unit tests |
| Incomparable added-model runs | High | Candidate admission protocol and matched benchmark plan |
| Agent does not improve Dice | High | Evaluate triage/review value separately; never force an improvement claim |
| Test contamination | Critical | Quarantine; one explicit, locked, approved evaluation only |
| `.venv` damaged during cleanup | High | Preserve environment; read-only inventory; no package mutation |
| `.git` history/remote damaged | Critical | Preserve `.git`; no re-init/reset/clean/remote mutation/push during audit |
| Large data/checkpoints committed accidentally | High | Audit tracked files, ignore rules and artifact-storage plan |
| UI consumes research time | Medium | Keep Streamlit if sufficient; only adopt full-stack if rubric/value justifies it |
| 8 GB GPU resource exhaustion | High | Sequential GPU jobs, measured feasibility and bounded experiments |
| Unsupported clinical/novelty claims | High | Evidence ledger; academic language and limitations review |

---

## 20. Troubleshooting workflow

Use a diagnosis-first approach. Change one variable at a time.

| Symptom | First checks | Do not do |
|---|---|---|
| Dice changes unexpectedly | Manifest, subject split, preprocessing, label mapping, checkpoint hash, sampler and aggregation | Do not add a model or change loss immediately |
| CUDA OOM | Existing processes, tensor shapes, batch/patch size, mixed precision and peak memory | Do not launch concurrent GPU jobs |
| Empty masks | Channel order, normalization, class mapping, threshold, target distribution and neighbouring slices | Do not remove empty slices from evaluation |
| Entropy does not track Dice | Entropy mask, calibration, empty masks, sampled voxels and subject grouping | Do not tune on the evaluation/test cohort |
| Agent fallback adds false positives | Context radius, neighbour rule, healthy-slice false alarms and cost | Do not report recovered misses alone |
| Agent Dice equals primary model | Check whether selected mask changes; separate calibration and triage metrics | Do not claim calibration improvement is segmentation improvement |
| UI disagrees with offline metrics | Check checkpoint, preprocessing, threshold, slice index, post-processing and output version | Do not modify test data to reproduce a UI bug |
| Python package suddenly fails | Confirm `.venv` interpreter, package versions and environment variables | Do not recreate or upgrade `.venv` as a first response |
| Git push/reconnect problem | Inspect branch/status/remotes and credentials safely | Do not re-init, reset, clean or guess a remote URL |
| Coding agent fails | Capture command, environment, stdout/stderr, exit code and diff | Do not mark complete because the process launched |

**Recovery loop:** `STOP -> PRESERVE LOGS -> REPRODUCE SMALL -> IDENTIFY ONE CAUSE -> PATCH ONE THING -> TEST -> RECORD`.

If the issue involves data integrity, checkpoint provenance, test leakage, Git history or environment corruption, stop and reopen the relevant gate.

---

## 21. Working protocol for ChatGPT, Antigravity and coding agents

### 21.1 Required context at the start of each substantial task

Read in this order:

1. `MASTER_PROJECT_BLUEPRINT_v2.0.md`.
2. `PROJECT_STATUS.md` (once created and verified).
3. `MODEL_AND_EXPERIMENT_REGISTRY.csv` (once created and verified).
4. `ARCHITECTURE_AND_RESEARCH_DECISIONS.md` / decision log.
5. `EVIDENCE_AND_ARTIFACT_INDEX.md` and the exact source reports for the task.
6. The latest session handoff.

Do not rely on old prompts, chat summaries or stale planning files when a more recent evidence artifact exists.

### 21.2 Task specification format

`CONTEXT + GOAL + EXACT FILES + INPUTS/OUTPUTS + CONSTRAINTS + ACCEPTANCE TESTS + DO NOT CHANGE + EVIDENCE TO RETURN`.

### 21.3 Default coding-agent rules

- Prefer read-only inspection before editing.
- State the exact files to be changed before implementation.
- No parallel writes to shared source/config/docs files.
- No training, inference, test access, dependency change or Git mutation unless explicitly included in the task.
- Return exact commands, exit codes, logs, hashes, metrics and tests.
- Distinguish created files from tested, verified, committed and pushed files.
- Update the status and changelog after an authorized milestone; do not rewrite historical evidence.
- If evidence conflicts, stop and report the conflict rather than choosing the convenient value.

### 21.4 Session handoff template

| Field | Required entry |
|---|---|
| Date / operator | Who performed the task and when |
| Current gate | G0–G8 and PASS/CONDITIONAL/BLOCKED |
| Objective | Exact work completed |
| Files read/changed | Full paths; state read-only if applicable |
| Commands | Exact command lines and working directory |
| Environment | Interpreter, package/GPU context |
| Evidence | Run ID, manifest, checkpoint/config hashes, logs, result files |
| Tests | Count, pass/fail/skip, exit code |
| Risks/open issues | What remains unresolved |
| Next authorized action | One concrete task, not a wishlist |

---

## 22. Supporting records and expected root-level companion files

The following compact files should be maintained alongside this blueprint after the read-only repository audit. Do not create duplicates if current valid equivalents already exist; audit them first.

| File | Purpose |
|---|---|
| `PROJECT_STATUS.md` | Current milestone, blockers, last completed task and next authorized task |
| `MODEL_AND_EXPERIMENT_REGISTRY.csv` | One row per architecture/run/candidate, with status and provenance |
| `ARCHITECTURE_AND_RESEARCH_DECISIONS.md` | Decisions, rationale, alternatives, owner/date and evidence |
| `EVIDENCE_AND_ARTIFACT_INDEX.md` | Human-readable map of authoritative reports, logs, manifests and hashes |
| `PROJECT_CHANGELOG.md` | Dated changes to scope, research baseline, code, environment and documents |
| `README.md` | Setup, how to run the demo/tests and link to this blueprint |
| `REPOSITORY_INVENTORY_READONLY.md` | Read-only tree/dependency/Git/environment audit output |

The Markdown file is the canonical editable source for AI-assisted maintenance. DOCX/PDF copies are reading/printing formats. Update the Markdown first, then regenerate the presentation formats from it so the versions do not diverge.

---

## 23. Immediate next actions — exact order

1. **Place this V2.0 document at the root** of `D:\MajorProject` as `MASTER_PROJECT_BLUEPRINT_v2.0.md`. Keep the original V1.1 PDF/DOCX as historical records; do not delete them.
2. Run the read-only repository, Git and `.venv` inventory. Preserve `.git`, `.venv`, data, manifests, checkpoints and historical outputs.
3. Reconcile the open provenance items: BraTS edition, ECE artifact/definition, dissertation CSV parse integrity, exact local checkpoint/manifest hashes and current model candidate identities.
4. Identify the 2–3 proposed models by exact name and evidence. Apply the admission checklist before training or integration.
5. Decide and implement the upstream modality-presence/input-integrity gate; test missing-channel behavior explicitly.
6. Complete only the necessary agent and application integration work after interfaces are stable.
7. Decide whether 220 historical cases are eligible and whether final 941-case training is required. Recalibrate if any checkpoint changes.
8. Freeze model weights, calibration, policy, software and evaluation code before requesting Gate 6B approval.
9. Run the held-out test only after explicit authorization, as a locked one-time evaluation.
10. Rebuild dissertation, presentation, demo and viva materials from verified evidence.

Do not reorder these dependencies to chase more model runs or visual polish.

---

## 24. Source register and evidence map

### 24.1 Primary project planning source

- `Brain_Tumor_Segmentation_Master_Roadmap_v1.1.pdf` — 31-page V1.1 roadmap supplied for this revision. Its useful research questions, evidence rules, model-comparison discipline, staged gates, team workflow, troubleshooting and tool-use rules were retained. Its old “next action”, dataset-edition certainty and pre-remediation Gate 6A state were updated or explicitly qualified.

### 24.2 November 10, 2026 corrected research outputs supplied for this revision

- `FINAL_CORRECTED_RESEARCH_BASELINE.md`
- `MISSING_MODALITY_POLICY_REEVALUATION.md`
- `FINAL_MODEL_COMPARISON.md`
- `FINAL_AGENT_EVALUATION.md`
- `FINAL_EVIDENCE_AND_LIMITATIONS.md`
- `DISSERTATION_RESULTS_TABLES.csv`
- `reevaluated_modality_metrics.csv`
- `subject_level_modality_stats.csv`
- `modality_policy_comparison.json`
- `FORENSIC_RECONCILIATION_REPORT.md` and related artifact/claim ledgers where present in the project results.

### 24.3 Important evidence paths reported in the supplied records

- `results/final_research_baseline/`
- `results/full_training_15ep/resunet_2d/best_model.pt`
- `results/full_training_15ep/unet_resnet34/best_model.pt`
- `results/dataset_splits/test_manifest.json`
- `src/agent/policy.py`
- `tools/reevaluate_missing_modality_policy.py`
- `tools/demo_app.py`

Verify all paths and files locally. This document does not establish that every path exists in the current folder or that a file has been committed to Git.

---

## 25. Final operating statement

**One project. One evidence-controlled research question. One canonical blueprint.**

The project should demonstrate a trustworthy segmentation workflow and an honest reliability-aware decision policy—not a long list of architectures or fashionable tools. New models and features are accepted only when their purpose, cost, evidence and impact on the research question are clear. Preserve the current scientific baseline, protect the test cohort, keep the existing `.venv` and Git history safe, and update this blueprint only when evidence-backed decisions genuinely change.

**This V2.0 document is intended to be the first official root-level operating blueprint in `D:\MajorProject`.** It does not authorize repository cleanup, package changes, Git reconnection, model training or test-set access.
