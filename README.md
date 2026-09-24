# Major Project — Brain Tumor Segmentation & Reliability-Aware AI Agent

**Team:** Nishant Kumar, Mohammad Kaif, Tariq  
**Batch:** B.Tech CSE 2023–2027  
**Institute:** Jaypee Institute of Information Technology (JIIT), Noida  
**Domain:** AI / ML  
**Current direction:** Uncertainty-aware, reliability-focused AI agent for brain tumor segmentation under heterogeneous MRI conditions.

> **Status:** Development + research. The repository currently contains the validated dataset pipeline, three 2D segmentation baselines, benchmark results, predictive uncertainty analysis, failure analysis, and initial MRI-condition/reliability experiments. The next major phase is an explicit reliability-aware agent.

---

## 1. Project Goal

Mentor-proposed direction:

> “Uncertainty-aware agent dynamically selects segmentation strategies, modalities, and specialized models for reliable brain tumor segmentation under heterogeneous MRI conditions.”

The project is not intended to be only a segmentation-model comparison.

```text
MRI input
   ↓
Input / condition analysis
   ↓
Uncertainty estimation
   ↓
Reliability-aware agent
   ↓
Select model / strategy / modality
   ↓
Segmentation
   ↓
Reliability + uncertainty report
   ↓
Optional fallback / escalation
```

The central research/development question is whether an agent can use model predictions and uncertainty/reliability signals to make safer segmentation decisions than simply using one fixed model.

---

## 2. Project Constraints

- Maximum team size: 3.
- Domain: P5 — AI / ML.
- Project must include substantial development; pure research alone is not sufficient.
- Final report target: 50–60+ pages.
- Mid-semester showcase/viva: **26 September 2026**.
- Supervisor meeting diary must be maintained.
- Mission AI encourages AI Agents / Agentic AI / MCP where genuinely applicable.
- The agent must be an implemented component, not only a diagram.
- Research claims must be supported by experiments.
- Keras, TensorFlow and MATLAB are not allowed.
- PyTorch is the allowed deep-learning framework.
- MCP should only be included if it provides a real function; it must not be added merely for terminology.

Engineering rules:

- Keep experiments controlled.
- Record dataset split, preprocessing, hyperparameters, seed and hardware.
- Do not call validation Dice “accuracy”.
- Do not select a final uncertainty threshold only because it looks good once.
- Do not silently ignore complete segmentation failures.
- Do not claim clinical validity from this development work.
- Current uncertainty experiments are exploratory.

---

## 3. Dataset

We are using the **BraTS 2023 GLI** training data.

Four MRI modalities:

- t1n
- t1c
- t2f
- t2w

Dataset audit completed:

- Total cases found: **198**
- Complete cases: **197**
- Incomplete cases: **1**
- Incomplete case: **BraTS-GLI-00646-001**
- Complete case shape: **(240, 240, 155)**
- Development split:
  - Train: **158 cases**
  - Validation: **39 cases**
  - Train slices: **13,074**
  - Validation slices: **3,223**

The development subset is intentionally used while methodology is being developed because full-scale MRI experiments are computationally expensive.

Metadata:

```text
data/metadata/BraTS2023_2017_GLI_Mapping.xlsx
data/metadata/README (1).md
```

Root archives currently include the BraTS 2023 GLI training and validation ZIP files.

**Important:** Do not infer semantic meanings for mask labels from the mapping Excel alone. Verify label semantics from official dataset documentation before making scientific claims.

---

## 4. Repository Structure

```text
D:\MajorProject
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata/
│       ├── BraTS2023_2017_GLI_Mapping.xlsx
│       └── README (1).md
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── test_dataset.py
│   ├── models/
│   │   ├── unet.py
│   │   ├── unetpp.py
│   │   └── attention_unet.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── visualization/
│   └── uncertainty/
│       ├── entropy.py
│       ├── analyze_uncertainty.py
│       ├── predictive_uncertainty.py
│       ├── uncertainty_bins.py
│       ├── model_uncertainty_comparison.py
│       ├── compare_model_uncertainty_stats.py
│       ├── analyze_failure_cases.py
│       ├── reliability_thresholds.py
│       ├── reliability_auc.py
│       ├── model_selection_analysis.py
│       ├── feature_selection_baseline.py
│       ├── feature_selection_case_split.py
│       ├── case_level_reliability.py
│       ├── input_condition_analysis.py
│       └── case_input_reliability.py
│
├── experiments/
├── results/
├── configs/
├── docs/
├── .gitignore
├── README.md
└── requirements.txt
```

Raw datasets, checkpoints and experiment outputs are intentionally excluded from Git where appropriate.

---

## 5. Environment

Current development machine:

- Windows
- NVIDIA RTX 4070 Laptop GPU
- 8 GB VRAM
- CUDA available
- Python 3.11 virtual environment
- Virtual environment: `D:\MajorProject\.venv`

Validated core packages:

- PyTorch 2.11.0+cu128
- MONAI 1.6.0
- nibabel 5.4.2
- scikit-learn 1.9.1
- matplotlib 3.11.2
- pandas 3.0.6
- tqdm 4.70.1
- NumPy 2.4.6
- SciPy 1.17.1

---

## 6. Installation / Terminal Setup

PowerShell:

```powershell
cd D:\MajorProject
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements.txt
```

CUDA verification:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected current environment:

```text
2.11.0+cu128
True
NVIDIA GeForce RTX 4070 Laptop GPU
```

If rebuilding on another machine, install the appropriate PyTorch CUDA build for that machine first. Do not assume the current CUDA wheel is universally correct.

---

## 7. Important Run Commands

Activate the environment first:

```powershell
cd D:\MajorProject
.\.venv\Scripts\Activate.ps1
```

Dataset smoke test:

```powershell
python -m src.data.test_dataset
```

U-Net baseline training:

```powershell
python -m src.training.train
```

Uncertainty:

```powershell
python -m src.uncertainty.analyze_uncertainty
python -m src.uncertainty.predictive_uncertainty
python -m src.uncertainty.uncertainty_bins
```

Model uncertainty:

```powershell
python -m src.uncertainty.model_uncertainty_comparison
python -m src.uncertainty.compare_model_uncertainty_stats
```

Failure/reliability:

```powershell
python -m src.uncertainty.analyze_failure_cases
python -m src.uncertainty.reliability_thresholds
python -m src.uncertainty.reliability_auc
```

Model selection:

```powershell
python -m src.uncertainty.model_selection_analysis
python -m src.uncertainty.feature_selection_case_split
```

MRI condition analysis:

```powershell
python -m src.uncertainty.input_condition_analysis
python -m src.uncertainty.case_input_reliability
```

Git:

```powershell
git status
git log --oneline --decorate -10
```

---

## 8. GitHub Repository

Repository:

https://github.com/NK-NISHANT/major-project-brain-tumor-segmentation

Current repository is private.

Important commits:

```text
c158c51 Initial project setup
0fbeb4f Add BraTS dataset pipeline
55aec99 Add U-Net baseline training pipeline
28e7a36 Add U-Net baseline visualization
1239c27 Add U-Net++ benchmark model
a4738fb Document U-Net and U-Net++ benchmark results
3bb3c38 Add Attention U-Net benchmark model
083c297 Document Attention U-Net benchmark results
df30b97 Ignore experiment outputs
3bcfaaf Add MRI condition reliability analysis
```

---

# 9. Completed Work

## M1 — Dataset Pipeline

Completed:

- Dataset discovery and audit.
- Complete/incomplete case detection.
- Case-wise train/validation split.
- Four-modality loading.
- Slice extraction.
- Non-zero voxel normalization.
- Mask loading.
- Dataset smoke testing.
- Case-level caching.
- Case-wise batch sampling.
- DataLoader performance optimization.

The performance optimization loads and normalizes a case once rather than repeatedly loading individual slices from disk.

---

## M2 — U-Net Baseline

Implemented class: **UNet2D**

Parameters: **7,763,428**

5-epoch development benchmark:

| Metric | Result |
|---|---:|
| Best epoch | 5 |
| Best mean Dice | 0.7711 |
| Best mean IoU | 0.6329 |
| Runtime | ~39.9 min |

**0.7711 is mean Dice, not accuracy.**

---

## M3 — U-Net++

Implemented class: **UNetPlusPlus2D**

Parameters: **9,160,068**

5-epoch benchmark:

| Metric | Result |
|---|---:|
| Best epoch | 4 |
| Best mean Dice | 0.7937 |
| Best mean IoU | 0.6582 |
| Runtime | ~67.3 min |

This configuration produced higher validation Dice/IoU than U-Net, while also using more parameters and taking longer.

This is a configuration-specific development result, not a universal claim that U-Net++ is always better.

---

## M4 — Attention U-Net

Implemented class: **AttentionUNet2D**

Parameters: **7,852,160**

5-epoch benchmark:

| Metric | Result |
|---|---:|
| Best epoch | 5 |
| Best mean Dice | 0.7291 |
| Best mean IoU | 0.5886 |
| Runtime | ~47.1 min |

---

# 10. Controlled Benchmark Setup

Current three-model comparison:

- Same development dataset split.
- Same train/validation cases.
- Same four input modalities.
- Same 240×240 spatial size.
- Batch size 2.
- Learning rate 1e-3.
- AdamW.
- Dice + Cross Entropy loss.
- Seed 42.
- 5-epoch development budget.
- No augmentation in the current benchmark.
- Same RTX 4070 8 GB GPU.

Core principle:

> When comparing architectures, change the architecture while keeping the rest of the experimental setup controlled.

---

# 11. Uncertainty / Reliability Work

Normalized softmax entropy is implemented in:

```text
src/uncertainty/entropy.py
```

Higher entropy indicates a less concentrated probability distribution.

### Oracle diagnostic

An initial experiment measured entropy over ground-truth foreground pixels.

This is useful diagnostically but is **not deployable**, because ground truth is unavailable at inference time.

### Predictive foreground uncertainty

The next implementation measures entropy over the model's **predicted foreground pixels**.

Current U-Net validation result:

- Valid predicted-foreground slices: **2,460**
- Spearman uncertainty vs Dice: **rho ≈ -0.6542**
- Spearman uncertainty vs IoU: **rho ≈ -0.6864**

Higher predictive uncertainty tended to occur with lower segmentation quality in this development validation data.

The displayed IoU p-value of 0.0 is numerical underflow, not literally zero.

---

# 12. Uncertainty Bins

U-Net validation slices were divided into five uncertainty quantile bins:

| Bin | Mean uncertainty | Mean Dice | Mean IoU |
|---|---:|---:|---:|
| Very Low | 0.0433 | 0.8217 | 0.7532 |
| Low | 0.0626 | 0.7697 | 0.6827 |
| Medium | 0.0827 | 0.6520 | 0.5527 |
| High | 0.1236 | 0.5712 | 0.4589 |
| Very High | 0.2875 | 0.3008 | 0.2102 |

These bins are exploratory and are not final clinical thresholds.

---

# 13. Model-Wise Predictive Uncertainty

| Model | Valid slices | Dice rho | IoU rho |
|---|---:|---:|---:|
| U-Net | 2460 | -0.6542 | -0.6864 |
| U-Net++ | 2255 | -0.6084 | -0.6446 |
| Attention U-Net | 2454 | -0.5977 | -0.6357 |

This supports studying predictive entropy as a reliability signal across multiple architectures.

---

# 14. Complete-Miss Analysis

A complete miss means ground-truth foreground exists but the model predicts zero foreground pixels.

| Model | Complete misses | Miss rate |
|---|---:|---:|
| U-Net | 118 / 2578 | 4.58% |
| U-Net++ | 323 / 2578 | 12.53% |
| Attention U-Net | 124 / 2578 | 4.81% |

Important implication:

> Empty predictions cannot produce predicted-foreground entropy, so the final reliability system must explicitly handle complete misses as a high-risk state.

---

# 15. Reliability AUC

Exploratory definition:

**poor segmentation = Dice < 0.50**

This is a development-analysis definition, not a clinical standard.

| Model | ROC-AUC |
|---|---:|
| U-Net | 0.8059 |
| U-Net++ | 0.7646 |
| Attention U-Net | 0.7550 |

This is currently one of the strongest reliability findings, but it still needs stronger case-level/generalization evaluation.

---

# 16. Model Selection Findings

Question tested:

> Can the model with the lowest uncertainty automatically be selected as the model with the best actual Dice?

Across **2,246 slices** where all three models had valid predictive foreground uncertainty:

- Lowest-uncertainty model agreed with highest-Dice model on **48.35%** of slices.

Conclusion:

> **Minimum uncertainty alone is not sufficient for model selection.**

This is an important negative result. The final agent needs multiple signals.

---

# 17. Feature-Selection Findings

A Random Forest model-selection experiment was attempted.

A same-data experiment reached about **58.19%** agreement, but it was optimistic because of leakage.

A case-wise split gave:

- Training cases: 31
- Test cases: 8
- Baseline agreement: **45.27%**

Adding simple MRI input-condition features reduced agreement to:

**39.71%**

Therefore we currently have no evidence that arbitrary global intensity statistics improve unseen-case model selection.

---

# 18. MRI Input-Condition Analysis

Current case-level features include:

- non-zero fraction
- mean
- standard deviation
- minimum
- maximum

for each of the four modalities.

Total numeric modality features: **20**

Validation cases: **39**

Some exploratory correlations were observed between MRI intensity statistics and reliability/failure measures.

However:

- only 39 cases were used;
- many correlations were tested;
- slices within a case are not independent;
- this analysis is exploratory;
- it should not be treated as causal evidence.

The first condition-aware feature-selection attempt reduced unseen-case agreement, so arbitrary feature expansion should stop.

---

# 19. Methodological Lessons

1. **Uncertainty is useful:** predictive entropy correlates with segmentation quality in the current validation experiments.
2. **Uncertainty alone is not enough:** minimum uncertainty did not reliably identify the best model.
3. **Complete misses need explicit handling:** empty predictions need a separate high-risk state.
4. **Case-level reasoning matters:** the final agent should reason over MRI cases/volumes, not independently switch models for every slice.
5. **Simple condition features are not automatically useful:** raw intensity statistics did not improve case-wise model selection.
6. **Leakage must be avoided:** case-wise splitting is required for generalization tests.

---

# 20. Current Position

```text
Dataset
   ✓
Preprocessing / data pipeline
   ✓
U-Net baseline
   ✓
U-Net++ benchmark
   ✓
Attention U-Net benchmark
   ✓
Predictive uncertainty
   ✓
Failure analysis
   ✓
Initial reliability evaluation
   ✓
Initial MRI condition analysis
   ✓
--------------------------------
Reliability-aware agent
   → NEXT
```

The project is **not finished**. The segmentation/reliability groundwork is now sufficiently developed to move into the explicit agent phase.

---

# 21. Next Major Development Phase

Immediate next implementation:

```text
src/uncertainty/agent_state.py
```

Start with a simple, interpretable state representation.

Candidate state information:

- input condition indicators
- model identity
- predictive uncertainty
- predicted foreground size
- complete-miss flag
- segmentation quality proxies
- model disagreement
- modality/condition information where justified

Do not jump directly to an LLM agent.

Development order:

```text
state representation
      ↓
rule-based reliability policy
      ↓
evaluate policy
      ↓
case-level decision making
      ↓
improve agent
      ↓
optional LLM/tool layer
```

---

# 22. Planned Agent Architecture

```text
Input MRI
   ↓
Run candidate model(s)
   ↓
Collect reliability signals
   ↓
Agent state
   ↓
Decision policy
   ↓
Choose:
   - accept result
   - switch model
   - retry / alternate strategy
   - flag low reliability
   ↓
Final segmentation + reliability report
```

Initial rule concept:

```text
IF complete_miss:
    flag_high_risk
    try_alternative_model

ELSE IF uncertainty is high:
    evaluate alternative model

ELSE:
    accept current prediction
```

These are policy concepts only. Final thresholds must be experimentally selected.

---

# 23. Planned Additional Models

Potential additions:

1. ResUNet
2. U-Net with ResNet encoder
3. TransUNet
4. Optional 3D U-Net if memory/data pipeline permit

Do not add models merely to increase the count. Every added architecture needs an experimental reason.

---

# 24. Full Dataset Strategy

1. Stabilize methodology on the development subset.
2. Finalize reliability features and agent logic.
3. Select informative experiments.
4. Scale selected/final experiments to the full training dataset.
5. Run final evaluation and robustness analysis.
6. Freeze final results for report/demo.

Do not train every exploratory model at full scale immediately.

---

# 25. 2D vs 3D

Current segmentation models are **2D**.

The final agent should preferably reason at the **case/volume level**.

Possible architecture:

```text
3D MRI case
   ↓
2D slice-level model predictions
   ↓
aggregate reliability signals
   ↓
case-level agent state
   ↓
agent decision
```

A 3D U-Net can be explored later if hardware and preprocessing allow it.

---

# 26. Agentic AI / LLM / MCP

The project should not artificially add an LLM.

Possible later extension:

- deterministic reliability engine performs numerical analysis;
- agent consumes structured state;
- LLM provides human-readable explanation if useful;
- MCP is used only if an actual tool/context interface is required.

The core reliability decision should remain grounded in measurable model outputs.

---

# 27. Mid-Semester Viva — 26 September 2026

Suggested 10-minute story:

1. Problem — 1 min
2. Motivation — 1 min
3. Dataset/pipeline — 1 min
4. Three baselines — 2 min
5. Uncertainty — 2 min
6. Failure cases — 1 min
7. Proposed agent — 1 min
8. Future work — 1 min

Show actual outputs rather than only architecture diagrams.

Do not claim:

- 77% accuracy;
- clinical validation;
- universal superiority of U-Net++;
- an implemented agent if it is not implemented;
- MCP if it is not implemented;
- full-dataset training if it has not happened;
- a clinically correct uncertainty threshold.

Use wording such as:

> “In our current development validation experiment…”

---

# 28. Final Roadmap

### Phase A — Completed

- Dataset audit
- Data pipeline
- U-Net
- U-Net++
- Attention U-Net
- Benchmarking
- Uncertainty
- Failure analysis
- Reliability AUC
- Initial model selection
- Initial MRI condition analysis

### Phase B — Immediate

- Agent state
- Transparent reliability rules
- Complete-miss handling
- Multi-model reliability signals
- Case-level aggregation

### Phase C — Agent Evaluation

- Define action space.
- Define baseline policies.
- Compare fixed-model baseline vs agent.
- Measure model switching.
- Measure failure detection.
- Measure reliability improvement.
- Test unseen cases.

### Phase D — More Models

- ResUNet
- ResNet encoder U-Net
- TransUNet
- Optional 3D U-Net

### Phase E — Full Dataset

- Scale selected experiments.
- Re-run final benchmarks.
- Re-run reliability analysis.
- Evaluate generalization.

### Phase F — Mission AI Layer

- Structured agent state.
- Tool-based agent if useful.
- Optional LLM explanation.
- Optional MCP only if technically justified.

### Phase G — Final Evaluation

- Segmentation metrics.
- Reliability metrics.
- Failure detection.
- Model-selection/switching metrics.
- Case-level analysis.
- Ablation studies.
- Runtime/parameter analysis.
- Robustness/generalization.

### Phase H — Final Deliverables

- Working software/demo.
- 50–60+ page final report.
- Research-paper style write-up.
- Architecture diagrams.
- Experiment tables.
- GitHub repository.
- Meeting diary.
- Final presentation.
- Final viva preparation.

---

# 29. Suggested Final Report Structure

1. Abstract
2. Introduction
3. Problem Statement
4. Motivation
5. Literature Review
6. Dataset
7. Data Preprocessing
8. Baseline Segmentation Architectures
9. Experimental Methodology
10. Segmentation Results
11. Uncertainty Estimation
12. Reliability Analysis
13. Agent Architecture
14. Agent Decision Policy
15. Model Selection / Switching
16. Robustness Analysis
17. Full Dataset Evaluation
18. Ablation Studies
19. Discussion
20. Limitations
21. Future Work
22. Conclusion
23. References
24. Appendix

---

# 30. Mission AI / Engineering Alignment

The project is designed to provide evidence for:

- **WP1:** deep engineering knowledge through segmentation, uncertainty and agent design.
- **WP2:** heterogeneous MRI conditions and conflicting reliability signals.
- **WP3:** no obvious single-model solution; agent design requires experimentation.
- **WP4:** reliability-aware decision making is more complex than direct segmentation.
- **WP5:** the system goes beyond simply applying a standard segmentation architecture.
- **WP6:** coordination among project members, supervisor and research resources.
- **WP7:** data, models, uncertainty, agent decisions and evaluation treated as a system.
- **EA1:** diverse data/models/tools.
- **EA3:** creative combination of segmentation, uncertainty and agentic reasoning.
- **EA4:** reliability is explicitly studied because segmentation errors can have significant consequences.
- **EA5:** agentic reliability evaluation extends beyond the initial baseline experience.

These mappings must be supported by actual implementation and experiments in the final report.

---

# 31. README Update Policy

This is a **living handoff document**.

Do not update it after every tiny code change.

Update it after every major development chunk.

Recommended checkpoints:

### Update 1 — Current checkpoint
This version.

### Update 2 — Agent state + first rule agent
Add state features, action space, rule logic and first results.

### Update 3 — Additional model benchmark
Add architecture, parameters, runtime, Dice/IoU and reliability behavior.

### Update 4 — Agent evaluation
Add baseline vs agent, switching statistics, failure detection and case-level results.

### Update 5 — Full dataset scaling
Add final dataset size, final training setup, final metrics and generalization results.

### Update 6 — Final submission
Freeze final architecture, final results, final commands, final demo instructions and final report structure.

**Never add an unverified claim to this README.**

---

# 32. Current Do-Not-Break Rules

1. Do not delete existing benchmark results.
2. Do not overwrite checkpoints without a reason.
3. Keep experiment names/versioned.
4. Keep case-wise splits for reliability/generalization tests.
5. Keep model-specific metrics separate.
6. Never reuse U-Net metrics for another model.
7. Keep complete misses explicitly represented.
8. Do not use ground truth in deployable uncertainty features.
9. Do not leak validation cases into training.
10. Record important experiments.
11. Commit meaningful milestones to Git.
12. Update this README after major milestones.
13. Never claim an experiment was performed if it was not.
14. Do not add AI-agent/MCP terminology without actual implementation.

---

# 33. Immediate Next Step

Implement:

```text
src/uncertainty/agent_state.py
```

Start with a small, interpretable state representation.

The next major milestone is a **working rule-based reliability agent with measurable model-selection/fallback behavior**.

---

## Current Status in One Line

**Dataset → pipeline → 3-model benchmark → predictive uncertainty → failure/reliability analysis → initial condition analysis → NEXT: explicit reliability-aware agent.**
