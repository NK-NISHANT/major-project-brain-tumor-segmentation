# Project Operations and Reproducibility Runbook

**Project Root:** `D:\MajorProject`
**Python Environment:** `D:\MajorProject\.venv\Scripts\python.exe`
**Device:** Windows 11 (PowerShell), NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM, 75.15 GB free disk `[MEASURED]`)
**Parent Roadmap:** `../../docs/historical/MASTER_PROJECT_ROADMAP.md`
**Last Updated:** October 26, 2026 (Targeted Planning Corrections Applied)

---

## 1. Environment Verification & Activation

Every terminal session must start with activating the dedicated virtual environment and confirming CUDA availability:

```powershell
# 1. Navigate to project root
cd D:\MajorProject

# 2. Activate virtual environment
.\.venv\Scripts\Activate.ps1

# 3. Verify Python version (Expected: 3.11.x)
python --version

# 4. Verify PyTorch and CUDA availability (TESTED: PASSED)
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# 5. Verify core scientific libraries (TESTED: PASSED)
python -c "import monai, nibabel, sklearn, matplotlib, pandas, numpy, scipy; print('Core dependencies OK')"
```

---

## 2. Unit Testing & Baseline Verification

Run the formal unit test suite before modifying code or evaluating models:

```powershell
# Run all 71 agent and full-volume unit tests (TESTED: PASSED, ~2.4 seconds)
python -m unittest discover -s src/agent -p "test_*.py"

# Run dataset smoke test (TESTED: PASSED)
python -m src.data.test_dataset --num-samples 3
```

---

## 3. Existing Model Evaluation & Benchmarks

### A. Run Full-Volume Evaluation (Milestone 7 Benchmark)
This command evaluates the 6 comparative systems (U-Net, U-Net++, Attention U-Net, Baseline Agent, Option A, Option B) across all 155 slices of each of the 39 validation cases (6,045 slices/system = 36,270 records `[MEASURED]`):

```powershell
# TESTED: PASSED (Execution time: ~15.5 minutes on RTX 4070, Peak VRAM: 350 MB)
python -m src.agent.evaluate_full_volume `
    --raw-dir "D:\MajorProject\data\raw" `
    --output-dir "D:\MajorProject\results\agent_evaluation\milestone7_full_volume" `
    --device "cuda"
```
- **Output Artifacts Generated:**
  - `results/agent_evaluation/milestone7_full_volume/slice_evaluations.csv`
  - `results/agent_evaluation/milestone7_full_volume/evaluation_summary.json`
  - `results/agent_evaluation/milestone7_full_volume/report.md`

### B. Dry-Run / Smoke Test Mode (Single Case, 4 Slices)
Use this command to verify pipeline changes quickly without launching the 15-minute full benchmark:

```powershell
# TESTED: PASSED (Execution time: ~3 seconds)
python -m src.agent.evaluate_full_volume `
    --raw-dir "D:\MajorProject\data\raw" `
    --output-dir "D:\MajorProject\results\agent_evaluation\smoke_test_run" `
    --device "cuda" `
    --max-cases 1 `
    --max-slices-per-case 4
```

---

## 4. Local Interactive Demonstration (Streamlit Showcase)

The Phase 3 research demonstration prototype is verified and operational:

```powershell
# Set console encoding to UTF-8 on Windows to avoid Unicode arrow display issues
$env:PYTHONIOENCODING="utf-8"

# Launch Streamlit showcase on http://localhost:8501 (TESTED: PASSED)
& "D:\MajorProject\.venv\Scripts\python.exe" -m streamlit run tools/demo_app.py
```

---

## 5. Controlled Model Training Procedures (Manual Terminal Runs Only)

All heavy GPU training jobs (Phase B pilots and Phase E full-scale training) must be executed manually in PowerShell with explicit user authorization. **Do not run training inside background agent tasks. Never run concurrent GPU training jobs.**

### A. Phase B Architecture Pilot Runs (5 Epochs, 158 Cases)
*Requires Gate 2 approval.*
```powershell
# ResUNet-2D Pilot (~45 min [ESTIMATE])
python -m src.training.train --model-name resunet --epochs 5 --batch-size 16 --results-dir results/pilot_resunet_5ep

# TransUNet Pilot (~1.8h [ESTIMATE])
python -m src.training.train --model-name transunet --epochs 5 --batch-size 8 --results-dir results/pilot_transunet_5ep
```

### B. Phase E Full-Scale Training Runs (~875 Cases [ESTIMATE])
*Strictly BLOCKED until Gates 1–5 pass and explicit user approval is granted.*
```powershell
# Sequential staged runs (save to separate directories):
# Standard U-Net (~3.7 hours [ESTIMATE])
python -m src.training.train --model-name unet --epochs 5 --batch-size 16 --data-manifest data/splits/train_manifest.json --results-dir experiments/full_scale_unet

# U-Net++ (~6.2 hours [ESTIMATE])
python -m src.training.train --model-name unetpp --epochs 5 --batch-size 16 --data-manifest data/splits/train_manifest.json --results-dir experiments/full_scale_unetpp

# Attention U-Net (~4.3 hours [ESTIMATE])
python -m src.training.train --model-name attention_unet --epochs 5 --batch-size 16 --data-manifest data/splits/train_manifest.json --results-dir experiments/full_scale_attnunet

# ResUNet-2D (~4.1 hours [ESTIMATE])
python -m src.training.train --model-name resunet --epochs 5 --batch-size 16 --data-manifest data/splits/train_manifest.json --results-dir experiments/full_scale_resunet
```

---

## 6. Report Compilation Workflow

The project's Word thesis is programmatically generated from benchmark data and figures:

```powershell
# Build Final Word Thesis Document (TESTED: PASSED)
python tools/build_final_report.py
```
- **Target Output:** `D:\MajorProject\Final_Report_Brain_Tumor_Segmentation.docx`

---

## 7. Decoupled Full-Stack Web Application (Phase H)

Once implemented, the full-stack system is launched via two services:

```powershell
# Terminal 1: Launch FastAPI Backend Server
cd D:\MajorProject
.\.venv\Scripts\Activate.ps1
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Launch React + Vite Frontend
cd D:\MajorProject\web_app
npm run dev
# Browser opens to http://localhost:5173
```

---

## 8. Recovery & Troubleshooting Procedures

| Symptom / Error | Root Cause | Recovery Procedure |
|---|---|---|
| `CUDA out of memory` | VRAM exceeded by batch size or 3D volume | Ensure `batch_size = 16` for standard 2D or `8` for TransUNet. In full-volume evaluation, verify `torch.cuda.empty_cache()` after each case. |
| `ModuleNotFoundError: No module named 'src'` | Running script outside project root or without `sys.path` injection | Run with `python -m src.module.path` from `D:\MajorProject`, or verify `PROJECT_ROOT` is prepended to `sys.path`. |
| Windows CP1252 Unicode Error in Streamlit | Streamlit CLI prints Unicode arrows incompatible with Windows default console | Prepend `$env:PYTHONIOENCODING="utf-8"` in PowerShell before launching Streamlit. |
| JSON serialization error: `Out of range float values` | Metrics contained `NaN` or `Infinity` from zero-foreground slices | Always process output dictionaries through `src.agent.evaluate_full_volume.sanitize_for_json()` before `json.dump()`. |
