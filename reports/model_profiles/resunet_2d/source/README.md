# SOURCE CODE & ASSET GENERATION PIPELINE: ResUNet-2D
**Model:** ResUNet-2D (`ResUNet2D`)  
**Directory:** `reports/model_profiles/resunet_2d/source/`  
**Workspace:** `D:\MajorProject`

This directory contains the complete, automated Python and PowerShell scripts used to audit evidence and generate all deliverables for the ResUNet-2D model profile package.

---

## SCRIPT INVENTORY & RESPONSIBILITIES

| Script Filename | Language / Runtime | Output Destination | Description & Capabilities |
|:---|:---:|:---|:---|
| **`generate_technical_assets.py`** | Python 3.11 (`.venv`) | `../assets/technical/` | Generates all 25 high-resolution (300 DPI) scientific figures (`01_` to `25_`), covering architecture diagrams, loss curves, calibration reliability diagrams, entropy distributions, and C3 decision flowcharts. |
| **`generate_non_technical_assets.py`** | Python 3.11 (`.venv`) | `../assets/non_technical/` | Generates 16 intuitive 16:9 widescreen cards (`simple_01` to `simple_16`) explaining concepts simply for non-specialists, junior teammates, and viva examiners. |
| **`generate_qualitative_assets.py`** | Python 3.11 (`.venv`) | `../assets/qualitative_examples/` | Runs live calibrated ResUNet inference on 3 internal validation patients (`BraTS-GLI-01069-000`, `BraTS-GLI-00008-000`, `BraTS-GLI-00753-001`) to generate publication-grade multi-panel figures with error maps and calibrated entropy heatmaps. |
| **`generate_reusable_icons.py`** | Python 3.11 (`.venv`) | `../assets/reusable_icons/` | Generates 6 transparent badge/icon graphics (`icon_accept.png`, `icon_fallback.png`, `icon_flag.png`, etc.) for presentations and documentation. |
| **`generate_one_page_overview.py`** | Python 3.11 (`.venv`) | `../one_page/` | Generates the definitive A3 Landscape (16.54 x 11.69 in) One-Page Overview in high-res PNG (300 DPI), vector PDF, and vector SVG. |
| **`generate_explainer_deck.ps1`** | PowerShell (Windows 11) | `../presentation/` | Leverages PowerPoint COM automation to construct the 8-slide widescreen (16:9) presentation deck (`ResUNet_2D_Explainer_Deck.pptx`) with embedded images and complete speaker scripts. |
| **`build_resunet_dossier_v1_1.py`** | Python 3.11 + Word COM | `../dossier/` | Generates the master 20-section Dossier v1.1 in Markdown (`.md`), Word Master (`.docx`), and converts it to PDF (`.pdf`) via Microsoft Word COM automation. |

---

## EXECUTION ORDER & REPRODUCTION INSTRUCTIONS

To reproduce the entire ResUNet-2D deliverable package from scratch, execute the scripts in the following sequence from the repository root (`D:\MajorProject`):

```powershell
# 1. Activate Python virtual environment
.venv\Scripts\Activate.ps1

# 2. Generate the 25 Technical Scientific Assets
python reports\model_profiles\resunet_2d\source\generate_technical_assets.py

# 3. Generate the 16 Non-Technical Explainer Cards
python reports\model_profiles\resunet_2d\source\generate_non_technical_assets.py

# 4. Generate the 3 Qualitative Example Panels
python reports\model_profiles\resunet_2d\source\generate_qualitative_assets.py

# 5. Generate Reusable Icons and Badges
python reports\model_profiles\resunet_2d\source\generate_reusable_icons.py

# 6. Generate the One-Page Model Overview (PNG, PDF, SVG)
python reports\model_profiles\resunet_2d\source\generate_one_page_overview.py

# 7. Generate the 8-Slide PowerPoint Presentation Deck
powershell -ExecutionPolicy Bypass -File reports\model_profiles\resunet_2d\source\generate_explainer_deck.ps1

# 8. Build the Master Dossier v1.1 (Markdown, DOCX, and PDF)
python reports\model_profiles\resunet_2d\source\build_resunet_dossier_v1_1.py
```

---

## DEPENDENCIES & ENVIRONMENT REQUIREMENTS

- **Python Version:** Python 3.11.9 (`.venv`)
- **Key Python Packages:** `torch`, `torchvision`, `numpy`, `matplotlib`, `python-docx`, `nibabel`, `scipy`
- **Host Platform:** Windows 11 Pro / Enterprise
- **Office Automation:** Microsoft Office 16 (Word & PowerPoint COM interfaces enabled)
- **GPU Acceleration:** NVIDIA CUDA 12.8 (RTX 4070 Laptop GPU supported, CPU fallback available)
