"""Complete ResUNet-2D Model Dossier Generator.
Generates Markdown, Word DOCX, and PDF editions.

Workspace: D:\MajorProject
Output Directory: reports/model_profiles/resunet_2d/
"""

import os
import sys
import json
import hashlib
import pandas as pd
import numpy as np
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

OUTPUT_DIR = r'D:\MajorProject\reports\model_profiles\resunet_2d'
ASSETS_DIR = os.path.join(OUTPUT_DIR, 'assets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MD_PATH = os.path.join(OUTPUT_DIR, 'ResUNet_2D_Model_Dossier_v1.0.md')
DOCX_PATH = os.path.join(OUTPUT_DIR, 'ResUNet_2D_Model_Dossier_v1.0.docx')
PDF_PATH = os.path.join(OUTPUT_DIR, 'ResUNet_2D_Model_Dossier_v1.0.pdf')

# Palette Constants
COLOR_NAVY_HEX = "1F3A60"
COLOR_SLATE_HEX = "475569"
COLOR_TEAL_HEX = "0D9488"
COLOR_CYAN_HEX = "0284C7"
COLOR_CORAL_HEX = "E11D48"
COLOR_AMBER_HEX = "D97706"
COLOR_LIGHT_BG_HEX = "F8FAFC"
COLOR_BORDER_HEX = "CBD5E1"
COLOR_CARD_BG_HEX = "F1F5F9"

NAVY_RGB = RGBColor(31, 58, 96)
SLATE_RGB = RGBColor(71, 85, 105)
TEAL_RGB = RGBColor(13, 148, 136)
CYAN_RGB = RGBColor(2, 132, 199)
DARK_RGB = RGBColor(15, 23, 42)


# -----------------------------------------------------------------------------
# XML Helper Functions for DOCX styling
# -----------------------------------------------------------------------------
def set_cell_background(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_border(cell, **kwargs):
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        edge_data = kwargs.get(edge)
        if edge_data:
            tag = f'w:{edge}'
            element = OxmlElement(tag)
            element.set(qn('w:val'), edge_data.get('val', 'single'))
            element.set(qn('w:sz'), str(edge_data.get('sz', 4)))
            element.set(qn('w:space'), str(edge_data.get('space', 0)))
            element.set(qn('w:color'), edge_data.get('color', 'CBD5E1'))
            tcBorders.append(element)
    tcPr.append(tcBorders)

def add_callout_box(doc, title, text, border_color_hex=COLOR_TEAL_HEX, bg_color_hex=COLOR_LIGHT_BG_HEX):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.columns[0].width = Inches(6.5)

    cell = table.cell(0, 0)
    set_cell_background(cell, bg_color_hex)
    set_cell_margins(cell, top=140, bottom=140, left=180, right=180)
    set_cell_border(cell,
                    left=dict(val='single', sz=24, color=border_color_hex),
                    top=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                    bottom=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                    right=dict(val='single', sz=4, color=COLOR_BORDER_HEX))

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    run_t = p.add_run(title + "\n")
    run_t.font.name = 'Segoe UI'
    run_t.font.size = Pt(10.0)
    run_t.font.bold = True
    run_t.font.color.rgb = NAVY_RGB

    run_b = p.add_run(text)
    run_b.font.name = 'Segoe UI'
    run_b.font.size = Pt(9.0)
    run_b.font.color.rgb = SLATE_RGB

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def add_metric_card_row(doc, cards):
    cols = len(cards)
    table = doc.add_table(rows=1, cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    col_width = Inches(6.5 / cols)

    for i, (title, value, subtitle) in enumerate(cards):
        cell = table.cell(0, i)
        cell.width = col_width
        set_cell_background(cell, COLOR_CARD_BG_HEX)
        set_cell_margins(cell, top=120, bottom=120, left=80, right=80)
        set_cell_border(cell,
                        top=dict(val='single', sz=12, color=COLOR_NAVY_HEX),
                        bottom=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                        left=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                        right=dict(val='single', sz=4, color=COLOR_BORDER_HEX))

        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2)

        run_t = p.add_run(title.upper() + "\n")
        run_t.font.name = 'Segoe UI'
        run_t.font.size = Pt(7.5)
        run_t.font.bold = True
        run_t.font.color.rgb = SLATE_RGB

        run_v = p.add_run(value + "\n")
        run_v.font.name = 'Segoe UI'
        run_v.font.size = Pt(13.0)
        run_v.font.bold = True
        run_v.font.color.rgb = NAVY_RGB

        run_s = p.add_run(subtitle)
        run_s.font.name = 'Segoe UI'
        run_s.font.size = Pt(7.5)
        run_s.font.color.rgb = TEAL_RGB

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def format_table(table, col_widths, col_alignments, header_bg=COLOR_NAVY_HEX):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False

    header_row = table.rows[0]
    trPr = header_row._tr.get_or_add_trPr()
    trPr.append(OxmlElement('w:tblHeader'))

    for j, cell in enumerate(header_row.cells):
        cell.width = col_widths[j]
        set_cell_background(cell, header_bg)
        set_cell_margins(cell, top=90, bottom=90, left=90, right=90)
        set_cell_border(cell,
                        top=dict(val='single', sz=6, color=COLOR_NAVY_HEX),
                        bottom=dict(val='single', sz=12, color=COLOR_NAVY_HEX),
                        left=dict(val='none'), right=dict(val='none'))
        p = cell.paragraphs[0]
        p.alignment = col_alignments[j]
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        for r in p.runs:
            r.font.name = 'Segoe UI'
            r.font.size = Pt(8.5)
            r.font.bold = True
            r.font.color.rgb = RGBColor(255, 255, 255)

    for i, row in enumerate(table.rows[1:], start=1):
        bg = COLOR_LIGHT_BG_HEX if i % 2 == 1 else "FFFFFF"
        for j, cell in enumerate(row.cells):
            cell.width = col_widths[j]
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=90, right=90)
            set_cell_border(cell,
                            top=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                            bottom=dict(val='single', sz=4, color=COLOR_BORDER_HEX),
                            left=dict(val='none'), right=dict(val='none'))
            p = cell.paragraphs[0]
            p.alignment = col_alignments[j]
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            for r in p.runs:
                r.font.name = 'Segoe UI'
                r.font.size = Pt(8.0)
                r.font.color.rgb = DARK_RGB

def add_heading_styled(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    if level == 1:
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(5)
        for r in h.runs:
            r.font.name = 'Segoe UI'
            r.font.size = Pt(13)
            r.font.bold = True
            r.font.color.rgb = NAVY_RGB
    elif level == 2:
        h.paragraph_format.space_before = Pt(10)
        h.paragraph_format.space_after = Pt(4)
        for r in h.runs:
            r.font.name = 'Segoe UI'
            r.font.size = Pt(11)
            r.font.bold = True
            r.font.color.rgb = TEAL_RGB
    elif level == 3:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
        for r in h.runs:
            r.font.name = 'Segoe UI'
            r.font.size = Pt(9.5)
            r.font.bold = True
            r.font.color.rgb = SLATE_RGB
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Segoe UI'
        r_pre.font.size = Pt(9.0)
        r_pre.font.bold = True
        r_pre.font.color.rgb = DARK_RGB
    r = p.add_run(text)
    r.font.name = 'Segoe UI'
    r.font.size = Pt(9.0)
    r.font.color.rgb = DARK_RGB
    return p

def add_figure_with_caption(doc, image_path, caption_text, width=Inches(6.2)):
    if os.path.exists(image_path):
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(2)
        p_img.paragraph_format.keep_with_next = True
        p_img.add_run().add_picture(image_path, width=width)

        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(2)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = 'Segoe UI'
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = SLATE_RGB


# -----------------------------------------------------------------------------
# BUILD MARKDOWN SOURCE DOSSIER
# -----------------------------------------------------------------------------
def build_markdown_dossier():
    with open('results/full_training_15ep/resunet_2d/summary.json') as f:
        summary = json.load(f)
    with open('results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json') as f:
        cal_params = json.load(f)

    md = """# MODEL DOSSIER 01 — RESUNET-2D
## Comprehensive Architecture, Training, Performance, Calibration & Reliability Report

**Project:** Reliability-Aware AI Agent for Brain Tumor Segmentation  
**Workspace:** `D:\\MajorProject`  
**Model Name:** ResUNet-2D  
**Architecture Role:** Primary Champion Segmentation Model / Core Reliability Agent Engine  
**Checkpoint Profile:** `STAGE_C_15EP`  
**Best Checkpoint Epoch:** Epoch 13  
**Checkpoint SHA-256:** `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d`  
**Optimal Calibrated Temperature:** $T^* = 2.8804590702056885$  
**Evaluation Cohort:** Held-out `Val_eval` (105 Cases, 16,275 Slices, Subject-isolated from `Val_cal`)  
**Document Version:** v1.0 (Audited Scientific Dossier)  
**Date of Audit:** November 18, 2026  

---

## 1. EXECUTIVE MODEL SNAPSHOT

ResUNet-2D serves as the authoritative **Primary Champion** model within the five-model portfolio of the Reliability-Aware AI Agent for Brain Tumor Segmentation. Engineered as a 5-stage deep residual U-Net, ResUNet-2D combines pre-activation residual convolutional units with Group Normalization ($G=8$) to enable stable, high-fidelity segmentation across the multi-parametric MRI modalities of the BraTS-GLI benchmark.

| Metric Dimension | Authoritative Value | Benchmark Context & Significance |
|---|---:|---|
| **Total Trainable Parameters** | **8,114,604** | 100% trainable; compact footprint (92.97 MB checkpoint). |
| **Best Training Epoch** | **Epoch 13** | Validation Macro Mean Dice reaches global maximum of **0.8670**. |
| **Volumetric Mean Dice (`Val_eval`)** | **0.8641** | 95% CI [0.8265, 0.8930] across 105 held-out evaluation volumes. |
| **Whole Tumor (WT) Dice** | **0.9039** | Bounded HD95: 15.29 mm (Median 4.12 mm, 95% CI [11.38, 20.05] mm). |
| **Tumor Core (TC) Dice** | **0.8597** | Bounded HD95: 8.98 mm (Median 3.16 mm, 95% CI [6.08, 12.74] mm). |
| **Enhancing Tumor (ET) Dice** | **0.8286** | Bounded HD95: 9.92 mm (Median 2.00 mm, 95% CI [4.71, 18.66] mm). |
| **Optimal Temperature ($T^*$)** | **2.880459** | Fitted via L-BFGS on `Val_cal` (50 cases, 5,020,730 sampled voxels). |
| **Uncalibrated ECE (`Val_eval`)** | **8.59%** | Pre-calibration baseline exhibits pronounced clinical overconfidence. |
| **Calibrated ECE (`Val_eval`)** | **3.41%** | **60.32% relative reduction**; ECE drops to **0.84%** on `Val_cal`. |
| **Calibrated NLL (`Val_eval`)** | **0.4415** | Reduced from 1.0277 (**57.04% relative reduction**). |
| **Foreground Predictive Entropy** | **0.2686 ± 0.0903** | Median: 0.2474, IQR: [0.1995, 0.3185], Range: [0.0000, 0.6146]. |
| **Agent Action: ACCEPT ($H < 0.35$)** | **5,619 slices (34.53%)** | 83.02% of all ground-truth tumor slices retained with Mean Dice 0.7855. |
| **Agent Action: FALLBACK_SWITCH** | **147 slices (0.90%)** | Rescues 54 ground-truth tumor slices missed by ResUNet-2D via UNetResNet34. |
| **Agent Retained Quality Impact** | **+0.0653 Mean Dice** | Retained Dice improves from 0.7150 (unfiltered) to **0.7803**; severe failures drop by 34.1%. |

---

## 2. MODEL IDENTITY & CONFIGURATION

The detailed execution configuration of ResUNet-2D was preserved in `results/full_training_15ep/resunet_2d/config.json` and `src/models/resunet.py`.

- **Official Model Name:** `resunet_2d` (ResUNet-2D)
- **Model Family:** 2D Deep Residual Convolutional Encoder-Decoder (U-Net derivative)
- **Research Role:** Primary Champion Segmentation Model; primary inference engine for the Stage C Reliability Agent.
- **Checkpoint Profile:** `STAGE_C_15EP`
- **Architecture Class:** `src.models.resunet.ResUNet2D`
- **Input Channels:** 4 co-registered MRI modalities:
  1. Channel 0: Native T1-weighted (`t1n`)
  2. Channel 1: Post-contrast T1-weighted (`t1c`)
  3. Channel 2: T2-weighted (`t2w`)
  4. Channel 3: T2 Fluid-Attenuated Inversion Recovery (`t2f` / FLAIR)
- **Output Classes:** 4 mutually exclusive voxel categories:
  - Class 0: Background / Non-tumor tissue (`BG`)
  - Class 1: Necrotic Tumor Core (`NCR`)
  - Class 2: Peritumoral Edematous Tissue (`ED`)
  - Class 3: GD-Enhancing Tumor (`ET`)
- **Total Trainable Parameters:** 8,114,604 (100% requires_grad = True).
- **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.01).
- **Learning Rate Schedule:** Base learning rate $\eta = 0.001$, 2-epoch linear warmup followed by Cosine Annealing decay down to $\eta_{\text{min}} = 2.44 \times 10^{-5}$ at Epoch 15.
- **Gradient Clipping:** Maximum gradient norm threshold $1.0$ (monitored pre- and post-clip).
- **Loss Function:** Compound Loss: $\mathcal{L} = \mathcal{L}_{\text{Dice}} + \mathcal{L}_{\text{CE}}$ (unweighted sum of soft multi-class Dice loss and cross-entropy loss).
- **Batch Size:** 2 slices (2D axial slices sampled per mini-batch).
- **Dataset Census:** BraTS-GLI dataset.
  - Training cohort: 721 cases (59,020 axial slices).
  - Validation cohort: 155 cases (12,364 axial slices).
  - Split manifest SHA-256: `e753caa256950dee0ed921ee4e8a08f6f515dcb062862c1fb7a699e08f69059d`.
- **Hardware Platform:** NVIDIA GeForce RTX 4070 Laptop GPU (8,192 MB VRAM), CUDA 12.8.
- **Runtime Environment:** Windows 11, Python 3.11.9, PyTorch 2.11.0+cu128.
- **Checkpoint Selection Methodology:** The model checkpoint was selected based on the highest validation macro mean Dice score across 15 full training epochs. Epoch 13 achieved **0.867045**, outperforming both Epoch 14 (0.866129) and the final Epoch 15 (0.863087).

---

## 3. ARCHITECTURE & DATA FLOW

The ResUNet-2D architecture implements a symmetric 5-stage U-Net topology enhanced with pre-activation residual blocks and Group Normalization.

![ResUNet-2D Architecture](assets/architecture_diagram.png)

### 3.1 Pre-Activation Residual Convolutional Block
Standard residual blocks place normalization and activation layers after convolution, which obstructs the identity shortcut with non-linear operations. ResUNet-2D utilizes full **Pre-Activation Residual Blocks** (`PreActResidualBlock`):
$$\mathcal{F}(x) = \text{Conv}_{3\times 3}(\text{ReLU}(\text{GN}(\text{Conv}_{3\times 3}(\text{ReLU}(\text{GN}(x))))))$$
$$\text{Output} = \mathcal{F}(x) + \text{Shortcut}(x)$$
Where $\text{Shortcut}(x)$ is a $1\times 1$ convolution with matching stride if channel dimensions change or downsampling occurs, and the identity operator otherwise. This preserves an unobstructed linear gradient highway throughout backpropagation.

### 3.2 Group Normalization vs. Batch Normalization
Medical image segmentation models trained on high-resolution 2D slices ($240 \times 240 \times 4$) are constrained to small mini-batch sizes ($B=2$) due to GPU memory constraints. Under $B=2$, Batch Normalization exhibits high variance in its mini-batch mean and variance estimates, degrading stability. ResUNet-2D employs **Group Normalization** with $G=8$ groups. Group Normalization computes statistics per instance across channel groups, ensuring training and inference dynamics remain entirely invariant to batch size.

### 3.3 Stage-by-Stage Parameter Breakdown

| Module Stage | In Channels | Out Channels | Spatial Dim | Stride | Parameter Count | % Total |
|---|---:|---:|---:|---:|---:|---:|
| **Encoder 1** | 4 | 32 | $240 \times 240$ | 1 | 10,632 | 0.13% |
| **Encoder 2** | 32 | 64 | $120 \times 120$ | 2 | 57,664 | 0.71% |
| **Encoder 3** | 64 | 128 | $60 \times 60$ | 2 | 230,016 | 2.83% |
| **Encoder 4** | 128 | 256 | $30 \times 30$ | 2 | 918,784 | 11.32% |
| **Bottleneck (Level 5)** | 256 | 512 | $15 \times 15$ | 2 | 3,672,576 | 45.26% |
| **Upconv 4 + Decoder 4** | 512 + 256 | 256 | $30 \times 30$ | 2 (transp) | 2,427,136 | 29.91% |
| **Upconv 3 + Decoder 3** | 256 + 128 | 128 | $60 \times 60$ | 2 (transp) | 607,360 | 7.49% |
| **Upconv 2 + Decoder 2** | 128 + 64 | 64 | $120 \times 120$ | 2 (transp) | 152,128 | 1.87% |
| **Upconv 1 + Decoder 1** | 64 + 32 | 32 | $240 \times 240$ | 2 (transp) | 38,176 | 0.47% |
| **Classifier Head** | 32 | 4 | $240 \times 240$ | 1 | 132 | 0.00% |
| **Total Model** | — | — | — | — | **8,114,604** | **100.00%** |

*Crucial Design Feature:* Encoder Level 1 operates with stride 1, preserving the native $240 \times 240$ spatial resolution without early downsampling. This prevents fine enhancing tumor (ET) margins and small satellite lesions from being lost in the earliest receptive field.

---

## 4. 15-EPOCH TRAINING PERFORMANCE ANALYSIS

The training trajectory of ResUNet-2D was audited directly from `results/full_training_15ep/resunet_2d/history.csv`. Total cumulative training time was **8.0851 hours** (485.10 minutes), with average epoch durations of 32.3 minutes.

![Training Trajectory Curves](assets/training_curves.png)

### 4.1 Complete 15-Epoch Numerical History

| Epoch | Train Loss | Val Loss | Mean Dice | Mean IoU | BG Dice | NCR Dice | ED Dice | ET Dice | Clipped Batches | Learning Rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.306681 | 0.316821 | 0.806517 | 0.679018 | 0.997904 | 0.734574 | 0.824143 | 0.860835 | 12,764 | 0.000100 |
| 2 | 0.258361 | 0.375369 | 0.775118 | 0.633851 | 0.997890 | 0.734534 | 0.809827 | 0.780992 | 5,464 | 0.000550 |
| 3 | 0.237095 | 0.297229 | 0.802887 | 0.671905 | 0.997920 | 0.759683 | 0.810975 | 0.838003 | 2,922 | 0.001000 |
| 4 | 0.222604 | 0.266618 | 0.823700 | 0.703768 | 0.998326 | 0.749168 | 0.845808 | 0.876123 | 2,036 | 0.000986 |
| 5 | 0.215468 | 0.251587 | 0.838288 | 0.723659 | 0.998388 | 0.784334 | 0.848862 | 0.881668 | 1,835 | 0.000943 |
| 6 | 0.208286 | 0.250357 | 0.839315 | 0.725064 | 0.998384 | 0.785766 | 0.853319 | 0.878861 | 1,655 | 0.000876 |
| 7 | 0.202484 | 0.238729 | 0.853782 | 0.746555 | 0.998477 | 0.811942 | 0.850589 | 0.898813 | 1,497 | 0.000786 |
| 8 | 0.202149 | 0.269076 | 0.815500 | 0.690143 | 0.998538 | 0.764368 | 0.853166 | 0.828966 | 1,518 | 0.000681 |
| 9 | 0.196123 | 0.244924 | 0.831809 | 0.713379 | 0.998214 | 0.796827 | 0.823708 | 0.874892 | 1,427 | 0.000565 |
| 10 | 0.193770 | 0.224416 | 0.855294 | 0.749435 | 0.998620 | 0.799873 | 0.866533 | 0.899476 | 1,542 | 0.000445 |
| 11 | 0.189466 | 0.214959 | 0.856866 | 0.751846 | 0.998639 | 0.801259 | 0.868664 | 0.900675 | 1,509 | 0.000329 |
| 12 | 0.183335 | 0.219434 | 0.857094 | 0.751479 | 0.998590 | 0.813922 | 0.860037 | 0.897322 | 1,597 | 0.000224 |
| **13 (Best)** | **0.180275** | **0.207292** | **0.867045** | **0.766909** | **0.998738** | **0.820963** | **0.876465** | **0.903708** | **1,599** | **0.000134** |
| 14 | 0.178053 | 0.206310 | 0.866129 | 0.765715 | 0.998718 | 0.817413 | 0.874477 | 0.906497 | 1,700 | 0.000067 |
| 15 | 0.176589 | 0.208839 | 0.863087 | 0.761295 | 0.998717 | 0.809666 | 0.873938 | 0.905658 | 1,807 | 0.000024 |

![Class-wise Progression](assets/class_dice_curves.png)

### 4.2 Training Dynamics and Convergence Assessment
1. **Warmup Phase (Epochs 1–2):** Learning rate linearly increases from $0.0001$ to $0.0010$. The initial batch gradient norm is high, clipping 12,764 batches in Epoch 1, quickly stabilizing to 5,464 by Epoch 2.
2. **Rapid Feature Acquisition (Epochs 3–7):** Train loss rapidly declines from 0.2371 to 0.2025. Validation mean Dice rises from 0.8029 to 0.8538.
3. **Transient Instability at Epoch 8:** Validation loss spikes temporarily to 0.2691 and Dice drops to 0.8155, driven by an acute dip in Enhancing Tumor Dice (0.8290) and Necrotic Core Dice (0.7644).
4. **Peak Optimization Window (Epochs 11–15):** The model enters asymptotic stability. Training loss decreases steadily from 0.1895 (Epoch 11) to 0.1766 (Epoch 15). Validation loss reaches its global minimum at Epoch 14 (0.206310). Peak validation mean Dice occurs at **Epoch 13 (0.867045)**.
5. **Scientific Caution on Convergence:** While training loss continued a slight monotonic downward drift through Epoch 15, validation loss stabilized between 0.2063 and 0.2088 and validation Dice plateaued between 0.8631 and 0.8670. This denotes empirical stability and diminishing returns for 2D slice training rather than absolute mathematical convergence. Epoch 13 represents the optimal generalization point on the validation frontier.

---

## 5. SEGMENTATION PERFORMANCE

ResUNet-2D was evaluated on the held-out `Val_eval` partition (105 cases, 16,275 slices, 96 unique subjects).

### 5.1 Authoritative 3D Volumetric Segmentation Metrics

| Segmentation Sub-Region | Volumetric Dice | 95% Confidence Interval | Bounded HD95 (mm) | HD95 95% CI (mm) | Median HD95 (mm) |
|---|---:|---:|---:|---:|---:|
| **Whole Tumor (WT)** | **0.9039** | [0.8765, 0.9241] | **15.29** | [11.38, 20.05] | **4.12** |
| **Tumor Core (TC)** | **0.8597** | [0.8076, 0.8986] | **8.98** | [6.08, 12.74] | **3.16** |
| **Enhancing Tumor (ET)** | **0.8286** | [0.7794, 0.8684] | **9.92** | [4.71, 18.66] | **2.00** |
| **Macro Volumetric Mean** | **0.8641** | [0.8265, 0.8930] | **11.40** | [7.39, 17.15] | **3.09** |

*Important Clarification of Evaluation Protocols:*
- **Volumetric Mean Dice (0.8641):** Computed by accumulating true-positive, false-positive, and false-negative voxel counts across the full 3D volume for each case before calculating the harmonic mean. Larger slices naturally carry higher weight.
- **Epoch 13 Checkpoint-Selection Dice (0.8670):** Computed across validation slices during training using running batch tallies.
- **Slice-Level Unfiltered Baseline Dice (0.7150):** Computed as the arithmetic mean of 2D Dice scores across individual slices. Empty slices and small peripheral tumor margins with few pixels drag this unweighted mean lower.

### 5.2 Case-Level Statistical Distributions (N=105 Cases)

| Metric | Mean | Std Dev | Min | 25% (Q1) | 50% (Median) | 75% (Q3) | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| **WT Dice** | 0.9039 | 0.1122 | 0.4146 | 0.8986 | 0.9464 | 0.9661 | 0.9831 |
| **TC Dice** | 0.8597 | 0.2204 | 0.0000 | 0.8932 | 0.9419 | 0.9633 | 1.0000 |
| **ET Dice** | 0.8286 | 0.2143 | 0.0000 | 0.8284 | 0.9092 | 0.9420 | 1.0000 |
| **WT Bounded HD95 (mm)** | 15.29 | 22.28 | 1.00 | 2.00 | 4.12 | 18.83 | 90.57 |
| **TC Bounded HD95 (mm)** | 8.98 | 16.26 | 0.00 | 1.73 | 3.16 | 7.28 | 81.82 |
| **ET Bounded HD95 (mm)** | 9.92 | 38.16 | 0.00 | 1.00 | 2.00 | 5.00 | 373.12 |

### 5.3 Case Performance Extremes
- **Top Performing Cases:**
  1. `BraTS-GLI-01335-000`: WT Dice **0.9831**, TC Dice **0.9534**, ET Dice **0.9391** (Large, well-circumscribed lesion with sharp contrast enhancement).
  2. `BraTS-GLI-01069-000`: WT Dice **0.9819**, TC Dice **0.9839**, ET Dice **0.9665**.
  3. `BraTS-GLI-00187-000`: WT Dice **0.9809**, TC Dice **0.9845**, ET Dice **0.9581**.
- **Challenging Cases (Failure Modes):**
  1. `BraTS-GLI-00753-001`: WT Dice **0.4146**, TC Dice **0.3091**, ET Dice **0.2690** (Diffuse, infiltrative non-enhancing edema with poorly demarcated boundaries).
  2. `BraTS-GLI-01119-000`: WT Dice **0.4232**, TC Dice **0.7029**, ET Dice **0.6867** (Extensive edema under-segmented on FLAIR).
  3. `BraTS-GLI-00331-000`: WT Dice **0.4479**, TC Dice **0.0536**, ET Dice **0.4064** (Tiny necrotic focus of <50 voxels missed in multiple axial slices).

---

## 6. PROBABILITY CALIBRATION ANALYSIS (STAGE C2)

Deep neural networks trained with cross-entropy and soft Dice loss are notoriously overconfident. ResUNet-2D underwent formal post-hoc **Temperature Scaling** in Stage C2.

![Calibration Reliability Diagrams](assets/calibration_comparison.png)

### 6.1 Temperature Scaling Mathematical Formulation
Given unnormalized model logit vector $z(x) \in \mathbb{R}^K$ for input slice $x$ over $K=4$ classes, the calibrated posterior probability distribution is computed via scalar temperature $T > 0$:
$$\hat{p}_k(x; T) = \frac{\exp(z_k(x) / T)}{\sum_{j=1}^K \exp(z_j(x) / T)}$$
Because $T$ is a strictly positive scalar, the transformation is strictly monotonic:
$$\operatorname{argmax}_{k} \hat{p}_k(x; T) = \operatorname{argmax}_{k} z_k(x)$$
**Argmax Invariance:** Calibrated probabilities adjust prediction confidence without altering the hard discrete segmentation mask. In Stage C2 verification, exactly **0 mismatches** were observed across all 16,275 slices.

### 6.2 Fitting Diagnostics
- **Fitting Cohort (`Val_cal`):** 50 cases, 47 subjects, 5,020,730 sampled foreground and boundary voxels.
- **Optimization Algorithm:** L-BFGS with strong Wolfe line search.
- **Initial State:** $T_0 = 1.0$, Initial NLL = 0.420767.
- **Convergence:** Terminated successfully in **36 iterations**.
- **Fitted Optimal Temperature:**
  $$T^* = 2.8804590702056885$$
- **Final Fitting NLL:** 0.2173115313053131.

### 6.3 Calibration Performance Comparison

| Metric | Val_cal Uncalibrated | Val_cal Calibrated ($T^*$) | Relative Change | Val_eval Uncalibrated | Val_eval Calibrated ($T^*$) | Relative Change |
|---|---:|---:|---:|---:|---:|---:|
| **Negative Log-Likelihood (NLL)** | 0.4154 | 0.2173 | **−47.69%** | 1.0277 | 0.4415 | **−57.04%** |
| **Expected Calibration Error (ECE)** | 5.02% | 0.84% | **−83.25%** | 8.59% | 3.41% | **−60.32%** |
| **Maximum Calibration Error (MCE)** | 28.11% | 7.26% | **−74.17%** | 30.73% | 26.42% | **−14.03%** |
| **Brier Score** | 0.1122 | 0.0994 | **−11.37%** | 0.1851 | 0.1660 | **−10.32%** |

*Key Findings:* Pre-calibration ResUNet-2D probabilities exhibited severe overconfidence (uncalibrated ECE 8.59% on `Val_eval`). Scaling by $T^* = 2.8805$ softens the probability distribution, pulling high-confidence bins into alignment with empirical segmentation accuracy.

---

## 7. PREDICTIVE UNCERTAINTY ANALYSIS

Predictive uncertainty provides the core gating mechanism for autonomous triage in the Stage C reliability agent.

![Predictive Uncertainty Distribution](assets/entropy_distribution.png)

### 7.1 Normalized Shannon Entropy Formulation
For each voxel $i$ with calibrated posterior class probabilities $\hat{p}_i = [\hat{p}_{i,0}, \dots, \hat{p}_{i,3}]$, voxel-wise predictive entropy is defined as:
$$H_i = -\frac{1}{\ln 4} \sum_{k=0}^3 \hat{p}_{i,k} \ln \hat{p}_{i,k} \quad \in [0, 1]$$
For an axial slice $S$, slice-level foreground predictive entropy $\bar{H}(S)$ is computed as the mean entropy across all voxels assigned to tumor foreground ($k \in \{1, 2, 3\}$):
$$\bar{H}(S) = \frac{1}{|S_{\text{fg}}|} \sum_{i \in S_{\text{fg}}} H_i$$
When $|S_{\text{fg}}| = 0$ (empty prediction), entropy is mathematically undefined and handled via 2.5D axial context validation.

### 7.2 Population Uncertainty Metrics (`Val_eval`, 105 Cases, 16,275 Slices)

| Uncertainty Dimension | Numerical Statistic | Research Interpretation |
|---|---:|---|
| **Total Evaluated Slices** | 16,275 | 105 cases $\times$ 155 axial slices per volume. |
| **Empty Prediction Slices** | 9,401 (57.76%) | Non-tumor slices or severe false negatives; entropy undefined. |
| **Non-Empty Foreground Slices** | 6,874 (42.24%) | Evaluated for foreground predictive entropy. |
| **Mean Foreground Entropy** | **0.2686** | Moderate overall confidence across valid tumor slices. |
| **Standard Deviation** | **0.0903** | Tight distribution with low inter-slice dispersion. |
| **Median Foreground Entropy** | **0.2474** | Centered substantially below the 0.35 policy threshold. |
| **First Quartile (Q1)** | **0.1995** | 25% of non-empty slices have $H \le 0.1995$ (very high certainty). |
| **Third Quartile (Q3)** | **0.3185** | 75% of non-empty slices have $H \le 0.3185$. |
| **Minimum Observed Entropy** | **0.0000** | Perfect certainty on dense, central tumor slices. |
| **Maximum Observed Entropy** | **0.6146** | High uncertainty on ambiguous, infiltrative boundaries. |
| **Non-Empty Slices with $H < 0.35$** | **5,619 (81.74%)** | Direct candidates for autonomous deployment (**ACCEPT**). |
| **Non-Empty Slices with $H \ge 0.35$** | **1,255 (18.26%)** | Automatically flagged for expert review (**FLAG_UNRELIABLE**). |

---

## 8. ROLE IN THE RELIABILITY-AWARE AGENT

ResUNet-2D does not operate in isolation; it serves as the **Primary Champion** inside the frozen Stage C3 v1.1.3 reliability-aware agent.

![Agent Flow Protocol](assets/agent_flow_diagram.png)

### 8.1 Agent Triage Architecture
The live agent coordinates:
- **Primary Model:** ResUNet-2D ($T^* = 2.880459$).
- **Secondary Fallback Model:** UNetResNet34 ($T^* = 2.626091$).
- **Entropy Gating Threshold:** $\tau = 0.35$.
- **2.5D Axial Context Radius:** $k = \pm 2$ (4 neighbor slots: $z-2, z-1, z+1, z+2$). Out-of-volume slots at $z=0$ or $z=154$ count as unavailable.
- **Minimum Axial Foreground Neighbors:** $N_{\text{min}} = 1$.

### 8.2 Three-Action Decision Semantics
1. **`ACCEPT`:**
   Triggered when ResUNet-2D produces a non-empty segmentation and its normalized foreground entropy satisfies $H < 0.35$.
   *Outcome:* 5,619 slices (34.53% of all slices; 83.02% of ground-truth tumor slices). Mask deployed autonomously.
2. **`FALLBACK_SWITCH`:**
   Triggered when ResUNet-2D predicts an empty mask, but 2.5D axial context confirms neighboring tumor ($\ge 1$ neighbor has foreground tumor), and the secondary model (UNetResNet34) detects non-empty tumor.
   *Outcome:* 147 slices (0.90% of all slices; 54 ground-truth tumor slices rescued). Secondary mask deployed. If secondary entropy $H_{\text{sec}} \ge 0.35$, the action remains `FALLBACK_SWITCH` and records a high-uncertainty audit warning.
3. **`FLAG_UNRELIABLE`:**
   Triggered when:
   - ResUNet-2D is non-empty but $H \ge 0.35$ (high uncertainty).
   - ResUNet-2D is empty and axial context confirms tumor, but secondary model is also empty (consensus empty).
   - ResUNet-2D is empty and axial context does not confirm tumor (isolated or true-negative slice).
   *Outcome:* 10,509 slices (64.57% of all slices; includes 9,401 empty-prediction slices without axial support). Triage referral to expert radiologist.

### 8.3 System-Level Reliability Impact on Tumor Slices (N=6,549 Slices)

| Configuration | Retained Coverage (%) | Retained Mean Dice | Referred Mean Dice | Retained Severe Failures (Dice < 0.5) | Severe Failure Rate (%) |
|---|---:|---:|---:|---:|---:|
| **Unfiltered Primary Baseline** | 100.00% (6,549) | 0.7150 | — | 1,343 | 20.51% |
| **Entropy-Only Policy ($\tau=0.35$)** | 83.02% (5,437) | 0.7855 | 0.3702 | 699 | 12.86% |
| **Fallback-Only Policy (No $\tau$)** | 95.63% (6,263) | 0.7499 | 0.0000 | 1,046 | 16.70% |
| **Full Stage C3 v1.1.3 Policy** | **83.84% (5,491)** | **0.7803** | **0.3891** | **742** | **13.51%** |

*Crucial Scientific Synthesis:* The Stage C3 agent improves retained slice quality from 0.7150 to **0.7803** (+0.0653 Dice improvement) and reduces severe failures from 20.51% to 13.51% (**34.1% relative reduction**). The agent does not modify the weights of ResUNet-2D; rather, it selectively routes high-uncertainty and discordant slices to human specialists.

---

## 9. SCIENTIFIC LIMITATIONS & DISCLOSURES

In accordance with academic rigor, the following ten scientific disclosures must be recognized:

1. **Single-Seed Training Limitation:** ResUNet-2D was trained exclusively under seed 42. Variability across alternative stochastic initializations has not been quantified.
2. **Internal Validation Scope:** Performance was evaluated on the internal BraTS-GLI validation cohort. The model has not been validated on external multi-center clinical cohorts with distinct scanner protocols.
3. **Checkpoint-Selection Exposure:** Selection of Epoch 13 was based on the highest validation mean Dice. Slight selection bias on the internal validation split is inherent to this procedure.
4. **Calibration Separation Boundaries:** While `Val_cal` (50 cases) and `Val_eval` (105 cases) are strictly subject-isolated, both subsets belong to the same broader BraTS-GLI data distribution.
5. **Entropy Heuristics:** Shannon entropy measures posterior probability dispersion under the trained distribution; it does not constitute a guaranteed detector for out-of-distribution imaging artifacts or confident false negatives.
6. **High Empty-Prediction Frequency:** 57.76% of all slices contain no tumor. In 2D slice inference, axial context validation is essential to prevent erroneous referral of true-negative slices.
7. **Small-Population MCE Sensitivity:** Maximum Calibration Error on `Val_eval` is 26.42%, influenced by sparse bin counts in intermediate probability intervals.
8. **Missing-Modality Vulnerability:** ResUNet-2D relies heavily on multi-parametric MRI co-registration. Evaluation under missing modalities reveals severe vulnerabilities:
   - Withholding T1c (contrast) drops Tumor Core Dice to **0.3132** and Enhancing Tumor Dice to **0.1684** (Mean Dice collapses to **0.4449**).
   - Withholding T2w drops Whole Tumor Dice to **0.4388** (Mean Dice collapses to **0.6058**).
9. **Academic Prototype Status:** ResUNet-2D and the Stage C agent are research prototypes. They are not approved by the FDA, CDSCO, or any regulatory body for autonomous clinical diagnosis.
10. **Sealed Quarantine Test Cohort Preservation:** The official test cohort (`data/quarantine_test/`) was strictly quarantined and unaccessed throughout training, calibration, and dossier compilation.

---

## 10. ARTIFACT PROVENANCE & SOURCE REGISTER

Every numerical result in this dossier is verifiable against immutable workspace artifacts:

| Artifact Description | Workspace Relative Path | File Size | SHA-256 Checksum |
|---|---|---:|---|
| **Best Model Checkpoint** | `results/full_training_15ep/resunet_2d/best_model.pt` | 97,486,845 B | `81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d` |
| **Final Model Checkpoint** | `results/full_training_15ep/resunet_2d/final_model.pt` | 97,488,175 B | `3c6e265f9b6093963ce346468d5dc1d40e878cf335184a4d0a309f145331103f` |
| **Training History** | `results/full_training_15ep/resunet_2d/history.csv` | 4,218 B | Verified 15 epochs, peak Dice epoch 13 |
| **Training Summary** | `results/full_training_15ep/resunet_2d/summary.json` | 1,446 B | Total runtime 8.0851 hours |
| **Calibration Parameters** | `results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json` | 32,421 B | Optimal $T^* = 2.8804590702056885$ |
| **C2 Run Manifest** | `results/stage_c/phase2/c2_calibration_v1/c2_run_manifest.json` | 2,756 B | Argmax invariance 100% verified |
| **Evaluation Metrics** | `results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv` | 1,842 B | Val_eval ECE 3.41%, Volumetric Dice 0.8641 |
| **Entropy Summary** | `results/stage_c/phase2/c2_calibration_v1/entropy_summary.csv` | 1,120 B | 6,874 non-empty slices, mean entropy 0.2686 |
| **HD95 Evaluation Summary** | `results/hd95_evaluation/hd95_summary.json` | 3,892 B | WT HD95 15.29 mm, TC HD95 8.98 mm |
| **C3 Policy Comparison** | `results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv` | 1,648 B | Retained Mean Dice 0.7803 |
| **Architecture Definition** | `src/models/resunet.py` | 7,984 B | 8,114,604 trainable parameters |

---

## 11. FINAL RESEARCH SUMMARY

ResUNet-2D achieves strong baseline segmentation accuracy on the BraTS-GLI benchmark (**0.8641 Volumetric Mean Dice**, **0.9039 WT Dice**). Through post-hoc temperature scaling ($T^* = 2.8805$), the model's expected calibration error is reduced by 60.3% to **3.41%** without altering argmax predictions. Embedded as the Primary Champion in the Stage C3 reliability agent, ResUNet-2D's calibrated predictions enable selective autonomous deployment on 83.8% of tumor slices with **0.7803 Mean Dice**, while routing high-uncertainty and discordant slices to human specialists.
"""

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(md)
    print(f'[✓] Markdown dossier generated: {MD_PATH} ({len(md)} chars)')


# -----------------------------------------------------------------------------
# BUILD WORD DOCX DOSSIER
# -----------------------------------------------------------------------------
def build_docx_dossier():
    doc = docx.Document()

    # Configure Margins: 0.75 in (54 pt)
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.different_first_page_header_footer = True

        # Header / Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("MODEL DOSSIER 01: RESUNET-2D | RELIABILITY-AWARE AI AGENT")
        hrun.font.name = 'Segoe UI'
        hrun.font.size = Pt(7.5)
        hrun.font.color.rgb = SLATE_RGB

        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
        frun = fp.add_run("B.Tech Major Project — Research Dossier | Strict Confidential Scientific Audit")
        frun.font.name = 'Segoe UI'
        frun.font.size = Pt(7.5)
        frun.font.color.rgb = SLATE_RGB

    # -------------------------------------------------------------------------
    # COVER / TITLE BLOCK
    # -------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(2)
    r_kicker = p_title.add_run("RESEARCH AUDIT & MODEL PROFILE — DOSSIER 01\n")
    r_kicker.font.name = 'Segoe UI'
    r_kicker.font.size = Pt(10)
    r_kicker.font.bold = True
    r_kicker.font.color.rgb = TEAL_RGB

    r_main = p_title.add_run("ResUNet-2D: Primary Champion Model\n")
    r_main.font.name = 'Segoe UI'
    r_main.font.size = Pt(22)
    r_main.font.bold = True
    r_main.font.color.rgb = NAVY_RGB

    r_sub = p_title.add_run("Comprehensive Architecture, 15-Epoch Training, Performance, Calibration & Reliability Report")
    r_sub.font.name = 'Segoe UI'
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    r_sub.font.color.rgb = SLATE_RGB

    add_callout_box(
        doc,
        "PROJECT GOVERNANCE & PROVENANCE METADATA",
        "• Project: Reliability-Aware AI Agent for Brain Tumor Segmentation (B.Tech Major Project)\n"
        "• Model Role: Primary Champion Segmentation Model & Core Inference Engine for Stage C Agent\n"
        "• Checkpoint Profile: STAGE_C_15EP | Best Checkpoint: Epoch 13 (SHA-256: 81ef568df7dc47e3...)\n"
        "• Stage C2 Calibrated Temperature: T* = 2.8804590702056885 (L-BFGS on Val_cal 50 cases)\n"
        "• Authoritative Evaluation Cohort: Held-Out Val_eval (105 Cases, 16,275 Slices, 96 Subjects)\n"
        "• Document Status: Audited Scientific Technical Dossier v1.0 | Date: November 18, 2026",
        border_color_hex=COLOR_NAVY_HEX,
        bg_color_hex=COLOR_LIGHT_BG_HEX
    )

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE MODEL SNAPSHOT
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "1. Executive Model Snapshot", level=1)
    add_body_p(
        doc,
        "ResUNet-2D is the primary champion segmentation architecture in our five-model portfolio. "
        "Engineered as a 5-stage deep residual U-Net with Group Normalization (G=8) and pre-activation residual units, "
        "the model delivers superior multi-class segmentation fidelity across the 4 co-registered MRI modalities of the BraTS-GLI benchmark. "
        "The snapshot below summarizes verified headline performance across training, evaluation, calibration, and reliability agent deployment."
    )

    # Metric Cards Row 1
    add_metric_card_row(doc, [
        ("Trainable Parameters", "8,114,604", "100% Trainable (92.97 MB)"),
        ("Best Training Epoch", "Epoch 13", "Mean Dice: 0.8670"),
        ("Volumetric Mean Dice", "0.8641", "Val_eval (105 Cases)"),
        ("Optimal Temperature", "2.8805", "L-BFGS Fit (Val_cal)"),
    ])

    # Metric Cards Row 2
    add_metric_card_row(doc, [
        ("Uncalibrated ECE", "8.59%", "Val_eval Baseline"),
        ("Calibrated ECE", "3.41%", "−60.3% Rel. Reduction"),
        ("Mean FG Entropy", "0.2686", "Median: 0.2474 (IQR 0.119)"),
        ("Agent Triage Dice", "0.7803", "+0.0653 vs Unfiltered"),
    ])

    # -------------------------------------------------------------------------
    # SECTION 2: MODEL IDENTITY & CONFIGURATION
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "2. Model Identity & Configuration", level=1)
    add_body_p(
        doc,
        "The specifications below are extracted directly from the authoritative configuration artifact "
        "(results/full_training_15ep/resunet_2d/config.json) and source definition (src/models/resunet.py)."
    )

    id_table = doc.add_table(rows=11, cols=2)
    id_widths = [Inches(2.5), Inches(4.0)]
    id_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]

    id_data = [
        ("Official Model Identifier", "resunet_2d (Display: ResUNet-2D)"),
        ("Model Family / Topology", "2D Deep Residual U-Net (Encoder-Decoder with Long Skips)"),
        ("Research Portfolio Role", "Primary Champion Model & Core Engine for Stage C Reliability Agent"),
        ("Input Modalities (4 Channels)", "T1n (Ch 0), T1c / Contrast (Ch 1), T2w (Ch 2), T2-FLAIR (Ch 3) @ 240×240"),
        ("Output Target Classes (4)", "0: Background, 1: Necrotic Core (NCR), 2: Edema (ED), 3: Enhancing Tumor (ET)"),
        ("Total Trainable Parameters", "8,114,604 (100% active gradients; zero frozen layers)"),
        ("Optimization Pipeline", "AdamW (lr=0.001, weight_decay=0.01), 2-Epoch Linear Warmup, Cosine Annealing"),
        ("Loss Function", "Compound Loss: Soft Multi-Class Dice Loss + Categorical Cross-Entropy Loss"),
        ("Training Dataset Partition", "BraTS-GLI: 721 Training Cases (59,020 Slices), Batch Size B=2"),
        ("Hardware & Environment", "NVIDIA GeForce RTX 4070 Laptop GPU (8GB), PyTorch 2.11.0+cu128, Python 3.11.9"),
        ("Checkpoint Selection Strategy", "Highest Validation Macro Mean Dice: Epoch 13 (0.867045 vs Epoch 15: 0.863087)"),
    ]
    for row_idx, (k, v) in enumerate(id_data):
        row = id_table.rows[row_idx]
        row.cells[0].paragraphs[0].add_run(k)
        row.cells[1].paragraphs[0].add_run(v)
    format_table(id_table, id_widths, id_align)

    # -------------------------------------------------------------------------
    # SECTION 3: ARCHITECTURE & DATA FLOW
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "3. Architecture & Data Flow", level=1)
    add_body_p(
        doc,
        "ResUNet-2D integrates pre-activation residual blocks into a 5-stage U-Net topology. "
        "The architecture diagram below displays the exact feedforward pathway, channel progressions, and long skip concatenations."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'architecture_diagram.png'),
        "Figure 1: ResUNet-2D 5-Stage Architecture with Pre-Activation Residual Units and Group Normalization (G=8)."
    )

    add_body_p(
        doc,
        "Detailed parameter breakdown across all encoder, decoder, and bottleneck stages:",
        bold_prefix="Layer Parameter Breakdown: "
    )

    param_table = doc.add_table(rows=12, cols=6)
    p_widths = [Inches(1.5), Inches(0.8), Inches(0.8), Inches(1.0), Inches(1.2), Inches(1.2)]
    p_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.CENTER,
               WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT]

    p_headers = ["Stage Module", "In Ch", "Out Ch", "Spatial Dim", "Parameters", "% Total"]
    for j, h in enumerate(p_headers):
        param_table.rows[0].cells[j].paragraphs[0].add_run(h)

    p_rows = [
        ("Encoder Level 1", "4", "32", "240×240", "10,632", "0.13%"),
        ("Encoder Level 2", "32", "64", "120×120", "57,664", "0.71%"),
        ("Encoder Level 3", "64", "128", "60×60", "230,016", "2.83%"),
        ("Encoder Level 4", "128", "256", "30×30", "918,784", "11.32%"),
        ("Bottleneck (L5)", "256", "512", "15×15", "3,672,576", "45.26%"),
        ("Decoder Level 4", "512+256", "256", "30×30", "2,427,136", "29.91%"),
        ("Decoder Level 3", "256+128", "128", "60×60", "607,360", "7.49%"),
        ("Decoder Level 2", "128+64", "64", "120×120", "152,128", "1.87%"),
        ("Decoder Level 1", "64+32", "32", "240×240", "38,176", "0.47%"),
        ("Classifier Head", "32", "4", "240×240", "132", "0.00%"),
        ("Total Network", "4", "4", "240×240", "8,114,604", "100.00%"),
    ]
    for row_idx, r_data in enumerate(p_rows, start=1):
        row = param_table.rows[row_idx]
        for col_idx, val in enumerate(r_data):
            row.cells[col_idx].paragraphs[0].add_run(val)
    format_table(param_table, p_widths, p_align)

    # -------------------------------------------------------------------------
    # SECTION 4: 15-EPOCH TRAINING PERFORMANCE ANALYSIS
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "4. 15-Epoch Training Performance Analysis", level=1)
    add_body_p(
        doc,
        "ResUNet-2D completed 15 full training epochs on 59,020 axial slices from 721 training cases. "
        "The model spent 8.0851 hours in training, averaging 32.3 minutes per epoch. "
        "The charts below illustrate the training vs validation loss trajectory and multi-class Dice progression."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'training_curves.png'),
        "Figure 2: Loss trajectories (left) and validation mean Dice progression (right) across 15 epochs, annotating Epoch 13."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'class_dice_curves.png'),
        "Figure 3: Multi-class validation Dice progression across 15 epochs for Background, NCR, ED, and ET sub-regions."
    )

    add_body_p(
        doc,
        "Authoritative 15-epoch numerical history extracted from results/full_training_15ep/resunet_2d/history.csv:",
        bold_prefix="Complete Epoch Log: "
    )

    hist_table = doc.add_table(rows=16, cols=8)
    h_widths = [Inches(0.6), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85)]
    h_align = [WD_ALIGN_PARAGRAPH.CENTER] + [WD_ALIGN_PARAGRAPH.RIGHT] * 7

    h_headers = ["Epoch", "Train Loss", "Val Loss", "Mean Dice", "BG Dice", "NCR Dice", "ED Dice", "ET Dice"]
    for j, h in enumerate(h_headers):
        hist_table.rows[0].cells[j].paragraphs[0].add_run(h)

    history_csv = 'results/full_training_15ep/resunet_2d/history.csv'
    df_h = pd.read_csv(history_csv)
    for idx, r in df_h.iterrows():
        row = hist_table.rows[idx + 1]
        row.cells[0].paragraphs[0].add_run(str(int(r['epoch'])))
        row.cells[1].paragraphs[0].add_run(f"{r['train_loss']:.4f}")
        row.cells[2].paragraphs[0].add_run(f"{r['val_loss']:.4f}")
        row.cells[3].paragraphs[0].add_run(f"{r['mean_dice']:.4f}")
        row.cells[4].paragraphs[0].add_run(f"{r['dice_class_0']:.4f}")
        row.cells[5].paragraphs[0].add_run(f"{r['dice_class_1']:.4f}")
        row.cells[6].paragraphs[0].add_run(f"{r['dice_class_2']:.4f}")
        row.cells[7].paragraphs[0].add_run(f"{r['dice_class_3']:.4f}")
    format_table(hist_table, h_widths, h_align)

    add_body_p(
        doc,
        "Scientific Convergence Assessment: Training loss declined monotonically across epochs 11–15 (0.1895 to 0.1766), "
        "while validation loss stabilized between 0.2063 (Epoch 14) and 0.2088 (Epoch 15). "
        "Validation Mean Dice reached its peak of 0.8670 at Epoch 13 before leveling off at 0.8631 at Epoch 15. "
        "This indicates that the model reached optimal validation performance at Epoch 13 with diminishing returns from additional epochs."
    )

    # -------------------------------------------------------------------------
    # SECTION 5: SEGMENTATION PERFORMANCE
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "5. Segmentation Performance", level=1)
    add_body_p(
        doc,
        "Evaluation results on the held-out Val_eval cohort (105 cases, 16,275 slices, 96 subjects). "
        "Metrics reflect 3D volumetric evaluation aggregated across all axial slices per subject."
    )

    seg_table = doc.add_table(rows=5, cols=6)
    s_widths = [Inches(1.8), Inches(0.9), Inches(1.1), Inches(1.0), Inches(1.1), Inches(0.6)]
    s_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER,
               WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.CENTER, WD_ALIGN_PARAGRAPH.RIGHT]

    s_headers = ["Sub-Region", "Mean Dice", "Dice 95% CI", "Mean HD95", "HD95 95% CI", "Median"]
    for j, h in enumerate(s_headers):
        seg_table.rows[0].cells[j].paragraphs[0].add_run(h)

    s_rows = [
        ("Whole Tumor (WT)", "0.9039", "[0.8765, 0.9241]", "15.29 mm", "[11.38, 20.05] mm", "4.12 mm"),
        ("Tumor Core (TC)", "0.8597", "[0.8076, 0.8986]", "8.98 mm", "[6.08, 12.74] mm", "3.16 mm"),
        ("Enhancing Tumor (ET)", "0.8286", "[0.7794, 0.8684]", "9.92 mm", "[4.71, 18.66] mm", "2.00 mm"),
        ("Macro Mean", "0.8641", "[0.8265, 0.8930]", "11.40 mm", "[7.39, 17.15] mm", "3.09 mm"),
    ]
    for row_idx, r_data in enumerate(s_rows, start=1):
        row = seg_table.rows[row_idx]
        for col_idx, val in enumerate(r_data):
            row.cells[col_idx].paragraphs[0].add_run(val)
    format_table(seg_table, s_widths, s_align)

    add_body_p(
        doc,
        "Distribution extremes across the 105 evaluated cases: The top performing case BraTS-GLI-01335-000 achieved WT Dice 0.9831, "
        "TC Dice 0.9534, and ET Dice 0.9391. Conversely, infiltrative and small lesions posed greater difficulty: "
        "BraTS-GLI-00753-001 (diffuse edema) achieved WT Dice 0.4146, while BraTS-GLI-00331-000 (<50 voxel necrotic core) achieved TC Dice 0.0536."
    )

    # -------------------------------------------------------------------------
    # SECTION 6: PROBABILITY CALIBRATION ANALYSIS (STAGE C2)
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "6. Probability Calibration Analysis (Stage C2)", level=1)
    add_body_p(
        doc,
        "In Stage C2, ResUNet-2D underwent scalar Temperature Scaling to calibrate predicted probabilities. "
        "The optimal temperature T* = 2.880459 was fitted using L-BFGS optimization on Val_cal (50 cases, 5,020,730 sampled voxels)."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'calibration_comparison.png'),
        "Figure 4: Reliability diagrams for uncalibrated vs calibrated probabilities and ECE reduction summary across cohorts."
    )

    add_body_p(
        doc,
        "Authoritative Stage C2 calibration metrics comparing pre- and post-calibration across both validation partitions:",
        bold_prefix="Calibration Verification Metrics: "
    )

    cal_table = doc.add_table(rows=5, cols=7)
    c_widths = [Inches(1.5), Inches(0.8), Inches(0.8), Inches(0.9), Inches(0.8), Inches(0.8), Inches(0.9)]
    c_align = [WD_ALIGN_PARAGRAPH.LEFT] + [WD_ALIGN_PARAGRAPH.RIGHT] * 6

    c_headers = ["Metric", "Val_cal Pre", "Val_cal Post", "Val_cal Red.", "Val_eval Pre", "Val_eval Post", "Val_eval Red."]
    for j, h in enumerate(c_headers):
        cal_table.rows[0].cells[j].paragraphs[0].add_run(h)

    c_rows = [
        ("NLL", "0.4154", "0.2173", "−47.69%", "1.0277", "0.4415", "−57.04%"),
        ("ECE", "5.02%", "0.84%", "−83.25%", "8.59%", "3.41%", "−60.32%"),
        ("MCE", "28.11%", "7.26%", "−74.17%", "30.73%", "26.42%", "−14.03%"),
        ("Brier Score", "0.1122", "0.0994", "−11.37%", "0.1851", "0.1660", "−10.32%"),
    ]
    for row_idx, r_data in enumerate(c_rows, start=1):
        row = cal_table.rows[row_idx]
        for col_idx, val in enumerate(r_data):
            row.cells[col_idx].paragraphs[0].add_run(val)
    format_table(cal_table, c_widths, c_align)

    add_body_p(
        doc,
        "Argmax Invariance: Because scalar temperature scaling is a monotonic transformation, argmax class assignments are strictly preserved. "
        "Stage C2 verified exactly 0 mismatches between pre- and post-calibration segmentation masks across all 16,275 slices."
    )

    # -------------------------------------------------------------------------
    # SECTION 7: PREDICTIVE UNCERTAINTY ANALYSIS
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "7. Predictive Uncertainty Analysis", level=1)
    add_body_p(
        doc,
        "Normalized foreground Shannon predictive entropy serves as the primary uncertainty quantification metric. "
        "The distribution across all 6,874 non-empty foreground slices on Val_eval is plotted below."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'entropy_distribution.png'),
        "Figure 5: Histogram and boxplot of foreground predictive entropy on Val_eval, with policy threshold τ = 0.35 annotated."
    )

    ent_table = doc.add_table(rows=7, cols=3)
    e_widths = [Inches(2.5), Inches(1.5), Inches(2.5)]
    e_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.LEFT]

    e_rows = [
        ("Total Slices Evaluated", "16,275", "105 cases × 155 slices per volume"),
        ("Empty Prediction Slices", "9,401 (57.76%)", "No tumor voxels detected; entropy undefined"),
        ("Non-Empty Foreground Slices", "6,874 (42.24%)", "Evaluated for foreground entropy"),
        ("Mean Foreground Entropy (± Std)", "0.2686 ± 0.0903", "Overall confidence across tumor slices"),
        ("Median [IQR] Entropy", "0.2474 [0.1995, 0.3185]", "IQR: 0.1190, Min: 0.0000, Max: 0.6146"),
        ("Non-Empty Slices with H < 0.35", "5,619 (81.74%)", "Accepted by agent (low uncertainty)"),
        ("Non-Empty Slices with H ≥ 0.35", "1,255 (18.26%)", "Flagged by agent (high uncertainty)"),
    ]
    for row_idx, (k, v, desc) in enumerate(e_rows):
        row = ent_table.rows[row_idx]
        row.cells[0].paragraphs[0].add_run(k)
        row.cells[1].paragraphs[0].add_run(v)
        row.cells[2].paragraphs[0].add_run(desc)
    format_table(ent_table, e_widths, e_align)

    # -------------------------------------------------------------------------
    # SECTION 8: ROLE IN THE STAGE C RELIABILITY AGENT
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "8. Role in the Stage C Reliability Agent", level=1)
    add_body_p(
        doc,
        "ResUNet-2D serves as the Primary Champion in the Stage C3 v1.1.3 reliability-aware agent. "
        "The secondary fallback model is UNetResNet34 (T* = 2.6261). "
        "The agent implements a 3-action policy (ACCEPT, FALLBACK_SWITCH, FLAG_UNRELIABLE) using an entropy threshold of τ = 0.35 "
        "and 2.5D axial context validation (radius k = ±2)."
    )

    add_figure_with_caption(
        doc,
        os.path.join(ASSETS_DIR, 'agent_flow_diagram.png'),
        "Figure 6: Stage C3 v1.1.3 Decision Protocol showing triage branches across entropy, axial context, and fallback models."
    )

    add_body_p(
        doc,
        "System-level reliability metrics comparing the unfiltered ResUNet-2D baseline against the Stage C3 agent policy (on 6,549 tumor slices):",
        bold_prefix="Agent Policy Impact on Tumor Slices: "
    )

    pol_table = doc.add_table(rows=5, cols=6)
    pol_widths = [Inches(1.8), Inches(0.9), Inches(1.0), Inches(1.0), Inches(0.9), Inches(0.9)]
    pol_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT,
                 WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT, WD_ALIGN_PARAGRAPH.RIGHT]

    pol_headers = ["Policy Configuration", "Coverage", "Retained Dice", "Referred Dice", "Severe Fails", "Fail Rate"]
    for j, h in enumerate(pol_headers):
        pol_table.rows[0].cells[j].paragraphs[0].add_run(h)

    pol_rows = [
        ("Unfiltered Primary", "100.00%", "0.7150", "—", "1,343", "20.51%"),
        ("Entropy-Only (τ=0.35)", "83.02%", "0.7855", "0.3702", "699", "12.86%"),
        ("Fallback-Only (No τ)", "95.63%", "0.7499", "0.0000", "1,046", "16.70%"),
        ("Full Stage C3 Policy", "83.84%", "0.7803", "0.3891", "742", "13.51%"),
    ]
    for row_idx, r_data in enumerate(pol_rows, start=1):
        row = pol_table.rows[row_idx]
        for col_idx, val in enumerate(r_data):
            row.cells[col_idx].paragraphs[0].add_run(val)
    format_table(pol_table, pol_widths, pol_align)

    add_body_p(
        doc,
        "Key Agent Finding: The Stage C3 agent boosts mean retained segmentation quality from 0.7150 to 0.7803 (+0.0653 Dice) "
        "and reduces severe segmentation failures (Dice < 0.5) from 20.51% to 13.51% (a 34.1% relative reduction). "
        "The agent selectively refers lower-confidence slices (mean referred Dice = 0.3891) to human experts."
    )

    # -------------------------------------------------------------------------
    # SECTION 9: SCIENTIFIC LIMITATIONS & DISCLOSURES
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "9. Scientific Limitations & Disclosures", level=1)
    add_body_p(
        doc,
        "In accordance with sound research methodology, ten critical scientific disclosures are documented below:"
    )

    disclosures = [
        ("1. Single-Seed Training: ", "The model was trained exclusively on seed 42. Multi-seed variance has not been evaluated."),
        ("2. Internal Validation Only: ", "All results reflect internal validation on BraTS-GLI; the model has not been validated on external hospital cohorts."),
        ("3. Checkpoint Selection Exposure: ", "Epoch 13 was selected using validation Dice; slight selection exposure is inherent to internal evaluation splits."),
        ("4. Calibration Separation Scope: ", "While Val_cal (50 cases) and Val_eval (105 cases) are strictly subject-isolated, both originate from the same broader BraTS-GLI distribution."),
        ("5. Predictive Entropy Limitations: ", "Normalized entropy is an empirical uncertainty heuristic; it does not guarantee detection of confident false negatives."),
        ("6. Empty-Prediction Prevalence: ", "57.76% of all slices are empty. 2.5D axial context validation is required to avoid misclassifying true-negative slices."),
        ("7. MCE Sample Sensitivity: ", "Maximum Calibration Error (26.42% on Val_eval) is driven by sparse confidence bins in intermediate probability ranges."),
        ("8. Missing-Modality Vulnerability: ", "Withholding T1c collapses Tumor Core Dice to 0.3132 and Enhancing Tumor Dice to 0.1684 (overall Mean Dice: 0.4449). Withholding T2w collapses Whole Tumor Dice to 0.4388 (overall Mean Dice: 0.6058). Full 4-channel input is required."),
        ("9. Research Prototype Classification: ", "ResUNet-2D and the Stage C agent are academic research prototypes, not approved medical devices (no FDA or CDSCO approval)."),
        ("10. Sealed Test Cohort Preservation: ", "The quarantine test partition (data/quarantine_test/) remained strictly sealed, untouched, and unaccessed."),
    ]
    for pfx, txt in disclosures:
        add_body_p(doc, txt, bold_prefix=pfx, space_after=3)

    # -------------------------------------------------------------------------
    # SECTION 10: ARTIFACT PROVENANCE REGISTER
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "10. Artifact Provenance Register", level=1)
    add_body_p(
        doc,
        "All reported metrics are verifiable against immutable workspace artifacts:"
    )

    prov_table = doc.add_table(rows=8, cols=3)
    pr_widths = [Inches(1.8), Inches(3.2), Inches(1.5)]
    pr_align = [WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.LEFT]

    pr_rows = [
        ("Best Model Checkpoint", "results/full_training_15ep/resunet_2d/best_model.pt", "SHA-256: 81ef568df7dc47e3... (97.49 MB)"),
        ("Final Model Checkpoint", "results/full_training_15ep/resunet_2d/final_model.pt", "SHA-256: 3c6e265f9b609396... (97.49 MB)"),
        ("Training History Log", "results/full_training_15ep/resunet_2d/history.csv", "15 Epochs, Peak Dice Epoch 13"),
        ("Calibration Artifact", "results/stage_c/phase2/c2_calibration_v1/calibration_parameters.json", "Optimal T* = 2.880459"),
        ("C2 Evaluation CSV", "results/stage_c/phase2/c2_calibration_v1/evaluation_metrics.csv", "Val_eval ECE: 3.41%, Dice: 0.8641"),
        ("HD95 Summary JSON", "results/hd95_evaluation/hd95_summary.json", "WT: 15.29 mm, TC: 8.98 mm, ET: 9.92 mm"),
        ("C3 Policy Summary", "results/stage_c/phase3/c3_selective_prediction_v1_1_3/policy_comparison_summary.csv", "Retained Mean Dice: 0.7803"),
        ("Source Architecture", "src/models/resunet.py", "8,114,604 Trainable Parameters"),
    ]
    for row_idx, (k, path, hash_desc) in enumerate(pr_rows):
        row = prov_table.rows[row_idx]
        row.cells[0].paragraphs[0].add_run(k)
        row.cells[1].paragraphs[0].add_run(path)
        row.cells[2].paragraphs[0].add_run(hash_desc)
    format_table(prov_table, pr_widths, pr_align)

    # -------------------------------------------------------------------------
    # SECTION 11: FINAL RESEARCH SUMMARY
    # -------------------------------------------------------------------------
    add_heading_styled(doc, "11. Final Research Summary", level=1)
    add_body_p(
        doc,
        "ResUNet-2D demonstrates strong segmentation fidelity on the BraTS-GLI benchmark, achieving 0.8641 Volumetric Mean Dice "
        "and 0.9039 Whole Tumor Dice. Post-hoc probability calibration via scalar Temperature Scaling (T* = 2.8805) reduces Expected Calibration Error "
        "by 60.3% to 3.41% without modifying argmax predictions. Deployed as the Primary Champion in the Stage C3 reliability agent, "
        "calibrated uncertainty enables selective autonomous deployment on 83.8% of tumor slices with 0.7803 Mean Dice, "
        "while systematically routing high-uncertainty and discordant predictions to human specialists."
    )

    doc.save(DOCX_PATH)
    print(f'[✓] Word DOCX dossier generated: {DOCX_PATH}')


# -----------------------------------------------------------------------------
# CONVERT DOCX TO PDF VIA WORD COM
# -----------------------------------------------------------------------------
def convert_docx_to_pdf():
    ps_script = f"""
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {{
    $doc = $word.Documents.Open('{DOCX_PATH}')
    $doc.SaveAs([ref]'{PDF_PATH}', [ref]17)
    $doc.Close()
    Write-Host "Successfully converted DOCX to PDF at {PDF_PATH}"
}} catch {{
    Write-Error $_
}} finally {{
    $word.Quit()
    [System.Runtime.Interopservices.Marshal]::ReleaseComObject($word) | Out-Null
}}
"""
    ps_file = os.path.join(OUTPUT_DIR, 'convert_to_pdf.ps1')
    with open(ps_file, 'w', encoding='utf-8') as f:
        f.write(ps_script)

    ret = os.system(f'powershell -ExecutionPolicy Bypass -File "{ps_file}"')
    if os.path.exists(PDF_PATH):
        size = os.path.getsize(PDF_PATH)
        print(f'[✓] PDF generated successfully: {PDF_PATH} ({size:,} bytes)')
    else:
        print(f'[!] PDF conversion failed with code {ret}')


if __name__ == '__main__':
    print('Starting ResUNet-2D Dossier generation...')
    build_markdown_dossier()
    build_docx_dossier()
    convert_docx_to_pdf()
    print('ResUNet-2D Dossier generation complete.')
