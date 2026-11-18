"""
tools/build_unet_dossier_docx_pdf.py
Compiles the comprehensive, publication-grade U-Net 2D Model Master Dossier into:
1. Microsoft Word Master (.docx): D:\\MajorProject\\documents\\UNet_2D_Model_Master_Dossier.docx
2. Printout-Ready PDF (.pdf): D:\\MajorProject\\documents\\UNet_2D_Model_Master_Dossier.pdf (via Word COM)
Also mirrors files to reports/model_profiles/unet_2d/dossier/
"""

import os
import sys
import subprocess
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

PROJECT_ROOT = Path(r"D:\MajorProject")
REPORTS_DOSSIER_DIR = PROJECT_ROOT / "reports" / "model_profiles" / "unet_2d" / "dossier"
DOCS_DIR = REPORTS_DOSSIER_DIR  # Decoupled
DOCS_FIG_DIR = PROJECT_ROOT / "reports" / "model_profiles" / "unet_2d" / "assets"

# DOCS_DIR mkdir removed
REPORTS_DOSSIER_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# DOCX Color Palette & XML Helpers
# -----------------------------------------------------------------------------
COLOR_NAVY = RGBColor(15, 23, 42)       # Slate 900
COLOR_BLUE = RGBColor(29, 78, 216)      # Blue 700
COLOR_MUTED = RGBColor(100, 116, 139)   # Slate 500
COLOR_DARK = RGBColor(30, 41, 59)       # Slate 800

HEX_HEADER_BG = "0F172A"
HEX_ZEBRA_BG = "F8FAFC"
HEX_CARD_BORDER = "CBD5E1"
HEX_ACCENT_BG = "EFF6FF"
HEX_CALLOUT_BORDER = "3B82F6"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def add_callout_box(doc, title, text, border_hex="2563EB", bg_hex="EFF6FF"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"★ {title}\n")
    run_t.font.name = "Segoe UI"
    run_t.font.size = Pt(11)
    run_t.font.bold = True
    run_t.font.color.rgb = COLOR_NAVY

    run_b = p.add_run(text)
    run_b.font.name = "Segoe UI"
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = COLOR_DARK
    doc.add_paragraph().paragraph_format.space_after = Pt(6)

def style_heading_1(doc, text):
    h = doc.add_heading(text, level=1)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = "Segoe UI"
    run.font.size = Pt(18)
    run.font.bold = True
    run.font.color.rgb = COLOR_NAVY
    return h

def style_heading_2(doc, text):
    h = doc.add_heading(text, level=2)
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.runs[0]
    run.font.name = "Segoe UI"
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = COLOR_BLUE
    return h

def add_body_p(doc, text, bold_prefix=None, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Segoe UI"
        r_pre.font.size = Pt(10.5)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_NAVY
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(10.5)
    r.font.color.rgb = COLOR_DARK
    return p

def add_figure_with_caption(doc, img_path, caption_num, caption_title, description):
    if not img_path.exists():
        print(f"Warning: image {img_path} not found!")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    doc.add_picture(str(img_path), width=Inches(6.5))

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(12)
    p_cap.paragraph_format.keep_with_next = True

    r_num = p_cap.add_run(f"Figure {caption_num}: ")
    r_num.font.name = "Segoe UI"
    r_num.font.size = Pt(9.5)
    r_num.font.bold = True
    r_num.font.color.rgb = COLOR_NAVY

    r_tit = p_cap.add_run(f"{caption_title} — ")
    r_tit.font.name = "Segoe UI"
    r_tit.font.size = Pt(9.5)
    r_tit.font.bold = True
    r_tit.font.color.rgb = COLOR_BLUE

    r_desc = p_cap.add_run(description)
    r_desc.font.name = "Segoe UI"
    r_desc.font.size = Pt(9.5)
    r_desc.font.color.rgb = COLOR_MUTED


def build_docx():
    print("Building U-Net 2D Master Dossier (.docx)...")
    doc = Document()

    # Set page margins to standard 0.75 in
    for section in doc.sections:
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)

    # -------------------------------------------------------------------------
    # COVER / HEADER BANNER
    # -------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(2)
    r_main = p_title.add_run("MODEL DOSSIER 01: U-NET 2D")
    r_main.font.name = "Segoe UI"
    r_main.font.size = Pt(24)
    r_main.font.bold = True
    r_main.font.color.rgb = COLOR_NAVY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_after = Pt(12)
    r_sub = p_sub.add_run("Foundational Semantic Segmentation Architecture, Full 15-Epoch Training Dynamics, Post-Hoc Probability Calibration, Predictive Uncertainty & Stage C Multi-Model Benchmark")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = COLOR_MUTED

    # Meta Table
    tbl_meta = doc.add_table(rows=4, cols=2)
    tbl_meta.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_info = [
        ("Project & Governing Authority", "Reliability-Aware AI Agent for Brain Tumor Segmentation (MASTER_PROJECT_BLUEPRINT_v2.0.md)"),
        ("Model Identifier & Architecture", "U-Net 2D (Standard 4-Level Plain U-Net, UNet2D) | 7,763,428 Trainable Parameters"),
        ("Operational Status & Best Epoch", "Fully Completed 15 Epochs | Best Checkpoint: Epoch 15 (Validation Mean Dice: 0.8594)"),
        ("Cryptographic Checkpoint SHA-256", "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef (93.3 MB)")
    ]
    for row_idx, (k, v) in enumerate(meta_info):
        c0 = tbl_meta.cell(row_idx, 0)
        c1 = tbl_meta.cell(row_idx, 1)
        c0.width = Inches(2.2)
        c1.width = Inches(4.5)
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, top=60, bottom=60, left=100, right=100)
        set_cell_margins(c1, top=60, bottom=60, left=100, right=100)
        p0 = c0.paragraphs[0]
        p0.add_run(k).font.bold = True
        p0.runs[0].font.name = "Segoe UI"
        p0.runs[0].font.size = Pt(9.5)
        p0.runs[0].font.color.rgb = COLOR_NAVY
        p1 = c1.paragraphs[0]
        p1.add_run(v).font.name = "Segoe UI"
        p1.runs[0].font.size = Pt(9.5)
        p1.runs[0].font.color.rgb = COLOR_DARK

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE PERFORMANCE SUMMARY (AT A GLANCE)
    # -------------------------------------------------------------------------
    style_heading_1(doc, "1. Executive Performance Summary (At A Glance)")
    add_body_p(doc, "The foundational U-Net 2D model represents the architectural baseline of our five-model deep learning portfolio for multi-parametric MRI (mpMRI) brain tumor segmentation. Evaluated on the authorized 721-case development cohort and held-out 155-case validation pool under strict test quarantine protocol, U-Net 2D completed all 15 planned epochs, achieving optimal convergence at Epoch 15 with maximum validation mean Dice of 0.8594 and outstanding post-hoc probability calibration of 4.83% ECE.")

    add_callout_box(doc, "Key Architectural Finding",
                    "U-Net 2D achieved the highest parameter efficiency (7.76M parameters vs 24.33M for ResNet34 and 9.16M for U-Net++) and fastest inference latency (9.60 ms per slice). In our multi-model reliability agent evaluation, U-Net served as the primary fallback engine, rescuing 65.9% of complete tumor misses produced by deeper nested networks.")

    # Executive KPI Table
    exec_table = doc.add_table(rows=14, cols=4)
    exec_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Metric Dimension", "Protocol / Dataset Census", "Empirical Value / Performance", "Significance & Portfolio Context"]
    for i, h in enumerate(headers):
        cell = exec_table.cell(0, i)
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=100, bottom=100, left=120, right=120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    rows_data = [
        ("Trainable Parameters", "Standard 4-level U-Net (FP32)", "7,763,428 (7.76 M)", "Most lightweight & parameter-efficient in portfolio"),
        ("Validation Mean Dice", "Epoch 15 Checkpoint (155 Val Cases)", "0.8594 (85.94%)", "Optimal checkpoint across 15-epoch training budget"),
        ("Enhancing Tumor (ET) Dice", "Epoch 15 Checkpoint (Class 3)", "0.9026 (90.26%)", "Sub-compartment peak; highly sensitive to active core"),
        ("Peritumoral Edema (ED) Dice", "Epoch 15 Checkpoint (Class 2)", "0.8662 (86.62%)", "Consistent infiltrative margin segmentation"),
        ("Necrotic Core (NCR) Dice", "Epoch 15 Checkpoint (Class 1)", "0.8094 (80.94%)", "Strong delineation of hypointense dead core"),
        ("Validation Mean IoU", "Epoch 15 Checkpoint (155 Val Cases)", "0.7554 (75.54%)", "Jaccard intersection over union index"),
        ("Volumetric Macro Dice", "Val_eval Cohort (105 Held-out Cases)", "0.8574 (85.74%)", "WT: 89.00%, TC: 85.27%, ET: 82.94%"),
        ("Optimal Temperature (T*)", "Val_cal Cohort (50 Cases, 5.02M voxels)", "T* = 2.6268 (L-BFGS, 43 iter)", "Learned positive scalar logit temperature"),
        ("Expected Calibration Error", "Val_eval Cohort (105 Cases, 311K voxels)", "10.85% → 4.83% (-55.5%)", "Second lowest calibrated ECE; beats ResNet34 & UNet++"),
        ("Negative Log-Likelihood", "Val_eval Cohort (105 Cases, 311K voxels)", "0.8273 → 0.3939 (-52.4%)", "Dramatic reduction in probabilistic prediction penalty"),
        ("Predictive Entropy Mean", "16,275 Slices (6,177 Non-Empty)", "0.2923 ± 0.0896 (Median 0.2701)", "Clinical referral threshold τ = 0.35 gates autonomous vs MD"),
        ("Inference Latency", "NVIDIA RTX 4070 Laptop GPU", "9.60 ms per slice", "Fastest model in portfolio (1.5× faster than UNet++)"),
        ("Confirmed Miss Rate", "3,223 Slices Evaluated", "4.58% (118 complete misses)", "Far superior to UNet++ (12.53% miss rate, 323 misses)")
    ]

    for row_idx, r_data in enumerate(rows_data, start=1):
        for col_idx, text in enumerate(r_data):
            cell = exec_table.cell(row_idx, col_idx)
            bg = HEX_ZEBRA_BG if row_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=70, bottom=70, left=100, right=100)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Segoe UI"
            r.font.size = Pt(9.0)
            r.font.color.rgb = COLOR_DARK
            if col_idx == 2:
                r.font.bold = True
                r.font.color.rgb = COLOR_BLUE

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Figure 7: Infographic Poster
    add_figure_with_caption(doc, DOCS_FIG_DIR / "07_unet_executive_summary_infographic.png",
                           "1.1", "U-Net 2D Executive Summary Dashboard & Performance Scorecard",
                           "Six major KPI metric cards, parameter footprint, calibration reduction, and clinical agent triage distribution.")

    # -------------------------------------------------------------------------
    # SECTION 2: NETWORK ARCHITECTURE & MATHEMATICAL FORMULATION
    # -------------------------------------------------------------------------
    style_heading_1(doc, "2. Network Architecture & Mathematical Formulation")
    add_body_p(doc, "The U-Net 2D architecture implemented in `src/models/unet.py` follows the canonical encoder-decoder topology introduced by Ronneberger et al. (2015), customized specifically for 4-channel multi-parametric brain MRI inputs and 4 mutually exclusive voxel classification categories.")

    add_body_p(doc, "The network consists of a contractive encoder path (extracting contextual representation across 4 resolution scales), a central bottleneck latent stage, an expansive decoder path (restoring full spatial detail), and long skip connections that directly concatenate high-resolution feature maps from the encoder to the corresponding decoder stage.")

    # Layer Breakdown Table
    layer_tbl = doc.add_table(rows=11, cols=5)
    layer_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    layer_headers = ["Stage / Block", "Sub-Modules & Operator Flow", "Feature Channels & Spatial Grid", "Trainable Parameters", "Param %"]
    for i, h in enumerate(layer_headers):
        cell = layer_tbl.cell(0, i)
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=80, bottom=80, left=100, right=100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    layer_data = [
        ("Encoder 1", "2× [Conv3x3 (4→32), BN, ReLU]", "32 ch @ 240 × 240", "10,496", "0.14%"),
        ("Encoder 2", "MaxPool2d(2×2) + 2× [Conv3x3 (32→64), BN, ReLU]", "64 ch @ 120 × 120", "55,552", "0.72%"),
        ("Encoder 3", "MaxPool2d(2×2) + 2× [Conv3x3 (64→128), BN, ReLU]", "128 ch @ 60 × 60", "221,696", "2.86%"),
        ("Encoder 4", "MaxPool2d(2×2) + 2× [Conv3x3 (128→256), BN, ReLU]", "256 ch @ 30 × 30", "885,760", "11.41%"),
        ("Bottleneck", "MaxPool2d(2×2) + 2× [Conv3x3 (256→512), BN, ReLU]", "512 ch @ 15 × 15", "3,540,992", "45.61%"),
        ("Decoder 4", "ConvTranspose2d (512→256) + Concat(256) + 2× Conv3x3", "256 ch @ 30 × 30", "2,295,040", "29.56%"),
        ("Decoder 3", "ConvTranspose2d (256→128) + Concat(128) + 2× Conv3x3", "128 ch @ 60 × 60", "574,080", "7.39%"),
        ("Decoder 2", "ConvTranspose2d (128→64) + Concat(64) + 2× Conv3x3", "64 ch @ 120 × 120", "143,680", "1.85%"),
        ("Decoder 1", "ConvTranspose2d (64→32) + Concat(32) + 2× Conv3x3", "32 ch @ 240 × 240", "36,000", "0.46%"),
        ("Classifier Head", "1×1 Conv2d (32 → 4)", "4 Class Logits @ 240 × 240", "132", "<0.01%")
    ]
    for row_idx, r_data in enumerate(layer_data, start=1):
        for col_idx, text in enumerate(r_data):
            cell = layer_tbl.cell(row_idx, col_idx)
            bg = HEX_ZEBRA_BG if row_idx % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK
            if col_idx in [3, 4]:
                r.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Figure 1: Architecture
    add_figure_with_caption(doc, DOCS_FIG_DIR / "01_unet_complete_architecture_diagram.png",
                           "2.1", "U-Net 2D Complete 4-Level Encoder-Decoder Architecture",
                           "Detailed dataflow, channel widths, downsampling/upsampling operators, skip connections, and exact layer parameter counts.")

    # -------------------------------------------------------------------------
    # SECTION 3: 15-EPOCH TRAINING DYNAMICS & CONVERGENCE PROFILE
    # -------------------------------------------------------------------------
    style_heading_1(doc, "3. 15-Epoch Training Dynamics & Convergence Profile")
    add_body_p(doc, "U-Net 2D was trained on the Acer Predator Helios 16 workstation utilizing the NVIDIA GeForce RTX 4070 Laptop GPU (8 GB GDDR6). The training cohort comprised 721 authorized cases (667 unique subjects, 59,020 slices per epoch). The validation pool comprised 155 cases (143 unique subjects, 12,364 slices). The run executed across 15 epochs under AdamW optimization with a 2-epoch linear warmup and 13-epoch cosine decay.")

    # Complete 15 Epoch History Table
    hist_table = doc.add_table(rows=16, cols=7)
    hist_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hist_headers = ["Epoch", "Train Loss", "Val Loss", "Val Mean Dice", "Val ET (C3)", "Val ED (C2)", "Val NCR (C1)"]
    for i, h in enumerate(hist_headers):
        cell = hist_table.cell(0, i)
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9.0)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    hist_rows = [
        ("1", "0.4527", "0.3549", "0.7977", "0.8602", "0.8061", "0.7268"),
        ("2", "0.2605", "0.4827", "0.6841", "0.6845", "0.7490", "0.6188"),
        ("3", "0.2438", "0.3815", "0.7932", "0.8420", "0.7893", "0.7483"),
        ("4", "0.2309", "0.3267", "0.7844", "0.8692", "0.8345", "0.6496"),
        ("5", "0.2195", "0.4320", "0.8425", "0.8858", "0.8566", "0.7851"),
        ("6", "0.2133", "0.4767", "0.8369", "0.8819", "0.8437", "0.7851"),
        ("7", "0.2074", "0.4408", "0.8481", "0.8966", "0.8363", "0.8114"),
        ("8", "0.2061", "0.5589", "0.8125", "0.8421", "0.8525", "0.7428"),
        ("9", "0.1991", "0.5135", "0.8409", "0.8850", "0.8317", "0.8059"),
        ("10", "0.1963", "0.5184", "0.8526", "0.8951", "0.8625", "0.8002"),
        ("11", "0.1914", "0.4911", "0.8506", "0.8972", "0.8659", "0.7885"),
        ("12", "0.1878", "0.5224", "0.8384", "0.8890", "0.8431", "0.7830"),
        ("13", "0.1852", "0.5381", "0.8399", "0.8993", "0.8583", "0.7622"),
        ("14", "0.1832", "0.4966", "0.8481", "0.9024", "0.8542", "0.7877"),
        ("15 (Best)", "0.1815", "0.4589", "0.8594", "0.9026", "0.8662", "0.8094")
    ]
    for row_idx, r_data in enumerate(hist_rows, start=1):
        for col_idx, text in enumerate(r_data):
            cell = hist_table.cell(row_idx, col_idx)
            is_best = (row_idx == 15)
            bg = "DCFCE7" if is_best else (HEX_ZEBRA_BG if row_idx % 2 == 1 else "FFFFFF")
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=80, right=80)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK
            if is_best or col_idx == 3:
                r.font.bold = True
                if is_best:
                    r.font.color.rgb = RGBColor(22, 101, 52)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Figure 2 & Figure 3
    add_figure_with_caption(doc, DOCS_FIG_DIR / "02_unet_15epoch_loss_convergence.png",
                           "3.1", "15-Epoch Loss Trajectory & Optimization Schedule",
                           "Train loss descent (0.4527 → 0.1815), validation loss, cosine learning rate decay, and gradient norm stability.")

    add_figure_with_caption(doc, DOCS_FIG_DIR / "03_unet_subcompartment_dice_iou_progression.png",
                           "3.2", "Sub-Compartment Dice & IoU Progression Across 15 Epochs",
                           "Enhancing Tumor (ET), Peritumoral Edema (ED), Necrotic Core (NCR), and Macro Mean scores peaking simultaneously at Epoch 15.")

    # -------------------------------------------------------------------------
    # SECTION 4: POST-HOC PROBABILITY CALIBRATION (TEMPERATURE SCALING)
    # -------------------------------------------------------------------------
    style_heading_1(doc, "4. Post-Hoc Probability Calibration & Temperature Scaling")
    add_body_p(doc, "Modern deep neural networks trained with cross-entropy and soft Dice loss are notoriously overconfident. Uncalibrated softmax outputs cannot be interpreted as true posterior probabilities. Under Work Package C2, post-hoc temperature scaling was fitted exclusively on Val_cal (50 cases, 47 unique subjects, 5,020,730 voxels) using L-BFGS optimization.")

    add_body_p(doc, "The learned optimal scalar temperature is T* = 2.6268 (converged in 43 iterations). When evaluated on the held-out Val_eval cohort (105 cases, 96 unique subjects, 311,546 voxels), temperature scaling achieved dramatic error reductions:")
    add_body_p(doc, "• Expected Calibration Error (ECE): Reduced from 10.85% to 4.83% (a 55.5% relative reduction).", bold_prefix="1. ECE Reduction: ")
    add_body_p(doc, "• Maximum Calibration Error (MCE): Reduced from 44.22% to 21.40% (a 51.6% relative reduction).", bold_prefix="2. MCE Reduction: ")
    add_body_p(doc, "• Negative Log-Likelihood (NLL): Reduced from 0.8273 to 0.3939 (a 52.4% relative reduction).", bold_prefix="3. NLL Reduction: ")
    add_body_p(doc, "• Brier Score: Reduced from 0.2263 to 0.1923 (a 15.0% relative reduction).", bold_prefix="4. Brier Score: ")
    add_body_p(doc, "• Mathematical Argmax Invariance: Temperature scaling strictly preserves discrete predictions: argmax(z/T) == argmax(z). Empirically verified across all 16,275 evaluated slices with exactly 0 voxel mismatches.", bold_prefix="5. Argmax Invariance: ")

    add_figure_with_caption(doc, DOCS_FIG_DIR / "04_unet_calibration_reliability_diagram.png",
                           "4.1", "U-Net 2D Reliability Diagrams: Uncalibrated vs Calibrated",
                           "15-bin confidence vs accuracy comparison on 105 held-out patients, showing elimination of the overconfidence gap.")

    # -------------------------------------------------------------------------
    # SECTION 5: UNCERTAINTY QUANTIFICATION & CLINICAL AGENT PROFILE
    # -------------------------------------------------------------------------
    style_heading_1(doc, "5. Uncertainty Quantification & Clinical Agent Profile")
    add_body_p(doc, "Predictive uncertainty was quantified across 16,275 slices from 105 held-out patients using Shannon predictive entropy: H(x) = -sum p_c log(p_c). Slices with zero predicted foreground evaluate strictly to None (preventing false 0.0 entropy substitution). Across 6,177 non-empty slices, mean predictive entropy was 0.2923 ± 0.0896 (median: 0.2701, IQR: [0.2293, 0.3325]).")

    add_body_p(doc, "Under the Stage C clinical operating threshold of tau = 0.35, the multi-model reliability agent triages predictions into three operational pathways:")
    add_body_p(doc, "• Autonomous Acceptance (H < 0.35): Approximately 81.6% of tumor slices are classified with high confidence and accepted for autonomous radiation treatment and surgical planning.", bold_prefix="Action 1: ")
    add_body_p(doc, "• Specialist Referral (H >= 0.35): Approximately 18.4% of tumor slices exhibit elevated boundary uncertainty and are flagged for mandatory human expert review.", bold_prefix="Action 2: ")
    add_body_p(doc, "• Fallback Model Rescue: When primary models produce complete false-negative misses, U-Net 2D acts as the high-sensitivity backup engine, successfully rescuing 65.9% of missed tumor slices.", bold_prefix="Action 3: ")

    add_figure_with_caption(doc, DOCS_FIG_DIR / "05_unet_predictive_entropy_distribution.png",
                           "5.1", "Slice Predictive Entropy Distribution & Clinical Triage",
                           "Density profile across 6,177 non-empty slices and cohort census breakdown between autonomous acceptance and specialist review.")

    # -------------------------------------------------------------------------
    # SECTION 6: FIVE-MODEL BENCHMARK COMPARISON
    # -------------------------------------------------------------------------
    style_heading_1(doc, "6. Five-Model Benchmark Comparison")
    add_body_p(doc, "To establish rigorous scientific context, U-Net 2D was benchmarked against all five frozen 15-epoch architectures under standardized evaluation protocols on the 105 held-out cases of Val_eval:")

    bench_table = doc.add_table(rows=6, cols=7)
    bench_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    bench_headers = ["Architecture", "Parameters", "Val Mean Dice", "Val_eval Vol Dice", "Calibrated ECE", "Inference Latency", "Miss Rate"]
    for i, h in enumerate(bench_headers):
        cell = bench_table.cell(0, i)
        set_cell_background(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=80, bottom=80, left=80, right=80)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.name = "Segoe UI"
        r.font.size = Pt(8.5)
        r.font.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)

    bench_rows = [
        ("U-Net 2D (Model 1)", "7.76 M", "0.8594 (Ep 15)", "0.8574", "4.83%", "9.60 ms", "4.58%"),
        ("Attention U-Net", "7.85 M", "0.8633 (Ep 15)", "0.8581", "5.68%", "14.09 ms", "4.81%"),
        ("ResUNet-2D", "8.11 M", "0.8659 (Ep 13)", "0.8641", "3.41%", "10.20 ms", "4.52%"),
        ("U-Net++ (Nested)", "9.16 M", "0.8601 (Ep 15)", "0.8549", "5.51%", "15.14 ms", "12.53%"),
        ("UNet-ResNet34", "24.33 M", "0.8674 (Ep 15)", "0.8535", "4.96%", "18.50 ms", "4.65%")
    ]
    for row_idx, r_data in enumerate(bench_rows, start=1):
        for col_idx, text in enumerate(r_data):
            cell = bench_table.cell(row_idx, col_idx)
            is_unet = (row_idx == 1)
            bg = "EFF6FF" if is_unet else (HEX_ZEBRA_BG if row_idx % 2 == 1 else "FFFFFF")
            set_cell_background(cell, bg)
            set_cell_margins(cell, top=60, bottom=60, left=70, right=70)
            p = cell.paragraphs[0]
            r = p.add_run(text)
            r.font.name = "Segoe UI"
            r.font.size = Pt(8.5)
            r.font.color.rgb = COLOR_DARK
            if is_unet:
                r.font.bold = True
                if col_idx in [0, 4, 5]:
                    r.font.color.rgb = COLOR_BLUE

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    add_figure_with_caption(doc, DOCS_FIG_DIR / "06_five_model_benchmark_radar_and_bars.png",
                           "6.1", "Five-Model Comparative Benchmark Analysis",
                           "Volumetric segmentation accuracy, parameter footprint, calibrated ECE, and runtime latency comparisons.")

    # -------------------------------------------------------------------------
    # SECTION 7: METHODOLOGICAL DISCLOSURES & SCIENTIFIC REPRODUCIBILITY
    # -------------------------------------------------------------------------
    style_heading_1(doc, "7. Methodological Disclosures & Scientific Reproducibility")
    add_body_p(doc, "In accordance with medical AI reporting standards (STARD-AI / TRIPOD-AI) and project governance blueprint v2.0, the following methodological boundaries are formally disclosed:")

    add_body_p(doc, "Validation Dice scores evaluated on the 155-case validation pool are formally classified as selection-exposed internal validation metrics because checkpoints were selected based on maximum performance on this pool.", bold_prefix="1. Selection Exposure Disclosure: ")
    add_body_p(doc, "The sealed 155-case test cohort (data/quarantine_test/) and test manifest (results/dataset_splits/test_manifest.json) remained 100% UNTOUCHED throughout model training, calibration, and evaluation. Zero test data was accessed.", bold_prefix="2. Test Quarantine Integrity: ")
    add_body_p(doc, "Positive scalar temperature scaling preserves discrete classification assignments: argmax_c (z_c / T) == argmax_c z_c. Empirically proven with 0 errors across all 16,275 slices.", bold_prefix="3. Mathematical Invariance: ")
    add_body_p(doc, "All weights, hyperparameters, logs, and evaluation metrics are preserved under cryptographic SHA-256 hashes for 100% bit-for-bit auditability.", bold_prefix="4. Checkpoint Integrity: ")

    # Save Document
    docx_out = DOCS_DIR / "UNet_2D_Model_Master_Dossier.docx"
    doc.save(str(docx_out))
    print(f"Successfully saved Word Document: {docx_out}")

    # Output is already saved directly into REPORTS_DOSSIER_DIR
    print(f"Saved directly to reports dossier: {docx_out}")

    return docx_out


def convert_docx_to_pdf(docx_path):
    print("Converting DOCX to PDF via Word COM...")
    pdf_out = docx_path.with_suffix(".pdf")
    ps_cmd = f"""
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    try {{
        $doc = $word.Documents.Open('{str(docx_path)}')
        $doc.SaveAs([ref]'{str(pdf_out)}', [ref]17)
        $doc.Close()
        Write-Output 'PDF conversion successful'
    }} catch {{
        Write-Output "PDF conversion error: $_"
    }} finally {{
        $word.Quit()
    }}
    """
    res = subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True, text=True)
    print(res.stdout)
    if pdf_out.exists():
        print(f"Successfully generated PDF directly in reports: {pdf_out}")
    else:
        print("PDF conversion did not create file. Check Word COM.")


if __name__ == "__main__":
    docx_file = build_docx()
    convert_docx_to_pdf(docx_file)
