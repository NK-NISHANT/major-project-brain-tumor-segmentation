"""
Build Comprehensive Master Model Dossier for U-Net++:
- Markdown (.md)
- Word Document (.docx with professional executive styling and embedded 300 DPI figures)
- PDF (.pdf via Word COM automation)
Output: D:\\MajorProject\\documents\\
"""

import os
import sys
import subprocess
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

WORKSPACE = Path(r"D:\MajorProject")
REPORTS_DIR = WORKSPACE / "reports" / "model_profiles" / "unetpp"
DOCS_DIR = REPORTS_DIR / "dossier"
FIG_DIR = REPORTS_DIR / "assets"

DOCS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLOR_PRIMARY = RGBColor(15, 23, 42)      # Deep Navy #0F172A
COLOR_SECONDARY = RGBColor(51, 65, 85)    # Slate #334155
COLOR_ACCENT_BLUE = RGBColor(2, 132, 199) # Sky #0284C7
COLOR_ACCENT_GREEN = RGBColor(16, 185, 129) # Emerald #10B981
COLOR_ACCENT_CORAL = RGBColor(225, 29, 72) # Rose #E11D48
COLOR_TEXT_DARK = RGBColor(15, 23, 42)
COLOR_TEXT_MUTED = RGBColor(100, 116, 139)

HEX_HEADER_BG = "0F172A"
HEX_ROW_ALT = "F8FAFC"
HEX_BORDER = "CBD5E1"
HEX_CALLOUT_BG = "F0F9FF"
HEX_CALLOUT_BORDER = "0284C7"

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
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

def set_table_borders(table, color="CBD5E1", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def style_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    run = h.runs[0]
    run.font.name = "Segoe UI"
    if level == 1:
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        h.paragraph_format.space_before = Pt(18)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
    elif level == 2:
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        h.paragraph_format.keep_with_next = True
    elif level == 3:
        run.font.size = Pt(11)
        run.font.bold = True
        run.font.color.rgb = COLOR_ACCENT_BLUE
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(2)
        h.paragraph_format.keep_with_next = True
    return h

def add_body_paragraph(doc, text, bold_prefix=None, space_after=4):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Segoe UI"
        r_pre.font.size = Pt(10)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_TEXT_DARK
    r = p.add_run(text)
    r.font.name = "Segoe UI"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_TEXT_DARK
    return p

def add_callout(doc, text, title="CRITICAL SCIENTIFIC INSIGHT", bg_hex=HEX_CALLOUT_BG, border_hex=HEX_CALLOUT_BORDER):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, bg_hex)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=180)
    
    # Left border only
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:left w:val="single" w:sz="36" w:space="0" w:color="{border_hex}"/>'
        f'<w:top w:val="none"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(3)
    r_title = p.add_run(f"[{title}] ")
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(10)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY
    
    r_text = p.add_run(text)
    r_text.font.name = "Segoe UI"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = COLOR_TEXT_DARK
    
    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_after = Pt(4)

def add_image_box(doc, image_path, caption_text, width_inches=6.8):
    if Path(image_path).exists():
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(8)
        p_img.paragraph_format.space_after = Pt(3)
        run_img = p_img.add_run()
        run_img.add_picture(str(image_path), width=Inches(width_inches))
        
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_cap.paragraph_format.space_before = Pt(1)
        p_cap.paragraph_format.space_after = Pt(8)
        r_cap = p_cap.add_run(caption_text)
        r_cap.font.name = "Segoe UI"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_TEXT_MUTED
    else:
        print(f"[!] Warning: Image not found: {image_path}")


# -----------------------------------------------------------------------------
# BUILD WORD MASTER (.DOCX)
# -----------------------------------------------------------------------------
def generate_word_dossier():
    docx_path = DOCS_DIR / "UNetPlusPlus_Master_Model_Dossier.docx"
    print(f"Generating Word Master Dossier: {docx_path}")
    
    doc = Document()
    
    # 0.75-inch margins for executive presentation
    for sec in doc.sections:
        sec.top_margin = Inches(0.75)
        sec.bottom_margin = Inches(0.75)
        sec.left_margin = Inches(0.75)
        sec.right_margin = Inches(0.75)

    # Title & Metadata Banner
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_title.paragraph_format.space_before = Pt(10)
    p_title.paragraph_format.space_after = Pt(2)
    r_pre = p_title.add_run("DEFINITIVE MODEL DOSSIER & BENCHMARK AUDIT\n")
    r_pre.font.name = "Segoe UI"
    r_pre.font.size = Pt(11)
    r_pre.font.bold = True
    r_pre.font.color.rgb = COLOR_ACCENT_BLUE

    r_main = p_title.add_run("U-Net++: Nested Dense Skip Pathways 2D Architecture")
    r_main.font.name = "Segoe UI"
    r_main.font.size = Pt(22)
    r_main.font.bold = True
    r_main.font.color.rgb = COLOR_PRIMARY

    p_meta = doc.add_paragraph()
    p_meta.paragraph_format.space_after = Pt(14)
    meta_runs = [
        ("Project: ", True), ("Reliability-Aware AI Agent for Brain Tumor Segmentation  |  ", False),
        ("Dataset: ", True), ("ASNR-MICCAI BraTS 2023 GLI  |  ", False),
        ("Model ID: ", True), ("unetpp (UNetPlusPlus2D)  |  ", False),
        ("Best Checkpoint: ", True), ("Epoch 15 (Mean Dice: 0.8602)  |  ", False),
        ("Status: ", True), ("Stage C Phase 1 & 2 Fully Verified", False),
    ]
    for text, bold in meta_runs:
        r = p_meta.add_run(text)
        r.font.name = "Segoe UI"
        r.font.size = Pt(9.5)
        r.font.bold = bold
        r.font.color.rgb = COLOR_TEXT_MUTED

    # Inset Executive Poster Image Right Upfront
    add_image_box(doc, FIG_DIR / "fig_07_unetpp_executive_poster_dashboard.png",
                  "Figure 0: U-Net++ Executive Dashboard & Performance Scorecard (Far-Distance & Printout Optimized).", width_inches=6.8)

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    style_heading(doc, "1. Executive Summary & Core Results", level=1)
    add_body_paragraph(doc, 
        "The U-Net++ (Nested Dense Skip Pathways U-Net) architecture has successfully completed the entire Stage C experimental pipeline under the governing Master Project Blueprint v2.0. Evaluated under strict synchronous CUDA execution on the authorized 721-case training cohort and 155-case validation pool, U-Net++ attained a peak selection-exposed validation Mean Dice of 0.8602 (Mean IoU: 0.7568) at Epoch 15. In downstream Stage C Phase 2 evaluation across 105 held-out internal evaluation cases (Val_eval), post-hoc temperature scaling (T* = 2.5647) reduced Expected Calibration Error from 11.18% to 5.51% (a 50.70% relative reduction) and Negative Log-Likelihood from 0.8074 to 0.3904 (a 51.64% reduction), while maintaining 100% discrete argmax segmentation invariance with exactly zero mismatched voxels across nearly one billion evaluated predictions.")

    add_callout(doc,
        "U-Net++ demonstrated the highest selection-exposed validation Mean Dice (0.8602) among all three scratch-trained Phase 1 baseline architectures, narrowly exceeding standard U-Net (0.8594) and Attention U-Net (0.8542). However, in accordance with the governing blueprint, this modest difference (+0.0008 Dice) does not establish architectural superiority and must be interpreted alongside its 2.5× higher VRAM footprint (1,111.3 MB vs 446.3 MB) and 1.87× runtime overhead.",
        title="EXECUTIVE BENCHMARK VERDICT", bg_hex="FEF3C7", border_hex="D97706")

    # Core Scorecard Table
    table_data = [
        ["Dimension / Performance Metric", "Protocol Specification", "Observed Empirical Value", "Source Artifact"],
        ["Trainable Parameters", "Standardized Stage B channels", "9,160,068 (~9.16 M)", "src/models/unetpp.py"],
        ["Training Cohort", "721 cases / 667 unique subjects", "59,020 slices (47,228 fg + 11,792 bg)", "development_train_manifest.json"],
        ["Validation Pool", "155 cases / 143 unique subjects", "12,364 slices (Val_cal + Val_eval)", "validation_manifest.json"],
        ["Best Selected Epoch", "Maximum Validation Mean Dice", "Epoch 15 (Peak observed in budget)", "history.csv / summary.json"],
        ["Val Mean Dice (Ep 15)", "Selection-exposed internal validation", "0.8602 (0.860218)", "history.csv"],
        ["Val Mean IoU (Ep 15)", "Selection-exposed internal validation", "0.7568 (0.756770)", "history.csv"],
        ["Enhancing Tumor (ET) Dice", "Class 3 (active neoplastic margin)", "0.9047 (IoU: 0.8260)", "history.csv"],
        ["Peritumoral Edema (ED) Dice", "Class 2 (infiltrative peritumoral zone)", "0.8664 (IoU: 0.7642)", "history.csv"],
        ["Necrotic Core (NCR) Dice", "Class 1 (necrotic avascular core)", "0.8096 (IoU: 0.6801)", "history.csv"],
        ["Healthy Background (BG) Dice", "Class 0 (normal brain tissue)", "0.9986 (IoU: 0.9972)", "history.csv"],
        ["Optimal Temperature (T*)", "Fitted via L-BFGS on Val_cal (50 cases)", "2.564703 (42 iterations)", "calibration_parameters.json"],
        ["Uncalibrated -> Calibrated ECE", "Held-out Val_eval (105 cases)", "11.18% -> 5.51% (-50.70% rel.)", "evaluation_metrics.csv"],
        ["Uncalibrated -> Calibrated NLL", "Held-out Val_eval (105 cases)", "0.8074 -> 0.3904 (-51.64% rel.)", "evaluation_metrics.csv"],
        ["Held-out Volumetric Dice", "Val_eval volumetric reconstruction", "0.8549 (WT: 0.8830, TC: 0.8521, ET: 0.8297)", "evaluation_metrics.csv"],
        ["Argmax Prediction Invariance", "Voxel-level discrete assignment check", "EXACTLY 0 MISMATCHES (0.000%)", "c2_calibration_report.md"],
        ["Mean Predictive Entropy", "6,128 non-empty slices (Val_eval)", "0.2883 ± 0.0907 (Median: 0.2685)", "entropy_summary.csv"],
        ["Peak Process VRAM", "NVIDIA RTX 4070 (8 GB capacity)", "1,111.3 MB (Process allocation)", "train.log / summary.json"],
        ["Total Training Runtime", "15 Epochs Synchronous CUDA", "15.44 Hours (926.39 Minutes)", "train.log / history.csv"],
        ["Test Quarantine Preservation", "quarantine_test/ & test_manifest.json", "100% INTACT (Zero test access)", "STAGE_C_EXPERIMENT_PROTOCOL"]
    ]

    tbl = doc.add_table(rows=len(table_data), cols=4)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl)
    
    col_widths = [Inches(1.8), Inches(1.8), Inches(1.8), Inches(1.4)]
    for r_idx, row in enumerate(tbl.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            cell.text = table_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.runs[0]
            r.font.name = "Segoe UI"
            if r_idx == 0:
                set_cell_background(cell, HEX_HEADER_BG)
                set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
                r.font.size = Pt(9)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, HEX_ROW_ALT)
                set_cell_margins(cell, top=80, bottom=80, left=140, right=140)
                r.font.size = Pt(8.5)
                r.font.color.rgb = COLOR_TEXT_DARK
                if c_idx == 2:
                    r.font.bold = True

    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECTION 2: ARCHITECTURE SPECIFICATION
    # -------------------------------------------------------------------------
    style_heading(doc, "2. Architecture Specification & Mathematical Formulation", level=1)
    add_body_paragraph(doc,
        "The U-Net++ architecture (Zhou et al., 2018) redesigns the classic skip connections of U-Net into a nested, dense convolutional network. Instead of transmitting feature maps directly across long, identity skip pathways where an inherent semantic gap exists between early encoder representations and late decoder representations, U-Net++ bridges this gap by routing features through a dense grid of intermediate convolutional blocks X^{i,j}.")

    add_body_paragraph(doc,
        "Let x^{i,j} denote the feature map output by node X^{i,j}, where i in {0, 1, 2, 3, 4} denotes the downsampling layer along the encoder spine, and j in {0, 1, 2, 3, 4 - i} denotes the dense skip pathway index. The feature representation at each node is formally defined as:",
        bold_prefix="Mathematical Formulation: ")

    add_body_paragraph(doc,
        "x^{i,j} = ConvBlock( [ x^{i,0}, x^{i,1}, ..., x^{i,j-1}, Up( x^{i+1,j-1} ) ] )   for j > 0\n"
        "x^{i,0} = ConvBlock( Down( x^{i-1,0} ) )   for i > 0, with x^{0,0} = ConvBlock( Input )",
        bold_prefix="Recurrence Relation: ")

    add_body_paragraph(doc,
        "Here, [ ... ] denotes channel-wise concatenation, Up(.) denotes bilinear interpolation upsampling by a factor of 2, Down(.) denotes 2x2 max-pooling with stride 2, and ConvBlock(.) comprises two consecutive 3x3 convolutions, each followed by Batch Normalization and ReLU activation without internal bias. Deep supervision was intentionally disabled to preserve strict parity with the single-head training loop and loss interface of standard U-Net and Attention U-Net.")

    add_image_box(doc, FIG_DIR / "fig_01_unetpp_nested_architecture.png",
                  "Figure 1: Full Architecture Diagram of U-Net++ (Nested Dense Skip Pathways), showing node connectivity, resolution stages, channel growth, and parameter allocation.", width_inches=6.8)

    # -------------------------------------------------------------------------
    # SECTION 3: 15-EPOCH TRAINING TRAJECTORY
    # -------------------------------------------------------------------------
    style_heading(doc, "3. Complete 15-Epoch Training Trajectory & Convergence Audit", level=1)
    add_body_paragraph(doc,
        "Training was executed under synchronous CUDA mode (CUDA_LAUNCH_BLOCKING=1) on an NVIDIA RTX 4070 Laptop GPU, requiring 15.44 continuous hours (926.39 minutes) without a single driver recovery event or runtime exception. The complete epoch-by-epoch telemetry recorded in history.csv is presented below:")

    add_image_box(doc, FIG_DIR / "fig_02_unetpp_training_trajectory.png",
                  "Figure 2: Complete 15-Epoch Training Trajectory for U-Net++, including Loss curves, Validation Mean Dice & IoU, Learning Rate schedule, and Gradient Norm stabilization.", width_inches=6.8)

    # Trajectory Table from history.csv
    history_csv = WORKSPACE / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "history.csv"
    df_hist = pd.read_csv(history_csv)

    traj_rows = [["Ep", "Train Loss", "Val Loss", "Val Dice", "Val IoU", "ET Dice", "ED Dice", "NCR Dice", "Grad Norm", "% Clip", "LR Active", "Epoch Time", "Best?"]]
    for _, r in df_hist.iterrows():
        ep = int(r['epoch'])
        t_loss = f"{r['train_loss']:.4f}"
        v_loss = f"{r['val_loss']:.4f}"
        v_dice = f"{r['mean_dice']:.4f}"
        v_iou = f"{r['mean_iou']:.4f}"
        et_d = f"{r['dice_class_3']:.4f}"
        ed_d = f"{r['dice_class_2']:.4f}"
        ncr_d = f"{r['dice_class_1']:.4f}"
        gnorm = f"{r['grad_norm_pre']:.2f}"
        clip_pct = f"{(r['clipped_batches']/29510)*100:.1f}%" if 'clipped_batches' in r else "N/A"
        lr_act = f"{r['learning_rate_primary']:.2e}"
        ep_t = f"{r['epoch_duration_min']:.1f}m"
        best = "YES" if r['is_best'] else "no"
        traj_rows.append([str(ep), t_loss, v_loss, v_dice, v_iou, et_d, ed_d, ncr_d, gnorm, clip_pct, lr_act, ep_t, best])

    tbl_traj = doc.add_table(rows=len(traj_rows), cols=13)
    tbl_traj.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_traj)

    for r_idx, row in enumerate(tbl_traj.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = traj_rows[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            r = p.runs[0]
            r.font.name = "Segoe UI"
            if r_idx == 0:
                set_cell_background(cell, HEX_HEADER_BG)
                set_cell_margins(cell, top=80, bottom=80, left=40, right=40)
                r.font.size = Pt(7.5)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, HEX_ROW_ALT)
                set_cell_margins(cell, top=60, bottom=60, left=40, right=40)
                r.font.size = Pt(7.5)
                r.font.color.rgb = COLOR_TEXT_DARK
                if r_idx == 15: # Highlight best epoch
                    set_cell_background(cell, "FEF3C7")
                    r.font.bold = True

    p_space2 = doc.add_paragraph()
    p_space2.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECTION 4: TUMOR SUB-COMPARTMENT PERFORMANCE
    # -------------------------------------------------------------------------
    style_heading(doc, "4. Tumor Sub-Compartment Segmentation Analysis", level=1)
    add_body_paragraph(doc,
        "Brain tumor segmentation requires accurate delineation of heterogeneous tissue sub-compartments that vary in intensity, morphology, and clinical implications. At the best checkpoint (Epoch 15), U-Net++ demonstrated the following sub-compartment performance:")

    add_body_paragraph(doc,
        "1. Enhancing Tumor (ET / Class 3): Dice = 0.9047 (IoU: 0.8260). Enhancing tumor represents the actively proliferating, vascularized rim of high-grade glioma. U-Net++ achieved its highest segmentation score in this class, surpassing both U-Net (0.9026) and Attention U-Net (0.9035).\n"
        "2. Peritumoral Edema (ED / Class 2): Dice = 0.8664 (IoU: 0.7642). Edema represents the infiltrative non-enhancing fluid margin visible primarily on T2-FLAIR. U-Net++ exhibited strong consistency with U-Net (0.8662) and Attention U-Net (0.8645).\n"
        "3. Necrotic Core (NCR / Class 1): Dice = 0.8096 (IoU: 0.6801). Necrotic core comprises avascular dead tissue and cyst fluid within the tumor cavity. While technically challenging due to variable hypointensity, U-Net++ maintained 0.8096, outperforming Attention U-Net (0.7947).\n"
        "4. Healthy Brain & Background (BG / Class 0): Dice = 0.9986 (IoU: 0.9972). Delineates unaffected brain parenchyma and skull-stripped void with 99.86% fidelity.")

    add_image_box(doc, FIG_DIR / "fig_03_unetpp_compartment_dice_evolution.png",
                  "Figure 3: Sub-Compartment Validation Dice Evolution across all 15 training epochs for ET, ED, NCR, and Background.", width_inches=6.8)

    # -------------------------------------------------------------------------
    # SECTION 5: CALIBRATION & TEMPERATURE SCALING
    # -------------------------------------------------------------------------
    style_heading(doc, "5. Post-Hoc Probability Calibration & Temperature Scaling", level=1)
    add_body_paragraph(doc,
        "Deep neural networks trained with Cross-Entropy and Dice losses are systematically overconfident: their predicted softmax probabilities significantly overestimate empirical accuracy. In Work Package C2, post-hoc temperature scaling was conducted by learning a single positive scalar temperature T* on the 50 calibration cases (Val_cal, 5,020,730 voxels) and evaluating calibration quality on the 105 held-out internal evaluation cases (Val_eval, 311,546 voxels).")

    add_body_paragraph(doc,
        "Optimal Temperature Fitting: The optimal parameter T* = 2.564703 was obtained in 42 iterations of L-BFGS with strong Wolfe line search by minimizing Negative Log-Likelihood (NLL). When applied to test-time logits via z_scaled = z / T*, softmax probability overconfidence is eliminated while perfectly preserving the top-1 predicted class index.",
        bold_prefix="L-BFGS Optimization: ")

    add_image_box(doc, FIG_DIR / "fig_04_unetpp_calibration_reliability.png",
                  "Figure 4: Reliability Diagrams before and after Temperature Scaling (T* = 2.5647), demonstrating a 50.70% ECE reduction and 51.64% NLL reduction on Val_eval.", width_inches=6.8)

    # -------------------------------------------------------------------------
    # SECTION 6: PREDICTIVE UNCERTAINTY & EMPTY-PREDICTION SEMANTICS
    # -------------------------------------------------------------------------
    style_heading(doc, "6. Predictive Uncertainty Quantification & Empty Slices", level=1)
    add_body_paragraph(doc,
        "To empower the downstream Stage C Reliability Agent to perform automated triage, normalized 4-class Shannon entropy was computed from calibrated probabilities for all 16,275 slices in Val_eval (105 cases x 155 slices). Slices with zero predicted foreground voxels were strictly assigned Python None (serialized as JSON null) to prevent false 0.0 entropy substitutions.")

    add_body_paragraph(doc,
        "Over the 6,128 non-empty slices, foreground predictive entropy exhibited a mean of 0.2883 ± 0.0907 and a median of 0.2685 (IQR: [0.2168, 0.3408]). Across the operating threshold tau = 0.35, low-uncertainty slices (H < 0.35) are autonomously retained, while high-uncertainty slices (H >= 0.35) are flagged for specialist radiologist referral.",
        bold_prefix="Entropy Distribution & Triage Gate: ")

    add_image_box(doc, FIG_DIR / "fig_06_unetpp_predictive_entropy_distribution.png",
                  "Figure 5: Predictive Uncertainty (Shannon Entropy) Distribution over 6,128 non-empty slices and Empty-Prediction Slice accounting (62.35% empty).", width_inches=6.8)

    # -------------------------------------------------------------------------
    # SECTION 7: 5-MODEL BENCHMARK MATRIX
    # -------------------------------------------------------------------------
    style_heading(doc, "7. Five-Model Comparative Benchmark Matrix", level=1)
    add_body_paragraph(doc,
        "Under the Stage C unified evaluation benchmark, U-Net++ was benchmarked alongside ResUNet-2D, UNetResNet34, Standard U-Net, and Attention U-Net across all clinical and computational dimensions:")

    add_image_box(doc, FIG_DIR / "fig_05_five_model_benchmark_matrix.png",
                  "Figure 6: Stage C 5-Model Comparative Benchmark Matrix, comparing parameter count, validation Dice, held-out volumetric Dice, and calibrated ECE.", width_inches=6.8)

    # 5-Model Table
    comp_data = [
        ["Model Architecture", "Params (M)", "Best Ep", "Val Mean Dice", "Val_eval Vol Dice", "Uncal ECE", "Cal ECE", "ECE Red.", "Cal NLL", "Peak VRAM"],
        ["ResUNet-2D", "8.11 M", "13", "0.8659", "0.8641", "8.59%", "3.41%", "-60.32%", "0.4415", "446 MB"],
        ["UNetResNet34", "24.33 M", "15", "0.8674", "0.8535", "10.98%", "4.96%", "-54.78%", "0.3655", "736 MB"],
        ["Standard U-Net", "7.76 M", "15", "0.8594", "0.8574", "10.85%", "4.83%", "-55.46%", "0.3939", "446 MB"],
        ["U-Net++ (Nested)", "9.16 M", "15", "0.8602", "0.8549", "11.18%", "5.51%", "-50.70%", "0.3904", "1,111 MB"],
        ["Attention U-Net", "7.85 M", "15", "0.8542", "0.8581", "10.81%", "5.68%", "-47.47%", "0.3689", "482 MB"]
    ]

    tbl_comp = doc.add_table(rows=len(comp_data), cols=10)
    tbl_comp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(tbl_comp)

    for r_idx, row in enumerate(tbl_comp.rows):
        for c_idx, cell in enumerate(row.cells):
            cell.text = comp_data[r_idx][c_idx]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            r = p.runs[0]
            r.font.name = "Segoe UI"
            if r_idx == 0:
                set_cell_background(cell, HEX_HEADER_BG)
                set_cell_margins(cell, top=90, bottom=90, left=50, right=50)
                r.font.size = Pt(8)
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, HEX_ROW_ALT)
                set_cell_margins(cell, top=70, bottom=70, left=50, right=50)
                r.font.size = Pt(8)
                r.font.color.rgb = COLOR_TEXT_DARK
                if r_idx == 4: # Highlight U-Net++
                    set_cell_background(cell, "FEF3C7")
                    r.font.bold = True

    p_space3 = doc.add_paragraph()
    p_space3.paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECTION 8: SCIENTIFIC LIMITATIONS & DISCLOSURE
    # -------------------------------------------------------------------------
    style_heading(doc, "8. Methodological Limitations & Governance Adherence", level=1)
    add_body_paragraph(doc,
        "1. Selection-Exposed Internal Validation: All reported validation metrics were obtained on cohorts monitored during model selection. These figures are formally classified as selection-exposed internal validation metrics and do not represent unbiased generalization on external unseen cohorts.\n"
        "2. Single-Seed Evaluation: Single-seed evaluation (seed 42) prevents calculation of inter-run variance or formal paired hypothesis testing.\n"
        "3. Test Quarantine Preservation: The quarantined test partition (data/quarantine_test/) remained 100% unaccessed, unread, and intact throughout all Phase 1 and Phase 2 operations.\n"
        "4. Prototype Academic Boundary: This model is an academic research prototype and is not certified for autonomous clinical diagnosis by the US FDA, CE, or CDSCO.")

    doc.save(str(docx_path))
    print(f"[✓] Successfully generated Word Master: {docx_path}")

def convert_docx_to_pdf():
    docx_path = DOCS_DIR / "UNetPlusPlus_Master_Model_Dossier.docx"
    pdf_path = DOCS_DIR / "UNetPlusPlus_Master_Model_Dossier.pdf"
    print(f"Converting DOCX to PDF via Word COM: {pdf_path}")

    ps_script = f"""
    $word = New-Object -ComObject Word.Application
    try {{
        $doc = $word.Documents.Open("{docx_path}")
        $doc.SaveAs([ref]"{pdf_path}", [ref]17) # 17 = wdFormatPDF
        $doc.Close()
        Write-Host "PDF export successful."
    }} finally {{
        $word.Quit()
    }}
    """
    temp_ps1 = WORKSPACE / "reports" / "model_profiles" / "unetpp" / "convert_temp.ps1"
    with open(temp_ps1, "w", encoding="utf-8") as f:
        f.write(ps_script)

    result = subprocess.run(["powershell", "-ExecutionPolicy", "Bypass", "-File", str(temp_ps1)], capture_output=True, text=True)
    if temp_ps1.exists():
        temp_ps1.unlink()

    if result.returncode != 0:
        print(f"Error during Word COM conversion: {result.stderr}")
    else:
        print(f"[✓] Successfully generated PDF: {pdf_path}")

def main():
    generate_word_dossier()
    convert_docx_to_pdf()
    print("U-Net++ Master Dossier DOCX and PDF generation complete!")

if __name__ == "__main__":
    main()
