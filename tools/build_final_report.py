# -*- coding: utf-8 -*-
"""
Build Final Research Edition Report for Major Project-II.
Project: Reliability-Aware AI Agent for Brain Tumor Segmentation
Governance Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Scientific Evidence Authority: G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md
Author: Nishant Kumar, Mohammad Tariq, Mohammad Kaif
Supervisor: Dr. Ruchika Bala
Institution: Jaypee Institute of Information Technology, Noida
"""

import argparse
import hashlib
import os
import sys
import uuid
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = SCRIPT_DIR.parent

DEFAULT_OUT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_3.docx"
)
DEFAULT_MATERIALS = WORKSPACE_ROOT / "Report_Materials"
DEFAULT_CHARTS_DIR = WORKSPACE_ROOT / "tmp" / "report_figures"
DEFAULT_ARTWORK_DIR = WORKSPACE_ROOT / "tmp" / "report_artwork_research_v4_1_3"
DEFAULT_SAMPLE_VIZ = (
    WORKSPACE_ROOT
    / "results"
    / "baseline_unet2d_5ep"
    / "visualizations"
    / "sample_01_BraTS-GLI-01416-000_slice_090.png"
)
PROTECTED_ORIGINAL_REPORT = (
    WORKSPACE_ROOT / "Final_Report_Brain_Tumor_Segmentation.docx"
)
PROTECTED_V1_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition.docx"
)
PROTECTED_V2_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v2.docx"
)
PROTECTED_V3_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v3.docx"
)
PROTECTED_V4_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx"
)
PROTECTED_V4_1_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx"
)
PROTECTED_V4_1_1_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx"
)
PROTECTED_V4_1_2_REPORT = (
    WORKSPACE_ROOT
    / "docs"
    / "reports"
    / "Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx"
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build the Final Major Project-II Research Edition Report DOCX."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path for generated output DOCX (default: %(default)s)",
    )
    parser.add_argument(
        "--materials",
        type=Path,
        default=DEFAULT_MATERIALS,
        help="Path to Report_Materials directory (default: %(default)s)",
    )
    parser.add_argument(
        "--charts-dir",
        type=Path,
        default=DEFAULT_CHARTS_DIR,
        help="Path to pre-rendered charts directory (strictly read-only) (default: %(default)s)",
    )
    parser.add_argument(
        "--artwork-dir",
        type=Path,
        default=DEFAULT_ARTWORK_DIR,
        help="Path to generated artwork directory (default: %(default)s)",
    )
    parser.add_argument(
        "--sample-viz",
        type=Path,
        default=DEFAULT_SAMPLE_VIZ,
        help="Path to sample visualization image for Figure 4.3 (default: %(default)s)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Explicitly permit replacing an existing output DOCX file",
    )
    return parser.parse_args(argv)


def is_relative_to(p1: Path, p2: Path) -> bool:
    try:
        p1.resolve().relative_to(p2.resolve())
        return True
    except ValueError:
        return False


def validate_preflight_inputs(
    out_path: Path,
    materials_dir: Path,
    charts_dir: Path,
    artwork_dir: Path,
    sample_viz: Path,
    overwrite: bool,
):
    resolved_charts = charts_dir.resolve()
    resolved_artwork = artwork_dir.resolve()
    resolved_materials = materials_dir.resolve()
    resolved_out = out_path.resolve()

    # Protection invariant: The historical original report must never be overwritten.
    if resolved_out == PROTECTED_ORIGINAL_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The original historical report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Specify a distinct output destination."
        )

    # Protection invariant: Research Edition V1 must never be overwritten.
    if resolved_out == PROTECTED_V1_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V1 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx."
        )

    # Protection invariant: Research Edition V2 must never be overwritten.
    if resolved_out == PROTECTED_V2_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V2 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4.docx."
        )

    # Protection invariant: Research Edition V3 must never be overwritten.
    if resolved_out == PROTECTED_V3_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V3 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4.1 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1.docx."
        )

    # Protection invariant: Research Edition V4 must never be overwritten.
    if resolved_out == PROTECTED_V4_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V4 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4.1.1 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_1.docx."
        )

    # Protection invariant: Research Edition V4.1 must never be overwritten.
    if resolved_out == PROTECTED_V4_1_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V4.1 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4.1.2 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx."
        )

    # Protection invariant: Research Edition V4.1.1 must never be overwritten.
    if resolved_out == PROTECTED_V4_1_1_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V4.1.1 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4.1.2 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_2.docx."
        )

    # Protection invariant: Research Edition V4.1.2 must never be overwritten.
    if resolved_out == PROTECTED_V4_1_2_REPORT.resolve():
        raise PermissionError(
            f"Protection invariant violation: The historical Research Edition V4.1.2 report '{resolved_out}' "
            "is strictly protected and must never be overwritten. Target V4.1.3 instead: docs/reports/Final_Report_Brain_Tumor_Segmentation_Research_Edition_v4_1_3.docx."
        )

    # 1. Materials directory check
    if not materials_dir.exists():
        raise FileNotFoundError(f"Materials directory does not exist: {materials_dir}")
    if not materials_dir.is_dir():
        raise NotADirectoryError(f"Materials path is not a directory: {materials_dir}")

    # 2. Required materials check
    format_template = (
        materials_dir
        / "Final_Minor_Project-2_report_and_PPT_format_&_instructions_Odd2025[2].docx"
    )
    required_materials = [
        format_template,
        materials_dir / "Brain Tumor Segmentation Project Overview.png",
        materials_dir / "Brain Tumor Segmentation Benchmark Infographic.png",
        materials_dir / "Uncertainty Analysis for Brain Tumor Segmentation.png",
        materials_dir / "Reliability-Aware Brain Tumor AI Agent.png",
    ]
    for mat in required_materials:
        if not mat.exists():
            raise FileNotFoundError(f"Required material file missing: {mat}")
        if not os.access(mat, os.R_OK):
            raise PermissionError(f"Required material file is not readable: {mat}")

    try:
        _ = Document(str(format_template))
    except Exception as e:
        raise ValueError(
            f"Format template '{format_template}' is not a valid readable DOCX: {e}"
        )

    for mat in required_materials[1:]:
        try:
            with Image.open(mat) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Material image '{mat}' cannot be decoded: {e}")

    # 3. Charts directory check (strictly read-only source)
    if not charts_dir.exists():
        raise FileNotFoundError(f"Charts directory does not exist: {charts_dir}")
    if not charts_dir.is_dir():
        raise NotADirectoryError(f"Charts path is not a directory: {charts_dir}")

    required_charts = [
        charts_dir / "model_metrics.png",
        charts_dir / "per_class_dice.png",
        charts_dir / "loss_curves.png",
        charts_dir / "mean_dice_trend.png",
        charts_dir / "miss_rate.png",
    ]
    for chart in required_charts:
        if not chart.exists():
            raise FileNotFoundError(f"Required chart file missing: {chart}")
        if not os.access(chart, os.R_OK):
            raise PermissionError(f"Required chart file is not readable: {chart}")
        try:
            with Image.open(chart) as img:
                img.verify()
        except Exception as e:
            raise ValueError(f"Chart image '{chart}' cannot be decoded: {e}")

    # 4. Sample visualization check
    if not sample_viz.exists():
        raise FileNotFoundError(f"Sample visualization file missing: {sample_viz}")
    if not os.access(sample_viz, os.R_OK):
        raise PermissionError(f"Sample visualization file is not readable: {sample_viz}")
    try:
        with Image.open(sample_viz) as img:
            img.verify()
    except Exception as e:
        raise ValueError(f"Sample visualization '{sample_viz}' cannot be decoded: {e}")

    # 5. Directory Safety & Overlap Protection
    if resolved_charts == resolved_artwork:
        raise ValueError(
            f"Path collision: --charts-dir and --artwork-dir must not point to the identical directory: {resolved_charts}"
        )
    if is_relative_to(resolved_artwork, resolved_charts):
        raise ValueError(
            f"Path collision: --artwork-dir ({resolved_artwork}) cannot be nested inside --charts-dir ({resolved_charts})"
        )
    if is_relative_to(resolved_charts, resolved_artwork):
        raise ValueError(
            f"Path collision: --charts-dir ({resolved_charts}) cannot be nested inside --artwork-dir ({resolved_artwork})"
        )
    if resolved_artwork == resolved_materials or is_relative_to(
        resolved_artwork, resolved_materials
    ):
        raise ValueError(
            f"Path collision: --artwork-dir ({resolved_artwork}) cannot be or be inside --materials ({resolved_materials})"
        )
    if is_relative_to(resolved_materials, resolved_artwork):
        raise ValueError(
            f"Path collision: --materials ({resolved_materials}) cannot be inside --artwork-dir ({resolved_artwork})"
        )
    if resolved_charts == resolved_materials or is_relative_to(
        resolved_charts, resolved_materials
    ):
        raise ValueError(
            f"Path collision: --charts-dir ({resolved_charts}) cannot be or be inside --materials ({resolved_materials})"
        )
    if is_relative_to(resolved_materials, resolved_charts):
        raise ValueError(
            f"Path collision: --materials ({resolved_materials}) cannot be inside --charts-dir ({resolved_charts})"
        )

    if is_relative_to(resolved_out, resolved_charts):
        raise ValueError(
            f"Output file path ({resolved_out}) cannot be inside --charts-dir ({resolved_charts})"
        )
    if is_relative_to(resolved_out, resolved_artwork):
        raise ValueError(
            f"Output file path ({resolved_out}) cannot be inside --artwork-dir ({resolved_artwork})"
        )
    if is_relative_to(resolved_out, resolved_materials):
        raise ValueError(
            f"Output file path ({resolved_out}) cannot be inside --materials ({resolved_materials})"
        )

    out_parent = resolved_out.parent
    if not out_parent.exists():
        raise FileNotFoundError(
            f"Output destination directory does not exist: {out_parent}"
        )
    if not os.access(out_parent, os.W_OK):
        raise PermissionError(
            f"Output destination directory is not writable: {out_parent}"
        )

    if resolved_out.exists() and not overwrite:
        raise FileExistsError(
            f"Output file '{resolved_out}' already exists. Use --overwrite to permit replacement."
        )

    # Invariant: If artwork directory already exists and contains any files, reject unconditionally.
    if resolved_artwork.exists():
        existing_artwork_items = list(resolved_artwork.iterdir())
        if existing_artwork_items:
            raise FileExistsError(
                f"Artwork directory '{resolved_artwork}' already contains {len(existing_artwork_items)} pre-existing file(s). "
                "Refusing to overwrite or contaminate existing artwork directory. Specify an empty or new directory. "
                "--overwrite authorizes replacement of the DOCX output only and must never authorize overwriting artwork files."
            )


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def set_cell_text(cell, text, bold=False, color=None, size=9, align=WD_ALIGN_PARAGRAPH.CENTER):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run(str(text))
    r.bold = bold
    r.font.name = "Times New Roman"
    r._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    r.font.size = Pt(size)
    if color:
        r.font.color.rgb = RGBColor(*color)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def set_page_number_format(section, fmt="decimal", start=None):
    sect_pr = section._sectPr
    old = sect_pr.find(qn("w:pgNumType"))
    if old is not None:
        sect_pr.remove(old)
    pg = OxmlElement("w:pgNumType")
    pg.set(qn("w:fmt"), fmt)
    if start is not None:
        pg.set(qn("w:start"), str(start))
    sect_pr.append(pg)


def add_page_field(paragraph, roman=False):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    run.font.name = "Times New Roman"
    run.font.size = Pt(10)
    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE \\* roman" if roman else "PAGE")
    run._r.addnext(fld)


def setup_section(section, first_page=False, page_format=None, page_start=None, roman=False):
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(3.5)
    section.right_margin = Cm(1.25)
    section.top_margin = Cm(2.5)
    section.bottom_margin = Cm(1.25)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.7)
    section.different_first_page_header_footer = first_page
    section.footer.is_linked_to_previous = False
    section.first_page_footer.is_linked_to_previous = False
    if page_format:
        set_page_number_format(section, page_format, page_start)
    footer = section.footer.paragraphs[0]
    footer.clear()
    add_page_field(footer, roman=roman)
    if first_page:
        section.first_page_footer.paragraphs[0].clear()


def add_first_page_border(section):
    borders = OxmlElement("w:pgBorders")
    borders.set(qn("w:display"), "firstPage")
    borders.set(qn("w:offsetFrom"), "page")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), "12")
        border.set(qn("w:space"), "24")
        border.set(qn("w:color"), "auto")
        borders.append(border)
    section._sectPr.insert_element_before(borders, "w:pgNumType", "w:cols", "w:docGrid")


def base_style(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    for name in ["Heading 1", "Heading 2", "Heading 3"]:
        style = doc.styles[name]
        style.font.name = "Times New Roman"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
        style.font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Heading 1"].font.size = Pt(13)
    doc.styles["Heading 1"].font.bold = True
    doc.styles["Heading 2"].font.size = Pt(12)
    doc.styles["Heading 2"].font.bold = True
    doc.styles["Heading 3"].font.size = Pt(11)
    doc.styles["Heading 3"].font.bold = True


def add_para(doc, text="", bold_prefix=None, align=None, size=11, after=0, before=0):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.5
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.space_before = Pt(before)
    if align is not None:
        p.alignment = align
    if bold_prefix and text.startswith(bold_prefix):
        r = p.add_run(bold_prefix)
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r = p.add_run(text[len(bold_prefix):])
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
    else:
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.space_before = Pt(10 if level == 1 else 6)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(10)


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(item)
        r.font.name = "Times New Roman"
        r.font.size = Pt(11)


def add_table(doc, headers, rows, widths=None, left_align_cols=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.style = "Table Grid"
    if left_align_cols is None:
        left_align_cols = []

    # Table formatting: repeat header row on page split and prevent row splitting
    header_tr = table.rows[0]._tr.get_or_add_trPr()
    header_tr.append(OxmlElement("w:tblHeader"))
    header_tr.append(OxmlElement("w:cantSplit"))

    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        set_cell_shading(cell, "1F4E78")
        align = WD_ALIGN_PARAGRAPH.LEFT if i in left_align_cols else WD_ALIGN_PARAGRAPH.CENTER
        set_cell_text(cell, h, bold=True, color=(255, 255, 255), align=align)
        for p in cell.paragraphs:
            p.paragraph_format.keep_with_next = True
        if widths:
            cell.width = Cm(widths[i])
    for row in rows:
        r = table.add_row()
        trPr = r._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))
        cells = r.cells
        for i, value in enumerate(row):
            align = WD_ALIGN_PARAGRAPH.LEFT if i in left_align_cols else WD_ALIGN_PARAGRAPH.CENTER
            set_cell_text(cells[i], value, align=align)
            if widths:
                cells[i].width = Cm(widths[i])
    return table


def add_figure(doc, image, caption, width=5.65):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(image), width=Inches(width))
    add_caption(doc, caption)


def page_break(doc):
    doc.add_page_break()


def chapter_section(doc, start=None):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(section, first_page=True, page_format="decimal", page_start=start)
    return section


def numbered_section(doc):
    section = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(section, first_page=False, page_format="decimal")
    return section


def prefatory_heading(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(40)
    p.paragraph_format.space_after = Pt(16)
    r = p.add_run(text)
    r.bold = True
    r.font.name = "Times New Roman"
    r.font.size = Pt(13)
    return p


def generate_model_portfolio_figure(out_file: Path):
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    models = [
        "U-Net\n(Pilot Baseline)",
        "U-Net++\n(Pilot Baseline)",
        "Attention U-Net\n(Pilot Baseline)",
        "ResUNet-2D\n(Primary Baseline)",
        "UNetResNet34\n(Reference Baseline)",
    ]
    params = [7.763428, 9.160068, 7.852160, 8.114604, 24.327524]
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#2E7D32", "#1565C0"]

    bars = ax.bar(models, params, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Trainable Parameters (Millions)", fontsize=11, fontweight="bold")
    ax.set_title("Five-Model Architecture Portfolio & Parameter Complexity", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 28)

    for bar, p in zip(bars, params):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.6,
            f"{p:.2f}M\n({int(round(p * 1e6)):,})",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def generate_entropy_threshold_figure(out_file: Path):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=300)
    intervals = ["[0.00, 0.05]", "(0.05, 0.07]", "(0.07, 0.10]", "(0.10, 0.16]", "> 0.16"]
    slice_counts = [820, 713, 648, 537, 505]
    percentages = [c / 3223 * 100 for c in slice_counts]
    colors = ["#2E7D32", "#66BB6A", "#FFA726", "#EF5350", "#C62828"]

    bars = ax.bar(intervals, slice_counts, color=colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax.set_xlabel("Normalized Predictive Entropy Interval (H)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Axial Slices", fontsize=10, fontweight="bold")
    ax.set_title(
        "Figure 6.5A: Pilot Validation Slices by Rounded Entropy Thresholds\n(Total Population N = 3,223 Validation Slices)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_ylim(0, 950)

    for bar, count, pct in zip(bars, slice_counts, percentages):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 15,
            f"{count:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def generate_entropy_quintile_figure(out_file: Path):
    fig, ax1 = plt.subplots(figsize=(8.5, 4.5), dpi=300)
    quintiles = ["Q1 (Very Low)", "Q2 (Low)", "Q3 (Medium)", "Q4 (High)", "Q5 (Very High)"]
    mean_entropy = [0.0433, 0.0626, 0.0827, 0.1236, 0.2875]
    mean_dice = [0.8217, 0.7697, 0.6520, 0.5712, 0.3008]
    mean_iou = [0.7532, 0.6827, 0.5527, 0.4589, 0.2102]

    x = np.arange(len(quintiles))
    width = 0.32

    rects1 = ax1.bar(
        x - width / 2, mean_dice, width, label="Mean Dice", color="#1565C0", edgecolor="black", linewidth=0.7
    )
    rects2 = ax1.bar(
        x + width / 2, mean_iou, width, label="Mean IoU", color="#42A5F5", edgecolor="black", linewidth=0.7
    )

    ax1.set_xlabel("Equal-Frequency Entropy Quintile (n = 492 foreground slices each)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Segmentation Overlap Score", fontsize=10, fontweight="bold", color="#1565C0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(quintiles, fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(x, mean_entropy, color="#D32F2F", marker="o", linewidth=2.2, markersize=7, label="Mean Entropy (H)")
    ax2.set_ylabel("Mean Predictive Entropy (H)", fontsize=10, fontweight="bold", color="#D32F2F")
    ax2.set_ylim(0, 0.40)

    for rect in rects1:
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold"
        )

    for rect in rects2:
        h = rect.get_height()
        ax1.text(
            rect.get_x() + rect.get_width() / 2.0, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8
        )

    for i, e in enumerate(mean_entropy):
        ax2.text(i, e + 0.012, f"{e:.4f}", ha="center", va="bottom", fontsize=8, color="#D32F2F", fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", ncol=3, framealpha=0.9, fontsize=9)

    ax1.set_title(
        "Figure 6.5B: Segmentation Quality Monotonically Degrades with Uncertainty\n(N = 2,460 Valid Foreground Slices, 5 Quintiles of 492 Slices Each)",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def generate_agent_architecture_diagram(out_file: Path):
    import matplotlib.patches as patches

    fig, ax = plt.subplots(figsize=(13.0, 6.8), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title Banner
    ax.text(50, 96.5, "Figure 7.1: Deterministic Reliability-Aware Agent Architecture & Triage Pipeline",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#1A237E")
    ax.text(50, 93.0, "Four-stage deterministic workflow: tensor integrity, calibration, spatial context, and triage dispatch",
            ha="center", va="center", fontsize=9.5, fontstyle="italic", color="#455A64")

    # ==================== COLUMN 1: INPUT & INTEGRITY ====================
    ax.text(12.5, 87.5, "STAGE 1: INPUT INTEGRITY", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#283593")

    # 1A. Input Box
    box_1a = patches.FancyBboxPatch((3.0, 62.0), 19.0, 22.0, boxstyle="round,pad=0.6",
                                    facecolor="#E8EAF6", edgecolor="#3949AB", linewidth=1.4)
    ax.add_patch(box_1a)
    ax.text(12.5, 80.0, "4-Modality mpMRI Tensor", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1A237E")
    ax.text(12.5, 71.0, "• Ch 0: T1n (native)\n• Ch 1: T1c (contrast)\n• Ch 2: T2f (FLAIR)\n• Ch 3: T2w (T2-wt)\nShape: [1, 4, H, W]",
            ha="center", va="center", fontsize=8.2, color="#283593", linespacing=1.25)

    # Connector 1A -> 1B
    ax.annotate("", xy=(12.5, 55.5), xytext=(12.5, 61.5),
                arrowprops=dict(arrowstyle="->", color="#3949AB", lw=1.6))

    # 1B. Tensor-Integrity Validation Gate
    box_1b = patches.FancyBboxPatch((3.0, 35.5), 19.0, 20.0, boxstyle="round,pad=0.6",
                                    facecolor="#FFF3E0", edgecolor="#E65100", linewidth=1.4)
    ax.add_patch(box_1b)
    ax.text(12.5, 51.5, "Tensor-Integrity Gate", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#E65100")
    ax.text(12.5, 42.5, "• Checks torch.Tensor\n• Asserts ndim == 4\n• Confirms 4 channels\n• Asserts H, W > 0",
            ha="center", va="center", fontsize=8.2, color="#BF360C", linespacing=1.25)

    # Connector 1B -> 1C (Halt)
    ax.annotate("", xy=(12.5, 28.5), xytext=(12.5, 35.0),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.6, ls="--"))
    ax.text(13.0, 31.8, "Invalid\nShape / Type", color="#C62828", fontsize=7.4, fontweight="bold", ha="left", va="center")

    # 1C. Input Rejection Box (Dashed red box, pre-inference exception)
    box_1c = patches.FancyBboxPatch((3.0, 6.5), 19.0, 22.0, boxstyle="round,pad=0.6",
                                    facecolor="#FFEBEE", edgecolor="#C62828", linewidth=1.5, linestyle="--")
    ax.add_patch(box_1c)
    ax.text(12.5, 24.5, "INPUT REJECTION", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#B71C1C")
    ax.text(12.5, 14.5, "• Execution halted\n• Raises TypeError / ValueError\n• Pre-inference exception\n• Separate from agent triage\n(Not an Agent referral)",
            ha="center", va="center", fontsize=7.8, color="#C62828", linespacing=1.25)

    # ==================== COLUMN 2: PRIMARY & CALIBRATION ====================
    # Connector Col 1 -> Col 2
    ax.annotate("", xy=(26.5, 68.0), xytext=(22.5, 45.5),
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.8, connectionstyle="arc3,rad=-0.15"))
    ax.text(24.0, 58.0, "Valid\nTensor", color="#2E7D32", fontsize=7.8, fontweight="bold", ha="center", va="center")

    ax.text(36.5, 87.5, "STAGE 2: PRIMARY & CALIBRATION", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1B5E20")

    # 2A. Primary Model Box
    box_2a = patches.FancyBboxPatch((27.0, 54.0), 19.0, 30.0, boxstyle="round,pad=0.6",
                                    facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.4)
    ax.add_patch(box_2a)
    ax.text(36.5, 79.5, "Primary Segmenter", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1B5E20")
    ax.text(36.5, 66.5, "• ResUNet-2D (Epoch 13)\n• 4 channels in, 4 classes out\n• Val Mean Dice: 0.8670\n• Generates raw voxel logits",
            ha="center", va="center", fontsize=8.2, color="#2E7D32", linespacing=1.25)

    # Connector 2A -> 2B
    ax.annotate("", xy=(36.5, 46.5), xytext=(36.5, 53.5),
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.6))

    # 2B. Calibration & Uncertainty Box
    box_2b = patches.FancyBboxPatch((27.0, 6.5), 19.0, 40.0, boxstyle="round,pad=0.6",
                                    facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.4)
    ax.add_patch(box_2b)
    ax.text(36.5, 42.0, "Calibration & Uncertainty", ha="center", va="center", fontsize=9.6, fontweight="bold", color="#1B5E20")
    ax.text(36.5, 23.5, "• Temperature Scaling:\n  T* = 2.8808 (ResUNet)\n  T* = 2.6267 (UNetResNet)\n• Probability Calibration:\n  ECE: 0.0861 -> 0.0344 (-60%)\n• Foreground Entropy:\n  H(p) normalized in [0, 1]\n• Axial Context Guard:\n  Radius k = +/-2 slices",
            ha="center", va="center", fontsize=8.0, color="#1B5E20", linespacing=1.25)

    # ==================== COLUMN 3: TRIAGE ENGINE ====================
    # Connector Col 2 -> Col 3
    ax.annotate("", xy=(50.5, 45.0), xytext=(46.5, 26.5),
                arrowprops=dict(arrowstyle="->", color="#00695C", lw=1.8, connectionstyle="arc3,rad=-0.12"))
    ax.text(48.5, 36.0, "Calibrated\nOutputs", color="#00695C", fontsize=7.5, fontweight="bold", ha="center", va="center")

    ax.text(61.0, 87.5, "STAGE 3: TRIAGE ENGINE", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#E65100")

    # 3. Deterministic Decision Engine Box
    box_3 = patches.FancyBboxPatch((51.0, 6.5), 20.0, 77.5, boxstyle="round,pad=0.6",
                                   facecolor="#FFF8E1", edgecolor="#FFA000", linewidth=1.5)
    ax.add_patch(box_3)
    ax.text(61.0, 79.5, "Deterministic Rule Engine", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#E65100")
    ax.text(61.0, 75.5, "(src/agent/policy.py)", ha="center", va="center", fontsize=8.0, fontstyle="italic", color="#BF360C")
    ax.text(61.0, 44.5, "1. Empty Mask Check:\n"
                              "   Is primary mask non-empty?\n\n"
                              "2. Uncertainty Check:\n"
                              "   Is foreground H < 0.35?\n\n"
                              "3. Spatial Context Check:\n"
                              "   If empty primary, check\n"
                              "   +/- 2 axial slices for\n"
                              "   neighboring tumor burden\n\n"
                              "4. Fallback Dispatch:\n"
                              "   Query UNetResNet34;\n"
                              "   substitute non-empty\n"
                              "   candidate mask\n\n"
                              "* Candidate switching on\n"
                              "  entropy is disabled in B2",
            ha="center", va="center", fontsize=8.0, color="#BF360C", linespacing=1.2)

    # ==================== COLUMN 4: ACTIONS ====================
    ax.text(86.5, 87.5, "STAGE 4: TRIAGE ACTIONS", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1B5E20")

    # Connectors Col 3 -> Col 4
    # Route to Action 1
    ax.annotate("", xy=(75.5, 71.5), xytext=(71.5, 71.5),
                arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.8))
    ax.text(73.5, 73.8, "H < 0.35 &\nNon-Empty", color="#2E7D32", fontsize=7.4, fontweight="bold", ha="center", va="bottom")

    # Route to Action 2
    ax.annotate("", xy=(75.5, 45.0), xytext=(71.5, 45.0),
                arrowprops=dict(arrowstyle="->", color="#E65100", lw=1.8))
    ax.text(73.5, 47.3, "Empty Primary\n+ Context (k=2)", color="#E65100", fontsize=7.4, fontweight="bold", ha="center", va="bottom")

    # Route to Action 3
    ax.annotate("", xy=(75.5, 18.0), xytext=(71.5, 18.0),
                arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.8))
    ax.text(73.5, 20.3, "H >= 0.35 OR\nUnresolved", color="#C62828", fontsize=7.4, fontweight="bold", ha="center", va="bottom")

    # 4A. Action 1: ACCEPT
    box_4a = patches.FancyBboxPatch((76.0, 59.5), 21.5, 24.5, boxstyle="round,pad=0.6",
                                    facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=1.5)
    ax.add_patch(box_4a)
    ax.text(86.75, 79.5, "ACTION 1: ACCEPT", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#1B5E20")
    ax.text(86.75, 69.5, "• Criteria: H < 0.35 & non-empty\n• Retain ResUNet-2D mask\n• 5,437 slices (83.02%)\n• Autonomous output released",
            ha="center", va="center", fontsize=8.2, color="#2E7D32", linespacing=1.25)

    # 4B. Action 2: FALLBACK SWITCH
    box_4b = patches.FancyBboxPatch((76.0, 33.0), 21.5, 24.5, boxstyle="round,pad=0.6",
                                    facecolor="#FFF8E1", edgecolor="#FFA000", linewidth=1.5)
    ax.add_patch(box_4b)
    ax.text(86.75, 53.0, "ACTION 2: FALLBACK SWITCH", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#E65100")
    ax.text(86.75, 43.0, "• Criteria: Empty + context (k=2)\n• Query UNetResNet34 (T*=2.63)\n• Retain non-empty candidate\n• 54 slices (0.82%) rescued",
            ha="center", va="center", fontsize=8.2, color="#BF360C", linespacing=1.25)

    # 4C. Action 3: FLAG / REFER
    box_4c = patches.FancyBboxPatch((76.0, 6.5), 21.5, 24.5, boxstyle="round,pad=0.6",
                                    facecolor="#FFEBEE", edgecolor="#E53935", linewidth=1.5)
    ax.add_patch(box_4c)
    ax.text(86.75, 26.5, "ACTION 3: FLAG / REFER", ha="center", va="center", fontsize=9.8, fontweight="bold", color="#B71C1C")
    ax.text(86.75, 15.5, "• H >= 0.35 or unresolved empty\n• Autonomous output withheld\n• Simulated referral for review\n• 1,058 slices (16.16% referral)\n[Total Evaluated: 6,549 slices]",
            ha="center", va="center", fontsize=7.8, color="#C62828", linespacing=1.25)

    plt.tight_layout()
    fig.savefig(out_file, dpi=300)
    plt.close(fig)


def prepare_report_artwork(materials_dir: Path, charts_dir: Path, artwork_dir: Path):
    artwork_dir.mkdir(parents=True, exist_ok=True)

    format_template = (
        materials_dir
        / "Final_Minor_Project-2_report_and_PPT_format_&_instructions_Odd2025[2].docx"
    )
    jiit_logo = artwork_dir / "jiit_logo.png"

    if not jiit_logo.exists():
        template = Document(str(format_template))
        logo_rel = next(
            rel for rel in template.part.rels.values() if rel.reltype.endswith("/image")
        )
        jiit_logo.write_bytes(logo_rel.target_part.blob)

    # 1. Figure 4.1: MRI Modalities
    mri_modalities = artwork_dir / "mri_modalities.png"
    with Image.open(materials_dir / "Brain Tumor Segmentation Project Overview.png") as image:
        image.crop((18, 780, 1038, 1075)).save(mri_modalities)

    # 2. Figure 4.2: Five-Model Architecture Portfolio
    generate_model_portfolio_figure(artwork_dir / "model_portfolio.png")

    # 3. Figure 6.5A: Slice Distribution Across Rounded Entropy Thresholds
    generate_entropy_threshold_figure(artwork_dir / "entropy_threshold_distribution.png")

    # 4. Figure 6.5B: Segmentation Quality Across Equal-Frequency Foreground Quintiles
    generate_entropy_quintile_figure(artwork_dir / "entropy_quintile_quality.png")

    # 5. Figure 7.1: Agent Architecture Diagram
    generate_agent_architecture_diagram(artwork_dir / "agent_architecture.png")


def cover(doc, logo_path: Path):
    def cover_line(text, size, bold=False, before=0, after=0):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(before)
        p.paragraph_format.space_after = Pt(after)
        p.paragraph_format.keep_together = True
        r = p.add_run(text)
        r.font.name = "Times New Roman"
        r.font.size = Pt(size)
        r.bold = bold
        return p

    cover_line("Jaypee Institute of Information Technology, Noida", 18, True, after=5)
    cover_line(
        "DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING AND INFORMATION TECHNOLOGY",
        13,
        bold=True,
        after=0,
    )

    logo = doc.add_paragraph()
    logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
    logo.paragraph_format.space_before = Pt(40)
    logo.paragraph_format.space_after = Pt(28)
    logo.add_run().add_picture(str(logo_path), width=Inches(1.65))

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.line_spacing = 1.15
    title.paragraph_format.space_after = Pt(16)
    label = title.add_run("Project Title: ")
    label.bold = True
    label.font.name = "Times New Roman"
    label.font.size = Pt(14)
    value = title.add_run(
        "Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent\n"
        "(Research Edition: Architectural Baselines, Calibration, and Reliability Analysis)"
    )
    value.font.name = "Times New Roman"
    value.font.size = Pt(13)
    value.bold = True

    roster = doc.add_table(rows=4, cols=2)
    roster.alignment = WD_TABLE_ALIGNMENT.CENTER
    roster.autofit = False
    roster.columns[0].width = Cm(4.3)
    roster.columns[1].width = Cm(5.2)
    border_set = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "nil")
        border_set.append(border)
    roster._tbl.tblPr.append(border_set)
    roster_data = [
        ("Enrol. No.", "Name of Student"),
        ("9923103009", "Nishant Kumar"),
        ("9923103011", "Mohammad Tariq"),
        ("9923103012", "Mohammad Kaif"),
    ]
    for row_index, row in enumerate(roster_data):
        for col_index, text in enumerate(row):
            set_cell_text(
                roster.cell(row_index, col_index),
                text,
                bold=(row_index == 0),
                size=12,
            )
            roster.cell(row_index, col_index).width = roster.columns[col_index].width

    for r in roster.rows:
        trPr = r._tr.get_or_add_trPr()
        trPr.append(OxmlElement("w:cantSplit"))

    cover_line("Course Name: Major Project-II", 12, before=18, after=4)
    cover_line("Course Code: 15B19CI791", 12, after=4)
    cover_line("Program: B. Tech. CSE", 12, after=4)
    cover_line("4th Year 8th Semester", 12, after=10)
    cover_line("Under the guidance of", 11, True, after=3)
    cover_line("Dr. Ruchika Bala", 12, True, after=2)
    cover_line("Assistant Professor (Senior Grade)", 11, after=7)
    cover_line("2026 - 2027", 12, True)


def build_preliminary_pages(doc):
    pre = doc.add_section(WD_SECTION.NEW_PAGE)
    setup_section(pre, page_format="lowerRoman", page_start=1, roman=True)

    # 1. Acknowledgement
    prefatory_heading(doc, "ACKNOWLEDGEMENT")
    add_para(
        doc,
        "We place on record our profound gratitude and sincere respect to Dr. Ruchika Bala, Assistant Professor "
        "(Senior Grade), Department of Computer Science & Engineering and Information Technology, Jaypee Institute "
        "of Information Technology, Noida, for her invaluable mentorship, rigorous technical feedback, and continuous "
        "encouragement throughout the course of this major research project. Her guidance in experimental design, "
        "methodological rigor, and scientific writing was instrumental in shaping both the engineering implementation "
        "and academic presentation of this work.",
    )
    add_para(
        doc,
        "We express our sincere appreciation to the Department of Computer Science & Engineering and Information Technology "
        "and the institute administration for providing academic facilities, computational resources, and scholarly "
        "environment supporting our research. Model training and empirical evaluations were conducted on dedicated "
        "local compute infrastructure featuring an NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM). We are also grateful to our faculty members, academic evaluators, and peers for their "
        "constructive critiques and technical suggestions during project reviews and milestone presentations.",
    )
    add_para(
        doc,
        "We gratefully acknowledge the ASNR-MICCAI Brain Tumor Segmentation (BraTS) 2023 Challenge organizers and the "
        "participating medical institutions for curating and making publicly available the multi-parametric MRI dataset "
        "that served as the empirical foundation for this study. We also acknowledge the open-source scientific computing "
        "community behind PyTorch, MONAI, SimpleITK, scikit-learn, and related libraries, whose robust software architectures "
        "enabled reproducible deep learning research. Any remaining oversights or limitations remain entirely our own.",
    )
    add_para(
        doc,
        "Nishant Kumar (9923103009)     Mohammad Tariq (9923103011)     Mohammad Kaif (9923103012)",
        align=WD_ALIGN_PARAGRAPH.CENTER,
        after=0,
        before=30,
    )
    page_break(doc)

    # 2. Declaration
    prefatory_heading(doc, "DECLARATION")
    add_para(
        doc,
        "We hereby declare that this Major Project-II report titled \"Brain Tumor Segmentation Using Predictive Uncertainty "
        "and Towards a Reliability-Aware AI Agent (Research Edition: Architectural Baselines, Calibration, and Reliability Analysis)\" "
        "is an authentic record of our own research carried out under the supervision of Dr. Ruchika Bala, Assistant Professor "
        "(Senior Grade), Department of Computer Science & Engineering and Information Technology, Jaypee Institute of Information "
        "Technology, Noida.",
    )
    add_para(
        doc,
        "To the best of our knowledge and belief, this report contains no material previously published or written by another "
        "person, nor material submitted for the award of any other degree, diploma, or certificate at this or any other university "
        "or institution of higher education, except where due academic citation and acknowledgement are explicitly provided in the text. "
        "All data partitions, model parameter counts, evaluation metrics, calibration statistics, and agent triage outcomes reported "
        "herein reflect verified empirical executions backed by reproducible code and logged audit manifests.",
    )
    add_para(doc, "Place: Noida", before=36)
    add_para(doc, "Date: November 2026")
    add_para(doc, "Nishant Kumar     Enrolment No.: 9923103009", before=24)
    add_para(doc, "Mohammad Tariq    Enrolment No.: 9923103011")
    add_para(doc, "Mohammad Kaif     Enrolment No.: 9923103012")
    page_break(doc)

    # 3. Certificate
    prefatory_heading(doc, "CERTIFICATE")
    add_para(
        doc,
        "This is to certify that the work titled \"Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent "
        "(Research Edition: Architectural Baselines, Calibration, and Reliability Analysis)\" submitted by Nishant Kumar (Enrolment No. 9923103009), "
        "Mohammad Tariq (Enrolment No. 9923103011), and Mohammad Kaif (Enrolment No. 9923103012) of B.Tech of Jaypee Institute of Information "
        "Technology, Noida has been carried out under my supervision. This work has not been submitted partially or wholly to any other "
        "University or Institute for the award of any other degree or diploma.",
    )
    add_para(
        doc,
        "\n\n\n_____________________________________\n"
        "(Signature of Supervisor)\n\n"
        "Name of Supervisor: Dr. Ruchika Bala\n"
        "Designation: Assistant Professor (Senior Grade)\n"
        "Department: Department of Computer Science & Engineering and Information Technology\n"
        "Institution: Jaypee Institute of Information Technology, Noida\n"
        "Date: ________________________",
        before=36,
    )
    page_break(doc)

    # 4. Abstract
    prefatory_heading(doc, "ABSTRACT")
    add_para(
        doc,
        "Accurate and objective delineation of brain tumor subregions from multi-parametric magnetic resonance imaging (mpMRI) "
        "is essential for neuro-oncological diagnosis, surgical resection planning, radiation therapy targeting, and longitudinal "
        "treatment response monitoring. While deep convolutional neural networks produce impressive voxel-wise segmentation "
        "masks, standard segmentation models deployed with deterministic softmax decoders are known to suffer from uncalibrated "
        "overconfidence. Conventional overlap metrics, such as the Dice Similarity Coefficient (Dice), aggregate performance across "
        "thousands of slices but completely conceal localized anatomical failures, catastrophic slice-level mispredictions, and "
        "boundary ambiguity. In safety-critical computer-aided diagnosis applications, an autonomous segmentation tool that silently outputs "
        "erroneous contours poses substantial diagnostic hazards. Consequently, dependable decision-support prototypes demand not merely "
        "high segmentation accuracy, but orthogonal predictive uncertainty estimation, rigorous probability calibration, and safe "
        "selective triage mechanisms that systematically abstain or flag slices for review when model confidence degrades.",
    )
    add_para(
        doc,
        "This research develops, benchmarks, and rigorously audits a comprehensive 2D brain tumor segmentation and reliability-aware "
        "triage framework utilizing the ASNR-MICCAI BraTS 2023 Adult Glioma (GLI) dataset. A complete subject-level dataset census "
        "encompassing 1,251 cases across 1,133 distinct subjects was conducted to establish strict patient-disjoint partitions: a Train "
        "partition of 941 cases (721 actively consumed by 15-epoch models, 220 historical reserve), a Validation pool of 155 cases "
        "divided into a 50-case calibration subset (Val_cal, 47 subjects) and a 105-case internal evaluation subset (Val_eval, 96 subjects), "
        "and a sealed test cohort of 155 cases (143 subjects) preserved under strict read-only quarantine. Five distinct 2D segmentation "
        "architectures were implemented and evaluated across two formal experimental stages: Stage A (a historical 5-epoch pilot benchmark "
        "on 39 validation cases comparing standard U-Net [7.76M params, Mean Dice 0.7711], U-Net++ [9.16M params, Mean Dice 0.7937], "
        "and Attention U-Net [7.85M params, Mean Dice 0.7291]); and Stage B (mature 15-epoch research baselines trained on 721 cases / "
        "59,020 axial slices comparing the Primary Research Baseline ResUNet-2D [8.11M params] and the Reference Research Baseline "
        "UNetResNet34 [24.33M trainable params]). On the 105-case internal evaluation subset, ResUNet-2D achieved a Mean Dice of "
        "0.8641 [95% CI: 0.827, 0.893] (WT: 0.9039, TC: 0.8597, ET: 0.8286; WT median HD95: 4.12 mm) versus UNetResNet34's Mean Dice of "
        "0.8535 [95% CI: 0.814, 0.889] (WT: 0.8843, TC: 0.8423, ET: 0.8339; ET median HD95: 1.73 mm), highlighting a clear trade-off "
        "between global regional overlap and subcompartment boundary precision.",
    )
    add_para(
        doc,
        "To operationalize reliability, voxel-wise predictive entropy was computed and normalized across the four target classes. In pilot "
        "evaluations, slice-level foreground entropy exhibited a strong negative Spearman rank correlation with Dice (rho = -0.6542, "
        "nominal slice-level p < 10^-15; subject-level clustering acknowledged) and an ROC-AUC of 0.8059 for discriminating low-quality segmentations (Dice < 0.50). Furthermore, evaluation of 2,460 "
        "valid foreground slices across equal-frequency quintiles demonstrated monotonic quality degradation: Mean Dice dropped steadily "
        "from 0.8217 in the lowest-entropy quintile (mean H = 0.0433) to 0.3008 in the highest-entropy quintile (mean H = 0.2875). To correct "
        "inherent neural network overconfidence, post-hoc temperature scaling was fitted on the 50 calibration cases via negative log-likelihood "
        "minimization (T* = 2.8808 for ResUNet-2D; T* = 2.6267 for UNetResNet34). Temperature scaling reduced Expected Calibration Error (ECE) "
        "from 0.0502 to 0.0084 (-83.3%) on calibration data, and achieved substantial ECE reductions when evaluated on the internal 105-case evaluation pool (Val_eval), reducing ECE from "
        "0.0861 to 0.0344 (-60.0%) for ResUNet-2D and from 0.1098 to 0.0496 (-54.8%) for UNetResNet34. A deterministic reliability-aware "
        "triage agent was implemented with pre-inference input tensor-integrity validation, temperature-calibrated entropy estimation, empty-prediction guards, "
        "and dual-model fallback switching. Evaluated on 6,549 foreground evaluation slices at an operating threshold of tau = 0.35, the agent retained "
        "83.84% of slices (5,491 slices total: 5,437 directly accepted from ResUNet-2D and 54 retained via fallback switch to UNetResNet34), "
        "flagged 16.16% (1,058 slices) for simulated referral, yielding an aggregate Dice of 0.8640 compared with 0.8641 for standalone ResUNet-2D. "
        "This confirms that the primary triage utility of the reliability agent lies in selective triage and risk containment under the tested operating policy rather than raw overlap enhancement.",
    )
    add_para(
        doc,
        "A controlled synthetic channel-ablation stress test evaluated across all five missing-modality conditions (C0 to C4 via post-normalization zero-clamping) "
        "exposed a critical failure mode: under missing contrast-enhanced T1c (C2), enhancing tumor Dice collapsed precipitously to 0.1684 and tumor core Dice to 0.3132, "
        "while whole tumor Dice remained 0.8532 due to intact edema signal on remaining T2 sequences. Under the tested synthetic post-normalization T1c zero-clamping condition, "
        "severe ET and TC degradation was not accompanied by a statistically significant increase in aggregate referral (pooled referral shifted only to 17.44%, p_adj = 1.0000). "
        "This demonstrates a specific limitation of the tested entropy-based policy and motivates explicit modality-presence and identity validation upstream. Conversely, under missing T2w (C3), the empty-prediction guard intercepted 2,621 empty masks, "
        "surging pooled referral to 62.65% (4,103 / 6,549 slices, p_adj = 7.12 x 10^-17). Finally, methodological limitations are transparently disclosed: the 105-case evaluation "
        "pool was included in the 155-case validation pool used during checkpoint selection, inducing checkpoint-selection bias; consequently, reported metrics reflect internal "
        "model validation rather than independent clinical testing. The sealed test cohort remains intact and uninspected for future blinded validation.",
    )
    add_para(
        doc,
        "Keywords: Brain Tumor Segmentation; Multi-parametric MRI; Deep Learning; ResUNet-2D; UNetResNet34; Predictive Uncertainty; "
        "Shannon Entropy; Probability Calibration; Expected Calibration Error; Temperature Scaling; Reliability-Aware AI Agent; BraTS 2023.",
        bold_prefix="Keywords: ",
        before=12,
        after=0,
    )
    page_break(doc)

    # 5. Table of Contents
    prefatory_heading(doc, "TABLE OF CONTENTS")
    toc_rows = [
        ("Acknowledgement", "i"),
        ("Declaration", "ii"),
        ("Certificate", "iii"),
        ("Abstract", "iv"),
        ("Table of Contents", "vi"),
        ("List of Tables", "vii"),
        ("List of Figures", "viii"),
        ("List of Abbreviations", "ix"),
        ("Chapter 1: INTRODUCTION", "1"),
        ("    1.1 Clinical and Technical Context", "1"),
        ("    1.2 Limitations of Segmentation Dice as a Sole Reliability Measure", "1"),
        ("    1.3 Predictive Uncertainty and Reliability-Aware Decision-Making", "2"),
        ("    1.4 Research Motivation, Objectives, and Scope", "2"),
        ("    1.5 Summary of Project Contributions", "3"),
        ("    1.6 Report Organization", "4"),
        ("Chapter 2: BACKGROUND STUDY AND RELATED WORK", "5"),
        ("    2.1 Multi-modal MRI and Glioma Subregions", "5"),
        ("    2.2 Deep Learning Architectures for Medical Image Segmentation", "5"),
        ("    2.3 Evaluation, Overlap, and Distance Metrics", "6"),
        ("    2.4 Predictive Uncertainty Quantification in Medical Imaging", "6"),
        ("    2.5 Probability Calibration in Neural Networks", "7"),
        ("    2.6 Selective Prediction, Triage, and Reliability-Aware AI", "7"),
        ("Chapter 3: DATASET CENSUS AND PARTITIONING METHODOLOGY", "8"),
        ("    3.1 BraTS 2023 GLI Dataset Overview", "8"),
        ("    3.2 Subject-Level Dataset Census and Split Accounting", "8"),
        ("    3.3 Historical 197-Case Pilot Split vs Mature Cohort", "9"),
        ("    3.4 Data Preprocessing and Tensor Normalization", "9"),
        ("Chapter 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE", "11"),
        ("    4.1 Five-Model Architecture Portfolio", "11"),
        ("    4.2 Structural Details of Stage A Architectures", "12"),
        ("    4.3 Structural Details of Stage B Architectures", "13"),
        ("    4.4 Training Dynamics and Loss Optimization Framework", "13"),
        ("    4.5 Software Architecture and Repository Design", "14"),
        ("Chapter 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS", "15"),
        ("    5.1 Two-Stage Experimental Progression", "15"),
        ("    5.2 Stage A: Historical Five-Epoch Pilot Benchmark", "15"),
        ("    5.3 Stage B: Mature 15-Epoch Research Baselines", "17"),
        ("    5.4 Checkpoint-Selection Bias and Methodological Boundaries", "18"),
        ("Chapter 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE", "19"),
        ("    6.1 Predictive Uncertainty and Segmentation Quality Correlation", "19"),
        ("    6.2 Separation of Uncertainty Binning and Slice Distribution", "20"),
        ("    6.3 Post-Hoc Probability Calibration and Temperature Scaling", "21"),
        ("    6.4 Implemented Deterministic Reliability-Aware Agent", "22"),
        ("Chapter 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS", "23"),
        ("    7.1 Missing-Modality Robustness Stress Study", "23"),
        ("    7.2 Methodological Boundaries and Comprehensive Limitations", "24"),
        ("    7.3 Exploratory 3D / Foundation Model Pilot Studies", "25"),
        ("    7.4 Future Research Directions and Conclusion", "25"),
        ("References", "27"),
        ("Appendix A: Full Dataset Split Manifest and Census Audit", "28"),
        ("Appendix B: Reproducibility Record and Hyperparameter Specifications", "28"),
        ("Appendix C: Research Governance, Verification Checklist, and Integrity Declaration", "29"),
    ]
    add_table(doc, ["Contents", "Page No."], toc_rows, [14.5, 2.8], left_align_cols=[0])
    page_break(doc)

    # 6. List of Tables
    prefatory_heading(doc, "LIST OF TABLES")
    table_list_rows = [
        ("Table 2.1", "Evaluation, overlap, distance, and uncertainty measures used in this report", "6"),
        ("Table 3.1", "Full subject-level dataset census and partition allocation for BraTS 2023 GLI", "9"),
        ("Table 3.2", "Historical 197-case pilot development split vs mature 1,251-case cohort partitioning", "9"),
        ("Table 4.1", "Five-model 2D segmentation architecture portfolio and verified runtime parameter counts", "11"),
        ("Table 4.2", "Core implementation modules and functional responsibilities across the codebase", "14"),
        ("Table 5.1", "Stage A historical five-epoch pilot benchmark on the 39-case validation cohort", "15"),
        ("Table 5.2", "Stage A foreground class-wise Dice at model-specific best validation epochs", "15"),
        ("Table 5.3", "Stage B mature 15-epoch baseline comparison on the 105-case internal evaluation pool", "18"),
        ("Table 6.1", "Historical pilot uncertainty and segmentation quality correlation analysis", "19"),
        ("Table 6.2", "Post-hoc temperature scaling and ECE reduction across calibration and evaluation cohorts", "21"),
        ("Table 6.3", "Reconciled triage and referral action distribution of the deterministic reliability-aware agent", "22"),
        ("Table 7.1", "Authoritative Analysis B2 missing-modality stress test across all five input conditions on 105 evaluation cases", "23"),
        ("Table 7.2", "Qualitative comparison of deep learning brain tumor segmentation paradigms", "25"),
        ("Table A.1", "Subject-level dataset split manifest summary", "28"),
        ("Table B.1", "Reproducibility checklist and hyperparameter specifications", "28"),
        ("Table C.1", "Research governance, claim boundary, and methodological integrity checklist", "29"),
    ]
    add_table(doc, ["Table", "Title", "Page"], table_list_rows, [2.5, 12.3, 2.5], left_align_cols=[0, 1])
    page_break(doc)

    # 7. List of Figures
    prefatory_heading(doc, "LIST OF FIGURES")
    figure_list_rows = [
        ("Figure 4.1", "Multi-modal MRI input channels and segmentation subregions with modality mapping", "11"),
        ("Figure 4.2", "Five-model 2D segmentation architecture portfolio and parameter complexity", "12"),
        ("Figure 4.3", "Representative qualitative segmentation output from historical 5-epoch U-Net pilot", "12"),
        ("Figure 6.1", "Best validation Dice and IoU scores across the historical three-model pilot", "16"),
        ("Figure 6.2", "Foreground Dice comparison by encoded tumor label at model-specific best epochs", "16"),
        ("Figure 6.3", "Training and validation loss recorded at each epoch for historical pilot models", "17"),
        ("Figure 6.4", "Validation mean foreground Dice across the five pilot training epochs", "17"),
        ("Figure 6.5A", "Pilot validation slice distribution across rounded predictive entropy thresholds (N = 3,223)", "20"),
        ("Figure 6.5B", "Segmentation quality across equal-frequency predictive entropy quintiles (N = 2,460)", "20"),
        ("Figure 6.6", "Complete foreground miss rate among validation slices containing tumor pixels", "21"),
        ("Figure 7.1", "Architecture of the implemented deterministic reliability-aware AI agent and triage pipeline", "26"),
    ]
    add_table(doc, ["Figure", "Title", "Page"], figure_list_rows, [2.5, 12.3, 2.5], left_align_cols=[0, 1])
    page_break(doc)

    # 8. List of Abbreviations
    prefatory_heading(doc, "LIST OF ABBREVIATIONS")
    abbrev_rows = [
        ("AI", "Artificial Intelligence"),
        ("AUC", "Area Under the Receiver Operating Characteristic Curve"),
        ("BraTS", "Brain Tumor Segmentation Challenge"),
        ("CE", "Cross-Entropy"),
        ("CI", "Confidence Interval"),
        ("CNN", "Convolutional Neural Network"),
        ("Dice", "Dice Similarity Coefficient"),
        ("ECE", "Expected Calibration Error"),
        ("ED", "Peritumoral Edematous / Infiltrated Tissue (Label 2)"),
        ("ET", "Enhancing Tumor (Label 3)"),
        ("FLAIR", "Fluid-Attenuated Inversion Recovery"),
        ("GPU", "Graphics Processing Unit"),
        ("HD95", "95th Percentile Hausdorff Distance"),
        ("IoU", "Intersection over Union (Jaccard Index)"),
        ("LLM", "Large Language Model"),
        ("MC", "Monte Carlo"),
        ("MONAI", "Medical Open Network for Artificial Intelligence"),
        ("mpMRI", "Multi-parametric Magnetic Resonance Imaging"),
        ("MRI", "Magnetic Resonance Imaging"),
        ("NCR", "Necrotic Core and Non-Enhancing Tumor (Label 1)"),
        ("NIfTI", "Neuroimaging Informatics Technology Initiative"),
        ("NLL", "Negative Log-Likelihood"),
        ("PACS", "Picture Archiving and Communication System"),
        ("ReLU", "Rectified Linear Unit"),
        ("ResNet", "Residual Neural Network"),
        ("ROC", "Receiver Operating Characteristic"),
        ("TC", "Tumor Core (NCR + ET)"),
        ("VRAM", "Video Random Access Memory"),
        ("WT", "Whole Tumor (NCR + ED + ET)"),
    ]
    add_table(doc, ["Abbreviation", "Meaning"], abbrev_rows, [4.0, 13.3], left_align_cols=[0, 1])


def build_chapter1_introduction(doc):
    chapter_section(doc, start=1)
    add_heading(doc, "CHAPTER 1: INTRODUCTION", 1)

    add_heading(doc, "1.1 Clinical and Technical Context", 2)
    add_para(
        doc,
        "Gliomas represent the most prevalent and biologically aggressive primary intrinsic neoplasms of the central "
        "nervous system, accounting for the vast majority of primary malignant brain tumors and conferring significant patient "
        "morbidity and mortality. In clinical neuro-oncology, high-grade gliomas (such as glioblastoma multiforme, WHO Grade 4) "
        "exhibit profound intratumoral heterogeneity, rapid cellular proliferation, microvascular proliferation, and widespread "
        "diffuse infiltration into surrounding functional brain parenchyma. Accurate anatomical delineation of the tumor\'s "
        "constituent histological subcompartments—specifically the necrotic non-enhancing core, the contrast-enhancing active "
        "tumor rim, and the peritumoral vasogenic edema/infiltrated brain tissue—is indispensable across the clinical care pathway. "
        "It directly governs neurosurgical resection planning, target volume contouring in conformal radiation therapy, and "
        "quantitative volumetric assessment of treatment response or disease progression.",
    )
    add_para(
        doc,
        "To capture these diverse physiological and morphological characteristics, clinical protocol relies on multi-parametric "
        "magnetic resonance imaging (mpMRI). Standard diagnostic protocols routinely acquire four complementary structural pulse "
        "sequences: native T1-weighted (T1n), gadolinium contrast-enhanced T1-weighted (T1c), T2-weighted (T2w), and T2 Fluid-Attenuated "
        "Inversion Recovery (T2f). Each sequence provides distinct biophysical contrast: T1c highlights regions of active blood-brain "
        "barrier breakdown, T1n establishes native baseline tissue contrast, T2w delineates fluid-rich areas and cellular margins, "
        "and T2f selectively suppresses bulk cerebrospinal fluid signal to emphasize hyperintense peritumoral edema. Despite the "
        "rich diagnostic information provided by mpMRI, manual slice-by-slice voxel contouring by expert raters is "
        "exceedingly labor-intensive and subject to substantial inter-observer and intra-observer variability across tumor subregions [1]. "
        "Automated, reproducible, and objective deep learning segmentation pipelines are therefore essential to support "
        "neuro-radiological image analysis workflows [1], [2].",
    )

    add_heading(doc, "1.2 Limitations of Segmentation Dice as a Sole Reliability Measure", 2)
    add_para(
        doc,
        "Over the past decade, deep learning research in medical image analysis has predominantly treated brain tumor segmentation as "
        "a purely competitive benchmark driven by overlap metrics—most notably the Dice Similarity Coefficient (Dice) and Intersection "
        "over Union (IoU). While a high global Dice score (e.g., Dice > 0.85) indicates strong spatial coincidence between predicted and "
        "ground-truth masks in aggregate, it is profoundly inadequate as a sole measure of clinical reliability and algorithmic safety. "
        "Global Dice is fundamentally an aggregate volumetric statistic that conceals catastrophic localized failures: it fails to signal "
        "when a model predicts plausible-looking contours in anatomically impossible locations, completely misses small but clinically "
        "critical enhancing tumor satellites, or produces severely degraded segmentations on atypical scans.",
    )
    add_para(
        doc,
        "Crucially, standard deep convolutional neural networks with deterministic softmax classification decoders are notorious for "
        "producing uncalibrated, overconfident predictions [6]. A standard model outputs a probability vector that sums to one across "
        "classes, but the maximum softmax probability frequently approaches 0.99 even when the underlying image features are corrupt, "
        "ambiguous, or completely out of distribution. In a safety-critical medical deployment, a silent algorithmic failure—where a "
        "deep learning model outputs an erroneous contour with apparent high confidence—poses severe clinical hazards. If a neurosurgeon "
        "or radiation oncologist relies on an unverified false contour, healthy eloquent brain tissue may be resected or critical tumor "
        "margins undertreated. Consequently, a clinically dependable AI tool cannot simply output a deterministic segmentation mask; "
        "it must provide an orthogonal, calibrated confidence signal that communicates when its predictions are uncertain, questionable, "
        "or untrustworthy.",
    )

    add_heading(doc, "1.3 Predictive Uncertainty and Reliability-Aware Decision-Making", 2)
    add_para(
        doc,
        "To transcend the blind spots of uncalibrated deep learning, this research investigates predictive uncertainty estimation as "
        "a foundation for reliability-aware medical AI. Information-theoretic Shannon entropy, computed over the predicted softmax "
        "class probability distribution and normalized across the target classes, provides an intuitive, mathematically grounded "
        "measure of confidence concentration [9]. When a model is certain, its predicted probability distribution is sharply peaked on a "
        "single anatomical class, resulting in near-zero entropy. Conversely, when tissue boundaries are ambiguous, contrast is degraded, "
        "or input features deviate from the training manifold, the distribution disperses across multiple classes, driving entropy toward "
        "its theoretical maximum.",
    )
    add_para(
        doc,
        "Predictive uncertainty enables the operationalization of the selective prediction (or abstention) paradigm in medical AI [13]. "
        "Under this paradigm, an intelligent clinical support agent does not force an autonomous decision on every case. Instead, it "
        "operates within a structured triage framework: predictions with low, calibrated uncertainty (H < 0.35) and non-empty foreground are "
        "autonomously accepted; empty primary predictions with confirmed spatial axial-context support (radius k = 2) trigger an automated "
        "fallback switch to candidate UNetResNet34; and predictions with elevated uncertainty (H >= 0.35) or unresolved empty masks are "
        "systematically abstained from autonomous execution and flagged for simulated referral for review (abstention). Pre-inference input "
        "tensor-integrity validation halts execution via exceptions if input tensors are malformed. This observe-decide-act triage "
        "architecture bridges the gap between raw algorithmic benchmark performance and safe, human-in-the-loop clinical integration.",
    )

    add_heading(doc, "1.4 Research Motivation, Objectives, and Scope", 2)
    add_para(
        doc,
        "The overarching motivation of this project is to develop, rigorously evaluate, and scientifically audit a multi-stage deep "
        "learning framework that combines high-performance multi-modal brain tumor segmentation with calibrated predictive uncertainty "
        "and deterministic reliability-aware triage. The investigation is structured around six specific research objectives:",
    )
    add_bullets(
        doc,
        [
            "Architectural Benchmarking: Implement, train, and benchmark five distinct 2D convolutional neural network architectures—standard "
            "U-Net, U-Net++, Attention U-Net, ResUNet-2D, and UNetResNet34—under controlled, reproducible training protocols.",
            "Rigorous Dataset Census and Auditing: Conduct a comprehensive subject-level census of the BraTS 2023 Adult Glioma dataset "
            "(1,251 cases across 1,133 distinct subjects) and enforce strict patient-disjoint partitioning to prevent data leakage.",
            "Probability Calibration: Evaluate neural network miscalibration and implement post-hoc temperature scaling on a dedicated "
            "calibration cohort to minimize Expected Calibration Error (ECE) across foreground tumor classes.",
            "Uncertainty Quantification and Quality Correlation: Analyze the empirical relationship between normalized predictive entropy "
            "and true segmentation overlap (Dice/IoU), establishing whether uncertainty reliably tracks empirical degradation.",
            "Missing-Modality Stress Testing: Systematically evaluate model robustness and uncertainty responsiveness under missing-channel "
            "stress conditions across all four MRI inputs (ablating T1n, T1c, T2w, and T2f), discovering failure modes and uncertainty blind spots.",
            "Deterministic Reliability Agent Implementation: Design, implement, and benchmark an automated reliability agent executing "
            "pre-inference input tensor-integrity validation, calibrated entropy triage, axial-context empty-prediction guards, and multi-model fallback switching.",
        ],
    )
    add_para(
        doc,
        "Scope and Boundaries: The empirical scope of this report is strictly confined to 2D axial slice segmentation using the BraTS 2023 "
        "Adult Glioma development cohort. All models, calibration routines, and agent logic are developed and evaluated as an academic "
        "research prototype. This system is not cleared as a medical device, has not undergone prospective clinical trials, contains no "
        "PACS/DICOM clinical integration, and must not be utilized for clinical diagnosis, treatment planning, or direct patient care.",
    )

    add_heading(doc, "1.5 Summary of Project Contributions", 2)
    add_para(
        doc,
        "This project delivers four major technical and scientific contributions to the medical imaging literature:",
    )
    add_bullets(
        doc,
        [
            "Two-Stage Architectural Progression: We present both a historical 5-epoch pilot benchmark across three models (U-Net, U-Net++, "
            "Attention U-Net) and mature 15-epoch research baselines (ResUNet-2D as Primary Research Baseline, UNetResNet34 as Reference "
            "Research Baseline) trained on 721 cases / 59,020 axial slices.",
            "Rigorous Subject-Level Dataset Accounting: We establish an authoritative subject-level split manifest across 1,251 cases / "
            "1,133 subjects, completely isolating a 155-case validation pool (50 calibration cases, 105 evaluation cases) and preserving "
            "a 155-case sealed test cohort under strict read-only quarantine.",
            "Empirical Calibration and Quality Triage: We demonstrate that post-hoc temperature scaling reduces Expected Calibration Error "
            "by 60.0% on evaluation data (0.0861 to 0.0344), and show that equal-frequency entropy quintiles exhibit monotonic quality "
            "degradation (Dice 0.8217 down to 0.3008).",
            "Failure Mode Discovery and Policy Limitations: Demonstrates a specific limitation of the tested entropy-based policy under synthetic "
            "post-normalization T1c zero-clamping, where severe enhancing tumor and tumor core degradation (ET Dice collapses to 0.1684) was not "
            "accompanied by a statistically significant increase in aggregate referral (p_adj = 1.0000), motivating explicit upstream modality-presence "
            "and sequence-identity validation.",
        ],
    )

    page_break(doc)
    add_heading(doc, "1.6 Report Organization", 2)
    add_para(
        doc,
        "The remainder of this report is organized as follows: Chapter 2 reviews background literature on multi-modal MRI, deep segmentation "
        "architectures, calibration, and selective prediction. Chapter 3 details the BraTS 2023 dataset census, subject-level split accounting, "
        "and preprocessing protocols. Chapter 4 presents the five-model portfolio, structural configurations, loss formulations, and codebase "
        "architecture. Chapter 5 details comparative experimental results across Stage A (pilot) and Stage B (mature baselines), openly disclosing "
        "checkpoint-selection bias. Chapter 6 examines predictive uncertainty, probability calibration, and the deterministic reliability agent. "
        "Chapter 7 presents missing-modality stress tests, comprehensive limitations, future scope, and conclusions. Formal references and "
        "audit appendices complete the document.",
    )


def build_chapter2_background(doc):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 2: BACKGROUND STUDY AND RELATED WORK", 1)

    add_heading(doc, "2.1 Multi-modal MRI and Glioma Subregions", 2)
    add_para(
        doc,
        "Multi-parametric magnetic resonance imaging (mpMRI) is the gold standard diagnostic imaging modality for brain tumor evaluation. "
        "Because different tissue components exhibit distinct magnetic relaxation properties, standard clinical protocols acquire multiple "
        "complementary pulse sequences to comprehensively characterize the tumor microenvironment [1], [2]. The BraTS benchmark "
        "standardizes four primary structural sequences: native T1-weighted (T1n), post-gadolinium contrast-enhanced T1-weighted (T1c), "
        "T2-weighted (T2w), and T2 Fluid-Attenuated Inversion Recovery (T2f).",
    )
    add_para(
        doc,
        "In diffuse gliomas, the tumor lesion is structurally and biologically heterogeneous, comprising three distinct pathologically "
        "meaningful subregions that must be delineated: (1) Enhancing Tumor (ET, encoded as Label 3), representing actively dividing "
        "hypervascular malignant cells with disrupted blood-brain barrier permeability, appearing bright on T1c; (2) Necrotic Core and "
        "Non-Enhancing Tumor (NCR, encoded as Label 1), representing hypovascular or avascular necrotic tissue and non-enhancing infiltrative "
        "components, appearing hypointense on T1c and hyperintense on T2w; and (3) Peritumoral Edematous / Infiltrated Tissue (ED, encoded "
        "as Label 2), representing vasogenic edema combined with microscopically infiltrating glioma cells, appearing prominently hyperintense "
        "on T2f. Under the standardized BraTS hierarchical evaluation protocol, these encoded labels are grouped into clinically actionable "
        "composite structures: Whole Tumor (WT = NCR + ED + ET), Tumor Core (TC = NCR + ET), and Enhancing Tumor (ET alone).",
    )

    add_heading(doc, "2.2 Deep Learning Architectures for Medical Image Segmentation", 2)
    add_para(
        doc,
        "Convolutional neural networks have revolutionized biomedical image segmentation, establishing the standard paradigm of an "
        "encoder-decoder network with skip connections. The foundational architecture in this domain is the U-Net, introduced by Ronneberger "
        "et al. [3]. U-Net consists of a contracting encoder path that extracts high-level semantic context through successive convolutions "
        "and spatial downsampling, paired with a symmetric expansive decoder path that restores high-resolution spatial localization through "
        "transposed convolutions or upsampling. Critical to U-Net\'s success are lateral skip connections, which directly transfer high-resolution "
        "feature maps from the encoder to the decoder, preserving fine anatomical boundary information that would otherwise be lost during pooling.",
    )
    add_para(
        doc,
        "Subsequent architectural innovations have sought to optimize feature fusion and attention. Zhou et al. [4] developed U-Net++, "
        "which replaces standard linear skip connections with nested, dense convolutional blocks that progressively bridge the semantic "
        "gap between encoder and decoder features. Oktay et al. [5] proposed Attention U-Net, introducing additive soft attention gates "
        "within the skip connections that automatically learn to suppress activations in irrelevant background regions while highlighting "
        "salient tumor boundaries. To enable deeper feature representations without suffering from vanishing gradients, He et al. [8] "
        "introduced deep residual learning, utilizing identity shortcut mappings. This concept was combined with encoder-decoder segmentation by "
        "Zhang et al. [11], who proposed the Residual U-Net (ResUNet) architecture by embedding residual blocks into the U-Net contracting and "
        "expanding paths. Originally introduced for remote sensing feature delineation, the residual U-Net framework was subsequently widely adopted "
        "across biomedical image segmentation to improve gradient propagation and boundary delineation. Furthermore, transfer learning using "
        "encoder backbones pretrained on natural images (e.g., ResNet-34) has demonstrated superior feature extraction and boundary fidelity "
        "in medical imaging domains [12].",
    )

    add_heading(doc, "2.3 Evaluation, Overlap, and Distance Metrics", 2)
    add_para(
        doc,
        "Evaluating medical image segmentation requires measuring both volumetric overlap and boundary surface fidelity. Table 2.1 "
        "summarizes the primary overlap, distance, and uncertainty measures utilized throughout this research.",
    )

    headers_2_1 = ["Measure", "Mathematical Definition", "Operational Interpretation in Study"]
    rows_2_1 = [
        ("Dice Similarity Coefficient (Dice)", "2 |P \u2229 G| / (|P| + |G|)", "Primary volumetric overlap measure; harmonic mean of precision and recall (foreground classes)."),
        ("Intersection over Union (IoU)", "|P \u2229 G| / |P \u222a G|", "Jaccard overlap coefficient; penalizes over- and under-segmentation more severely than Dice."),
        ("95th Percentile Hausdorff (HD95)", "95th percentile d(P, G)", "Surface distance metric measuring boundary fidelity in millimeters (mm); robust to isolated outliers."),
        ("Normalized Predictive Entropy (H)", "- (1 / log C) \u2211 p_c log p_c", "Voxel- and slice-level predictive uncertainty derived from softmax probabilities across C=4 classes."),
        ("Expected Calibration Error (ECE)", "\u2211 (B_m / N) |acc(B_m) - conf(B_m)|", "Weighted difference between model confidence and empirical accuracy across M=15 probability bins."),
    ]
    add_table(doc, headers_2_1, rows_2_1, [3.2, 5.8, 7.5], left_align_cols=[0, 1, 2])
    add_caption(doc, "Table 2.1: Evaluation, overlap, distance, and uncertainty measures used in this report.")

    add_para(
        doc,
        "During model training, regional overlap and pixel-wise classification are optimized simultaneously using a combined compound "
        "loss function: L_total = L_Dice + L_CE, where L_Dice promotes continuous spatial overlap and L_CE enforces penalization of class "
        "prediction discrepancies across all voxels.",
        before=6,
    )

    add_heading(doc, "2.4 Predictive Uncertainty Quantification in Medical Imaging", 2)
    add_para(
        doc,
        "Uncertainty in deep neural networks is broadly categorized into two fundamental forms: epistemic (model or parameter) uncertainty "
        "and aleatoric (data or measurement) uncertainty [10]. Epistemic uncertainty arises from limitations in the training data, model "
        "capacity, or out-of-distribution feature spaces; in principle, it can be mitigated with additional training data. Aleatoric uncertainty "
        "reflects intrinsic sensor noise, image motion artifacts, partial volume averaging, and biological ambiguity at fuzzy tissue boundaries; "
        "it cannot be eliminated by increasing training volume.",
    )
    add_para(
        doc,
        "In deep learning segmentation pipelines requiring high computational throughput, single-forward-pass softmax entropy provides an "
        "efficient and mathematically grounded measure of predictive uncertainty. Derived from Shannon\'s information theory (Shannon, "
        "1948) [9], predictive entropy quantifies the dispersion of the predicted probability vector across the class spectrum. We normalize "
        "the raw entropy by dividing by log(C) (where C = 4), constraining H \u2208 [0, 1]. In literature, Monte Carlo Dropout (Gal & "
        "Ghahramani, 2016) [7] is frequently cited as a Bayesian approximation for estimating epistemic uncertainty by sampling multiple "
        "stochastic passes at test time. We clarify that while MC Dropout represents an important theoretical foundation, our primary "
        "pipeline evaluates single-forward-pass deterministic softmax entropy to achieve practical clinical execution speeds without "
        "requiring multi-pass inference overhead.",
    )

    add_heading(doc, "2.5 Probability Calibration in Neural Networks", 2)
    add_para(
        doc,
        "A model is defined as perfectly calibrated when its predicted class probability reflects its true empirical likelihood of "
        "correctness: P(Y = y | P_hat = p) = p for all p \u2208 [0, 1]. While classical machine learning models (such as logistic regression) "
        "are often well-calibrated, modern deep neural networks with batch normalization, residual connections, and high parameter "
        "counts suffer from pervasive miscalibration and overconfidence, as systematically documented by Guo et al. (2017) [6].",
    )
    add_para(
        doc,
        "To measure miscalibration, the Expected Calibration Error (ECE) is computed by partitioning predictions into M equal-width "
        "confidence bins (typically M = 15) and calculating the weighted absolute difference between bin accuracy and average bin confidence. "
        "To remediate miscalibration without altering model parameters or degrading segmentation accuracy, post-hoc Temperature Scaling "
        "is applied. Temperature scaling introduces a single learnable scalar parameter T > 0 that rescales logit activations prior to "
        "the softmax function: p_hat_i = exp(z_i / T) / \u2211_j exp(z_j / T). When T > 1, the softmax distribution is softened, "
        "counteracting overconfidence. The optimal temperature T* is fitted strictly on an independent calibration cohort by minimizing "
        "negative log-likelihood (NLL) via gradient descent.",
    )

    add_heading(doc, "2.6 Selective Prediction, Triage, and Reliability-Aware AI", 2)
    add_para(
        doc,
        "Selective classification, formalized by Geifman & El-Yaniv (2017) [13], defines a decision framework wherein a machine learning model "
        "is equipped with a selection function g(x) \u2208 {0, 1} that determines whether to output an autonomous prediction (g(x) = 1) or "
        "abstain (g(x) = 0). This framework establishes a controllable trade-off between coverage (the proportion of retained samples) and "
        "risk (the error rate on retained samples).",
    )
    add_para(
        doc,
        "In this study, we operationalize selective classification into a multi-tier deterministic triage pipeline. When the primary "
        "segmentation model (ResUNet-2D) yields a calibrated foreground entropy below an operational threshold (tau = 0.35) on a non-empty "
        "prediction (H < 0.35), the segmentation is accepted autonomously (ACCEPT). When the primary model predicts an empty foreground mask, "
        "the agent inspects adjacent axial slices (radius k = 2) and, upon confirming neighboring foreground support, executes an autonomous "
        "fallback switch to a secondary architecture (UNetResNet34). If predictive entropy is elevated (H >= 0.35) or an empty prediction cannot "
        "be resolved with axial support, the agent abstains and generates a simulated referral flag for review. In the canonical B2 configuration, "
        "uncertainty-based candidate switching is disabled. This structured triage establishes a verifiable mechanism for selective prediction, "
        "accepting high-confidence predictions while routing uncertain or aberrant cases for simulated review.",
    )


def build_chapter3_dataset_methodology(doc):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 3: DATASET CENSUS AND PARTITIONING METHODOLOGY", 1)

    add_heading(doc, "3.1 BraTS 2023 GLI Dataset Overview", 2)
    add_para(
        doc,
        "The empirical investigations conducted in this research utilize the multi-institutional Brain Tumor Segmentation "
        "(BraTS) 2023 Adult Glioma (GLI) Challenge dataset [1], [2]. The BraTS benchmark represents the international gold standard "
        "for evaluating computer-assisted glioma segmentation algorithms, comprising retrospectively collected pre-operative "
        "multi-parametric MRI scans acquired across multiple clinical centers with diverse 1.5T and 3.0T MRI scanners and varying "
        "acquisition parameters.",
    )
    add_para(
        doc,
        "All imaging volumes provided in the challenge repository were rigidly co-registered to the standardized SRI24 anatomical "
        "brain atlas, resampled to an isotropic voxel resolution of 1.0 x 1.0 x 1.0 mm^3, and skull-stripped to remove non-brain tissues, "
        "yielding uniform volumetric matrices of dimension 240 x 240 x 155 voxels. Each patient examination comprises four co-registered "
        "structural sequences: native T1-weighted (T1n), post-gadolinium contrast-enhanced T1-weighted (T1c), T2-weighted (T2w), and "
        "T2 Fluid-Attenuated Inversion Recovery (T2f). Ground-truth segmentation masks were delineated manually by trained neuro-imaging "
        "annotators and subsequently approved by board-certified neuroradiologists, assigning each voxel to one of four mutually exclusive "
        "categories: background (0), necrotic core and non-enhancing tumor (1), peritumoral edema (2), or enhancing tumor (3).",
    )
    add_para(
        doc,
        "Input Channel Ordering: In our implemented data pipeline and PyTorch dataset class (src/data/dataset.py), the 4-channel input "
        "tensor fed into all neural network architectures strictly follows the canonical sequence order: Channel 0: T1n (native T1), "
        "Channel 1: T1c (contrast-enhanced T1), Channel 2: T2f (T2 FLAIR), and Channel 3: T2w (T2-weighted). Maintaining this explicit channel "
        "mapping is vital for reproducible tensor ingestion and cross-architecture compatibility. Note that downstream pre-inference input "
        "tensor-integrity validation verifies tensor type, 4D shape [1, 4, H, W], and positive spatial dimensions, but semantic modality "
        "presence and correct sequence identification remain strictly an upstream data-loading responsibility.",
    )

    add_heading(doc, "3.2 Subject-Level Dataset Census and Split Accounting", 2)
    add_para(
        doc,
        "A foundational contribution of this research is a complete, audited subject-level census of the entire BraTS 2023 GLI development "
        "cohort residing in the repository. In medical machine learning, a common and critical methodological flaw is case-level splitting, "
        "where longitudinal or multi-acquisition scans from the same biological subject are inadvertently distributed across training and "
        "evaluation partitions. Such data contamination induces severe patient leakage, causing models to memorize subject-specific anatomical "
        "features and inflating reported evaluation scores.",
    )
    add_para(
        doc,
        "To guarantee complete methodological integrity, our audit established that the 1,251 cases in the project repository belong to "
        "1,133 unique biological subjects. We instituted strict subject-level grouping to partition the census into completely subject-disjoint "
        "cohorts. Table 3.1 details the full project dataset accounting.",
    )

    headers_3_1 = ["Partition", "Cases", "Subjects", "Axial Slices", "Operational Status / Usage"]
    rows_3_1 = [
        ("Train Partition (Total)", "941", "847", "77,020", "Full training partition allocated in split manifest"),
        ("  -- Active 15-Epoch Training", "721", "667", "59,020", "Actively loaded & consumed by ResUNet & UNetResNet34"),
        ("  -- Historical Dev Reserve", "220", "180", "18,000", "Allocated to Train in manifest; unconsumed by 15-ep runs"),
        ("Validation Pool (Total)", "155", "143", "12,364", "Full validation pool consumed during epoch selection"),
        ("  -- Calibration Subset (Val_cal)", "50", "47", "3,982", "Used strictly for post-hoc temperature scaling fitting"),
        ("  -- Internal Evaluation (Val_eval)", "105", "96", "8,382", "Used for baseline comparison & agent triage evaluation"),
        ("Verified Development Total", "1,096", "990", "89,384", "Verified development population (Train + Validation pool)"),
        ("Sealed Test Cohort (Test_sealed)", "155", "143", "Quarantined", "Strict read-only quarantine; unopened, uninspected"),
        ("Total Project Census", "1,251", "1,133", "Uninspected*", "Total slice count not reported due to test quarantine"),
    ]
    add_table(doc, headers_3_1, rows_3_1, [4.0, 2.0, 2.0, 2.5, 6.0], left_align_cols=[0, 4])
    add_caption(doc, "Table 3.1: Full subject-level dataset census and partition allocation for BraTS 2023 GLI.")

    add_para(
        doc,
        "Critical Methodological Disclosures: Two vital methodological distinctions must be emphasized: First, within the 941-case "
        "Train partition, the mature 15-epoch models actually consumed 721 cases (667 subjects; 59,020 axial slices), while 220 cases "
        "(180 subjects) remained in reserve due to historical compute constraints. Second, while the 50-case calibration subset (Val_cal) "
        "and the 105-case internal evaluation subset (Val_eval) are completely subject-disjoint, the full 155-case validation pool was "
        "monitored during training to select the best validation checkpoint. Consequently, the 105-case internal evaluation subset was "
        "subject to checkpoint-selection bias and cannot be characterized as an untouched, independent test set. Third, the sealed test "
        "cohort of 155 cases (143 subjects) remains strictly quarantined and uninspected, preserving its blind integrity for future validation.",
        before=6,
    )

    add_heading(doc, "3.3 Historical 197-Case Pilot Split vs Mature Cohort", 2)
    add_para(
        doc,
        "Prior to the full-cohort census, an exploratory pilot study was conducted on a historical 197-case subset to evaluate initial "
        "pipeline feasibility and baseline model behavior. Table 3.2 compares the historical pilot split with the mature census partitioning.",
    )

    headers_3_2 = ["Attribute", "Historical Pilot Benchmark (Stage A)", "Mature Full-Cohort Baselines (Stage B)"]
    rows_3_2 = [
        ("Total Cohort Size", "197 complete cases", "1,251 complete cases / 1,133 unique subjects"),
        ("Training Set Size", "158 cases (13,074 axial slices)", "721 active cases (59,020 axial slices) [941 in partition]"),
        ("Validation Set Size", "39 cases (3,223 axial slices)", "155 cases (12,364 slices) [50 cal + 105 eval]"),
        ("Splitting Strategy", "Case-level random split (seed 42)", "Subject-level strict grouping (seed 42)"),
        ("Subject Leakage Risk", "Potential subject leakage present", "Zero subject leakage across partitions"),
        ("Benchmarked Models", "U-Net, U-Net++, Attention U-Net", "ResUNet-2D (Primary), UNetResNet34 (Reference)"),
        ("Training Budget", "5 epochs per model", "15 epochs per model"),
    ]
    add_table(doc, headers_3_2, rows_3_2, [4.0, 6.2, 6.3], left_align_cols=[0, 1, 2])
    add_caption(doc, "Table 3.2: Historical 197-case pilot development split vs mature 1,251-case cohort partitioning.")

    add_para(
        doc,
        "Because the historical 197-case split was conducted at case level rather than subject level, potential subject leakage "
        "could not be ruled out. The pilot results are therefore preserved strictly as historical development findings, while the "
        "15-epoch baselines represent the authoritative research baselines.",
        before=6,
    )

    add_heading(doc, "3.4 Data Preprocessing and Tensor Normalization", 2)
    add_para(
        doc,
        "MRI intensity values lack absolute physical units (unlike CT Hounsfield units) and exhibit substantial inter-scanner and "
        "intra-scanner intensity variations. To standardize inputs, each 3D volume underwent intensity normalization prior to slice "
        "extraction. Normalization statistics (mean and standard deviation) were calculated exclusively over non-zero brain voxels, "
        "completely excluding air background voxels: x_norm = (x - \u03bc_brain) / \u03c3_brain. This prevents zero-background voxels "
        "from artificially depressing the mean and distorting brain tissue contrast.",
    )
    add_para(
        doc,
        "Following volume normalization, 2D axial slices of dimension 240 x 240 voxels were extracted along the transaxial plane. Slices "
        "lacking sufficient brain tissue content were discarded during dataset preparation to optimize training efficiency. The resulting "
        "4-channel 2D tensors were cached to solid-state storage to eliminate repetitive GZIP decompression bottlenecks during training.",
    )


def build_chapter4_models_architecture(doc, artwork_dir: Path, sample_viz: Path):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 4: MODEL PORTFOLIO AND IMPLEMENTATION ARCHITECTURE", 1)

    add_heading(doc, "4.1 Five-Model Architecture Portfolio", 2)
    add_para(
        doc,
        "To systematically investigate architectural trade-offs in brain tumor segmentation, this project implemented and evaluated "
        "five distinct 2D convolutional neural network architectures. These models span the spectrum from foundational medical "
        "segmentation baselines to advanced residual and pretrained encoder-decoder networks. Table 4.1 details the complete portfolio, "
        "runtime parameter counts, and research roles.",
    )

    headers_4_1 = ["Model Architecture", "Encoder Backbone", "Trainable Parameters", "Non-Trainable Buffers", "Total Tensor Elements", "Research Role"]
    rows_4_1 = [
        ("U-Net", "4-level DoubleConv", "7,763,428", "5,906", "7,769,334", "Historical pilot (Stage A)"),
        ("U-Net++", "Nested dense skips", "9,160,068", "7,326", "9,167,394", "Historical pilot (Stage A)"),
        ("Attention U-Net", "Additive attention", "7,852,160", "6,886", "7,859,046", "Historical pilot (Stage A)"),
        ("ResUNet-2D", "Pre-activation residual", "8,114,604", "0", "8,114,604", "Primary Baseline (Stage B)"),
        ("UNetResNet34", "Pretrained ResNet-34", "24,327,524", "18,988", "24,346,512", "Reference Baseline (Stage B)"),
    ]
    add_table(doc, headers_4_1, rows_4_1, [3.0, 3.2, 2.5, 2.4, 2.5, 2.9], left_align_cols=[0, 1, 5])
    add_caption(doc, "Table 4.1: Five-model 2D segmentation architecture portfolio and verified runtime parameter counts.")

    add_para(
        doc,
        "Parameter Count Auditing: All parameter counts in Table 4.1 were audited via live PyTorch runtime inspection by querying both "
        "trainable parameters (p.numel() across model.parameters()) and non-trainable persistent tensor buffers (b.numel() across model.buffers(), "
        "such as batch normalization running mean and variance). For U-Net, U-Net++, and Attention U-Net, batch normalization contributes 5,906, "
        "7,326, and 6,886 buffer elements respectively. ResUNet-2D uses Group Normalization without persistent running statistics, yielding "
        "0 non-trainable buffers and 8,114,604 total tensor elements. UNetResNet34 contains 24,327,524 trainable parameters and 18,988 "
        "non-trainable persistent buffer elements from its ResNet-34 encoder and decoder batch-norm layers, totaling 24,346,512 tensor elements. "
        "Earlier informal reports citing 24,435,556 or 8,126,820 were identified as analytical typos and superseded.",
        before=6,
    )

    add_figure(
        doc,
        artwork_dir / "mri_modalities.png",
        "Figure 4.1: Multi-modal MRI input channels and segmentation subregions. The panel labels reflect historical BraTS terminology "
        "(T1, T1ce, T2, FLAIR) mapping to the canonical project modalities: T1 \u2192 T1n (native T1), T1ce \u2192 T1c (contrast-enhanced T1), "
        "T2 \u2192 T2w (T2-weighted), and FLAIR \u2192 T2f (T2-FLAIR). Model input tensor channel ordering in code is strictly "
        "('t1n', 't1c', 't2f', 't2w'). Ground truth labels delineate Whole Tumor (WT: green + yellow + red), Tumor Core (TC: yellow + red), "
        "and Enhancing Tumor (ET: yellow).",
        width=5.75,
    )

    add_figure(
        doc,
        artwork_dir / "model_portfolio.png",
        "Figure 4.2: Five-model 2D segmentation architecture portfolio and parameter complexity across the two experimental stages. "
        "Stage A (historical pilot): standard U-Net, U-Net++, and Attention U-Net. Stage B (mature research baselines): ResUNet-2D "
        "(Primary Research Baseline) and UNetResNet34 (Reference Research Baseline). Parameter counts reflect verified runtime PyTorch elements.",
        width=5.65,
    )

    add_figure(
        doc,
        sample_viz,
        "Figure 4.3: Representative qualitative segmentation output from the historical 5-epoch U-Net pilot for subject BraTS-GLI-01416-000 "
        "(axial slice 090). Note: In the final subject-level dataset census, BraTS-GLI-01416-000 is allocated to the Train partition "
        "(721 development cases) and is presented here solely for historical pilot qualitative demonstration, not as a sample from the "
        "independent 105-case evaluation pool.",
        width=5.65,
    )

    add_heading(doc, "4.2 Structural Details of Stage A Architectures", 2)
    add_para(
        doc,
        "The three architectures benchmarked during Stage A were implemented in custom PyTorch with a consistent base channel capacity "
        "of 32 features at the initial encoder level, doubling at each downsampling stage (32 \u2192 64 \u2192 128 \u2192 256 \u2192 512).",
    )
    add_bullets(
        doc,
        [
            "Standard U-Net: Features a 4-level contracting encoder with repeated 3x3 convolutions, Batch Normalization, and ReLU activations, "
            "interleaved with 2x2 max-pooling. The expansive decoder utilizes 2x2 bilinear upsampling followed by concatenation of skip features "
            "and double convolutions, terminating in a 1x1 convolution mapping 32 features to 4 class logits (7.76M parameters).",
            "U-Net++: Implements dense nested convolutional pathways along the skip connections. Instead of passing encoder features directly "
            "to the decoder, intermediate convolution nodes accumulate feature maps from all previous nodes at the same resolution level, "
            "significantly reducing semantic divergence at the cost of higher parameter complexity (9.16M parameters).",
            "Attention U-Net: Integrates additive soft attention gates into the skip pathways. Each attention gate takes the skip feature map x "
            "and decoder gating signal g, computes intermediate linear transformations with 1x1 convolutions, applies additive combination "
            "and sigmoid normalization, and outputs attention coefficients alpha \u2208 [0, 1] that re-weight skip features (7.85M parameters).",
        ],
    )

    add_heading(doc, "4.3 Structural Details of Stage B Architectures", 2)
    add_para(
        doc,
        "For the mature 15-epoch research experiments (Stage B), two advanced architectures were selected based on theoretical maturity "
        "and empirical stability:",
    )
    add_bullets(
        doc,
        [
            "ResUNet-2D (Primary Research Baseline): Combines the symmetric encoder-decoder topology of U-Net with pre-activation residual "
            "units (He et al., 2016) [8]. Each block computes x_l+1 = x_l + F(x_l, W_l), where F consists of Batch Normalization, ReLU, "
            "and 3x3 convolutions. The identity shortcut path allows gradients to flow unimpeded directly through the depth of the network, "
            "accelerating optimization and stabilizing deep feature extraction without inflating parameter counts (8.11M parameters).",
            "UNetResNet34 (Reference Research Baseline): Leverages a standard ImageNet-pretrained ResNet-34 encoder coupled with a MONAI-backed "
            "decoder. The encoder consists of 34 convolutional layers structured into 4 residual stages. Pretrained low-level edge and texture "
            "extractors provide exceptional boundary sensitivity, while the high-capacity decoder performs multi-scale skip feature fusion "
            "(24.33M trainable parameters; 24.35M total elements).",
        ],
    )

    add_heading(doc, "4.4 Training Dynamics and Loss Optimization Framework", 2)
    add_para(
        doc,
        "All models were trained using PyTorch with CUDA acceleration on an NVIDIA RTX 4070 Laptop GPU (8 GB VRAM). Training was conducted "
        "using full-precision single-precision floating-point arithmetic (FP32) without automated mixed precision (AMP) to ensure strict "
        "numerical consistency across all architectural benchmarks. Optimization was performed using the AdamW optimizer with an initial "
        "learning rate of 1 x 10^-3, fixed random seed 42, and stage-specific batch sizes: Stage A (historical 5-epoch pilot benchmark) utilized a mini-batch size of 2 "
        "axial slices with weight decay of 1 x 10^-4 on the 197-case pilot split; Stage B (mature 15-epoch research baselines) scaled training to a mini-batch size "
        "of 16 axial slices with cosine learning-rate decay and gradient clipping norm 1.0 on 721 training cases.",
    )
    add_para(
        doc,
        "The optimization objective optimized a combined compound loss: L_total = L_Dice + L_CE. The Dice loss measures soft multi-class "
        "overlap across the three tumor subcompartments: L_Dice = 1 - (1/K) \u2211_k [ (2 \u2211_i p_ik g_ik + \u03f5) / (\u2211_i p_ik^2 + \u2211_i g_ik^2 + \u03f5) ], "
        "where \u03f5 = 10^-5 prevents division by zero. The cross-entropy loss penalizes voxel-wise misclassifications: "
        "L_CE = - (1/N) \u2211_i \u2211_k g_ik log(p_ik). Both loss terms were assigned equal weights (1.0).",
    )

    page_break(doc)
    add_heading(doc, "4.5 Software Architecture and Repository Design", 2)
    add_para(
        doc,
        "The software codebase is structured modularly within the major-project-brain-tumor-segmentation repository to ensure strict "
        "reproducibility and clean separation of concerns. Table 4.2 outlines the primary repository modules.",
    )

    headers_4_2 = ["Subsystem", "Directory / File Path", "Primary Functional Responsibility"]
    rows_4_2 = [
        ("Data Ingestion", "src/data/dataset.py", "Volume loading, non-zero normalization, 2D slice caching, subject split enforcement"),
        ("Architectures", "src/models/*.py", "PyTorch implementations of U-Net, U-Net++, Attention U-Net, ResUNet, UNetResNet34"),
        ("Training Engine", "src/training/train_scaled.py", "Full-precision FP32 training, validation loop, checkpointing and JSON history logging"),
        ("Evaluation", "src/evaluation/metrics.py", "Voxel-wise Dice, IoU, class-wise overlap, and 95th percentile Hausdorff (HD95)"),
        ("Uncertainty", "src/uncertainty/calibration.py", "Softmax predictive entropy, NLL temperature scaling, ECE computation"),
        ("Agent Triage", "src/agent/models.py, policy.py", "Deterministic rule engine: input tensor-integrity validation, calibrated entropy triage, fallback switching"),
        ("Results & Logs", "results/*", "Execution manifests, checkpoint weights, training histories, visualization artifacts"),
    ]
    add_table(doc, headers_4_2, rows_4_2, [3.2, 4.5, 8.8], left_align_cols=[0, 1, 2])
    add_caption(doc, "Table 4.2: Core implementation modules and functional responsibilities across the codebase.")


def build_chapter5_experimental_results(doc, charts_dir: Path):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 5: EXPERIMENTAL BENCHMARK AND COMPARATIVE RESULTS", 1)

    add_heading(doc, "5.1 Two-Stage Experimental Progression", 2)
    add_para(
        doc,
        "To present an honest, comprehensive account of project development, this research report explicitly distinguishes "
        "two distinct experimental stages: Stage A (the historical five-epoch pilot benchmark) and Stage B (the mature fifteen-epoch "
        "research baselines). The historical pilot established initial comparative feasibility across three foundational architectures "
        "under constrained computational budgets. The mature research baselines scaled the training dataset by more than 4.5-fold "
        "(from 158 to 721 cases), expanded the training duration to fifteen epochs, and evaluated advanced residual architectures with "
        "comprehensive distance metrics. The historical pilot findings are preserved as developmental context, but are not conflated "
        "with the mature research baselines.",
    )

    add_heading(doc, "5.2 Stage A: Historical Five-Epoch Pilot Benchmark", 2)
    add_para(
        doc,
        "The historical benchmark evaluated standard U-Net, U-Net++, and Attention U-Net on the 197-case pilot split (158 training, "
        "39 validation cases; 3,223 validation slices). All models were trained under identical conditions: 5 epochs, batch size 2, "
        "AdamW optimizer, learning rate 0.001, compound loss (Dice + CE), and fixed random seed 42. Table 5.1 summarizes the best "
        "validation performance observed for each architecture.",
    )

    headers_5_1 = ["Model Architecture", "Parameters", "Best Mean Dice", "Best Mean IoU", "Best Epoch", "Training Runtime"]
    rows_5_1 = [
        ("U-Net", "7,763,428", "0.7711", "0.6329", "5", "39.9 min"),
        ("U-Net++", "9,160,068", "0.7937", "0.6582", "4", "67.3 min"),
        ("Attention U-Net", "7,852,160", "0.7291", "0.5886", "5", "47.1 min"),
    ]
    add_table(doc, headers_5_1, rows_5_1, [3.8, 2.5, 2.7, 2.7, 2.2, 2.6], left_align_cols=[0])
    add_caption(doc, "Table 5.1: Stage A historical five-epoch pilot benchmark on the 39-case validation cohort.")

    add_para(
        doc,
        "Important Metric Nuance: For U-Net++, the best validation checkpoint occurred at epoch 4, achieving a Mean Dice of 0.7937 and "
        "a best Mean IoU of 0.6582. At the final epoch (epoch 5), U-Net++ exhibited slight validation oscillation, recording a final Mean "
        "IoU of 0.630749. This distinction illustrates the necessity of preserving best-epoch checkpoints rather than assuming the final "
        "epoch is optimal.",
        before=6,
    )

    headers_5_2 = ["Model Architecture", "Best Epoch", "Label 1 Dice (NCR)", "Label 2 Dice (ED)", "Label 3 Dice (ET)"]
    rows_5_2 = [
        ("U-Net", "5", "0.6828", "0.7744", "0.8562"),
        ("U-Net++", "4", "0.7846", "0.7850", "0.8116"),
        ("Attention U-Net", "5", "0.5579", "0.7720", "0.8572"),
    ]
    add_table(doc, headers_5_2, rows_5_2, [4.5, 2.5, 3.2, 3.2, 3.1], left_align_cols=[0])
    add_caption(doc, "Table 5.2: Stage A foreground class-wise Dice at model-specific best validation epochs.")

    add_para(
        doc,
        "Class-Wise Behavior: Table 5.2 and Figure 6.2 reveal distinct architectural strengths. U-Net++ achieved the highest score on "
        "the necrotic core (Label 1: 0.7846), benefiting from dense skip connections that capture fine internal tumor structures. Standard "
        "U-Net and Attention U-Net performed strongly on enhancing tumor (Label 3: 0.8562 and 0.8572). However, Attention U-Net struggled "
        "on the necrotic core (0.5579) under the brief 5-epoch schedule, likely because attention gates require longer training to stabilize "
        "gating coefficients on small, irregular subregions.",
        before=6,
    )

    add_figure(
        doc,
        charts_dir / "model_metrics.png",
        "Figure 6.1: Best validation Dice and IoU scores across the historical three-model pilot (39 validation cases).",
        width=5.65,
    )

    add_figure(
        doc,
        charts_dir / "per_class_dice.png",
        "Figure 6.2: Foreground Dice comparison by encoded tumor label at model-specific best epochs in the historical pilot.",
        width=5.65,
    )

    add_figure(
        doc,
        charts_dir / "loss_curves.png",
        "Figure 6.3: Training and validation loss recorded at each epoch for historical pilot models.",
        width=5.65,
    )

    add_figure(
        doc,
        charts_dir / "mean_dice_trend.png",
        "Figure 6.4: Validation mean foreground Dice across the five pilot training epochs.",
        width=5.65,
    )

    add_para(
        doc,
        "Complete Foreground Misses: Analysis of complete misses across 2,578 validation slices containing true tumor pixels reveals "
        "notable differences across models: standard U-Net missed 4.58% (118 slices), Attention U-Net missed 4.81% (124 slices), while "
        "U-Net++ missed 12.53% (323 slices). This counter-intuitive finding highlights a critical trade-off: although U-Net++ achieved the "
        "highest aggregate Dice (0.7937), its dense multi-scale skip architecture was more prone to completely missing small, faint, "
        "peripheral tumor slices. The complete miss rate comparison is detailed alongside uncertainty failure modes in Section 6.2 (Figure 6.6).",
        before=6,
    )

    add_heading(doc, "5.3 Stage B: Mature 15-Epoch Research Baselines", 2)
    add_para(
        doc,
        "Stage B established the mature research baselines of the project, expanding the dataset to 721 training cases (667 subjects; "
        "59,020 axial slices) and extending the training duration to 15 epochs with mini-batch size 16. Two advanced architectures were "
        "trained: ResUNet-2D (Primary Research Baseline) and UNetResNet34 (Reference Research Baseline). Checkpoint selection was "
        "monitored on the full 155-case validation pool: ResUNet-2D achieved its best validation Mean Dice of 0.8670 at epoch 13; "
        "UNetResNet34 reached its best validation Mean Dice of 0.8554 at epoch 15.",
    )
    add_para(
        doc,
        "Internal Evaluation on 105 Cases: Performance was evaluated in depth on the 105-case internal evaluation subset (Val_eval, "
        "8,382 axial slices), reporting hierarchical Dice (Whole Tumor [WT], Tumor Core [TC], Enhancing Tumor [ET]) alongside 95th "
        "percentile Hausdorff Distance (HD95). Table 5.3 presents the audited comparison.",
    )

    headers_5_3 = ["Evaluation Metric", "ResUNet-2D (Primary Baseline)", "UNetResNet34 (Reference Baseline)", "Optimal Architecture"]
    rows_5_3 = [
        ("Mean Foreground Dice [95% CI]", "0.8641 [0.827, 0.893]", "0.8535 [0.814, 0.889]", "ResUNet-2D (+0.0106)"),
        ("Whole Tumor (WT) Dice", "0.9039", "0.8843", "ResUNet-2D (+0.0196)"),
        ("Tumor Core (TC) Dice", "0.8597", "0.8423", "ResUNet-2D (+0.0174)"),
        ("Enhancing Tumor (ET) Dice", "0.8286", "0.8339", "UNetResNet34 (+0.0053)"),
        ("Whole Tumor (WT) Median HD95", "4.12 mm", "5.00 mm", "ResUNet-2D (0.88 mm lower)"),
        ("Tumor Core (TC) Median HD95", "3.16 mm", "2.64 mm", "UNetResNet34 (0.52 mm lower)"),
        ("Enhancing Tumor (ET) Median HD95", "2.00 mm", "1.73 mm", "UNetResNet34 (0.27 mm lower)"),
    ]
    add_table(doc, headers_5_3, rows_5_3, [4.8, 4.2, 4.2, 3.3], left_align_cols=[0, 3])
    add_caption(doc, "Table 5.3: Stage B mature 15-epoch baseline comparison on the 105-case internal evaluation pool.")

    add_para(
        doc,
        "Architectural Trade-Off Analysis: Table 5.3 reveals a highly nuanced, clinically meaningful trade-off between the two baselines. "
        "ResUNet-2D achieves superior overall Mean Dice (0.8641 vs 0.8535) and substantially better Whole Tumor delineation (WT Dice: 0.9039 "
        "vs 0.8843; WT HD95: 4.12 mm vs 5.00 mm). Its lightweight pre-activation residual blocks provide smooth spatial regularization "
        "across diffuse edema margins. In contrast, UNetResNet34 demonstrates superior performance on the active tumor core and enhancing "
        "margin: it achieves higher ET Dice (0.8339 vs 0.8286), a lower median ET HD95 (1.73 mm vs 2.00 mm), and a lower median TC HD95 "
        "(2.64 mm vs 3.16 mm). The high representational capacity of the pretrained ResNet-34 encoder provides sharper edge localization "
        "around compact hyperintense structures. This complementary divergence strongly validates our dual-model portfolio design.",
        before=6,
    )

    add_heading(doc, "5.4 Checkpoint-Selection Bias and Methodological Boundaries", 2)
    add_para(
        doc,
        "Critical Methodological Disclosure: In adherence to the highest standards of scientific integrity, we explicitly disclose "
        "that the 105-case internal evaluation subset (Val_eval) participated in checkpoint selection because it formed part of the "
        "155-case validation pool monitored during training. Consequently, the evaluation metrics reported in Table 5.3 are subject to "
        "checkpoint-selection bias (hyperparameter tuning on validation data). These metrics represent internal model validation rather "
        "than completely untouched, independent test results. True out-of-sample generalization can only be confirmed on the 155-case "
        "sealed test cohort, which remains strictly unopened and uninspected.",
    )


def build_chapter6_uncertainty_calibration_agent(doc, charts_dir: Path, artwork_dir: Path):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 6: PREDICTIVE UNCERTAINTY, PROBABILITY CALIBRATION, AND RELIABILITY-AWARE TRIAGE", 1)

    add_heading(doc, "6.1 Predictive Uncertainty and Segmentation Quality Correlation", 2)
    add_para(
        doc,
        "To evaluate whether predictive uncertainty serves as an effective surrogate for true segmentation quality in the absence "
        "of ground truth, we analyzed the statistical relationship between normalized softmax entropy and empirical overlap metrics. "
        "Voxel-wise entropy H(x) was averaged across predicted foreground tumor voxels to yield a slice-level uncertainty metric: "
        "H_slice = (1 / |\u03a9_fg|) \u2211_{i \u2208 \u03a9_fg} H(x_i).",
    )
    add_para(
        doc,
        "In the historical pilot analysis of the standard U-Net model across the 39-case validation cohort, slice-level predictive entropy "
        "demonstrated a strong, statistically significant negative monotonic relationship with segmentation quality. Table 6.1 summarizes "
        "the correlation and discrimination findings.",
    )

    headers_6_1 = ["Evaluated Relationship / Task", "Observed Statistical Metric", "Empirical Interpretation in Study"]
    rows_6_1 = [
        ("Uncertainty vs Dice Overlap", "Spearman rho = -0.6542 (nominal p < 10^-15)", "Strong inverse correlation; elevated entropy reliably accompanies degraded Dice overlap."),
        ("Uncertainty vs IoU Overlap", "Spearman rho = -0.6864 (nominal p < 10^-15)", "Slightly stronger inverse correlation with Jaccard index due to sharper union penalization."),
        ("Low-Quality Slice Detection (Dice < 0.50)", "ROC-AUC = 0.8059", "Strong discriminative capability for flagging severely compromised segmentation contours."),
        ("Uncertainty-Only Model Selection", "Agreement = 48.35% (1,086 / 2,246 slices)", "Uncertainty alone is insufficient to select the Dice-best model among competing architectures."),
    ]
    add_table(doc, headers_6_1, rows_6_1, [4.5, 4.5, 7.5], left_align_cols=[0, 1, 2])
    add_caption(doc, "Table 6.1: Historical pilot uncertainty and segmentation quality correlation analysis (39-case pilot cohort).")

    add_para(
        doc,
        "Statistical Dependence Disclosure: We explicitly disclose that the Spearman rank correlations reported in Table 6.1 "
        "treated all N = 3,223 2D axial slices pooled across the 39 pilot validation subjects as independent observations. Because multiple "
        "slices originating from the same biological subject share anatomical spatial autocorrelation, slice pooling inflates effective degrees "
        "of freedom. While the observed rank coefficients (rho = -0.6542 and rho = -0.6864) establish a robust slice-level monotonic inverse association, "
        "the nominal p-value (< 10^-15) reflects slice-level pooling rather than patient-level statistical independence.",
        before=6,
    )
    add_para(
        doc,
        "Limits of Uncertainty Alone: Importantly, Table 6.1 highlights an essential limitation: while entropy reliably flags low-quality "
        "slices (ROC-AUC 0.8059), an exploratory policy that selected whichever model exhibited the lowest slice entropy matched the "
        "true Dice-best model on only 48.35% of slices (1,086 of 2,246 evaluated slices). This establishes that raw uncertainty alone "
        "cannot serve as an autonomous model selector; it must be coupled with calibration, spatial guards, and structured triage rules.",
        before=6,
    )

    page_break(doc)
    add_heading(doc, "6.2 Separation of Uncertainty Binning and Slice Distribution", 2)
    add_para(
        doc,
        "Resolution of Prior Presentation Flaw: In earlier historical report materials, an infographic (Figure 6.5) conflated two distinct "
        "grouping methodologies: it displayed slice counts derived from fixed, rounded entropy threshold intervals alongside Dice and IoU "
        "scores calculated over equal-frequency quintiles. This created the misleading visual impression that both statistics shared the same "
        "cohort partitions. In this Research Edition, we rigorously resolve this ambiguity by separating the data into two distinct, "
        "mathematically consistent visual panels: Figure 6.5A and Figure 6.5B.",
    )

    add_figure(
        doc,
        artwork_dir / "entropy_threshold_distribution.png",
        "Figure 6.5A: Slice distribution across rounded predictive entropy thresholds for all 3,223 pilot validation slices (foreground "
        "and non-foreground). Slice counts: [0.00, 0.05]: 820 slices (25.4%); (0.05, 0.07]: 713 slices (22.1%); (0.07, 0.10]: 648 slices "
        "(20.1%); (0.10, 0.16]: 537 slices (16.7%); and > 0.16: 505 slices (15.7%).",
        width=4.6,
    )

    add_figure(
        doc,
        artwork_dir / "entropy_quintile_quality.png",
        "Figure 6.5B: Segmentation quality across equal-frequency predictive entropy quintiles for 2,460 valid foreground validation slices "
        "(492 slices per quintile). Mean Dice drops monotonically from 0.8217 in Q1 (Very Low, mean H = 0.0433) to 0.7697 in Q2 (Low, H = 0.0626), "
        "0.6520 in Q3 (Medium, H = 0.0827), 0.5712 in Q4 (High, H = 0.1236), and 0.3008 in Q5 (Very High, H = 0.2875).",
        width=4.6,
    )

    add_para(
        doc,
        "Monotonic Quality Degradation: As demonstrated in Figure 6.5B, when slices are binned into strict equal-frequency quintiles "
        "(n = 492 slices each), segmentation overlap exhibits strict monotonic degradation as uncertainty rises. In the lowest-uncertainty "
        "quintile (Q1), Mean Dice reaches 0.8217 (IoU: 0.7532). By Q3, Dice drops to 0.6520 (IoU: 0.5527), and in the highest-uncertainty "
        "quintile (Q5), Mean Dice collapses to 0.3008 (IoU: 0.2102). This demonstrates, within the empirical boundaries of the historical "
        "39-case pilot cohort, that predictive entropy functions as an effective, monotonic proxy for segmentation degradation.",
        before=6,
    )

    add_figure(
        doc,
        charts_dir / "miss_rate.png",
        "Figure 6.6: Complete foreground miss rate among validation slices containing tumor pixels across historical pilot models.",
        width=2.4,
    )

    add_para(
        doc,
        "Complete Foreground Misses and Failure Manifestation: Figure 6.6 details the complete foreground miss distribution "
        "across 2,578 validation slices with true tumor burden. Complete misses occur when the model predicts zero foreground pixels, "
        "resulting in an indeterminate or zero slice Dice. Crucially, foreground entropy cannot be calculated over empty predictions, "
        "creating a critical blind spot for uncertainty triage that necessitates an explicit empty-prediction guard in the agent pipeline.",
    )

    add_heading(doc, "6.3 Post-Hoc Probability Calibration and Temperature Scaling", 2)
    add_para(
        doc,
        "Because raw deep neural network logits produce miscalibrated probabilities, post-hoc temperature scaling was implemented to "
        "align predicted probabilities with empirical accuracy. The optimal temperature parameter T* was fitted strictly on the 50-case "
        "calibration subset (Val_cal, 47 subjects, 3,982 slices) by minimizing negative log-likelihood (NLL) using L-BFGS optimization. "
        "The fitted scalar was T* = 2.880783 for ResUNet-2D and T* = 2.626696 for UNetResNet34. Both temperatures exceed 2.6, confirming "
        "that both uncalibrated architectures exhibited substantial overconfidence.",
    )
    add_para(
        doc,
        "Internal Evaluation Assessment: To evaluate calibration performance on the internal Val_eval subset, Expected Calibration Error (ECE) "
        "with M = 15 probability bins was measured across both the 50-case calibration cohort and the 105-case internal evaluation pool. Table 6.2 reports the results.",
    )

    headers_6_2 = ["Evaluated Cohort / Architecture", "Uncalibrated ECE", "Fitted Temperature (T*)", "Calibrated ECE", "Relative ECE Reduction"]
    rows_6_2 = [
        ("Val_cal (50 cases) — ResUNet-2D", "0.0502", "2.8808", "0.0084", "-83.3%"),
        ("Val_cal (50 cases) — UNetResNet34", "0.0655", "2.6267", "0.0093", "-85.8%"),
        ("Val_eval (105 cases) — ResUNet-2D", "0.0861", "2.8808 (applied)", "0.0344", "-60.0%"),
        ("Val_eval (105 cases) — UNetResNet34", "0.1098", "2.6267 (applied)", "0.0496", "-54.8%"),
    ]
    add_table(doc, headers_6_2, rows_6_2, [4.8, 2.8, 3.2, 2.7, 3.0], left_align_cols=[0, 4])
    add_caption(doc, "Table 6.2: Post-hoc temperature scaling and ECE reduction across calibration and evaluation cohorts.")

    add_para(
        doc,
        "Evaluation of Calibration on Internal Val_eval Subset: On the calibration cohort (50 cases), temperature scaling achieved an 83.3% reduction in ECE "
        "for ResUNet-2D (0.0502 \u2192 0.0084) and an 85.8% reduction for UNetResNet34 (0.0655 \u2192 0.0093). On internal evaluation "
        "(Val_eval, 105 cases), temperature scaling yielded substantial ECE reductions: ResUNet-2D ECE decreased from 0.0861 to 0.0344 (-60.0%), "
        "and UNetResNet34 ECE decreased from 0.1098 to 0.0496 (-54.8%). The observed calibration gap (0.0084 on calibration vs 0.0344 on internal evaluation "
        "for ResUNet) reflects the expected attenuation when transferring to cases outside the calibration fit. Both Val_cal and Val_eval "
        "were part of the 155-case validation pool monitored during epoch selection, inducing checkpoint-selection bias.",
    )

    add_heading(doc, "6.4 Implemented Deterministic Reliability-Aware Agent", 2)
    add_para(
        doc,
        "The core engineering contribution of this research is a deterministic, rule-based reliability agent (src/agent/policy.py, runner.py) "
        "that operationalizes selective prediction and automated triage through a structured, auditable decision workflow:",
    )
    add_bullets(
        doc,
        [
            "Stage 1: Input Tensor-Integrity Validation: Upstream data loading is responsible for assembling inputs in the expected canonical channel order (Channel 0: T1n, Channel 1: T1c, Channel 2: T2f, Channel 3: T2w). The validator checks the tensor interface, dimensionality, batch and channel counts ([1, 4, H, W]), and positive spatial dimensions (H, W > 0). It does not independently identify semantic modality content or detect every all-zero placeholder. Malformed tensors raise TypeError or ValueError exceptions and halt execution prior to inference (an interface exception, not an agent triage decision).",
            "Stage 2: Primary Segmentation & Temperature Calibration: Evaluates primary ResUNet-2D (Epoch 13), applies temperature scaling "
            "(T* = 2.8808), and computes voxel-wise softmax probabilities alongside normalized foreground Shannon entropy H(p).",
            "Stage 3: Anomaly & Empty-Prediction Guard: Intercepts slices where the primary model predicts zero foreground voxels despite "
            "neighboring axial context (radius k = 2) confirming persistent tumor burden (spatial discontinuity).",
            "Stage 4: Triage Decision Rules: Operates a selective prediction threshold (tau = 0.35). Predictions satisfying calibrated entropy "
            "H < 0.35 with non-empty foreground are autonomously accepted (ACCEPT).",
            "Stage 5: Dual-Model Fallback & Action Dispatch: Axial-context-supported empty predictions (k = 2) query UNetResNet34 (T* = 2.6267); "
            "non-empty masks are substituted (FALLBACK_SWITCH). Slices with H >= 0.35 or unresolved empty failures are flagged for simulated "
            "review (FLAG_UNRELIABLE). Candidate switching on entropy is disabled.",
        ],
    )
    add_para(
        doc,
        "Canonical B2 Evaluation Results: In the Gate 6A (Analysis B2) evaluation across 105 internal evaluation cases (6,549 foreground slices), "
        "actions were logged slice-by-slice. Table 6.3 details the reconciled action distribution with exact mutual exclusivity:",
    )

    headers_6_3 = ["Agent Triage Action Category", "Slice Count", "Percentage of Slices", "Empirical Outcome / Action Rationale"]
    rows_6_3 = [
        ("Autonomous Retain (Direct ResUNet-2D Accept)", "5,437 slices", "83.02%", "Calibrated foreground entropy H < 0.35 with non-empty foreground; accepted directly from primary ResUNet-2D."),
        ("Autonomous Retain (Fallback Switch to UNetResNet34)", "54 slices", "0.82%", "Primary ResUNet-2D predicted zero foreground voxels on slices supported by neighboring axial context; candidate substituted autonomously."),
        ("Simulated Referral (Flagged for Review)", "1,058 slices", "16.16%", "Calibrated foreground entropy H >= 0.35 or unresolvable empty-primary prediction; autonomous output withheld and flagged for review."),
        ("Total Evaluated Foreground Cohort", "6,549 slices", "100.00%", "Complete census of foreground slices across all 105 evaluation cases (5,437 + 54 + 1,058 = 6,549 slices)."),
    ]
    add_table(doc, headers_6_3, rows_6_3, [4.8, 2.4, 2.8, 6.8], left_align_cols=[0, 3])
    add_caption(doc, "Table 6.3: Reconciled triage and referral action distribution of the deterministic reliability-aware agent on 6,549 foreground slices.")

    add_para(
        doc,
        "Aggregate Segmentation Overlap Note: Across all 6,549 evaluated slices, aggregate pipeline Mean Dice is 0.8640 (vs 0.8641 for "
        "standalone ResUNet-2D). The utility of the reliability agent lies in selective triage and risk containment under the tested operating policy: "
        "by autonomously retaining 83.84% of slices (5,491 slices: 5,437 direct primary accepts + 54 fallback switches) and referring 16.16% "
        "(1,058 slices), the agent demonstrates effective selective prediction.",
        before=2,
    )


def build_chapter7_stress_limitations_conclusion(doc, artwork_dir: Path):
    chapter_section(doc)
    add_heading(doc, "CHAPTER 7: MISSING-MODALITY ROBUSTNESS, LIMITATIONS, AND FUTURE DIRECTIONS", 1)

    add_heading(doc, "7.1 Missing-Modality Robustness Stress Study", 2)
    add_para(
        doc,
        "Experimental Scope and Protocol Clarification: To investigate the robustness of the primary ResUNet-2D baseline under corrupted or "
        "incomplete inputs, we conducted a controlled synthetic channel-ablation stress test across the 105-case evaluation pool (Val_eval, "
        "comprising 96 unique subjects, 16,275 total slices, and 6,549 foreground slices). In this protocol, channel omission was simulated by "
        "clamping the target normalized input channel strictly to zero in a frozen 4-channel model without retraining. We emphasize the critical "
        "distinction between tensor-integrity validation in the implemented input pipeline (which raises exceptions on malformed tensors before inference) "
        "and these synthetic channel-zeroing experiments (which evaluate network behavior under zeroed input arrays). This experiment evaluates "
        "post-normalization zero-clamping behavior in a fixed-input architecture; it does not evaluate native missing-file acquisition handling, "
        "dynamic network reconfiguration, or validation of the upstream input tensor-integrity validation gate itself. An all-zero placeholder channel represents "
        "an artificial input artifact that differs from clinical acquisition failure.",
    )
    add_para(
        doc,
        "Policy Comparison: Analysis A vs Analysis B2: The missing-modality experiment was audited across two distinct policy regimes: "
        "(1) Analysis A (Simplified Entropy-Only Ablation): Slices with zero predicted foreground voxels were assigned a synthetic entropy of 0.0, "
        "causing complete false-negative misses (where H = 0.0 <= 0.35) to be erroneously retained as 100% confident. Under missing T2w (C3), "
        "this induced severe triage inversion: accepted slice Dice collapsed to 0.2466 while rejected slice Dice was 0.3417. "
        "(2) Analysis B2 (Full Gate 6A Multi-Model Agent Policy): Implements the authoritative decision rules (tau = 0.35, empty-prediction guard, "
        "and axial-context-supported fallback to UNetResNet34). Slices with empty primary predictions are intercepted by the empty guard and flagged "
        "as unreliable (or rescued by fallback), successfully restoring monotonic separation (accepted slice Dice 0.5307 vs rejected slice Dice 0.1535). "
        "Table 7.1 details the official Analysis B2 results across all five input conditions.",
    )

    headers_7_1 = ["Condition", "Omitted Modality Channel", "Case Mean Dice", "WT Dice", "TC Dice", "ET Dice", "B2 Pooled Referral % (Num / Denom)", "B2 Subject-Avg Referral %"]
    rows_7_1 = [
        ("C0: Baseline", "None (All 4 channels)", "0.8641", "0.9039", "0.8597", "0.8286", "16.16% (1,058 / 6,549)", "17.24%"),
        ("C1: Native T1 Ablated", "Omitted T1n (channel 0)", "0.8517", "0.8939", "0.8494", "0.8118", "18.16% (1,189 / 6,549)", "19.04%"),
        ("C2: Contrast T1c Ablated", "Omitted T1c (channel 1)", "0.4449", "0.8532", "0.3132", "0.1684", "17.44% (1,142 / 6,549)", "18.55%"),
        ("C3: T2-Weighted Ablated", "Omitted T2w (channel 3)", "0.6058", "0.4388", "0.6901", "0.6885", "62.65% (4,103 / 6,549)", "63.34%"),
        ("C4: T2-FLAIR Ablated", "Omitted T2f (channel 2)", "0.8394", "0.8748", "0.8342", "0.8091", "19.61% (1,284 / 6,549)", "20.70%"),
    ]
    add_table(doc, headers_7_1, rows_7_1, [2.5, 3.2, 1.8, 1.6, 1.6, 1.6, 2.7, 2.0], left_align_cols=[0, 1])
    add_caption(doc, "Table 7.1: Authoritative Analysis B2 missing-modality stress test across all five input conditions on 105 evaluation cases (N = 6,549 foreground slices).")

    add_para(
        doc,
        "Critical Discovery of the Contrast Uncertainty Blind Spot: Table 7.1 reveals a profound vulnerability in post-hoc predictive uncertainty. "
        "When native T1 is ablated (C1), overall performance degrades only slightly (Mean Dice 0.8517 vs 0.8641), indicating anatomical redundancy. "
        "However, when contrast-enhanced T1c is ablated (C2), Enhancing Tumor (ET) Dice collapses precipitously by 79.7% (from 0.8286 to 0.1684) and "
        "Tumor Core (TC) Dice falls by 63.6% (from 0.8597 to 0.3132). Crucially, Whole Tumor (WT) Dice remains high (0.8532) because peritumoral edema "
        "remains clearly visible on remaining T2 sequences. Under the official B2 policy, the pooled referral rate on C2 shifts only from 16.16% to 17.44% "
        "(subject-averaged: 17.24% to 18.55%), a change that is statistically non-significant (Bonferroni-adjusted Wilcoxon p_adj = 1.0000). "
        "Under the tested synthetic post-normalization T1c zero-clamping condition, severe ET and TC degradation was not accompanied by a statistically "
        "significant increase in aggregate referral. This demonstrates a specific limitation of the tested entropy-based policy and motivates "
        "explicit modality-presence and identity validation upstream prior to tensor assembly. The proportion of severe compartment-level failures "
        "referred remains unmeasured, as existing records report only aggregate foreground referral.",
    )
    add_para(
        doc,
        "Behavior Under Missing T2-Weighted and T2-FLAIR Input: In sharp contrast to C2, withholding T2w (C3) destroys whole tumor boundary delineation "
        "(WT Dice collapses to 0.4388), causing 2,621 foreground slices to produce complete background predictions. Under Analysis B2, the empty-prediction "
        "guard intercepts these misses, driving pooled referral to 62.65% (4,103 / 6,549 slices, p_adj = 7.12 x 10^-17). When T2f is ablated (C4), "
        "moderate degradation occurs (Mean Dice 0.8394), and referral increases modestly to 19.61% (p_adj = 0.00048). These empirical findings confirm "
        "that the empty-prediction guard provides robust protection against extensive tissue loss (C3); however, under the tested C2 condition "
        "(synthetic post-normalization T1c zero-clamping), the entropy-based policy did not reliably flag the observed ET and TC degradation.",
        before=6,
    )

    add_heading(doc, "7.2 Methodological Boundaries and Comprehensive Limitations", 2)
    add_para(
        doc,
        "To ensure transparency and academic integrity, we summarize the primary methodological boundaries, experimental constraints, "
        "and technical limitations of this study:",
    )
    add_bullets(
        doc,
        [
            "Checkpoint-Selection Bias: As disclosed in Section 5.4, the 105-case internal evaluation subset (Val_eval) was included in the "
            "155-case validation pool monitored during epoch selection. Consequently, the reported evaluation metrics reflect internal "
            "model validation rather than completely untouched, independent test results.",
            "Case-Level Pilot Partitioning: The historical 5-epoch pilot benchmark was evaluated on a case-level split (158 train / 39 val) "
            "that did not strictly enforce subject grouping, introducing potential patient leakage in the pilot results.",
            "2D Axial Slicing vs 3D Volumetric Context: Models were trained on 2D axial slices, forfeiting through-plane spatial context "
            "along the sagittal and coronal axes. While 2D processing enables rapid training and larger mini-batch sizes, it is susceptible "
            "to slice-to-slice boundary inconsistencies.",
            "Exploratory Foundation Model and 3D Pilot Constraints: Exploratory investigations into SegResNet-3D were constrained to a "
            "bounded patch-based 5-epoch feasibility pilot (10 cases, 96 x 96 x 96 crops) due to VRAM limitations. LiteMedSAM was evaluated as a "
            "custom promptable-decoder simulation without official pretrained weights. Full-scale nnU-Net v2 was not trained, and MedSAM2 was not evaluated.",
            "Non-Clinical Academic Prototype: The software contains no DICOM parser, PACS integration, or prospective clinical validation, "
            "and does not make diagnostic or treatment recommendations.",
        ],
    )

    headers_7_2 = ["Segmentation Paradigm", "Core Strengths", "Primary Technical Limitations", "Suitability for Clinical Workflow"]
    rows_7_2 = [
        ("2D CNNs (ResUNet / UNet++)", "Low compute requirements, fast inference, high in-plane resolution", "Lack 3D volumetric context, slice-to-slice boundary jitter", "Suitable for rapid triage and screening prototypes"),
        ("3D Volumetric CNNs (Patch/Full-Volume)", "True 3D spatial context, continuous volumetric boundary surfaces", "High VRAM consumption, small mini-batch limits, high computational cost", "Established research standard for volumetric segmentation benchmarks"),
        ("Promptable Foundation Models", "Zero-shot transfer, interactive promptable refinement", "Require external prompts (boxes/points), high parameter footprint", "Promising for interactive human-in-the-loop editing"),
        ("Reliability Triage Agents", "Selective prediction, risk containment, calibrated uncertainty", "Requires complex rule tuning, risk of uncertainty blind spots", "Essential layer for dependable automated triage"),
    ]
    add_table(doc, headers_7_2, rows_7_2, [3.5, 4.2, 4.8, 4.0], left_align_cols=[0, 1, 2, 3])
    add_caption(doc, "Table 7.2: Qualitative comparison of deep learning brain tumor segmentation paradigms.")

    add_heading(doc, "7.3 Exploratory 3D / Foundation Model Pilot Studies", 2)
    add_para(
        doc,
        "In addition to the primary 2D baselines, two exploratory investigations were undertaken to assess advanced segmentation paradigms: "
        "(1) SegResNet-3D Feasibility Pilot: An exploratory patch-based 3D convolutional network (MONAI SegResNet, 1,176,852 parameters) "
        "was evaluated to assess the engineering feasibility of 3D volumetric context under local hardware constraints. Rather than full-volume "
        "training or a converged benchmark, this experiment comprised a bounded 5-epoch feasibility run (50 optimization steps across 10 training "
        "cases) utilizing sub-volume patch crops (96 x 96 x 96, batch size 1). Under this exploratory configuration, compound loss decreased from "
        "2.0956 to 1.7366 with peak training VRAM of 622.3 MB, achieving an exploratory reconstruction Whole Tumor (WT) Dice of 0.2658. These metrics "
        "reflect an un-converged engineering feasibility test demonstrating volumetric data flow rather than competitive benchmark parity with the "
        "mature 2D baselines. (2) LiteMedSAM: An exploratory promptable segmentation pipeline was simulated using "
        "a lightweight ViT image encoder and a custom promptable mask decoder conditioned on bounding-box coordinates derived from ground-truth "
        "bounding boxes with random spatial jitter. Because official pretrained LiteMedSAM weights were not integrated, this experiment represents "
        "a proof-of-concept simulation of promptable refinement rather than a formal benchmark. State-of-the-art self-configuring 3D pipelines "
        "(nnU-Net v2) and next-generation video/volumetric models (MedSAM2) were not evaluated in this project scope.",
    )

    add_heading(doc, "7.4 Future Research Directions and Conclusion", 2)
    add_para(
        doc,
        "Future Research Roadmap: Building upon the validated baselines established in this report, future work should prioritize four "
        "strategic research directions: (1) Scaled 3D nnU-Net v2 Implementation: Transitioning the core segmentation backbone to full-volume "
        "3D self-configuring nnU-Net v2 on high-memory multi-GPU infrastructure; (2) Multi-Modal LLM Reasoning Agent: Integrating a vision-language "
        "agent (e.g., Med-PaLM or specialized multimodal LLM) capable of interpreting the quantitative uncertainty map, analyzing clinical "
        "metadata, and generating conversational, natural-language diagnostic summaries for neuroradiologists; (3) Multi-Center External "
        "Validation: Benchmarking calibrated triage policies on multi-center clinical cohorts outside BraTS to test cross-scanner generalization; "
        "and (4) Sealed Test Cohort Unblinding: Conducting a formal, audited evaluation on the 155-case sealed test cohort (Test_sealed) under "
        "strict pre-registered protocol governance.",
    )

    add_figure(
        doc,
        artwork_dir / "agent_architecture.png",
        "Figure 7.1: Architecture of the implemented deterministic reliability-aware AI agent and triage pipeline. "
        "The diagram illustrates the implemented deterministic workflow: input tensor-integrity validation across the 4 canonical channels "
        "(Channel 0: T1n, Channel 1: T1c, Channel 2: T2f, Channel 3: T2w; verifying torch.Tensor, 4D shape [1, 4, H, W], and positive spatial "
        "dimensions, where invalid inputs raise TypeError or ValueError exceptions prior to inference rather than producing an agent decision), "
        "primary ResUNet-2D segmentation (Epoch 13), post-hoc temperature scaling (T* = 2.8808), foreground Shannon entropy estimation, "
        "axial-context-supported empty prediction guards (radius k = 2), and deterministic three-way decision routing (accept H < 0.35 and "
        "non-empty, 5,437 slices, 83.02%; fallback switch to UNetResNet34 upon axial-context-supported empty primary prediction, 54 slices, 0.82%; "
        "or simulated referral for review when H >= 0.35 or unresolvable empty prediction, 1,058 slices, 16.16%). In the canonical B2 configuration, "
        "candidate switching on entropy alone is disabled. Key Architecture Invariant: Pre-inference tensor-integrity checks halt execution on "
        "malformed inputs via interface exceptions (not an agent decision or referral). Across the 6,549 evaluated slices, autonomous triage yields "
        "83.02% direct accepts, 0.82% context fallback switches, and 16.16% simulated referrals. Upstream data loading is responsible for assembling "
        "correct modality order and presence; synthetic zero-clamped channels pass tensor-integrity validation as valid numeric tensors.",
        width=6.0,
    )

    add_para(
        doc,
        "Conclusion: This major research project successfully demonstrated that while deep neural networks achieve high overlap accuracy in "
        "brain tumor segmentation, dependable medical AI demands orthogonal reliability mechanisms. By establishing an authoritative subject-level "
        "census of 1,251 cases, benchmarking five distinct 2D architectures across two developmental stages, implementing post-hoc temperature "
        "scaling that reduced ECE by 60%, and deploying a deterministic rule-based reliability agent that autonomously accepted 83.84% of predictions "
        "while flagging 16.16% for simulated review, this research establishes a reproducible, scientifically defensible foundation for reliability-aware "
        "neuro-oncological AI.",
        before=6,
    )


def build_references(doc):
    numbered_section(doc)
    add_heading(doc, "REFERENCES", 1)

    references = [
        "[1] B. H. Menze et al., \"The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS),\" IEEE Transactions on Medical Imaging, vol. 34, no. 10, pp. 1993-2024, Oct. 2015.",
        "[2] U. Baid et al., \"The RSNA-ASNR-MICCAI BraTS 2021 Benchmark on Brain Tumor Segmentation and Radiogenomic Classification,\" arXiv:2107.02314, 2021.",
        "[3] O. Ronneberger, P. Fischer, and T. Brox, \"U-Net: Convolutional Networks for Biomedical Image Segmentation,\" in Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2015, pp. 234-241.",
        "[4] Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, \"UNet++: A Nested U-Net Architecture for Medical Image Segmentation,\" in Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support, 2018, pp. 3-11.",
        "[5] O. Oktay et al., \"Attention U-Net: Learning Where to Look for the Pancreas,\" in Medical Imaging with Deep Learning (MIDL), 2018.",
        "[6] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On Calibration of Modern Neural Networks,\" in Proceedings of the 34th International Conference on Machine Learning (ICML), 2017, pp. 1321-1330.",
        "[7] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning,\" in Proceedings of the 33rd International Conference on Machine Learning (ICML), 2016, pp. 1050-1059.",
        "[8] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep Residual Learning for Image Recognition,\" in IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.",
        "[9] C. E. Shannon, \"A Mathematical Theory of Communication,\" The Bell System Technical Journal, vol. 27, no. 3, pp. 379-423, Jul. 1948.",
        "[10] A. Kendall and Y. Gal, \"What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?\" in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 5574-5584.",
        "[11] Z. Zhang, Q. Liu, and Y. Wang, \"Road Extraction by Deep Residual U-Net,\" IEEE Geoscience and Remote Sensing Letters, vol. 15, no. 5, pp. 749-753, May 2018.",
        "[12] M. J. Cardoso et al., \"MONAI: An open-source framework for deep learning in healthcare,\" arXiv:2211.02701, 2022.",
        "[13] Y. Geifman and R. El-Yaniv, \"Selective Classification for Deep Neural Networks,\" in Advances in Neural Information Processing Systems (NeurIPS), 2017, pp. 4878-4887.",
        "[14] A. Paszke et al., \"PyTorch: An Imperative Style, High-Performance Deep Learning Library,\" in Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 8024-8035.",
        "[15] ASNR-MICCAI BraTS 2023 Glioma Segmentation Challenge, Synapse Platform, syn53708249, 2023. [Online]. Available: https://www.synapse.org/#!Synapse:syn53708249",
        "[16] NK-NISHANT, \"Major Project: Reliability-Aware AI Agent for Brain Tumor Segmentation,\" GitHub Repository, 2026. [Online]. Available: https://github.com/NK-NISHANT/major-project-brain-tumor-segmentation",
    ]

    for ref in references:
        p = add_para(doc, ref, size=10, after=3)
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        p.paragraph_format.line_spacing = 1.15


def build_appendices(doc):
    numbered_section(doc)
    add_heading(doc, "APPENDIX A: FULL DATASET SPLIT MANIFEST AND CENSUS AUDIT", 1)
    add_para(
        doc,
        "The following audited manifest summarizes the complete subject-level cohort allocation across the 1,251 cases "
        "and 1,133 distinct biological subjects residing in the project repository. Strict patient-level grouping was enforced "
        "to eliminate cross-partition data leakage. Total slice count not reported because the sealed test cohort's slice count "
        "remains uninspected under strict quarantine. The verified development total comprises 89,384 slices across 1,096 cases.",
    )

    headers_a = ["Cohort Partition", "Case Count", "Subject Count", "Axial Slices", "Integrity & Partitioning Protocol"]
    rows_a = [
        ("Train Partition", "941 cases", "847 subjects", "77,020 slices", "Patient-disjoint; 721 cases consumed in 15-ep runs, 220 reserve"),
        ("Validation Pool", "155 cases", "143 subjects", "12,364 slices", "Patient-disjoint pool monitored during epoch selection"),
        ("  -- Val_cal Subset", "50 cases", "47 subjects", "3,982 slices", "Dedicated calibration set; fitted temperature scaling T*"),
        ("  -- Val_eval Subset", "105 cases", "96 subjects", "8,382 slices", "Internal evaluation pool; subject to checkpoint-selection bias"),
        ("Verified Dev Total", "1,096 cases", "990 subjects", "89,384 slices", "Verified development population (Train + Validation pool)"),
        ("Sealed Test Cohort", "155 cases", "143 subjects", "Quarantined", "Strict read-only quarantine; unopened, uninspected slice count"),
        ("Total Project Census", "1,251 cases", "1,133 subjects", "Uninspected*", "Complete repository census (*total slice count not reported)"),
    ]
    add_table(doc, headers_a, rows_a, [3.8, 2.5, 2.7, 2.8, 4.7], left_align_cols=[0, 4])
    add_caption(doc, "Table A.1: Subject-level dataset split manifest summary.")

    add_heading(doc, "APPENDIX B: REPRODUCIBILITY RECORD AND HYPERPARAMETER SPECIFICATIONS", 1)
    add_para(
        doc,
        "To ensure full academic and engineering reproducibility, Table B.1 details the software environment, hardware infrastructure, "
        "hyperparameter schedules, and deterministic controls utilized across all experiments.",
    )

    headers_b = ["System Parameter / Component", "Recorded Specification / Experimental Value"]
    rows_b = [
        ("Repository & Commit", "github.com/NK-NISHANT/major-project-brain-tumor-segmentation (audited branch)"),
        ("Python & Deep Learning Engine", "Python 3.11.9 (.venv); PyTorch 2.11.0+cu128; MONAI 1.6.0"),
        ("Hardware Compute Platform", "NVIDIA GeForce RTX 4070 Laptop GPU (8 GB VRAM); 13th Gen Intel(R) Core(TM) i9-13900HX"),
        ("Training Execution Precision", "Full single precision (FP32, torch.float32); no automatic mixed precision (AMP) utilized during training"),
        ("Stage A Hyperparameters", "5 epochs; batch size 2; AdamW (lr = 1e-3, weight decay = 1e-4); seed = 42"),
        ("Stage B Hyperparameters", "15 epochs; batch size 16; AdamW (lr = 1e-3, cosine decay); gradient clip norm = 1.0; seed = 42"),
        ("Compound Loss Formulation", "L_total = 1.0 * L_Dice + 1.0 * L_CrossEntropy (Dice epsilon = 1e-5)"),
        ("Primary Baseline Checkpoint", "results/full_training_15ep/resunet_2d/best_model.pt (Epoch 13)"),
        ("Reference Baseline Checkpoint", "results/full_training_15ep/unet_resnet34/best_model.pt (Epoch 15)"),
        ("HD95 Cohort Summary Record", "results/hd95_evaluation/hd95_summary.json"),
        ("Stage A Pilot Checkpoints", "results/baseline_unet2d_5ep, results/unetpp2d_5ep, results/attention_unet2d_5ep"),
        ("Fitted Temperatures (NLL)", "ResUNet-2D: T* = 2.880783; UNetResNet34: T* = 2.626696 (fitted on Val_cal)"),
        ("Reliability Agent Operating Point", "tau = 0.35, empty-prediction guard enabled, axial context window k = 2"),
    ]
    add_table(doc, headers_b, rows_b, [5.5, 11.0], left_align_cols=[0, 1])
    add_caption(doc, "Table B.1: Reproducibility checklist and hyperparameter specifications.")

    page_break(doc)
    add_heading(doc, "APPENDIX C: RESEARCH GOVERNANCE, VERIFICATION CHECKLIST, AND INTEGRITY DECLARATION", 1)
    add_para(
        doc,
        "In strict compliance with the project governance authority (MASTER_PROJECT_BLUEPRINT_v2.0.md) and the authoritative "
        "scientific audit (G1_A3_SCIENTIFIC_ACCURACY_AUDIT.md), Table C.1 documents the verification status and claim boundaries "
        "governing all reported results.",
    )

    headers_c = ["Research Governance Item", "Verified Status", "Scientific Authority & Boundary Declaration"]
    rows_c = [
        ("Protected Original Report", "VERIFIED BYTE-FOR-BYTE", "Original Minor Project report DOCX remains completely untouched."),
        ("Sealed Test Cohort Quarantine", "STRICTLY PRESERVED", "Test cohort was never opened, evaluated, or used for parameter tuning."),
        ("Checkpoint-Selection Bias", "OPENLY DISCLOSED", "105-case evaluation pool metrics reflect internal validation bias."),
        ("Missing-Modality Limitation", "PROMINENTLY DISCLOSED", "Under tested C2 condition, entropy policy did not reliably flag ET/TC degradation."),
        ("Clinical Deployment Boundary", "NON-CLINICAL PROTOTYPE", "Academic research prototype; no diagnostic or clinical claims asserted."),
    ]
    add_table(doc, headers_c, rows_c, [4.8, 3.8, 7.9], left_align_cols=[0, 1, 2])
    add_caption(doc, "Table C.1: Research governance, claim boundary, and methodological integrity checklist.")


def main(argv=None):
    args = parse_args(argv)
    out_path = Path(args.out).resolve()
    materials_dir = Path(args.materials).resolve()
    charts_dir = Path(args.charts_dir).resolve()
    artwork_dir = Path(args.artwork_dir).resolve()
    sample_viz = Path(args.sample_viz).resolve()
    overwrite = args.overwrite

    validate_preflight_inputs(
        out_path=out_path,
        materials_dir=materials_dir,
        charts_dir=charts_dir,
        artwork_dir=artwork_dir,
        sample_viz=sample_viz,
        overwrite=overwrite,
    )

    prepare_report_artwork(materials_dir, charts_dir, artwork_dir)
    doc = Document()
    base_style(doc)

    # 1. Cover Page (Section 0)
    setup_section(doc.sections[0], first_page=True)
    add_first_page_border(doc.sections[0])
    cover(doc, artwork_dir / "jiit_logo.png")

    # 2. Preliminary Pages (Section 1: Roman numerals)
    build_preliminary_pages(doc)

    # 3. Chapter 1: Introduction (Section 2: Arabic numerals starting at 1)
    build_chapter1_introduction(doc)

    # 4. Chapter 2: Background Study and Related Work
    build_chapter2_background(doc)

    # 5. Chapter 3: Dataset Census and Partitioning Methodology
    build_chapter3_dataset_methodology(doc)

    # 6. Chapter 4: Model Portfolio and Implementation Architecture
    build_chapter4_models_architecture(doc, artwork_dir, sample_viz)

    # 7. Chapter 5: Experimental Benchmark and Comparative Results
    build_chapter5_experimental_results(doc, charts_dir)

    # 8. Chapter 6: Predictive Uncertainty, Probability Calibration, and Reliability-Aware Triage
    build_chapter6_uncertainty_calibration_agent(doc, charts_dir, artwork_dir)

    # 9. Chapter 7: Missing-Modality Robustness, Limitations, and Future Directions
    build_chapter7_stress_limitations_conclusion(doc, artwork_dir)

    # 10. References
    build_references(doc)

    # 11. Appendices A, B, C
    build_appendices(doc)

    # Ensure subsequent sections do not inherit cover border
    for section in doc.sections[1:]:
        border = section._sectPr.find(qn("w:pgBorders"))
        if border is not None:
            section._sectPr.remove(border)

    doc.core_properties.title = (
        "Brain Tumor Segmentation Using Predictive Uncertainty and Towards a Reliability-Aware AI Agent "
        "(Research Edition)"
    )
    doc.core_properties.author = "Nishant Kumar; Mohammad Tariq; Mohammad Kaif"
    doc.core_properties.subject = "Major Project-II Research Edition Report, Course Code: 15B19CI791"

    # Atomic write pattern via temporary file
    tmp_out = out_path.parent / f".tmp_{uuid.uuid4().hex[:8]}_{out_path.name}"
    try:
        doc.save(str(tmp_out))
        with ZipFile(tmp_out, "r") as zf:
            namelist = zf.namelist()
            if "word/document.xml" not in namelist:
                raise ValueError(
                    f"Generated DOCX is corrupt: missing word/document.xml in {tmp_out}"
                )
            if "[Content_Types].xml" not in namelist:
                raise ValueError(
                    f"Generated DOCX is corrupt: missing [Content_Types].xml in {tmp_out}"
                )

        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Destination file '{out_path}' already exists. Use --overwrite to permit replacement."
            )
        os.replace(tmp_out, out_path)
    except Exception as exc:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except Exception:
                pass
        raise exc

    print(out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
