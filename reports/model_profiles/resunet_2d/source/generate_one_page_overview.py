"""
Script: generate_one_page_overview.py
Description: Generates the definitive A3 Landscape One-Page Overview for ResUNet-2D:
             - ResUNet_2D_One_Page_Overview.png (300 DPI)
             - ResUNet_2D_One_Page_Overview.pdf
             - ResUNet_2D_One_Page_Overview.svg
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

OUT_DIR = r"D:\MajorProject\reports\model_profiles\resunet_2d\one_page"
os.makedirs(OUT_DIR, exist_ok=True)

# Styling palette
BG_DARK = "#090D16"        # Deepest Slate
PANEL_BG = "#131C2E"       # Dark Navy Slate
CARD_BG = "#1A253C"        # Navy Card
BORDER_COL = "#263554"     # Card Border
TEXT_WHITE = "#FFFFFF"     # Pure White
TEXT_MUTED = "#94A3B8"     # Muted Slate
ACCENT_BLUE = "#38BDF8"    # Sky Blue
ACCENT_GREEN = "#10B981"   # Emerald
ACCENT_AMBER = "#F59E0B"   # Amber
ACCENT_RED = "#EF4444"     # Rose Red
ACCENT_PURPLE = "#A855F7"  # Purple
ACCENT_CYAN = "#06B6D4"    # Cyan

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'text.color': TEXT_WHITE,
    'axes.labelcolor': TEXT_MUTED,
    'figure.facecolor': BG_DARK,
    'axes.facecolor': PANEL_BG,
})

def build_one_page_overview():
    # A3 Landscape dimensions: 16.54 x 11.69 inches
    fig = plt.figure(figsize=(16.54, 11.69), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16.54)
    ax.set_ylim(0, 11.69)
    ax.axis('off')

    # Canvas Background
    ax.add_patch(patches.Rectangle((0, 0), 16.54, 11.69, color=BG_DARK))

    # =========================================================================
    # HEADER SECTION (Y: 10.45 to 11.45)
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 10.35), 15.54, 1.05, boxstyle="round,pad=0.08",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.5))

    # Left: Title & Identity
    ax.text(0.75, 11.08, "RESUNET-2D — MODEL DOSSIER & SYSTEM ARCHITECTURE",
            color=TEXT_WHITE, fontsize=17, fontweight='bold', va='center')
    ax.text(0.75, 10.65, "Primary Champion Segmentation Model  |  Stage C Reliability Agent Decision Backbone  |  15-Epoch Profile",
            color=ACCENT_BLUE, fontsize=10.5, fontweight='bold', va='center')

    # Right: Provenance & Key Stats Badges
    badges = [
        ("CHECKPOINT", "STAGE_C_15EP (Ep 13)", ACCENT_GREEN, 10.0),
        ("SHA-256", "1d85be47... (Verified)", ACCENT_CYAN, 12.2),
        ("PARAMETERS", "8.11M (32.4 MB)", ACCENT_AMBER, 14.3),
    ]
    for b_title, b_val, col, bx in badges:
        ax.add_patch(patches.FancyBboxPatch((bx, 10.52), 1.9, 0.72, boxstyle="round,pad=0.05",
                                            facecolor=CARD_BG, edgecolor=col, linewidth=1.2))
        ax.text(bx + 0.95, 10.98, b_title, color=col, fontsize=7.5, fontweight='bold', ha='center', va='center')
        ax.text(bx + 0.95, 10.70, b_val, color=TEXT_WHITE, fontsize=8.5, fontweight='bold', ha='center', va='center')

    # =========================================================================
    # 4 MAIN VERTICAL COLUMNS (Y: 1.15 to 10.15, Height: 9.0)
    # Column width: 3.70, Gap: 0.24, Margins: 0.50
    # =========================================================================
    col_w = 3.70
    gap = 0.24
    top_y = 1.15
    col_h = 9.0

    # -------------------------------------------------------------------------
    # COLUMN 1: Architecture & Engineering Specifications
    # -------------------------------------------------------------------------
    c1_x = 0.50
    ax.add_patch(patches.FancyBboxPatch((c1_x, top_y), col_w, col_h, boxstyle="round,pad=0.08",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.2))

    # Col 1 Header
    ax.add_patch(patches.FancyBboxPatch((c1_x + 0.15, 9.60), col_w - 0.30, 0.45, boxstyle="round,pad=0.05",
                                        facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1))
    ax.text(c1_x + col_w/2, 9.825, "1. ARCHITECTURE & SPECIFICATIONS", color=ACCENT_BLUE, fontsize=10.5, fontweight='bold', ha='center', va='center')

    # Col 1 Card 1: 4-Channel Input
    ax.add_patch(patches.FancyBboxPatch((c1_x + 0.15, 8.05), col_w - 0.30, 1.40, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c1_x + 0.3, 9.20, "Input Modalities (4-Channel 2D)", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    m_txt = (
        "• T1n: Anatomical baseline (240 × 240)\n"
        "• T1c: Active enhancing tumor core (Gadolinium)\n"
        "• T2w: Edema, fluid & ventricular architecture\n"
        "• T2f: FLAIR fluid attenuation (infiltrative margins)\n"
        "Preprocessing: Z-score normalized per brain ROI"
    )
    ax.text(c1_x + 0.3, 9.05, m_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 1 Card 2: Residual Block & Skip Connections
    ax.add_patch(patches.FancyBboxPatch((c1_x + 0.15, 5.75), col_w - 0.30, 2.15, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c1_x + 0.3, 7.65, "Network Topology & Flow", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    topo_txt = (
        "• Encoder: 4 stages with residual skip-blocks\n"
        "  Channels: 16 → 32 → 64 → 128 → Bottleneck 256\n"
        "• Residual Function: y = F(x, {Wi}) + x\n"
        "  Solves vanishing gradient; smooths loss landscape\n"
        "• Decoder: 4 transpose-conv stages (256 → 16)\n"
        "• Skip Connections: Direct feature re-injection\n"
        "  Concatenates encoder features with decoder maps\n"
        "• Output Head: 1×1 Conv → 4 raw class logits\n"
        "  Classes: Background (0), NCR (1), ED (2), ET (3)"
    )
    ax.text(c1_x + 0.3, 7.50, topo_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 1 Card 3: Computational Footprint
    ax.add_patch(patches.FancyBboxPatch((c1_x + 0.15, 3.85), col_w - 0.30, 1.75, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c1_x + 0.3, 5.35, "Computational & Memory Footprint", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    foot_txt = (
        "• Total Parameters: 8,111,892 (8.11 M)\n"
        "• Trainable Parameters: 8,111,892 (100%)\n"
        "• Checkpoint Disk Size: 32.4 MB (FP32)\n"
        "• Slice Latency (RTX 4070): ~1.8 ms / slice\n"
        "• Full 3D Volume Latency: ~0.28 s (155 slices)\n"
        "• GPU Memory Footprint: ~1.2 GB VRAM peak"
    )
    ax.text(c1_x + 0.3, 5.20, foot_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 1 Card 4: Loss Function
    ax.add_patch(patches.FancyBboxPatch((c1_x + 0.15, 1.30), col_w - 0.30, 2.40, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c1_x + 0.3, 3.45, "Training Formulation", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    train_txt = (
        "• Loss: Combined Soft Dice + Cross-Entropy\n"
        "  L_total = 0.5 × L_Dice + 0.5 × L_CE\n"
        "• Optimizer: Adam (lr = 1e-4, weight_decay = 1e-5)\n"
        "• Epochs: 15 epochs (Batch size = 16)\n"
        "• Hardware: NVIDIA GeForce RTX 4070 (8 GB)\n"
        "• Checkpoint Selection: Best validation Macro Dice\n"
        "  Peak reached at Epoch 13 (Dice = 0.8670)"
    )
    ax.text(c1_x + 0.3, 3.30, train_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # -------------------------------------------------------------------------
    # COLUMN 2: Segmentation Performance & Evaluation Cohort
    # -------------------------------------------------------------------------
    c2_x = c1_x + col_w + gap
    ax.add_patch(patches.FancyBboxPatch((c2_x, top_y), col_w, col_h, boxstyle="round,pad=0.08",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.2))

    # Col 2 Header
    ax.add_patch(patches.FancyBboxPatch((c2_x + 0.15, 9.60), col_w - 0.30, 0.45, boxstyle="round,pad=0.05",
                                        facecolor=CARD_BG, edgecolor=ACCENT_GREEN, linewidth=1))
    ax.text(c2_x + col_w/2, 9.825, "2. SEGMENTATION PERFORMANCE", color=ACCENT_GREEN, fontsize=10.5, fontweight='bold', ha='center', va='center')

    # Col 2 KPI Grid: 4 Metric Cards
    kpis = [
        ("Macro Mean Dice (Val_eval)", "0.8641", "105 Held-out Cases", ACCENT_GREEN),
        ("Whole Tumor (WT) Dice", "0.9039", "95% CI: [0.8878, 0.9189]", ACCENT_CYAN),
        ("Tumor Core (TC) Dice", "0.8597", "95% CI: [0.8351, 0.8821]", ACCENT_AMBER),
        ("Enhancing Tumor (ET) Dice", "0.8286", "95% CI: [0.7963, 0.8580]", ACCENT_PURPLE),
    ]
    for idx, (k_title, k_val, k_sub, col) in enumerate(kpis):
        kx = c2_x + 0.15 + (idx % 2) * 1.75
        ky = 8.15 - (idx // 2) * 1.35
        ax.add_patch(patches.FancyBboxPatch((kx, ky), 1.65, 1.20, boxstyle="round,pad=0.05",
                                            facecolor=CARD_BG, edgecolor=col, linewidth=1.2))
        ax.text(kx + 0.15, ky + 0.98, k_title, color=TEXT_MUTED, fontsize=7.5, fontweight='bold')
        ax.text(kx + 0.15, ky + 0.50, k_val, color=col, fontsize=16, fontweight='bold')
        ax.text(kx + 0.15, ky + 0.18, k_sub, color=TEXT_WHITE, fontsize=7)

    # Col 2 Card 2: Checkpoint Selection Dice vs Volumetric Dice Distinction
    ax.add_patch(patches.FancyBboxPatch((c2_x + 0.15, 4.05), col_w - 0.30, 2.45, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c2_x + 0.3, 6.25, "Distinct Evaluation Protocols Explained", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    proto_txt = (
        "1. Checkpoint Selection Validation (Epoch 13):\n"
        "   Foreground Macro Mean Dice = 0.8670\n"
        "   (NCR: 0.8210, ED: 0.8765, ET: 0.9037)\n"
        "   * 4-Class Mean including Background = 0.899968\n\n"
        "2. 3D Volumetric Evaluation (105 Val_eval Cases):\n"
        "   Volumetric Macro Mean Dice = 0.8641\n"
        "   Computed by reconstructing full 3D patient volumes\n"
        "   across all 16,275 contiguous slices.\n\n"
        "3. Slice-Level Unfiltered Macro Dice: 0.7150\n"
        "   Averaged across 6,549 ground-truth tumor slices."
    )
    ax.text(c2_x + 0.3, 6.10, proto_txt, color=TEXT_MUTED, fontsize=7.8, va='top', linespacing=1.30)

    # Col 2 Card 3: Hausdorff Distance HD95
    ax.add_patch(patches.FancyBboxPatch((c2_x + 0.15, 1.30), col_w - 0.30, 2.60, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c2_x + 0.3, 3.65, "95th Percentile Hausdorff Distance (HD95)", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    hd_txt = (
        "• Physical boundary distance in millimeters (mm):\n"
        "  - Enhancing Tumor (ET): Median = 2.00 mm\n"
        "    (Mean: 9.92 mm, 95% CI: [4.71, 18.66])\n"
        "  - Tumor Core (TC): Median = 3.16 mm\n"
        "    (Mean: 8.98 mm, 95% CI: [6.08, 12.74])\n"
        "  - Whole Tumor (WT): Median = 4.12 mm\n"
        "    (Mean: 15.29 mm, 95% CI: [11.38, 20.05])\n\n"
        "• Missing Modality Robustness:\n"
        "  - Drop T1c: TC Dice drops to 0.3132, ET to 0.1684\n"
        "  - Drop T2w: WT Dice drops to 0.4388\n"
        "  Confirms high clinical reliance on specific sequences."
    )
    ax.text(c2_x + 0.3, 3.50, hd_txt, color=TEXT_MUTED, fontsize=7.8, va='top', linespacing=1.30)

    # -------------------------------------------------------------------------
    # COLUMN 3: Calibration & Predictive Uncertainty
    # -------------------------------------------------------------------------
    c3_x = c2_x + col_w + gap
    ax.add_patch(patches.FancyBboxPatch((c3_x, top_y), col_w, col_h, boxstyle="round,pad=0.08",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.2))

    # Col 3 Header
    ax.add_patch(patches.FancyBboxPatch((c3_x + 0.15, 9.60), col_w - 0.30, 0.45, boxstyle="round,pad=0.05",
                                        facecolor=CARD_BG, edgecolor=ACCENT_AMBER, linewidth=1))
    ax.text(c3_x + col_w/2, 9.825, "3. CALIBRATION & UNCERTAINTY", color=ACCENT_AMBER, fontsize=10.5, fontweight='bold', ha='center', va='center')

    # Col 3 Card 1: Temperature Scaling
    ax.add_patch(patches.FancyBboxPatch((c3_x + 0.15, 7.15), col_w - 0.30, 2.30, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c3_x + 0.3, 9.20, "Post-Hoc Temperature Scaling (T*)", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    cal_txt = (
        "• Optimal Calibration Temperature: T* = 2.880459\n"
        "  Optimized on Val_cal (50 cases) via L-BFGS (36 iters)\n"
        "• Calibration Set (Val_cal) Results:\n"
        "  - ECE: 5.02% → 0.84% (83.3% relative reduction)\n"
        "  - NLL: 0.4154 → 0.2173 (47.7% reduction)\n"
        "  - MCE: 28.11% → 7.26%\n"
        "• Evaluation Set (Val_eval 105 cases) Generalization:\n"
        "  - ECE: 8.59% → 3.41% (60.3% relative reduction)\n"
        "  - NLL: 1.0277 → 0.4415 (57.0% reduction)\n"
        "  - MCE: 30.73% → 26.42%"
    )
    ax.text(c3_x + 0.3, 9.05, cal_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 3 Card 2: Predictive Entropy Distribution
    ax.add_patch(patches.FancyBboxPatch((c3_x + 0.15, 4.35), col_w - 0.30, 2.65, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c3_x + 0.3, 6.75, "Shannon Predictive Entropy (H)", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    ent_txt = (
        "• Formula: H = - Σ p_c · ln(p_c) / ln(4) in [0, 1]\n"
        "  Evaluated on calibrated probabilities (T = 2.8805)\n"
        "• Distribution Across 6,874 Non-Empty Slices:\n"
        "  - Mean Entropy: 0.2686 ± 0.0903\n"
        "  - Median: 0.2474   |   IQR: [0.1995, 0.3185]\n"
        "  - Min / Max: 0.0000 / 0.6146\n"
        "• Frozen Operating Threshold: τ = 0.35\n"
        "  - H < 0.35: 5,619 slices (81.74% of non-empty)\n"
        "    High agreement, mean Dice = 0.7803\n"
        "  - H ≥ 0.35: 1,255 slices (18.26% of non-empty)\n"
        "    Severe uncertainty, mean Dice drops to 0.4320"
    )
    ax.text(c3_x + 0.3, 6.60, ent_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 3 Card 3: Boundary Sampling Protocol
    ax.add_patch(patches.FancyBboxPatch((c3_x + 0.15, 1.30), col_w - 0.30, 2.90, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c3_x + 0.3, 3.95, "Calibration Protocol Safeguards", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    safe_txt = (
        "• Pre-Flight Verification Handshake:\n"
        "  SHA-256 cryptographic match between calibration JSON\n"
        "  and model checkpoint ensures no temperature drift.\n\n"
        "• Disjoint Cohort Split:\n"
        "  - Val_cal: 50 cases (50 patient subjects)\n"
        "  - Val_eval: 105 cases (96 patient subjects)\n"
        "  Zero patient leakage between calibration & test.\n\n"
        "• Statistical Clustering:\n"
        "  Cluster-bootstrap (1,000 resamples) grouped by\n"
        "  patient subject ID accounts for within-patient slice\n"
        "  correlations in all reported confidence intervals."
    )
    ax.text(c3_x + 0.3, 3.80, safe_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # -------------------------------------------------------------------------
    # COLUMN 4: Stage C3 Decision Triage & Clinical Reliability
    # -------------------------------------------------------------------------
    c4_x = c3_x + col_w + gap
    ax.add_patch(patches.FancyBboxPatch((c4_x, top_y), col_w, col_h, boxstyle="round,pad=0.08",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.2))

    # Col 4 Header
    ax.add_patch(patches.FancyBboxPatch((c4_x + 0.15, 9.60), col_w - 0.30, 0.45, boxstyle="round,pad=0.05",
                                        facecolor=CARD_BG, edgecolor=ACCENT_PURPLE, linewidth=1))
    ax.text(c4_x + col_w/2, 9.825, "4. STAGE C RELIABILITY AGENT", color=ACCENT_PURPLE, fontsize=10.5, fontweight='bold', ha='center', va='center')

    # Col 4 Card 1: 3-Action Triage Policy
    ax.add_patch(patches.FancyBboxPatch((c4_x + 0.15, 6.30), col_w - 0.30, 3.15, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c4_x + 0.3, 9.20, "Frozen C3 Policy (v1.1.3 Semantics)", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    
    # 3 Action breakdown rows
    actions = [
        ("ACCEPT", "5,619 Slices (34.53%)", "81.74% of non-empty; H < 0.35; Autonomous", ACCENT_GREEN, 8.85),
        ("FALLBACK_SWITCH", "147 Slices (0.90%)", "Rescues 54 true tumors via UNetResNet34", ACCENT_AMBER, 7.95),
        ("FLAG_UNRELIABLE", "10,509 Slices (64.57%)", "Includes 9,401 true empty slices + 1,108 uncertain", ACCENT_RED, 7.05),
    ]
    for act_name, act_count, act_desc, col, ay in actions:
        ax.add_patch(patches.FancyBboxPatch((c4_x + 0.25, ay - 0.45), col_w - 0.50, 0.75, boxstyle="round,pad=0.04",
                                            facecolor=BG_DARK, edgecolor=col, linewidth=1.2))
        ax.text(c4_x + 0.40, ay + 0.08, act_name, color=col, fontsize=8.5, fontweight='bold')
        ax.text(c4_x + col_w - 0.40, ay + 0.08, act_count, color=TEXT_WHITE, fontsize=8, ha='right')
        ax.text(c4_x + 0.40, ay - 0.22, act_desc, color=TEXT_MUTED, fontsize=7.2)

    # Col 4 Card 2: Clinical Impact & Severe Failure Reduction
    ax.add_patch(patches.FancyBboxPatch((c4_x + 0.15, 3.85), col_w - 0.30, 2.30, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c4_x + 0.3, 5.90, "Clinical Impact on Tumor Slices", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    imp_txt = (
        "• Evaluated on 6,549 Ground-Truth Tumor Slices:\n"
        "  - Retained by Agent: 5,491 slices (83.84%)\n"
        "    Retained Mean Macro Dice = 0.7803\n"
        "  - Referred to Clinicians: 1,058 slices (16.16%)\n"
        "    Referred Mean Macro Dice = 0.3891\n"
        "• Severe Failure Rate (Dice < 0.50):\n"
        "  - Baseline (No Agent): 1,343 slices (20.51%)\n"
        "  - With Stage C Agent: 742 slices (13.51%)\n"
        "  - Net Reduction: 34.1% relative cut in severe errors!"
    )
    ax.text(c4_x + 0.3, 5.75, imp_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # Col 4 Card 3: Axial Continuity Rule
    ax.add_patch(patches.FancyBboxPatch((c4_x + 0.15, 1.30), col_w - 0.30, 2.40, boxstyle="round,pad=0.06",
                                        facecolor=CARD_BG, edgecolor=BORDER_COL, linewidth=1))
    ax.text(c4_x + 0.3, 3.45, "Axial Context Guardrails", color=TEXT_WHITE, fontsize=9.5, fontweight='bold')
    ax_txt = (
        "• Context Window: Radius k = 2 ([z-2, z-1, z+1, z+2])\n"
        "• Spatial Support Rule: Requires ≥ 1 evaluated neighbor\n"
        "  with non-empty foreground before triggering fallback.\n"
        "• Boundary Accounting (Batch 1C.1 Parity):\n"
        "  Out-of-volume slices (z < 0, z > 154) count strictly as\n"
        "  unavailable neighbors (2 unavailable at z=0, z=154).\n"
        "• Prevents spurious model switching on healthy slices."
    )
    ax.text(c4_x + 0.3, 3.30, ax_txt, color=TEXT_MUTED, fontsize=8, va='top', linespacing=1.35)

    # =========================================================================
    # FOOTER SECTION (Y: 0.30 to 0.95)
    # =========================================================================
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.28), 15.54, 0.72, boxstyle="round,pad=0.06",
                                        facecolor=PANEL_BG, edgecolor=BORDER_COL, linewidth=1.2))

    disc_txt = (
        "EVIDENCE & BOUNDARY DISCLOSURES: All metrics computed from frozen Stage C evaluation artifacts on held-out internal validation cohort (105 cases, 16,275 slices). "
        "Confidence intervals use subject-level cluster bootstrap (1,000 resamples, seed 42). "
        "Strict Quarantine Test Set (155 cases) remains sealed and untouched. "
        "ACADEMIC NOTICE: Academic engineering prototype. NOT approved by FDA/CDSCO for clinical diagnosis or surgical guidance."
    )
    ax.text(0.70, 0.76, "SCIENTIFIC EVIDENCE PROVENANCE & REGULATORY NOTICE:",
            color=ACCENT_AMBER, fontsize=7.5, fontweight='bold')
    ax.text(0.70, 0.46, disc_txt, color=TEXT_MUTED, fontsize=6.8, va='center')

    # Save outputs
    png_path = os.path.join(OUT_DIR, "ResUNet_2D_One_Page_Overview.png")
    pdf_path = os.path.join(OUT_DIR, "ResUNet_2D_One_Page_Overview.pdf")
    svg_path = os.path.join(OUT_DIR, "ResUNet_2D_One_Page_Overview.svg")

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.savefig(svg_path)
    plt.close()
    print(f"[✓] Successfully generated {png_path}")
    print(f"[✓] Successfully generated {pdf_path}")
    print(f"[✓] Successfully generated {svg_path}")

if __name__ == "__main__":
    build_one_page_overview()
