"""
Generate 7 High-Impact, Big-Font Scientific Visual Assets for U-Net++.
Optimized for far-distance readability, viva presentations, and high-DPI printouts.
"""

import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Paths
WORKSPACE = Path(r"D:\MajorProject")
REPORTS_ASSETS_DIR = WORKSPACE / "reports" / "model_profiles" / "unetpp" / "assets"
DOCS_FIG_DIR = REPORTS_ASSETS_DIR  # Decoupled
# DOCS_FIG_DIR mkdir removed
REPORTS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Typography settings for maximum visibility
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = False
plt.rcParams['axes.edgecolor'] = '#334155'
plt.rcParams['axes.linewidth'] = 1.8

# High contrast palette
NAVY = '#0F172A'       # Slate 900
SLATE = '#334155'      # Slate 700
BLUE = '#0284C7'       # Sky 600
CYAN = '#06B6D4'       # Cyan 500
TEAL = '#0D9488'       # Teal 600
GREEN = '#10B981'      # Emerald 500
CORAL = '#E11D48'      # Rose 600
AMBER = '#D97706'      # Amber 600
PURPLE = '#7C3AED'     # Violet 600
LIGHT_BG = '#F8FAFC'   # Slate 50
CARD_BG = '#FFFFFF'    # White
GRID_COLOR = '#E2E8F0' # Slate 200

def save_fig(fig, filename):
    for out_dir in [REPORTS_ASSETS_DIR]:
        path = out_dir / filename
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)
    print(f"[✓] Saved: {filename}")


# -----------------------------------------------------------------------------
# FIG 1: FULL ARCHITECTURE DIAGRAM (NESTED DENSE SKIPS)
# -----------------------------------------------------------------------------
def gen_01_architecture():
    fig = plt.figure(figsize=(18, 10), facecolor='#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Header
    ax.text(0.5, 0.95, "U-Net++ (Nested Dense Skip Pathways Architecture)",
            ha='center', va='center', fontsize=26, fontweight='bold', color=NAVY)
    ax.text(0.5, 0.91, "Input: 4-Channel mpMRI (240×240) | Output: 4-Class Segmentation Logits | Total Parameters: 9,160,068 (~9.16M)",
            ha='center', va='center', fontsize=16, fontweight='semibold', color=SLATE)

    # Helper function for nodes
    def draw_node(x, y, w, h, name, desc, color, text_color='#FFFFFF', border_color=None):
        border = border_color if border_color else color
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                      facecolor=color, edgecolor=border, linewidth=2.0, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, name, ha='center', va='center',
                fontsize=15, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w/2, y + h*0.28, desc, ha='center', va='center',
                fontsize=11.5, fontweight='medium', color=text_color, zorder=4)

    # Grid of nodes: level i (0 to 4), stage j (0 to 4-i)
    node_coords = {}
    x_bases = [0.12, 0.28, 0.44, 0.60, 0.76]
    y_bases = [0.73, 0.56, 0.39, 0.22, 0.05]
    node_w = 0.11
    node_h = 0.095

    # Channels and resolutions
    stage_info = [
        ("32 ch | 240×240"),
        ("64 ch | 120×120"),
        ("128 ch | 60×60"),
        ("256 ch | 30×30"),
        ("512 ch | 15×15"),
    ]

    # Draw Nodes
    for i in range(5):
        for j in range(5 - i):
            x = x_bases[j]
            y = y_bases[i]
            node_coords[(i, j)] = (x, y)
            name = f"X^{{{i},{j}}}"
            desc = stage_info[i]
            
            if j == 0:
                # Encoder
                c = TEAL if i < 4 else NAVY
            elif i == 0 and j == 4:
                # Final output head
                c = BLUE
            else:
                # Dense intermediate skip node
                c = PURPLE if j % 2 == 1 else CYAN

            draw_node(x, y, node_w, node_h, name, desc, c)

    # Input node
    draw_node(0.015, 0.73, 0.075, 0.095, "mpMRI", "4 Ch\n240²", SLATE)
    ax.annotate('', xy=(0.12, 0.777), xytext=(0.09, 0.777),
                arrowprops=dict(facecolor=SLATE, edgecolor=SLATE, width=2.5, headwidth=8, shrink=0.05))

    # Output node
    draw_node(0.895, 0.73, 0.09, 0.095, "Output", "1×1 Conv\n4 Classes", CORAL)
    ax.annotate('', xy=(0.895, 0.777), xytext=(0.87, 0.777),
                arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, width=2.5, headwidth=8, shrink=0.05))

    # Downsampling arrows (Encoder column j=0)
    for i in range(4):
        x1, y1 = node_coords[(i, 0)]
        x2, y2 = node_coords[(i+1, 0)]
        ax.annotate('', xy=(x2 + node_w/2, y2 + node_h), xytext=(x1 + node_w/2, y1),
                    arrowprops=dict(facecolor=TEAL, edgecolor=TEAL, width=2.5, headwidth=8, shrink=0.08))
        ax.text(x1 + node_w/2 + 0.035, (y1 + y2 + node_h)/2, "MaxPool (2×2)",
                fontsize=11, fontweight='bold', color=TEAL, ha='left', va='center')

    # Upsampling arrows and Dense Skips
    for i in range(4):
        for j in range(4 - i):
            # Upsample from (i+1, j) to (i, j+1)
            xu, yu = node_coords[(i+1, j)]
            xt, yt = node_coords[(i, j+1)]
            ax.annotate('', xy=(xt, yt + node_h*0.3), xytext=(xu + node_w, yu + node_h*0.7),
                        arrowprops=dict(facecolor=GREEN, edgecolor=GREEN, width=2.0, headwidth=7, linestyle=':', shrink=0.08))

            # Horizontal dense skip connections from (i, k) for k <= j to (i, j+1)
            for k in range(j + 1):
                xs, ys = node_coords[(i, k)]
                if k == j:
                    ax.annotate('', xy=(xt, yt + node_h*0.7), xytext=(xs + node_w, ys + node_h*0.7),
                                arrowprops=dict(facecolor=AMBER, edgecolor=AMBER, width=2.2, headwidth=7, shrink=0.08))

    # Legend Card
    leg_x, leg_y, leg_w, leg_h = 0.44, 0.05, 0.54, 0.13
    rect = patches.FancyBboxPatch((leg_x, leg_y), leg_w, leg_h, boxstyle="round,pad=0.015",
                                  facecolor='#FFFFFF', edgecolor=GRID_COLOR, linewidth=2.0, zorder=5)
    ax.add_patch(rect)
    ax.text(leg_x + 0.02, leg_y + leg_h - 0.03, "KEY ARCHITECTURAL ELEMENTS:",
            fontsize=13, fontweight='bold', color=NAVY, zorder=6)
    
    items = [
        ("Encoder Spine (Levels 0-4)", TEAL),
        ("Dense Skip Nodes (X^{i,j})", CYAN),
        ("Decoder Output Node (X^{0,4})", BLUE),
        ("Dense Horizontal Skips", AMBER),
        ("Bilinear Upsample (2×)", GREEN),
        ("4-Class Output Head", CORAL)
    ]
    for idx, (label, col) in enumerate(items):
        col_x = leg_x + 0.02 + (idx % 3) * 0.175
        col_y = leg_y + 0.065 if idx < 3 else leg_y + 0.025
        circ = patches.Circle((col_x + 0.008, col_y + 0.005), 0.006, facecolor=col, edgecolor=NAVY, linewidth=1, zorder=6)
        ax.add_patch(circ)
        ax.text(col_x + 0.02, col_y + 0.005, label, fontsize=11, fontweight='bold', color=SLATE, va='center', zorder=6)

    save_fig(fig, "fig_01_unetpp_nested_architecture.png")


# -----------------------------------------------------------------------------
# FIG 2: 15-EPOCH TRAINING TRAJECTORY & CONVERGENCE
# -----------------------------------------------------------------------------
def gen_02_trajectory():
    history_path = WORKSPACE / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "history.csv"
    df = pd.read_csv(history_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), facecolor='#FFFFFF')
    fig.suptitle("U-Net++: Complete 15-Epoch Training Trajectory & Convergence Metrics",
                 fontsize=22, fontweight='bold', color=NAVY, y=0.98)

    epochs = df['epoch'].values

    # Panel 1: Loss
    ax1 = axes[0, 0]
    ax1.set_facecolor(LIGHT_BG)
    ax1.plot(epochs, df['train_loss'], 'o-', color=BLUE, linewidth=3.0, markersize=8, label='Training Loss')
    ax1.plot(epochs, df['val_loss'], 's--', color=CORAL, linewidth=2.8, markersize=8, label='Validation Loss')
    ax1.set_title("Loss Trajectory (Dice + Cross-Entropy)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax1.set_xlabel("Epoch", fontsize=14, fontweight='bold', color=SLATE)
    ax1.set_ylabel("Loss Value", fontsize=14, fontweight='bold', color=SLATE)
    ax1.set_xticks(epochs)
    ax1.tick_params(labelsize=12)
    ax1.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax1.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax1.annotate(f"Final Train Loss: {df['train_loss'].iloc[-1]:.4f}\nFinal Val Loss: {df['val_loss'].iloc[-1]:.4f}",
                 xy=(15, df['train_loss'].iloc[-1]), xytext=(10.5, 0.32),
                 fontsize=12, fontweight='bold', color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=BLUE, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=BLUE, lw=2.0))

    # Panel 2: Mean Dice & IoU
    ax2 = axes[0, 1]
    ax2.set_facecolor(LIGHT_BG)
    ax2.plot(epochs, df['mean_dice'], 'o-', color=GREEN, linewidth=3.2, markersize=9, label='Validation Mean Dice')
    ax2.plot(epochs, df['mean_iou'], '^-', color=PURPLE, linewidth=2.8, markersize=8, label='Validation Mean IoU')
    ax2.set_title("Validation Segmentation Accuracy", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax2.set_xlabel("Epoch", fontsize=14, fontweight='bold', color=SLATE)
    ax2.set_ylabel("Score", fontsize=14, fontweight='bold', color=SLATE)
    ax2.set_xticks(epochs)
    ax2.set_ylim(0.50, 0.90)
    ax2.tick_params(labelsize=12)
    ax2.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax2.legend(fontsize=13, loc='lower right', framealpha=0.95)
    best_dice = df['mean_dice'].iloc[-1]
    best_iou = df['mean_iou'].iloc[-1]
    ax2.plot(15, best_dice, '*', color=AMBER, markersize=20, markeredgecolor=NAVY, markeredgewidth=1.5, zorder=10)
    ax2.annotate(f"BEST CHECKPOINT (Epoch 15)\nMean Dice: {best_dice:.4f} (86.02%)\nMean IoU: {best_iou:.4f} (75.68%)",
                 xy=(15, best_dice), xytext=(8.0, 0.58),
                 fontsize=12.5, fontweight='bold', color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#FEF3C7", ec=AMBER, lw=2.0),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.5))

    # Panel 3: Learning Rate
    ax3 = axes[1, 0]
    ax3.set_facecolor(LIGHT_BG)
    lr_vals = df['learning_rate_primary'].values
    ax3.plot(epochs, lr_vals, 'd-', color=AMBER, linewidth=3.0, markersize=8)
    ax3.set_title("Step Learning Rate Schedule (Warmup + Cosine Decay)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax3.set_xlabel("Epoch", fontsize=14, fontweight='bold', color=SLATE)
    ax3.set_ylabel("Learning Rate (LR)", fontsize=14, fontweight='bold', color=SLATE)
    ax3.set_xticks(epochs)
    ax3.tick_params(labelsize=12)
    ax3.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax3.annotate("2-Epoch Linear Warmup\n(1e-4 -> 1e-3)", xy=(2, lr_vals[1]), xytext=(2.5, 0.0007),
                 fontsize=12, fontweight='bold', color=SLATE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=AMBER, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.0))
    ax3.annotate(f"Cosine Floor: 2.44e-5\n(Epoch 15)", xy=(15, lr_vals[-1]), xytext=(10.5, 0.00025),
                 fontsize=12, fontweight='bold', color=SLATE,
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FFFFFF", ec=AMBER, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=AMBER, lw=2.0))

    # Panel 4: Gradient Norm & Clipping
    ax4 = axes[1, 1]
    ax4.set_facecolor(LIGHT_BG)
    ax4.plot(epochs, df['grad_norm_pre'], 'o-', color=TEAL, linewidth=2.8, markersize=8, label='Pre-Clip L2 Norm')
    ax4.plot(epochs, df['grad_norm_post'], 's-', color=SLATE, linewidth=2.4, markersize=7, label='Post-Clip L2 Norm (Max 1.0)')
    ax4.set_title("Gradient Norm Stability & Clip Enforcement", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax4.set_xlabel("Epoch", fontsize=14, fontweight='bold', color=SLATE)
    ax4.set_ylabel("L2 Gradient Norm", fontsize=14, fontweight='bold', color=SLATE)
    ax4.set_xticks(epochs)
    ax4.tick_params(labelsize=12)
    ax4.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax4.legend(fontsize=13, loc='upper right', framealpha=0.95)
    ax4.annotate("Initial stabilization: 52.0% clipped\nEpochs 5-15: Stable at ~2.2% clipped",
                 xy=(7, 0.25), xytext=(5.5, 0.85),
                 fontsize=12, fontweight='bold', color=NAVY,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec=TEAL, lw=1.5),
                 arrowprops=dict(arrowstyle="->", color=TEAL, lw=2.0))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_02_unetpp_training_trajectory.png")


# -----------------------------------------------------------------------------
# FIG 3: SUB-COMPARTMENT ACCURACY EVOLUTION
# -----------------------------------------------------------------------------
def gen_03_compartment_dice():
    history_path = WORKSPACE / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "history.csv"
    df = pd.read_csv(history_path)
    epochs = df['epoch'].values

    fig, ax = plt.subplots(figsize=(16, 9), facecolor='#FFFFFF')
    ax.set_facecolor(LIGHT_BG)

    ax.plot(epochs, df['dice_class_0'], 'o-', color=SLATE, linewidth=2.6, markersize=7, label='Class 0: Healthy Background (BG)')
    ax.plot(epochs, df['dice_class_3'], 's-', color=CORAL, linewidth=3.4, markersize=9, label='Class 3: Enhancing Tumor (ET)')
    ax.plot(epochs, df['dice_class_2'], '^-', color=GREEN, linewidth=3.2, markersize=9, label='Class 2: Peritumoral Edema (ED)')
    ax.plot(epochs, df['dice_class_1'], 'd-', color=BLUE, linewidth=3.0, markersize=8, label='Class 1: Necrotic Tumor Core (NCR)')

    ax.set_title("U-Net++: Tumor Sub-Compartment Validation Dice Evolution (15 Epochs)",
                 fontsize=22, fontweight='bold', color=NAVY, pad=18)
    ax.set_xlabel("Training Epoch", fontsize=16, fontweight='bold', color=SLATE, labelpad=10)
    ax.set_ylabel("Validation Dice Coefficient", fontsize=16, fontweight='bold', color=SLATE, labelpad=10)
    ax.set_xticks(epochs)
    ax.set_ylim(0.60, 1.02)
    ax.tick_params(labelsize=14)
    ax.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.85)
    ax.legend(fontsize=14, loc='lower right', framealpha=0.95, facecolor='#FFFFFF')

    # Value Callout badges at Epoch 15
    final_metrics = [
        ("Background (BG)", df['dice_class_0'].iloc[-1], SLATE, 0.9986, 0.9972),
        ("Enhancing Tumor (ET)", df['dice_class_3'].iloc[-1], CORAL, 0.9047, 0.8260),
        ("Peritumoral Edema (ED)", df['dice_class_2'].iloc[-1], GREEN, 0.8664, 0.7642),
        ("Necrotic Core (NCR)", df['dice_class_1'].iloc[-1], BLUE, 0.8096, 0.6801),
    ]

    for name, val, col, d_val, i_val in final_metrics:
        ax.plot(15, val, 'o', color=col, markersize=12, markeredgecolor=NAVY, markeredgewidth=1.8, zorder=10)
        ax.annotate(f"{name}\nDice: {d_val:.4f} | IoU: {i_val:.4f}",
                    xy=(15, val), xytext=(15.3, val),
                    fontsize=12, fontweight='bold', color=col, va='center',
                    bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec=col, lw=1.8))

    # Add clinical explanation card at bottom left
    card_text = (
        "CLINICAL INTERPRETATION OF SUB-COMPARTMENTS:\n"
        "• Enhancing Tumor (ET: 0.9047) — Active neovascular tumor margin; critical for surgical resection planning.\n"
        "• Peritumoral Edema (ED: 0.8664) — Infiltrative fluid zone; critical for radiation therapy field boundaries.\n"
        "• Necrotic Core (NCR: 0.8096) — Non-enhancing avascular core; marks tumor necrosis & hypoxia.\n"
        "• Whole Tumor (WT = ET + ED + NCR) & Tumor Core (TC = ET + NCR) synthesized from these predictions."
    )
    ax.text(0.02, 0.04, card_text, transform=ax.transAxes,
            fontsize=12, fontweight='semibold', color=NAVY,
            bbox=dict(boxstyle="round,pad=0.6", fc="#EFF6FF", ec=BLUE, lw=1.8))

    ax.set_xlim(0.8, 18.5)
    plt.tight_layout()
    save_fig(fig, "fig_03_unetpp_compartment_dice_evolution.png")


# -----------------------------------------------------------------------------
# FIG 4: PROBABILITY CALIBRATION & RELIABILITY DIAGRAM
# -----------------------------------------------------------------------------
def gen_04_calibration_reliability():
    eval_json_path = WORKSPACE / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "evaluation_metrics" / "unetpp_eval_metrics.json"
    with open(eval_json_path, 'r') as f:
        data = json.load(f)

    uncal = data['uncalibrated']
    cal = data['calibrated']
    T = data['optimal_temperature_T']

    fig, axes = plt.subplots(1, 2, figsize=(18, 9), facecolor='#FFFFFF')
    fig.suptitle(f"U-Net++: Post-Hoc Probability Calibration via Temperature Scaling (T* = {T:.4f})",
                 fontsize=22, fontweight='bold', color=NAVY, y=0.98)

    bins = np.linspace(0, 1, 16)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Panel 1: Uncalibrated (T=1.0)
    ax1 = axes[0]
    ax1.set_facecolor(LIGHT_BG)
    ax1.plot([0, 1], [0, 1], '--', color=SLATE, linewidth=2.5, label='Perfect Calibration (y = x)')
    ax1.plot(uncal['bin_confidences'], uncal['bin_accuracies'], 's-', color=CORAL, linewidth=3.2,
             markersize=9, label=f"Uncalibrated (T=1.00)\nECE: {uncal['ece']*100:.2f}%\nNLL: {uncal['nll']:.4f}")
    ax1.fill_between(uncal['bin_confidences'], uncal['bin_accuracies'], uncal['bin_confidences'],
                     color=CORAL, alpha=0.18, label='Calibration Error Gap')
    ax1.set_title("Uncalibrated Predictions (Systemic Overconfidence)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax1.set_xlabel("Mean Predicted Confidence", fontsize=14, fontweight='bold', color=SLATE)
    ax1.set_ylabel("Empirical True Accuracy", fontsize=14, fontweight='bold', color=SLATE)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.tick_params(labelsize=13)
    ax1.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax1.legend(fontsize=13, loc='upper left', framealpha=0.95)

    # Panel 2: Calibrated (T = T*)
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT_BG)
    ax2.plot([0, 1], [0, 1], '--', color=SLATE, linewidth=2.5, label='Perfect Calibration (y = x)')
    ax2.plot(cal['bin_confidences'], cal['bin_accuracies'], 'o-', color=GREEN, linewidth=3.2,
             markersize=9, label=f"Calibrated (T* = {T:.4f})\nECE: {cal['ece']*100:.2f}%\nNLL: {cal['nll']:.4f}")
    ax2.fill_between(cal['bin_confidences'], cal['bin_accuracies'], cal['bin_confidences'],
                     color=GREEN, alpha=0.18, label='Residual Error Gap')
    ax2.set_title(f"Calibrated Predictions (ECE Reduced by -50.70%)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax2.set_xlabel("Mean Predicted Confidence", fontsize=14, fontweight='bold', color=SLATE)
    ax2.set_ylabel("Empirical True Accuracy", fontsize=14, fontweight='bold', color=SLATE)
    ax2.set_xlim(-0.02, 1.02)
    ax2.set_ylim(-0.02, 1.02)
    ax2.tick_params(labelsize=13)
    ax2.grid(True, linestyle='--', color=GRID_COLOR, alpha=0.8)
    ax2.legend(fontsize=13, loc='upper left', framealpha=0.95)

    # Inset Summary Comparison Card across bottom of figure
    summary_card = (
        f"QUANTITATIVE CALIBRATION GAINS ON HELD-OUT VAL_EVAL COHORT (105 CASES, 311,546 VOXELS):\n"
        f"• Expected Calibration Error (ECE):  {uncal['ece']*100:.2f}%  →  {cal['ece']*100:.2f}%   (-50.70% Relative Reduction)\n"
        f"• Negative Log-Likelihood (NLL):     {uncal['nll']:.4f}  →  {cal['nll']:.4f}   (-51.64% Relative Reduction)\n"
        f"• Brier Score:                       {uncal['brier_score']:.4f}  →  {cal['brier_score']:.4f}   (-14.98% Relative Reduction)\n"
        f"• Discrete Argmax Segmentation Invariance: EXACTLY 0 MISMATCHES across all 937,440,000 voxel predictions"
    )
    fig.text(0.5, 0.04, summary_card, ha='center', va='center', fontsize=13, fontweight='semibold', color=NAVY,
             bbox=dict(boxstyle="round,pad=0.7", fc="#F0FDF4", ec=GREEN, lw=2.0))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    save_fig(fig, "fig_04_unetpp_calibration_reliability.png")


# -----------------------------------------------------------------------------
# FIG 5: 5-MODEL BENCHMARK MATRIX (COMPARATIVE RADAR & BARS)
# -----------------------------------------------------------------------------
def gen_05_benchmark_matrix():
    eval_csv_path = WORKSPACE / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "evaluation_metrics.csv"
    df = pd.read_csv(eval_csv_path)

    models = df['model'].tolist()
    params_m = (df['parameters'] / 1e6).tolist()
    val_eval_dice = df['mean_volumetric_dice'].tolist()
    cal_ece = (df['val_eval_cal_ece'] * 100).tolist()
    ece_red = [float(x.replace('%', '')) for x in df['ece_reduction_pct'].tolist()]

    fig, axes = plt.subplots(2, 2, figsize=(17, 11), facecolor='#FFFFFF')
    fig.suptitle("Stage C Phase 2: Comprehensive 5-Model Comparative Benchmark Matrix",
                 fontsize=22, fontweight='bold', color=NAVY, y=0.98)

    # Color mapping highlighting U-Net++
    colors = [CYAN if m != 'U-Net++' else AMBER for m in models]
    edge_colors = [NAVY if m != 'U-Net++' else CORAL for m in models]
    linewidths = [1.5 if m != 'U-Net++' else 2.5 for m in models]

    # Panel 1: Parameter Count
    ax1 = axes[0, 0]
    ax1.set_facecolor(LIGHT_BG)
    bars1 = ax1.bar(models, params_m, color=colors, edgecolor=edge_colors, linewidth=linewidths)
    ax1.set_title("Trainable Parameters (Millions)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax1.set_ylabel("Parameters (M)", fontsize=14, fontweight='bold', color=SLATE)
    ax1.tick_params(axis='x', rotation=15, labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)
    ax1.grid(True, axis='y', linestyle='--', color=GRID_COLOR)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.2f} M",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color=NAVY)

    # Panel 2: Volumetric Mean Dice
    ax2 = axes[0, 1]
    ax2.set_facecolor(LIGHT_BG)
    bars2 = ax2.bar(models, val_eval_dice, color=colors, edgecolor=edge_colors, linewidth=linewidths)
    ax2.set_title("Volumetric Mean Dice (105 Evaluation Cases)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax2.set_ylabel("Macro Mean Dice", fontsize=14, fontweight='bold', color=SLATE)
    ax2.set_ylim(0.80, 0.89)
    ax2.tick_params(axis='x', rotation=15, labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)
    ax2.grid(True, axis='y', linestyle='--', color=GRID_COLOR)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.0015, f"{yval:.4f}",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color=NAVY)

    # Panel 3: Calibrated ECE
    ax3 = axes[1, 0]
    ax3.set_facecolor(LIGHT_BG)
    bars3 = ax3.bar(models, cal_ece, color=colors, edgecolor=edge_colors, linewidth=linewidths)
    ax3.set_title("Calibrated ECE (%) [Lower is Better]", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax3.set_ylabel("Calibrated ECE (%)", fontsize=14, fontweight='bold', color=SLATE)
    ax3.set_ylim(0, 7.0)
    ax3.tick_params(axis='x', rotation=15, labelsize=12)
    ax3.tick_params(axis='y', labelsize=12)
    ax3.grid(True, axis='y', linestyle='--', color=GRID_COLOR)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 0.15, f"{yval:.2f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color=NAVY)

    # Panel 4: Relative ECE Reduction
    ax4 = axes[1, 1]
    ax4.set_facecolor(LIGHT_BG)
    bars4 = ax4.bar(models, ece_red, color=colors, edgecolor=edge_colors, linewidth=linewidths)
    ax4.set_title("Relative ECE Reduction after Calibration (%)", fontsize=16, fontweight='bold', color=NAVY, pad=12)
    ax4.set_ylabel("ECE Reduction (%)", fontsize=14, fontweight='bold', color=SLATE)
    ax4.set_ylim(0, 70.0)
    ax4.tick_params(axis='x', rotation=15, labelsize=12)
    ax4.tick_params(axis='y', labelsize=12)
    ax4.grid(True, axis='y', linestyle='--', color=GRID_COLOR)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"-{yval:.1f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color=NAVY)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "fig_05_five_model_benchmark_matrix.png")


# -----------------------------------------------------------------------------
# FIG 6: PREDICTIVE UNCERTAINTY & SHANNON ENTROPY DISTRIBUTION
# -----------------------------------------------------------------------------
def gen_06_predictive_entropy():
    eval_json_path = WORKSPACE / "results" / "stage_c" / "phase2" / "c2_calibration_v1" / "evaluation_metrics" / "unetpp_eval_metrics.json"
    with open(eval_json_path, 'r') as f:
        data = json.load(f)

    ent = data['predictive_entropy']

    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5), facecolor='#FFFFFF')
    fig.suptitle("U-Net++: Predictive Uncertainty (Shannon Entropy) & Slice Accounting",
                 fontsize=22, fontweight='bold', color=NAVY, y=0.98)

    # Left: Donut Chart of Slice Population
    ax1 = axes[0]
    ax1.set_facecolor('#FFFFFF')
    labels = [
        f"Empty Prediction Slices\n({ent['empty_slices']:,} slices | {ent['empty_slice_pct']:.1f}%)\n[Entropy = None / null]",
        f"Non-Empty Tumor Slices\n({ent['non_empty_slices']:,} slices | {100 - ent['empty_slice_pct']:.1f}%)\n[Active Foreground Entropy]"
    ]
    sizes = [ent['empty_slices'], ent['non_empty_slices']]
    pie_colors = [SLATE, BLUE]
    explode = (0.04, 0)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                                       startangle=140, colors=pie_colors,
                                       textprops=dict(color=NAVY, fontsize=13, fontweight='bold'),
                                       wedgeprops=dict(width=0.45, edgecolor=NAVY, linewidth=1.8))
    for at in autotexts:
        at.set_fontsize(14)
        at.set_color('#FFFFFF')
        at.set_weight('bold')
    ax1.set_title(f"Total Evaluated Slices: {ent['total_slices']:,} (105 Cases × 155 Slices)",
                  fontsize=16, fontweight='bold', color=NAVY, pad=14)

    # Right: Metric Distribution Box / Summary
    ax2 = axes[1]
    ax2.set_facecolor(LIGHT_BG)
    ax2.axis('off')

    # Draw Card for entropy statistics
    rect = patches.FancyBboxPatch((0.05, 0.08), 0.90, 0.84, boxstyle="round,pad=0.03",
                                  facecolor='#FFFFFF', edgecolor=BLUE, linewidth=2.5)
    ax2.add_patch(rect)

    ax2.text(0.5, 0.85, "CALIBRATED FOREGROUND PREDICTIVE ENTROPY STATS",
             ha='center', va='center', fontsize=16, fontweight='bold', color=NAVY)
    ax2.text(0.5, 0.80, "(Computed via Normalized 4-Class Shannon Entropy: H in [0, 1])",
             ha='center', va='center', fontsize=13, fontweight='medium', color=SLATE)

    metrics_text = [
        ("Evaluated Tumor Slices:", f"{ent['non_empty_slices']:,} slices"),
        ("Mean Foreground Entropy (μ):", f"{ent['mean']:.4f} ± {ent['std']:.4f}"),
        ("Median Predictive Entropy:", f"{ent['median']:.4f}"),
        ("Interquartile Range [p25, p75]:", f"[{ent['p25']:.4f}, {ent['p75']:.4f}]"),
        ("Full Observed Range [Min, Max]:", f"[{ent['min']:.4f}, {ent['max']:.4f}]"),
        ("Stage C Triage Operating Gate:", f"τ = 0.35 (Retention Threshold)"),
        ("Discrete Argmax Mismatches:", f"0 (100% Invariance Verified)")
    ]

    start_y = 0.68
    for label, val in metrics_text:
        ax2.text(0.12, start_y, label, fontsize=14, fontweight='bold', color=SLATE, va='center')
        ax2.text(0.88, start_y, val, fontsize=14, fontweight='bold', color=NAVY, ha='right', va='center')
        start_y -= 0.075

    # Bottom alert block
    alert_box = patches.FancyBboxPatch((0.10, 0.12), 0.80, 0.10, boxstyle="round,pad=0.015",
                                       facecolor="#EFF6FF", edgecolor=BLUE, linewidth=1.5)
    ax2.add_patch(alert_box)
    ax2.text(0.5, 0.17, "CRITICAL SAFETY ENFORCEMENT: Empty slices strictly record None/null;\nzero instances of false 0.0 entropy substitution occurred.",
             ha='center', va='center', fontsize=12, fontweight='bold', color=BLUE)

    plt.tight_layout()
    save_fig(fig, "fig_06_unetpp_predictive_entropy_distribution.png")


# -----------------------------------------------------------------------------
# FIG 7: EXECUTIVE DASHBOARD POSTER (A3 / FAR-DISTANCE VISIBILITY)
# -----------------------------------------------------------------------------
def gen_07_executive_poster():
    fig = plt.figure(figsize=(20, 11.25), facecolor='#FFFFFF')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(LIGHT_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Top Header Banner
    header_rect = patches.Rectangle((0, 0.88), 1.0, 0.12, facecolor=NAVY)
    ax.add_patch(header_rect)
    ax.text(0.04, 0.955, "MODEL PROFILE & BENCHMARK DOSSIER: U-NET++ (2D NESTED DENSE U-NET)",
            fontsize=26, fontweight='bold', color='#FFFFFF', va='center')
    ax.text(0.04, 0.915, "Multi-Parametric MRI Brain Tumor Segmentation | BraTS 2023 Cohort | Stage C Full Benchmark Suite",
            fontsize=16, fontweight='semibold', color=CYAN, va='center')
    ax.text(0.96, 0.935, "EXECUTIVE POSTER",
            fontsize=20, fontweight='bold', color='#FFFFFF', ha='right', va='center',
            bbox=dict(boxstyle="round,pad=0.4", fc=CORAL, ec='none'))

    # Helper function to draw KPI cards
    def draw_kpi(x, y, w, h, title, value, subtitle, accent_col):
        card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                      facecolor='#FFFFFF', edgecolor=GRID_COLOR, linewidth=2.0)
        ax.add_patch(card)
        bar = patches.Rectangle((x, y + h - 0.012), w, 0.012, facecolor=accent_col)
        ax.add_patch(bar)
        ax.text(x + w/2, y + h - 0.035, title, ha='center', va='center',
                fontsize=13, fontweight='bold', color=SLATE)
        ax.text(x + w/2, y + h*0.48, value, ha='center', va='center',
                fontsize=28, fontweight='bold', color=accent_col)
        ax.text(x + w/2, y + 0.03, subtitle, ha='center', va='center',
                fontsize=11.5, fontweight='semibold', color=SLATE)

    # 4 Top KPI Cards
    draw_kpi(0.04, 0.72, 0.21, 0.14, "SELECTION VAL MEAN DICE", "0.8602", "Epoch 15 Peak (721 Train / 155 Val)", GREEN)
    draw_kpi(0.28, 0.72, 0.21, 0.14, "ENHANCING TUMOR (ET) DICE", "0.9047", "Top Segmentation Sub-Compartment", CORAL)
    draw_kpi(0.52, 0.72, 0.21, 0.14, "CALIBRATION REDUCTION", "-50.70%", f"ECE: 11.18% → 5.51% (T* = 2.5647)", BLUE)
    draw_kpi(0.76, 0.72, 0.20, 0.14, "TOTAL TRAINABLE PARAMS", "9.16 M", "9,160,068 (Nested Dense Skips)", PURPLE)

    # 3 Major Content Blocks below
    # Block A: Architecture & Training Specs (Left)
    card_a = patches.FancyBboxPatch((0.04, 0.05), 0.29, 0.64, boxstyle="round,pad=0.015",
                                    facecolor='#FFFFFF', edgecolor=GRID_COLOR, linewidth=2.0)
    ax.add_patch(card_a)
    ax.text(0.06, 0.65, "1. ARCHITECTURAL SPECIFICATION", fontsize=16, fontweight='bold', color=NAVY)
    
    specs = [
        ("Model Family:", "UNetPlusPlus2D (Nested U-Net)"),
        ("Nested Skips:", "Levels 0 to 4 (X^{i,j} blocks)"),
        ("Deep Supervision:", "Disabled (Standardized Stage B)"),
        ("Input Modalities:", "4 Channels (FLAIR, T1, T1ce, T2)"),
        ("Input Grid:", "240 × 240 Axial Slices"),
        ("Output Classes:", "4 (BG, NCR, ED, ET)"),
        ("Training Cohort:", "721 Cases / 667 Unique Subjects"),
        ("Validation Pool:", "155 Cases / 143 Unique Subjects"),
        ("Training Slices:", "59,020 (47,228 fg + 11,792 bg)"),
        ("Epoch Budget:", "15 Epochs Complete"),
        ("Execution Mode:", "Synchronous CUDA (Zero Crashes)"),
        ("Process Peak VRAM:", "1,111.3 MB (RTX 4070 8GB)"),
        ("Total Runtime:", "15.44 Hours (926.39 Minutes)"),
        ("Checkpoint Hash:", "fbd237a42bdf12f9... (Intact)")
    ]
    sy = 0.60
    for k, v in specs:
        ax.text(0.06, sy, k, fontsize=11.5, fontweight='bold', color=SLATE)
        ax.text(0.31, sy, v, fontsize=11.5, fontweight='bold', color=NAVY, ha='right')
        sy -= 0.038

    # Block B: Benchmark Segmentation Performance (Center)
    card_b = patches.FancyBboxPatch((0.36, 0.05), 0.29, 0.64, boxstyle="round,pad=0.015",
                                    facecolor='#FFFFFF', edgecolor=GRID_COLOR, linewidth=2.0)
    ax.add_patch(card_b)
    ax.text(0.38, 0.65, "2. COMPREHENSIVE BENCHMARKS", fontsize=16, fontweight='bold', color=NAVY)

    perf = [
        ("Val Mean Dice (Ep 15):", "0.8602 (0.860218)"),
        ("Val Mean IoU (Ep 15):", "0.7568 (0.756770)"),
        ("Enhancing Tumor (ET):", "0.9047 (IoU: 0.8260)"),
        ("Peritumoral Edema (ED):", "0.8664 (IoU: 0.7642)"),
        ("Necrotic Core (NCR):", "0.8096 (IoU: 0.6801)"),
        ("Healthy Background (BG):", "0.9986 (IoU: 0.9972)"),
        ("Final Training Loss:", "0.1728 (Dice + CE)"),
        ("Final Validation Loss:", "0.4280"),
        ("Held-out Val_eval Vol Dice:", "0.8549 (105 Cases)"),
        ("Whole Tumor (WT) Dice:", "0.8830 (Val_eval)"),
        ("Tumor Core (TC) Dice:", "0.8521 (Val_eval)"),
        ("ET Volumetric Dice:", "0.8297 (Val_eval)"),
        ("5-Epoch Baseline Dice:", "0.7937 (Historical Stage B)")
    ]
    sy = 0.60
    for k, v in perf:
        ax.text(0.38, sy, k, fontsize=11.5, fontweight='bold', color=SLATE)
        ax.text(0.63, sy, v, fontsize=11.5, fontweight='bold', color=NAVY, ha='right')
        sy -= 0.041

    # Block C: Calibration & Reliability Agent Safety (Right)
    card_c = patches.FancyBboxPatch((0.68, 0.05), 0.28, 0.64, boxstyle="round,pad=0.015",
                                    facecolor='#FFFFFF', edgecolor=GRID_COLOR, linewidth=2.0)
    ax.add_patch(card_c)
    ax.text(0.70, 0.65, "3. CALIBRATION & TRIAGE SAFETY", fontsize=16, fontweight='bold', color=NAVY)

    calib = [
        ("Learned Temperature (T*):", "2.5647 (L-BFGS, Val_cal)"),
        ("Uncalibrated ECE:", "11.18% (Overconfident)"),
        ("Calibrated ECE:", "5.51% (-50.70% Relative)"),
        ("Uncalibrated NLL:", "0.8074"),
        ("Calibrated NLL:", "0.3904 (-51.64% Relative)"),
        ("Calibrated Brier Score:", "0.1983 (-14.98% Relative)"),
        ("Argmax Mismatches:", "EXACTLY 0 (100% Invariant)"),
        ("Evaluated Slices:", "16,275 (105 Cases × 155 Slices)"),
        ("Empty Slices (None):", "10,147 (62.35%)"),
        ("Non-Empty Slices:", "6,128 (37.65%)"),
        ("Mean Foreground Entropy:", "0.2883 ± 0.0907"),
        ("Median Entropy:", "0.2685 (IQR: [0.22, 0.34])"),
        ("Operating Gate:", "τ = 0.35 (Retention Threshold)"),
        ("Methodological Status:", "Internal Selection-Exposed")
    ]
    sy = 0.60
    for k, v in calib:
        ax.text(0.70, sy, k, fontsize=11.5, fontweight='bold', color=SLATE)
        ax.text(0.94, sy, v, fontsize=11.5, fontweight='bold', color=NAVY, ha='right')
        sy -= 0.038

    save_fig(fig, "fig_07_unetpp_executive_poster_dashboard.png")
    # Also save directly in documents root as printable poster
    REPORTS_ONEPAGE_DIR = WORKSPACE / "reports" / "model_profiles" / "unetpp" / "one_page"
    REPORTS_ONEPAGE_DIR.mkdir(parents=True, exist_ok=True)
    poster_png = REPORTS_ONEPAGE_DIR / "UNetPlusPlus_Executive_Summary_Printout.png"
    poster_pdf = REPORTS_ONEPAGE_DIR / "UNetPlusPlus_Executive_Summary_Printout.pdf"
    fig.savefig(poster_png, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    fig.savefig(poster_pdf, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    print("[✓] Saved executive printout poster (PNG & PDF) to reports/model_profiles/unetpp/one_page/")


def main():
    print("Generating U-Net++ Scientific and Executive Figures...")
    gen_01_architecture()
    gen_02_trajectory()
    gen_03_compartment_dice()
    gen_04_calibration_reliability()
    gen_05_benchmark_matrix()
    gen_06_predictive_entropy()
    gen_07_executive_poster()
    print("All 7 U-Net++ figures successfully generated at 300 DPI!")

if __name__ == "__main__":
    main()
