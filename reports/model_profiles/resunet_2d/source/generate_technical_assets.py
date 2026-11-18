"""Generate 25 Technical and Scientific Visual Assets for ResUNet-2D.

Workspace: D:\MajorProject
Output: reports/model_profiles/resunet_2d/assets/technical/
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

TECH_DIR = r'D:\MajorProject\reports\model_profiles\resunet_2d\assets\technical'
os.makedirs(TECH_DIR, exist_ok=True)

# Color Palette
NAVY = '#1F3A60'
SLATE = '#475569'
TEAL = '#0D9488'
CYAN = '#0284C7'
CORAL = '#E11D48'
AMBER = '#D97706'
PURPLE = '#7C3AED'
LIGHT_BG = '#F8FAFC'
GRID_COLOR = '#E2E8F0'
WHITE = '#FFFFFF'
DARK = '#0F172A'


# -----------------------------------------------------------------------------
# 01. Full Architecture
# -----------------------------------------------------------------------------
def gen_01_architecture():
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.96, "ResUNet-2D: Complete 5-Stage Deep Residual U-Net Architecture",
            ha='center', va='center', fontsize=22, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.92, "Input: 4-Channel mpMRI (240×240) | Output: 4-Class Logits | Total Trainable Parameters: 8,114,604",
            ha='center', va='center', fontsize=14, color=SLATE, transform=ax.transAxes)

    def draw_card(x, y, w, h, title, subtitle, color, text_color=WHITE, edge_color=None):
        if edge_color is None:
            edge_color = color
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                      facecolor=color, edgecolor=edge_color, linewidth=1.8, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.64, title, ha='center', va='center', fontsize=12, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=9.5, color=text_color, zorder=4)

    # Input Block
    draw_card(0.02, 0.72, 0.12, 0.13, "mpMRI Input", "4 Channels @ 240×240\n[T1n, T1c, T2w, T2f]", '#334155')

    # Encoder Stages (Left)
    draw_card(0.17, 0.72, 0.16, 0.13, "Encoder Level 1", "PreAct ResBlock (32 ch)\nStride 1 (240×240)\n10,632 params (0.13%)", TEAL)
    draw_card(0.17, 0.54, 0.16, 0.13, "Encoder Level 2", "PreAct ResBlock (64 ch)\nStride 2 (120×120)\n57,664 params (0.71%)", TEAL)
    draw_card(0.17, 0.36, 0.16, 0.13, "Encoder Level 3", "PreAct ResBlock (128 ch)\nStride 2 (60×60)\n230,016 params (2.83%)", TEAL)
    draw_card(0.17, 0.18, 0.16, 0.13, "Encoder Level 4", "PreAct ResBlock (256 ch)\nStride 2 (30×30)\n918,784 params (11.32%)", TEAL)

    # Bottleneck
    draw_card(0.40, 0.04, 0.20, 0.13, "Bottleneck (Level 5)", "PreAct ResBlock (512 ch)\nStride 2 (15×15)\n3,672,576 params (45.26%)", NAVY)

    # Decoder Stages (Right)
    draw_card(0.67, 0.18, 0.17, 0.13, "Decoder Level 4", "Upconv 2×2 (512→256)\nConcat Enc4 (512 ch) → ResBlock\n2,427,136 params (29.91%)", CYAN)
    draw_card(0.67, 0.36, 0.17, 0.13, "Decoder Level 3", "Upconv 2×2 (256→128)\nConcat Enc3 (256 ch) → ResBlock\n607,360 params (7.49%)", CYAN)
    draw_card(0.67, 0.54, 0.17, 0.13, "Decoder Level 2", "Upconv 2×2 (128→64)\nConcat Enc2 (128 ch) → ResBlock\n152,128 params (1.87%)", CYAN)
    draw_card(0.67, 0.72, 0.17, 0.13, "Decoder Level 1", "Upconv 2×2 (64→32)\nConcat Enc1 (64 ch) → ResBlock\n38,176 params (0.47%)", CYAN)

    # Output Classifier
    draw_card(0.87, 0.72, 0.11, 0.13, "1×1 Classifier", "Conv2d (32→4)\nLogits @ 240×240\n132 params (<0.01%)", DARK)

    # Connections & Arrows
    arr = dict(facecolor=SLATE, edgecolor=SLATE, width=2.0, headwidth=7, shrink=0.08)
    ax.annotate('', xy=(0.17, 0.785), xytext=(0.14, 0.785), arrowprops=arr)
    ax.annotate('', xy=(0.25, 0.67), xytext=(0.25, 0.72), arrowprops=arr)
    ax.annotate('', xy=(0.25, 0.49), xytext=(0.25, 0.54), arrowprops=arr)
    ax.annotate('', xy=(0.25, 0.31), xytext=(0.25, 0.36), arrowprops=arr)
    ax.annotate('', xy=(0.40, 0.105), xytext=(0.33, 0.18), arrowprops=arr)

    ax.annotate('', xy=(0.67, 0.22), xytext=(0.60, 0.105), arrowprops=arr)
    ax.annotate('', xy=(0.755, 0.36), xytext=(0.755, 0.31), arrowprops=arr)
    ax.annotate('', xy=(0.755, 0.54), xytext=(0.755, 0.49), arrowprops=arr)
    ax.annotate('', xy=(0.755, 0.72), xytext=(0.755, 0.67), arrowprops=arr)
    ax.annotate('', xy=(0.87, 0.785), xytext=(0.84, 0.785), arrowprops=arr)

    # Skip Connections
    skip = dict(facecolor=AMBER, edgecolor=AMBER, width=2.2, headwidth=8, linestyle='--')
    for y, label in [(0.785, "Long Skip 1 (240×240)"), (0.605, "Long Skip 2 (120×120)"),
                     (0.425, "Long Skip 3 (60×60)"), (0.245, "Long Skip 4 (30×30)")]:
        ax.annotate('', xy=(0.67, y), xytext=(0.33, y), arrowprops=skip)
        ax.text(0.50, y + 0.018, label, ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=AMBER)

    # Callout card
    box = patches.FancyBboxPatch((0.36, 0.35), 0.28, 0.22, boxstyle="round,pad=0.015",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.5, zorder=5)
    ax.add_patch(box)
    ax.text(0.50, 0.53, "Key Architectural Innovations", ha='center', va='center', fontsize=12, fontweight='bold', color=NAVY, zorder=6)
    points = (
        "• PreActResidualBlock: F(x) = GN8 → ReLU → Conv3x3\n"
        "  → GN8 → ReLU → Conv3x3; Shortcut: Conv1x1 or Id.\n"
        "• Unobstructed Gradient Highway via pre-activation.\n"
        "• Group Normalization (G=8): batch-size invariant.\n"
        "• Stride-1 Level 1 preserves enhancing tumor margins."
    )
    ax.text(0.38, 0.44, points, ha='left', va='center', fontsize=9.5, color=SLATE, zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '01_resunet_full_architecture.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 01_resunet_full_architecture.png')


# -----------------------------------------------------------------------------
# 02. Residual Block Explained
# -----------------------------------------------------------------------------
def gen_02_residual_block():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.95, "Pre-Activation Residual Block Topology (PreActResidualBlock)",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.90, "Topology: Norm1 → ReLU → Conv1 → Norm2 → ReLU → Conv2 with Direct Residual Addition",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    def draw_node(x, y, w, h, title, subtitle, color, tc=WHITE):
        r = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                   facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3)
        ax.add_patch(r)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=11.5, fontweight='bold', color=tc, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=9.5, color=tc, zorder=4)

    # Input x
    draw_node(0.42, 0.78, 0.16, 0.08, "Input Tensor x", "[B, C_in, H, W]", '#334155')

    # Main branch
    draw_node(0.20, 0.65, 0.18, 0.07, "GroupNorm 1", "G=8, C=C_in", TEAL)
    draw_node(0.20, 0.55, 0.18, 0.07, "ReLU Activation", "In-place non-linearity", CYAN)
    draw_node(0.20, 0.45, 0.18, 0.07, "Conv2d (3×3)", "stride=s, padding=1, bias=F", NAVY)
    draw_node(0.20, 0.35, 0.18, 0.07, "GroupNorm 2", "G=8, C=C_out", TEAL)
    draw_node(0.20, 0.25, 0.18, 0.07, "ReLU Activation", "In-place non-linearity", CYAN)
    draw_node(0.20, 0.15, 0.18, 0.07, "Conv2d (3×3)", "stride=1, padding=1, bias=F", NAVY)

    # Shortcut branch
    draw_node(0.62, 0.45, 0.24, 0.12, "Shortcut Path", "If stride!=1 or C_in!=C_out:\nConv 1×1 (stride=s) + GN(G=8)\nElse: nn.Identity()", AMBER, tc=WHITE)

    # Addition node
    circle = patches.Circle((0.50, 0.08), 0.035, facecolor='#10B981', edgecolor=NAVY, linewidth=1.5, zorder=4)
    ax.add_patch(circle)
    ax.text(0.50, 0.08, "+", ha='center', va='center', fontsize=18, fontweight='bold', color=WHITE, zorder=5)

    # Output node
    draw_node(0.42, -0.05, 0.16, 0.06, "Output: F(x) + shortcut(x)", "[B, C_out, H/s, W/s]", '#059669')

    # Arrows
    arr = dict(facecolor=SLATE, edgecolor=SLATE, width=2.0, headwidth=7, shrink=0.05)
    ax.annotate('', xy=(0.29, 0.72), xytext=(0.42, 0.82), arrowprops=arr)
    ax.annotate('', xy=(0.74, 0.57), xytext=(0.58, 0.82), arrowprops=arr)

    ax.annotate('', xy=(0.29, 0.62), xytext=(0.29, 0.65), arrowprops=arr)
    ax.annotate('', xy=(0.29, 0.52), xytext=(0.29, 0.55), arrowprops=arr)
    ax.annotate('', xy=(0.29, 0.42), xytext=(0.29, 0.45), arrowprops=arr)
    ax.annotate('', xy=(0.29, 0.32), xytext=(0.29, 0.35), arrowprops=arr)
    ax.annotate('', xy=(0.29, 0.22), xytext=(0.29, 0.25), arrowprops=arr)

    ax.annotate('', xy=(0.465, 0.08), xytext=(0.29, 0.15), arrowprops=arr)
    ax.annotate('', xy=(0.535, 0.08), xytext=(0.74, 0.45), arrowprops=arr)
    ax.annotate('', xy=(0.50, 0.01), xytext=(0.50, 0.045), arrowprops=arr)

    ax.set_xlim(0.1, 0.9)
    ax.set_ylim(-0.08, 1.0)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '02_residual_block_explained.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 02_residual_block_explained.png')


# -----------------------------------------------------------------------------
# 03. Input to Output Pipeline
# -----------------------------------------------------------------------------
def gen_03_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.93, "End-to-End ResUNet-2D Inference & Segmentation Pipeline",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.86, "From 4-Channel Co-registered MRI to Multi-Class Segmented Tumor Margins",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    steps = [
        ("1. Co-registered MRI", "4 Modalities (240×240)\nT1n, T1c, T2w, T2f\nIntensities > 0", '#334155'),
        ("2. Preprocessing", "Z-score normalization\nwithin foreground brain mask\nConcatenate channels", TEAL),
        ("3. ResUNet-2D Forward", "5-Stage Residual U-Net\n8,114,604 parameters\nReturns 4 raw logits", NAVY),
        ("4. Temperature Scaling", "Calibrated Posterior Prob.\np_k = softmax(z_k / T*)\nT* = 2.880459", PURPLE),
        ("5. Argmax Decision", "Pixel-wise argmax(p_k)\nPreserves hard masks\n4 Discrete Classes", CYAN),
        ("6. Segmentation Mask", "Class 0: Background\nClass 1: NCR, Class 2: ED\nClass 3: ET @ 240×240", '#059669'),
    ]

    w, h, y = 0.13, 0.42, 0.32
    xs = np.linspace(0.04, 0.83, 6)

    for i, (title, sub, col) in enumerate(steps):
        rect = patches.FancyBboxPatch((xs[i], y), w, h, boxstyle="round,pad=0.015",
                                      facecolor=col, edgecolor=NAVY, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(xs[i] + w/2, y + h*0.75, title, ha='center', va='center', fontsize=11, fontweight='bold', color=WHITE, zorder=4)
        ax.text(xs[i] + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=9.5, color=WHITE, zorder=4)

        if i < 5:
            arr = dict(facecolor=AMBER, edgecolor=AMBER, width=2.5, headwidth=8, shrink=0.08)
            ax.annotate('', xy=(xs[i+1], y + h/2), xytext=(xs[i] + w, y + h/2), arrowprops=arr)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '03_input_to_output_pipeline.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 03_input_to_output_pipeline.png')


# -----------------------------------------------------------------------------
# 04. Four-Channel MRI Explainer
# -----------------------------------------------------------------------------
def gen_04_mri_channels():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=WHITE, dpi=300)

    channels = [
        ("Channel 0: T1-Native (T1n)", "Non-contrast T1-weighted\nAnatomical reference frame;\nnormal gray/white matter contrast.", '#475569'),
        ("Channel 1: T1-Contrast (T1c)", "Post-Gadolinium T1-weighted\nHighlights breakdown of BBB;\nprimary driver for Enhancing Tumor (ET).", '#0284C7'),
        ("Channel 2: T2-Weighted (T2w)", "T2-weighted spin echo\nSensitive to fluid and water;\ndelineates Whole Tumor (WT) volume.", '#0D9488'),
        ("Channel 3: T2-FLAIR (T2f)", "Fluid-Attenuated Inversion Recovery\nSuppresses free CSF signal;\nisolates peritumoral edema (ED).", '#D97706'),
    ]

    for ax, (title, desc, color) in zip(axes, channels):
        ax.set_facecolor(LIGHT_BG)
        circle = patches.Circle((0.5, 0.65), 0.25, facecolor=color, alpha=0.85, edgecolor=NAVY, linewidth=2)
        ax.add_patch(circle)
        ax.text(0.5, 0.65, title.split(':')[0], ha='center', va='center', fontsize=14, fontweight='bold', color=WHITE)

        ax.text(0.5, 0.32, title, ha='center', va='center', fontsize=12, fontweight='bold', color=NAVY)
        ax.text(0.5, 0.15, desc, ha='center', va='center', fontsize=10, color=SLATE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    plt.suptitle("Multiparametric 4-Channel MRI Input Space (BraTS-GLI Standard)", fontsize=18, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '04_four_channel_mri_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 04_four_channel_mri_explainer.png')


# -----------------------------------------------------------------------------
# 05. Training and Validation Loss
# -----------------------------------------------------------------------------
def gen_05_loss_curve():
    csv_path = r'D:\MajorProject\results\full_training_15ep\resunet_2d\history.csv'
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=WHITE, dpi=300)
    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values

    ax.plot(epochs, train_loss, marker='o', color=NAVY, linewidth=2.5, label='Training Loss (Compound Dice + CE)')
    ax.plot(epochs, val_loss, marker='s', color=CORAL, linewidth=2.5, label='Validation Loss')

    min_idx = 13  # Epoch 14
    ax.scatter([14], [val_loss[min_idx]], color=CORAL, s=140, zorder=5)
    ax.annotate(
        f'Lowest Validation Loss: {val_loss[min_idx]:.4f}\n(Epoch 14)',
        xy=(14, val_loss[min_idx]),
        xytext=(10.5, val_loss[min_idx] + 0.04),
        arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, width=1.8, headwidth=7, shrink=0.1),
        fontsize=11, fontweight='bold', color=CORAL,
        bbox=dict(boxstyle='round,pad=0.5', facecolor=WHITE, edgecolor=CORAL, linewidth=1.5)
    )

    ax.axvline(x=13, color=TEAL, linestyle=':', linewidth=1.8, alpha=0.8)
    ax.annotate(
        f'Epoch 13 (Best Dice Checkpoint)\nVal Loss: {val_loss[12]:.4f}',
        xy=(13, val_loss[12]),
        xytext=(8.0, val_loss[12] - 0.05),
        arrowprops=dict(facecolor=TEAL, edgecolor=TEAL, width=1.5, headwidth=6, shrink=0.1),
        fontsize=10.5, fontweight='bold', color=TEAL,
        bbox=dict(boxstyle='round,pad=0.4', facecolor=WHITE, edgecolor=TEAL, linewidth=1.2)
    )

    ax.set_title('ResUNet-2D Training and Validation Loss Trajectory (Epochs 1–15)', fontsize=15, fontweight='bold', color=NAVY, pad=14)
    ax.set_xlabel('Training Epoch', fontsize=12, color=SLATE)
    ax.set_ylabel('Compound Loss (Dice + Cross-Entropy)', fontsize=12, color=SLATE)
    ax.set_xticks(epochs)
    ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax.set_facecolor(LIGHT_BG)
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR, fontsize=11, loc='upper right')

    plt.savefig(os.path.join(TECH_DIR, '05_training_validation_loss.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 05_training_validation_loss.png')


# -----------------------------------------------------------------------------
# 06. Classwise Dice Progression
# -----------------------------------------------------------------------------
def gen_06_classwise_dice():
    csv_path = r'D:\MajorProject\results\full_training_15ep\resunet_2d\history.csv'
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=WHITE, dpi=300)
    epochs = df['epoch'].values
    dice_bg = df['dice_class_0'].values
    dice_ncr = df['dice_class_1'].values
    dice_ed = df['dice_class_2'].values
    dice_et = df['dice_class_3'].values

    ax.plot(epochs, dice_bg, marker='o', color='#64748B', linewidth=1.8, linestyle='--', label='Class 0: Background (BG)')
    ax.plot(epochs, dice_et, marker='^', color=CYAN, linewidth=2.4, label='Class 3: Enhancing Tumor (ET)')
    ax.plot(epochs, dice_ed, marker='s', color='#10B981', linewidth=2.4, label='Class 2: Peritumoral Edema (ED)')
    ax.plot(epochs, dice_ncr, marker='v', color=AMBER, linewidth=2.4, label='Class 1: Necrotic Core (NCR)')

    ax.axvline(x=13, color=NAVY, linestyle=':', linewidth=1.8, alpha=0.8)
    ax.annotate(
        f'Epoch 13 Class Breakdown:\n'
        f'• BG:  {dice_bg[12]:.4f}\n'
        f'• ET:  {dice_et[12]:.4f}\n'
        f'• ED:  {dice_ed[12]:.4f}\n'
        f'• NCR: {dice_ncr[12]:.4f}',
        xy=(13, 0.82),
        xytext=(9.2, 0.65),
        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.1),
        fontsize=10.5, color=NAVY, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor=WHITE, edgecolor=NAVY, linewidth=1.2)
    )

    ax.set_title('Class-wise Validation Dice Similarity Coefficient Progression (Epochs 1–15)', fontsize=15, fontweight='bold', color=NAVY, pad=14)
    ax.set_xlabel('Training Epoch', fontsize=12, color=SLATE)
    ax.set_ylabel('Validation Dice Similarity Coefficient (DSC)', fontsize=12, color=SLATE)
    ax.set_xticks(epochs)
    ax.set_ylim(0.65, 1.01)
    ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax.set_facecolor(LIGHT_BG)
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR, fontsize=11, loc='lower right')

    plt.savefig(os.path.join(TECH_DIR, '06_classwise_dice_progression.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 06_classwise_dice_progression.png')


# -----------------------------------------------------------------------------
# 07. Foreground Macro Dice Progression (Corrected Label)
# -----------------------------------------------------------------------------
def gen_07_foreground_macro_dice():
    csv_path = r'D:\MajorProject\results\full_training_15ep\resunet_2d\history.csv'
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=WHITE, dpi=300)
    epochs = df['epoch'].values
    fg_mean_dice = df['mean_dice'].values
    four_class_mean = (df['dice_class_0'] + df['dice_class_1'] + df['dice_class_2'] + df['dice_class_3']) / 4.0

    ax.plot(epochs, fg_mean_dice, marker='D', color=TEAL, linewidth=2.8, label='Foreground Macro Mean Dice (NCR, ED, ET)')
    ax.plot(epochs, four_class_mean, marker='o', color='#94A3B8', linewidth=1.8, linestyle='--', label='4-Class Macro Mean (Includes Background)')

    ax.axvline(x=13, color=NAVY, linestyle=':', linewidth=2.0)
    ax.scatter([13], [fg_mean_dice[12]], color=NAVY, s=140, zorder=5)
    ax.annotate(
        f'BEST CHECKPOINT (Epoch 13)\n'
        f'Foreground Macro Mean: {fg_mean_dice[12]:.4f}\n'
        f'(4-Class Mean: {four_class_mean[12]:.4f})\n'
        f'Selected for Stage C Agent',
        xy=(13, fg_mean_dice[12]),
        xytext=(7.2, fg_mean_dice[12] - 0.05),
        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.8, headwidth=7, shrink=0.1),
        fontsize=11, fontweight='bold', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0FDF4', edgecolor=TEAL, linewidth=1.5)
    )

    ax.set_title('Foreground Macro Mean Dice Progression Across 15 Epochs', fontsize=15, fontweight='bold', color=NAVY, pad=14)
    ax.set_xlabel('Training Epoch', fontsize=12, color=SLATE)
    ax.set_ylabel('Validation Dice Score', fontsize=12, color=SLATE)
    ax.set_xticks(epochs)
    ax.set_ylim(0.75, 0.93)
    ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax.set_facecolor(LIGHT_BG)
    ax.legend(frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR, fontsize=11, loc='lower right')

    plt.savefig(os.path.join(TECH_DIR, '07_foreground_macro_dice_progression.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 07_foreground_macro_dice_progression.png')


# -----------------------------------------------------------------------------
# 08. Checkpoint Selection Explainer
# -----------------------------------------------------------------------------
def gen_08_checkpoint_selection():
    fig, ax = plt.subplots(figsize=(13, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Model Checkpoint Selection Criterion Rationale",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Distinguishing Epoch 13 (Peak Foreground Macro Dice) vs Epoch 14 (Minimum Validation Loss)",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    r1 = patches.FancyBboxPatch((0.08, 0.22), 0.38, 0.58, boxstyle="round,pad=0.02",
                                facecolor='#F0FDF4', edgecolor=TEAL, linewidth=2.0, zorder=3)
    ax.add_patch(r1)
    ax.text(0.27, 0.74, "EPOCH 13: SELECTED CHECKPOINT", ha='center', va='center', fontsize=13, fontweight='bold', color=TEAL)
    t1 = (
        "• Criterion: Peak Validation Foreground Macro Dice\n"
        "• Foreground Macro Mean Dice: 0.8670 (Global Peak)\n"
        "• Enhancing Tumor (ET) Dice: 0.9037\n"
        "• Peritumoral Edema (ED) Dice: 0.8765\n"
        "• Necrotic Core (NCR) Dice: 0.8210\n"
        "• Validation Loss: 0.2073\n\n"
        "Research Rationale:\n"
        "In clinical tumor segmentation, maximizing spatial\n"
        "overlap (Dice) on critical biological sub-regions takes\n"
        "precedence over marginal logit loss improvements."
    )
    ax.text(0.11, 0.47, t1, ha='left', va='center', fontsize=10.5, color=DARK, linespacing=1.3)

    r2 = patches.FancyBboxPatch((0.54, 0.22), 0.38, 0.58, boxstyle="round,pad=0.02",
                                facecolor='#FEF2F2', edgecolor=CORAL, linewidth=1.5, zorder=3)
    ax.add_patch(r2)
    ax.text(0.73, 0.74, "EPOCH 14: MINIMUM LOSS ALTERNATIVE", ha='center', va='center', fontsize=13, fontweight='bold', color=CORAL)
    t2 = (
        "• Criterion: Minimum Validation Loss\n"
        "• Validation Loss: 0.2063 (Global Minimum)\n"
        "• Foreground Macro Mean Dice: 0.8661 (−0.0009)\n"
        "• Enhancing Tumor (ET) Dice: 0.9065 (+0.0028)\n"
        "• Peritumoral Edema (ED) Dice: 0.8745 (−0.0020)\n"
        "• Necrotic Core (NCR) Dice: 0.8174 (−0.0036)\n\n"
        "Evaluation:\n"
        "While cross-entropy slightly favored Epoch 14,\n"
        "necrotic core and edema overlap degraded.\n"
        "Epoch 13 was preserved per project governance."
    )
    ax.text(0.57, 0.47, t2, ha='left', va='center', fontsize=10.5, color=DARK, linespacing=1.3)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.1, 1.0)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '08_checkpoint_selection_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 08_checkpoint_selection_explainer.png')


# -----------------------------------------------------------------------------
# 09. Segmentation Performance Summary
# -----------------------------------------------------------------------------
def gen_09_segmentation_summary():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=WHITE, dpi=300)

    subregions = ['Whole Tumor\n(WT)', 'Tumor Core\n(TC)', 'Enhancing Tumor\n(ET)', 'Macro Volumetric\nMean']
    dices = [0.9039, 0.8597, 0.8286, 0.8641]
    ci_lowers = [0.8765, 0.8076, 0.7794, 0.8265]
    ci_uppers = [0.9241, 0.8986, 0.8684, 0.8930]
    errors = [
        [d - l for d, l in zip(dices, ci_lowers)],
        [u - d for d, u in zip(dices, ci_uppers)]
    ]

    bars1 = ax1.bar(subregions, dices, yerr=errors, capsize=6, color=[TEAL, CYAN, AMBER, NAVY], alpha=0.85, edgecolor=NAVY, linewidth=1.2)
    ax1.set_title('3D Volumetric Dice Similarity Coefficient\n(Val_eval 105 Cases, Subject-Level Cluster Bootstrap 95% CI)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax1.set_ylabel('Volumetric Dice Score', fontsize=11, color=SLATE)
    ax1.set_ylim(0.7, 0.98)
    ax1.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
    ax1.set_facecolor(LIGHT_BG)

    for bar, d in zip(bars1, dices):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.04, f'{d:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold', color=WHITE)

    hd_subs = ['Whole Tumor\n(WT)', 'Tumor Core\n(TC)', 'Enhancing Tumor\n(ET)']
    mean_hd = [15.29, 8.98, 9.92]
    med_hd = [4.12, 3.16, 2.00]
    hd_ci_l = [11.38, 6.08, 4.71]
    hd_ci_u = [20.05, 12.74, 18.66]
    hd_err = [
        [m - l for m, l in zip(mean_hd, hd_ci_l)],
        [u - m for m, u in zip(mean_hd, hd_ci_u)]
    ]

    x = np.arange(len(hd_subs))
    w = 0.35
    b1 = ax2.bar(x - w/2, mean_hd, w, yerr=hd_err, capsize=6, label='Mean Bounded HD95', color=CORAL, alpha=0.85, edgecolor=CORAL)
    b2 = ax2.bar(x + w/2, med_hd, w, label='Median HD95', color=CYAN, alpha=0.85, edgecolor=CYAN)

    ax2.set_title('Surface Distance: 95th Percentile Hausdorff Distance\n(Val_eval 105 Cases, Millimeters; Spacing: 1.0×1.0×1.0 mm)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax2.set_ylabel('Distance (mm)', fontsize=11, color=SLATE)
    ax2.set_xticks(x)
    ax2.set_xticklabels(hd_subs, fontsize=10.5, color=SLATE)
    ax2.set_ylim(0, 24)
    ax2.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
    ax2.set_facecolor(LIGHT_BG)
    ax2.legend(frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR, fontsize=10)

    for bar, val in zip(b1, mean_hd):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1.2, f'{val:.1f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=CORAL)
    for bar, val in zip(b2, med_hd):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}', ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=CYAN)

    plt.savefig(os.path.join(TECH_DIR, '09_segmentation_performance_summary.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 09_segmentation_performance_summary.png')


# -----------------------------------------------------------------------------
# 10 & 11. Standalone Reliability Diagrams (Val_cal & Val_eval)
# -----------------------------------------------------------------------------
def gen_10_11_reliability_diagrams():
    bins_file = r'D:\MajorProject\results\stage_c\phase2\c2_calibration_v1\calibration_parameters.json'
    with open(bins_file) as f:
        params = json.load(f)

    val_cal_data = params['models']['resunet_2d']['val_cal_metrics']
    uncal = val_cal_data['uncalibrated']
    cal = val_cal_data['calibrated']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=WHITE, dpi=300)
    for ax, data, title, col in [
        (ax1, uncal, f"Val_cal Uncalibrated (T = 1.0)\nECE: {uncal['ece']*100:.2f}% | NLL: {uncal['nll']:.4f}", CORAL),
        (ax2, cal, f"Val_cal Calibrated (T* = 2.8805)\nECE: {cal['ece']*100:.2f}% | NLL: {cal['nll']:.4f}", TEAL)
    ]:
        conf = np.array(data['bin_confidences'])
        acc = np.array(data['bin_accuracies'])
        mask = np.array(data['bin_counts']) > 0

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
        ax.bar(conf[mask], acc[mask], width=0.045, color=col, alpha=0.65, edgecolor=col, label='Observed Accuracy')
        ax.scatter(conf[mask], acc[mask], color=col, s=40, zorder=4)
        ax.set_title(title, fontsize=12, fontweight='bold', color=NAVY)
        ax.set_xlabel('Mean Confidence', fontsize=11, color=SLATE)
        ax.set_ylabel('Empirical Accuracy', fontsize=11, color=SLATE)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.set_facecolor(LIGHT_BG)
        ax.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor=WHITE)

    plt.suptitle("Val_cal Calibration Reliability Diagrams (50 Cases, 5,020,730 Voxels)", fontsize=16, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '10_reliability_diagram_val_cal.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 10_reliability_diagram_val_cal.png')

    bins_eval = r'D:\MajorProject\results\stage_c\phase2\c2_calibration_v1\reliability_diagrams\resunet_2d_bins.json'
    with open(bins_eval) as f:
        eval_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=WHITE, dpi=300)
    for ax, data, title, col in [
        (ax1, eval_data['uncalibrated'], f"Val_eval Uncalibrated (T = 1.0)\nECE: {eval_data['uncalibrated']['ece']*100:.2f}% | NLL: {eval_data['uncalibrated']['nll']:.4f}", CORAL),
        (ax2, eval_data['calibrated'], f"Val_eval Calibrated (T* = 2.8805)\nECE: {eval_data['calibrated']['ece']*100:.2f}% | NLL: {eval_data['calibrated']['nll']:.4f}", TEAL)
    ]:
        conf = np.array(data['bin_confidences'])
        acc = np.array(data['bin_accuracies'])
        mask = np.array(data['bin_counts']) > 0

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
        ax.bar(conf[mask], acc[mask], width=0.045, color=col, alpha=0.65, edgecolor=col, label='Observed Accuracy')
        ax.scatter(conf[mask], acc[mask], color=col, s=40, zorder=4)
        ax.set_title(title, fontsize=12, fontweight='bold', color=NAVY)
        ax.set_xlabel('Mean Confidence', fontsize=11, color=SLATE)
        ax.set_ylabel('Empirical Accuracy', fontsize=11, color=SLATE)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
        ax.set_facecolor(LIGHT_BG)
        ax.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor=WHITE)

    plt.suptitle("Val_eval Calibration Reliability Diagrams (105 Held-out Cases, 311,546 Voxels)", fontsize=16, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '11_reliability_diagram_val_eval.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 11_reliability_diagram_val_eval.png')


# -----------------------------------------------------------------------------
# 12. Calibration Metrics Comparison
# -----------------------------------------------------------------------------
def gen_12_calibration_comparison():
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), facecolor=WHITE, dpi=300)

    metrics = [
        ("Negative Log-Likelihood", [0.4154, 0.2173], [1.0277, 0.4415], "−47.7% / −57.0%"),
        ("Expected Calib. Error (ECE)", [5.02, 0.84], [8.59, 3.41], "−83.3% / −60.3%"),
        ("Maximum Calib. Error (MCE)", [28.11, 7.26], [30.73, 26.42], "−74.2% / −14.0%"),
        ("Brier Score", [0.1122, 0.0994], [0.1851, 0.1660], "−11.4% / −10.3%"),
    ]

    for ax, (title, val_cal, val_eval, red_str) in zip(axes, metrics):
        x = np.arange(2)
        w = 0.35
        b1 = ax.bar(x - w/2, [val_cal[0], val_eval[0]], w, label='Uncalibrated', color=CORAL, alpha=0.85, edgecolor=CORAL)
        b2 = ax.bar(x + w/2, [val_cal[1], val_eval[1]], w, label='Calibrated', color=TEAL, alpha=0.85, edgecolor=TEAL)

        ax.set_title(title, fontsize=11.5, fontweight='bold', color=NAVY)
        ax.set_xticks(x)
        ax.set_xticklabels(['Val_cal\n(50 cases)', 'Val_eval\n(105 cases)'], fontsize=10, color=SLATE)
        ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
        ax.set_facecolor(LIGHT_BG)
        ax.text(0.5, 0.88, f"Rel. Red.: {red_str}", transform=ax.transAxes, ha='center', fontsize=9, fontweight='bold', color=NAVY,
                bbox=dict(boxstyle='round,pad=0.3', facecolor=WHITE, edgecolor=GRID_COLOR))

    axes[0].legend(frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR, fontsize=9.5)
    plt.suptitle("Stage C2 Calibration Metrics: Pre- vs Post-Temperature Scaling (T* = 2.8805)", fontsize=16, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '12_calibration_metrics_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 12_calibration_metrics_comparison.png')


# -----------------------------------------------------------------------------
# 13. Temperature Scaling Explainer
# -----------------------------------------------------------------------------
def gen_13_temperature_scaling_explainer():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Post-Hoc Probability Calibration: Temperature Scaling",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Adjusting Predictive Confidence without Modifying Network Weights or Discrete Mask Boundaries",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t_math = (
        "Mathematical Formulation:\n"
        "Given raw unnormalized logits z = [z_0, z_1, z_2, z_3] from ResUNet-2D,\n"
        "calibrated class posterior probabilities are computed via positive scalar T*:\n\n"
        "    p_k(x; T*) = exp(z_k / T*) / sum_j exp(z_j / T*)\n\n"
        "Fitted ResUNet-2D Optimal Parameter:  T* = 2.8804590702\n\n"
        "Why Temperature Scaling is Essential in Medical AI:\n"
        "1. Networks trained with soft Dice + CE become severely overconfident.\n"
        "2. T* > 1 softens extreme probability logits toward realistic empirical frequencies.\n"
        "3. Strictly Monotonic: argmax_k(z_k / T*) = argmax_k(z_k) for all T* > 0.\n"
        "   -> Argmax segmentation masks are 100% invariant (0 mismatches observed).\n"
        "4. Non-Destructive: No model weights are modified; inference remains deterministic."
    )
    box = patches.FancyBboxPatch((0.08, 0.14), 0.84, 0.68, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.48, t_math, ha='left', va='center', fontsize=11.5, color=DARK, linespacing=1.35, family='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '13_temperature_scaling_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 13_temperature_scaling_explainer.png')


# -----------------------------------------------------------------------------
# 14. Calibration Sampling Protocol
# -----------------------------------------------------------------------------
def gen_14_sampling_protocol():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Stage C2 Calibration Sampling Protocol & Cohort Isolation",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Rigorous Partitioning of 155 Validation Cases into Fitting (Val_cal) and Evaluation (Val_eval) Sets",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Dataset Split Integrity (calibration_splits.json, SHA-256: e753caa25...):\n\n"
        "1. Val_cal (Fitting Cohort - 50 Cases, 47 Biological Subjects):\n"
        "   • Sampled Voxels: 5,020,730 voxels total across 3,982 axial slices.\n"
        "   • Stratified Sampling: Focused on foreground tumor and boundary transition zones.\n"
        "   • Optimization: L-BFGS optimizer minimizing Negative Log-Likelihood (36 iterations).\n\n"
        "2. Val_eval (Held-Out Evaluation Cohort - 105 Cases, 96 Biological Subjects):\n"
        "   • Slices Evaluated: 16,275 full 2D axial slices.\n"
        "   • Sampled Evaluation Voxels: 311,546 voxels evaluated across identical coordinates.\n"
        "   • Subject-Isolation: Zero subject overlap between Val_cal and Val_eval.\n\n"
        "3. Methodological Disclosures:\n"
        "   • Calibration metrics reflect the documented sampled-voxel protocol, not all volume voxels.\n"
        "   • Checkpoint selection exposure: Epoch 13 was selected on the validation set prior to split."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=TEAL, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '14_calibration_sampling_protocol.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 14_calibration_sampling_protocol.png')


# -----------------------------------------------------------------------------
# 15. Predictive Entropy Distribution
# -----------------------------------------------------------------------------
def gen_15_entropy_distribution():
    pred_dir = r'D:\MajorProject\results\stage_c\phase2\c2_calibration_v1\evaluation_predictions\resunet_2d'
    entropies = []
    empty_slices = 0

    for fname in os.listdir(pred_dir):
        if fname.endswith('.json'):
            with open(os.path.join(pred_dir, fname)) as f:
                d = json.load(f)
                for s in d['slice_records']:
                    val = s.get('mean_fg_entropy')
                    if val is not None and not np.isnan(val):
                        entropies.append(val)
                    else:
                        empty_slices += 1

    entropies = np.array(entropies)
    non_empty = len(entropies)
    total = non_empty + empty_slices

    fig, ax = plt.subplots(figsize=(12, 6.5), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.axvspan(0.0, 0.35, color='#F0FDF4', alpha=0.6, label='ACCEPT Zone (H < 0.35, 81.74% of Non-Empty)')
    ax.axvspan(0.35, 0.65, color='#FFF1F2', alpha=0.6, label='FLAG_UNRELIABLE Triage Zone (H ≥ 0.35, 18.26%)')
    ax.axvline(x=0.35, color=CORAL, linestyle='--', linewidth=2.2, label='Policy Threshold τ = 0.35')

    ax.hist(entropies, bins=50, color=CYAN, alpha=0.75, edgecolor=WHITE, density=True)

    mean_val = np.mean(entropies)
    med_val = np.median(entropies)
    ax.axvline(x=mean_val, color=NAVY, linestyle='-', linewidth=2.0, label=f'Mean: {mean_val:.4f}')
    ax.axvline(x=med_val, color=TEAL, linestyle='-.', linewidth=2.0, label=f'Median: {med_val:.4f}')

    q1 = np.percentile(entropies, 25)
    q3 = np.percentile(entropies, 75)
    stats_str = (
        f"Val_eval Population (105 Cases):\n"
        f"• Total Slices: {total:,}\n"
        f"• Non-Empty Slices: {non_empty:,} (42.24%)\n"
        f"• Empty Slices: {empty_slices:,} (57.76%)\n"
        f"• Mean ± Std: {mean_val:.4f} ± {np.std(entropies):.4f}\n"
        f"• Median [IQR]: {med_val:.4f} [{q1:.4f}, {q3:.4f}]\n"
        f"• Min / Max: {np.min(entropies):.4f} / {np.max(entropies):.4f}\n"
        f"• Slices with H ≥ 0.35: 1,255 (18.26%)"
    )
    ax.text(0.68, 0.95, stats_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor=WHITE, edgecolor=NAVY, alpha=0.95))

    ax.set_title('Normalized Foreground Shannon Predictive Entropy Distribution (Val_eval N=6,874)', fontsize=15, fontweight='bold', color=NAVY, pad=14)
    ax.set_xlabel('Normalized Foreground Entropy H in [0, 1]', fontsize=12, color=SLATE)
    ax.set_ylabel('Probability Density', fontsize=12, color=SLATE)
    ax.set_xlim(0, 0.65)
    ax.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax.legend(loc='upper left', fontsize=10, frameon=True, facecolor=WHITE, edgecolor=GRID_COLOR)

    plt.savefig(os.path.join(TECH_DIR, '15_predictive_entropy_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 15_predictive_entropy_distribution.png')


# -----------------------------------------------------------------------------
# 16. Entropy Threshold Explainer
# -----------------------------------------------------------------------------
def gen_16_threshold_explainer():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Operational Threshold τ = 0.35 Gating Semantics",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Balancing Autonomous Coverage Against Segmentation Reliability",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Operating Threshold Design (Stage C3 Policy Definition):\n\n"
        "1. Low-Uncertainty Autonomous Acceptance Zone (H < 0.35):\n"
        "   • Applies strictly to non-empty primary predictions.\n"
        "   • 5,619 slices in Val_eval (81.74% of non-empty predictions; 34.53% of all slices).\n"
        "   • Captures 83.02% of all true ground-truth tumor slices.\n"
        "   • Retained slice Mean Dice reaches 0.7855 on accepted primary predictions.\n\n"
        "2. High-Uncertainty Specialist Referral Zone (H ≥ 0.35):\n"
        "   • 1,255 non-empty slices in Val_eval (18.26% of non-empty predictions).\n"
        "   • Action: FLAG_UNRELIABLE. Automatically routes ambiguous slices to expert review.\n"
        "   • Mean Dice of referred slices is 0.3891, demonstrating effective triage of errors.\n\n"
        "3. Critical Scientific Caveat:\n"
        "   • τ = 0.35 is an operational decision threshold, NOT a certified clinical safety boundary.\n"
        "   • Entropy is a dispersion indicator and cannot guarantee absence of silent errors."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '16_entropy_threshold_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 16_entropy_threshold_explainer.png')


# -----------------------------------------------------------------------------
# 17. Entropy Population Breakdown
# -----------------------------------------------------------------------------
def gen_17_population_breakdown():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=WHITE, dpi=300)

    labels1 = ['Empty Primary\n(9,401 / 57.76%)', 'Non-Empty Low-H\n(5,619 / 34.53%)', 'Non-Empty High-H\n(1,255 / 7.71%)']
    sizes1 = [9401, 5619, 1255]
    colors1 = ['#94A3B8', TEAL, CORAL]
    ax1.pie(sizes1, labels=labels1, autopct='%1.1f%%', startangle=140, colors=colors1,
            wedgeprops=dict(width=0.45, edgecolor=WHITE, linewidth=2), textprops=dict(fontsize=10.5, color=DARK, fontweight='bold'))
    ax1.set_title('All Evaluated Slices in Val_eval\n(Total: 16,275 Slices across 105 Cases)', fontsize=13, fontweight='bold', color=NAVY, pad=12)

    labels2 = ['H < 0.35 (ACCEPT)\n(5,619 / 81.74%)', 'H ≥ 0.35 (FLAG)\n(1,255 / 18.26%)']
    sizes2 = [5619, 1255]
    colors2 = [TEAL, CORAL]
    ax2.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=140, colors=colors2,
            wedgeprops=dict(width=0.45, edgecolor=WHITE, linewidth=2), textprops=dict(fontsize=11, color=DARK, fontweight='bold'))
    ax2.set_title('Non-Empty Primary Predictions\n(Total: 6,874 Slices, 42.24% of Cohort)', fontsize=13, fontweight='bold', color=NAVY, pad=12)

    plt.suptitle("Val_eval Slice Population Uncertainty Breakdown", fontsize=16, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '17_entropy_population_breakdown.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 17_entropy_population_breakdown.png')


# -----------------------------------------------------------------------------
# 18. Full C3 Decision Flow (Redesigned & Unclipped)
# -----------------------------------------------------------------------------
def gen_18_full_decision_flow():
    fig, ax = plt.subplots(figsize=(16, 9.5), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.96, "Stage C3 v1.1.3 Reliability Agent Decision Protocol",
            ha='center', va='center', fontsize=22, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.92, "Rigorous Three-Action Triage Logic: ACCEPT | FALLBACK_SWITCH | FLAG_UNRELIABLE",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    def draw_node(x, y, w, h, title, subtitle, color, shape="round", tc=WHITE):
        if shape == "round":
            p = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                      facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3)
        elif shape == "diamond":
            verts = [(x + w/2, y + h), (x + w, y + h/2), (x + w/2, y), (x, y + h/2)]
            p = patches.Polygon(verts, closed=True, facecolor=color, edgecolor=NAVY, linewidth=1.8, zorder=3)
        ax.add_patch(p)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=10.5, fontweight='bold', color=tc, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=8.5, color=tc, zorder=4)

    draw_node(0.04, 0.65, 0.18, 0.15, "Primary Champion", "ResUNet-2D (T*=2.8805)\nInference on Slice z", NAVY)
    draw_node(0.28, 0.65, 0.16, 0.15, "Empty Foreground?", "Argmax tumor voxels = 0?", '#FEF3C7', shape="diamond", tc=NAVY)
    draw_node(0.50, 0.72, 0.18, 0.15, "Uncertainty Check", "Foreground Entropy H\nvs Threshold τ = 0.35", '#FEF3C7', shape="diamond", tc=NAVY)

    draw_node(0.76, 0.81, 0.20, 0.12, "ACTION: ACCEPT", "H < 0.35 (Low Uncertainty)\nDeploy ResUNet-2D Mask\n(5,619 Slices / 34.53%)", '#059669')
    draw_node(0.76, 0.64, 0.20, 0.12, "ACTION: FLAG_UNRELIABLE", "H ≥ 0.35 (High Uncertainty)\nTriage for Expert Review\n(1,255 Slices / 7.71%)", CORAL)

    draw_node(0.28, 0.35, 0.16, 0.15, "2.5D Axial Context?", "Radius k=±2 (4 slots)\nTumor Neighbors ≥ 1?", '#FEF3C7', shape="diamond", tc=NAVY)
    draw_node(0.28, 0.12, 0.16, 0.12, "ACTION: FLAG_UNRELIABLE", "0 Neighboring Tumor\n(True Negative / Margin)\n(9,254 Slices / 56.86%)", CORAL)

    draw_node(0.50, 0.35, 0.18, 0.15, "Secondary Fallback", "UNetResNet34 (T*=2.6261)\nEvaluate Slice z", NAVY)
    draw_node(0.74, 0.35, 0.16, 0.15, "Secondary Non-Empty?", "Candidate detects\ntumor voxels > 0?", '#FEF3C7', shape="diamond", tc=NAVY)
    draw_node(0.74, 0.12, 0.22, 0.13, "ACTION: FALLBACK_SWITCH", "Adopt UNetResNet34 Mask\n(147 Slices / 0.90%)\n[+Warning if H_sec ≥ 0.35]", TEAL)
    draw_node(0.74, -0.01, 0.22, 0.08, "Secondary Also Empty → FLAG", "Consensus Empty / Triage", CORAL)

    arr = dict(facecolor=SLATE, edgecolor=SLATE, width=2.0, headwidth=7, shrink=0.05)
    ax.annotate('', xy=(0.28, 0.725), xytext=(0.22, 0.725), arrowprops=arr)
    ax.annotate('', xy=(0.50, 0.795), xytext=(0.44, 0.725), arrowprops=arr)
    ax.text(0.44, 0.78, "NO (Non-Empty)", fontsize=9.5, fontweight='bold', color=SLATE)

    ax.annotate('', xy=(0.36, 0.50), xytext=(0.36, 0.65), arrowprops=arr)
    ax.text(0.37, 0.57, "YES (Empty)", fontsize=9.5, fontweight='bold', color=SLATE)

    ax.annotate('', xy=(0.76, 0.87), xytext=(0.68, 0.82), arrowprops=arr)
    ax.text(0.69, 0.87, "H < 0.35", fontsize=9.5, fontweight='bold', color='#059669')

    ax.annotate('', xy=(0.76, 0.70), xytext=(0.68, 0.77), arrowprops=arr)
    ax.text(0.69, 0.72, "H ≥ 0.35", fontsize=9.5, fontweight='bold', color=CORAL)

    ax.annotate('', xy=(0.50, 0.425), xytext=(0.44, 0.425), arrowprops=arr)
    ax.text(0.44, 0.44, "YES (≥1 Tumor)", fontsize=9.5, fontweight='bold', color=SLATE)

    ax.annotate('', xy=(0.36, 0.24), xytext=(0.36, 0.35), arrowprops=arr)
    ax.text(0.37, 0.29, "NO (0 Tumor)", fontsize=9.5, fontweight='bold', color=SLATE)

    ax.annotate('', xy=(0.74, 0.425), xytext=(0.68, 0.425), arrowprops=arr)
    ax.annotate('', xy=(0.85, 0.25), xytext=(0.82, 0.35), arrowprops=arr)
    ax.text(0.85, 0.30, "YES (Non-Empty)", fontsize=9.5, fontweight='bold', color=TEAL)

    ax.annotate('', xy=(0.85, 0.07), xytext=(0.85, 0.12), arrowprops=arr)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.0)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '18_full_c3_decision_flow.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 18_full_c3_decision_flow.png')


# -----------------------------------------------------------------------------
# 19. Primary & Secondary Interaction
# -----------------------------------------------------------------------------
def gen_19_primary_secondary():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Primary Champion & Secondary Fallback Interaction",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Why the Secondary Model is Selectively Invoked Only Under Empty-Primary / Context-Supported Conditions",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Design Rationale for Constrained Secondary Invocation:\n\n"
        "1. Primary Champion (ResUNet-2D, T* = 2.8805):\n"
        "   • Superior overall volumetric overlap (WT Dice: 0.9039, TC: 0.8597, ET: 0.8286).\n"
        "   • Handles 95%+ of routine slice segmentations with low uncertainty.\n\n"
        "2. Secondary Fallback (UNetResNet34, T* = 2.6261):\n"
        "   • Heavier encoder backbone (24.33M parameters vs 8.11M).\n"
        "   • Invoked selectively ONLY when ResUNet-2D predicts empty inside a known tumor volume.\n"
        "   • Avoids redundant dual-model execution overhead on 99% of slices.\n\n"
        "3. High-Uncertainty Fallback Semantics (Frozen C3 v1.1.3 Rule):\n"
        "   • If secondary model detects tumor but has secondary entropy H_sec ≥ 0.35, action remains\n"
        "     FALLBACK_SWITCH with an audit warning (candidate_high_uncertainty_warning).\n"
        "   • Retaining the secondary mask prevents dropping rescued tumor slices while alerting radiologists."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '19_primary_secondary_model_interaction.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 19_primary_secondary_model_interaction.png')


# -----------------------------------------------------------------------------
# 20. Agent Action Distribution
# -----------------------------------------------------------------------------
def gen_20_action_distribution():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=WHITE, dpi=300)

    actions = ['ACCEPT\n(Primary Low-H)', 'FALLBACK_SWITCH\n(Rescued by Sec)', 'FLAG_UNRELIABLE\n(Referred / Empty)']
    counts1 = [5619, 147, 10509]
    pcts1 = [34.53, 0.90, 64.57]
    colors = ['#059669', TEAL, CORAL]

    b1 = ax1.bar(actions, counts1, color=colors, alpha=0.85, edgecolor=NAVY, linewidth=1.2)
    ax1.set_title('Triage Actions Across All Evaluated Slices\n(Denominator: N = 16,275 Slices, 105 Cases)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax1.set_ylabel('Slice Count', fontsize=11, color=SLATE)
    ax1.set_ylim(0, 12000)
    ax1.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
    ax1.set_facecolor(LIGHT_BG)

    for bar, c, p in zip(b1, counts1, pcts1):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 250, f'{c:,}\n({p:.2f}%)', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=DARK)

    gt_actions = ['ACCEPT\n(Primary Retained)', 'FALLBACK_SWITCH\n(Rescued Tumor)', 'FLAG_UNRELIABLE\n(Referred Tumor)']
    counts2 = [5437, 54, 1058]
    pcts2 = [83.02, 0.82, 16.16]

    b2 = ax2.bar(gt_actions, counts2, color=colors, alpha=0.85, edgecolor=NAVY, linewidth=1.2)
    ax2.set_title('Triage Actions on Ground-Truth Tumor Slices\n(Denominator: N = 6,549 Slices; Retained: 83.84%)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax2.set_ylabel('Slice Count', fontsize=11, color=SLATE)
    ax2.set_ylim(0, 6500)
    ax2.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
    ax2.set_facecolor(LIGHT_BG)

    for bar, c, p in zip(b2, counts2, pcts2):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 120, f'{c:,}\n({p:.2f}%)', ha='center', va='bottom', fontsize=10.5, fontweight='bold', color=DARK)

    plt.suptitle("Stage C3 Reliability Agent Triage Action Distribution", fontsize=16, fontweight='bold', color=NAVY, y=0.98)
    plt.savefig(os.path.join(TECH_DIR, '20_agent_action_distribution.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 20_agent_action_distribution.png')


# -----------------------------------------------------------------------------
# 21. Empty Prediction Fallback Explainer
# -----------------------------------------------------------------------------
def gen_21_empty_fallback():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Empty-Prediction Fallback Recovery Mechanism",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "How 2.5D Axial Context Rescues False-Negative Slice Dropouts",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "False-Negative Slice Dropout Problem in 2D Slice Inference:\n"
        "• In 2D slice-by-slice models, individual axial slices through the center of a tumor can occasionally\n"
        "  predict 0 foreground pixels due to local contrast variation or subtle artifact.\n"
        "• In an unfiltered system, this creates an erroneous false negative slice dropout.\n\n"
        "Stage C3 2.5D Axial Context Solution:\n"
        "1. Detection: Primary model predicts 0 foreground pixels on slice z.\n"
        "2. Context Inspection: Inspect neighboring primary predictions within radius k = ±2 (z-2, z-1, z+1, z+2).\n"
        "3. Axial Support Confirmation: If at least 1 neighbor contains confirmed tumor, axial support is confirmed.\n"
        "4. Secondary Invocation: Query UNetResNet34 on slice z.\n"
        "5. Fallback Switch: If secondary detects tumor, adopt its prediction (FALLBACK_SWITCH).\n"
        "   -> Rescued 54 true ground-truth tumor slices in Val_eval with 0 human intervention."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=TEAL, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '21_empty_prediction_fallback_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 21_empty_prediction_fallback_explainer.png')


# -----------------------------------------------------------------------------
# 22. Context Window Explainer
# -----------------------------------------------------------------------------
def gen_22_context_window():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "2.5D Axial Context Neighborhood Topology & Boundary Accounting",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Radius k = ±2 Window with Authoritative Boundary Slot Availability Accounting",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Axial Neighborhood Parameters & Accounting (k = 2, N_min = 1):\n\n"
        "1. Interior Slice (e.g., z = 50, Total Volume Slices: 155):\n"
        "   • Evaluated Neighbor Slots: z-2 (48), z-1 (49), z+1 (51), z+2 (52) [Total: 4 slots].\n"
        "   • All 4 slots available. Condition: Sum(foreground_neighbors) ≥ 1.\n\n"
        "2. Boundary Slice z = 0 (Inferior Margin):\n"
        "   • Out-of-volume slots: z-2 (−2) and z-1 (−1) are strictly UNAVAILABLE.\n"
        "   • Available slots: z+1 (1) and z+2 (2) [Total: 2 available slots].\n"
        "   • Authoritative Convention: Out-of-volume slots count as unavailable, not as negative support.\n\n"
        "3. Boundary Slice z = 154 (Superior Margin):\n"
        "   • Out-of-volume slots: z+1 (155) and z+2 (156) are UNAVAILABLE.\n"
        "   • Available slots: z-2 (152) and z-1 (153) [Total: 2 available slots].\n\n"
        "4. Fallback Rule: If context is confirmed, invoke secondary candidate; else FLAG_UNRELIABLE."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '22_context_window_explainer.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 22_context_window_explainer.png')


# -----------------------------------------------------------------------------
# 23. Checkpoint Provenance
# -----------------------------------------------------------------------------
def gen_23_checkpoint_provenance():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Cryptographic Checkpoint Identity & Integrity Card",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Immutable Physical Model Weights Verification",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Physical Checkpoint Verification Details:\n\n"
        "• Model Architecture: ResUNet-2D (PreActResidualBlock, GroupNorm G=8)\n"
        "• Checkpoint Profile: STAGE_C_15EP\n"
        "• Target Checkpoint File: results/full_training_15ep/resunet_2d/best_model.pt\n"
        "• Selected Epoch: Epoch 13 (Global Validation Peak)\n"
        "• Physical File Size: 97,486,845 bytes (92.97 MB)\n"
        "• Verified SHA-256 Digest:\n"
        "  81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d\n\n"
        "• Associated Final Checkpoint (Epoch 15): results/.../final_model.pt\n"
        "  File Size: 97,488,175 bytes | SHA-256: 3c6e265f9b6093963ce346468d5dc1d40e878cf...\n\n"
        "• Total Parameter Count: Exactly 8,114,604 active trainable float32 weights.\n"
        "• Storage Format: PyTorch State Dictionary (.pt) with full key-weight mapping."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=NAVY, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35, family='monospace')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '23_model_checkpoint_provenance.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 23_model_checkpoint_provenance.png')


# -----------------------------------------------------------------------------
# 24. Evaluation Cohort Summary
# -----------------------------------------------------------------------------
def gen_24_cohort_summary():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Dataset Census & Cohort Partitioning Register",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Strict Separation of Training, Calibration, Evaluation, and Quarantine Test Cohorts",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Dataset Split Architecture (BraTS-GLI Census, Split SHA-256: e753caa25...):\n\n"
        "1. Training Set: 721 Cases (59,020 Slices)\n"
        "   • Used exclusively for backpropagation model training over 15 epochs.\n\n"
        "2. Validation Calibration Set (Val_cal): 50 Cases (3,982 Slices, 47 Biological Subjects)\n"
        "   • Used exclusively for post-hoc Temperature Scaling fitting (5,020,730 sampled voxels).\n\n"
        "3. Validation Evaluation Set (Val_eval): 105 Cases (16,275 Slices, 96 Biological Subjects)\n"
        "   • Held-out evaluation benchmark for volumetric Dice, HD95, ECE, and Stage C3 simulation.\n\n"
        "4. Quarantined Test Set: 155 Cases (Strictly Sealed & Unaccessed)\n"
        "   • Protected under data/quarantine_test/ and test_manifest.json; untouched."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=TEAL, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '24_evaluation_cohort_summary.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 24_evaluation_cohort_summary.png')


# -----------------------------------------------------------------------------
# 25. Model Limitations and Boundaries
# -----------------------------------------------------------------------------
def gen_25_limitations_card():
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=WHITE, dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.94, "Scientific Disclosures, Boundary Guardrails & Non-Clinical Status",
            ha='center', va='center', fontsize=20, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.87, "Essential Context for Teammates, Technical Reviewers, and Viva Panels",
            ha='center', va='center', fontsize=13, color=SLATE, transform=ax.transAxes)

    t = (
        "Non-Negotiable Research Boundary Guardrails:\n\n"
        "1. Academic Prototype: ResUNet-2D is an experimental research system. It has NOT received FDA,\n"
        "   CDSCO, or CE mark regulatory clearance for autonomous diagnostic or surgical planning use.\n"
        "2. Single-Seed Scope: Model was trained on random seed 42; multi-seed variance is unquantified.\n"
        "3. Internal Validation Only: Evaluated on BraTS-GLI internal splits; external multi-center generalizability\n"
        "   across varying scanners, field strengths (1.5T vs 3.0T), and institutions is unverified.\n"
        "4. Missing Modality Failure: Missing T1c collapses Tumor Core Dice to 0.3132; missing T2w collapses\n"
        "   Whole Tumor Dice to 0.4388. All 4 mpMRI sequences must be co-registered and present.\n"
        "5. Human-in-the-Loop Mandate: All FLAG_UNRELIABLE slices must undergo radiologist review."
    )
    box = patches.FancyBboxPatch((0.08, 0.12), 0.84, 0.70, boxstyle="round,pad=0.02",
                                 facecolor=WHITE, edgecolor=CORAL, linewidth=1.8, zorder=3)
    ax.add_patch(box)
    ax.text(0.12, 0.47, t, ha='left', va='center', fontsize=11, color=DARK, linespacing=1.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    plt.savefig(os.path.join(TECH_DIR, '25_model_limitations_and_evidence_boundaries.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print('[✓] 25_model_limitations_and_evidence_boundaries.png')


if __name__ == '__main__':
    print('Generating 25 Technical Assets...')
    gen_01_architecture()
    gen_02_residual_block()
    gen_03_pipeline()
    gen_04_mri_channels()
    gen_05_loss_curve()
    gen_06_classwise_dice()
    gen_07_foreground_macro_dice()
    gen_08_checkpoint_selection()
    gen_09_segmentation_summary()
    gen_10_11_reliability_diagrams()
    gen_12_calibration_comparison()
    gen_13_temperature_scaling_explainer()
    gen_14_sampling_protocol()
    gen_15_entropy_distribution()
    gen_16_threshold_explainer()
    gen_17_population_breakdown()
    gen_18_full_decision_flow()
    gen_19_primary_secondary()
    gen_20_action_distribution()
    gen_21_empty_fallback()
    gen_22_context_window()
    gen_23_checkpoint_provenance()
    gen_24_cohort_summary()
    gen_25_limitations_card()
    print('All 25 Technical Assets successfully generated in:', TECH_DIR)
