"""Generate publication-grade figures for ResUNet-2D Model Dossier.

Workspace: D:\MajorProject
Output Directory: reports/model_profiles/resunet_2d/assets/
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

# Set visual styling
plt.rcParams['font.sans-serif'] = 'Segoe UI', 'DejaVu Sans', 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.autolayout'] = True

ASSETS_DIR = r'D:\MajorProject\reports\model_profiles\resunet_2d\assets'
os.makedirs(ASSETS_DIR, exist_ok=True)

# Colors
NAVY = '#1F3A60'
SLATE = '#475569'
TEAL = '#0D9488'
CYAN = '#0284C7'
CORAL = '#E11D48'
AMBER = '#D97706'
LIGHT_BG = '#F8FAFC'
GRID_COLOR = '#E2E8F0'


# -----------------------------------------------------------------------------
# 1. Training Curves (Loss & Mean Dice)
# -----------------------------------------------------------------------------
def plot_training_curves():
    csv_path = r'D:\MajorProject\results\full_training_15ep\resunet_2d\history.csv'
    df = pd.read_csv(csv_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), facecolor='white', dpi=300)

    epochs = df['epoch'].values
    train_loss = df['train_loss'].values
    val_loss = df['val_loss'].values
    mean_dice = df['mean_dice'].values

    # Left: Losses
    ax1.plot(epochs, train_loss, marker='o', color=NAVY, linewidth=2.2, label='Training Loss (Compound Dice+CE)')
    ax1.plot(epochs, val_loss, marker='s', color=CORAL, linewidth=2.2, label='Validation Loss')
    ax1.set_title('Training & Validation Loss Trajectory (15 Epochs)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax1.set_xlabel('Epoch', fontsize=11, color=SLATE)
    ax1.set_ylabel('Loss Value', fontsize=11, color=SLATE)
    ax1.set_xticks(epochs)
    ax1.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax1.set_facecolor(LIGHT_BG)

    # Annotate minimum validation loss at Epoch 14
    min_val_loss_idx = int(np.argmin(val_loss))
    ax1.annotate(
        f'Lowest Val Loss: {val_loss[min_val_loss_idx]:.4f}\n(Epoch {epochs[min_val_loss_idx]})',
        xy=(epochs[min_val_loss_idx], val_loss[min_val_loss_idx]),
        xytext=(epochs[min_val_loss_idx] - 3.5, val_loss[min_val_loss_idx] + 0.04),
        arrowprops=dict(facecolor=CORAL, edgecolor=CORAL, width=1.5, headwidth=6, shrink=0.1),
        fontsize=9.5, fontweight='bold', color=CORAL,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=CORAL, alpha=0.9)
    )
    ax1.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=10)

    # Right: Validation Mean Dice
    ax2.plot(epochs, mean_dice, marker='D', color=TEAL, linewidth=2.5, label='Validation Mean Dice')
    ax2.set_title('Validation Mean Dice Across Epochs', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax2.set_xlabel('Epoch', fontsize=11, color=SLATE)
    ax2.set_ylabel('Mean Dice Score', fontsize=11, color=SLATE)
    ax2.set_xticks(epochs)
    ax2.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax2.set_facecolor(LIGHT_BG)

    # Annotate Best Checkpoint at Epoch 13
    best_idx = 12  # Epoch 13
    ax2.axvline(x=13, color=NAVY, linestyle=':', linewidth=1.8, alpha=0.7)
    ax2.scatter([13], [mean_dice[best_idx]], color=NAVY, s=120, zorder=5)
    ax2.annotate(
        f'BEST CHECKPOINT (Epoch 13)\nMean Dice: {mean_dice[best_idx]:.4f}\nSelected for Stage C Agent',
        xy=(13, mean_dice[best_idx]),
        xytext=(7.5, mean_dice[best_idx] - 0.045),
        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.8, headwidth=7, shrink=0.1),
        fontsize=9.5, fontweight='bold', color=NAVY,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F0FDF4', edgecolor=TEAL, linewidth=1.5)
    )
    ax2.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=10, loc='lower right')

    out_file = os.path.join(ASSETS_DIR, 'training_curves.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


# -----------------------------------------------------------------------------
# 2. Class-wise Dice Curves
# -----------------------------------------------------------------------------
def plot_class_dice_curves():
    csv_path = r'D:\MajorProject\results\full_training_15ep\resunet_2d\history.csv'
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(11, 6), facecolor='white', dpi=300)

    epochs = df['epoch'].values
    dice_bg = df['dice_class_0'].values
    dice_ncr = df['dice_class_1'].values
    dice_ed = df['dice_class_2'].values
    dice_et = df['dice_class_3'].values
    mean_dice = df['mean_dice'].values

    ax.plot(epochs, dice_bg, marker='o', color='#64748B', linewidth=1.8, linestyle='--', label='Class 0: Background (BG)')
    ax.plot(epochs, dice_et, marker='^', color='#0284C7', linewidth=2.3, label='Class 3: Enhancing Tumor (ET)')
    ax.plot(epochs, dice_ed, marker='s', color='#10B981', linewidth=2.3, label='Class 2: Peritumoral Edema (ED)')
    ax.plot(epochs, dice_ncr, marker='v', color='#F59E0B', linewidth=2.3, label='Class 1: Necrotic Core (NCR)')
    ax.plot(epochs, mean_dice, marker='D', color=NAVY, linewidth=2.8, label='Macro Mean Dice (All 4 Classes)')

    ax.axvline(x=13, color=NAVY, linestyle=':', linewidth=1.8, alpha=0.8)

    # Annotate values at Epoch 13
    ax.annotate(
        f'Epoch 13 Peak Breakdown:\n'
        f'• BG:  {dice_bg[12]:.4f}\n'
        f'• ET:  {dice_et[12]:.4f}\n'
        f'• ED:  {dice_ed[12]:.4f}\n'
        f'• NCR: {dice_ncr[12]:.4f}\n'
        f'• Mean:{mean_dice[12]:.4f}',
        xy=(13, 0.82),
        xytext=(9.2, 0.65),
        arrowprops=dict(facecolor=NAVY, edgecolor=NAVY, width=1.5, headwidth=6, shrink=0.1),
        fontsize=9.5, color=NAVY, fontweight='semibold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8FAFC', edgecolor=NAVY, linewidth=1.2)
    )

    ax.set_title('ResUNet-2D Multi-Class Segmentation Progression (Epochs 1–15)', fontsize=13, fontweight='bold', color=NAVY, pad=12)
    ax.set_xlabel('Training Epoch', fontsize=11, color=SLATE)
    ax.set_ylabel('Validation Dice Similarity Coefficient (DSC)', fontsize=11, color=SLATE)
    ax.set_xticks(epochs)
    ax.set_ylim(0.60, 1.01)
    ax.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax.set_facecolor(LIGHT_BG)
    ax.legend(frameon=True, facecolor='white', edgecolor=GRID_COLOR, fontsize=10, loc='lower right')

    out_file = os.path.join(ASSETS_DIR, 'class_dice_curves.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


# -----------------------------------------------------------------------------
# 3. Calibration Comparison (Reliability Diagrams & ECE Bar Chart)
# -----------------------------------------------------------------------------
def plot_calibration_comparison():
    bins_file = r'D:\MajorProject\results\stage_c\phase2\c2_calibration_v1\reliability_diagrams\resunet_2d_bins.json'
    with open(bins_file) as f:
        bins = json.load(f)

    fig = plt.figure(figsize=(15, 5.5), facecolor='white', dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.1, 0.8], wspace=0.28)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    uncal = bins['uncalibrated']
    cal = bins['calibrated']

    # Subplot 1: Uncalibrated Reliability Diagram
    uncal_conf = np.array(uncal['bin_confidences'])
    uncal_acc = np.array(uncal['bin_accuracies'])
    mask_uncal = np.array(uncal['bin_counts']) > 0

    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax1.bar(uncal_conf[mask_uncal], uncal_acc[mask_uncal], width=0.045, color=CORAL, alpha=0.65, edgecolor=CORAL, label='Empirical Accuracy')
    ax1.scatter(uncal_conf[mask_uncal], uncal_acc[mask_uncal], color=CORAL, s=40, zorder=4)
    ax1.set_title(f"Uncalibrated (T = 1.0)\nVal_eval ECE: {uncal['ece']*100:.2f}% | NLL: {uncal['nll']:.4f}", fontsize=11, fontweight='bold', color=NAVY)
    ax1.set_xlabel('Mean Predicted Confidence', fontsize=10.5, color=SLATE)
    ax1.set_ylabel('Empirical Accuracy', fontsize=10.5, color=SLATE)
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.02)
    ax1.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax1.set_facecolor(LIGHT_BG)
    ax1.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white')

    # Subplot 2: Calibrated Reliability Diagram
    cal_conf = np.array(cal['bin_confidences'])
    cal_acc = np.array(cal['bin_accuracies'])
    mask_cal = np.array(cal['bin_counts']) > 0

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Perfect Calibration')
    ax2.bar(cal_conf[mask_cal], cal_acc[mask_cal], width=0.045, color=TEAL, alpha=0.65, edgecolor=TEAL, label='Empirical Accuracy')
    ax2.scatter(cal_conf[mask_cal], cal_acc[mask_cal], color=TEAL, s=40, zorder=4)
    ax2.set_title(f"Calibrated (T* = 2.8805)\nVal_eval ECE: {cal['ece']*100:.2f}% | NLL: {cal['nll']:.4f}", fontsize=11, fontweight='bold', color=NAVY)
    ax2.set_xlabel('Mean Predicted Confidence', fontsize=10.5, color=SLATE)
    ax2.set_ylabel('Empirical Accuracy', fontsize=10.5, color=SLATE)
    ax2.set_xlim(0, 1.02)
    ax2.set_ylim(0, 1.02)
    ax2.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR)
    ax2.set_facecolor(LIGHT_BG)
    ax2.legend(loc='upper left', fontsize=9, frameon=True, facecolor='white')

    # Subplot 3: ECE and NLL Reduction Summary
    cohorts = ['Val_cal\n(50 cases)', 'Val_eval\n(105 cases)']
    x = np.arange(len(cohorts))
    width = 0.35

    # ECE data in percentage
    uncal_eces = [5.02, 8.59]
    cal_eces = [0.84, 3.41]

    rects1 = ax3.bar(x - width/2, uncal_eces, width, label='Uncalibrated', color=CORAL, alpha=0.85, edgecolor=CORAL)
    rects2 = ax3.bar(x + width/2, cal_eces, width, label='Calibrated (T*=2.8805)', color=TEAL, alpha=0.85, edgecolor=TEAL)

    ax3.set_title('ECE Reduction by Cohort', fontsize=11, fontweight='bold', color=NAVY)
    ax3.set_ylabel('Expected Calibration Error (%)', fontsize=10.5, color=SLATE)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cohorts, fontsize=10, color=SLATE)
    ax3.grid(True, linestyle='--', alpha=0.5, color=GRID_COLOR, axis='y')
    ax3.set_facecolor(LIGHT_BG)
    ax3.set_ylim(0, 10.5)

    # Add data labels
    for rect in rects1:
        h = rect.get_height()
        ax3.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=CORAL)
    for rect in rects2:
        h = rect.get_height()
        ax3.annotate(f'{h:.2f}%', xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold', color=TEAL)

    # Reduction annotations
    ax3.annotate('−83.3% rel.', xy=(0, 1.5), ha='center', fontsize=8.5, color=NAVY, fontweight='bold')
    ax3.annotate('−60.3% rel.', xy=(1, 4.2), ha='center', fontsize=8.5, color=NAVY, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8.5, frameon=True, facecolor='white')

    out_file = os.path.join(ASSETS_DIR, 'calibration_comparison.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


# -----------------------------------------------------------------------------
# 4. Predictive Uncertainty Distribution
# -----------------------------------------------------------------------------
def plot_entropy_distribution():
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

    fig, (ax_box, ax_hist) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                          gridspec_kw={'height_ratios': [0.2, 0.8]},
                                          facecolor='white', dpi=300)

    # Top: Boxplot
    bp = ax_box.boxplot(entropies, vert=False, patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor='#E0F2FE', color=CYAN, linewidth=1.5),
                        medianprops=dict(color=NAVY, linewidth=2.2),
                        whiskerprops=dict(color=SLATE, linewidth=1.3),
                        capprops=dict(color=SLATE, linewidth=1.3),
                        flierprops=dict(marker='.', color=SLATE, alpha=0.3, markersize=3))
    ax_box.set_facecolor(LIGHT_BG)
    ax_box.set_yticks([])
    ax_box.grid(True, linestyle='--', alpha=0.4, color=GRID_COLOR, axis='x')
    ax_box.set_title('ResUNet-2D Predictive Entropy Distribution (Val_eval N=6,874 Non-Empty Slices)',
                     fontsize=13, fontweight='bold', color=NAVY, pad=12)

    # Threshold line on boxplot
    ax_box.axvline(x=0.35, color=CORAL, linestyle='--', linewidth=2.0)

    # Bottom: Histogram
    counts, bin_edges, patches_hist = ax_hist.hist(entropies, bins=50, color=CYAN, alpha=0.6, edgecolor='white', density=True)
    
    # Shade regions
    ax_hist.axvspan(0.0, 0.35, color='#F0FDF4', alpha=0.6, label='ACCEPT Zone (H < τ = 0.35)')
    ax_hist.axvspan(0.35, 0.65, color='#FFF1F2', alpha=0.6, label='REFERRAL / FLAG_UNRELIABLE Zone (H ≥ 0.35)')
    ax_hist.axvline(x=0.35, color=CORAL, linestyle='--', linewidth=2.2, label='Policy Threshold τ = 0.35')

    # Re-draw histogram on top of span
    ax_hist.hist(entropies, bins=50, color=CYAN, alpha=0.75, edgecolor='white', density=True)

    # Mean and Median lines
    mean_val = np.mean(entropies)
    median_val = np.median(entropies)
    ax_hist.axvline(x=mean_val, color=NAVY, linestyle='-', linewidth=1.8, label=f'Mean: {mean_val:.4f}')
    ax_hist.axvline(x=median_val, color=TEAL, linestyle='-.', linewidth=1.8, label=f'Median: {median_val:.4f}')

    # Stats text box
    q1 = np.percentile(entropies, 25)
    q3 = np.percentile(entropies, 75)
    pct_above_tau = (np.sum(entropies >= 0.35) / non_empty) * 100

    stats_str = (
        f"Population Metrics (Val_eval):\n"
        f"• Total Slices: {total:,}\n"
        f"• Non-Empty Slices: {non_empty:,} ({non_empty/total*100:.2f}%)\n"
        f"• Empty Slices: {empty_slices:,} ({empty_slices/total*100:.2f}%)\n"
        f"• Mean ± Std: {mean_val:.4f} ± {np.std(entropies):.4f}\n"
        f"• Median [IQR]: {median_val:.4f} [{q1:.4f}, {q3:.4f}]\n"
        f"• Min / Max: {np.min(entropies):.4f} / {np.max(entropies):.4f}\n"
        f"• Non-Empty with H ≥ 0.35: {pct_above_tau:.2f}% (Triage Ref.)"
    )
    ax_hist.text(0.68, 0.95, stats_str, transform=ax_hist.transAxes, fontsize=9.5,
                 verticalalignment='top', bbox=dict(boxstyle='round,pad=0.6', facecolor='white', edgecolor=NAVY, alpha=0.92))

    ax_hist.set_xlabel('Normalized Foreground Shannon Predictive Entropy (H)', fontsize=11, color=SLATE)
    ax_hist.set_ylabel('Probability Density', fontsize=11, color=SLATE)
    ax_hist.set_xlim(0, 0.65)
    ax_hist.grid(True, linestyle='--', alpha=0.6, color=GRID_COLOR)
    ax_hist.set_facecolor(LIGHT_BG)
    ax_hist.legend(loc='upper left', fontsize=9.5, frameon=True, facecolor='white', edgecolor=GRID_COLOR)

    out_file = os.path.join(ASSETS_DIR, 'entropy_distribution.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


# -----------------------------------------------------------------------------
# 5. Architecture Diagram (ResUNet-2D Detailed Block Diagram)
# -----------------------------------------------------------------------------
def plot_architecture_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), facecolor='white', dpi=300)
    ax.set_facecolor(LIGHT_BG)

    # Title & Metadata Card
    ax.text(0.5, 0.96, "ResUNet-2D: 5-Stage Deep Residual U-Net Architecture",
            ha='center', va='center', fontsize=16, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.92, "Input: 4-Channel mpMRI (T1n, T1c, T2w, T2f) @ 240×240 | Output: 4-Class Segmentation Logits | Parameters: 8,114,604",
            ha='center', va='center', fontsize=10.5, color=SLATE, transform=ax.transAxes)

    def draw_box(x, y, w, h, title, subtitle, color, text_color='white', edge_color=None):
        if edge_color is None:
            edge_color = color
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                                      facecolor=color, edgecolor=edge_color, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=8, color=text_color, zorder=4)

    # 1. Input Block
    draw_box(0.03, 0.72, 0.12, 0.12, "Input mpMRI", "4 ch @ 240×240", '#334155')

    # 2. Encoder Blocks (Left Column)
    # Enc1: 240x240, 32 ch
    draw_box(0.18, 0.72, 0.16, 0.12, "Encoder 1", "PreAct ResBlock (32 ch)\nStride=1 (240×240)\n10,632 params", TEAL)
    # Enc2: 120x120, 64 ch
    draw_box(0.18, 0.54, 0.16, 0.12, "Encoder 2", "PreAct ResBlock (64 ch)\nStride=2 (120×120)\n57,664 params", TEAL)
    # Enc3: 60x60, 128 ch
    draw_box(0.18, 0.36, 0.16, 0.12, "Encoder 3", "PreAct ResBlock (128 ch)\nStride=2 (60×60)\n230,016 params", TEAL)
    # Enc4: 30x30, 256 ch
    draw_box(0.18, 0.18, 0.16, 0.12, "Encoder 4", "PreAct ResBlock (256 ch)\nStride=2 (30×30)\n918,784 params", TEAL)

    # 3. Bottleneck (Bottom Center)
    # Bottleneck: 15x15, 512 ch
    draw_box(0.42, 0.05, 0.18, 0.12, "Bottleneck", "PreAct ResBlock (512 ch)\nStride=2 (15×15)\n3,672,576 params (45.3%)", NAVY)

    # 4. Decoder Blocks (Right Column)
    # Dec4: 30x30, 256 ch
    draw_box(0.68, 0.18, 0.16, 0.12, "Decoder 4", "Upconv (512→256)\nConcat Enc4 (512 ch)\nResBlock: 1,902,592 p", CYAN)
    # Dec3: 60x60, 128 ch
    draw_box(0.68, 0.36, 0.16, 0.12, "Decoder 3", "Upconv (256→128)\nConcat Enc3 (256 ch)\nResBlock: 476,160 p", CYAN)
    # Dec2: 120x120, 64 ch
    draw_box(0.68, 0.54, 0.16, 0.12, "Decoder 2", "Upconv (128→64)\nConcat Enc2 (128 ch)\nResBlock: 119,296 p", CYAN)
    # Dec1: 240x240, 32 ch
    draw_box(0.68, 0.72, 0.16, 0.12, "Decoder 1", "Upconv (64→32)\nConcat Enc1 (64 ch)\nResBlock: 29,952 p", CYAN)

    # 5. Classifier & Output
    draw_box(0.86, 0.72, 0.11, 0.12, "Classifier Head", "1×1 Conv (32→4)\nLogits @ 240×240\n132 params", '#0F172A')

    # Draw Arrows for Encoder Downsampling
    arrow_props = dict(facecolor=SLATE, edgecolor=SLATE, width=1.5, headwidth=6, shrink=0.1)
    # Input -> Enc1
    ax.annotate('', xy=(0.18, 0.78), xytext=(0.15, 0.78), arrowprops=arrow_props)
    # Enc1 -> Enc2
    ax.annotate('', xy=(0.26, 0.66), xytext=(0.26, 0.72), arrowprops=arrow_props)
    # Enc2 -> Enc3
    ax.annotate('', xy=(0.26, 0.48), xytext=(0.26, 0.54), arrowprops=arrow_props)
    # Enc3 -> Enc4
    ax.annotate('', xy=(0.26, 0.30), xytext=(0.26, 0.36), arrowprops=arrow_props)
    # Enc4 -> Bottleneck
    ax.annotate('', xy=(0.42, 0.11), xytext=(0.34, 0.18), arrowprops=arrow_props)

    # Draw Arrows for Decoder Upsampling
    # Bottleneck -> Dec4
    ax.annotate('', xy=(0.68, 0.22), xytext=(0.60, 0.11), arrowprops=arrow_props)
    # Dec4 -> Dec3
    ax.annotate('', xy=(0.76, 0.36), xytext=(0.76, 0.30), arrowprops=arrow_props)
    # Dec3 -> Dec2
    ax.annotate('', xy=(0.76, 0.54), xytext=(0.76, 0.48), arrowprops=arrow_props)
    # Dec2 -> Dec1
    ax.annotate('', xy=(0.76, 0.72), xytext=(0.76, 0.66), arrowprops=arrow_props)
    # Dec1 -> Classifier
    ax.annotate('', xy=(0.86, 0.78), xytext=(0.84, 0.78), arrowprops=arrow_props)

    # Skip Connections (Dashed horizontal lines)
    skip_props = dict(facecolor=AMBER, edgecolor=AMBER, width=1.8, headwidth=7, linestyle='--')
    for y_pos in [0.78, 0.60, 0.42, 0.24]:
        ax.annotate('', xy=(0.68, y_pos), xytext=(0.34, y_pos), arrowprops=skip_props)

    # Skip Connection Label
    ax.text(0.51, 0.795, "Long Skip Connection (Concat)", ha='center', va='bottom', fontsize=9, fontweight='semibold', color=AMBER)
    ax.text(0.51, 0.615, "Long Skip Connection (Concat)", ha='center', va='bottom', fontsize=9, fontweight='semibold', color=AMBER)
    ax.text(0.51, 0.435, "Long Skip Connection (Concat)", ha='center', va='bottom', fontsize=9, fontweight='semibold', color=AMBER)
    ax.text(0.51, 0.255, "Long Skip Connection (Concat)", ha='center', va='bottom', fontsize=9, fontweight='semibold', color=AMBER)

    # Explanatory Inset Card on Pre-Activation Residual Block
    inset_rect = patches.FancyBboxPatch((0.36, 0.38), 0.30, 0.18, boxstyle="round,pad=0.015",
                                        facecolor='white', edgecolor=NAVY, linewidth=1.5, zorder=5)
    ax.add_patch(inset_rect)
    ax.text(0.51, 0.53, "PreActResidualBlock Topology", ha='center', va='center', fontsize=9.5, fontweight='bold', color=NAVY, zorder=6)
    topo_text = (
        "• F(x): GN(8) → ReLU → Conv 3×3\n"
        "        → GN(8) → ReLU → Conv 3×3\n"
        "• Shortcut: Conv 1×1 (if downsample) else Identity\n"
        "• Output: F(x) + Shortcut(x)\n"
        "• Property: Unobstructed Linear Gradient Highway"
    )
    ax.text(0.38, 0.45, topo_text, ha='left', va='center', fontsize=8, color=SLATE, zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    out_file = os.path.join(ASSETS_DIR, 'architecture_diagram.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


# -----------------------------------------------------------------------------
# 6. Agent Flow Diagram (Stage C3 Reliability Triage Decision Tree)
# -----------------------------------------------------------------------------
def plot_agent_flow_diagram():
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white', dpi=300)
    ax.set_facecolor(LIGHT_BG)

    ax.text(0.5, 0.96, "Stage C3 v1.1.3 Reliability-Aware Agent Decision Protocol",
            ha='center', va='center', fontsize=15, fontweight='bold', color=NAVY, transform=ax.transAxes)
    ax.text(0.5, 0.92, "Three-Action Triage Policy: ACCEPT | FALLBACK_SWITCH | FLAG_UNRELIABLE",
            ha='center', va='center', fontsize=10.5, color=SLATE, transform=ax.transAxes)

    def draw_node(x, y, w, h, title, subtitle, color, shape="round", text_color='white'):
        if shape == "round":
            patch = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                          facecolor=color, edgecolor=NAVY, linewidth=1.2, zorder=3)
        elif shape == "diamond":
            verts = [
                (x + w/2, y + h),      # top
                (x + w, y + h/2),      # right
                (x + w/2, y),          # bottom
                (x, y + h/2),          # left
            ]
            patch = patches.Polygon(verts, closed=True, facecolor=color, edgecolor=NAVY, linewidth=1.5, zorder=3)
        else:
            patch = patches.Rectangle((x, y), w, h, facecolor=color, edgecolor=NAVY, linewidth=1.2, zorder=3)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h*0.60, title, ha='center', va='center', fontsize=9.5, fontweight='bold', color=text_color, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=8, color=text_color, zorder=4)

    # 1. Primary Model Input
    draw_node(0.04, 0.65, 0.18, 0.14, "Primary Champion", "ResUNet-2D (T* = 2.8805)\nInference on Slice z", NAVY)

    # 2. Decision Diamond 1: Empty Foreground?
    draw_node(0.28, 0.65, 0.16, 0.14, "Empty Foreground?", "Voxel Argmax Count = 0\n(No Tumor Detected)", '#FEF3C7', shape="diamond", text_color=NAVY)

    # 3A. Non-Empty Branch (Upwards) -> Check Uncertainty
    draw_node(0.50, 0.72, 0.18, 0.14, "Uncertainty Check", "Foreground Entropy H\nvs Threshold τ = 0.35", '#FEF3C7', shape="diamond", text_color=NAVY)

    # Action: ACCEPT
    draw_node(0.76, 0.80, 0.20, 0.12, "ACTION: ACCEPT", "H < 0.35 (Low Uncertainty)\nDeploy ResUNet-2D Mask\n(5,619 Slices / 34.5%)", '#059669')

    # Action: FLAG_UNRELIABLE (from high entropy)
    draw_node(0.76, 0.63, 0.20, 0.12, "ACTION: FLAG_UNRELIABLE", "H ≥ 0.35 (High Uncertainty)\nTriage for Expert Review", CORAL)

    # 3B. Empty Branch (Downwards) -> Check Axial Context
    draw_node(0.28, 0.35, 0.16, 0.14, "2.5D Axial Context?", "Radius k=±2 (4 slots)\nForeground Support ≥ 1?", '#FEF3C7', shape="diamond", text_color=NAVY)

    # Action: FLAG_UNRELIABLE (from empty with no axial support)
    draw_node(0.28, 0.12, 0.16, 0.12, "ACTION: FLAG_UNRELIABLE", "0 Foreground Neighbors\n(True Negative / Margin)", CORAL)

    # Query Secondary Model
    draw_node(0.50, 0.35, 0.18, 0.14, "Secondary Fallback", "UNetResNet34 (T* = 2.6261)\nEvaluate Slice z", NAVY)

    # Decision Diamond 3: Secondary has foreground?
    draw_node(0.74, 0.35, 0.16, 0.14, "Secondary Non-Empty?", "Candidate Detected\nTumor Voxel > 0?", '#FEF3C7', shape="diamond", text_color=NAVY)

    # Action: FALLBACK_SWITCH
    draw_node(0.74, 0.12, 0.22, 0.13, "ACTION: FALLBACK_SWITCH", "Adopt UNetResNet34 Mask\n(147 Slices / 0.9%)\n[+Warning if H_sec ≥ 0.35]", TEAL)

    # Secondary Empty -> FLAG_UNRELIABLE
    draw_node(0.74, 0.00, 0.22, 0.08, "Secondary Also Empty → FLAG_UNRELIABLE", "Consensus Empty / Triage", CORAL)

    # Draw Directed Connectors
    arrow_props = dict(facecolor=SLATE, edgecolor=SLATE, width=1.5, headwidth=6, shrink=0.05)

    # Primary -> Diamond 1
    ax.annotate('', xy=(0.28, 0.72), xytext=(0.22, 0.72), arrowprops=arrow_props)
    
    # Diamond 1 -> Uncertainty Check (Non-empty: NO)
    ax.annotate('', xy=(0.50, 0.79), xytext=(0.44, 0.72), arrowprops=arrow_props)
    ax.text(0.44, 0.77, "NO (Non-Empty)", fontsize=8.5, fontweight='bold', color=SLATE)

    # Diamond 1 -> Axial Context (Empty: YES)
    ax.annotate('', xy=(0.36, 0.49), xytext=(0.36, 0.65), arrowprops=arrow_props)
    ax.text(0.37, 0.56, "YES (Empty)", fontsize=8.5, fontweight='bold', color=SLATE)

    # Uncertainty Check -> ACCEPT (H < 0.35)
    ax.annotate('', xy=(0.76, 0.86), xytext=(0.68, 0.82), arrowprops=arrow_props)
    ax.text(0.69, 0.86, "H < 0.35", fontsize=8.5, fontweight='bold', color='#059669')

    # Uncertainty Check -> FLAG_UNRELIABLE (H >= 0.35)
    ax.annotate('', xy=(0.76, 0.69), xytext=(0.68, 0.76), arrowprops=arrow_props)
    ax.text(0.69, 0.71, "H ≥ 0.35", fontsize=8.5, fontweight='bold', color=CORAL)

    # Axial Context -> Secondary Model (Support Confirmed: YES)
    ax.annotate('', xy=(0.50, 0.42), xytext=(0.44, 0.42), arrowprops=arrow_props)
    ax.text(0.44, 0.44, "YES (≥1 Tumor)", fontsize=8.5, fontweight='bold', color=SLATE)

    # Axial Context -> FLAG_UNRELIABLE (Support NOT Confirmed: NO)
    ax.annotate('', xy=(0.36, 0.24), xytext=(0.36, 0.35), arrowprops=arrow_props)
    ax.text(0.37, 0.29, "NO (0 Tumor)", fontsize=8.5, fontweight='bold', color=SLATE)

    # Secondary Model -> Decision Diamond 3
    ax.annotate('', xy=(0.74, 0.42), xytext=(0.68, 0.42), arrowprops=arrow_props)

    # Secondary Non-Empty -> FALLBACK_SWITCH
    ax.annotate('', xy=(0.85, 0.25), xytext=(0.82, 0.35), arrowprops=arrow_props)
    ax.text(0.85, 0.30, "YES (Non-Empty)", fontsize=8.5, fontweight='bold', color=TEAL)

    # Secondary Empty -> Secondary Also Empty
    ax.annotate('', xy=(0.85, 0.08), xytext=(0.85, 0.12), arrowprops=arrow_props)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    out_file = os.path.join(ASSETS_DIR, 'agent_flow_diagram.png')
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'[✓] Generated: {out_file}')


if __name__ == '__main__':
    plot_training_curves()
    plot_class_dice_curves()
    plot_calibration_comparison()
    plot_entropy_distribution()
    plot_architecture_diagram()
    plot_agent_flow_diagram()
    print('All 6 dossier figures successfully generated in:', ASSETS_DIR)
