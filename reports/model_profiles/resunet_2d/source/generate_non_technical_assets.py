"""
Script: generate_non_technical_assets.py
Description: Generates 16 beautifully designed, high-resolution non-technical / intuitive
             explainer cards (16:9 ratio, 1920x1080) for ResUNet-2D.
             Tailored for faculty viva panels, clinical collaborators, and interdisciplinary teammates.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.patheffects as pe

OUT_DIR = r"D:\MajorProject\reports\model_profiles\resunet_2d\assets\non_technical"
os.makedirs(OUT_DIR, exist_ok=True)

# Cohesive modern styling
BG_DARK = "#0F172A"       # Slate 900
CARD_BG = "#1E293B"       # Slate 800
CARD_BORDER = "#334155"   # Slate 700
TEXT_WHITE = "#F8FAFC"    # Slate 50
TEXT_MUTED = "#94A3B8"    # Slate 400
ACCENT_BLUE = "#38BDF8"   # Sky 400
ACCENT_GREEN = "#34D399"  # Emerald 400
ACCENT_AMBER = "#FBBF24"  # Amber 400
ACCENT_RED = "#F87171"    # Rose 400
ACCENT_PURPLE = "#C084FC" # Violet 400
ACCENT_CYAN = "#22D3EE"   # Cyan 400

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'DejaVu Sans', 'Arial'],
    'text.color': TEXT_WHITE,
    'axes.labelcolor': TEXT_MUTED,
    'figure.facecolor': BG_DARK,
    'axes.facecolor': CARD_BG,
    'savefig.facecolor': BG_DARK,
    'savefig.edgecolor': BG_DARK,
})

def create_base_card(title, subtitle, category="RESUNET-2D EXPLAINER SERIES", card_num="CARD 01/16"):
    fig = plt.figure(figsize=(16, 9), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')

    # Background canvas
    ax.add_patch(patches.Rectangle((0, 0), 16, 9, color=BG_DARK))

    # Header section
    # Category tag
    ax.add_patch(patches.FancyBboxPatch((0.8, 8.1), 3.2, 0.45, boxstyle="round,pad=0.08",
                                        facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1.2))
    ax.text(2.4, 8.325, category, color=ACCENT_BLUE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Card number tag
    ax.add_patch(patches.FancyBboxPatch((13.6, 8.1), 1.6, 0.45, boxstyle="round,pad=0.08",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))
    ax.text(14.4, 8.325, card_num, color=TEXT_MUTED, fontsize=11, fontweight='bold', ha='center', va='center')

    # Main Title & Subtitle
    ax.text(0.8, 7.6, title, color=TEXT_WHITE, fontsize=24, fontweight='bold', va='center')
    ax.text(0.8, 7.05, subtitle, color=TEXT_MUTED, fontsize=13, va='center')

    # Top accent line
    ax.plot([0.8, 15.2], [6.65, 6.65], color=CARD_BORDER, linewidth=1.2)

    # Bottom footer strip
    ax.plot([0.8, 15.2], [0.85, 0.85], color=CARD_BORDER, linewidth=1.2)
    ax.text(0.8, 0.45, "BRAIN TUMOR SEGMENTATION & RELIABILITY FRAMEWORK", color=TEXT_MUTED, fontsize=10, fontweight='bold')
    ax.text(15.2, 0.45, "PRIMARY CHAMPION MODEL  |  STAGE C AGENT", color=ACCENT_BLUE, fontsize=10, fontweight='bold', ha='right')

    return fig, ax

def add_takeaway_box(ax, x, y, w, h, icon, headline, description, accent=ACCENT_BLUE):
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=accent, linewidth=1.5))
    # Icon circle
    ax.add_patch(patches.Circle((x + 0.45, y + h - 0.45), 0.28, facecolor=accent, edgecolor='none'))
    ax.text(x + 0.45, y + h - 0.45, icon, color=BG_DARK, fontsize=11, fontweight='bold', ha='center', va='center')

    # Headline
    ax.text(x + 0.9, y + h - 0.45, headline, color=TEXT_WHITE, fontsize=13, fontweight='bold', va='center')

    # Description text
    ax.text(x + 0.25, y + 0.25, description, color=TEXT_MUTED, fontsize=10.5, va='bottom', wrap=True)

# -------------------------------------------------------------------------------------------------
# Card 01: What is ResUNet?
# -------------------------------------------------------------------------------------------------
def generate_card_01():
    fig, ax = create_base_card("What is ResUNet-2D?",
                               "The AI 'Eye' Trained to Spot Brain Tumors Slice by Slice with Residual Precision",
                               card_num="CARD 01/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Conceptual diagram of ResUNet
    # Input MRI stack -> Encoder (ResNet) -> Bottleneck -> Decoder -> Tumor Segmentation
    # Input box
    ax.add_patch(patches.Rectangle((1.2, 3.2), 1.6, 2.0, facecolor="#1E3A8A", edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(2.0, 4.3, "4-Channel\nMRI Slice", color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(2.0, 3.5, "240 x 240", color=ACCENT_CYAN, fontsize=10, ha='center')

    # Arrow 1
    ax.annotate("", xy=(3.3, 4.2), xytext=(2.9, 4.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=2.5))

    # Encoder
    ax.add_patch(patches.Polygon([[3.4, 5.4], [4.8, 4.6], [4.8, 3.8], [3.4, 3.0]],
                                 closed=True, facecolor="#065F46", edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(4.0, 4.2, "Encoder\n(ResNet)", color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Residual loops indicator
    ax.annotate("", xy=(4.6, 4.8), xytext=(3.6, 5.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color=ACCENT_AMBER, lw=2))
    ax.text(4.1, 5.5, "+ Residual Shortcut", color=ACCENT_AMBER, fontsize=9, ha='center')

    # Bottleneck
    ax.add_patch(patches.Rectangle((4.9, 3.8), 0.9, 0.8, facecolor="#581C87", edgecolor=ACCENT_PURPLE, linewidth=1.5))
    ax.text(5.35, 4.2, "Latent\nRep", color=TEXT_WHITE, fontsize=9, fontweight='bold', ha='center', va='center')

    # Decoder
    ax.add_patch(patches.Polygon([[5.9, 4.6], [7.3, 5.4], [7.3, 3.0], [5.9, 3.8]],
                                 closed=True, facecolor="#065F46", edgecolor=ACCENT_GREEN, linewidth=1.5))
    ax.text(6.7, 4.2, "Decoder\n(Up-Conv)", color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Skip Connection
    ax.annotate("", xy=(6.8, 5.7), xytext=(3.9, 5.7),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.25", color=ACCENT_CYAN, lw=2, ls='--'))
    ax.text(5.35, 6.2, "Direct Skip Connections (Preserve Tiny Details)", color=ACCENT_CYAN, fontsize=10, ha='center')

    # Arrow 2
    ax.annotate("", xy=(7.8, 4.2), xytext=(7.4, 4.2),
                arrowprops=dict(arrowstyle="->", color=ACCENT_BLUE, lw=2.5))

    # Output mask
    ax.add_patch(patches.Rectangle((7.9, 3.2), 1.2, 2.0, facecolor="#7F1D1D", edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(8.5, 4.3, "Tumor\nMask", color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center', va='center')
    ax.text(8.5, 3.5, "NCR/ED/ET", color=ACCENT_AMBER, fontsize=9, ha='center')

    # Bottom caption on graphic
    ax.text(5.05, 1.6, "Combines the U-Net segmentation shape with Deep Residual learning,\npreventing degradation while detecting fine tumor margins.",
            color=TEXT_MUTED, fontsize=11, ha='center', style='italic')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "1", "Deep Residual Architecture",
                     "Uses residual skip-blocks (8.11 Million parameters) that let\ngradients flow smoothly, allowing deep feature extraction\nwithout gradient vanishing.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "2", "High Accuracy Champion",
                     "Achieves 0.8670 Macro Mean Dice across all tumor classes\nand 0.9039 Whole Tumor Dice on 105 held-out evaluation\npatients, leading our 5-model portfolio.", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "3", "Stage C Reliability Backbone",
                     "Serves as the primary decision engine in our reliability agent,\nsupplying temperature-calibrated predictions paired with\nreal-time uncertainty estimation.", ACCENT_PURPLE)

    plt.savefig(os.path.join(OUT_DIR, "simple_01_what_is_resunet.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 02: Why Brain Tumors are Hard
# -------------------------------------------------------------------------------------------------
def generate_card_02():
    fig, ax = create_base_card("Why Brain Tumors Are So Hard to Segment",
                               "Diffuse Margins, Irregular Geometries, and Severe Class Imbalance",
                               card_num="CARD 02/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Conceptual sketch: Brain slice with tiny diffuse tumor vs huge background
    ax.add_patch(patches.Ellipse((5.0, 3.8), 5.5, 4.2, facecolor="#0B132B", edgecolor=CARD_BORDER, linewidth=2))
    ax.text(5.0, 5.4, "Brain Healthy Parenchyma (98.6% of Volume)", color=TEXT_MUTED, fontsize=11, ha='center')

    # Diffuse edema patch
    ax.add_patch(patches.Ellipse((5.8, 3.6), 2.4, 1.8, facecolor="#10B981", alpha=0.35, edgecolor=ACCENT_GREEN, linewidth=1.5, ls=':'))
    ax.text(6.4, 4.3, "Infiltrating Edema\n(Fuzzy Boundaries)", color=ACCENT_GREEN, fontsize=10, ha='center')

    # Enhancing core
    ax.add_patch(patches.Circle((5.4, 3.4), 0.7, facecolor="#EF4444", alpha=0.6, edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(5.4, 3.4, "Active\nTumor", color=TEXT_WHITE, fontsize=10, fontweight='bold', ha='center', va='center')

    # Necrotic center
    ax.add_patch(patches.Circle((5.3, 3.3), 0.25, facecolor="#F59E0B", edgecolor=ACCENT_AMBER, linewidth=1.2))

    # Callout line for Class Imbalance
    ax.annotate("Extreme Class Imbalance:\nTumor is <1.4% of total brain voxels",
                xy=(5.4, 2.7), xytext=(2.6, 1.8),
                arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=1.5),
                color=ACCENT_AMBER, fontsize=10, fontweight='bold')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "A", "Infiltrative 'Tentacles'",
                     "Gliomas do not have neat borders; microscopic cancer cells\ninfiltrate healthy brain tissues like tendrils, making\nvisual boundary detection ambiguous even for radiologists.", ACCENT_RED)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "B", "The Needle in the Haystack Problem",
                     "98.6% of all voxels are healthy tissue or air. A naive model\npredicting 'all healthy' gets 98.6% accuracy but 0.0% utility.\nResUNet uses Combined Cross-Entropy + Dice loss to fix this.", ACCENT_AMBER)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "C", "Heterogeneous Multimodal Signal",
                     "No single MRI scan shows the complete picture. Necrosis shows\non T1, active tumor on T1c, and edema on FLAIR. The AI must\nfuse all 4 simultaneous channels to make a safe call.", ACCENT_CYAN)

    plt.savefig(os.path.join(OUT_DIR, "simple_02_why_brain_tumors_are_hard.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 03: Four MRI Modalities Explained
# -------------------------------------------------------------------------------------------------
def generate_card_03():
    fig, ax = create_base_card("The Four MRI Modalities Explained Simply",
                               "How Combining 4 Different MRI Sequences Gives AI 'Super-Vision'",
                               card_num="CARD 03/16")

    # 4 distinct visual blocks side by side
    mods = [
        ("T1-Native (T1n)", "Anatomical Baseline", "#1E293B", ACCENT_BLUE,
         "Shows standard brain structure and gray/white matter boundaries.\nTumor appears as dark hypointense distortion."),
        ("T1-Contrast (T1c)", "Active Tumor Core", "#3B1E2B", ACCENT_RED,
         "Gadolinium contrast leaks through disrupted blood-brain barrier.\nActive enhancing tumor glows brightly!"),
        ("T2-Weighted (T2w)", "Fluid & Swelling", "#1E3B2B", ACCENT_GREEN,
         "Water and CSF light up bright white. Identifies overall edema,\nfluid accumulation, and tissue inflammation."),
        ("T2-FLAIR (T2f)", "Edema Without Clutter", "#3B351E", ACCENT_AMBER,
         "Suppresses dark free water/CSF while keeping tissue fluid bright.\nCrucial for tracing outer infiltrating tumor boundaries.")
    ]

    for i, (name, role, bg, accent, desc) in enumerate(mods):
        x = 0.8 + i * 3.65
        ax.add_patch(patches.FancyBboxPatch((x, 1.3), 3.45, 5.1, boxstyle="round,pad=0.1",
                                            facecolor=bg, edgecolor=accent, linewidth=1.8))
        # Top badge
        ax.add_patch(patches.FancyBboxPatch((x + 0.3, 5.8), 2.85, 0.45, boxstyle="round,pad=0.05",
                                            facecolor=CARD_BG, edgecolor=accent, linewidth=1))
        ax.text(x + 1.725, 6.025, name, color=accent, fontsize=11, fontweight='bold', ha='center', va='center')

        # Role
        ax.text(x + 1.725, 5.4, role, color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center')

        # Conceptual MRI Box
        ax.add_patch(patches.Rectangle((x + 0.6, 3.4), 2.25, 1.7, facecolor="#0B132B", edgecolor=CARD_BORDER, linewidth=1))
        ax.text(x + 1.725, 4.25, f"[MRI Image\nChannel {i+1}]", color=TEXT_MUTED, fontsize=10, ha='center', va='center')

        # Description
        ax.text(x + 0.25, 1.6, desc, color=TEXT_MUTED, fontsize=10, va='bottom')

    plt.savefig(os.path.join(OUT_DIR, "simple_03_four_mri_modalities_explained.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 04: Three Tumor Regions Explained
# -------------------------------------------------------------------------------------------------
def generate_card_04():
    fig, ax = create_base_card("The Three Tumor Sub-Regions Explained",
                               "How Clinicians Decompose Complex Gliomas into Actionable Targets",
                               card_num="CARD 04/16")

    # Left: Nested Tumor Graphic
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Outer: Peritumoral Edema
    ax.add_patch(patches.Ellipse((5.0, 3.7), 6.5, 4.4, facecolor="#10B981", alpha=0.3, edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(5.0, 5.5, "Whole Tumor (WT): 0.9039 Dice", color=ACCENT_GREEN, fontsize=13, fontweight='bold', ha='center')

    # Middle: Non-Enhancing / Necrosis + Enhancing = Tumor Core
    ax.add_patch(patches.Ellipse((5.0, 3.5), 4.2, 2.9, facecolor="#F59E0B", alpha=0.35, edgecolor=ACCENT_AMBER, linewidth=2))
    ax.text(5.0, 4.5, "Tumor Core (TC): 0.8597 Dice", color=ACCENT_AMBER, fontsize=12, fontweight='bold', ha='center')

    # Center: Enhancing Tumor
    ax.add_patch(patches.Circle((5.0, 3.2), 1.0, facecolor="#EF4444", alpha=0.6, edgecolor=ACCENT_RED, linewidth=2))
    ax.text(5.0, 3.2, "Enhancing (ET)\n0.8286 Dice", color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "ET", "Enhancing Tumor (ET) — Red",
                     "The most aggressive, actively dividing cancerous cells with\nleaky blood vessels. Primary target for surgical resection\nand localized radiotherapy.", ACCENT_RED)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "TC", "Tumor Core (TC = NCR + ET) — Amber",
                     "Includes dead tissue (necrosis) starved of oxygen plus the\nactive tumor. Identifies the bulk mass that surgeons aim\nto debulk without injuring eloquent brain tracts.", ACCENT_AMBER)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "WT", "Whole Tumor (WT = ED + NCR + ET) — Green",
                     "Encompasses all swollen infiltrating edema plus the core.\nMarks the entire compromised territory used by oncologists\nto plan safety margins for radiation fields.", ACCENT_GREEN)

    plt.savefig(os.path.join(OUT_DIR, "simple_04_three_tumor_regions_explained.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 05: Encoder-Decoder Concept
# -------------------------------------------------------------------------------------------------
def generate_card_05():
    fig, ax = create_base_card("The Encoder-Decoder Concept: Funnel & Reconstruct",
                               "How Neural Networks Compress Big Pictures into Core Meaning, Then Paint Pixel Maps",
                               card_num="CARD 05/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Funnel sketch
    ax.add_patch(patches.Polygon([[1.5, 5.5], [4.5, 4.2], [4.5, 3.2], [1.5, 1.9]],
                                 closed=True, facecolor="#1E3A8A", edgecolor=ACCENT_BLUE, linewidth=2))
    ax.text(3.0, 3.7, "ENCODER\nCompresses 'Where'\ninto 'What'", color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center', va='center')

    # Bottleneck
    ax.add_patch(patches.Rectangle((4.6, 3.2), 0.9, 1.0, facecolor="#581C87", edgecolor=ACCENT_PURPLE, linewidth=2))
    ax.text(5.05, 3.7, "Latent\nCode", color=TEXT_WHITE, fontsize=9, fontweight='bold', ha='center', va='center')

    # Decoder expanding
    ax.add_patch(patches.Polygon([[5.6, 4.2], [8.6, 5.5], [8.6, 1.9], [5.6, 3.2]],
                                 closed=True, facecolor="#065F46", edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(7.1, 3.7, "DECODER\nReconstructs 'What'\nback to 'Where'", color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center', va='center')

    # Skip arrow over the top
    ax.annotate("", xy=(7.5, 5.6), xytext=(2.5, 5.6),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color=ACCENT_AMBER, lw=2.5))
    ax.text(5.0, 6.2, "Skip Connections: 'Don't Forget the Exact Coordinates!'", color=ACCENT_AMBER, fontsize=11, fontweight='bold', ha='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "1", "Step 1: Abstracting Meaning",
                     "The encoder downsamples the image 4 times (240 -> 120 -> 60\n-> 30 -> 15). At 15x15, individual pixels vanish, but the network\nknows: 'there is an aggressive glioma in the frontal lobe.'", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "2", "Step 2: Rebuilding Geometry",
                     "The decoder upsamples back to full 240x240 resolution. It\ntakes the high-level tumor understanding and paints it into\nexact voxel classifications.", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "3", "The Secret Sauce: Skip Connections",
                     "Compression loses sharp edges. Skip connections act like direct\ntelephones from early layers to late layers, copying exact\nmicroscopic boundary locations across the network.", ACCENT_AMBER)

    plt.savefig(os.path.join(OUT_DIR, "simple_05_encoder_decoder_concept.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 06: Residual Connections Explained
# -------------------------------------------------------------------------------------------------
def generate_card_06():
    fig, ax = create_base_card("Residual Connections: The Secret to Going Deep",
                               "Learning the Difference (F(x) = H(x) - x) Instead of Learning from Scratch",
                               card_num="CARD 06/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Flow: Input x -> Conv 1 -> Conv 2 -> Addition (+) -> Output x + F(x)
    ax.add_patch(patches.Rectangle((1.5, 3.2), 1.2, 1.0, facecolor=CARD_BG, edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(2.1, 3.7, "Input\nx", color=TEXT_WHITE, fontsize=13, fontweight='bold', ha='center', va='center')

    # Arrow to Conv 1
    ax.annotate("", xy=(3.5, 3.7), xytext=(2.7, 3.7),
                arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, lw=2))

    # Conv 1 & 2
    ax.add_patch(patches.Rectangle((3.6, 3.1), 1.4, 1.2, facecolor="#1E3A8A", edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(4.3, 3.7, "3x3 Conv\n+ BN + ReLU", color=TEXT_WHITE, fontsize=10, fontweight='bold', ha='center', va='center')

    ax.annotate("", xy=(5.7, 3.7), xytext=(5.0, 3.7),
                arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, lw=2))

    ax.add_patch(patches.Rectangle((5.8, 3.1), 1.4, 1.2, facecolor="#1E3A8A", edgecolor=ACCENT_BLUE, linewidth=1.5))
    ax.text(6.5, 3.7, "3x3 Conv\n+ BN", color=TEXT_WHITE, fontsize=10, fontweight='bold', ha='center', va='center')

    # Adder circle (+)
    ax.add_patch(patches.Circle((8.0, 3.7), 0.45, facecolor="#065F46", edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(8.0, 3.7, "+", color=TEXT_WHITE, fontsize=18, fontweight='bold', ha='center', va='center')

    # Main path into adder
    ax.annotate("", xy=(7.55, 3.7), xytext=(7.2, 3.7),
                arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, lw=2))

    # Shortcut connection bypass arc
    ax.annotate("", xy=(8.0, 4.15), xytext=(2.1, 4.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.4", color=ACCENT_AMBER, lw=3))
    ax.text(5.0, 5.3, "Shortcut: Passes Identity 'x' Directly!", color=ACCENT_AMBER, fontsize=12, fontweight='bold', ha='center')

    # Output
    ax.text(8.8, 3.7, "= F(x) + x", color=ACCENT_GREEN, fontsize=13, fontweight='bold', va='center')

    # Bottom caption
    ax.text(5.05, 1.6, "If the layers learn nothing new, the block safely defaults to identity.\nThis prevents deep neural networks from degrading during training.",
            color=TEXT_MUTED, fontsize=11, ha='center', style='italic')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "A", "The Vanishing Gradient Problem",
                     "In traditional 20+ layer networks, mathematical signals shrink\nto zero during backpropagation. Early layers receive no learning\nsignal and remain completely random.", ACCENT_RED)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "B", "The Highway Analogy",
                     "Residual shortcuts act like express highway lanes. Gradients\nzoom back through the additions without being multiplied by\ntiny fractions, enabling stable training.", ACCENT_AMBER)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "C", "Why ResUNet Wins",
                     "By embedding residual blocks inside the U-Net, ResUNet-2D\nreaches peak validation accuracy at Epoch 13 (0.8670 Macro\nDice), outperforming plain convolution backbones.", ACCENT_GREEN)

    plt.savefig(os.path.join(OUT_DIR, "simple_06_residual_connections_explained.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 07: What Dice Score Means
# -------------------------------------------------------------------------------------------------
def generate_card_07():
    fig, ax = create_base_card("What Does a 'Dice Score' Mean in Practice?",
                               "Understanding Overlap Metrics Between AI Prediction and Ground Truth",
                               card_num="CARD 07/16")

    # Left Graphic Box: Two overlapping circles (Venn diagram)
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Circle A: Ground Truth (Radiologist)
    ax.add_patch(patches.Circle((4.3, 3.8), 1.8, facecolor="#3B82F6", alpha=0.4, edgecolor=ACCENT_BLUE, linewidth=2))
    ax.text(3.4, 4.2, "Ground Truth\n(Expert)", color=ACCENT_BLUE, fontsize=12, fontweight='bold', ha='center')

    # Circle B: AI Prediction
    ax.add_patch(patches.Circle((5.7, 3.8), 1.8, facecolor="#10B981", alpha=0.4, edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(6.6, 4.2, "ResUNet-2D\nPrediction", color=ACCENT_GREEN, fontsize=12, fontweight='bold', ha='center')

    # Center Overlap
    ax.text(5.0, 3.8, "INTERSECTION\n(A ∩ B)", color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Mathematical Formula Box
    ax.add_patch(patches.FancyBboxPatch((2.2, 1.5), 5.7, 0.9, boxstyle="round,pad=0.08",
                                        facecolor="#0B132B", edgecolor=ACCENT_AMBER, linewidth=1.5))
    ax.text(5.05, 1.95, r"Dice Similarity Coefficient = $\frac{2 \times |A \cap B|}{|A| + |B|}$",
            color=ACCENT_AMBER, fontsize=14, ha='center', va='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "1.0", "1.0 = Flawless Match",
                     "A score of 1.0 means the predicted tumor outline exactly\nmatches the ground truth down to every single voxel.\n0.0 means completely disjoint (zero overlap).", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "0.86", "ResUNet-2D = 0.8641 (Whole Cohort)",
                     "Our model achieves 0.9039 for Whole Tumor and 0.8597 for\nTumor Core. In medical imaging literature, any Dice above 0.80\nis considered excellent clinical congruence.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Note", "Why Dice Trumps Raw Accuracy",
                     "Because tumors make up only 1.4% of the brain, a dummy model\nwith 0% tumor overlap still scores 98.6% raw accuracy!\nDice penalizes both false positives and false negatives equally.", ACCENT_AMBER)

    plt.savefig(os.path.join(OUT_DIR, "simple_07_what_dice_score_means.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 08: What HD95 Means
# -------------------------------------------------------------------------------------------------
def generate_card_08():
    fig, ax = create_base_card("What is HD95? Measuring Boundary Distance in Millimeters",
                               "95th Percentile Hausdorff Distance: Ensuring the AI Doesn't Stray from the Real Edge",
                               card_num="CARD 08/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Two wavy contours representing surface boundaries
    t = np.linspace(0, 2*np.pi, 100)
    r1 = 1.6 + 0.15 * np.sin(4*t)
    x1 = 5.0 + r1 * np.cos(t)
    y1 = 3.8 + r1 * np.sin(t)

    r2 = 1.9 + 0.25 * np.cos(3*t)
    x2 = 5.0 + r2 * np.cos(t)
    y2 = 3.8 + r2 * np.sin(t)

    ax.plot(x1, y1, color=ACCENT_BLUE, linewidth=2.5, label="Expert Boundary")
    ax.plot(x2, y2, color=ACCENT_GREEN, linewidth=2.5, linestyle='--', label="Predicted Boundary")

    # Distance caliper
    ax.annotate("", xy=(6.9, 4.3), xytext=(6.5, 4.0),
                arrowprops=dict(arrowstyle="<->", color=ACCENT_AMBER, lw=2.5))
    ax.text(7.1, 4.0, "HD95 Distance\n(in mm)", color=ACCENT_AMBER, fontsize=10, fontweight='bold', va='center')

    # Outlier spike
    ax.plot([3.2, 2.5], [4.5, 5.2], color=ACCENT_RED, linewidth=1.5, ls=':')
    ax.text(2.3, 5.4, "5% Outliers Discarded\n(Spurious Noise)", color=ACCENT_RED, fontsize=9, ha='center')

    ax.legend(loc="lower center", facecolor=BG_DARK, edgecolor=CARD_BORDER, fontsize=10)

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "mm", "Physical Millimeter Metric",
                     "Unlike Dice (unitless 0-1), HD95 measures physical distance\nin millimeters (mm) between the predicted boundary and the\ntrue anatomical tumor surface.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "95%", "Why 95% Instead of 100% Max?",
                     "Maximum Hausdorff distance is ruined by a single stray false-\npositive speck miles away. The 95th percentile clips extreme\n1-pixel outliers to reflect the true operational margin.", ACCENT_AMBER)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Med", "Median HD95: 2.0 mm to 4.1 mm",
                     "On 105 held-out patients, ResUNet-2D achieves a median HD95\nof just 2.00 mm for Enhancing Tumor, 3.16 mm for Tumor Core,\nand 4.12 mm for Whole Tumor.", ACCENT_GREEN)

    plt.savefig(os.path.join(OUT_DIR, "simple_08_what_hd95_means.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 09: Why AI Gets Overconfident
# -------------------------------------------------------------------------------------------------
def generate_card_09():
    fig, ax = create_base_card("Why Deep Neural Networks Get Dangerously Overconfident",
                               "Modern Deep Learning Maximizes Accuracy by Exaggerating Output Confidence",
                               card_num="CARD 09/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Bar chart showing Uncalibrated confidence (99%) vs actual accuracy (75%)
    bars = [("Uncalibrated Confidence", 0.96, ACCENT_RED), ("Actual Correct Rate", 0.76, ACCENT_BLUE)]
    for i, (label, val, col) in enumerate(bars):
        y = 4.2 - i * 1.5
        ax.add_patch(patches.Rectangle((1.5, y), 6.5 * val, 0.9, facecolor=col, edgecolor='none'))
        ax.text(1.7, y + 0.45, label, color=TEXT_WHITE, fontsize=12, fontweight='bold', va='center')
        ax.text(1.5 + 6.5 * val + 0.2, y + 0.45, f"{val*100:.0f}%", color=col, fontsize=14, fontweight='bold', va='center')

    # Warning banner
    ax.add_patch(patches.FancyBboxPatch((1.5, 1.5), 7.1, 0.9, boxstyle="round,pad=0.08",
                                        facecolor="#7F1D1D", edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(5.05, 1.95, "DANGER: AI says '99% confident' even when it is totally wrong!",
            color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "!", "The Softmax Illusion",
                     "Modern cross-entropy loss pushes model logits to extreme\nvalues so softmax probabilities hit 0.99 or 0.00. The network\nlearns to sound utterly confident even on boundary guesses.", ACCENT_RED)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "Med", "Fatal in Medical Diagnosis",
                     "If a surgeon relies on an uncalibrated 98% confidence score\nto excise brain tissue, a false positive means permanently\ndamaging healthy speech or motor cortex.", ACCENT_AMBER)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Fix", "The Solution: Post-Hoc Calibration",
                     "Before trusting ResUNet-2D in clinical triage, we perform\nrigorous Temperature Scaling (T* = 2.8805) to restore honest,\nstatistically reliable probabilities.", ACCENT_GREEN)

    plt.savefig(os.path.join(OUT_DIR, "simple_09_why_ai_gets_overconfident.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 10: Temperature Scaling Visualized
# -------------------------------------------------------------------------------------------------
def generate_card_10():
    fig, ax = create_base_card("Temperature Scaling: Cooling Down the AI's Ego",
                               "A Thermostat That Softens Logits Without Changing Which Class Wins",
                               card_num="CARD 10/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Thermometer graphic
    ax.add_patch(patches.Rectangle((1.5, 2.0), 0.6, 3.5, facecolor="#0B132B", edgecolor=CARD_BORDER, linewidth=1.5))
    ax.add_patch(patches.Circle((1.8, 1.8), 0.55, facecolor=ACCENT_RED, edgecolor='none'))
    ax.add_patch(patches.Rectangle((1.6, 2.0), 0.4, 2.2, facecolor=ACCENT_RED, edgecolor='none'))
    ax.text(1.8, 5.8, "T* = 2.88", color=ACCENT_AMBER, fontsize=12, fontweight='bold', ha='center')

    # Probability redistribution bars
    # Before (T=1.0)
    ax.text(3.5, 5.4, "Before Calibration (T=1.0)", color=ACCENT_RED, fontsize=11, fontweight='bold')
    ax.add_patch(patches.Rectangle((3.5, 4.4), 4.5 * 0.92, 0.6, facecolor=ACCENT_RED, alpha=0.8))
    ax.text(3.7, 4.7, "Tumor: 92% (Overconfident)", color=TEXT_WHITE, fontsize=10, va='center')
    ax.add_patch(patches.Rectangle((3.5, 3.6), 4.5 * 0.08, 0.6, facecolor=TEXT_MUTED, alpha=0.5))
    ax.text(4.0, 3.9, "Healthy: 8%", color=TEXT_WHITE, fontsize=10, va='center')

    # Down arrow
    ax.annotate("", xy=(5.5, 2.9), xytext=(5.5, 3.4),
                arrowprops=dict(arrowstyle="->", color=ACCENT_AMBER, lw=2.5))
    ax.text(5.9, 3.15, "Divide by T = 2.8805", color=ACCENT_AMBER, fontsize=10, fontweight='bold', va='center')

    # After (T=2.88)
    ax.text(3.5, 2.5, "After Calibration (T=2.8805)", color=ACCENT_GREEN, fontsize=11, fontweight='bold')
    ax.add_patch(patches.Rectangle((3.5, 1.7), 4.5 * 0.74, 0.6, facecolor=ACCENT_GREEN, alpha=0.8))
    ax.text(3.7, 2.0, "Tumor: 74% (Honest Probability)", color=TEXT_WHITE, fontsize=10, va='center')
    ax.add_patch(patches.Rectangle((3.5 + 4.5*0.74, 1.7), 4.5 * 0.26, 0.6, facecolor=TEXT_MUTED, alpha=0.5))
    ax.text(3.5 + 4.5*0.74 + 0.2, 2.0, "Healthy: 26%", color=TEXT_WHITE, fontsize=10, va='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "T", "Single Learnable Parameter",
                     "Temperature scaling divides raw model outputs (logits) by\na scalar T > 1 before softmax. It does not alter argmax,\nso model accuracy remains 100% identical.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "ECE", "Expected Calibration Error Drops 60%",
                     "On 105 held-out evaluation patients, Expected Calibration\nError plummets from 8.59% down to 3.41%. Negative Log\nLikelihood drops from 1.0277 to 0.4415.", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Safe", "Enables Safe Clinical Thresholding",
                     "Once probabilities reflect true empirical frequencies, our\nreliability agent can compute genuine entropy and safely reject\nuncertain slices before they reach doctors.", ACCENT_PURPLE)

    plt.savefig(os.path.join(OUT_DIR, "simple_10_temperature_scaling_visualized.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 11: Entropy as Traffic Light
# -------------------------------------------------------------------------------------------------
def generate_card_11():
    fig, ax = create_base_card("Predictive Entropy as a Clinical Traffic Light",
                               "How Shannon Entropy Warns Clinicians When the Model is Confused",
                               card_num="CARD 11/16")

    # Left Graphic Box: Traffic Light
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Traffic light housing
    ax.add_patch(patches.FancyBboxPatch((1.6, 1.5), 1.8, 4.5, boxstyle="round,pad=0.15",
                                        facecolor="#0B132B", edgecolor=CARD_BORDER, linewidth=2))

    # Red Light (High Entropy >= 0.35)
    ax.add_patch(patches.Circle((2.5, 4.9), 0.55, facecolor=ACCENT_RED, edgecolor='white', linewidth=1.5))
    ax.text(3.8, 4.9, "RED: H >= 0.35\nHigh Uncertainty / Conflict", color=ACCENT_RED, fontsize=12, fontweight='bold', va='center')

    # Amber Light (Borderline / Transition)
    ax.add_patch(patches.Circle((2.5, 3.7), 0.55, facecolor=ACCENT_AMBER, edgecolor='none'))
    ax.text(3.8, 3.7, "AMBER: H ≈ 0.27 - 0.34\nBorderline Confidence", color=ACCENT_AMBER, fontsize=11, fontweight='bold', va='center')

    # Green Light (Low Entropy < 0.35)
    ax.add_patch(patches.Circle((2.5, 2.5), 0.55, facecolor=ACCENT_GREEN, edgecolor='white', linewidth=1.5))
    ax.text(3.8, 2.5, "GREEN: H < 0.35\nCrisp Consensus / High Certainty", color=ACCENT_GREEN, fontsize=12, fontweight='bold', va='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "H", "What is Shannon Entropy?",
                     "Entropy measures information dispersion. If the AI is 99%\nsure of tumor, entropy is near 0. If it is torn 25%-25%-25%-25%\nacross classes, entropy hits maximum 1.386 nats.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "τ", "The Frozen Threshold (τ = 0.35)",
                     "Calibrated against Stage C validation cohorts. Slices with\nH < 0.35 represent dependable segmentations with high\nmean Dice (0.7803), safely passed to ACCEPT.", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Flag", "Automatic Red Flagging",
                     "When entropy exceeds 0.35, the AI admits its uncertainty and\nflags the slice for human expert review, preventing rogue\npredictions from entering medical workflows.", ACCENT_RED)

    plt.savefig(os.path.join(OUT_DIR, "simple_11_entropy_as_traffic_light.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 12: Axial Context Sandwich
# -------------------------------------------------------------------------------------------------
def generate_card_12():
    fig, ax = create_base_card("The Axial Context Sandwich: 3D Anatomical Sanity Check",
                               "Checking Neighboring MRI Slices to Ensure Tumors Don't Vanish into Thin Air",
                               card_num="CARD 12/16")

    # Left Graphic Box: Stack of 5 Slices (z-2 to z+2)
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    slices = [
        ("Slice z+2", "#1E293B", ACCENT_BLUE, "Upper Neighbor (+2)"),
        ("Slice z+1", "#1E293B", ACCENT_BLUE, "Upper Neighbor (+1)"),
        ("Slice z (Current)", "#3B1E2B", ACCENT_AMBER, "Target Slice (Empty Primary!)"),
        ("Slice z-1", "#1E293B", ACCENT_BLUE, "Lower Neighbor (-1)"),
        ("Slice z-2", "#1E293B", ACCENT_BLUE, "Lower Neighbor (-2)")
    ]

    for i, (name, bg, col, role) in enumerate(slices):
        y = 5.3 - i * 0.95
        ax.add_patch(patches.FancyBboxPatch((1.5, y), 3.2, 0.75, boxstyle="round,pad=0.08",
                                            facecolor=bg, edgecolor=col, linewidth=1.5))
        ax.text(3.1, y + 0.375, name, color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center', va='center')
        ax.text(5.0, y + 0.375, role, color=col, fontsize=11, va='center')

    # Connecting bracket
    ax.plot([4.9, 4.8, 4.8, 4.9], [5.6, 5.6, 1.8, 1.8], color=ACCENT_CYAN, linewidth=2)
    ax.text(5.5, 1.4, "Radius r = 2\nRequires >= 1 Foreground Neighbor", color=ACCENT_CYAN, fontsize=10, fontweight='bold')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "3D", "Biology Exists in 3D Space",
                     "A 3D glioblastoma cannot suddenly disappear in a single 1mm\nslice and reappear in the next. An isolated empty slice nestled\nbetween tumor slices is almost certainly a false negative.", ACCENT_BLUE)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "Rule", "The Axial Support Rule",
                     "Our Stage C agent checks a 5-slice window [z-2, z-1, z, z+1, z+2].\nIf at least one neighbor contains confirmed tumor, axial\nsupport is verified.", ACCENT_GREEN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "Act", "Triggers Secondary Rescue",
                     "Confirmed axial support grants the agent permission to query\nour secondary specialist (UNetResNet34) to rescue the\nmissed tumor slice.", ACCENT_AMBER)

    plt.savefig(os.path.join(OUT_DIR, "simple_12_axial_context_sandwich.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 13: Three-Action Triage Decision
# -------------------------------------------------------------------------------------------------
def generate_card_13():
    fig, ax = create_base_card("The Three-Action Reliability Triage System",
                               "How the Stage C Agent Sorts Every Slice into Safe Decisions",
                               card_num="CARD 13/16")

    # 3 distinct columns for ACCEPT, FALLBACK_SWITCH, FLAG_UNRELIABLE
    actions = [
        ("ACCEPT", "Autonomous Deployment", "#064E3B", ACCENT_GREEN,
         "5,619 Slices (34.5%)\n(81.7% of non-empty slices)",
         "• Primary model predicted tumor\n• Calibrated entropy H < 0.35\n• High agreement & certainty\n• Deployed without human review"),
        ("FALLBACK_SWITCH", "Specialist Rescue", "#78350F", ACCENT_AMBER,
         "147 Slices (0.90%)\n(Rescues 54 True Tumors)",
         "• Primary predicted empty\n• Axial context confirmed tumor\n• Secondary UNetResNet34 non-empty\n• Rescued with secondary mask"),
        ("FLAG_UNRELIABLE", "Expert Clinical Referral", "#7F1D1D", ACCENT_RED,
         "10,509 Slices (64.6%)\n(Includes 9,401 true healthy)",
         "• Primary H >= 0.35 (uncertain)\n• Empty without axial support\n• Secondary empty or no context\n• Safeguards patient safety")
    ]

    for i, (act, role, bg, accent, stats, desc) in enumerate(actions):
        x = 0.8 + i * 4.9
        ax.add_patch(patches.FancyBboxPatch((x, 1.2), 4.6, 5.2, boxstyle="round,pad=0.1",
                                            facecolor=bg, edgecolor=accent, linewidth=2))
        # Action title badge
        ax.add_patch(patches.FancyBboxPatch((x + 0.3, 5.7), 4.0, 0.55, boxstyle="round,pad=0.08",
                                            facecolor=CARD_BG, edgecolor=accent, linewidth=1.5))
        ax.text(x + 2.3, 5.975, act, color=accent, fontsize=14, fontweight='bold', ha='center', va='center')

        # Subtitle
        ax.text(x + 2.3, 5.3, role, color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center')

        # Stats banner
        ax.add_patch(patches.Rectangle((x + 0.3, 4.3), 4.0, 0.75, facecolor="#0B132B", edgecolor=CARD_BORDER))
        ax.text(x + 2.3, 4.675, stats, color=ACCENT_CYAN, fontsize=10.5, fontweight='bold', ha='center', va='center')

        # Bullet points
        ax.text(x + 0.4, 1.6, desc, color=TEXT_WHITE, fontsize=11, va='bottom', linespacing=1.6)

    plt.savefig(os.path.join(OUT_DIR, "simple_13_three_action_triage_decision.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 14: Why Fallback Saves Empty Predictions
# -------------------------------------------------------------------------------------------------
def generate_card_14():
    fig, ax = create_base_card("Why Fallback Rescues Missed Tumors",
                               "How Teaming Up Two Complementary AI Models Closes False-Negative Gaps",
                               card_num="CARD 14/16")

    # Left Graphic Box
    ax.add_patch(patches.FancyBboxPatch((0.8, 1.1), 8.5, 5.3, boxstyle="round,pad=0.1",
                                        facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=1))

    # Flowchart:
    # 1. Primary ResUNet says "EMPTY" (0 voxels)
    ax.add_patch(patches.FancyBboxPatch((1.2, 4.3), 3.4, 1.4, boxstyle="round,pad=0.08",
                                        facecolor="#7F1D1D", edgecolor=ACCENT_RED, linewidth=1.5))
    ax.text(2.9, 5.3, "Primary ResUNet-2D", color=TEXT_WHITE, fontsize=11, fontweight='bold', ha='center')
    ax.text(2.9, 4.7, "Prediction: EMPTY (0 voxels)", color=ACCENT_RED, fontsize=10, ha='center')

    # Arrow down to axial check
    ax.annotate("", xy=(2.9, 3.4), xytext=(2.9, 4.2),
                arrowprops=dict(arrowstyle="->", color=TEXT_WHITE, lw=2))

    # Axial Context Check
    ax.add_patch(patches.FancyBboxPatch((1.2, 2.2), 3.4, 1.1, boxstyle="round,pad=0.08",
                                        facecolor="#1E293B", edgecolor=ACCENT_CYAN, linewidth=1.5))
    ax.text(2.9, 2.9, "Axial Context Check", color=ACCENT_CYAN, fontsize=11, fontweight='bold', ha='center')
    ax.text(2.9, 2.5, "Upper/Lower slices have tumor!", color=TEXT_WHITE, fontsize=9.5, ha='center')

    # Arrow to Secondary
    ax.annotate("", xy=(5.0, 2.75), xytext=(4.7, 2.75),
                arrowprops=dict(arrowstyle="->", color=ACCENT_GREEN, lw=2.5))

    # Secondary UNetResNet34
    ax.add_patch(patches.FancyBboxPatch((5.1, 2.0), 3.8, 2.4, boxstyle="round,pad=0.08",
                                        facecolor="#064E3B", edgecolor=ACCENT_GREEN, linewidth=2))
    ax.text(7.0, 4.0, "Secondary Specialist\n(UNetResNet34)", color=TEXT_WHITE, fontsize=12, fontweight='bold', ha='center')
    ax.text(7.0, 3.3, "Finds Infiltrating Tumor\nRescues 54 True Tumors!", color=ACCENT_GREEN, fontsize=11, ha='center')
    ax.text(7.0, 2.4, "Action: FALLBACK_SWITCH", color=ACCENT_AMBER, fontsize=10, fontweight='bold', ha='center')

    # Right 3 Callout Cards
    add_takeaway_box(ax, 9.6, 4.7, 5.6, 1.7, "1", "The Blind Spot of 2D Slices",
                     "In 2D segmentation, faint tumor fringes at the top or bottom\nof a volume look ambiguous. Primary ResUNet conservatively\noutputs empty background.", ACCENT_RED)

    add_takeaway_box(ax, 9.6, 2.9, 5.6, 1.7, "2", "Axial Validation Prevents False Alarms",
                     "Secondary fallback is NOT called randomly. It requires axial\nsanction from neighboring slices, preventing unnecessary model\nswitching on healthy brain tissue.", ACCENT_CYAN)

    add_takeaway_box(ax, 9.6, 1.1, 5.6, 1.7, "3", "True Tumors Rescued",
                     "Across 105 held-out patients, this exact mechanism rescues 54\ngenuine tumor slices that would have otherwise been missed\nor discarded as background.", ACCENT_GREEN)

    plt.savefig(os.path.join(OUT_DIR, "simple_14_why_fallback_saves_empty_predictions.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 15: Clinical Safety Story
# -------------------------------------------------------------------------------------------------
def generate_card_15():
    fig, ax = create_base_card("The Clinical Safety Story: From Raw AI to Safe Practice",
                               "Transforming a Raw Model into a Fail-Safe Medical Diagnostic Assistant",
                               card_num="CARD 15/16")

    # 4 Chronological Pillars
    pillars = [
        ("Phase 1: Raw AI", "#1E293B", ACCENT_RED,
         "• High raw accuracy (0.8670 Dice)\n• Dangerously overconfident\n• Zero uncertainty feedback\n• Unsafe for standalone use"),
        ("Phase 2: Calibrated", "#1E293B", ACCENT_AMBER,
         "• Temp scaled (T* = 2.8805)\n• ECE drops to 3.41%\n• Probabilities match reality\n• Honest confidence values"),
        ("Phase 3: Agent Triage", "#1E293B", ACCENT_BLUE,
         "• Entropy threshold τ = 0.35\n• Axial context sanity check\n• Secondary model fallback\n• Automated triaging"),
        ("Phase 4: Clinical Safety", "#064E3B", ACCENT_GREEN,
         "• 83.8% tumor slices autonomous\n• 16.2% referred to specialists\n• Severe failures drop 34%\n• Clinician trust established")
    ]

    for i, (title, bg, col, desc) in enumerate(pillars):
        x = 0.8 + i * 3.65
        ax.add_patch(patches.FancyBboxPatch((x, 1.3), 3.45, 5.1, boxstyle="round,pad=0.1",
                                            facecolor=bg, edgecolor=col, linewidth=1.8))
        # Step number circle
        ax.add_patch(patches.Circle((x + 0.6, 5.8), 0.35, facecolor=col, edgecolor='none'))
        ax.text(x + 0.6, 5.8, str(i+1), color=BG_DARK, fontsize=12, fontweight='bold', ha='center', va='center')

        # Title
        ax.text(x + 1.2, 5.8, title, color=TEXT_WHITE, fontsize=12, fontweight='bold', va='center')

        # Flow arrow between cards
        if i < 3:
            ax.annotate("", xy=(x + 3.75, 3.8), xytext=(x + 3.45, 3.8),
                        arrowprops=dict(arrowstyle="->", color=TEXT_MUTED, lw=2.5))

        # Description
        ax.text(x + 0.3, 2.0, desc, color=TEXT_WHITE, fontsize=10.5, va='bottom', linespacing=1.6)

    plt.savefig(os.path.join(OUT_DIR, "simple_15_clinical_safety_story.png"), dpi=150)
    plt.close()

# -------------------------------------------------------------------------------------------------
# Card 16: Performance Snapshot Dashboard
# -------------------------------------------------------------------------------------------------
def generate_card_16():
    fig, ax = create_base_card("ResUNet-2D: Complete Performance Snapshot Dashboard",
                               "Key Performance Indicators, Calibration Metrics, and Cohort Verifications at a Glance",
                               card_num="CARD 16/16")

    # Grid of 6 KPI Cards
    kpis = [
        ("Macro Mean Dice (Val_eval)", "0.8641", "3D Volumetric on 105 held-out patients", ACCENT_GREEN),
        ("Foreground Macro Dice (Best)", "0.8670", "NCR: 0.8210 | ED: 0.8765 | ET: 0.9037", ACCENT_BLUE),
        ("Whole Tumor (WT) Dice", "0.9039", "Outer boundary detection on Val_eval", ACCENT_CYAN),
        ("Optimal Calibration Temp", "2.8805", "ECE drops 8.59% -> 3.41% on Val_eval", ACCENT_AMBER),
        ("Entropy Threshold (τ)", "0.3500", "81.7% non-empty accepted at 0.7803 Dice", ACCENT_PURPLE),
        ("Severe Failure Reduction", "-34.1%", "Slice failures (<0.5 Dice) cut 20.5% -> 13.5%", ACCENT_RED),
    ]

    for i, (title, val, desc, col) in enumerate(kpis):
        col_idx = i % 3
        row_idx = i // 3
        x = 0.8 + col_idx * 4.9
        y = 3.9 - row_idx * 2.6

        ax.add_patch(patches.FancyBboxPatch((x, y), 4.6, 2.3, boxstyle="round,pad=0.1",
                                            facecolor=CARD_BG, edgecolor=col, linewidth=1.8))
        ax.text(x + 0.4, y + 1.9, title, color=TEXT_MUTED, fontsize=11, fontweight='bold')
        ax.text(x + 0.4, y + 1.15, val, color=col, fontsize=28, fontweight='bold')
        ax.text(x + 0.4, y + 0.4, desc, color=TEXT_WHITE, fontsize=10)

    plt.savefig(os.path.join(OUT_DIR, "simple_16_performance_snapshot_dashboard.png"), dpi=150)
    plt.close()

def main():
    print("Generating 16 Non-Technical Explainer Cards...")
    generate_card_01()
    print("[✓] simple_01_what_is_resunet.png")
    generate_card_02()
    print("[✓] simple_02_why_brain_tumors_are_hard.png")
    generate_card_03()
    print("[✓] simple_03_four_mri_modalities_explained.png")
    generate_card_04()
    print("[✓] simple_04_three_tumor_regions_explained.png")
    generate_card_05()
    print("[✓] simple_05_encoder_decoder_concept.png")
    generate_card_06()
    print("[✓] simple_06_residual_connections_explained.png")
    generate_card_07()
    print("[✓] simple_07_what_dice_score_means.png")
    generate_card_08()
    print("[✓] simple_08_what_hd95_means.png")
    generate_card_09()
    print("[✓] simple_09_why_ai_gets_overconfident.png")
    generate_card_10()
    print("[✓] simple_10_temperature_scaling_visualized.png")
    generate_card_11()
    print("[✓] simple_11_entropy_as_traffic_light.png")
    generate_card_12()
    print("[✓] simple_12_axial_context_sandwich.png")
    generate_card_13()
    print("[✓] simple_13_three_action_triage_decision.png")
    generate_card_14()
    print("[✓] simple_14_why_fallback_saves_empty_predictions.png")
    generate_card_15()
    print("[✓] simple_15_clinical_safety_story.png")
    generate_card_16()
    print("[✓] simple_16_performance_snapshot_dashboard.png")
    print(f"All 16 Non-Technical Assets successfully generated in: {OUT_DIR}")

if __name__ == "__main__":
    main()
