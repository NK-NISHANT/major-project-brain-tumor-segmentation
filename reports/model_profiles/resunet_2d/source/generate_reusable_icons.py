"""
Script: generate_reusable_icons.py
Description: Generates clean, high-resolution badge and icon graphics for ResUNet-2D.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT_DIR = r"D:\MajorProject\reports\model_profiles\resunet_2d\assets\reusable_icons"
os.makedirs(OUT_DIR, exist_ok=True)

BG_DARK = "#0F172A"
CARD_BG = "#1E293B"

def save_icon(name, draw_fn, size=(4, 4), dpi=200):
    fig = plt.figure(figsize=size, dpi=dpi, facecolor='none')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    draw_fn(ax)
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=dpi, transparent=True)
    plt.close()
    print(f"[✓] {name}")

def draw_accept(ax):
    # Circle
    ax.add_patch(patches.Circle((5, 5), 4.5, facecolor="#064E3B", edgecolor="#34D399", linewidth=4))
    # Checkmark
    ax.plot([2.8, 4.4, 7.4], [5.0, 3.4, 6.8], color="#34D399", linewidth=8, solid_capstyle='round')

def draw_fallback(ax):
    # Circle
    ax.add_patch(patches.Circle((5, 5), 4.5, facecolor="#78350F", edgecolor="#FBBF24", linewidth=4))
    # Switch arrows
    ax.annotate("", xy=(7.2, 5.8), xytext=(2.8, 5.8),
                arrowprops=dict(arrowstyle="->", color="#FBBF24", lw=6))
    ax.annotate("", xy=(2.8, 4.2), xytext=(7.2, 4.2),
                arrowprops=dict(arrowstyle="->", color="#FBBF24", lw=6))

def draw_flag(ax):
    # Triangle
    triangle = patches.Polygon([[5, 8.8], [1.2, 2.0], [8.8, 2.0]],
                               closed=True, facecolor="#7F1D1D", edgecolor="#F87171", linewidth=4)
    ax.add_patch(triangle)
    # Exclamation mark
    ax.plot([5, 5], [6.8, 4.2], color="#FFFFFF", linewidth=6, solid_capstyle='round')
    ax.plot([5, 5], [2.8, 2.8], color="#FFFFFF", marker='o', markersize=9)

def draw_calibration(ax):
    # Thermometer
    ax.add_patch(patches.Circle((5, 5), 4.5, facecolor="#1E293B", edgecolor="#38BDF8", linewidth=4))
    ax.add_patch(patches.Rectangle((4.3, 3.2), 1.4, 4.2, facecolor="#0B132B", edgecolor="#38BDF8", linewidth=2))
    ax.add_patch(patches.Circle((5, 3.0), 1.3, facecolor="#F87171", edgecolor="#38BDF8", linewidth=2))
    ax.add_patch(patches.Rectangle((4.6, 3.0), 0.8, 3.0, facecolor="#F87171", edgecolor='none'))

def draw_brain_mri(ax):
    # Brain outline circle badge
    ax.add_patch(patches.Circle((5, 5), 4.5, facecolor="#1E293B", edgecolor="#C084FC", linewidth=4))
    # Two hemispheres
    ax.add_patch(patches.Ellipse((3.7, 5.2), 2.2, 3.4, facecolor="#3B1E48", edgecolor="#C084FC", linewidth=2))
    ax.add_patch(patches.Ellipse((6.3, 5.2), 2.2, 3.4, facecolor="#3B1E48", edgecolor="#C084FC", linewidth=2))
    # Small highlighted lesion
    ax.add_patch(patches.Circle((4.2, 5.6), 0.6, facecolor="#EF4444", edgecolor="#FBBF24", linewidth=1.5))

def draw_resunet_badge(ax):
    # Hexagon badge
    ax.add_patch(patches.RegularPolygon((5, 5), numVertices=6, radius=4.6,
                                       facecolor="#0B132B", edgecolor="#38BDF8", linewidth=4))
    ax.text(5, 5.6, "ResUNet", color="#F8FAFC", fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(5, 3.8, "2D", color="#38BDF8", fontsize=20, fontweight='bold', ha='center', va='center')

def main():
    print("Generating Reusable Icons & Badges...")
    save_icon("icon_accept.png", draw_accept)
    save_icon("icon_fallback.png", draw_fallback)
    save_icon("icon_flag.png", draw_flag)
    save_icon("icon_calibration.png", draw_calibration)
    save_icon("icon_brain_mri.png", draw_brain_mri)
    save_icon("icon_resunet_badge.png", draw_resunet_badge)
    print("All icons successfully generated.")

if __name__ == "__main__":
    main()
