"""Gate 6A: Generate publication-quality figures for uncertainty calibration,
reliability diagrams, risk-coverage curves, and patient-level segmentation metrics.

Workspace: D:\\MajorProject
Output: results/gate6a_reliability_calibration/figures/
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "gate6a_reliability_calibration"
FIGURES_DIR = RESULTS_DIR / "figures"


def setup_style() -> None:
    """Set clean, publication-ready matplotlib styling."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_reliability_diagrams(cal_params: dict) -> Path:
    """Plot reliability diagrams before and after temperature scaling for ResUNet-2D and UNetResNet34."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "reliability_diagrams_calibration.png"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)

    models_info = [
        ("resunet_2d", "ResUNet-2D (Primary Model)", axes[0]),
        ("unet_resnet34", "UNetResNet34 (Fallback Candidate)", axes[1]),
    ]

    for model_key, title, ax in models_info:
        data = cal_params["calibration_metrics_val_cal"][model_key]
        uncal = data["uncalibrated"]
        cal = data["calibrated"]
        t_opt = cal_params["fitted_temperatures"][model_key]["optimal_temperature_T"]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=1.5, label="Perfect Calibration")

        # Bin midpoints
        bin_edges = np.array(uncal["bin_edges"])
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        acc_uncal = np.array(uncal["bin_accuracies"])
        conf_uncal = np.array(uncal["bin_confidences"])
        counts_uncal = np.array(uncal["bin_counts"])
        valid_uncal = counts_uncal > 0

        acc_cal = np.array(cal["bin_accuracies"])
        conf_cal = np.array(cal["bin_confidences"])
        counts_cal = np.array(cal["bin_counts"])
        valid_cal = counts_cal > 0

        # Plot curves
        ax.plot(
            conf_uncal[valid_uncal],
            acc_uncal[valid_uncal],
            "o-",
            color="#D9534F",
            linewidth=2,
            markersize=6,
            label=f"Uncalibrated (T=1.00, ECE={uncal['ece']*100:.2f}%)",
        )
        ax.plot(
            conf_cal[valid_cal],
            acc_cal[valid_cal],
            "s-",
            color="#2E7D32",
            linewidth=2,
            markersize=6,
            label=f"Calibrated (T*={t_opt:.2f}, ECE={cal['ece']*100:.2f}%)",
        )

        ax.set_title(title, pad=10, fontweight="bold")
        ax.set_xlabel("Confidence (Top-1 Probability)")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))

        # Annotations text box
        ece_drop = (1.0 - (cal["ece"] / uncal["ece"])) * 100.0
        info_text = (
            f"ECE: {uncal['ece']*100:.2f}% -> {cal['ece']*100:.2f}% (-{ece_drop:.1f}%)\n"
            f"NLL: {uncal['nll']:.4f} -> {cal['nll']:.4f}\n"
            f"Brier: {uncal['brier_score']:.4f} -> {cal['brier_score']:.4f}"
        )
        ax.text(
            0.05,
            0.68,
            info_text,
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9),
            fontsize=9.5,
        )
        ax.legend(loc="lower right")

    axes[0].set_ylabel("Empirical Accuracy")
    plt.suptitle("Multiclass Reliability Diagrams Before and After Temperature Scaling ($Val_{cal}$)", y=0.98, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_risk_coverage(cal_params: dict) -> Path:
    """Plot risk-coverage curves illustrating selective classification trade-offs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "risk_coverage_selective_prediction.png"

    rc_data = cal_params["threshold_exploration"]["risk_coverage_curve"]
    coverage = np.array(rc_data["coverage"]) * 100.0  # Percentage
    mean_dice = np.array(rc_data["mean_dice"])
    risk = np.array(rc_data["risk"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: Coverage vs Retained Mean Dice
    ax1.plot(coverage, mean_dice, color="#1565C0", linewidth=2.5, label="Retained Mean Dice")
    ax1.axvline(100.0, color="gray", linestyle=":", alpha=0.7)
    
    # Highlight operating point (approx tau=0.35 threshold coverage)
    # At high coverage (e.g. ~95%), retained dice increases
    ax1.set_title("Selective Segmentation Fidelity", pad=10, fontweight="bold")
    ax1.set_xlabel("Coverage (% Slices Retained)")
    ax1.set_ylabel("Mean Dice on Retained Slices")
    ax1.set_xlim(0, 102)
    ax1.set_ylim(0.78, 1.01)
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax1.legend(loc="lower left")

    # Panel 2: Coverage vs Selective Risk
    ax2.plot(coverage, risk, color="#C62828", linewidth=2.5, label="Empirical Risk (1 - Accuracy)")
    ax2.set_title("Selective Risk vs. Coverage", pad=10, fontweight="bold")
    ax2.set_xlabel("Coverage (% Slices Retained)")
    ax2.set_ylabel("Voxel Misclassification Risk")
    ax2.set_xlim(0, 102)
    ax2.set_ylim(0.0, max(risk) * 1.15)
    ax2.legend(loc="upper left")

    plt.suptitle("Risk-Coverage Trade-off for Selective Triage via Calibrated Uncertainty", y=0.98, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_evaluation_summary(eval_summary: dict) -> Path:
    """Plot patient-level Mean Dice comparisons across evaluated configurations on Val_eval."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / "val_eval_configuration_comparison.png"

    stats = eval_summary["statistical_summary"]
    configs = [
        ("resunet_uncal", "ResUNet-2D\n(Uncalibrated)"),
        ("resunet_cal", "ResUNet-2D\n(Calibrated)"),
        ("resnet34_uncal", "UNetResNet34\n(Uncalibrated)"),
        ("resnet34_cal", "UNetResNet34\n(Calibrated)"),
        ("agent_baseline", "Agent Baseline\n(Empty+Axial)"),
        ("agent_calibrated", "Calibrated Agent\n(Gate 6A Triage)"),
        ("agent_calibrated_fb", "Calibrated Agent\n(Uncertainty FB)"),
    ]

    labels = [c[1] for c in configs]
    mean_dices = [stats[c[0]]["mean_dice"]["mean"] for c in configs]
    ci_lowers = [stats[c[0]]["mean_dice"]["ci_95_lower"] for c in configs]
    ci_uppers = [stats[c[0]]["mean_dice"]["ci_95_upper"] for c in configs]

    # Error bars (asymmetric)
    yerr_lower = [m - l for m, l in zip(mean_dices, ci_lowers)]
    yerr_upper = [u - m for m, u in zip(mean_dices, ci_uppers)]
    yerr = [yerr_lower, yerr_upper]

    colors = ["#90CAF9", "#1976D2", "#FFE082", "#FFA000", "#A5D6A7", "#2E7D32", "#388E3C"]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(labels, mean_dices, yerr=yerr, capsize=5, color=colors, edgecolor="#333333", linewidth=1.2, width=0.6)

    ax.set_ylabel("Patient-Level Mean Dice (Volumetric 3D)")
    ax.set_ylim(0.80, 0.92)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.02))
    ax.set_title("Volumetric Segmentation Performance on $Val_{eval}$ Cohort (105 Cases, 95% Bootstrap CIs)", pad=12, fontweight="bold")

    # Add numeric labels on bars
    for bar, m, l, u in zip(bars, mean_dices, ci_lowers, ci_uppers):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.006,
            f"{m:.4f}\n[{l:.4f}, {u:.4f}]",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path


def main() -> None:
    setup_style()
    cal_file = RESULTS_DIR / "calibration_parameters.json"
    if not cal_file.exists():
        print(f"Error: {cal_file} not found.")
        sys.exit(1)

    with open(cal_file, "r", encoding="utf-8") as f:
        cal_params = json.load(f)

    p1 = plot_reliability_diagrams(cal_params)
    print(f"Saved: {p1}")

    p2 = plot_risk_coverage(cal_params)
    print(f"Saved: {p2}")

    eval_file = RESULTS_DIR / "evaluation_summary.json"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_summary = json.load(f)
        p3 = plot_evaluation_summary(eval_summary)
        print(f"Saved: {p3}")
    else:
        print("Note: evaluation_summary.json not yet available. Skipping comparison bar plot until evaluation completes.")


if __name__ == "__main__":
    main()
