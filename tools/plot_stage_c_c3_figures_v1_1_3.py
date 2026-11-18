"""
Plotting script for Stage C Phase 3: Work Package C3 v1.1.3.
Generates publication-ready figures with non-truncated axes, outside legends,
clean footnotes, and explicit denominator labels.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["resunet_2d", "unet_resnet34", "unet", "unetpp", "attention_unet"]
MODEL_DISPLAY_NAMES = {
    "resunet_2d": "ResUNet-2D (Primary)",
    "unet_resnet34": "UNetResNet34 (Secondary)",
    "unet": "U-Net",
    "unetpp": "U-Net++",
    "attention_unet": "Attention U-Net",
}
COLORS = {
    "resunet_2d": "#1f77b4",
    "unet_resnet34": "#ff7f0e",
    "unet": "#2ca02c",
    "unetpp": "#d62728",
    "attention_unet": "#9467bd",
}


def plot_figures(data_dir: Path, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_sizes = {}

    # -------------------------------------------------------------
    # 1. Four-Way Policy Ablation Comparison
    # -------------------------------------------------------------
    pol_df = pd.read_csv(data_dir / "policy_comparison_summary.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    cfgs = pol_df["configuration"].tolist()
    dices = pol_df["retained_mean_dice"].tolist()
    gt_ref_rates = pol_df["gt_referral_pct"].tolist()
    all_ref_rates = pol_df["all_flag_unreliable_pct"].tolist()
    sev_rates = pol_df["retained_severe_failure_rate_pct"].tolist()

    x = np.arange(len(cfgs))
    width = 0.38

    # Left Panel: Zero-based Y-axis to avoid misleading baseline truncation
    ax1.bar(x, dices, width, color=["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"], edgecolor="black")
    ax1.axhline(0.7150, color="#444444", linestyle="--", linewidth=1.3, label="Unfiltered Primary Baseline (0.7150)")
    ax1.set_ylabel("Mean Foreground Dice [Scale: 0.0 - 1.0]", fontsize=10)
    ax1.set_title("Retained-Subset Mean Foreground Dice (GT+ Slices)", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, fontsize=9, rotation=10)
    ax1.set_ylim(0.0, 1.0)
    for i, v in enumerate(dices):
        ax1.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    # Legend placed cleanly OUTSIDE plotting area below the subplot
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=True, fontsize=9)

    # Right Panel: Separate GT+ and All-slice referral denominators and failure rates
    w3 = 0.25
    ax2.bar(x - w3, gt_ref_rates, w3, label="GT+ Referral Rate (%) [Denom: 6,549]", color="#1f77b4", edgecolor="black")
    ax2.bar(x, all_ref_rates, w3, label="All-Slice Referral Rate (%) [Denom: 16,275]", color="#ff7f0e", edgecolor="black")
    ax2.bar(x + w3, sev_rates, w3, label="Retained Severe Failure Rate (%)", color="#d62728", edgecolor="black")
    ax2.set_ylabel("Percentage (%)", fontsize=10)
    ax2.set_title("Referral Burdens (GT+ vs. All-Slice) & Retained Failure Rates", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cfgs, fontsize=9, rotation=10)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")
    # Legend placed cleanly OUTSIDE plotting area below the subplot
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True, fontsize=8.5)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    p1 = output_dir / "policy_ablation_comparison.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    file_sizes["policy_ablation_comparison.png"] = p1.stat().st_size

    # -------------------------------------------------------------
    # 2. Five-Model Risk-Coverage Curves
    # -------------------------------------------------------------
    rc_df = pd.read_csv(data_dir / "risk_coverage_by_model.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)

    for m in MODELS:
        m_df = rc_df[(rc_df["model_name"] == m) & (rc_df["gt_pos_retained_count"] > 0) & (rc_df["retained_mean_dice"].notna())]
        covs = m_df["gt_pos_coverage"].values
        dices = m_df["retained_mean_dice"].values
        refs = m_df["gt_pos_referral_rate"].values
        sev_rates = m_df["retained_severe_failure_rate"].values * 100

        ax1.plot(covs, dices, marker="o", markersize=4, label=MODEL_DISPLAY_NAMES[m], color=COLORS[m], linewidth=1.8)
        ax2.plot(refs, sev_rates, marker="s", markersize=4, label=MODEL_DISPLAY_NAMES[m], color=COLORS[m], linewidth=1.8)

        base_row = m_df[m_df["is_baseline_reference"] == True]
        if len(base_row) > 0:
            b = base_row.iloc[0]
            ax1.scatter([b["gt_pos_coverage"]], [b["retained_mean_dice"]], color=COLORS[m], s=90, edgecolors="black", zorder=5)
            ax2.scatter([b["gt_pos_referral_rate"]], [b["retained_severe_failure_rate"] * 100], color=COLORS[m], s=90, edgecolors="black", zorder=5)

    ax1.set_title("Coverage vs. Retained Foreground Dice", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Foreground Coverage (Retained / 6,549 Slices)", fontsize=10)
    ax1.set_ylabel("Retained-Subset Mean Foreground Dice", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower left", fontsize=9)
    ax1.set_ylim(0.70, 0.95)

    ax2.set_title("Referral Rate vs. Retained Severe Failure Rate", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Foreground Referral Rate (Referred / 6,549 Slices)", fontsize=10)
    ax2.set_ylabel("Retained Severe Failure Rate (%) [Dice < 0.50]", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5)
    # Legend is fully visible without overlapping annotations
    ax2.legend(loc="upper right", fontsize=9)
    ax2.set_ylim(0, 30)

    # Suptitle explicitly identifying selection-exposed internal validation
    fig.suptitle(
        "Descriptive Five-Model Risk-Coverage Curves (Val_eval Selection-Exposed Internal Validation Analysis)",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    # Footnote cleanly positioned below the subplots
    footnote_text = (
        "Note on empirical non-monotonic rebounds: UNetResNet34 severe failure rate rebounds from 7.10% to 8.75% "
        "between tau=0.25 and 0.30; U-Net++ rebounds from 5.44% to 8.90% between tau=0.25 and 0.30."
    )
    fig.text(
        0.5,
        0.015,
        footnote_text,
        ha="center",
        fontsize=8.5,
        style="italic",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.95),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    p2 = output_dir / "risk_coverage_curves.png"
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    file_sizes["risk_coverage_curves.png"] = p2.stat().st_size

    # -------------------------------------------------------------
    # 3. Biological Subject-Level Distribution
    # -------------------------------------------------------------
    sub_df = pd.read_csv(data_dir / "subject_level_policy_metrics.csv")
    fig, ax = plt.subplots(figsize=(7.5, 5.2), dpi=300)
    sub_covs = sub_df["coverage"].values * 100
    sub_dices = sub_df["retained_mean_dice"].values

    ax.scatter(sub_covs, sub_dices, color="#1f77b4", alpha=0.75, edgecolors="black", s=55)
    ax.axhline(0.704260, color="#7f7f7f", linestyle="--", linewidth=1.2, label="Mean Subject Baseline Dice (0.7043)")
    ax.axhline(0.752105, color="#d62728", linestyle=":", linewidth=1.4, label="Mean Subject Retained Dice (0.7521)")
    ax.set_title("Descriptive Biological Subject Distribution (N=96 Biological Subjects)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Subject Foreground Coverage (%) [Denom: Subject GT+ Slices]", fontsize=10)
    ax.set_ylabel("Subject Retained-Subset Mean Dice", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", fontsize=8.5)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(50, 102)

    plt.tight_layout()
    p3 = output_dir / "subject_level_distributions.png"
    fig.savefig(p3, bbox_inches="tight")
    plt.close(fig)
    file_sizes["subject_level_distributions.png"] = p3.stat().st_size

    return file_sizes


if __name__ == "__main__":
    d_dir = Path("results/stage_c/phase3/c3_selective_prediction_v1_1_3")
    sizes = plot_figures(d_dir, d_dir)
    print("Regenerated figures in", d_dir)
    for k, v in sizes.items():
        print(f"  {k}: {v} bytes")
