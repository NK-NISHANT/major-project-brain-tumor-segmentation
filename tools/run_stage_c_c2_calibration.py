"""Work Package C2: Five-Model Probability Calibration & Uncertainty Analysis.

Workspace: D:\\MajorProject
Governing Blueprint: MASTER_PROJECT_BLUEPRINT_v2.0.md
Governing Protocol: STAGE_C_EXPERIMENT_PROTOCOL_DRAFT.md (Section 5.2)

Architectures Evaluated:
1. ResUNet-2D (frozen best checkpoint, epoch 13)
2. UNetResNet34 (frozen best checkpoint, epoch 15)
3. U-Net (frozen best checkpoint, epoch 15)
4. U-Net++ (frozen best checkpoint, epoch 15)
5. Attention U-Net (frozen best checkpoint, epoch 15)

Cohort Definitions:
- Val_cal: 50 cases (47 unique subjects) - Selection-exposed calibration data
- Val_eval: 105 cases (96 unique subjects) - Selection-exposed internal evaluation data
Total: 155 cases (143 unique subjects, 0 subject overlap)
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    CaseFiles,
    MODALITIES,
    SEGMENTATION_SUFFIX,
    _load_and_normalize_case,
)
from src.models.attention_unet import AttentionUNet2D
from src.models.resunet import ResUNet2D
from src.models.unet import UNet2D
from src.models.unet_resnet34 import UNetResNet34
from src.models.unetpp import UNetPlusPlus2D
from src.uncertainty.calibration import (
    TemperatureScaler,
    compute_multiclass_brier_score,
    compute_multiclass_ece,
    compute_nll,
)
from src.uncertainty.entropy import softmax_entropy


MODELS_CONFIG: list[dict[str, Any]] = [
    {
        "name": "resunet_2d",
        "display_name": "ResUNet-2D",
        "checkpoint_path": PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "best_model.pt",
        "config_path": PROJECT_ROOT / "results" / "full_training_15ep" / "resunet_2d" / "config.json",
        "factory": ResUNet2D,
        "kwargs": {"in_channels": 4, "num_classes": 4, "base_channels": 32},
        "expected_best_epoch": 13,
        "expected_sha256": "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d",
        "expected_params": 8114604,
    },
    {
        "name": "unet_resnet34",
        "display_name": "UNetResNet34",
        "checkpoint_path": PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "best_model.pt",
        "config_path": PROJECT_ROOT / "results" / "full_training_15ep" / "unet_resnet34" / "config.json",
        "factory": UNetResNet34,
        "kwargs": {"in_channels": 4, "num_classes": 4, "base_channels": 32},
        "expected_best_epoch": 15,
        "expected_sha256": "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d",
        "expected_params": 24327524,
    },
    {
        "name": "unet",
        "display_name": "U-Net",
        "checkpoint_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unet_721_seed42" / "best_model.pt",
        "config_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unet_721_seed42" / "config.json",
        "factory": UNet2D,
        "kwargs": {"in_channels": 4, "num_classes": 4, "base_channels": 32},
        "expected_best_epoch": 15,
        "expected_sha256": "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef",
        "expected_params": 7763428,
    },
    {
        "name": "unetpp",
        "display_name": "U-Net++",
        "checkpoint_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "best_model.pt",
        "config_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "unetpp_721_seed42" / "config.json",
        "factory": UNetPlusPlus2D,
        "kwargs": {"in_channels": 4, "num_classes": 4, "base_channels": 32},
        "expected_best_epoch": 15,
        "expected_sha256": "fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6",
        "expected_params": 9160068,
    },
    {
        "name": "attention_unet",
        "display_name": "Attention U-Net",
        "checkpoint_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "attention_unet_721_seed42" / "best_model.pt",
        "config_path": PROJECT_ROOT / "results" / "stage_c" / "phase1" / "attention_unet_721_seed42" / "config.json",
        "factory": AttentionUNet2D,
        "kwargs": {"in_channels": 4, "num_classes": 4, "base_channels": 32},
        "expected_best_epoch": 15,
        "expected_sha256": "5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e",
        "expected_params": 7852160,
    },
]


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def resolve_single_case(case_id: str, raw_dir: Path, archive_dir: Path) -> CaseFiles:
    cdir = (raw_dir / case_id) if (raw_dir / case_id).exists() else (archive_dir / case_id)
    if not cdir.exists():
        raise FileNotFoundError(f"Case directory {case_id} not found in raw or extracted_archive")

    mod_paths = {m: cdir / f"{case_id}-{m}.nii.gz" for m in MODALITIES}
    seg_path = cdir / f"{case_id}-{SEGMENTATION_SUFFIX}.nii.gz"
    shape = tuple(int(dim) for dim in nib.load(str(seg_path)).shape)

    return CaseFiles(
        case_id=case_id,
        case_dir=cdir,
        modality_paths=mod_paths,
        segmentation_path=seg_path,
        shape=shape,
    )


class VolumetricDiceAccumulator:
    """Accumulates 3D voxel confusion counts across all 155 slices of a single case."""

    def __init__(self) -> None:
        self.wt_tp = 0
        self.wt_fp = 0
        self.wt_fn = 0

        self.tc_tp = 0
        self.tc_fp = 0
        self.tc_fn = 0

        self.et_tp = 0
        self.et_fp = 0
        self.et_fn = 0

    def update_slice(self, pred_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
        p = pred_mask.detach().cpu().long()
        t = target_mask.detach().cpu().long()

        p_wt = p > 0
        t_wt = t > 0
        self.wt_tp += int((p_wt & t_wt).sum().item())
        self.wt_fp += int((p_wt & ~t_wt).sum().item())
        self.wt_fn += int((~p_wt & t_wt).sum().item())

        p_tc = (p == 1) | (p == 3)
        t_tc = (t == 1) | (t == 3)
        self.tc_tp += int((p_tc & t_tc).sum().item())
        self.tc_fp += int((p_tc & ~t_tc).sum().item())
        self.tc_fn += int((~p_tc & t_tc).sum().item())

        p_et = p == 3
        t_et = t == 3
        self.et_tp += int((p_et & t_et).sum().item())
        self.et_fp += int((p_et & ~t_et).sum().item())
        self.et_fn += int((~p_et & t_et).sum().item())

    def compute_dice(self) -> dict[str, float]:
        def _dice(tp: int, fp: int, fn: int) -> float:
            denom = 2 * tp + fp + fn
            if denom > 0:
                return (2.0 * tp) / denom
            return 1.0 if (tp + fp + fn) == 0 else 0.0

        wt_d = _dice(self.wt_tp, self.wt_fp, self.wt_fn)
        tc_d = _dice(self.tc_tp, self.tc_fp, self.tc_fn)
        et_d = _dice(self.et_tp, self.et_fp, self.et_fn)

        valid_dices = [d for d in (wt_d, tc_d, et_d) if not np.isnan(d)]
        mean_d = float(np.mean(valid_dices)) if valid_dices else float("nan")

        return {
            "wt_dice": wt_d,
            "tc_dice": tc_d,
            "et_dice": et_d,
            "mean_dice": mean_d,
        }


def compute_slice_dice(pred_mask: torch.Tensor, target_mask: torch.Tensor) -> tuple[float, float, float, float]:
    p = pred_mask.detach().cpu().long()
    t = target_mask.detach().cpu().long()

    # Per-class Dice (1=NCR, 2=ED, 3=ET)
    class_dices: list[float] = []
    for c in (1, 2, 3):
        tp = float(((p == c) & (t == c)).sum().item())
        fp = float(((p == c) & (t != c)).sum().item())
        fn = float(((p != c) & (t == c)).sum().item())
        denom = 2.0 * tp + fp + fn
        class_dices.append((2.0 * tp / denom) if denom > 0 else float("nan"))

    # WT: {1, 2, 3}
    p_wt = p > 0
    t_wt = t > 0
    wt_tp = float((p_wt & t_wt).sum().item())
    wt_fp = float((p_wt & ~t_wt).sum().item())
    wt_fn = float((~p_wt & t_wt).sum().item())
    wt_denom = 2.0 * wt_tp + wt_fp + wt_fn
    wt_dice = (2.0 * wt_tp / wt_denom) if wt_denom > 0 else float("nan")

    # TC: {1, 3}
    p_tc = (p == 1) | (p == 3)
    t_tc = (t == 1) | (t == 3)
    tc_tp = float((p_tc & t_tc).sum().item())
    tc_fp = float((p_tc & ~t_tc).sum().item())
    tc_fn = float((~p_tc & t_tc).sum().item())
    tc_denom = 2.0 * tc_tp + tc_fp + tc_fn
    tc_dice = (2.0 * tc_tp / tc_denom) if tc_denom > 0 else float("nan")

    # ET: {3}
    p_et = p == 3
    t_et = t == 3
    et_tp = float((p_et & t_et).sum().item())
    et_fp = float((p_et & ~t_et).sum().item())
    et_fn = float((~p_et & t_et).sum().item())
    et_denom = 2.0 * et_tp + et_fp + et_fn
    et_dice = (2.0 * et_tp / et_denom) if et_denom > 0 else float("nan")

    valid_class_dices = [d for d in class_dices if not math.isnan(d)]
    mean_fg_dice = float(np.mean(valid_class_dices)) if valid_class_dices else float("nan")
    return mean_fg_dice, wt_dice, tc_dice, et_dice


def plot_reliability_diagram(
    model_name: str,
    display_name: str,
    uncal_data: dict[str, Any],
    cal_data: dict[str, Any],
    t_opt: float,
    out_path: Path,
) -> None:
    """Plot single model reliability diagram (uncalibrated vs calibrated)."""
    fig, ax = plt.subplots(figsize=(6.5, 6), dpi=150)

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=1.5, label="Perfect Calibration")

    acc_uncal = np.array(uncal_data["bin_accuracies"])
    conf_uncal = np.array(uncal_data["bin_confidences"])
    counts_uncal = np.array(uncal_data["bin_counts"])
    valid_uncal = counts_uncal > 0

    acc_cal = np.array(cal_data["bin_accuracies"])
    conf_cal = np.array(cal_data["bin_confidences"])
    counts_cal = np.array(cal_data["bin_counts"])
    valid_cal = counts_cal > 0

    ax.plot(
        conf_uncal[valid_uncal],
        acc_uncal[valid_uncal],
        "o-",
        color="#D9534F",
        linewidth=2,
        markersize=6,
        label=f"Uncalibrated (T=1.00, ECE={uncal_data['ece']*100:.2f}%)",
    )
    ax.plot(
        conf_cal[valid_cal],
        acc_cal[valid_cal],
        "s-",
        color="#2E7D32",
        linewidth=2,
        markersize=6,
        label=f"Calibrated (T*={t_opt:.4f}, ECE={cal_data['ece']*100:.2f}%)",
    )

    ax.set_title(f"{display_name} - Reliability Diagram", pad=12, fontweight="bold")
    ax.set_xlabel("Confidence (Top-1 Softmax Probability)", fontsize=11)
    ax.set_ylabel("Empirical Accuracy", fontsize=11)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.grid(True, linestyle=":", alpha=0.5)

    ece_rel_drop = (1.0 - (cal_data["ece"] / uncal_data["ece"])) * 100.0 if uncal_data["ece"] > 0 else 0.0
    nll_rel_drop = (1.0 - (cal_data["nll"] / uncal_data["nll"])) * 100.0 if uncal_data["nll"] > 0 else 0.0
    info_text = (
        f"ECE: {uncal_data['ece']*100:.2f}% -> {cal_data['ece']*100:.2f}% (-{ece_rel_drop:.1f}%)\n"
        f"NLL: {uncal_data['nll']:.4f} -> {cal_data['nll']:.4f} (-{nll_rel_drop:.1f}%)\n"
        f"Brier: {uncal_data['brier_score']:.4f} -> {cal_data['brier_score']:.4f}\n"
        f"MCE: {uncal_data['mce']*100:.2f}% -> {cal_data['mce']*100:.2f}%"
    )
    ax.text(
        0.05,
        0.65,
        info_text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9),
    )

    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_five_model_reliability_diagrams(
    models_results: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Plot publication-grade 5-panel reliability diagram comparison."""
    fig, axes = plt.subplots(1, 5, figsize=(26, 5.2), sharey=True, dpi=200)

    for idx, (m_res, ax) in enumerate(zip(models_results, axes)):
        d_name = m_res["display_name"]
        uncal = m_res["eval_uncal"]
        cal = m_res["eval_cal"]
        t_opt = m_res["optimal_temperature"]

        ax.plot([0, 1], [0, 1], "k--", alpha=0.7, linewidth=1.5, label="Perfect" if idx == 0 else "")

        acc_u = np.array(uncal["bin_accuracies"])
        conf_u = np.array(uncal["bin_confidences"])
        cnt_u = np.array(uncal["bin_counts"])
        v_u = cnt_u > 0

        acc_c = np.array(cal["bin_accuracies"])
        conf_c = np.array(cal["bin_confidences"])
        cnt_c = np.array(cal["bin_counts"])
        v_c = cnt_c > 0

        ax.plot(
            conf_u[v_u],
            acc_u[v_u],
            "o-",
            color="#D9534F",
            linewidth=2,
            markersize=5,
            label=f"Uncal (ECE={uncal['ece']*100:.2f}%)",
        )
        ax.plot(
            conf_c[v_c],
            acc_c[v_c],
            "s-",
            color="#2E7D32",
            linewidth=2,
            markersize=5,
            label=f"Cal (T*={t_opt:.2f}, ECE={cal['ece']*100:.2f}%)",
        )

        ax.set_title(d_name, pad=10, fontweight="bold", fontsize=11)
        ax.set_xlabel("Confidence", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Empirical Accuracy", fontsize=10)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.05)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.grid(True, linestyle=":", alpha=0.5)

        ece_rel_drop = (1.0 - (cal["ece"] / uncal["ece"])) * 100.0 if uncal["ece"] > 0 else 0.0
        info_text = f"ECE: {uncal['ece']*100:.2f}% -> {cal['ece']*100:.2f}%\n(-{ece_rel_drop:.1f}%)\nT* = {t_opt:.4f}"
        ax.text(
            0.05,
            0.72,
            info_text,
            transform=ax.transAxes,
            fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#F8F9FA", edgecolor="#CCCCCC", alpha=0.9),
        )
        ax.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_pipeline() -> None:
    t_start_total = time.perf_counter()
    print("===============================================================================")
    print("STAGE C PHASE 2 (C2): FIVE-MODEL PROBABILITY CALIBRATION & UNCERTAINTY ANALYSIS")
    print("===============================================================================")
    print(f"Start Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {platform.python_version()} | PyTorch: {torch.__version__}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Device: {device} ({gpu_name})")

    out_base = PROJECT_ROOT / "results" / "stage_c" / "phase2" / "c2_calibration_v1"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    arch_dir = PROJECT_ROOT / "data" / "extracted_archive"

    # Subdirectories
    dir_prov = out_base / "provenance_and_splits"
    dir_params = out_base / "calibration_parameters"
    dir_fit = out_base / "calibration_fit"
    dir_preds = out_base / "evaluation_predictions"
    dir_metrics = out_base / "evaluation_metrics"
    dir_rel = out_base / "reliability_diagrams"
    dir_ent = out_base / "predictive_entropy"
    dir_logs = out_base / "execution_logs"
    dir_verif = out_base / "verification_records"

    for d in [dir_prov, dir_params, dir_fit, dir_preds, dir_metrics, dir_rel, dir_ent, dir_logs, dir_verif]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Mandatory Cohort Verification Gate
    print("\n--- 1. Mandatory Cohort Verification Gate ---")
    splits_path = dir_prov / "calibration_splits.json"
    if not splits_path.exists():
        src_splits = PROJECT_ROOT / "results" / "gate6a_remediation_v1" / "calibration_splits.json"
        import shutil
        shutil.copy2(src_splits, splits_path)

    with open(splits_path, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    cal_case_ids: list[str] = splits_data["calibration_set"]["case_ids"]
    eval_case_ids: list[str] = splits_data["evaluation_set"]["case_ids"]
    cal_subjects: list[str] = splits_data["calibration_set"]["subject_ids"]
    eval_subjects: list[str] = splits_data["evaluation_set"]["subject_ids"]

    assert len(cal_case_ids) == 50, f"Expected 50 cal cases, got {len(cal_case_ids)}"
    assert len(eval_case_ids) == 105, f"Expected 105 eval cases, got {len(eval_case_ids)}"
    assert len(cal_subjects) == 47, f"Expected 47 cal subjects, got {len(cal_subjects)}"
    assert len(eval_subjects) == 96, f"Expected 96 eval subjects, got {len(eval_subjects)}"

    cal_set = set(cal_case_ids)
    eval_set = set(eval_case_ids)
    assert len(cal_set.intersection(eval_set)) == 0, "Case overlap detected!"

    cal_subj_set = set(cal_subjects)
    eval_subj_set = set(eval_subjects)
    assert len(cal_subj_set.intersection(eval_subj_set)) == 0, "Subject overlap detected!"

    splits_sha256 = get_file_sha256(splits_path)
    print(f"Val_cal:  {len(cal_case_ids)} cases, {len(cal_subjects)} unique subjects.")
    print(f"Val_eval: {len(eval_case_ids)} cases, {len(eval_subjects)} unique subjects.")
    print(f"Cohort Disjointness Verified: 0 case overlap, 0 subject overlap.")
    print(f"Split Hash: {splits_sha256}")

    # 2. Checkpoint Verification Gate
    print("\n--- 2. Checkpoint Verification Gate ---")
    verified_checkpoints: dict[str, dict[str, Any]] = {}
    for m_cfg in MODELS_CONFIG:
        ckpt_p = m_cfg["checkpoint_path"]
        assert ckpt_p.exists(), f"Checkpoint not found: {ckpt_p}"
        sha = get_file_sha256(ckpt_p)
        assert sha == m_cfg["expected_sha256"], f"Hash mismatch for {m_cfg['name']}: {sha} vs {m_cfg['expected_sha256']}"
        
        # Test model instantiation and parameter count
        model_inst = m_cfg["factory"](**m_cfg["kwargs"])
        total_p = sum(p.numel() for p in model_inst.parameters())
        assert total_p == m_cfg["expected_params"], f"Param mismatch for {m_cfg['name']}: {total_p} vs {m_cfg['expected_params']}"
        
        # Test state dict load
        sd = torch.load(ckpt_p, map_location="cpu")
        weights = sd["model_state_dict"] if "model_state_dict" in sd else sd
        missing, unexpected = model_inst.load_state_dict(weights, strict=True)
        assert len(missing) == 0 and len(unexpected) == 0, f"State dict mismatch for {m_cfg['name']}"
        
        del model_inst
        del sd
        del weights

        verified_checkpoints[m_cfg["name"]] = {
            "checkpoint_path": str(ckpt_p.relative_to(PROJECT_ROOT)),
            "sha256": sha,
            "parameter_count": total_p,
            "expected_best_epoch": m_cfg["expected_best_epoch"],
        }
        print(f"  {m_cfg['display_name']:<16}: Ep {m_cfg['expected_best_epoch']:2d} | Params: {total_p:,} | SHA: {sha[:16]}... (VERIFIED)")

    # Save verification record
    with open(dir_verif / "checkpoint_and_split_verification.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "splits_hash": splits_sha256,
                "cohort_accounting": {
                    "val_cal": {"cases": len(cal_case_ids), "subjects": len(cal_subjects)},
                    "val_eval": {"cases": len(eval_case_ids), "subjects": len(eval_subjects)},
                    "total_validation": {"cases": 155, "subjects": 143},
                    "subject_disjointness_verified": True,
                },
                "models": verified_checkpoints,
            },
            f,
            indent=2,
        )

    # 3. Model Processing Loop
    models_evaluation_summary: list[dict[str, Any]] = []
    models_calibration_params: dict[str, Any] = {}
    models_entropy_summary: list[dict[str, Any]] = []
    all_case_level_records: list[dict[str, Any]] = []

    for m_cfg in MODELS_CONFIG:
        m_name = m_cfg["name"]
        d_name = m_cfg["display_name"]
        ckpt_p = m_cfg["checkpoint_path"]
        print(f"\n===============================================================================")
        print(f"PROCESSING MODEL: {d_name} ({m_name})")
        print(f"===============================================================================")
        t_model_start = time.perf_counter()

        # Load model onto GPU
        model = m_cfg["factory"](**m_cfg["kwargs"]).to(device)
        sd = torch.load(ckpt_p, map_location=device)
        weights = sd["model_state_dict"] if "model_state_dict" in sd else sd
        model.load_state_dict(weights, strict=True)
        model.eval()
        torch.backends.cudnn.benchmark = False

        # -------------------------------------------------------------
        # Phase A: Calibration Fitting on Val_cal (50 cases)
        # -------------------------------------------------------------
        print(f"\nPhase A: Voxel extraction & calibration fitting on Val_cal ({len(cal_case_ids)} cases)...")
        cal_sampled_logits: list[torch.Tensor] = []
        cal_sampled_targets: list[torch.Tensor] = []
        cal_case_voxel_counts: dict[str, int] = {}

        rng_cal = np.random.RandomState(42)
        total_cal_voxels = 0

        with torch.inference_mode():
            for c_idx, cid in enumerate(cal_case_ids, start=1):
                t_c0 = time.perf_counter()
                cfiles = resolve_single_case(cid, raw_dir, arch_dir)
                vol = _load_and_normalize_case(cfiles)
                total_z = vol.image.shape[3]  # 155

                c_fg_voxels = 0
                c_bg_voxels = 0

                # Process slices in batches of 16
                batch_size = 16
                for z_start in range(0, total_z, batch_size):
                    z_end = min(z_start + batch_size, total_z)
                    cur_b = z_end - z_start

                    # Image tensor [B, 4, 240, 240]
                    img_batch = torch.from_numpy(
                        np.ascontiguousarray(np.moveaxis(vol.image[:, :, :, z_start:z_end], 3, 0))
                    ).float().to(device)

                    logits_batch = model(img_batch).cpu()  # [B, 4, 240, 240]

                    for b_i in range(cur_b):
                        z = z_start + b_i
                        if z % 2 == 0:
                            tgt_z = torch.from_numpy(vol.mask[:, :, z].astype(np.int64))
                            tgt_np = tgt_z.numpy()
                            brain_mask = (vol.image[:, :, :, z].sum(axis=0) != 0)
                            fg_idx = np.where(tgt_np > 0)
                            n_fg = len(fg_idx[0])

                            if n_fg > 0:
                                l_z = logits_batch[b_i]  # [4, 240, 240]
                                l_fg = l_z[:, fg_idx[0], fg_idx[1]].permute(1, 0)
                                t_fg = tgt_z[fg_idx[0], fg_idx[1]]

                                bg_idx = np.where((tgt_np == 0) & brain_mask)
                                n_bg = len(bg_idx[0])
                                s_bg = min(n_bg, n_fg)

                                if s_bg > 0:
                                    chosen_bg = rng_cal.choice(n_bg, size=s_bg, replace=False)
                                    bg_r, bg_c = bg_idx[0][chosen_bg], bg_idx[1][chosen_bg]
                                    l_bg = l_z[:, bg_r, bg_c].permute(1, 0)
                                    t_bg = tgt_z[bg_r, bg_c]

                                    cat_l = torch.cat([l_fg, l_bg], dim=0)
                                    cat_t = torch.cat([t_fg, t_bg], dim=0)
                                else:
                                    cat_l = l_fg
                                    cat_t = t_fg

                                cal_sampled_logits.append(cat_l)
                                cal_sampled_targets.append(cat_t)
                                c_fg_voxels += n_fg
                                c_bg_voxels += s_bg
                                total_cal_voxels += cat_l.shape[0]

                cal_case_voxel_counts[cid] = c_fg_voxels + c_bg_voxels
                del vol
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if c_idx % 10 == 0 or c_idx == len(cal_case_ids):
                    dt = time.perf_counter() - t_c0
                    print(f"  [{c_idx:2d}/{len(cal_case_ids)}] {cid} | Total Sampled: {total_cal_voxels:,} voxels")

        assert total_cal_voxels == 5020730, f"Expected 5,020,730 calibration voxels, got {total_cal_voxels}"
        all_cal_l = torch.cat(cal_sampled_logits, dim=0)
        all_cal_t = torch.cat(cal_sampled_targets, dim=0)
        print(f"Aggregated Val_cal logits: {all_cal_l.shape} | Targets: {all_cal_t.shape}")

        # Compute uncalibrated metrics on Val_cal
        p_cal_uncal = torch.softmax(all_cal_l, dim=-1)
        uncal_cal_nll = compute_nll(p_cal_uncal, all_cal_t)
        uncal_cal_ece_dict = compute_multiclass_ece(p_cal_uncal, all_cal_t, n_bins=15)
        uncal_cal_brier = compute_multiclass_brier_score(p_cal_uncal, all_cal_t)

        # Fit Temperature Scaler
        print("Fitting TemperatureScaler using L-BFGS (strong Wolfe line search)...")
        t_fit_start = time.perf_counter()
        scaler = TemperatureScaler(initial_temperature=1.0)
        # Fit on device for acceleration
        fit_diagnostics = scaler.fit(all_cal_l, all_cal_t, lr=0.05, max_iter=100, device=device)
        t_fit_sec = time.perf_counter() - t_fit_start
        t_opt = scaler.temperature

        # Compute calibrated metrics on Val_cal
        p_cal_cal = torch.softmax(all_cal_l / t_opt, dim=-1)
        cal_cal_nll = compute_nll(p_cal_cal, all_cal_t)
        cal_cal_ece_dict = compute_multiclass_ece(p_cal_cal, all_cal_t, n_bins=15)
        cal_cal_brier = compute_multiclass_brier_score(p_cal_cal, all_cal_t)

        print(f"Optimal Temperature T*: {t_opt:.4f} (Fit Time: {t_fit_sec:.2f}s, Iterations: {fit_diagnostics['iterations']}, Converged: {fit_diagnostics['converged']})")
        print(f"  Val_cal NLL:   {uncal_cal_nll:.4f} -> {cal_cal_nll:.4f} (Rel: -{(1.0 - cal_cal_nll/uncal_cal_nll)*100:.1f}%)")
        print(f"  Val_cal ECE:   {uncal_cal_ece_dict['ece']*100:.2f}% -> {cal_cal_ece_dict['ece']*100:.2f}% (Rel: -{(1.0 - cal_cal_ece_dict['ece']/uncal_cal_ece_dict['ece'])*100:.1f}%)")
        print(f"  Val_cal MCE:   {uncal_cal_ece_dict['mce']*100:.2f}% -> {cal_cal_ece_dict['mce']*100:.2f}%")
        print(f"  Val_cal Brier: {uncal_cal_brier:.4f} -> {cal_cal_brier:.4f}")

        # Save model-specific calibration parameter JSON
        cal_param_record = {
            "model_name": m_name,
            "display_name": d_name,
            "checkpoint_sha256": verified_checkpoints[m_name]["sha256"],
            "parameter_count": verified_checkpoints[m_name]["parameter_count"],
            "expected_best_epoch": verified_checkpoints[m_name]["expected_best_epoch"],
            "optimal_temperature_T": t_opt,
            "fit_diagnostics": fit_diagnostics,
            "val_cal_metrics": {
                "uncalibrated": {
                    "nll": uncal_cal_nll,
                    "ece": uncal_cal_ece_dict["ece"],
                    "mce": uncal_cal_ece_dict["mce"],
                    "brier_score": uncal_cal_brier,
                    "bin_accuracies": uncal_cal_ece_dict["bin_accuracies"],
                    "bin_confidences": uncal_cal_ece_dict["bin_confidences"],
                    "bin_counts": uncal_cal_ece_dict["bin_counts"],
                },
                "calibrated": {
                    "nll": cal_cal_nll,
                    "ece": cal_cal_ece_dict["ece"],
                    "mce": cal_cal_ece_dict["mce"],
                    "brier_score": cal_cal_brier,
                    "bin_accuracies": cal_cal_ece_dict["bin_accuracies"],
                    "bin_confidences": cal_cal_ece_dict["bin_confidences"],
                    "bin_counts": cal_cal_ece_dict["bin_counts"],
                },
            },
            "fitting_cohort": {
                "name": "Val_cal",
                "cases_count": len(cal_case_ids),
                "unique_subjects": len(cal_subjects),
                "total_sampled_voxels": total_cal_voxels,
                "case_voxel_counts": cal_case_voxel_counts,
                "splits_hash": splits_sha256,
            },
        }

        with open(dir_params / f"{m_name}_temperature.json", "w", encoding="utf-8") as f:
            json.dump(cal_param_record, f, indent=2)

        models_calibration_params[m_name] = cal_param_record

        # Free Phase A tensors
        del cal_sampled_logits
        del cal_sampled_targets
        del all_cal_l
        del all_cal_t
        del p_cal_uncal
        del p_cal_cal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # -------------------------------------------------------------
        # Phase B: Evaluation & Predictive Uncertainty on Val_eval (105 cases)
        # -------------------------------------------------------------
        print(f"\nPhase B: Evaluation & Predictive Uncertainty on Val_eval ({len(eval_case_ids)} cases)...")
        eval_sampled_logits: list[torch.Tensor] = []
        eval_sampled_targets: list[torch.Tensor] = []
        eval_case_voxel_counts: dict[str, int] = {}

        m_case_results: list[dict[str, Any]] = []
        m_slice_entropies: list[float] = []  # Excludes None
        m_empty_slice_count = 0
        m_non_empty_slice_count = 0
        m_argmax_mismatch_voxels = 0
        total_eval_voxels = 0

        rng_eval = np.random.RandomState(42)
        model_pred_dir = dir_preds / m_name
        model_pred_dir.mkdir(parents=True, exist_ok=True)

        with torch.inference_mode():
            for c_idx, cid in enumerate(eval_case_ids, start=1):
                t_c0 = time.perf_counter()
                cfiles = resolve_single_case(cid, raw_dir, arch_dir)
                vol = _load_and_normalize_case(cfiles)
                total_z = vol.image.shape[3]  # 155

                vol_accum = VolumetricDiceAccumulator()
                case_sampled_count = 0
                case_slice_records: list[dict[str, Any]] = []

                batch_size = 16
                for z_start in range(0, total_z, batch_size):
                    z_end = min(z_start + batch_size, total_z)
                    cur_b = z_end - z_start

                    img_batch = torch.from_numpy(
                        np.ascontiguousarray(np.moveaxis(vol.image[:, :, :, z_start:z_end], 3, 0))
                    ).float().to(device)

                    logits_batch = model(img_batch).cpu()  # [B, 4, 240, 240]

                    for b_i in range(cur_b):
                        z = z_start + b_i
                        tgt_z = torch.from_numpy(vol.mask[:, :, z].astype(np.int64))
                        tgt_np = tgt_z.numpy()
                        l_z = logits_batch[b_i]  # [4, 240, 240]

                        # 1. Argmax Invariance Check
                        pred_uncal = torch.argmax(l_z, dim=0).to(torch.uint8)
                        pred_cal = torch.argmax(l_z / t_opt, dim=0).to(torch.uint8)
                        mismatches = int((pred_uncal != pred_cal).sum().item())
                        m_argmax_mismatch_voxels += mismatches

                        # 2. Volumetric Metric Accumulator
                        vol_accum.update_slice(pred_cal, tgt_z)

                        # 3. Slice Dice Calculation
                        m_dice, wt_d, tc_d, et_d = compute_slice_dice(pred_cal, tgt_z)

                        # 4. Normalized 4-Class Shannon Entropy on Calibrated Logits
                        # Note: softmax_entropy expects [B, C, H, W]
                        l_z_4d = (l_z / t_opt).unsqueeze(0)
                        ent_map = softmax_entropy(l_z_4d).squeeze(0)  # [240, 240] in [0, 1]

                        pred_fg_mask = pred_cal > 0
                        pred_fg_pixels = int(pred_fg_mask.sum().item())
                        gt_fg_pixels = int((tgt_z > 0).sum().item())

                        if pred_fg_pixels == 0:
                            # MANDATORY EMPTY PREDICTION SEMANTICS:
                            # Zero predicted foreground voxels MUST produce None (never 0.0)
                            mean_fg_entropy = None
                            m_empty_slice_count += 1
                        else:
                            mean_fg_entropy = float(ent_map[pred_fg_mask].mean().item())
                            m_slice_entropies.append(mean_fg_entropy)
                            m_non_empty_slice_count += 1

                        case_slice_records.append({
                            "slice_index": z,
                            "gt_fg_pixels": gt_fg_pixels,
                            "pred_fg_pixels": pred_fg_pixels,
                            "mean_fg_entropy": mean_fg_entropy,
                            "mean_fg_dice": m_dice,
                            "wt_dice": wt_d,
                            "tc_dice": tc_d,
                            "et_dice": et_d,
                            "argmax_mismatches": mismatches,
                        })

                        # 5. Evaluation Voxel Subsampling (every 4th slice with GT foreground)
                        if z % 4 == 0:
                            fg_idx = np.where(tgt_np > 0)
                            n_fg = len(fg_idx[0])
                            if n_fg > 0:
                                brain_mask = (vol.image[:, :, :, z].sum(axis=0) != 0)
                                s_fg = min(n_fg, 100)
                                chosen_fg = rng_eval.choice(n_fg, size=s_fg, replace=False)
                                fg_r, fg_c = fg_idx[0][chosen_fg], fg_idx[1][chosen_fg]

                                l_fg = l_z[:, fg_r, fg_c].permute(1, 0)
                                t_fg = tgt_z[fg_r, fg_c]

                                bg_idx = np.where((tgt_np == 0) & brain_mask)
                                n_bg = len(bg_idx[0])
                                s_bg = min(n_bg, s_fg)

                                if s_bg > 0:
                                    chosen_bg = rng_eval.choice(n_bg, size=s_bg, replace=False)
                                    bg_r, bg_c = bg_idx[0][chosen_bg], bg_idx[1][chosen_bg]
                                    l_bg = l_z[:, bg_r, bg_c].permute(1, 0)
                                    t_bg = tgt_z[bg_r, bg_c]

                                    cat_l = torch.cat([l_fg, l_bg], dim=0)
                                    cat_t = torch.cat([t_fg, t_bg], dim=0)
                                else:
                                    cat_l = l_fg
                                    cat_t = t_fg

                                eval_sampled_logits.append(cat_l)
                                eval_sampled_targets.append(cat_t)
                                case_sampled_count += cat_l.shape[0]
                                total_eval_voxels += cat_l.shape[0]

                eval_case_voxel_counts[cid] = case_sampled_count
                case_dice = vol_accum.compute_dice()
                case_elapsed = time.perf_counter() - t_c0

                case_record = {
                    "case_id": cid,
                    "model_name": m_name,
                    "volumetric_dice": case_dice,
                    "sampled_voxels": case_sampled_count,
                    "empty_prediction_slices": sum(1 for s in case_slice_records if s["pred_fg_pixels"] == 0),
                    "non_empty_slices": sum(1 for s in case_slice_records if s["pred_fg_pixels"] > 0),
                    "mean_case_entropy": float(np.mean([s["mean_fg_entropy"] for s in case_slice_records if s["mean_fg_entropy"] is not None])) if any(s["mean_fg_entropy"] is not None for s in case_slice_records) else None,
                    "elapsed_sec": case_elapsed,
                    "slice_records": case_slice_records,
                }
                m_case_results.append(case_record)
                all_case_level_records.append({
                    "case_id": cid,
                    "model_name": m_name,
                    "mean_dice": case_dice["mean_dice"],
                    "wt_dice": case_dice["wt_dice"],
                    "tc_dice": case_dice["tc_dice"],
                    "et_dice": case_dice["et_dice"],
                    "sampled_voxels": case_sampled_count,
                    "empty_slices": case_record["empty_prediction_slices"],
                    "mean_entropy": case_record["mean_case_entropy"],
                })

                with open(model_pred_dir / f"{cid}.json", "w", encoding="utf-8") as f:
                    json.dump(case_record, f, indent=2)

                del vol
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if c_idx % 20 == 0 or c_idx == len(eval_case_ids):
                    print(f"  [{c_idx:3d}/{len(eval_case_ids)}] {cid} | Vol Dice: {case_dice['mean_dice']:.4f} | Eval Voxels: {total_eval_voxels:,}")

        assert total_eval_voxels == 311546, f"Expected 311,546 evaluation voxels, got {total_eval_voxels}"
        assert m_argmax_mismatch_voxels == 0, f"ARGMAX INVARIANCE VIOLATION: {m_argmax_mismatch_voxels} mismatched voxels!"
        print(f"Argmax Invariance STRICTLY Verified: 0 mismatch across all {len(eval_case_ids)*155:,} slices!")

        all_eval_l = torch.cat(eval_sampled_logits, dim=0)
        all_eval_t = torch.cat(eval_sampled_targets, dim=0)

        # Uncalibrated metrics on Val_eval
        p_eval_uncal = torch.softmax(all_eval_l, dim=-1)
        uncal_eval_nll = compute_nll(p_eval_uncal, all_eval_t)
        uncal_eval_ece_dict = compute_multiclass_ece(p_eval_uncal, all_eval_t, n_bins=15)
        uncal_eval_brier = compute_multiclass_brier_score(p_eval_uncal, all_eval_t)

        # Calibrated metrics on Val_eval
        p_eval_cal = torch.softmax(all_eval_l / t_opt, dim=-1)
        cal_eval_nll = compute_nll(p_eval_cal, all_eval_t)
        cal_eval_ece_dict = compute_multiclass_ece(p_eval_cal, all_eval_t, n_bins=15)
        cal_eval_brier = compute_multiclass_brier_score(p_eval_cal, all_eval_t)

        print(f"\nEvaluation Results on Val_eval ({len(eval_case_ids)} cases, {total_eval_voxels:,} voxels):")
        print(f"  NLL:   {uncal_eval_nll:.4f} -> {cal_eval_nll:.4f} (Rel: -{(1.0 - cal_eval_nll/uncal_eval_nll)*100:.1f}%)")
        print(f"  ECE:   {uncal_eval_ece_dict['ece']*100:.2f}% -> {cal_eval_ece_dict['ece']*100:.2f}% (Rel: -{(1.0 - cal_eval_ece_dict['ece']/uncal_eval_ece_dict['ece'])*100:.1f}%)")
        print(f"  MCE:   {uncal_eval_ece_dict['mce']*100:.2f}% -> {cal_eval_ece_dict['mce']*100:.2f}%")
        print(f"  Brier: {uncal_eval_brier:.4f} -> {cal_eval_brier:.4f}")

        # Mean volumetric Dice across 105 cases
        mean_vol_d = float(np.mean([c["volumetric_dice"]["mean_dice"] for c in m_case_results]))
        wt_vol_d = float(np.mean([c["volumetric_dice"]["wt_dice"] for c in m_case_results]))
        tc_vol_d = float(np.mean([c["volumetric_dice"]["tc_dice"] for c in m_case_results]))
        et_vol_d = float(np.mean([c["volumetric_dice"]["et_dice"] for c in m_case_results]))
        print(f"  Volumetric Mean Dice: {mean_vol_d:.4f} (WT: {wt_vol_d:.4f}, TC: {tc_vol_d:.4f}, ET: {et_vol_d:.4f})")

        # Predictive entropy statistics across non-empty slices
        ent_arr = np.array(m_slice_entropies)
        ent_mean = float(np.mean(ent_arr))
        ent_std = float(np.std(ent_arr))
        ent_median = float(np.median(ent_arr))
        ent_p25 = float(np.percentile(ent_arr, 25))
        ent_p75 = float(np.percentile(ent_arr, 75))
        ent_min = float(np.min(ent_arr))
        ent_max = float(np.max(ent_arr))

        print(f"  Predictive Entropy (Calibrated, {m_non_empty_slice_count:,} non-empty slices):")
        print(f"    Mean: {ent_mean:.4f} ± {ent_std:.4f} | Median: {ent_median:.4f} [IQR: {ent_p25:.4f} - {ent_p75:.4f}]")
        print(f"    Range: [{ent_min:.4f}, {ent_max:.4f}] | Empty Slices: {m_empty_slice_count:,} ({m_empty_slice_count/(m_empty_slice_count+m_non_empty_slice_count)*100:.2f}%)")

        # Save single model evaluation metrics JSON
        eval_record = {
            "model_name": m_name,
            "display_name": d_name,
            "parameter_count": verified_checkpoints[m_name]["parameter_count"],
            "expected_best_epoch": verified_checkpoints[m_name]["expected_best_epoch"],
            "checkpoint_sha256": verified_checkpoints[m_name]["sha256"],
            "optimal_temperature_T": t_opt,
            "argmax_invariance_mismatches": m_argmax_mismatch_voxels,
            "volumetric_dice": {
                "mean_dice": mean_vol_d,
                "wt_dice": wt_vol_d,
                "tc_dice": tc_vol_d,
                "et_dice": et_vol_d,
            },
            "uncalibrated": {
                "nll": uncal_eval_nll,
                "ece": uncal_eval_ece_dict["ece"],
                "mce": uncal_eval_ece_dict["mce"],
                "brier_score": uncal_eval_brier,
                "bin_accuracies": uncal_eval_ece_dict["bin_accuracies"],
                "bin_confidences": uncal_eval_ece_dict["bin_confidences"],
                "bin_counts": uncal_eval_ece_dict["bin_counts"],
                "bin_edges": uncal_eval_ece_dict["bin_edges"],
            },
            "calibrated": {
                "nll": cal_eval_nll,
                "ece": cal_eval_ece_dict["ece"],
                "mce": cal_eval_ece_dict["mce"],
                "brier_score": cal_eval_brier,
                "bin_accuracies": cal_eval_ece_dict["bin_accuracies"],
                "bin_confidences": cal_eval_ece_dict["bin_confidences"],
                "bin_counts": cal_eval_ece_dict["bin_counts"],
                "bin_edges": cal_eval_ece_dict["bin_edges"],
            },
            "predictive_entropy": {
                "total_slices": m_empty_slice_count + m_non_empty_slice_count,
                "empty_slices": m_empty_slice_count,
                "non_empty_slices": m_non_empty_slice_count,
                "empty_slice_pct": (m_empty_slice_count / (m_empty_slice_count + m_non_empty_slice_count)) * 100.0,
                "mean": ent_mean,
                "std": ent_std,
                "median": ent_median,
                "p25": ent_p25,
                "p75": ent_p75,
                "min": ent_min,
                "max": ent_max,
            },
            "evaluation_cohort": {
                "name": "Val_eval",
                "cases_count": len(eval_case_ids),
                "unique_subjects": len(eval_subjects),
                "total_sampled_voxels": total_eval_voxels,
                "splits_hash": splits_sha256,
            },
            "elapsed_seconds": time.perf_counter() - t_model_start,
        }

        with open(dir_metrics / f"{m_name}_eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(eval_record, f, indent=2)

        # Save reliability diagram bin data
        with open(dir_rel / f"{m_name}_bins.json", "w", encoding="utf-8") as f:
            json.dump({
                "model_name": m_name,
                "optimal_temperature": t_opt,
                "uncalibrated": eval_record["uncalibrated"],
                "calibrated": eval_record["calibrated"],
            }, f, indent=2)

        # Plot individual reliability diagram
        plot_reliability_diagram(
            model_name=m_name,
            display_name=d_name,
            uncal_data=eval_record["uncalibrated"],
            cal_data=eval_record["calibrated"],
            t_opt=t_opt,
            out_path=dir_rel / f"{m_name}_reliability.png",
        )

        models_evaluation_summary.append({
            "model_name": m_name,
            "display_name": d_name,
            "optimal_temperature": t_opt,
            "eval_uncal": eval_record["uncalibrated"],
            "eval_cal": eval_record["calibrated"],
            "volumetric_dice": eval_record["volumetric_dice"],
            "entropy_stats": eval_record["predictive_entropy"],
            "elapsed_s": eval_record["elapsed_seconds"],
        })

        models_entropy_summary.append({
            "model": d_name,
            "total_slices": eval_record["predictive_entropy"]["total_slices"],
            "empty_prediction_slices": eval_record["predictive_entropy"]["empty_slices"],
            "non_empty_slices": eval_record["predictive_entropy"]["non_empty_slices"],
            "empty_slice_pct": f"{eval_record['predictive_entropy']['empty_slice_pct']:.2f}%",
            "mean_entropy": f"{ent_mean:.4f}",
            "std_entropy": f"{ent_std:.4f}",
            "median_entropy": f"{ent_median:.4f}",
            "iqr_entropy": f"[{ent_p25:.4f}, {ent_p75:.4f}]",
            "min_entropy": f"{ent_min:.4f}",
            "max_entropy": f"{ent_max:.4f}",
            "argmax_invariance_mismatches": m_argmax_mismatch_voxels,
        })

        # Cleanup model from GPU
        del model
        del eval_sampled_logits
        del eval_sampled_targets
        del all_eval_l
        del all_eval_t
        del p_eval_uncal
        del p_eval_cal
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"COMPLETED {d_name} in {time.perf_counter() - t_model_start:.1f}s")

    # -------------------------------------------------------------
    # Phase C: Multi-Model Visualization, Aggregations & Run Manifest
    # -------------------------------------------------------------
    print("\n--- Phase C: Aggregations, Visualizations, and Manifest ---")

    # 1. Multi-panel reliability diagram (5 models)
    five_panel_png = dir_rel / "five_model_reliability_diagrams.png"
    plot_five_model_reliability_diagrams(models_evaluation_summary, five_panel_png)
    print(f"Saved 5-model reliability diagram: {five_panel_png}")

    # 2. Consolidated calibration_parameters.json
    all_cal_params_path = out_base / "calibration_parameters.json"
    with open(all_cal_params_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "provenance": {
                    "splits_sha256": splits_sha256,
                    "cohort_cal": {"cases": len(cal_case_ids), "subjects": len(cal_subjects), "voxels": 5020730},
                    "cohort_eval": {"cases": len(eval_case_ids), "subjects": len(eval_subjects), "voxels": 311546},
                },
                "models": models_calibration_params,
            },
            f,
            indent=2,
        )
    print(f"Saved: {all_cal_params_path}")

    # 3. Consolidated calibration_metrics.csv (Val_cal)
    cal_metrics_csv_path = out_base / "calibration_metrics.csv"
    with open(cal_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "parameters",
            "best_epoch",
            "optimal_temperature",
            "val_cal_uncal_nll",
            "val_cal_cal_nll",
            "nll_relative_reduction_pct",
            "val_cal_uncal_ece",
            "val_cal_cal_ece",
            "ece_relative_reduction_pct",
            "val_cal_uncal_mce",
            "val_cal_cal_mce",
            "val_cal_uncal_brier",
            "val_cal_cal_brier",
            "iterations",
            "converged",
        ])
        for m_cfg in MODELS_CONFIG:
            m_k = m_cfg["name"]
            p = models_calibration_params[m_k]
            u = p["val_cal_metrics"]["uncalibrated"]
            c = p["val_cal_metrics"]["calibrated"]
            nll_red = (1.0 - (c["nll"] / u["nll"])) * 100.0
            ece_red = (1.0 - (c["ece"] / u["ece"])) * 100.0
            writer.writerow([
                m_cfg["display_name"],
                p["parameter_count"],
                p["expected_best_epoch"],
                f"{p['optimal_temperature_T']:.6f}",
                f"{u['nll']:.6f}",
                f"{c['nll']:.6f}",
                f"{nll_red:.2f}%",
                f"{u['ece']:.6f}",
                f"{c['ece']:.6f}",
                f"{ece_red:.2f}%",
                f"{u['mce']:.6f}",
                f"{c['mce']:.6f}",
                f"{u['brier_score']:.6f}",
                f"{c['brier_score']:.6f}",
                p["fit_diagnostics"]["iterations"],
                p["fit_diagnostics"]["converged"],
            ])
    print(f"Saved: {cal_metrics_csv_path}")

    # 4. Consolidated evaluation_metrics.csv (Val_eval)
    eval_metrics_csv_path = out_base / "evaluation_metrics.csv"
    with open(eval_metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "parameters",
            "best_epoch",
            "temperature",
            "val_eval_uncal_nll",
            "val_eval_cal_nll",
            "nll_reduction_pct",
            "val_eval_uncal_ece",
            "val_eval_cal_ece",
            "ece_reduction_pct",
            "val_eval_uncal_mce",
            "val_eval_cal_mce",
            "val_eval_uncal_brier",
            "val_eval_cal_brier",
            "mean_volumetric_dice",
            "wt_volumetric_dice",
            "tc_volumetric_dice",
            "et_volumetric_dice",
        ])
        for m_res in models_evaluation_summary:
            m_k = m_res["model_name"]
            d_k = m_res["display_name"]
            p = models_calibration_params[m_k]
            u = m_res["eval_uncal"]
            c = m_res["eval_cal"]
            v = m_res["volumetric_dice"]
            nll_red = (1.0 - (c["nll"] / u["nll"])) * 100.0
            ece_red = (1.0 - (c["ece"] / u["ece"])) * 100.0
            writer.writerow([
                d_k,
                p["parameter_count"],
                p["expected_best_epoch"],
                f"{m_res['optimal_temperature']:.6f}",
                f"{u['nll']:.6f}",
                f"{c['nll']:.6f}",
                f"{nll_red:.2f}%",
                f"{u['ece']:.6f}",
                f"{c['ece']:.6f}",
                f"{ece_red:.2f}%",
                f"{u['mce']:.6f}",
                f"{c['mce']:.6f}",
                f"{u['brier_score']:.6f}",
                f"{c['brier_score']:.6f}",
                f"{v['mean_dice']:.4f}",
                f"{v['wt_dice']:.4f}",
                f"{v['tc_dice']:.4f}",
                f"{v['et_dice']:.4f}",
            ])
    print(f"Saved: {eval_metrics_csv_path}")

    # 5. Consolidated entropy_summary.csv
    ent_summary_csv_path = out_base / "entropy_summary.csv"
    with open(ent_summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model",
            "total_slices",
            "empty_prediction_slices",
            "non_empty_slices",
            "empty_slice_pct",
            "mean_entropy",
            "std_entropy",
            "median_entropy",
            "iqr_entropy",
            "min_entropy",
            "max_entropy",
            "argmax_invariance_mismatches",
        ])
        writer.writeheader()
        for r in models_entropy_summary:
            writer.writerow(r)
    print(f"Saved: {ent_summary_csv_path}")

    # 6. Case-level evaluation summary CSV
    case_summary_csv_path = out_base / "evaluation_metrics" / "case_level_evaluation_summary.csv"
    with open(case_summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id",
            "model_name",
            "mean_dice",
            "wt_dice",
            "tc_dice",
            "et_dice",
            "sampled_voxels",
            "empty_slices",
            "mean_entropy",
        ])
        writer.writeheader()
        for r in all_case_level_records:
            writer.writerow(r)
    print(f"Saved: {case_summary_csv_path}")

    # 7. Comprehensive c2_run_manifest.json
    total_elapsed = time.perf_counter() - t_start_total
    manifest_record = {
        "work_package": "Stage C Phase 2 (C2): Five-Model Probability Calibration & Uncertainty Analysis",
        "timestamp_start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_start_total)),
        "timestamp_end": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": total_elapsed,
        "environment": {
            "os": platform.platform(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_device": gpu_name,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        },
        "provenance_hashes": {
            "splits_json_sha256": splits_sha256,
            "script_sha256": get_file_sha256(Path(__file__).resolve()),
            "checkpoints": {m: verified_checkpoints[m]["sha256"] for m in verified_checkpoints},
            "output_calibration_parameters_sha256": get_file_sha256(all_cal_params_path),
            "output_calibration_metrics_csv_sha256": get_file_sha256(cal_metrics_csv_path),
            "output_evaluation_metrics_csv_sha256": get_file_sha256(eval_metrics_csv_path),
            "output_entropy_summary_csv_sha256": get_file_sha256(ent_summary_csv_path),
        },
        "cohorts": {
            "val_cal": {"cases": len(cal_case_ids), "subjects": len(cal_subjects), "voxels": 5020730},
            "val_eval": {"cases": len(eval_case_ids), "subjects": len(eval_subjects), "voxels": 311546},
            "subject_disjointness_verified": True,
        },
        "fitted_temperatures": {m: models_calibration_params[m]["optimal_temperature_T"] for m in models_calibration_params},
        "argmax_invariance_all_zero": all(r["argmax_invariance_mismatches"] == 0 for r in models_entropy_summary),
    }

    manifest_path = out_base / "c2_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_record, f, indent=2)
    print(f"Saved: {manifest_path}")

    print("\n===============================================================================")
    print(f"C2 EXECUTION COMPLETE IN {total_elapsed/60.0:.2f} MINUTES")
    print("===============================================================================")


if __name__ == "__main__":
    run_pipeline()
