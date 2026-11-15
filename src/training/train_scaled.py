"""Manifest-driven full-scale training script for BraTS-GLI candidates.

Gate 5D Authorized Pipeline:
- Loads training cases from development_train_manifest.json (721 cases)
- Loads validation cases from validation_manifest.json (155 cases)
- Strictly enforces test quarantine (0 test cases allowed)
- Chained SequentialLR: 2-epoch LinearLR warmup (0.1x -> 1.0x) + 13-epoch CosineAnnealingLR
- Gradient clipping (max_grad_norm=1.0) and finite loss/gradient checks
- Best validation Mean Dice checkpointing and final model checkpointing
- Full telemetry logging: pre/post grad norms, clipped batch count, learning rates, VRAM, and runtime
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import nibabel as nib
import torch
import torch.nn as nn

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (
    BraTSGLISliceDataset,
    CaseFiles,
    CaseSliceBatchSampler,
    MODALITIES,
    SEGMENTATION_SUFFIX,
)
from src.evaluation.metrics import SegmentationMetricAccumulator
from src.models.attention_unet import AttentionUNet2D
from src.models.resunet import ResUNet2D
from src.models.unet import UNet2D
from src.models.unet_resnet34 import UNetResNet34
from src.models.unetpp import UNetPlusPlus2D
from src.training.train import DiceCrossEntropyLoss, count_parameters, set_reproducible_seed


# Ensure standard output and error use UTF-8 where possible with line buffering
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

class DualLogger:
    """Tee stdout to both console and a log file with line flushing and encoding resilience."""

    def __init__(self, filepath: Path) -> None:
        self.terminal = sys.stdout
        self.log_file = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            encoding = getattr(self.terminal, "encoding", "ascii") or "ascii"
            safe_msg = message.encode(encoding, errors="replace").decode(encoding)
            self.terminal.write(safe_msg)
        self.log_file.write(message)
        self.terminal.flush()
        self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()


def resolve_case_files(manifest_cases: list[dict], raw_dir: Path, archive_dir: Path) -> list[CaseFiles]:
    """Resolve case files from either data/raw or data/extracted_archive based on source_location."""
    case_files_list: list[CaseFiles] = []
    for entry in manifest_cases:
        cid = entry["case_id"]
        src = entry["source_location"]
        cdir = (raw_dir / cid) if src == "disk_raw" else (archive_dir / cid)

        if not cdir.exists():
            raise FileNotFoundError(f"Case directory not found: {cdir}")

        mod_paths = {m: cdir / f"{cid}-{m}.nii.gz" for m in MODALITIES}
        seg_path = cdir / f"{cid}-{SEGMENTATION_SUFFIX}.nii.gz"

        for mod, p in mod_paths.items():
            if not p.exists():
                raise FileNotFoundError(f"Missing {mod} file for {cid}: {p}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation mask for {cid}: {seg_path}")

        shape = tuple(int(dim) for dim in nib.load(str(seg_path)).shape)
        case_files_list.append(
            CaseFiles(
                case_id=cid,
                case_dir=cdir,
                modality_paths=mod_paths,
                segmentation_path=seg_path,
                shape=shape,
            )
        )
    return case_files_list


def create_model(model_name: str, base_channels: int = 32, num_classes: int = 4) -> nn.Module:
    normalized = model_name.lower().replace("-", "_")
    if normalized in ("unet", "unet_2d"):
        return UNet2D(in_channels=4, num_classes=num_classes, base_channels=base_channels)
    elif normalized in ("unetpp", "unet_pp", "unetplusplus", "unetplusplus_2d"):
        return UNetPlusPlus2D(
            in_channels=4,
            num_classes=num_classes,
            base_channels=base_channels,
            deep_supervision=False,
        )
    elif normalized in ("attention_unet", "attention_unet_2d", "att_unet"):
        return AttentionUNet2D(in_channels=4, num_classes=num_classes, base_channels=base_channels)
    elif normalized in ("resunet", "resunet_2d"):
        return ResUNet2D(in_channels=4, num_classes=num_classes, base_channels=base_channels)
    elif normalized in ("unet_resnet34", "resnet34_unet"):
        return UNetResNet34(in_channels=4, num_classes=num_classes, base_channels=base_channels, pretrained=True)
    else:
        raise ValueError(f"Unsupported model: {model_name}")


class PerGroupCosineAnnealingLR(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine Annealing scheduler supporting a separate eta_min per parameter group.

    If eta_min is a float, it applies uniformly to all parameter groups.
    If eta_min is a Sequence[float], each parameter group receives its respective eta_min floor.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_max: int,
        eta_min: float | list[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if isinstance(eta_min, (list, tuple)):
            self.eta_min = list(eta_min)
        else:
            self.eta_min = float(eta_min)
        super().__init__(optimizer, T_max, 0.0, last_epoch)
        if isinstance(eta_min, (list, tuple)):
            self.eta_min = list(eta_min)
        else:
            self.eta_min = float(eta_min)

    def _get_eta_mins(self) -> list[float]:
        if isinstance(self.eta_min, list):
            return self.eta_min
        return [float(self.eta_min)] * len(self.optimizer.param_groups)

    def get_lr(self) -> list[float]:
        eta_mins = self._get_eta_mins()
        if self._step_count == 1 and self.last_epoch > 0:
            return [
                eta
                + (base_lr - eta)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, eta, group in zip(
                    self.base_lrs, eta_mins, self.optimizer.param_groups
                )
            ]
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return [
                group["lr"]
                + (base_lr - eta) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, eta, group in zip(
                    self.base_lrs, eta_mins, self.optimizer.param_groups
                )
            ]
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - eta)
            + eta
            for group, eta in zip(self.optimizer.param_groups, eta_mins)
        ]

    def _get_closed_form_lr(self) -> list[float]:
        eta_mins = self._get_eta_mins()
        return [
            eta
            + (base_lr - eta)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr, eta in zip(self.base_lrs, eta_mins)
        ]


def build_optimizer_and_scheduler(
    model: nn.Module,
    model_name: str,
    base_lr: float,
    warmup_epochs: int,
    total_epochs: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.SequentialLR]:
    # Optimizer configuration
    if hasattr(model, "get_parameter_groups"):
        param_groups = model.get_parameter_groups(base_lr=base_lr)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        eta_min = [1e-6, 1e-5]  # encoder floor 1e-6, decoder floor 1e-5
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
        eta_min = 1e-5  # uniform floor 1e-5

    # Scheduler: LinearLR warmup followed by PerGroupCosineAnnealingLR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine_scheduler = PerGroupCosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=eta_min,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )
    return optimizer, scheduler


def run_train_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_grad_norm: float = 1.0,
    max_batches: int | None = None,
) -> tuple[float, float, float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_pre_norm = 0.0
    total_post_norm = 0.0
    clipped_count = 0
    num_batches = 0

    num_total_batches = min(len(data_loader), max_batches) if max_batches is not None else len(data_loader)
    epoch_start = time.perf_counter()

    for batch_idx, (images, masks) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, masks)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss encountered at epoch {epoch}, batch {batch_idx}: loss={loss.item()}")

        loss.backward()

        pre_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False)
        post_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float("inf"))

        if not torch.isfinite(pre_norm):
            raise FloatingPointError(f"Non-finite gradient norm at epoch {epoch}, batch {batch_idx}: norm={pre_norm.item()}")

        if pre_norm.item() > max_grad_norm:
            clipped_count += 1

        optimizer.step()

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_pre_norm += float(pre_norm.item())
        total_post_norm += float(post_norm.item())
        num_batches += 1

        if (batch_idx + 1) % 2500 == 0 or (batch_idx + 1) == num_total_batches:
            elapsed = (time.perf_counter() - epoch_start) / 60.0
            print(
                f"  train {epoch}/{total_epochs} [{batch_idx + 1}/{num_total_batches}] | "
                f"loss={loss.item():.4f} | pre_norm={pre_norm.item():.2f} | "
                f"elapsed={elapsed:.1f} min"
            )

    avg_loss = total_loss / max(total_samples, 1)
    avg_pre_norm = total_pre_norm / max(num_batches, 1)
    avg_post_norm = total_post_norm / max(num_batches, 1)
    return avg_loss, avg_pre_norm, avg_post_norm, clipped_count


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_batches: int | None = None,
) -> tuple[float, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    accumulator = SegmentationMetricAccumulator(num_classes=4, ignore_background=True)

    for batch_idx, (images, masks) in enumerate(data_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        batch_size = images.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        accumulator.update(logits, masks)

    avg_loss = total_loss / max(total_samples, 1)
    metrics = accumulator.compute()
    return avg_loss, metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    metrics: Any,
    peak_vram_mb: float,
) -> None:
    temp_path = path.with_suffix(".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "metrics": metrics.as_dict(),
            "peak_vram_mb": peak_vram_mb,
        },
        temp_path,
    )
    temp_path.replace(path)


def load_and_verify_cohort_manifests(
    splits_dir: Path,
    expected_train_cases: int = 721,
    expected_train_subjects: int = 667,
    expected_val_cases: int = 155,
    expected_val_subjects: int = 143,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load authorized training and validation manifests and verify strict cohort integrity.

    Verifications performed:
      1. Existence and valid JSON parsing of permitted manifests (development_train_manifest.json, validation_manifest.json).
      2. Key structure validation ('cases' list of dicts with 'case_id' and 'subject_id').
      3. Expected case census (721 train cases, 155 validation cases).
      4. Expected unique biological subject census (667 train subjects, 143 validation subjects).
      5. Strict case-ID disjointness (train_cases & val_cases == empty set).
      6. Strict biological-subject disjointness (train_subjects & val_subjects == empty set).

    Test Quarantine Enforcement:
      This function NEVER opens, reads, parses, enumerates, or hashes test_manifest.json
      or any quarantine path, preserving the pre-registered sealed test quarantine boundary.
    """
    train_manifest_path = splits_dir / "development_train_manifest.json"
    val_manifest_path = splits_dir / "validation_manifest.json"

    if not train_manifest_path.exists():
        raise FileNotFoundError(f"Missing required development training manifest: {train_manifest_path}")
    if not val_manifest_path.exists():
        raise FileNotFoundError(f"Missing required validation manifest: {val_manifest_path}")

    try:
        with open(train_manifest_path, "r", encoding="utf-8") as f:
            train_manifest = json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed training manifest JSON at {train_manifest_path}: {exc}") from exc

    try:
        with open(val_manifest_path, "r", encoding="utf-8") as f:
            val_manifest = json.load(f)
    except Exception as exc:
        raise ValueError(f"Malformed validation manifest JSON at {val_manifest_path}: {exc}") from exc

    if not isinstance(train_manifest, dict) or "cases" not in train_manifest or not isinstance(train_manifest["cases"], list):
        raise ValueError("Training manifest missing valid 'cases' list")
    if not isinstance(val_manifest, dict) or "cases" not in val_manifest or not isinstance(val_manifest["cases"], list):
        raise ValueError("Validation manifest missing valid 'cases' list")

    train_cases_info = train_manifest["cases"]
    val_cases_info = val_manifest["cases"]

    if len(train_cases_info) != expected_train_cases:
        raise ValueError(f"Expected {expected_train_cases} train cases, got {len(train_cases_info)}")
    if len(val_cases_info) != expected_val_cases:
        raise ValueError(f"Expected {expected_val_cases} val cases, got {len(val_cases_info)}")

    train_cids = {c["case_id"] for c in train_cases_info if isinstance(c, dict) and "case_id" in c}
    val_cids = {c["case_id"] for c in val_cases_info if isinstance(c, dict) and "case_id" in c}

    if len(train_cids) != expected_train_cases:
        raise ValueError(f"Malformed training manifest: expected {expected_train_cases} unique case_ids, found {len(train_cids)}")
    if len(val_cids) != expected_val_cases:
        raise ValueError(f"Malformed validation manifest: expected {expected_val_cases} unique case_ids, found {len(val_cids)}")

    overlap_cids = train_cids & val_cids
    if overlap_cids:
        raise ValueError(f"FATAL: Train and Val case ID overlap detected ({len(overlap_cids)} cases): {sorted(list(overlap_cids))[:5]}")

    train_sids = {c["subject_id"] for c in train_cases_info if isinstance(c, dict) and "subject_id" in c}
    val_sids = {c["subject_id"] for c in val_cases_info if isinstance(c, dict) and "subject_id" in c}

    if len(train_sids) != expected_train_subjects:
        raise ValueError(f"Expected {expected_train_subjects} unique train subjects, got {len(train_sids)}")
    if len(val_sids) != expected_val_subjects:
        raise ValueError(f"Expected {expected_val_subjects} unique val subjects, got {len(val_sids)}")

    overlap_sids = train_sids & val_sids
    if overlap_sids:
        raise ValueError(f"FATAL: Biological subject overlap detected between train and val ({len(overlap_sids)} subjects): {sorted(list(overlap_sids))[:5]}")

    return train_cases_info, val_cases_info


def train_candidate(
    model_name: str,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 2,
    base_lr: float = 1e-3,
    warmup_epochs: int = 2,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    resume: bool = False,
    max_batches_per_epoch: int | None = None,
) -> dict[str, Any]:
    # Prevent Windows from entering system sleep while training (allows display sleep)
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        except Exception:
            pass

    set_reproducible_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    splits_dir = PROJECT_ROOT / "results" / "dataset_splits"
    raw_dir = PROJECT_ROOT / "data" / "raw"
    archive_dir = PROJECT_ROOT / "data" / "extracted_archive"

    # --- 1. Load Manifests and Verify Cohort Integrity ---
    print(f"Loading authorized manifests for {model_name}...")
    train_cases_info, val_cases_info = load_and_verify_cohort_manifests(splits_dir)
    print("Cohort integrity verified: Train (721 cases / 667 subjects) and Val (155 cases / 143 subjects) are strictly disjoint.")

    # --- 2. Build Datasets and Loaders ---
    print("Resolving case files and building dataset slices...")
    train_cases = resolve_case_files(train_cases_info, raw_dir, archive_dir)
    val_cases = resolve_case_files(val_cases_info, raw_dir, archive_dir)

    train_dataset = BraTSGLISliceDataset(train_cases, background_slice_ratio=0.25, seed=seed)
    val_dataset = BraTSGLISliceDataset(val_cases, background_slice_ratio=0.25, seed=seed + 1)

    print(f"Dataset built: {len(train_dataset):,} train slices, {len(val_dataset):,} val slices.")

    train_sampler = CaseSliceBatchSampler(train_dataset, batch_size=batch_size, shuffle_cases=True, shuffle_slices=True, seed=seed)
    val_sampler = CaseSliceBatchSampler(val_dataset, batch_size=batch_size, shuffle_cases=False, shuffle_slices=False, seed=seed + 1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0, pin_memory=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0, pin_memory=False)

    # --- 3. Build Model, Criterion, Optimizer, Scheduler ---
    print(f"Instantiating model {model_name}...")
    model = create_model(model_name).to(device)
    param_count = count_parameters(model)
    print(f"Model parameters: {param_count:,}")

    criterion = DiceCrossEntropyLoss(num_classes=4, dice_weight=1.0, ce_weight=1.0)
    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        model_name=model_name,
        base_lr=base_lr,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = DualLogger(output_dir / "train.log")
    original_stdout = sys.stdout
    sys.stdout = logger

    best_checkpoint_path = output_dir / "best_model.pt"
    final_checkpoint_path = output_dir / "final_model.pt"
    history_csv_path = output_dir / "history.csv"
    summary_json_path = output_dir / "summary.json"
    config_json_path = output_dir / "config.json"

    # Save run configuration
    with open(config_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "parameters": param_count,
                "epochs": epochs,
                "batch_size": batch_size,
                "base_lr": base_lr,
                "warmup_epochs": warmup_epochs,
                "max_grad_norm": max_grad_norm,
                "seed": seed,
                "train_cases": len(train_cases),
                "val_cases": len(val_cases),
                "train_slices": len(train_dataset),
                "val_slices": len(val_dataset),
                "device": str(device),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            f,
            indent=2,
        )

    # --- 4. Training Loop ---
    start_epoch = 1
    best_mean_dice = -math.inf
    best_epoch = 0
    history: list[dict[str, Any]] = []
    prior_duration_min = 0.0
    run_start_time = time.perf_counter()

    if resume and final_checkpoint_path.exists():
        print(f"\nResuming training from checkpoint: {final_checkpoint_path}")
        ckpt = torch.load(final_checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Loaded checkpoint at epoch {ckpt['epoch']}. Resuming from epoch {start_epoch} to {epochs}.")

        train_sampler.epoch = ckpt["epoch"]
        val_sampler.epoch = ckpt["epoch"]

        if history_csv_path.exists():
            with open(history_csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    typed_row: dict[str, Any] = {}
                    for k, v in r.items():
                        if k in ("epoch", "clipped_batches"):
                            typed_row[k] = int(v)
                        elif k == "is_best":
                            typed_row[k] = v.lower() == "true"
                        else:
                            try:
                                typed_row[k] = float(v)
                            except ValueError:
                                typed_row[k] = v
                    history.append(typed_row)

        for r in history:
            if r.get("is_best", False) and r.get("mean_dice", -1.0) > best_mean_dice:
                best_mean_dice = r["mean_dice"]
                best_epoch = r["epoch"]
        if best_epoch == 0 and history:
            best_r = max(history, key=lambda x: x["mean_dice"])
            best_mean_dice = best_r["mean_dice"]
            best_epoch = best_r["epoch"]

        if history:
            prior_duration_min = history[-1].get("cumulative_duration_min", 0.0)

        print(f"Loaded {len(history)} prior history records. Prior best epoch: {best_epoch} with Mean Dice = {best_mean_dice:.4f}")

    print(f"\n=======================================================")
    print(f"STARTING 15-EPOCH TRAINING: {model_name} (Epochs {start_epoch} to {epochs})")
    print(f"Output Directory: {output_dir}")
    print(f"Device: {device} | Total Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")
    print(f"=======================================================\n")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        current_lrs = [pg["lr"] for pg in optimizer.param_groups]

        print(f"--- Epoch {epoch}/{epochs} | LRs: {[f'{lr:.6f}' for lr in current_lrs]} ---")

        train_loss, pre_norm, post_norm, clipped_count = run_train_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_grad_norm=max_grad_norm,
            max_batches=max_batches_per_epoch,
        )

        val_loss, metrics = run_validation_epoch(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            max_batches=max_batches_per_epoch,
        )

        # Step scheduler at epoch completion
        scheduler.step()
        next_lrs = [pg["lr"] for pg in optimizer.param_groups]

        epoch_elapsed_min = (time.perf_counter() - epoch_start) / 60.0
        total_elapsed_min = prior_duration_min + (time.perf_counter() - run_start_time) / 60.0
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0.0

        is_best = metrics.mean_dice > best_mean_dice
        if is_best:
            best_mean_dice = metrics.mean_dice
            best_epoch = epoch
            save_checkpoint(best_checkpoint_path, model, optimizer, scheduler, epoch, metrics, peak_vram_mb)

        # Save final checkpoint each epoch
        save_checkpoint(final_checkpoint_path, model, optimizer, scheduler, epoch, metrics, peak_vram_mb)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log epoch record
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_dice": metrics.mean_dice,
            "mean_iou": metrics.mean_iou,
            "dice_class_0": metrics.per_class_dice[0],
            "dice_class_1": metrics.per_class_dice[1],
            "dice_class_2": metrics.per_class_dice[2],
            "dice_class_3": metrics.per_class_dice[3],
            "iou_class_0": metrics.per_class_iou[0],
            "iou_class_1": metrics.per_class_iou[1],
            "iou_class_2": metrics.per_class_iou[2],
            "iou_class_3": metrics.per_class_iou[3],
            "grad_norm_pre": pre_norm,
            "grad_norm_post": post_norm,
            "clipped_batches": clipped_count,
            "learning_rate_primary": current_lrs[-1],
            "learning_rate_encoder": current_lrs[0],
            "epoch_duration_min": epoch_elapsed_min,
            "cumulative_duration_min": total_elapsed_min,
            "peak_vram_mb": peak_vram_mb,
            "is_best": is_best,
        }
        history.append(row)

        # Print detailed epoch summary
        best_marker = " [* BEST]" if is_best else ""
        total_batches_run = min(len(train_loader), max_batches_per_epoch) if max_batches_per_epoch is not None else len(train_loader)
        clip_pct = (clipped_count / max(total_batches_run, 1)) * 100.0
        print(
            f"Epoch {epoch:2d}/{epochs} Summary | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"mean_dice={metrics.mean_dice:.4f}{best_marker} | mean_iou={metrics.mean_iou:.4f} | "
            f"ET(C3)={metrics.per_class_dice[3]:.4f} | ED(C2)={metrics.per_class_dice[2]:.4f} | NCR(C1)={metrics.per_class_dice[1]:.4f} | "
            f"grad_norm={pre_norm:.2f} ({clip_pct:.1f}% clipped) | "
            f"epoch_time={epoch_elapsed_min:.1f}m | total={total_elapsed_min/60.0:.2f}h | "
            f"VRAM={peak_vram_mb:.0f}MB"
        )
        print(f"  Next Epoch LRs: {[f'{lr:.6f}' for lr in next_lrs]}\n")

        # Flush history to CSV
        with open(history_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerows(history)

    # --- 5. Compile Final Summary ---
    total_run_hours = (history[-1]["cumulative_duration_min"] / 60.0) if history else ((time.perf_counter() - run_start_time) / 3600.0)
    summary = {
        "model_name": model_name,
        "parameters": param_count,
        "total_epochs": epochs,
        "best_epoch": best_epoch,
        "best_mean_dice": best_mean_dice,
        "total_runtime_hours": total_run_hours,
        "final_epoch_metrics": history[-1],
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=======================================================")
    print(f"RUN COMPLETE: {model_name}")
    print(f"Best Epoch: {best_epoch} with Mean Dice = {best_mean_dice:.4f}")
    print(f"Total Runtime: {total_run_hours:.2f} hours")
    print(f"Best Checkpoint: {best_checkpoint_path}")
    print(f"=======================================================\n")

    sys.stdout = original_stdout
    logger.close()

    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        except Exception:
            pass

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-Scale Candidate Training Runner")
    parser.add_argument(
        "--model",
        choices=["unet", "unetpp", "attention_unet", "resunet_2d", "unet_resnet34"],
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from existing checkpoint")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None, help="Maximum batches per epoch (for smoke testing/debugging)")
    args = parser.parse_args()

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "full_training_15ep" / args.model

    train_candidate(
        model_name=args.model,
        output_dir=results_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        base_lr=args.base_lr,
        warmup_epochs=args.warmup_epochs,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        resume=args.resume,
        max_batches_per_epoch=args.max_batches_per_epoch,
    )


if __name__ == "__main__":
    main()
