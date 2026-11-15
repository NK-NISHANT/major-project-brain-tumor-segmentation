from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    BraTSGLISliceDataset,
    DEFAULT_RAW_DATA_DIR,
    CaseSliceBatchSampler,
    create_train_val_datasets,
)
from src.evaluation.metrics import SegmentationMetricAccumulator  # noqa: E402
from src.models.attention_unet import AttentionUNet2D  # noqa: E402
from src.models.resunet import ResUNet2D  # noqa: E402
from src.models.unet import UNet2D, count_parameters  # noqa: E402
from src.models.unet_resnet34 import UNetResNet34  # noqa: E402
from src.models.unetpp import UNetPlusPlus2D  # noqa: E402


try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback only used if tqdm is absent.
    tqdm = None


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "baseline_unet2d"
DEFAULT_UNETPP_RESULTS_DIR = PROJECT_ROOT / "results" / "unetpp2d_5ep"
DEFAULT_RESUNET_RESULTS_DIR = PROJECT_ROOT / "results" / "resunet2d_5ep"
DEFAULT_UNET_RESNET34_RESULTS_DIR = PROJECT_ROOT / "results" / "unet_resnet34_5ep"
MODEL_CHOICES = ("unet", "unetpp", "attention_unet", "resunet", "unet_resnet34")


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "unet"
    raw_dir: Path = DEFAULT_RAW_DATA_DIR
    results_dir: Path = DEFAULT_RESULTS_DIR
    epochs: int = 5
    batch_size: int = 2
    learning_rate: float = 1e-3
    dice_weight: float = 1.0
    ce_weight: float = 1.0
    val_fraction: float = 0.2
    background_slice_ratio: float = 0.25
    seed: int = 42
    num_workers: int = 0
    pin_memory: bool = True
    case_cache_size: int = 1
    case_wise_batches: bool = True
    base_channels: int = 32
    num_classes: int = 4
    exclude_val_subject_companions: bool = False
    pretrained: bool = True
    warmup_epochs: int = 0
    max_grad_norm: float | None = None


class DiceCrossEntropyLoss(nn.Module):
    """Combined Dice and Cross Entropy loss for imbalanced segmentation.

    Dice directly rewards overlap and is useful when tumor pixels occupy a small
    fraction of each slice. Cross Entropy keeps the class-ID supervision simple.
    """

    def __init__(
        self,
        num_classes: int = 4,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        include_background_in_dice: bool = False,
        smooth: float = 1e-5,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.include_background_in_dice = include_background_in_dice
        self.smooth = smooth
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = self.cross_entropy(logits, target)

        probabilities = torch.softmax(logits, dim=1)
        # Dice needs a one-hot target internally, but the dataset and CE target
        # stay as integer class IDs suitable for CrossEntropyLoss.
        one_hot_target = F.one_hot(target, num_classes=self.num_classes)
        one_hot_target = one_hot_target.permute(0, 3, 1, 2).to(dtype=probabilities.dtype)

        start_class = 0 if self.include_background_in_dice else 1
        probabilities = probabilities[:, start_class:, :, :]
        one_hot_target = one_hot_target[:, start_class:, :, :]

        reduce_dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot_target, dim=reduce_dims)
        denominator = torch.sum(probabilities, dim=reduce_dims) + torch.sum(one_hot_target, dim=reduce_dims)
        dice_score = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        dice_loss = 1.0 - dice_score.mean()

        return (self.dice_weight * dice_loss) + (self.ce_weight * ce_loss)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D segmentation model for BraTS-GLI.")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON configuration file.")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="unet")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--background-slice-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    pin_memory_group = parser.add_mutually_exclusive_group()
    pin_memory_group.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=True)
    pin_memory_group.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--case-cache-size", type=int, default=1)
    case_batch_group = parser.add_mutually_exclusive_group()
    case_batch_group.add_argument("--case-wise-batches", dest="case_wise_batches", action="store_true", default=True)
    case_batch_group.add_argument("--no-case-wise-batches", dest="case_wise_batches", action="store_false")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument(
        "--exclude-val-subject-companions",
        action="store_true",
        default=False,
        help="Exclude companion scans of validation subjects from training set to prevent patient leakage.",
    )
    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    pretrained_group.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear LR warmup epochs (default: 0).")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max gradient norm for clipping (default: None).")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--history-path", type=Path, default=None)
    parser.add_argument("--forward-test-only", action="store_true")
    parser.add_argument("--benchmark-data", action="store_true")
    parser.add_argument("--benchmark-batches", type=int, default=64)
    return parser.parse_args()


def resolve_training_config(args: argparse.Namespace) -> TrainingConfig:
    cfg: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "raw_dir" in cfg and cfg["raw_dir"] is not None:
            cfg["raw_dir"] = Path(cfg["raw_dir"])
        if "results_dir" in cfg and cfg["results_dir"] is not None:
            cfg["results_dir"] = Path(cfg["results_dir"])

    cli_flags = {arg.split("=")[0] for arg in sys.argv[1:] if arg.startswith("--")}

    if "--model" in cli_flags or "model_name" not in cfg:
        cfg["model_name"] = args.model
    if "--raw-dir" in cli_flags or "raw_dir" not in cfg:
        cfg["raw_dir"] = args.raw_dir
    if "--results-dir" in cli_flags or "results_dir" not in cfg:
        cfg["results_dir"] = resolve_results_dir(cfg.get("model_name", args.model), args.results_dir)
    elif "results_dir" in cfg and isinstance(cfg["results_dir"], str):
        cfg["results_dir"] = Path(cfg["results_dir"])

    if "--epochs" in cli_flags or "epochs" not in cfg:
        cfg["epochs"] = args.epochs
    if "--batch-size" in cli_flags or "batch_size" not in cfg:
        cfg["batch_size"] = args.batch_size
    if "--learning-rate" in cli_flags or "learning_rate" not in cfg:
        cfg["learning_rate"] = args.learning_rate
    if "--dice-weight" in cli_flags or "dice_weight" not in cfg:
        cfg["dice_weight"] = args.dice_weight
    if "--ce-weight" in cli_flags or "ce_weight" not in cfg:
        cfg["ce_weight"] = args.ce_weight
    if "--val-fraction" in cli_flags or "val_fraction" not in cfg:
        cfg["val_fraction"] = args.val_fraction
    if "--background-slice-ratio" in cli_flags or "background_slice_ratio" not in cfg:
        cfg["background_slice_ratio"] = args.background_slice_ratio
    if "--seed" in cli_flags or "seed" not in cfg:
        cfg["seed"] = args.seed
    if "--num-workers" in cli_flags or "num_workers" not in cfg:
        cfg["num_workers"] = args.num_workers
    if "--pin-memory" in cli_flags or "--no-pin-memory" in cli_flags or "pin_memory" not in cfg:
        cfg["pin_memory"] = args.pin_memory
    if "--case-cache-size" in cli_flags or "case_cache_size" not in cfg:
        cfg["case_cache_size"] = args.case_cache_size
    if "--case-wise-batches" in cli_flags or "--no-case-wise-batches" in cli_flags or "case_wise_batches" not in cfg:
        cfg["case_wise_batches"] = args.case_wise_batches
    if "--base-channels" in cli_flags or "base_channels" not in cfg:
        cfg["base_channels"] = args.base_channels
    if "--exclude-val-subject-companions" in cli_flags or "exclude_val_subject_companions" not in cfg:
        cfg["exclude_val_subject_companions"] = args.exclude_val_subject_companions
    if "--pretrained" in cli_flags or "--no-pretrained" in cli_flags or "pretrained" not in cfg:
        cfg["pretrained"] = args.pretrained
    if "--warmup-epochs" in cli_flags or "warmup_epochs" not in cfg:
        cfg["warmup_epochs"] = args.warmup_epochs
    if "--max-grad-norm" in cli_flags or "max_grad_norm" not in cfg:
        cfg["max_grad_norm"] = args.max_grad_norm

    valid_fields = {f.name for f in fields(TrainingConfig)}
    filtered_cfg = {k: v for k, v in cfg.items() if k in valid_fields}
    return TrainingConfig(**filtered_cfg)


def main() -> None:
    args = parse_args()
    if args.summary_only:
        print_training_summary(resolve_history_path(args.model, args.history_path))
        return
    if args.forward_test_only:
        run_forward_pass_test(model_name=args.model, base_channels=args.base_channels)
        return

    config = resolve_training_config(args)
    if args.benchmark_data:
        benchmark_data_loading(config, num_batches=args.benchmark_batches)
        return
    train(config)


def train(config: TrainingConfig) -> list[dict[str, float]]:
    validate_config(config)
    set_reproducible_seed(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset, val_dataset = create_train_val_datasets(
        raw_dir=config.raw_dir,
        val_fraction=config.val_fraction,
        background_slice_ratio=config.background_slice_ratio,
        seed=config.seed,
        case_cache_size=config.case_cache_size,
        exclude_val_subject_companions=config.exclude_val_subject_companions,
    )

    train_loader = build_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        device=device,
        case_wise_batches=config.case_wise_batches,
    )
    val_loader = build_data_loader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        device=device,
        case_wise_batches=config.case_wise_batches,
    )

    model = create_model(config).to(device)
    parameter_count = count_parameters(model)
    criterion = DiceCrossEntropyLoss(
        num_classes=config.num_classes,
        dice_weight=config.dice_weight,
        ce_weight=config.ce_weight,
    )
    if hasattr(model, "get_parameter_groups"):
        optimizer = torch.optim.AdamW(model.get_parameter_groups(base_lr=config.learning_rate))
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    config.results_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        config.results_dir / "config.json",
        build_experiment_config(config, device, parameter_count, train_dataset, val_dataset),
    )

    print_run_header(
        model_name=config.model_name,
        device=device,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        batch_size=config.batch_size,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        parameter_count=parameter_count,
    )

    history: list[dict[str, float]] = []
    best_mean_dice = -math.inf
    best_checkpoint_path = config.results_dir / "best_model.pt"
    final_checkpoint_path = config.results_dir / "final_model.pt"
    start_time = time.perf_counter()

    scheduler = None
    if config.warmup_epochs > 0:
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=config.warmup_epochs,
        )

    try:
        for epoch in range(1, config.epochs + 1):
            train_loss, pre_norm, post_norm, clipped_count = run_train_epoch(
                model=model,
                data_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                total_epochs=config.epochs,
                max_grad_norm=config.max_grad_norm,
            )
            val_loss, metrics = run_validation_epoch(model, val_loader, criterion, device, epoch, config.epochs)
            if scheduler is not None:
                scheduler.step()

            current_lr = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": float(epoch),
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "mean_dice": metrics.mean_dice,
                "mean_iou": metrics.mean_iou,
                "grad_norm_pre": pre_norm,
                "grad_norm_post": post_norm,
                "clipped_batches": float(clipped_count),
                "learning_rate": current_lr,
            }
            for class_id, value in enumerate(metrics.per_class_dice):
                row[f"dice_class_{class_id}"] = value
            for class_id, value in enumerate(metrics.per_class_iou):
                row[f"iou_class_{class_id}"] = value
            history.append(row)

            elapsed_minutes = (time.perf_counter() - start_time) / 60.0
            clip_info = f" | grad_norm={pre_norm:.2f}" + (f" (clipped {clipped_count})" if clipped_count > 0 else "")
            print(
                f"Epoch {epoch}/{config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"mean_dice={metrics.mean_dice:.4f} | "
                f"mean_iou={metrics.mean_iou:.4f}{clip_info} | "
                f"elapsed={elapsed_minutes:.1f} min"
            )

            if metrics.mean_dice > best_mean_dice:
                best_mean_dice = metrics.mean_dice
                save_checkpoint(
                    best_checkpoint_path,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    epoch=epoch,
                    metrics=row,
                    parameter_count=parameter_count,
                )

            save_history_csv(config.results_dir / "history.csv", history)
            save_json(config.results_dir / "history.json", history)

    except RuntimeError as error:
        if is_cuda_oom(error):
            clear_cuda_cache()
            raise RuntimeError(
                "CUDA ran out of memory during training. Re-run with --batch-size 1; "
                "the batch size was not changed automatically."
            ) from error
        raise

    save_checkpoint(
        final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        config=config,
        epoch=config.epochs,
        metrics=history[-1],
        parameter_count=parameter_count,
    )
    summary = {
        "best_checkpoint": str(best_checkpoint_path),
        "final_checkpoint": str(final_checkpoint_path),
        "best_mean_dice": best_mean_dice,
        "final_epoch": history[-1],
        "runtime_minutes": (time.perf_counter() - start_time) / 60.0,
    }
    save_json(config.results_dir / "summary.json", summary)
    print_training_summary(config.results_dir / "history.csv")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    return history


def run_train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    max_grad_norm: float | None = None,
) -> tuple[float, float, float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_pre_norm = 0.0
    total_post_norm = 0.0
    clipped_count = 0
    num_batches = 0

    for batch_idx, (image, mask) in enumerate(progress(data_loader, description=f"train {epoch}/{total_epochs}")):
        image = image.to(device=device, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(image)
        loss = criterion(logits, mask)

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss encountered at epoch {epoch}, batch {batch_idx}: loss={loss.item()}"
            )

        loss.backward()

        if max_grad_norm is not None and max_grad_norm > 0:
            pre_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm, error_if_nonfinite=False
            )
            post_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            if pre_norm.item() > max_grad_norm:
                clipped_count += 1
        else:
            pre_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float("inf")
            )
            post_norm = pre_norm

        if not torch.isfinite(pre_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm encountered at epoch {epoch}, batch {batch_idx}: norm={pre_norm.item()}"
            )

        optimizer.step()

        batch_size = image.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        total_pre_norm += float(pre_norm.item())
        total_post_norm += float(post_norm.item())
        num_batches += 1

    avg_loss = total_loss / max(total_samples, 1)
    avg_pre_norm = total_pre_norm / max(num_batches, 1)
    avg_post_norm = total_post_norm / max(num_batches, 1)
    return avg_loss, avg_pre_norm, avg_post_norm, clipped_count


@torch.no_grad()
def run_validation_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, object]:
    # Validation stays separated from training: no gradients, no optimizer step,
    # and no validation slices are used to update model weights.
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_accumulator = SegmentationMetricAccumulator(num_classes=4, ignore_background=True)

    for image, mask in progress(data_loader, description=f"val {epoch}/{total_epochs}"):
        image = image.to(device=device, non_blocking=True)
        mask = mask.to(device=device, non_blocking=True)

        logits = model(image)
        loss = criterion(logits, mask)
        metric_accumulator.update(logits, mask)

        batch_size = image.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1), metric_accumulator.compute()


def build_data_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    device: torch.device,
    case_wise_batches: bool = True,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    effective_pin_memory = pin_memory and device.type == "cuda"
    persistent_workers = num_workers > 0

    if case_wise_batches and isinstance(dataset, BraTSGLISliceDataset):
        batch_sampler = CaseSliceBatchSampler(
            dataset,
            batch_size=batch_size,
            shuffle_cases=shuffle,
            shuffle_slices=shuffle,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            pin_memory=effective_pin_memory,
            persistent_workers=persistent_workers,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=effective_pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )


def progress(iterable: Iterable, description: str) -> Iterable:
    if tqdm is not None:
        return tqdm(iterable, desc=description, leave=False)
    print(description)
    return iterable


def run_forward_pass_test(model_name: str = "unet", base_channels: int = 32) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TrainingConfig(model_name=model_name, base_channels=base_channels)
    model = create_model(config).to(device)
    model.eval()
    dummy_input = torch.randn(2, 4, 240, 240, device=device)
    with torch.no_grad():
        logits = model(dummy_input)
    print(f"Device: {device}")
    print(f"Model: {model_name}")
    print(f"Dummy input shape: {tuple(dummy_input.shape)}")
    print(f"Logit output shape: {tuple(logits.shape)}")
    print(f"Model parameters: {count_parameters(model):,}")


def benchmark_data_loading(config: TrainingConfig, num_batches: int = 64) -> None:
    validate_config(config)
    if num_batches <= 0:
        raise ValueError("num_batches must be positive.")

    set_reproducible_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    construct_start = time.perf_counter()
    train_dataset, val_dataset = create_train_val_datasets(
        raw_dir=config.raw_dir,
        val_fraction=config.val_fraction,
        background_slice_ratio=config.background_slice_ratio,
        seed=config.seed,
        case_cache_size=config.case_cache_size,
    )
    construct_seconds = time.perf_counter() - construct_start

    train_loader = build_data_loader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        seed=config.seed,
        device=device,
        case_wise_batches=config.case_wise_batches,
    )

    fetch_start = time.perf_counter()
    fetched_batches = 0
    fetched_samples = 0
    first_image_shape: tuple[int, ...] | None = None
    first_mask_shape: tuple[int, ...] | None = None

    for image, mask in train_loader:
        fetched_batches += 1
        fetched_samples += int(image.shape[0])
        if first_image_shape is None:
            first_image_shape = tuple(image.shape)
            first_mask_shape = tuple(mask.shape)
        if fetched_batches >= num_batches:
            break

    fetch_seconds = time.perf_counter() - fetch_start
    average_batch_seconds = fetch_seconds / max(fetched_batches, 1)

    print("Data-loading benchmark:")
    print(f"  device={device}")
    print(f"  train_samples={len(train_dataset)}")
    print(f"  val_samples={len(val_dataset)}")
    print(f"  batch_size={config.batch_size}")
    print(f"  requested_batches={num_batches}")
    print(f"  fetched_batches={fetched_batches}")
    print(f"  fetched_samples={fetched_samples}")
    print(f"  num_workers={config.num_workers}")
    print(f"  pin_memory={config.pin_memory and device.type == 'cuda'}")
    print(f"  persistent_workers={config.num_workers > 0}")
    print(f"  case_cache_size={config.case_cache_size}")
    print(f"  case_wise_batches={config.case_wise_batches}")
    print(f"  construct_seconds={construct_seconds:.3f}")
    print(f"  fetch_seconds={fetch_seconds:.3f}")
    print(f"  avg_batch_seconds={average_batch_seconds:.4f}")
    print(f"  first_image_batch_shape={first_image_shape}")
    print(f"  first_mask_batch_shape={first_mask_shape}")


def print_training_summary(history_path: Path) -> None:
    if not history_path.exists():
        print(f"No history file found at {history_path}")
        return

    with history_path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print(f"History file is empty: {history_path}")
        return

    best_row = max(rows, key=lambda row: float(row["mean_dice"]))
    final_row = rows[-1]
    print("Training summary:")
    print(
        f"  best_epoch={int(float(best_row['epoch']))} "
        f"best_mean_dice={float(best_row['mean_dice']):.4f} "
        f"best_mean_iou={float(best_row['mean_iou']):.4f}"
    )
    print(
        f"  final_epoch={int(float(final_row['epoch']))} "
        f"train_loss={float(final_row['training_loss']):.4f} "
        f"val_loss={float(final_row['validation_loss']):.4f} "
        f"mean_dice={float(final_row['mean_dice']):.4f} "
        f"mean_iou={float(final_row['mean_iou']):.4f}"
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    metrics: dict[str, float],
    parameter_count: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": serialize_config(config, None, parameter_count),
        "metrics": metrics,
    }
    torch.save(checkpoint, path)


def save_history_csv(path: Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_json(path: Path, data: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def build_experiment_config(
    config: TrainingConfig,
    device: torch.device,
    parameter_count: int,
    train_dataset: BraTSGLISliceDataset,
    val_dataset: BraTSGLISliceDataset,
) -> dict[str, object]:
    data = serialize_config(config, device, parameter_count)
    data.update(
        {
            "input_channels": 4,
            "output_classes": config.num_classes,
            "optimizer": "AdamW",
            "loss": "Dice + Cross Entropy",
            "deep_supervision": False if config.model_name == "unetpp" else None,
            "dataset_split": {
                "train_cases": train_dataset.case_count,
                "validation_cases": val_dataset.case_count,
                "train_slices": len(train_dataset),
                "validation_slices": len(val_dataset),
                "split_level": "case",
                "val_fraction": config.val_fraction,
                "seed": config.seed,
            },
        }
    )
    return data


def serialize_config(
    config: TrainingConfig,
    device: torch.device | None,
    parameter_count: int,
) -> dict[str, object]:
    data = asdict(config)
    data["raw_dir"] = str(config.raw_dir)
    data["results_dir"] = str(config.results_dir)
    data["parameter_count"] = parameter_count
    if device is not None:
        data["device"] = str(device)
    return data


def validate_config(config: TrainingConfig) -> None:
    if config.model_name not in MODEL_CHOICES:
        raise ValueError(f"model_name must be one of {MODEL_CHOICES}.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    if config.case_cache_size < 0:
        raise ValueError("case_cache_size must be non-negative.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if config.dice_weight < 0 or config.ce_weight < 0:
        raise ValueError("loss weights must be non-negative.")
    if config.dice_weight == 0 and config.ce_weight == 0:
        raise ValueError("at least one loss weight must be positive.")
    if config.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if config.max_grad_norm is not None and config.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when set.")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_run_header(
    model_name: str,
    device: torch.device,
    train_samples: int,
    val_samples: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    parameter_count: int,
) -> None:
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {val_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Model parameters: {parameter_count:,}")


def is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "cuda" in message and "out of memory" in message


def clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def create_model(config: TrainingConfig) -> nn.Module:
    if config.model_name == "unet":
        return UNet2D(
            in_channels=4,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
        )

    if config.model_name == "unetpp":
        return UNetPlusPlus2D(
            in_channels=4,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            deep_supervision=False,
        )

    if config.model_name == "attention_unet":
        return AttentionUNet2D(
            in_channels=4,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
        )

    if config.model_name == "resunet":
        return ResUNet2D(
            in_channels=4,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
        )

    if config.model_name == "unet_resnet34":
        return UNetResNet34(
            in_channels=4,
            num_classes=config.num_classes,
            base_channels=config.base_channels,
            pretrained=config.pretrained,
        )

    raise ValueError(f"Unsupported model: {config.model_name}")


def resolve_results_dir(model_name: str, requested_results_dir: Path | None) -> Path:
    if requested_results_dir is not None:
        return requested_results_dir
    if model_name == "unetpp":
        return DEFAULT_UNETPP_RESULTS_DIR
    if model_name == "resunet":
        return DEFAULT_RESUNET_RESULTS_DIR
    if model_name == "unet_resnet34":
        return DEFAULT_UNET_RESNET34_RESULTS_DIR
    return DEFAULT_RESULTS_DIR


def resolve_history_path(model_name: str, requested_history_path: Path | None) -> Path:
    if requested_history_path is not None:
        return requested_history_path
    return resolve_results_dir(model_name, None) / "history.csv"


if __name__ == "__main__":
    main()
