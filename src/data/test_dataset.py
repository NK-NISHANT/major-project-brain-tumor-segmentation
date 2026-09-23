from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DEFAULT_RAW_DATA_DIR, inspect_datasets  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the BraTS-GLI slice dataset.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--background-slice-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=3)
    args = parser.parse_args()

    train_dataset, val_dataset = inspect_datasets(
        raw_dir=args.raw_dir,
        val_fraction=args.val_fraction,
        background_slice_ratio=args.background_slice_ratio,
        seed=args.seed,
    )

    for split_name, dataset in (("train", train_dataset), ("val", val_dataset)):
        print(f"\n{split_name} sample checks:")
        if len(dataset) == 0:
            print("  no selected slices")
            continue

        for sample_index in range(min(args.num_samples, len(dataset))):
            image, mask = dataset[sample_index]
            unique_labels = torch.unique(mask).tolist()
            foreground_percent = float((mask > 0).float().mean().item() * 100.0)
            print(
                f"  {split_name}[{sample_index}] "
                f"image_shape={tuple(image.shape)} "
                f"image_dtype={image.dtype} "
                f"mask_shape={tuple(mask.shape)} "
                f"mask_dtype={mask.dtype} "
                f"labels={unique_labels} "
                f"foreground_pixels={foreground_percent:.4f}%"
            )


if __name__ == "__main__":
    main()
