from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm

from src.data.dataset import MODALITIES, discover_valid_cases


OUTPUT_DIR = Path("results/input_condition")
OUTPUT_FILE = OUTPUT_DIR / "case_input_features.csv"


def compute_modality_features(path: Path) -> dict[str, float]:
    """Compute simple input-quality/intensity features for one MRI modality."""

    volume = np.asarray(nib.load(path).dataobj, dtype=np.float32)

    finite_mask = np.isfinite(volume)
    nonzero_mask = finite_mask & (volume != 0)

    if not np.any(nonzero_mask):
        return {
            "nonzero_fraction": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    values = volume[nonzero_mask]

    return {
        "nonzero_fraction": float(nonzero_mask.mean()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main() -> None:
    raw_dir = Path("data/raw")

    cases = discover_valid_cases(raw_dir)

    if not cases:
        raise RuntimeError("No valid cases found.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    print(f"Found {len(cases)} complete cases.")
    print(f"Modalities: {', '.join(MODALITIES)}")

    for case in tqdm(cases, desc="Extracting input features"):
        row = {
            "case_id": case.case_id,
        }

        for modality in MODALITIES:
            path = case.modality_paths[modality]

            features = compute_modality_features(path)

            for feature_name, value in features.items():
                row[f"{modality}_{feature_name}"] = value

        rows.append(row)

    fieldnames = list(rows[0].keys())

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Cases: {len(rows)}")
    print(f"Features per modality: 5")
    print(f"Total numeric modality features: {len(fieldnames) - 1}")


if __name__ == "__main__":
    main()