from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Keep this order fixed so every sample has a stable channel layout.
MODALITIES: tuple[str, ...] = ("t1n", "t1c", "t2f", "t2w")
SEGMENTATION_SUFFIX = "seg"


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    case_dir: Path
    modality_paths: Mapping[str, Path]
    segmentation_path: Path
    shape: tuple[int, int, int]


@dataclass(frozen=True)
class SliceRecord:
    case_id: str
    slice_index: int
    has_foreground: bool


def discover_valid_cases(raw_dir: str | Path = DEFAULT_RAW_DATA_DIR) -> list[CaseFiles]:
    """Find case directories that contain all required MRI modalities and a mask."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {raw_path}")

    valid_cases: list[CaseFiles] = []
    for case_dir in sorted(path for path in raw_path.iterdir() if path.is_dir()):
        modality_paths: dict[str, Path] = {}
        missing_file = False

        for modality in MODALITIES:
            path = _find_nifti_file(case_dir, modality)
            if path is None:
                missing_file = True
                break
            modality_paths[modality] = path

        segmentation_path = _find_nifti_file(case_dir, SEGMENTATION_SUFFIX)
        if missing_file or segmentation_path is None:
            continue

        shape = tuple(int(dim) for dim in nib.load(str(segmentation_path)).shape)
        if len(shape) != 3:
            continue

        valid_cases.append(
            CaseFiles(
                case_id=case_dir.name,
                case_dir=case_dir,
                modality_paths=modality_paths,
                segmentation_path=segmentation_path,
                shape=shape,
            )
        )

    return valid_cases


def split_cases(
    cases: Sequence[CaseFiles],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[CaseFiles], list[CaseFiles]]:
    """Split cases deterministically so slices from one case never cross splits."""
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in the range [0.0, 1.0).")

    cases = list(cases)
    if len(cases) < 2 or val_fraction == 0.0:
        return cases, []

    # Case-level splitting is critical for segmentation research: adjacent slices
    # from the same patient are highly correlated, so slice-level splitting would
    # leak patient information into validation and inflate performance.
    shuffled_indices = list(range(len(cases)))
    random.Random(seed).shuffle(shuffled_indices)

    val_count = max(1, int(round(len(cases) * val_fraction)))
    val_count = min(val_count, len(cases) - 1)
    val_indices = set(shuffled_indices[:val_count])

    train_cases = [case for index, case in enumerate(cases) if index not in val_indices]
    val_cases = [case for index, case in enumerate(cases) if index in val_indices]
    return train_cases, val_cases


def create_train_val_datasets(
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    val_fraction: float = 0.2,
    background_slice_ratio: float = 0.25,
    seed: int = 42,
) -> tuple["BraTSGLISliceDataset", "BraTSGLISliceDataset"]:
    """Create train/validation datasets using a deterministic case-level split."""
    cases = discover_valid_cases(raw_dir)
    train_cases, val_cases = split_cases(cases, val_fraction=val_fraction, seed=seed)

    train_dataset = BraTSGLISliceDataset(
        train_cases,
        background_slice_ratio=background_slice_ratio,
        seed=seed,
    )
    val_dataset = BraTSGLISliceDataset(
        val_cases,
        background_slice_ratio=background_slice_ratio,
        seed=seed + 1,
    )
    return train_dataset, val_dataset


class BraTSGLISliceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Axial 2D slice dataset for BraTS-GLI MRI segmentation.

    The dataset intentionally performs no augmentation. It returns four MRI
    channels and one mask slice suitable for PyTorch CrossEntropyLoss.
    """

    def __init__(
        self,
        cases: Sequence[CaseFiles],
        background_slice_ratio: float = 0.25,
        seed: int = 42,
    ) -> None:
        if background_slice_ratio < 0.0:
            raise ValueError("background_slice_ratio must be non-negative.")

        self.cases = list(cases)
        self.case_by_id = {case.case_id: case for case in self.cases}
        self.background_slice_ratio = background_slice_ratio
        self.seed = seed
        self.slice_index = build_slice_index(
            self.cases,
            background_slice_ratio=background_slice_ratio,
            seed=seed,
        )
        self._normalization_stats: dict[str, dict[str, tuple[float, float]]] = {}

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.slice_index[index]
        case = self.case_by_id[record.case_id]
        stats = self._get_normalization_stats(case)

        image_channels: list[np.ndarray] = []
        for modality in MODALITIES:
            image = nib.load(str(case.modality_paths[modality]))
            image_slice = np.asarray(
                image.dataobj[:, :, record.slice_index],
                dtype=np.float32,
            )
            mean, std = stats[modality]
            image_channels.append(_normalize_mri_slice(image_slice, mean, std))

        segmentation = nib.load(str(case.segmentation_path))
        mask = np.asarray(segmentation.dataobj[:, :, record.slice_index])

        # Keep labels as plain class IDs 0, 1, 2, 3. We do not assign semantic
        # names or remap labels here; CrossEntropyLoss only needs integer IDs.
        mask = np.rint(mask).astype(np.int64, copy=False)

        image_array = np.stack(image_channels, axis=0).astype(np.float32, copy=False)
        image_tensor = torch.from_numpy(image_array)
        mask_tensor = torch.from_numpy(mask)
        return image_tensor, mask_tensor

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def _get_normalization_stats(self, case: CaseFiles) -> dict[str, tuple[float, float]]:
        if case.case_id not in self._normalization_stats:
            self._normalization_stats[case.case_id] = {
                modality: _compute_nonzero_mean_std(case.modality_paths[modality])
                for modality in MODALITIES
            }
        return self._normalization_stats[case.case_id]


def build_slice_index(
    cases: Sequence[CaseFiles],
    background_slice_ratio: float = 0.25,
    seed: int = 42,
) -> list[SliceRecord]:
    """Build a lightweight list of selected axial slices without storing volumes."""
    records: list[SliceRecord] = []
    rng = random.Random(seed)

    for case in cases:
        segmentation = nib.load(str(case.segmentation_path))
        segmentation_volume = np.asarray(segmentation.dataobj)
        foreground_by_slice = np.any(segmentation_volume > 0, axis=(0, 1))

        foreground_slices = [
            int(index)
            for index, has_foreground in enumerate(foreground_by_slice)
            if bool(has_foreground)
        ]
        background_slices = [
            int(index)
            for index, has_foreground in enumerate(foreground_by_slice)
            if not bool(has_foreground)
        ]

        records.extend(
            SliceRecord(case.case_id, slice_index, True)
            for slice_index in foreground_slices
        )

        # Most axial volumes contain many empty slices. Keeping every background
        # slice would overrepresent easy negatives, so we retain a configurable
        # subset while still exposing the model to true background examples.
        max_background = int(round(len(foreground_slices) * background_slice_ratio))
        if background_slice_ratio > 0.0 and foreground_slices:
            max_background = max(1, max_background)
        max_background = min(max_background, len(background_slices))

        selected_background_slices = (
            sorted(rng.sample(background_slices, max_background))
            if max_background > 0
            else []
        )
        records.extend(
            SliceRecord(case.case_id, slice_index, False)
            for slice_index in selected_background_slices
        )

    return records


def inspect_datasets(
    raw_dir: str | Path = DEFAULT_RAW_DATA_DIR,
    val_fraction: float = 0.2,
    background_slice_ratio: float = 0.25,
    seed: int = 42,
) -> tuple[BraTSGLISliceDataset, BraTSGLISliceDataset]:
    """Print a compact sanity check for the discovered dataset and split."""
    train_dataset, val_dataset = create_train_val_datasets(
        raw_dir=raw_dir,
        val_fraction=val_fraction,
        background_slice_ratio=background_slice_ratio,
        seed=seed,
    )

    sample_dataset = train_dataset if len(train_dataset) > 0 else val_dataset
    if len(sample_dataset) == 0:
        raise RuntimeError("No slices were selected from the discovered cases.")

    sample_index = _first_foreground_index(sample_dataset)
    image, mask = sample_dataset[sample_index]
    unique_labels = torch.unique(mask).tolist()
    foreground_pixel_percent = float((mask > 0).float().mean().item() * 100.0)

    print(f"Valid cases: {train_dataset.case_count + val_dataset.case_count}")
    print(f"Training cases: {train_dataset.case_count}")
    print(f"Validation cases: {val_dataset.case_count}")
    print(f"Selected training slices: {len(train_dataset)}")
    print(f"Selected validation slices: {len(val_dataset)}")
    print(f"Image tensor shape: {tuple(image.shape)}")
    print(f"Mask tensor shape: {tuple(mask.shape)}")
    print(f"Unique mask labels: {unique_labels}")
    print(f"Foreground pixels: {foreground_pixel_percent:.4f}%")

    return train_dataset, val_dataset


def _find_nifti_file(case_dir: Path, suffix: str) -> Path | None:
    matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    return matches[0] if matches else None


def _compute_nonzero_mean_std(path: Path) -> tuple[float, float]:
    image = nib.load(str(path))
    volume = np.asarray(image.dataobj, dtype=np.float32)
    nonzero_voxels = volume[volume != 0]

    # MRI volumes are padded with zeros outside the anatomy. Normalizing with
    # those zeros would bias statistics toward the background rather than the
    # observed signal, so mean/std are estimated from non-zero voxels only.
    if nonzero_voxels.size == 0:
        return 0.0, 1.0

    mean = float(nonzero_voxels.mean())
    std = float(nonzero_voxels.std())
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return mean, std


def _normalize_mri_slice(image_slice: np.ndarray, mean: float, std: float) -> np.ndarray:
    normalized = np.zeros_like(image_slice, dtype=np.float32)
    nonzero_mask = image_slice != 0
    normalized[nonzero_mask] = (image_slice[nonzero_mask] - mean) / std
    return normalized


def _first_foreground_index(dataset: BraTSGLISliceDataset) -> int:
    for index, record in enumerate(dataset.slice_index):
        if record.has_foreground:
            return index
    return 0


__all__ = [
    "BraTSGLISliceDataset",
    "CaseFiles",
    "SliceRecord",
    "MODALITIES",
    "DEFAULT_RAW_DATA_DIR",
    "build_slice_index",
    "create_train_val_datasets",
    "discover_valid_cases",
    "inspect_datasets",
    "split_cases",
]
