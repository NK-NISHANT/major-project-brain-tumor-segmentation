from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


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


@dataclass(frozen=True)
class CachedCaseVolume:
    image: np.ndarray
    mask: np.ndarray


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
    case_cache_size: int = 1,
) -> tuple["BraTSGLISliceDataset", "BraTSGLISliceDataset"]:
    """Create train/validation datasets using a deterministic case-level split."""
    cases = discover_valid_cases(raw_dir)
    train_cases, val_cases = split_cases(cases, val_fraction=val_fraction, seed=seed)

    train_dataset = BraTSGLISliceDataset(
        train_cases,
        background_slice_ratio=background_slice_ratio,
        seed=seed,
        case_cache_size=case_cache_size,
    )
    val_dataset = BraTSGLISliceDataset(
        val_cases,
        background_slice_ratio=background_slice_ratio,
        seed=seed + 1,
        case_cache_size=case_cache_size,
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
        case_cache_size: int = 1,
    ) -> None:
        if background_slice_ratio < 0.0:
            raise ValueError("background_slice_ratio must be non-negative.")
        if case_cache_size < 0:
            raise ValueError("case_cache_size must be non-negative.")

        self.cases = list(cases)
        self.case_by_id = {case.case_id: case for case in self.cases}
        self.background_slice_ratio = background_slice_ratio
        self.seed = seed
        self.case_cache_size = case_cache_size
        self.slice_index = build_slice_index(
            self.cases,
            background_slice_ratio=background_slice_ratio,
            seed=seed,
        )
        self.case_to_slice_indices = self._build_case_to_slice_indices()
        self._case_volume_cache: OrderedDict[str, CachedCaseVolume] = OrderedDict()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Windows DataLoader workers receive a pickled dataset. The cache is
        # per-process and can be large, so workers should start with an empty
        # bounded cache rather than copying volume arrays from the parent.
        state["_case_volume_cache"] = OrderedDict()
        return state

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.slice_index[index]
        case = self.case_by_id[record.case_id]
        case_volume = self._get_case_volume(case)

        image_slice = np.ascontiguousarray(case_volume.image[:, :, :, record.slice_index])
        mask_slice = case_volume.mask[:, :, record.slice_index]

        # Keep labels as plain class IDs 0, 1, 2, 3. We do not assign semantic
        # names or remap labels here; CrossEntropyLoss only needs integer IDs.
        mask_slice = np.ascontiguousarray(mask_slice.astype(np.int64, copy=False))

        image_tensor = torch.from_numpy(image_slice)
        mask_tensor = torch.from_numpy(mask_slice)
        return image_tensor, mask_tensor

    @property
    def case_count(self) -> int:
        return len(self.cases)

    def get_case_slice_indices(self) -> dict[str, list[int]]:
        return {case_id: indices.copy() for case_id, indices in self.case_to_slice_indices.items()}

    def _build_case_to_slice_indices(self) -> dict[str, list[int]]:
        case_to_indices = {case.case_id: [] for case in self.cases}
        for dataset_index, record in enumerate(self.slice_index):
            case_to_indices[record.case_id].append(dataset_index)
        return case_to_indices

    def _get_case_volume(self, case: CaseFiles) -> CachedCaseVolume:
        if self.case_cache_size == 0:
            return _load_and_normalize_case(case)

        cached_volume = self._case_volume_cache.get(case.case_id)
        if cached_volume is not None:
            self._case_volume_cache.move_to_end(case.case_id)
            return cached_volume

        cached_volume = _load_and_normalize_case(case)
        self._case_volume_cache[case.case_id] = cached_volume

        # The cache is deliberately bounded. One normalized case is roughly
        # 145 MB on CPU, so caching all 197 cases would be tens of GB. We keep a
        # small per-process cache and never use GPU memory for dataset storage.
        while len(self._case_volume_cache) > self.case_cache_size:
            self._case_volume_cache.popitem(last=False)

        return cached_volume


class CaseSliceBatchSampler(Sampler[list[int]]):
    """Yield batches grouped by case so the CPU case cache can be reused."""

    def __init__(
        self,
        dataset: BraTSGLISliceDataset,
        batch_size: int,
        shuffle_cases: bool = True,
        shuffle_slices: bool = True,
        seed: int = 42,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.case_to_slice_indices = dataset.get_case_slice_indices()
        self.case_ids = [
            case.case_id
            for case in dataset.cases
            if self.case_to_slice_indices.get(case.case_id)
        ]
        self.batch_size = batch_size
        self.shuffle_cases = shuffle_cases
        self.shuffle_slices = shuffle_slices
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        # Cases are shuffled, but train/validation membership is unchanged.
        # Keeping nearby batches from the same case avoids repeatedly opening
        # gzip-compressed NIfTI files for individual slices.
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        case_ids = self.case_ids.copy()
        if self.shuffle_cases:
            rng.shuffle(case_ids)

        for case_id in case_ids:
            indices = self.case_to_slice_indices[case_id].copy()
            if self.shuffle_slices:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self) -> int:
        batch_count = 0
        for indices in self.case_to_slice_indices.values():
            if self.drop_last:
                batch_count += len(indices) // self.batch_size
            else:
                batch_count += (len(indices) + self.batch_size - 1) // self.batch_size
        return batch_count


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
    return _compute_nonzero_mean_std_from_array(volume)


def _compute_nonzero_mean_std_from_array(volume: np.ndarray) -> tuple[float, float]:
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


def _load_and_normalize_case(case: CaseFiles) -> CachedCaseVolume:
    image_volume = np.zeros((len(MODALITIES), *case.shape), dtype=np.float32)

    for channel, modality in enumerate(MODALITIES):
        # Repeated random reads from .nii.gz are slow because gzip decompression
        # is not designed for thousands of tiny slice requests. Load each
        # modality once per cached case, compute normalization once, and then
        # serve selected slices from the normalized CPU array.
        volume = np.asarray(
            nib.load(str(case.modality_paths[modality])).dataobj,
            dtype=np.float32,
        )
        mean, std = _compute_nonzero_mean_std_from_array(volume)
        nonzero_mask = volume != 0
        normalized_channel = image_volume[channel]
        normalized_channel[nonzero_mask] = (volume[nonzero_mask] - mean) / std

    segmentation = np.asarray(nib.load(str(case.segmentation_path)).dataobj)
    mask_volume = np.rint(segmentation).astype(np.uint8, copy=False)
    return CachedCaseVolume(image=image_volume, mask=mask_volume)


def _first_foreground_index(dataset: BraTSGLISliceDataset) -> int:
    for index, record in enumerate(dataset.slice_index):
        if record.has_foreground:
            return index
    return 0


__all__ = [
    "BraTSGLISliceDataset",
    "CachedCaseVolume",
    "CaseSliceBatchSampler",
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
