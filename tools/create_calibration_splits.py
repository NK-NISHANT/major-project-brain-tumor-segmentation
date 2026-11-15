"""Create deterministic calibration and evaluation splits from validation partition.

Gate 6A: Reliability-Aware AI Agent Integration & Uncertainty Calibration
Splits 155 validation cases into:
  - 50 calibration cases (Val_cal)
  - 105 evaluation cases (Val_eval)
Deterministically using seed 42 sorted by case ID.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any


def hash_case_list(case_ids: list[str]) -> str:
    """Compute deterministic SHA-256 hash of a list of case IDs."""
    data_str = "\n".join(sorted(case_ids)).encode("utf-8")
    return hashlib.sha256(data_str).hexdigest()


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    splits_dir = project_root / "results" / "dataset_splits"
    out_dir = project_root / "results" / "gate6a_reliability_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_manifest_path = splits_dir / "validation_manifest.json"
    test_manifest_path = splits_dir / "test_manifest.json"
    train_manifest_path = splits_dir / "development_train_manifest.json"

    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(train_manifest_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    val_cases = val_data["cases"]
    test_cases = test_data["cases"]
    train_cases = train_data["cases"]

    val_case_ids = sorted([c["case_id"] for c in val_cases])
    test_case_ids = set([c["case_id"] for c in test_cases])
    train_case_ids = set([c["case_id"] for c in train_cases])

    assert len(val_case_ids) == 155, f"Expected 155 validation cases, got {len(val_case_ids)}"
    assert len(test_case_ids) == 155, f"Expected 155 test cases, got {len(test_case_ids)}"
    assert len(train_case_ids) == 721, f"Expected 721 train cases, got {len(train_case_ids)}"

    # Deterministic shuffle with seed 42
    rng = random.Random(42)
    shuffled_ids = list(val_case_ids)
    rng.shuffle(shuffled_ids)

    cal_case_ids = sorted(shuffled_ids[:50])
    eval_case_ids = sorted(shuffled_ids[50:])

    assert len(cal_case_ids) == 50, f"Expected 50 cal cases, got {len(cal_case_ids)}"
    assert len(eval_case_ids) == 105, f"Expected 105 eval cases, got {len(eval_case_ids)}"

    # Set validation
    cal_set = set(cal_case_ids)
    eval_set = set(eval_case_ids)
    val_set = set(val_case_ids)

    assert len(cal_set.intersection(eval_set)) == 0, "Non-zero overlap between calibration and evaluation sets!"
    assert cal_set.union(eval_set) == val_set, "Calibration + Evaluation union does not match validation set!"
    assert len(cal_set.intersection(test_case_ids)) == 0, "Quarantined test case leaked into calibration set!"
    assert len(eval_set.intersection(test_case_ids)) == 0, "Quarantined test case leaked into evaluation set!"
    assert len(cal_set.intersection(train_case_ids)) == 0, "Train case leaked into calibration set!"
    assert len(eval_set.intersection(train_case_ids)) == 0, "Train case leaked into evaluation set!"

    # Subject-level analysis for transparency
    case_to_subj = {c["case_id"]: c["subject_id"] for c in val_cases}
    cal_subjs = {case_to_subj[cid] for cid in cal_case_ids}
    eval_subjs = {case_to_subj[cid] for cid in eval_case_ids}
    shared_subjs = sorted(list(cal_subjs.intersection(eval_subjs)))

    cal_hash = hash_case_list(cal_case_ids)
    eval_hash = hash_case_list(eval_case_ids)
    val_hash = hash_case_list(val_case_ids)

    out_record: dict[str, Any] = {
        "protocol": "Gate 6A Uncertainty Calibration and Evaluation Split",
        "random_seed": 42,
        "source_manifest": "results/dataset_splits/validation_manifest.json",
        "total_validation_cases": len(val_case_ids),
        "validation_manifest_hash_sha256": val_hash,
        "calibration_set": {
            "name": "Val_cal",
            "count": len(cal_case_ids),
            "sha256": cal_hash,
            "case_ids": cal_case_ids,
        },
        "evaluation_set": {
            "name": "Val_eval",
            "count": len(eval_case_ids),
            "sha256": eval_hash,
            "case_ids": eval_case_ids,
            "methodological_disclosure": (
                "Val_eval is an internal evaluation subset of the validation cohort (105 cases), "
                "strictly distinct from Val_cal (50 cases). It is NOT an independent final test set. "
                "The 155 test cases in results/dataset_splits/test_manifest.json remain 100% quarantined."
            ),
        },
        "subject_level_accounting": {
            "unique_subjects_in_validation": len(set(case_to_subj.values())),
            "unique_subjects_in_calibration": len(cal_subjs),
            "unique_subjects_in_evaluation": len(eval_subjs),
            "longitudinal_subjects_spanning_both": shared_subjs,
            "spanning_subject_count": len(shared_subjs),
            "notes": (
                "Per protocol specification, splitting was executed deterministically by case ID with seed 42. "
                f"{len(shared_subjs)} longitudinal subjects have separate temporal scans in both sets."
            ),
        },
    }

    out_file = out_dir / "calibration_splits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_record, f, indent=2)

    print(f"Created calibration split successfully:")
    print(f"  Source validation cases: {len(val_case_ids)}")
    print(f"  Val_cal:  {len(cal_case_ids)} cases (SHA-256: {cal_hash[:16]}...)")
    print(f"  Val_eval: {len(eval_case_ids)} cases (SHA-256: {eval_hash[:16]}...)")
    print(f"  Overlap:  0 cases")
    print(f"  Saved to: {out_file}")


if __name__ == "__main__":
    main()
