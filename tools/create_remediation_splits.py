"""Create deterministic, subject-grouped calibration and evaluation splits for Gate 6A Remediation.

Workspace: D:\\MajorProject
Output: results/gate6a_remediation_v1/calibration_splits.json

Requirements:
- Input: results/dataset_splits/validation_manifest.json (155 cases, 143 subjects)
- Zero subject overlap between calibration and evaluation sets.
- All temporal scans of multi-scan subjects are strictly assigned together.
- Seed 42 deterministic assignment.
- Verified 0 test-case leakage against results/dataset_splits/test_manifest.json.
- Cryptographic SHA-256 bindings for manifest files and partition contents.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


def hash_string_list(items: list[str]) -> str:
    data_str = "\n".join(sorted(items)).encode("utf-8")
    return hashlib.sha256(data_str).hexdigest()


def main() -> None:
    splits_dir = PROJECT_ROOT / "results" / "dataset_splits"
    out_dir = PROJECT_ROOT / "results" / "gate6a_remediation_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_manifest_path = splits_dir / "validation_manifest.json"
    test_manifest_path = splits_dir / "test_manifest.json"
    train_manifest_path = splits_dir / "development_train_manifest.json"

    val_manifest_hash = get_file_sha256(val_manifest_path)
    test_manifest_hash = get_file_sha256(test_manifest_path)
    train_manifest_hash = get_file_sha256(train_manifest_path)

    with open(val_manifest_path, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    with open(test_manifest_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    with open(train_manifest_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    val_cases = val_data["cases"]
    test_case_ids = set(c["case_id"] for c in test_data["cases"])
    train_case_ids = set(c["case_id"] for c in train_data["cases"])

    # Group validation cases by subject_id
    subj_to_cases: dict[str, list[str]] = defaultdict(list)
    case_to_subj: dict[str, str] = {}
    for c in val_cases:
        cid = c["case_id"]
        sid = c["subject_id"]
        subj_to_cases[sid].append(cid)
        case_to_subj[cid] = sid

    unique_subjects = sorted(list(subj_to_cases.keys()))
    assert len(val_cases) == 155, f"Expected 155 validation cases, got {len(val_cases)}"
    assert len(unique_subjects) == 143, f"Expected 143 unique subjects, got {len(unique_subjects)}"

    # Deterministic subject shuffle with seed 42
    rng = random.Random(42)
    shuffled_subjects = list(unique_subjects)
    rng.shuffle(shuffled_subjects)

    cal_cases: list[str] = []
    eval_cases: list[str] = []
    cal_subs: list[str] = []
    eval_subs: list[str] = []

    target_cal_cases = 50

    for sid in shuffled_subjects:
        cids = subj_to_cases[sid]
        if len(cal_cases) + len(cids) <= target_cal_cases:
            cal_cases.extend(cids)
            cal_subs.append(sid)
        else:
            eval_cases.extend(cids)
            eval_subs.append(sid)

    cal_cases = sorted(cal_cases)
    eval_cases = sorted(eval_cases)
    cal_subs = sorted(cal_subs)
    eval_subs = sorted(eval_subs)

    # Rigorous verification assertions
    assert len(cal_cases) == 50, f"Expected 50 cal cases, got {len(cal_cases)}"
    assert len(eval_cases) == 105, f"Expected 105 eval cases, got {len(eval_cases)}"
    assert len(cal_subs) == 47, f"Expected 47 cal subjects, got {len(cal_subs)}"
    assert len(eval_subs) == 96, f"Expected 96 eval subjects, got {len(eval_subs)}"

    set_cal_subs = set(cal_subs)
    set_eval_subs = set(eval_subs)
    overlap_subs = set_cal_subs.intersection(set_eval_subs)
    assert len(overlap_subs) == 0, f"Subject overlap detected: {overlap_subs}"

    set_cal_cases = set(cal_cases)
    set_eval_cases = set(eval_cases)
    overlap_cases = set_cal_cases.intersection(set_eval_cases)
    assert len(overlap_cases) == 0, f"Case overlap detected: {overlap_cases}"

    union_cases = set_cal_cases.union(set_eval_cases)
    all_val_case_ids = set(c["case_id"] for c in val_cases)
    assert union_cases == all_val_case_ids, "Union of cal + eval does not match validation manifest cases!"

    # Strict quarantine assertions
    assert len(set_cal_cases.intersection(test_case_ids)) == 0, "Test cases leaked into calibration set!"
    assert len(set_eval_cases.intersection(test_case_ids)) == 0, "Test cases leaked into evaluation set!"
    assert len(set_cal_cases.intersection(train_case_ids)) == 0, "Train cases leaked into calibration set!"
    assert len(set_eval_cases.intersection(train_case_ids)) == 0, "Train cases leaked into evaluation set!"

    cal_case_hash = hash_string_list(cal_cases)
    eval_case_hash = hash_string_list(eval_cases)

    # Document multi-scan subject allocations
    multi_scan_subjects_cal = [s for s in cal_subs if len(subj_to_cases[s]) > 1]
    multi_scan_subjects_eval = [s for s in eval_subs if len(subj_to_cases[s]) > 1]

    out_payload: dict[str, Any] = {
        "metadata": {
            "protocol": "Gate 6A Remediation: Subject-Grouped Calibration and Evaluation Split",
            "version": "1.0.0",
            "date": "2026-10-29",
            "random_seed": 42,
            "source_validation_manifest": str(val_manifest_path.relative_to(PROJECT_ROOT)),
            "source_validation_manifest_file_sha256": val_manifest_hash,
            "total_validation_cases": len(val_cases),
            "total_validation_subjects": len(unique_subjects),
            "methodology_notes": (
                "Validation cohort is partitioned deterministically at the SUBJECT level using seed 42. "
                "All longitudinal timepoints for any given subject are strictly grouped within either calibration or evaluation, "
                "ensuring zero cross-partition subject leakage."
            ),
        },
        "calibration_set": {
            "name": "Val_cal_remediation",
            "case_count": len(cal_cases),
            "subject_count": len(cal_subs),
            "sha256_case_list": cal_case_hash,
            "multi_scan_subject_count": len(multi_scan_subjects_cal),
            "multi_scan_subjects": multi_scan_subjects_cal,
            "subject_ids": cal_subs,
            "case_ids": cal_cases,
        },
        "evaluation_set": {
            "name": "Val_eval_remediation",
            "case_count": len(eval_cases),
            "subject_count": len(eval_subs),
            "sha256_case_list": eval_case_hash,
            "multi_scan_subject_count": len(multi_scan_subjects_eval),
            "multi_scan_subjects": multi_scan_subjects_eval,
            "subject_ids": eval_subs,
            "case_ids": eval_cases,
            "disclosure": (
                "Val_eval_remediation (105 cases, 96 subjects) is an internal evaluation subset of the 155-case validation partition, "
                "strictly distinct from Val_cal_remediation (50 cases, 47 subjects). It is NOT an independent final test set. "
                "The 155 test cases in results/dataset_splits/test_manifest.json remain 100% quarantined and unaccessed."
            ),
        },
        "subject_level_accounting": {
            "unique_subjects_in_validation": len(unique_subjects),
            "unique_subjects_in_calibration": len(cal_subs),
            "unique_subjects_in_evaluation": len(eval_subs),
            "subject_overlap_count": 0,
            "longitudinal_subjects_spanning_both": [],
            "disjointness_verified": True,
        },
        "quarantine_audit": {
            "quarantined_test_manifest": str(test_manifest_path.relative_to(PROJECT_ROOT)),
            "quarantined_test_manifest_file_sha256": test_manifest_hash,
            "total_test_cases_quarantined": len(test_case_ids),
            "test_cases_in_calibration": 0,
            "test_cases_in_evaluation": 0,
            "quarantine_preserved": True,
        },
    }

    out_file = out_dir / "calibration_splits.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print("=== Gate 6A Remediation Subject-Grouped Split Generated Successfully ===")
    print(f"Output File: {out_file}")
    print(f"Validation Manifest SHA-256: {val_manifest_hash}")
    print(f"Test Manifest SHA-256:       {test_manifest_hash}")
    print(f"Calibration Partition:       {len(cal_cases)} cases across {len(cal_subs)} subjects")
    print(f"  Multi-scan subjects ({len(multi_scan_subjects_cal)}): {multi_scan_subjects_cal}")
    print(f"Evaluation Partition:        {len(eval_cases)} cases across {len(eval_subs)} subjects")
    print(f"  Multi-scan subjects ({len(multi_scan_subjects_eval)}): {multi_scan_subjects_eval}")
    print(f"Cross-Partition Overlap:     0 subjects (100% DISJOINT)")
    print(f"Test Set Leakage:            0 cases (100% QUARANTINED)")


if __name__ == "__main__":
    main()
