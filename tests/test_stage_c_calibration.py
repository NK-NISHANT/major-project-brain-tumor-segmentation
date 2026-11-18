"""Unit tests for Stage C2 Temperature Calibration Provenance & Configuration Loader.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Calibration Authority: Stage C Phase 2 (c2_calibration_v1)
Batch 1B: Stage C2 Calibration Provenance & Configuration Loader.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.agent.calibration import (
    DEFAULT_C2_CALIBRATION_JSON,
    DEFAULT_C2_RUN_MANIFEST,
    EXPECTED_C2_MANIFEST_SHA256,
    EXPECTED_C2_MANIFEST_SIZE,
    EXPECTED_C2_SPLITS_SHA256,
    EXPECTED_CALIBRATION_JSON_SHA256,
    EXPECTED_CALIBRATION_JSON_SIZE,
    STAGE_C2_REFERENCE_TEMPERATURES,
    StageCCalibrationConfig,
    load_stage_c_calibration,
)
from src.agent.models import (
    CheckpointProfile,
    ModelRegistry,
    STAGE_C_EXPECTED_HASHES,
    STAGE_C_MODELS,
    normalize_model_name,
)


def _get_base_valid_dicts() -> tuple[dict, dict]:
    """Load valid base dictionaries directly from the approved physical artifacts."""
    with open(DEFAULT_C2_CALIBRATION_JSON, "r", encoding="utf-8") as f:
        calib_dict = json.load(f)
    with open(DEFAULT_C2_RUN_MANIFEST, "r", encoding="utf-8") as f:
        manifest_dict = json.load(f)
    return copy.deepcopy(calib_dict), copy.deepcopy(manifest_dict)


def _write_temp_json(data: dict) -> Path:
    """Write dictionary to a temporary JSON file and return its Path."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(data, tmp, indent=2)
        return Path(tmp.name)


def _write_temp_bundle(calib_dict: dict, manifest_dict: dict | None = None) -> tuple[Path, Path]:
    """Write paired temporary calibration JSON and manifest with synchronized hash."""
    tmp_calib = _write_temp_json(calib_dict)
    with open(tmp_calib, "rb") as f:
        calib_hash = hashlib.sha256(f.read()).hexdigest()

    if manifest_dict is None:
        _, base_manifest = _get_base_valid_dicts()
        manifest_dict = base_manifest
    else:
        manifest_dict = copy.deepcopy(manifest_dict)

    manifest_dict["provenance_hashes"]["output_calibration_parameters_sha256"] = calib_hash
    tmp_manifest = _write_temp_json(manifest_dict)
    return tmp_calib, tmp_manifest


class TestStageCCalibration(unittest.TestCase):
    """Comprehensive test suite for Stage C2 calibration loader and provenance verification."""

    # -------------------------------------------------------------------------
    # 5.1 Successful Configuration Loading & Physical Artifact Verification
    # -------------------------------------------------------------------------

    def test_01_all_five_stage_c_temperatures_load_successfully(self) -> None:
        """Requirement 1: All five Stage C temperatures load successfully from verified C2 artifact."""
        cfg = load_stage_c_calibration()
        self.assertIsInstance(cfg, StageCCalibrationConfig)
        self.assertEqual(len(cfg.temperatures), 5)
        for model in STAGE_C_MODELS:
            temp = cfg.temperature_for(model)
            self.assertIsInstance(temp, float)
            self.assertGreater(temp, 0.0)
            self.assertFalse(math.isnan(temp))
            self.assertFalse(math.isinf(temp))

    def test_02_all_five_canonical_registry_model_ids_exposed(self) -> None:
        """Requirement 2: All five canonical registry model IDs are exposed correctly."""
        cfg = load_stage_c_calibration()
        expected_models = set(STAGE_C_MODELS)
        self.assertEqual(set(cfg.models()), expected_models)
        self.assertEqual(set(cfg.temperatures.keys()), expected_models)

    def test_03_resunet_2d_maps_correctly_to_resunet(self) -> None:
        """Requirement 3: C2 JSON key 'resunet_2d' maps correctly to canonical registry ID 'resunet'."""
        cfg = load_stage_c_calibration()
        self.assertIn("resunet", cfg.temperatures)
        self.assertNotIn("resunet_2d", cfg.temperatures)  # canonicalized to resunet
        # temperature_for accepts both canonical and raw JSON key
        self.assertEqual(cfg.temperature_for("resunet"), cfg.temperature_for("resunet_2d"))

    def test_04_supported_aliases_retrieve_same_temperature(self) -> None:
        """Requirement 4: Existing supported aliases retrieve the exact same model temperature."""
        cfg = load_stage_c_calibration()
        # Test aliases for each model
        self.assertEqual(cfg.temperature_for("unet"), cfg.temperature_for("u-net"))
        self.assertEqual(cfg.temperature_for("unet"), cfg.temperature_for("unet2d"))
        self.assertEqual(cfg.temperature_for("unetpp"), cfg.temperature_for("unet++"))
        self.assertEqual(cfg.temperature_for("unetpp"), cfg.temperature_for("unetplusplus2d"))
        self.assertEqual(cfg.temperature_for("attention_unet"), cfg.temperature_for("attentionunet"))
        self.assertEqual(cfg.temperature_for("resunet"), cfg.temperature_for("resunet_2d"))
        self.assertEqual(cfg.temperature_for("resunet"), cfg.temperature_for("resunet2d"))
        self.assertEqual(cfg.temperature_for("unet_resnet34"), cfg.temperature_for("resnet34_unet"))
        self.assertEqual(cfg.temperature_for("unet_resnet34"), cfg.temperature_for("unet-resnet34"))

    def test_05_returned_config_corresponds_to_verified_c2_artifact(self) -> None:
        """Requirement 5: The returned configuration matches authoritative Stage C2 reference values."""
        cfg = load_stage_c_calibration()
        for model, expected_temp in STAGE_C2_REFERENCE_TEMPERATURES.items():
            actual_temp = cfg.temperature_for(model)
            self.assertAlmostEqual(actual_temp, expected_temp, places=6)

    def test_06_manifest_and_calibration_json_agree_on_all_five_values(self) -> None:
        """Requirement 6: Run manifest and calibration JSON agree on all five temperature values."""
        with open(DEFAULT_C2_RUN_MANIFEST, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_temps = manifest_data["fitted_temperatures"]

        cfg = load_stage_c_calibration()
        for raw_key, m_temp in manifest_temps.items():
            canonical = normalize_model_name(raw_key)
            self.assertAlmostEqual(cfg.temperature_for(canonical), m_temp, places=6)

    def test_07_calibration_checkpoint_hashes_match_approved_registry_hashes(self) -> None:
        """Requirement 7: Calibration checkpoint hashes match approved Stage C registry hashes."""
        cfg = load_stage_c_calibration()
        for model in STAGE_C_MODELS:
            expected_hash = STAGE_C_EXPECTED_HASHES[model]
            actual_hash = cfg.checkpoint_hash_for(model)
            self.assertEqual(actual_hash, expected_hash)

    def test_07b_physical_file_sizes_and_hashes_match_specification(self) -> None:
        """Direct check: Physical C2 files on disk match exact expected byte sizes and SHA-256 digests."""
        self.assertTrue(DEFAULT_C2_CALIBRATION_JSON.exists())
        self.assertTrue(DEFAULT_C2_RUN_MANIFEST.exists())

        self.assertEqual(DEFAULT_C2_CALIBRATION_JSON.stat().st_size, EXPECTED_CALIBRATION_JSON_SIZE)
        self.assertEqual(DEFAULT_C2_RUN_MANIFEST.stat().st_size, EXPECTED_C2_MANIFEST_SIZE)

        with open(DEFAULT_C2_CALIBRATION_JSON, "rb") as f:
            calib_hash = hashlib.sha256(f.read()).hexdigest()
        with open(DEFAULT_C2_RUN_MANIFEST, "rb") as f:
            manifest_hash = hashlib.sha256(f.read()).hexdigest()

        self.assertEqual(calib_hash, EXPECTED_CALIBRATION_JSON_SHA256)
        self.assertEqual(manifest_hash, EXPECTED_C2_MANIFEST_SHA256)

    # -------------------------------------------------------------------------
    # 5.2 Artifact Integrity
    # -------------------------------------------------------------------------

    def test_08_correct_calibration_json_sha256_accepted(self) -> None:
        """Requirement 8: Correct calibration JSON SHA-256 is accepted."""
        cfg = load_stage_c_calibration(verify_hashes=True)
        self.assertEqual(cfg.calibration_json_sha256, EXPECTED_CALIBRATION_JSON_SHA256)

    def test_09_incorrect_calibration_json_sha256_rejected(self) -> None:
        """Requirement 9: Incorrect calibration JSON SHA-256 is rejected with ValueError."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["timestamp"] = "2099-01-01 00:00:00"  # Changes file hash
        tmp_calib = _write_temp_json(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, verify_hashes=True)
            self.assertIn("calibration artifact SHA-256 mismatch", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()

    def test_10_correct_run_manifest_sha256_accepted(self) -> None:
        """Requirement 10: Correct run manifest SHA-256 is accepted."""
        cfg = load_stage_c_calibration(verify_hashes=True)
        self.assertEqual(cfg.manifest_sha256, EXPECTED_C2_MANIFEST_SHA256)

    def test_11_incorrect_run_manifest_sha256_rejected(self) -> None:
        """Requirement 11: Incorrect run manifest SHA-256 is rejected with ValueError."""
        _, manifest_dict = _get_base_valid_dicts()
        manifest_dict["timestamp_start"] = "2099-01-01 00:00:00"  # Changes manifest hash
        tmp_manifest = _write_temp_json(manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(manifest_path=tmp_manifest, verify_hashes=True)
            self.assertIn("run manifest SHA-256 mismatch", str(ctx.exception))
        finally:
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_12_manifest_with_wrong_calibration_artifact_hash_rejected(self) -> None:
        """Requirement 12: A manifest that references the wrong calibration artifact hash is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["provenance_hashes"]["output_calibration_parameters_sha256"] = "0" * 64
        tmp_calib = _write_temp_json(calib_dict)
        tmp_manifest = _write_temp_json(manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("output_calibration_parameters_sha256", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_13_manifest_with_incorrect_split_provenance_rejected(self) -> None:
        """Requirement 13: A manifest with incorrect split provenance is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["provenance_hashes"]["splits_json_sha256"] = "f" * 64
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("splits_json_sha256", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    # -------------------------------------------------------------------------
    # 5.3 Checkpoint Identity Handshake
    # -------------------------------------------------------------------------

    def test_14_each_entry_maps_to_correct_stage_c_checkpoint_identity(self) -> None:
        """Requirement 14: Each calibration entry maps to the correct Stage C checkpoint identity."""
        cfg = load_stage_c_calibration()
        for m in STAGE_C_MODELS:
            self.assertEqual(cfg.checkpoint_hash_for(m), STAGE_C_EXPECTED_HASHES[m])

    def test_15_checkpoint_hash_mismatch_is_rejected(self) -> None:
        """Requirement 15: A checkpoint hash mismatch in the calibration JSON is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["checkpoint_sha256"] = "badhash" + "0" * 57
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("checkpoint hash mismatch for 'unet'", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_16_missing_checkpoint_identity_rejected(self) -> None:
        """Requirement 16: Missing checkpoint identity in a model record is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]["unet"]["checkpoint_sha256"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing valid 'checkpoint_sha256'", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_17_duplicate_model_mappings_rejected(self) -> None:
        """Requirement 17: Duplicate model mappings resolving to the same canonical ID are rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        # Add 'resunet' when 'resunet_2d' is already present
        calib_dict["models"]["resunet"] = copy.deepcopy(calib_dict["models"]["resunet_2d"])
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Duplicate model resolution", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_18_missing_models_rejected(self) -> None:
        """Requirement 18: Missing models in calibration JSON are rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]["attention_unet"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing approved Stage C models", str(ctx.exception))
            self.assertIn("attention_unet", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_19_unexpected_model_entries_rejected(self) -> None:
        """Requirement 19: Unexpected model entries outside Stage C are rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["vit_transformer_v2"] = copy.deepcopy(calib_dict["models"]["unet"])
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Unexpected model key", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_20_mismatch_between_json_key_and_internal_model_name_rejected(self) -> None:
        """Requirement 20: A mismatch between the JSON key and internal model_name is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        # Set internal model_name of unet to 'attention_unet'
        calib_dict["models"]["unet"]["model_name"] = "attention_unet"
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("resolves to 'unet' but internal 'model_name'", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    # -------------------------------------------------------------------------
    # 5.4 Temperature Validation
    # -------------------------------------------------------------------------

    def test_21_missing_temperature_rejected(self) -> None:
        """Requirement 21: Missing temperature in a model record is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]["unet"]["optimal_temperature_T"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing 'optimal_temperature_T'", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_22_nonnumeric_temperature_rejected(self) -> None:
        """Requirement 22: Nonnumeric temperature is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = "two_point_five"
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be numeric", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_23_boolean_temperature_rejected(self) -> None:
        """Requirement 23: Boolean temperature is explicitly rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = True
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be numeric (int/float), got bool", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_24_zero_temperature_rejected(self) -> None:
        """Requirement 24: Zero temperature is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = 0.0
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be strictly positive", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_25_negative_temperature_rejected(self) -> None:
        """Requirement 25: Negative temperature is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = -1.5
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be strictly positive", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_26_nan_temperature_rejected(self) -> None:
        """Requirement 26: NaN temperature is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = float("nan")
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("NaN", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_27_infinite_temperature_rejected(self) -> None:
        """Requirement 27: Positive and negative infinity are rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["optimal_temperature_T"] = float("inf")
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("infinite", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_28_inconsistent_fit_diagnostic_temperature_rejected(self) -> None:
        """Requirement 28: Inconsistent fit-diagnostic temperature is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["fit_diagnostics"]["optimal_temperature"] = 1.0  # differs from 2.62679
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Inconsistent temperatures", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_29_non_converged_fit_metadata_rejected(self) -> None:
        """Requirement 29: Non-converged fit metadata is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["fit_diagnostics"]["converged"] = False
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("did not converge", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    # -------------------------------------------------------------------------
    # 5.5 Error and Regression Tests
    # -------------------------------------------------------------------------

    def test_30_missing_calibration_json_raises_file_not_found(self) -> None:
        """Requirement 30: Missing calibration JSON produces an explicit FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_stage_c_calibration(calib_path=Path("non_existent_calibration.json"))
        self.assertIn("not found", str(ctx.exception))

    def test_31_missing_run_manifest_raises_file_not_found(self) -> None:
        """Requirement 31: Missing run manifest produces an explicit FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            load_stage_c_calibration(manifest_path=Path("non_existent_manifest.json"))
        self.assertIn("not found", str(ctx.exception))

    def test_32_malformed_json_produces_informative_error(self) -> None:
        """Requirement 32: Malformed JSON produces an informative ValueError."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            tmp.write("{ invalid json formatting ...")
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_path, verify_hashes=False)
            self.assertIn("Malformed JSON", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_33_incomplete_schema_rejected(self) -> None:
        """Requirement 33: Incomplete JSON schema is rejected."""
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing 'models' dictionary", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_34_unknown_model_identifiers_rejected(self) -> None:
        """Requirement 34: Unknown model identifiers raise KeyError."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(ValueError):  # normalize_model_name raises ValueError on unknown
            cfg.temperature_for("non_existent_model")

    def test_35_invalid_config_never_produces_fallback_or_historical_values(self) -> None:
        """Requirement 35: Invalid configuration fails explicitly, never returning 1.0 or Gate 6A values."""
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["resunet_2d"]["optimal_temperature_T"] = 0.0  # Invalid
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError):
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_36_configuration_loader_does_not_call_torch_load_or_load_weights(self) -> None:
        """Safety requirement: The configuration loader does not call torch.load() or load model weights."""
        with patch("torch.load") as mock_torch_load:
            cfg = load_stage_c_calibration()
            self.assertIsInstance(cfg, StageCCalibrationConfig)
            # torch.load must never be called during calibration loading
            mock_torch_load.assert_not_called()

    def test_37_existing_stage_c_checkpoint_registry_behavior_remains_unchanged(self) -> None:
        """Requirement 37: Existing Batch 1A/1A.1 registry behavior remains fully intact."""
        reg_stage_c = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)
        self.assertEqual(len(reg_stage_c.checkpoint_paths), 5)
        reg_legacy = ModelRegistry()
        self.assertEqual(reg_legacy.profile, CheckpointProfile.LEGACY_DEFAULT)

    # -------------------------------------------------------------------------
    # 5.6 Manifest Temperature Provenance & Root Structure Hardening (Batch 1B.1)
    # -------------------------------------------------------------------------

    def test_38_manifest_missing_one_temperature_rejected(self) -> None:
        """Requirement 1B.1: Manifest missing a single fitted temperature is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        del manifest_dict["fitted_temperatures"]["unet"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing entries for approved Stage C models", str(ctx.exception))
            self.assertIn("unet", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_39_manifest_missing_all_temperatures_rejected(self) -> None:
        """Requirement 1B.1: Manifest missing all fitted temperatures is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"] = {}
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing entries for approved Stage C models", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_40_manifest_unexpected_temperature_entry_rejected(self) -> None:
        """Requirement 1B.1: Manifest containing an unexpected model entry is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unknown_model_v9"] = 2.5
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Unexpected model key", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_41_manifest_duplicate_normalized_models_rejected(self) -> None:
        """Requirement 1B.1: Manifest containing duplicate canonical model mappings is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["resunet"] = 2.8804590702056885
        # both 'resunet_2d' and 'resunet' are now in fitted_temperatures
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Duplicate model resolution in manifest", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_42_manifest_boolean_temperature_rejected(self) -> None:
        """Requirement 1B.1: Manifest containing a boolean temperature is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = True
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be numeric (int/float), got bool", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_43_manifest_nonnumeric_temperature_rejected(self) -> None:
        """Requirement 1B.1: Manifest containing a nonnumeric temperature is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = "two_point_six"
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be numeric", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_44_manifest_nonfinite_temperatures_rejected(self) -> None:
        """Requirement 1B.1: Manifest containing NaN or Inf temperature is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = float("nan")
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("NaN", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

        # Inf test
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = float("inf")
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("infinite", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_45_manifest_zero_and_negative_temperature_rejected(self) -> None:
        """Requirement 1B.1: Manifest zero or negative temperature is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = 0.0
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be strictly positive", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

        calib_dict, manifest_dict = _get_base_valid_dicts()
        manifest_dict["fitted_temperatures"]["unet"] = -2.5
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("must be strictly positive", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_46_manifest_temperature_inconsistent_with_calibration_json_rejected(self) -> None:
        """Requirement 1B.1: Manifest temperature inconsistent with calibration JSON is rejected."""
        calib_dict, manifest_dict = _get_base_valid_dicts()
        # Set manifest temperature for unet to 1.5 while JSON has 2.62679...
        manifest_dict["fitted_temperatures"]["unet"] = 1.5
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict, manifest_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("Temperature mismatch between calibration JSON", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    def test_47_calibration_json_root_is_array_or_null_rejected(self) -> None:
        """Requirement 1B.1: Calibration JSON root being array or null is rejected with clear ValueError."""
        # Array root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write("[1, 2, 3]")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_path, verify_hashes=False)
            self.assertIn("calibration JSON root must be a dictionary, got list", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Null root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write("null")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_path, verify_hashes=False)
            self.assertIn("calibration JSON root must be a dictionary, got NoneType", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_48_manifest_json_root_is_array_or_null_rejected(self) -> None:
        """Requirement 1B.1: Manifest JSON root being array or null is rejected with clear ValueError."""
        # Array root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write("[{\"k\": \"v\"}]")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(manifest_path=tmp_path, verify_hashes=False)
            self.assertIn("run manifest root must be a dictionary, got list", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # Null root
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            tmp.write("null")
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(manifest_path=tmp_path, verify_hashes=False)
            self.assertIn("run manifest root must be a dictionary, got NoneType", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_49_malformed_fit_diagnostic_structure_rejected(self) -> None:
        """Requirement 1B.1: Malformed fit-diagnostic structure is explicitly rejected."""
        # 1. Missing fit_diagnostics
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]["unet"]["fit_diagnostics"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing required 'fit_diagnostics' dictionary", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

        # 2. fit_diagnostics is a list
        calib_dict, _ = _get_base_valid_dicts()
        calib_dict["models"]["unet"]["fit_diagnostics"] = ["not", "a", "dict"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("'fit_diagnostics' must be a dictionary, got list", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

        # 3. fit_diagnostics missing optimal_temperature
        calib_dict, _ = _get_base_valid_dicts()
        del calib_dict["models"]["unet"]["fit_diagnostics"]["optimal_temperature"]
        tmp_calib, tmp_manifest = _write_temp_bundle(calib_dict)
        try:
            with self.assertRaises(ValueError) as ctx:
                load_stage_c_calibration(calib_path=tmp_calib, manifest_path=tmp_manifest, verify_hashes=False)
            self.assertIn("missing 'optimal_temperature'", str(ctx.exception))
        finally:
            if tmp_calib.exists():
                tmp_calib.unlink()
            if tmp_manifest.exists():
                tmp_manifest.unlink()

    # -------------------------------------------------------------------------
    # 5.7 Configuration Immutability & Defensive Copying (Batch 1B.2)
    # -------------------------------------------------------------------------

    def test_50_temperature_mapping_cannot_be_modified(self) -> None:
        """Requirement 1B.2: Temperature mapping cannot be modified after loading."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(TypeError):
            cfg.temperatures["resunet"] = 999.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            del cfg.temperatures["resunet"]  # type: ignore[attr-defined]

    def test_51_checkpoint_hash_mapping_cannot_be_modified(self) -> None:
        """Requirement 1B.2: Checkpoint hash mapping cannot be modified after loading."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(TypeError):
            cfg.checkpoint_hashes["resunet"] = "0" * 64  # type: ignore[index]
        with self.assertRaises(TypeError):
            del cfg.checkpoint_hashes["resunet"]  # type: ignore[attr-defined]

    def test_52_display_name_mapping_cannot_be_modified(self) -> None:
        """Requirement 1B.2: Display name mapping cannot be modified after loading."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(TypeError):
            cfg.display_names["resunet"] = "Hacked Model Name"  # type: ignore[index]
        with self.assertRaises(TypeError):
            del cfg.display_names["resunet"]  # type: ignore[attr-defined]

    def test_53_fit_diagnostic_mapping_cannot_be_modified(self) -> None:
        """Requirement 1B.2: Fit-diagnostic top-level mapping cannot be modified."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(TypeError):
            cfg.fit_diagnostics["resunet"] = {}  # type: ignore[index]
        with self.assertRaises(TypeError):
            del cfg.fit_diagnostics["resunet"]  # type: ignore[attr-defined]

    def test_54_nested_diagnostic_structures_cannot_be_modified(self) -> None:
        """Requirement 1B.2: Nested diagnostic dictionaries cannot be modified."""
        cfg = load_stage_c_calibration()
        with self.assertRaises(TypeError):
            cfg.fit_diagnostics["resunet"]["converged"] = False  # type: ignore[index]
        with self.assertRaises(TypeError):
            cfg.fit_diagnostics_for("resunet")["optimal_temperature"] = 1.0  # type: ignore[index]
        with self.assertRaises(TypeError):
            del cfg.fit_diagnostics["resunet"]["converged"]  # type: ignore[attr-defined]

    def test_55_defensive_copying_prevents_mutation_via_source_dict(self) -> None:
        """Requirement 1B.2: Modifying source dictionaries does not affect the frozen config."""
        mutable_temps = {"resunet": 2.8804590702056885, "unet": 2.626793384552002}
        mutable_diag = {"resunet": {"converged": True}}
        config = StageCCalibrationConfig(
            temperatures=mutable_temps,
            checkpoint_hashes={},
            display_names={},
            fit_diagnostics=mutable_diag,
            calibration_json_sha256="dummy",
            manifest_sha256="dummy",
            splits_json_sha256="dummy",
            timestamp="dummy",
        )
        # Mutate the source dictionary externally
        mutable_temps["resunet"] = 999.0
        mutable_diag["resunet"]["converged"] = False

        # Config must remain unaffected by source mutations
        self.assertEqual(config.temperatures["resunet"], 2.8804590702056885)
        self.assertTrue(config.fit_diagnostics["resunet"]["converged"])

    def test_56_all_existing_read_operations_continue_to_work(self) -> None:
        """Requirement 1B.2: All existing public read operations and interfaces continue to work."""
        cfg = load_stage_c_calibration()
        # Subscripting, membership, iteration, get
        self.assertIn("resunet", cfg.temperatures)
        self.assertEqual(cfg.temperatures.get("resunet"), cfg.temperatures["resunet"])
        self.assertEqual(len(cfg.temperatures), 5)
        self.assertEqual(len(list(cfg.temperatures.keys())), 5)
        self.assertEqual(len(list(cfg.temperatures.values())), 5)
        self.assertEqual(len(list(cfg.temperatures.items())), 5)

        # Helper methods
        for m in STAGE_C_MODELS:
            self.assertGreater(cfg.temperature_for(m), 0.0)
            self.assertEqual(len(cfg.checkpoint_hash_for(m)), 64)
            self.assertTrue(len(cfg.display_name_for(m)) > 0)
            self.assertTrue(cfg.fit_diagnostics_for(m)["converged"])

    def test_57_five_authoritative_temperatures_remain_unchanged(self) -> None:
        """Requirement 1B.2: The five loaded temperatures match reference values exactly."""
        cfg = load_stage_c_calibration()
        for model, expected_temp in STAGE_C2_REFERENCE_TEMPERATURES.items():
            self.assertAlmostEqual(cfg.temperature_for(model), expected_temp, places=7)


if __name__ == "__main__":
    unittest.main()
