"""Unit tests for Stage C Checkpoint Profiles and Provenance Verification.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Batch 1A.1: Checkpoint Verification Hardening & Profile Semantics Clarification.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.agent.models import (
    CANDIDATE_CHECKPOINTS,
    CANDIDATE_MODEL_NAMES,
    CANONICAL_MODEL_NAMES,
    CheckpointProfile,
    DEFAULT_CHECKPOINTS,
    GATE5D_CHECKPOINTS,
    GENUINE_5EP_CHECKPOINTS,
    MODEL_ALIASES,
    MODEL_FACTORIES,
    ModelRegistry,
    STAGE_C_CHECKPOINTS,
    STAGE_C_EXPECTED_HASHES,
    STAGE_C_MODELS,
    compute_file_sha256,
    compute_stream_sha256,
    normalize_model_name,
    normalize_profile,
    verify_checkpoint_sha256,
)


class TestCheckpointProfiles(unittest.TestCase):
    """Test suite for ModelRegistry checkpoint profiles and SHA-256 verification."""

    # -------------------------------------------------------------------------
    # 1. Profile Resolution & Backward Compatibility
    # -------------------------------------------------------------------------

    def test_01_historical_profile_paths_preserved(self) -> None:
        """Requirement 1: Historical profile preserves its original checkpoint paths."""
        reg_default = ModelRegistry()
        reg_legacy_enum = ModelRegistry(profile=CheckpointProfile.LEGACY_DEFAULT)
        reg_hist_enum = ModelRegistry(profile=CheckpointProfile.HISTORICAL_5EP)
        reg_hist_str = ModelRegistry(profile="historical_5ep")

        self.assertEqual(reg_default.profile, CheckpointProfile.LEGACY_DEFAULT)
        self.assertEqual(reg_legacy_enum.profile, CheckpointProfile.LEGACY_DEFAULT)
        self.assertEqual(reg_hist_enum.profile, CheckpointProfile.HISTORICAL_5EP)
        self.assertEqual(reg_hist_str.profile, CheckpointProfile.HISTORICAL_5EP)

        # Baseline canonical models
        for name in CANONICAL_MODEL_NAMES:
            expected = DEFAULT_CHECKPOINTS[name]
            self.assertEqual(reg_default.checkpoint_paths[name], expected)
            self.assertEqual(reg_legacy_enum.checkpoint_paths[name], expected)
            self.assertEqual(reg_hist_enum.checkpoint_paths[name], expected)
            self.assertEqual(reg_hist_str.checkpoint_paths[name], expected)

        # Candidate models (preserve legacy candidate resolution)
        for name in CANDIDATE_MODEL_NAMES:
            expected = CANDIDATE_CHECKPOINTS[name]
            self.assertEqual(reg_default.checkpoint_paths[name], expected)
            self.assertEqual(reg_legacy_enum.checkpoint_paths[name], expected)
            self.assertEqual(reg_hist_enum.checkpoint_paths[name], expected)
            self.assertEqual(reg_hist_str.checkpoint_paths[name], expected)

    def test_02_stage_c_profile_resolves_all_five_authoritative_paths(self) -> None:
        """Requirement 2: Stage C profile resolves all five authoritative checkpoint paths."""
        reg_stage_c = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)
        self.assertEqual(reg_stage_c.profile, CheckpointProfile.STAGE_C_15EP)

        for name in STAGE_C_MODELS:
            self.assertIn(name, reg_stage_c.checkpoint_paths)
            self.assertEqual(reg_stage_c.checkpoint_paths[name], STAGE_C_CHECKPOINTS[name])

    def test_03_stage_c_each_model_maps_to_unique_intended_checkpoint(self) -> None:
        """Requirement 3: Each Stage C model maps to exactly one intended checkpoint."""
        reg_stage_c = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)
        resolved_paths = [reg_stage_c.checkpoint_paths[m] for m in STAGE_C_MODELS]

        # 5 distinct models must resolve to 5 distinct physical paths
        self.assertEqual(len(resolved_paths), 5)
        self.assertEqual(len(set(resolved_paths)), 5)

        # Confirm exact expected relative sub-locations
        self.assertTrue(str(reg_stage_c.checkpoint_paths["resunet"]).endswith(
            os.path.join("results", "full_training_15ep", "resunet_2d", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["unet_resnet34"]).endswith(
            os.path.join("results", "full_training_15ep", "unet_resnet34", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["unet"]).endswith(
            os.path.join("results", "stage_c", "phase1", "unet_721_seed42", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["unetpp"]).endswith(
            os.path.join("results", "stage_c", "phase1", "unetpp_721_seed42", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["attention_unet"]).endswith(
            os.path.join("results", "stage_c", "phase1", "attention_unet_721_seed42", "best_model.pt")
        ))

    def test_04_unknown_model_identifiers_rejected(self) -> None:
        """Requirement 4: Unknown model identifiers are rejected with clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            normalize_model_name("unknown_transformer_v9")
        self.assertIn("Unknown model name", str(ctx.exception))

        reg = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)
        with self.assertRaises(ValueError):
            reg.get_model("unknown_transformer_v9")
        with self.assertRaises(ValueError):
            reg.verify_checkpoint("unknown_transformer_v9")

    def test_05_unknown_profile_identifiers_rejected(self) -> None:
        """Requirement 5: Unknown profile identifiers are rejected with clear ValueError."""
        with self.assertRaises(ValueError) as ctx:
            normalize_profile("unsupported_future_profile")
        self.assertIn("Unknown checkpoint profile", str(ctx.exception))

        with self.assertRaises(ValueError):
            ModelRegistry(profile="unsupported_future_profile")

    def test_06_missing_checkpoint_files_raise_explicit_errors(self) -> None:
        """Requirement 6: Missing checkpoint files raise explicit FileNotFoundError."""
        non_existent = Path("results/stage_c/phase1/missing_model/best_model.pt")
        with self.assertRaises(FileNotFoundError) as ctx:
            verify_checkpoint_sha256(
                non_existent,
                expected_sha256="abc123",
                model_name="missing_model",
                profile="stage_c_15ep",
            )
        self.assertIn("missing_model", str(ctx.exception))
        self.assertIn("stage_c_15ep", str(ctx.exception))

    def test_07_stage_c_never_silently_downgrades_to_historical(self) -> None:
        """Requirement 7: Explicitly selected Stage C profile never silently downgrades to historical weights."""
        reg_hist = ModelRegistry(profile=CheckpointProfile.LEGACY_DEFAULT)
        reg_stage_c = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)

        # Baseline models in Stage C must NOT point to the 5-epoch baseline directories
        self.assertNotEqual(
            reg_stage_c.checkpoint_paths["unet"],
            reg_hist.checkpoint_paths["unet"],
        )
        self.assertNotEqual(
            reg_stage_c.checkpoint_paths["unetpp"],
            reg_hist.checkpoint_paths["unetpp"],
        )
        self.assertNotEqual(
            reg_stage_c.checkpoint_paths["attention_unet"],
            reg_hist.checkpoint_paths["attention_unet"],
        )
        self.assertIn("unet_721_seed42", str(reg_stage_c.checkpoint_paths["unet"]))
        self.assertIn("baseline_unet2d_5ep", str(reg_hist.checkpoint_paths["unet"]))

    # -------------------------------------------------------------------------
    # 2. Checkpoint Identity & Cryptographic Hashing
    # -------------------------------------------------------------------------

    def test_08_sha256_chunked_calculation_matches_known_digest(self) -> None:
        """Requirement 8: SHA-256 calculation matches an independently computed expected digest."""
        test_payload = b"MajorProject Research Prototype Chunked SHA-256 Verification\n" * 1024
        expected_digest = hashlib.sha256(test_payload).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(test_payload)
            tmp_path = Path(tmp.name)

        try:
            computed_digest = compute_file_sha256(tmp_path, chunk_size=512)
            self.assertEqual(computed_digest, expected_digest)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_09_sha256_mismatch_is_rejected(self) -> None:
        """Requirement 9: SHA-256 mismatch is rejected with ValueError."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(b"Controlled model checkpoint mock content")
            tmp_path = Path(tmp.name)

        wrong_hash = "0000000000000000000000000000000000000000000000000000000000000000"
        try:
            with self.assertRaises(ValueError):
                verify_checkpoint_sha256(
                    tmp_path,
                    expected_sha256=wrong_hash,
                    model_name="mock_model",
                    profile="stage_c_15ep",
                )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_10_mismatch_errors_report_expected_and_actual_hashes(self) -> None:
        """Requirement 10: Mismatch errors report expected and actual hashes, model name and profile."""
        payload = b"Specific corrupt checkpoint mock payload"
        actual_hash = hashlib.sha256(payload).hexdigest()
        wrong_hash = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(payload)
            tmp_path = Path(tmp.name)

        try:
            with self.assertRaises(ValueError) as ctx:
                verify_checkpoint_sha256(
                    tmp_path,
                    expected_sha256=wrong_hash,
                    model_name="test_resunet",
                    profile="stage_c_15ep",
                )
            error_msg = str(ctx.exception)
            self.assertIn(wrong_hash, error_msg)
            self.assertIn(actual_hash, error_msg)
            self.assertIn("test_resunet", error_msg)
            self.assertIn("stage_c_15ep", error_msg)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_11_stage_c_expected_hash_mapping_contains_five_approved_identities(self) -> None:
        """Requirement 11: The Stage C expected-hash mapping contains the five approved identities."""
        self.assertEqual(set(STAGE_C_EXPECTED_HASHES.keys()), set(STAGE_C_MODELS))
        self.assertEqual(
            STAGE_C_EXPECTED_HASHES["resunet"],
            "81ef568df7dc47e3b883d4c676abc54213261e35aef079c372ac5225eeb19a8d",
        )
        self.assertEqual(
            STAGE_C_EXPECTED_HASHES["unet_resnet34"],
            "891d869a2369c395dbcb9b2b1adf3fec36d42f5e9b16508f7a964198a9332d1d",
        )
        self.assertEqual(
            STAGE_C_EXPECTED_HASHES["unet"],
            "37d35bf578868356caadc7f4a173af7b82535bb427fdb6cce8df121a9bf06cef",
        )
        self.assertEqual(
            STAGE_C_EXPECTED_HASHES["unetpp"],
            "fbd237a42bdf12f9c945a1c8caa953dfb44a1672d3514cc842368fa535f4bfe6",
        )
        self.assertEqual(
            STAGE_C_EXPECTED_HASHES["attention_unet"],
            "5f7d8dae1fb68e86905d2c36f56446de8c0a09babe0ce550230b157aa11e591e",
        )

    def test_12_actual_five_checkpoints_match_authoritative_hashes(self) -> None:
        """Requirement 12: The actual five checkpoint identities match the authoritative hashes."""
        reg = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)
        digests = reg.verify_all_checkpoints()

        self.assertEqual(len(digests), 5)
        for model_name, actual_hash in digests.items():
            expected = STAGE_C_EXPECTED_HASHES[model_name]
            self.assertEqual(
                actual_hash,
                expected,
                f"Physical checkpoint hash mismatch for '{model_name}'!",
            )

    # -------------------------------------------------------------------------
    # 3. Backward Compatibility & Architecture Aliases
    # -------------------------------------------------------------------------

    def test_13_model_registry_preserves_original_constructor_behavior(self) -> None:
        """Requirement 13: ModelRegistry() preserves its original constructor behavior."""
        reg = ModelRegistry()
        self.assertEqual(reg.profile, CheckpointProfile.LEGACY_DEFAULT)
        self.assertIn("unet", reg.checkpoint_paths)
        self.assertIn("resunet", reg.checkpoint_paths)
        self.assertEqual(reg.checkpoint_paths["unet"], DEFAULT_CHECKPOINTS["unet"])
        self.assertEqual(reg.checkpoint_paths["resunet"], CANDIDATE_CHECKPOINTS["resunet"])

    def test_14_existing_model_architecture_aliases_remain_functional(self) -> None:
        """Requirement 14: Existing model architecture aliases remain functional."""
        self.assertEqual(normalize_model_name("u-net"), "unet")
        self.assertEqual(normalize_model_name("unet2d"), "unet")
        self.assertEqual(normalize_model_name("unet++"), "unetpp")
        self.assertEqual(normalize_model_name("unetplusplus2d"), "unetpp")
        self.assertEqual(normalize_model_name("attentionunet"), "attention_unet")
        self.assertEqual(normalize_model_name("resunet_2d"), "resunet")
        self.assertEqual(normalize_model_name("resunet2d"), "resunet")
        self.assertEqual(normalize_model_name("resnet34_unet"), "unet_resnet34")
        self.assertEqual(normalize_model_name("unet-resnet34"), "unet_resnet34")

    def test_15_historical_registry_mappings_and_custom_override_unchanged(self) -> None:
        """Requirement 15: Historical registry mappings and custom override behavior remain unchanged."""
        custom_override = {"unet": Path("custom_unet_5ep.pt")}
        reg = ModelRegistry(checkpoint_paths=custom_override, profile=CheckpointProfile.LEGACY_DEFAULT)
        self.assertEqual(reg.checkpoint_paths["unet"], Path("custom_unet_5ep.pt"))
        self.assertEqual(reg.checkpoint_paths["unetpp"], DEFAULT_CHECKPOINTS["unetpp"])

    # -------------------------------------------------------------------------
    # 4. Verification Hardening & Streaming Deserialization Regressions (Batch 1A.1)
    # -------------------------------------------------------------------------

    def test_16_valid_checkpoint_passes_verification(self) -> None:
        """A valid checkpoint passes cryptographic SHA-256 verification."""
        mock_payload = b"Valid mock checkpoint binary data block\n" * 128
        expected_hash = hashlib.sha256(mock_payload).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(mock_payload)
            tmp_path = Path(tmp.name)

        try:
            result_hash = verify_checkpoint_sha256(
                tmp_path,
                expected_sha256=expected_hash,
                model_name="mock_valid",
                profile="stage_c_15ep",
            )
            self.assertEqual(result_hash, expected_hash)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_17_modified_checkpoint_rejected_same_size(self) -> None:
        """A modified checkpoint is rejected even when its file size remains unchanged."""
        original_data = b"Original payload block 12345678"
        tampered_data = b"Tampered payload block 87654321"
        self.assertEqual(len(original_data), len(tampered_data))

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(original_data)
            tmp_path = Path(tmp.name)

        expected_hash = hashlib.sha256(original_data).hexdigest()

        try:
            # Overwrite with tampered data of exact same byte size
            with open(tmp_path, "wb") as f:
                f.write(tampered_data)

            self.assertEqual(tmp_path.stat().st_size, len(original_data))

            with self.assertRaises(ValueError) as ctx:
                verify_checkpoint_sha256(
                    tmp_path,
                    expected_sha256=expected_hash,
                    model_name="mock_same_size",
                    profile="stage_c_15ep",
                )
            self.assertIn("Checkpoint integrity verification failed", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_18_replacement_file_with_same_size_and_mtime_rejected(self) -> None:
        """A replacement file with the same size and restored st_mtime_ns is rejected."""
        original_data = b"Original baseline data for timestamp test"
        replaced_data = b"Tampered replaced data for timestamp test"
        self.assertEqual(len(original_data), len(replaced_data))

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(original_data)
            tmp_path = Path(tmp.name)

        expected_hash = hashlib.sha256(original_data).hexdigest()
        orig_stat = tmp_path.stat()
        orig_size = orig_stat.st_size
        orig_atime_ns = orig_stat.st_atime_ns
        orig_mtime_ns = orig_stat.st_mtime_ns

        try:
            # Replace file content with exact same size
            with open(tmp_path, "wb") as f:
                f.write(replaced_data)

            # Restore original nanosecond access and modification timestamps
            os.utime(tmp_path, ns=(orig_atime_ns, orig_mtime_ns))

            # Verify size and modification timestamps are identical to original
            new_stat = tmp_path.stat()
            self.assertEqual(new_stat.st_size, orig_size)
            self.assertEqual(new_stat.st_mtime_ns, orig_mtime_ns)

            # Verification must still reject because actual file bytes are hashed without shortcuts
            with self.assertRaises(ValueError) as ctx:
                verify_checkpoint_sha256(
                    tmp_path,
                    expected_sha256=expected_hash,
                    model_name="mock_mtime",
                    profile="stage_c_15ep",
                )
            self.assertIn("Checkpoint integrity verification failed", str(ctx.exception))
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_19_deserialization_does_not_occur_after_hash_mismatch(self) -> None:
        """Deserialization (torch.load) does not occur after a hash mismatch."""
        mock_corrupt_payload = b"Corrupted checkpoint mock binary payload\n"

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(mock_corrupt_payload)
            tmp_path = Path(tmp.name)

        try:
            reg = ModelRegistry(
                profile=CheckpointProfile.STAGE_C_15EP,
                checkpoint_paths={"unet": tmp_path},
            )
            # The corrupt payload's hash will not match STAGE_C_EXPECTED_HASHES['unet']
            with patch("torch.load") as mock_torch_load:
                with self.assertRaises(ValueError) as ctx:
                    reg.get_model("unet")

                # Critical assertion: torch.load was NEVER called!
                mock_torch_load.assert_not_called()

                # Verify explicit error message formatting
                err_msg = str(ctx.exception)
                self.assertIn("unet", err_msg)
                self.assertIn("stage_c_15ep", err_msg)
                self.assertIn(str(tmp_path), err_msg)
                self.assertIn("Expected SHA-256", err_msg)
                self.assertIn("Actual SHA-256", err_msg)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_20_verified_file_handle_is_passed_to_deserialization(self) -> None:
        """The verified file handle is the exact one rewound and passed to torch.load()."""
        mock_payload = b"Mock valid torch weights stream for handle test\n"
        mock_hash = hashlib.sha256(mock_payload).hexdigest()

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(mock_payload)
            tmp_path = Path(tmp.name)

        try:
            reg = ModelRegistry(
                profile=CheckpointProfile.STAGE_C_15EP,
                checkpoint_paths={"unet": tmp_path},
            )
            # Patch STAGE_C_EXPECTED_HASHES so 'unet' expects mock_hash for this test
            with patch.dict("src.agent.models.STAGE_C_EXPECTED_HASHES", {"unet": mock_hash}):
                with patch("torch.load") as mock_torch_load:
                    # Provide dummy model_state_dict to satisfy load_state_dict
                    mock_torch_load.return_value = {"model_state_dict": {}}
                    with patch.object(MODEL_FACTORIES["unet"], "load_state_dict"):
                        reg.get_model("unet")

                    # Verify torch.load was called exactly once
                    self.assertEqual(mock_torch_load.call_count, 1)
                    passed_handle = mock_torch_load.call_args[0][0]

                    # Assert passed argument is an open binary stream with read and seek
                    self.assertTrue(hasattr(passed_handle, "read"))
                    self.assertTrue(hasattr(passed_handle, "seek"))
                    # Assert it's the exact same file handle
                    self.assertEqual(
                        Path(passed_handle.name).resolve(),
                        tmp_path.resolve(),
                    )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_21_loaded_models_cache_prevents_repeated_disk_reads(self) -> None:
        """The existing _loaded_models cache returns loaded instances without re-reading disk."""
        reg = ModelRegistry(profile=CheckpointProfile.LEGACY_DEFAULT)
        # Pre-populate _loaded_models with a mock module
        mock_model = MagicMock(spec=MODEL_FACTORIES["unet"])
        mock_param = MagicMock()
        mock_param.device = reg.default_device
        mock_model.parameters.return_value = iter([mock_param])

        reg._loaded_models["unet"] = mock_model

        # Calling get_model returns cached instance directly without accessing disk
        retrieved = reg.get_model("unet")
        self.assertIs(retrieved, mock_model)

    def test_22_profile_semantics_explicit_distinction(self) -> None:
        """Profile semantics: explicit distinction between legacy default, genuine 5ep, and Stage C 15ep."""
        reg_default = ModelRegistry()
        reg_legacy = ModelRegistry(profile=CheckpointProfile.LEGACY_DEFAULT)
        reg_genuine_5ep = ModelRegistry(profile=CheckpointProfile.GENUINE_5EP)
        reg_stage_c = ModelRegistry(profile=CheckpointProfile.STAGE_C_15EP)

        # 1. Default constructor uses LEGACY_DEFAULT
        self.assertEqual(reg_default.profile, CheckpointProfile.LEGACY_DEFAULT)
        self.assertEqual(reg_default.checkpoint_paths, reg_legacy.checkpoint_paths)

        # 2. Legacy default: unet, unetpp, attention_unet are 5ep; resunet and unet_resnet34 are 15ep Gate 5D
        self.assertTrue(str(reg_legacy.checkpoint_paths["resunet"]).endswith(
            os.path.join("results", "full_training_15ep", "resunet_2d", "best_model.pt")
        ))
        self.assertTrue(str(reg_legacy.checkpoint_paths["unet_resnet34"]).endswith(
            os.path.join("results", "full_training_15ep", "unet_resnet34", "best_model.pt")
        ))
        self.assertTrue(str(reg_legacy.checkpoint_paths["unet"]).endswith(
            os.path.join("results", "baseline_unet2d_5ep", "best_model.pt")
        ))

        # 3. Genuine 5EP: ALL 5 models are pure 5-epoch
        self.assertTrue(str(reg_genuine_5ep.checkpoint_paths["resunet"]).endswith(
            os.path.join("results", "resunet2d_5ep", "best_model.pt")
        ))
        self.assertTrue(str(reg_genuine_5ep.checkpoint_paths["unet_resnet34"]).endswith(
            os.path.join("results", "unet_resnet34_5ep", "best_model.pt")
        ))
        self.assertTrue(str(reg_genuine_5ep.checkpoint_paths["unet"]).endswith(
            os.path.join("results", "baseline_unet2d_5ep", "best_model.pt")
        ))

        # 4. Stage C 15EP: All 5 models are 15-epoch scaled
        self.assertTrue(str(reg_stage_c.checkpoint_paths["unet"]).endswith(
            os.path.join("results", "stage_c", "phase1", "unet_721_seed42", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["unetpp"]).endswith(
            os.path.join("results", "stage_c", "phase1", "unetpp_721_seed42", "best_model.pt")
        ))
        self.assertTrue(str(reg_stage_c.checkpoint_paths["attention_unet"]).endswith(
            os.path.join("results", "stage_c", "phase1", "attention_unet_721_seed42", "best_model.pt")
        ))


if __name__ == "__main__":
    unittest.main()
