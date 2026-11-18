"""Stage C2 Temperature Calibration Configuration and Provenance Loader.

Workspace: D:\\MajorProject
Governing Authority: MASTER_PROJECT_BLUEPRINT_v2.0.md
Calibration Authority: Stage C Phase 2 (c2_calibration_v1)
Batch 1B: Stage C2 Calibration Provenance & Configuration Loader.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from src.agent.models import (
    STAGE_C_EXPECTED_HASHES,
    STAGE_C_MODELS,
    normalize_model_name,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Authoritative C2 Artifact Paths
DEFAULT_C2_CALIBRATION_JSON = (
    PROJECT_ROOT
    / "results"
    / "stage_c"
    / "phase2"
    / "c2_calibration_v1"
    / "calibration_parameters.json"
)

DEFAULT_C2_RUN_MANIFEST = (
    PROJECT_ROOT
    / "results"
    / "stage_c"
    / "phase2"
    / "c2_calibration_v1"
    / "c2_run_manifest.json"
)

# Authoritative Cryptographic Digests
EXPECTED_CALIBRATION_JSON_SHA256 = (
    "77819fe816255d75446f9497e4b361d5364cb674ff55904dbb52240e3a76f67e"
)
EXPECTED_C2_MANIFEST_SHA256 = (
    "e56617faf06a605e6c51bde3b117ba0e02f438019ef48b2947edd1b92c758460"
)
EXPECTED_C2_SPLITS_SHA256 = (
    "e753caa256950dee0ed921ee4e8a08f6f515dcb062862c1fb7a699e08f69059d"
)

# Reference File Sizes
EXPECTED_CALIBRATION_JSON_SIZE = 30620
EXPECTED_C2_MANIFEST_SIZE = 2148

# Strict floating-point tolerance for temperature comparisons
TEMPERATURE_COMPARISON_TOLERANCE = 1e-7

# Authoritative Reference Temperatures (Stage C Phase 2 c2_calibration_v1)
STAGE_C2_REFERENCE_TEMPERATURES: Mapping[str, float] = {
    "resunet": 2.8804590702056885,
    "unet_resnet34": 2.6260907649993896,
    "unet": 2.626793384552002,
    "unetpp": 2.5647027492523193,
    "attention_unet": 2.2868876457214355,
}


def _freeze_mapping(obj: Any) -> Any:
    """Recursively freeze mappings and sequences into immutable proxies and tuples.

    Performs deep defensive copying so that mutations on source objects do not affect
    the frozen representation.
    """
    if isinstance(obj, Mapping):
        return MappingProxyType({k: _freeze_mapping(v) for k, v in obj.items()})
    elif isinstance(obj, (list, tuple)):
        return tuple(_freeze_mapping(item) for item in obj)
    elif isinstance(obj, set):
        return frozenset(_freeze_mapping(item) for item in obj)
    return obj


@dataclass(frozen=True)
class StageCCalibrationConfig:
    """Immutable configuration container for verified Stage C2 temperature calibration.

    All mapping fields are defensively copied and recursively wrapped in MappingProxyType
    to guarantee genuine immutability against accidental modification.
    """

    temperatures: Mapping[str, float]
    checkpoint_hashes: Mapping[str, str]
    display_names: Mapping[str, str]
    fit_diagnostics: Mapping[str, Mapping[str, Any]]
    calibration_json_sha256: str
    manifest_sha256: str
    splits_json_sha256: str
    timestamp: str

    def __post_init__(self) -> None:
        """Enforce recursive defensive copying and mapping immutability."""
        object.__setattr__(self, "temperatures", _freeze_mapping(self.temperatures))
        object.__setattr__(self, "checkpoint_hashes", _freeze_mapping(self.checkpoint_hashes))
        object.__setattr__(self, "display_names", _freeze_mapping(self.display_names))
        object.__setattr__(self, "fit_diagnostics", _freeze_mapping(self.fit_diagnostics))

    def models(self) -> tuple[str, ...]:
        """Return tuple of canonical model identifiers."""
        return tuple(self.temperatures.keys())

    def temperature_for(self, model_name: str) -> float:
        """Retrieve calibrated temperature using canonical model identifier or alias.

        Args:
            model_name: Canonical model ID or recognized alias.

        Returns:
            Calibrated temperature float > 0.

        Raises:
            KeyError: If model is unrecognized or not configured in Stage C2.
        """
        canonical = normalize_model_name(model_name)
        if canonical not in self.temperatures:
            raise KeyError(
                f"Model '{canonical}' is not configured in Stage C2 calibration (available: {list(self.temperatures.keys())})."
            )
        return self.temperatures[canonical]

    def checkpoint_hash_for(self, model_name: str) -> str:
        """Retrieve checkpoint SHA-256 identity for a model."""
        canonical = normalize_model_name(model_name)
        if canonical not in self.checkpoint_hashes:
            raise KeyError(
                f"Model '{canonical}' is not configured in Stage C2 calibration."
            )
        return self.checkpoint_hashes[canonical]

    def display_name_for(self, model_name: str) -> str:
        """Retrieve human-readable display name for a model."""
        canonical = normalize_model_name(model_name)
        if canonical not in self.display_names:
            raise KeyError(
                f"Model '{canonical}' is not configured in Stage C2 calibration."
            )
        return self.display_names[canonical]

    def fit_diagnostics_for(self, model_name: str) -> Mapping[str, Any]:
        """Retrieve optimizer fit diagnostics for a model."""
        canonical = normalize_model_name(model_name)
        if canonical not in self.fit_diagnostics:
            raise KeyError(
                f"Model '{canonical}' is not configured in Stage C2 calibration."
            )
        return self.fit_diagnostics[canonical]


def load_stage_c_calibration(
    calib_path: Path | str | None = None,
    manifest_path: Path | str | None = None,
    verify_hashes: bool = True,
) -> StageCCalibrationConfig:
    """Load, verify, and return authoritative Stage C2 temperature calibration configuration.

    Args:
        calib_path: Path to calibration_parameters.json. Defaults to DEFAULT_C2_CALIBRATION_JSON.
        manifest_path: Path to c2_run_manifest.json. Defaults to DEFAULT_C2_RUN_MANIFEST.
        verify_hashes: Whether to cryptographically verify SHA-256 digests against authoritative values.

    Returns:
        StageCCalibrationConfig container with validated temperatures and provenance metadata.

    Raises:
        FileNotFoundError: If calibration JSON or manifest file is missing.
        ValueError: If file hashes, manifest relationships, model identities, or temperature values are invalid.
    """
    p_calib = Path(calib_path if calib_path is not None else DEFAULT_C2_CALIBRATION_JSON).resolve()
    p_manifest = Path(manifest_path if manifest_path is not None else DEFAULT_C2_RUN_MANIFEST).resolve()

    if not p_calib.exists() or not p_calib.is_file():
        raise FileNotFoundError(f"Stage C2 calibration parameters JSON not found at: {p_calib}")
    if not p_manifest.exists() or not p_manifest.is_file():
        raise FileNotFoundError(f"Stage C2 run manifest JSON not found at: {p_manifest}")

    # Read raw bytes once from each file
    with open(p_calib, "rb") as f:
        calib_bytes = f.read()
    with open(p_manifest, "rb") as f:
        manifest_bytes = f.read()

    actual_calib_sha256 = hashlib.sha256(calib_bytes).hexdigest()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    # Step 1: Verify raw file hashes if requested
    if verify_hashes:
        if actual_calib_sha256 != EXPECTED_CALIBRATION_JSON_SHA256:
            raise ValueError(
                f"Stage C2 calibration artifact SHA-256 mismatch!\n"
                f"  Artifact Path:   {p_calib}\n"
                f"  Expected SHA-256: {EXPECTED_CALIBRATION_JSON_SHA256}\n"
                f"  Actual SHA-256:   {actual_calib_sha256}"
            )
        if actual_manifest_sha256 != EXPECTED_C2_MANIFEST_SHA256:
            raise ValueError(
                f"Stage C2 run manifest SHA-256 mismatch!\n"
                f"  Manifest Path:   {p_manifest}\n"
                f"  Expected SHA-256: {EXPECTED_C2_MANIFEST_SHA256}\n"
                f"  Actual SHA-256:   {actual_manifest_sha256}"
            )

    # Step 2: Parse verified bytes (no reopening of file handles)
    try:
        calib_data = json.loads(calib_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in Stage C2 calibration file {p_calib}: {exc}") from exc

    try:
        manifest_data = json.loads(manifest_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in Stage C2 manifest file {p_manifest}: {exc}") from exc

    # Validate JSON root structures immediately
    if not isinstance(calib_data, dict):
        raise ValueError(
            f"Stage C2 calibration JSON root must be a dictionary, got {type(calib_data).__name__}."
        )
    if not isinstance(manifest_data, dict):
        raise ValueError(
            f"Stage C2 run manifest root must be a dictionary, got {type(manifest_data).__name__}."
        )

    # Step 3: Validate manifest relationships
    prov_hashes = manifest_data.get("provenance_hashes")
    if not isinstance(prov_hashes, dict):
        raise ValueError(f"Manifest at {p_manifest} missing valid 'provenance_hashes' object.")

    # Manifest must point to this calibration artifact
    manifest_out_calib_hash = prov_hashes.get("output_calibration_parameters_sha256")
    if manifest_out_calib_hash != actual_calib_sha256:
        raise ValueError(
            f"Manifest 'output_calibration_parameters_sha256' ({manifest_out_calib_hash}) "
            f"does not match actual calibration JSON digest ({actual_calib_sha256})."
        )

    # Manifest must point to the expected C2 split identity
    manifest_splits_hash = prov_hashes.get("splits_json_sha256")
    if manifest_splits_hash != EXPECTED_C2_SPLITS_SHA256:
        raise ValueError(
            f"Manifest 'splits_json_sha256' ({manifest_splits_hash}) "
            f"does not match expected C2 split digest ({EXPECTED_C2_SPLITS_SHA256})."
        )

    manifest_checkpoints = prov_hashes.get("checkpoints")
    if not isinstance(manifest_checkpoints, dict):
        raise ValueError(f"Manifest at {p_manifest} missing valid 'checkpoints' dictionary.")

    manifest_fitted_temps = manifest_data.get("fitted_temperatures")
    if not isinstance(manifest_fitted_temps, dict):
        raise ValueError(f"Manifest at {p_manifest} missing valid 'fitted_temperatures' dictionary.")

    # Validate complete manifest temperature provenance
    manifest_canonical_temps: dict[str, float] = {}
    seen_manifest_canonicals: dict[str, str] = {}  # canonical -> raw_key

    for raw_m_key, raw_m_temp in manifest_fitted_temps.items():
        try:
            m_canonical = normalize_model_name(raw_m_key)
        except ValueError as exc:
            raise ValueError(
                f"Unexpected model key '{raw_m_key}' in manifest fitted_temperatures: {exc}"
            ) from exc

        if m_canonical not in STAGE_C_MODELS:
            raise ValueError(
                f"Model '{raw_m_key}' (canonical: '{m_canonical}') in manifest is not an approved Stage C model."
            )

        if m_canonical in seen_manifest_canonicals:
            raise ValueError(
                f"Duplicate model resolution in manifest fitted_temperatures: '{raw_m_key}' and "
                f"'{seen_manifest_canonicals[m_canonical]}' both resolve to canonical '{m_canonical}'."
            )
        seen_manifest_canonicals[m_canonical] = raw_m_key

        # Strict validation of manifest temperature value
        if isinstance(raw_m_temp, bool) or not isinstance(raw_m_temp, (int, float)):
            raise ValueError(
                f"Manifest fitted temperature for '{raw_m_key}' must be numeric (int/float), "
                f"got {type(raw_m_temp).__name__}: {raw_m_temp}"
            )

        m_temp_float = float(raw_m_temp)
        if math.isnan(m_temp_float):
            raise ValueError(f"Manifest fitted temperature for '{raw_m_key}' is NaN.")
        if math.isinf(m_temp_float):
            raise ValueError(f"Manifest fitted temperature for '{raw_m_key}' is infinite: {m_temp_float}")
        if m_temp_float <= 0.0:
            raise ValueError(
                f"Manifest fitted temperature for '{raw_m_key}' must be strictly positive (> 0), got {m_temp_float}"
            )

        manifest_canonical_temps[m_canonical] = m_temp_float

    # Require fitted-temperature entries for all five approved Stage C models
    missing_manifest_models = set(STAGE_C_MODELS) - set(manifest_canonical_temps.keys())
    if missing_manifest_models:
        raise ValueError(
            f"Manifest fitted_temperatures is missing entries for approved Stage C models: {sorted(missing_manifest_models)}"
        )

    # Step 4: Validate calibration JSON structure
    models_dict = calib_data.get("models")
    if not isinstance(models_dict, dict):
        raise ValueError(f"Calibration JSON missing 'models' dictionary at {p_calib}.")

    # Check model completeness: exactly all 5 Stage C models
    raw_model_keys = set(models_dict.keys())
    seen_canonicals: dict[str, str] = {}  # canonical -> raw_key

    for raw_key in raw_model_keys:
        try:
            canonical = normalize_model_name(raw_key)
        except ValueError as exc:
            raise ValueError(f"Unexpected model key '{raw_key}' in calibration JSON: {exc}") from exc

        if canonical not in STAGE_C_MODELS:
            raise ValueError(f"Model '{raw_key}' (canonical: '{canonical}') is not an approved Stage C model.")

        if canonical in seen_canonicals:
            raise ValueError(
                f"Duplicate model resolution in calibration JSON: '{raw_key}' and '{seen_canonicals[canonical]}' "
                f"both resolve to canonical model '{canonical}'."
            )
        seen_canonicals[canonical] = raw_key

    # Check for missing Stage C models
    missing_models = set(STAGE_C_MODELS) - set(seen_canonicals.keys())
    if missing_models:
        raise ValueError(f"Calibration JSON is missing approved Stage C models: {sorted(missing_models)}")

    # Step 5: Check each model record in detail
    temperatures: dict[str, float] = {}
    checkpoint_hashes: dict[str, str] = {}
    display_names: dict[str, str] = {}
    fit_diagnostics: dict[str, dict[str, Any]] = {}

    for canonical in STAGE_C_MODELS:
        raw_key = seen_canonicals[canonical]
        m_record = models_dict[raw_key]
        if not isinstance(m_record, dict):
            raise ValueError(f"Model record for '{raw_key}' must be a dictionary.")

        # Validate internal model_name vs raw_key
        internal_model_name = m_record.get("model_name")
        if internal_model_name is None:
            raise ValueError(f"Model record '{raw_key}' is missing 'model_name' field.")
        if normalize_model_name(str(internal_model_name)) != canonical:
            raise ValueError(
                f"Model key '{raw_key}' resolves to '{canonical}' but internal 'model_name' "
                f"'{internal_model_name}' resolves to '{normalize_model_name(str(internal_model_name))}'."
            )

        # Validate checkpoint hash handshake
        rec_chkpt_hash = m_record.get("checkpoint_sha256")
        if not rec_chkpt_hash or not isinstance(rec_chkpt_hash, str):
            raise ValueError(f"Model record '{raw_key}' missing valid 'checkpoint_sha256'.")
        expected_chkpt_hash = STAGE_C_EXPECTED_HASHES[canonical]
        if rec_chkpt_hash.strip().lower() != expected_chkpt_hash:
            raise ValueError(
                f"Calibration record checkpoint hash mismatch for '{canonical}'!\n"
                f"  Expected: {expected_chkpt_hash}\n"
                f"  Found in JSON: {rec_chkpt_hash}"
            )

        # Cross-validate against manifest checkpoint hash
        manifest_chkpt_hash = manifest_checkpoints.get(raw_key) or manifest_checkpoints.get(canonical)
        if manifest_chkpt_hash is None:
            raise ValueError(f"Manifest missing checkpoint entry for '{raw_key}'.")
        if str(manifest_chkpt_hash).strip().lower() != expected_chkpt_hash:
            raise ValueError(
                f"Manifest checkpoint hash mismatch for '{canonical}'!\n"
                f"  Expected: {expected_chkpt_hash}\n"
                f"  Found in Manifest: {manifest_chkpt_hash}"
            )

        # Validate temperature: optimal_temperature_T
        if "optimal_temperature_T" not in m_record:
            raise ValueError(f"Model record '{raw_key}' missing 'optimal_temperature_T'.")
        raw_temp = m_record["optimal_temperature_T"]

        # Strict type checks: boolean is subclass of int, so MUST explicitly reject bool
        if isinstance(raw_temp, bool) or not isinstance(raw_temp, (int, float)):
            raise ValueError(
                f"Temperature for '{canonical}' must be numeric (int/float), got {type(raw_temp).__name__}: {raw_temp}"
            )

        temp_float = float(raw_temp)
        if math.isnan(temp_float):
            raise ValueError(f"Temperature for '{canonical}' is NaN.")
        if math.isinf(temp_float):
            raise ValueError(f"Temperature for '{canonical}' is infinite: {temp_float}")
        if temp_float <= 0.0:
            raise ValueError(f"Temperature for '{canonical}' must be strictly positive (> 0), got {temp_float}")

        # Validate fit diagnostics (mandatory dictionary in authoritative C2 schema)
        if "fit_diagnostics" not in m_record:
            raise ValueError(f"Model record '{raw_key}' is missing required 'fit_diagnostics' dictionary.")
        diag = m_record["fit_diagnostics"]
        if not isinstance(diag, dict):
            raise ValueError(
                f"Model record '{raw_key}' 'fit_diagnostics' must be a dictionary, got {type(diag).__name__}."
            )

        diag_temp = diag.get("optimal_temperature")
        if diag_temp is None:
            raise ValueError(f"Fit diagnostics for '{canonical}' missing 'optimal_temperature'.")
        if isinstance(diag_temp, bool) or not isinstance(diag_temp, (int, float)):
            raise ValueError(
                f"Fit diagnostic optimal_temperature for '{canonical}' must be numeric, got {type(diag_temp).__name__}."
            )
        diag_float = float(diag_temp)
        if math.isnan(diag_float) or math.isinf(diag_float) or diag_float <= 0.0:
            raise ValueError(f"Fit diagnostic optimal_temperature for '{canonical}' is invalid: {diag_float}")
        if not math.isclose(temp_float, diag_float, rel_tol=TEMPERATURE_COMPARISON_TOLERANCE, abs_tol=TEMPERATURE_COMPARISON_TOLERANCE):
            raise ValueError(
                f"Inconsistent temperatures in '{canonical}': optimal_temperature_T={temp_float} "
                f"vs fit_diagnostics.optimal_temperature={diag_float}"
            )

        converged = diag.get("converged")
        if converged is not True:
            raise ValueError(f"Optimization for '{canonical}' did not converge (converged={converged}).")

        fit_diagnostics[canonical] = dict(diag)

        # Cross-validate temperature against manifest (never silently skip)
        if canonical not in manifest_canonical_temps:
            raise ValueError(f"Manifest missing fitted temperature entry for '{canonical}'.")
        manifest_temp_val = manifest_canonical_temps[canonical]
        if not math.isclose(temp_float, manifest_temp_val, rel_tol=TEMPERATURE_COMPARISON_TOLERANCE, abs_tol=TEMPERATURE_COMPARISON_TOLERANCE):
            raise ValueError(
                f"Temperature mismatch between calibration JSON ({temp_float}) and "
                f"manifest ({manifest_temp_val}) for '{canonical}' "
                f"(tolerance: {TEMPERATURE_COMPARISON_TOLERANCE})."
            )

        temperatures[canonical] = temp_float
        checkpoint_hashes[canonical] = expected_chkpt_hash
        display_names[canonical] = str(m_record.get("display_name", canonical))

    timestamp = str(calib_data.get("timestamp", ""))
    manifest_splits_str = str(manifest_splits_hash or "")

    return StageCCalibrationConfig(
        temperatures=temperatures,
        checkpoint_hashes=checkpoint_hashes,
        display_names=display_names,
        fit_diagnostics=fit_diagnostics,
        calibration_json_sha256=actual_calib_sha256,
        manifest_sha256=actual_manifest_sha256,
        splits_json_sha256=manifest_splits_str,
        timestamp=timestamp,
    )
