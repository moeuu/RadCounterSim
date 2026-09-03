"""Fail-closed material-specific surface-treatment response data."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import yaml

TreatmentDataStatus = Literal["synthetic_validation_only", "experimentally_calibrated"]


@dataclass(frozen=True, slots=True)
class DryContactTreatment:
    rate_constant_s_inv: float
    efficiency_mean: float
    efficiency_std: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.rate_constant_s_inv) or self.rate_constant_s_inv <= 0.0:
            raise ValueError("dry-contact rate_constant_s_inv must be positive")
        if not 0.0 <= self.efficiency_mean <= 1.0:
            raise ValueError("dry-contact efficiency_mean must be in [0, 1]")
        if not math.isfinite(self.efficiency_std) or self.efficiency_std < 0.0:
            raise ValueError("dry-contact efficiency_std must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class WaterJetTreatment:
    removal_coefficient_m2_per_l: float
    washability_mean: float
    washability_std: float
    activity_capture_fraction: float
    runoff_redeposition_fraction: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.removal_coefficient_m2_per_l)
            or self.removal_coefficient_m2_per_l <= 0.0
        ):
            raise ValueError("water removal_coefficient_m2_per_l must be positive")
        if not math.isfinite(self.washability_mean) or self.washability_mean <= 0.0:
            raise ValueError("water washability_mean must be positive")
        if not math.isfinite(self.washability_std) or self.washability_std < 0.0:
            raise ValueError("water washability_std must be finite and nonnegative")
        for name, value in (
            ("activity_capture_fraction", self.activity_capture_fraction),
            ("runoff_redeposition_fraction", self.runoff_redeposition_fraction),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"water {name} must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TreatmentProvenance:
    source: str
    calibration_method: str
    measured_date: str
    units: dict[str, str]

    def __post_init__(self) -> None:
        if not self.source or not self.calibration_method or not self.measured_date:
            raise ValueError("treatment provenance source, method, and date are required")
        required_units = {"rate_constant", "removal_coefficient"}
        if not required_units.issubset(self.units):
            raise ValueError("treatment provenance is missing required units")


@dataclass(frozen=True, slots=True)
class TreatmentMaterialModel:
    model_id: str
    substrate_material_id: str
    status: TreatmentDataStatus
    dry_contact: DryContactTreatment
    water_jet: WaterJetTreatment
    provenance: TreatmentProvenance
    file_sha256: str
    numeric_sha256: str


def _numeric_digest(
    dry_contact: DryContactTreatment,
    water_jet: WaterJetTreatment,
) -> str:
    encoded = json.dumps(
        {
            "dry_contact": asdict(dry_contact),
            "water_jet": asdict(water_jet),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_treatment_material_model(
    path: str | Path,
    *,
    expected_file_sha256: str,
    required_status: TreatmentDataStatus | None = None,
    expected_substrate_material_id: str | None = None,
) -> TreatmentMaterialModel:
    """Load and fully validate one material response file without fallbacks."""

    model_path = Path(path).expanduser().resolve()
    if not expected_file_sha256:
        raise ValueError("treatment model SHA256 is required")
    file_bytes = model_path.read_bytes()
    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
    if file_sha256 != expected_file_sha256:
        raise ValueError("treatment model file SHA256 mismatch")
    payload = yaml.safe_load(file_bytes)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("treatment model requires schema_version 1")
    model_id = str(payload.get("model_id", ""))
    substrate = str(payload.get("substrate_material_id", ""))
    status = str(payload.get("status", ""))
    if not model_id or not substrate:
        raise ValueError("treatment model ID and substrate material ID are required")
    if status not in {"synthetic_validation_only", "experimentally_calibrated"}:
        raise ValueError(f"unsupported treatment data status: {status!r}")
    if required_status is not None and status != required_status:
        raise ValueError(
            f"treatment data status {status!r} does not satisfy {required_status!r}"
        )
    if expected_substrate_material_id is not None and substrate != expected_substrate_material_id:
        raise ValueError(
            f"treatment model substrate {substrate!r} does not match "
            f"{expected_substrate_material_id!r}"
        )
    dry_payload = payload.get("dry_contact")
    water_payload = payload.get("water_jet")
    provenance_payload = payload.get("provenance")
    if not isinstance(dry_payload, dict) or not isinstance(water_payload, dict):
        raise ValueError("treatment model requires dry_contact and water_jet sections")
    if not isinstance(provenance_payload, dict):
        raise ValueError("treatment model requires provenance")
    dry = DryContactTreatment(
        rate_constant_s_inv=float(dry_payload["rate_constant_s_inv"]),
        efficiency_mean=float(dry_payload["efficiency_mean"]),
        efficiency_std=float(dry_payload["efficiency_std"]),
    )
    water = WaterJetTreatment(
        removal_coefficient_m2_per_l=float(
            water_payload["removal_coefficient_m2_per_l"]
        ),
        washability_mean=float(water_payload["washability_mean"]),
        washability_std=float(water_payload["washability_std"]),
        activity_capture_fraction=float(water_payload["activity_capture_fraction"]),
        runoff_redeposition_fraction=float(
            water_payload["runoff_redeposition_fraction"]
        ),
    )
    provenance = TreatmentProvenance(
        source=str(provenance_payload.get("source", "")),
        calibration_method=str(provenance_payload.get("calibration_method", "")),
        measured_date=str(provenance_payload.get("measured_date", "")),
        units={
            str(key): str(value)
            for key, value in dict(provenance_payload.get("units", {})).items()
        },
    )
    return TreatmentMaterialModel(
        model_id=model_id,
        substrate_material_id=substrate,
        status=status,  # type: ignore[arg-type]
        dry_contact=dry,
        water_jet=water,
        provenance=provenance,
        file_sha256=file_sha256,
        numeric_sha256=_numeric_digest(dry, water),
    )
