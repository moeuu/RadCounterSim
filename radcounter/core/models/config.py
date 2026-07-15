"""Versioned Pydantic scenario configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EmissionLineConfig(StrictModel):
    energy_keV: float = Field(gt=0)
    photons_per_decay: float = Field(ge=0)


class IsotopeConfig(StrictModel):
    isotope_id: str = Field(min_length=1)
    emission_lines: tuple[EmissionLineConfig, ...] = Field(min_length=1)


class PointSourceConfig(StrictModel):
    source_id: str = Field(min_length=1)
    isotope_id: str = Field(min_length=1)
    position_world_m: tuple[float, float, float]
    activity_bq: float = Field(ge=0)
    enabled: bool = True
    hidden_from_estimator: bool = False


class DetectorConfig(StrictModel):
    detector_id: str = Field(min_length=1)
    energy_bin_edges_keV: tuple[float, ...] = Field(min_length=2)
    efficiency_energy_keV: tuple[float, ...] = Field(min_length=2)
    intrinsic_efficiency: tuple[float, ...] = Field(min_length=2)
    background_cps_per_bin: tuple[float, ...]
    dead_time_s: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> DetectorConfig:
        if tuple(sorted(self.energy_bin_edges_keV)) != self.energy_bin_edges_keV:
            raise ValueError("energy_bin_edges_keV must be increasing")
        if len(self.efficiency_energy_keV) != len(self.intrinsic_efficiency):
            raise ValueError("detector efficiency arrays must match")
        if len(self.background_cps_per_bin) != len(self.energy_bin_edges_keV) - 1:
            raise ValueError("background_cps_per_bin must match energy bins")
        return self


class MeasurementPoseConfig(StrictModel):
    position_world_m: tuple[float, float, float]
    orientation_world_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    duration_s: float = Field(gt=0)


class RuntimeConfig(StrictModel):
    seed: int = Field(ge=0)
    output_directory: str = "outputs"


class RadiationConfig(StrictModel):
    transport_backend: Literal["analytic", "embree"] = "analytic"
    scatter_model: Literal["none", "empirical_buildup", "truth_only_bias"] = "none"
    minimum_distance_m: float = Field(default=0.01, gt=0)


class ScenarioConfig(StrictModel):
    schema_version: Literal["1.0"]
    scenario_id: str = Field(min_length=1)
    runtime: RuntimeConfig
    radiation: RadiationConfig = RadiationConfig()
    isotopes: tuple[IsotopeConfig, ...] = Field(min_length=1)
    point_sources: tuple[PointSourceConfig, ...] = ()
    detector: DetectorConfig
    measurement_poses: tuple[MeasurementPoseConfig, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_references(self) -> ScenarioConfig:
        isotope_ids = {item.isotope_id for item in self.isotopes}
        if len(isotope_ids) != len(self.isotopes):
            raise ValueError("isotope_id values must be unique")
        source_ids = {item.source_id for item in self.point_sources}
        if len(source_ids) != len(self.point_sources):
            raise ValueError("source_id values must be unique")
        missing = {source.isotope_id for source in self.point_sources} - isotope_ids
        if missing:
            raise ValueError(f"source isotope references are missing: {sorted(missing)}")
        if self.radiation.transport_backend == "embree":
            raise ValueError("embree scenarios require the native runtime, not yet available")
        return self


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Load and strictly validate a scenario YAML file."""

    scenario_path = Path(path).expanduser().resolve()
    with scenario_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("scenario root must be a mapping")
    return ScenarioConfig.model_validate(raw)
