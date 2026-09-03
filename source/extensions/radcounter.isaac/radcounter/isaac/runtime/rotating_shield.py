"""Physical rotating-shield measurements through the synchronized USD scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radcounter.core.experiments import EvidenceClass
from radcounter.core.sensors import (
    RotatingShieldConfiguration,
    RotatingShieldMode,
    ShieldProgramMeasurement,
)

from .simulation import IsaacRadiationSimulation, RuntimeDetector

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PhysicalRotatingShieldResult:
    """One auditable posture program executed by direct USD component motion."""

    measurement: ShieldProgramMeasurement
    detector_prim_path: str
    shield_rotation_prim_path: str
    attenuation_geometry_paths: tuple[str, ...]
    changed_geometry_paths_per_posture: tuple[tuple[str, ...], ...]
    evidence_class: EvidenceClass = EvidenceClass.KINEMATIC_SCENE_EDIT


class IsaacPhysicalRotatingShield:
    """Rotate real attenuation geometry and evaluate the shared radiation runtime.

    This controller deliberately authors a scalar USD rotation operation.  Its
    output is therefore component-level ``kinematic_scene_edit`` evidence and
    cannot be relabeled as robot execution evidence.
    """

    def __init__(
        self,
        simulation: IsaacRadiationSimulation,
        configuration: RotatingShieldConfiguration,
        rng: np.random.Generator,
    ) -> None:
        if configuration.mode is not RotatingShieldMode.PHYSICAL_GEOMETRY:
            raise ValueError("Isaac physical rotating shield requires physical_geometry mode")
        binding = configuration.physical_geometry
        if binding is None:
            raise ValueError("physical rotating-shield configuration has no USD binding")
        self.simulation = simulation
        self.configuration = configuration
        self.binding = binding
        self._counter = configuration.create_counter(rng)
        self._changed_per_posture: list[tuple[str, ...]] = []

        rotation_prim = simulation.stage.GetPrimAtPath(binding.shield_rotation_prim_path)
        if not rotation_prim or not rotation_prim.IsValid():
            raise ValueError(
                f"rotating-shield prim does not exist: {binding.shield_rotation_prim_path}"
            )
        rotation_attribute = rotation_prim.GetAttribute(binding.rotation_attribute)
        if not rotation_attribute or not rotation_attribute.HasAuthoredValueOpinion():
            raise ValueError(
                "rotating-shield prim requires an authored scalar rotation attribute: "
                f"{binding.rotation_attribute}"
            )
        initial_angle = rotation_attribute.Get()
        if not isinstance(initial_angle, (int, float)) or not np.isfinite(float(initial_angle)):
            raise ValueError("rotating-shield rotation attribute must hold a finite scalar angle")
        self._rotation_attribute = rotation_attribute

        detector_locations = {
            location.prim_path: location for location in simulation.detectors
        }
        location = detector_locations.get(binding.detector_prim_path)
        if location is None:
            raise ValueError(
                f"rotating-shield detector prim is unavailable: {binding.detector_prim_path}"
            )
        if location.detector_id != configuration.detector.detector_id:
            raise ValueError(
                "rotating-shield detector ID does not match its USD detector binding"
            )
        runtime_detector = simulation.configuration.detectors[location.detector_id]
        self._require_matching_detector(runtime_detector)

        configured_sources = binding.source_prim_paths
        if configured_sources is not None:
            available_sources = {source.prim_path for source in simulation.sources}
            if not set(configured_sources).issubset(available_sources):
                raise ValueError("one or more rotating-shield source prim paths are unavailable")

        geometry_paths = getattr(simulation.transport, "geometry_paths", None)
        if geometry_paths is None:
            raise ValueError(
                "physical rotating-shield mode requires transport with mutable geometry"
            )
        prefix = binding.shield_rotation_prim_path.rstrip("/") + "/"
        attenuation_paths = tuple(
            path
            for path in geometry_paths
            if path == binding.shield_rotation_prim_path or path.startswith(prefix)
        )
        if not attenuation_paths:
            raise ValueError(
                "rotating-shield hierarchy contains no material-tagged attenuation geometry"
            )
        self._attenuation_geometry_paths = attenuation_paths

    def _require_matching_detector(self, runtime: RuntimeDetector) -> None:
        configured = self.configuration.detector
        arrays = (
            (configured.energy_bin_edges_keV, runtime.energy_bin_edges_keV),
            (configured.response_energy_keV, runtime.response_energy_keV),
            (configured.effective_area_m2_per_bin, runtime.effective_area_m2_per_bin),
            (configured.background_cps_per_bin, runtime.background_cps_per_bin),
        )
        if any(
            first.shape != second.shape
            or not np.allclose(first, second, rtol=0.0, atol=1.0e-15)
            for first, second in arrays
        ) or not np.isclose(
            configured.dead_time_s, runtime.dead_time_s, rtol=0.0, atol=1.0e-15
        ):
            raise ValueError(
                "rotating-shield detector response does not match the radiation runtime"
            )

    def _physical_rate(self, angle_deg: float) -> FloatArray:
        previous = float(self._rotation_attribute.Get())
        if not self._rotation_attribute.Set(float(angle_deg)):
            raise RuntimeError("failed to author rotating-shield posture")
        changed = tuple(self.simulation.synchronize())
        relevant = tuple(
            path for path in changed if path in self._attenuation_geometry_paths
        )
        if not np.isclose(previous, angle_deg, rtol=0.0, atol=1.0e-12) and not relevant:
            raise RuntimeError(
                "rotating-shield posture changed without updating attenuation geometry"
            )
        self._changed_per_posture.append(relevant)
        spectra = self.simulation.expected_spectra(
            detector_paths=[self.binding.detector_prim_path],
            source_paths=self.binding.source_prim_paths,
        )
        try:
            return spectra[self.binding.detector_prim_path]
        except KeyError as error:
            raise RuntimeError("rotating-shield detector produced no spectrum") from error

    def measure(self) -> PhysicalRotatingShieldResult:
        """Execute the configured posture sequence against physical scene geometry."""

        initial_angle = float(self._rotation_attribute.Get())
        self._changed_per_posture = []
        try:
            measurement = self._counter.measure_program(
                self.configuration.program,
                physical_rate_provider=self._physical_rate,
            )
        except Exception:
            self._rotation_attribute.Set(initial_angle)
            self.simulation.synchronize()
            raise
        if len(self._changed_per_posture) != len(self.configuration.program.angles_deg):
            raise RuntimeError("rotating-shield program did not evaluate every posture")
        return PhysicalRotatingShieldResult(
            measurement=measurement,
            detector_prim_path=self.binding.detector_prim_path,
            shield_rotation_prim_path=self.binding.shield_rotation_prim_path,
            attenuation_geometry_paths=self._attenuation_geometry_paths,
            changed_geometry_paths_per_posture=tuple(self._changed_per_posture),
        )


def physical_rotating_shield_payload(result: PhysicalRotatingShieldResult) -> dict[str, Any]:
    """Serialize the posture evidence without losing its evidence class."""

    measurement = result.measurement
    return {
        "evidence_class": result.evidence_class.value,
        "mode": measurement.mode.value,
        "detector_prim_path": result.detector_prim_path,
        "shield_rotation_prim_path": result.shield_rotation_prim_path,
        "attenuation_geometry_paths": list(result.attenuation_geometry_paths),
        "changed_geometry_paths_per_posture": [
            list(paths) for paths in result.changed_geometry_paths_per_posture
        ],
        "commanded_angles_deg": measurement.commanded_angles_deg.tolist(),
        "actual_angles_deg": measurement.actual_angles_deg.tolist(),
        "encoder_angles_deg": measurement.encoder_angles_deg.tolist(),
        "dwell_s": measurement.dwell_s.tolist(),
        "expected_counts_per_posture_bin": (
            measurement.expected_counts_per_posture_bin.tolist()
        ),
        "observed_counts_per_posture_bin": (
            measurement.observed_counts_per_posture_bin.tolist()
        ),
    }
