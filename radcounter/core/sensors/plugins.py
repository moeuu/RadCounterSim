"""Custom detector descriptors, Python factories, and live-reading adapters."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from importlib.metadata import entry_points
from pathlib import Path

import yaml

from .catalog import popular_detector_catalog
from .universal import (
    DeadTimeModel,
    DetectorDescriptor,
    DetectorFamily,
    DetectorModel,
    DetectorOutput,
    DetectorReading,
    Directionality,
    MeasurementRequest,
    ParametricDetectorModel,
    ParticleResponse,
    RadiationType,
    ResponseCurve,
)

DetectorFactory = Callable[[DetectorDescriptor], DetectorModel]


class DetectorRegistry:
    """Registry supporting built-ins, explicit plugins, and Python entry points."""

    ENTRY_POINT_GROUP = "radcounter.detectors"

    def __init__(self, include_popular: bool = True) -> None:
        self._descriptors: dict[str, DetectorDescriptor] = {}
        self._factories: dict[str, DetectorFactory] = {}
        if include_popular:
            for descriptor in popular_detector_catalog().values():
                self.register_descriptor(descriptor)

    def register_descriptor(
        self,
        descriptor: DetectorDescriptor,
        *,
        replace_existing: bool = False,
    ) -> None:
        if descriptor.model_id in self._descriptors and not replace_existing:
            raise ValueError(f"detector model already registered: {descriptor.model_id}")
        self._descriptors[descriptor.model_id] = descriptor

    def register_factory(
        self,
        model_id: str,
        factory: DetectorFactory,
        *,
        replace_existing: bool = False,
    ) -> None:
        if model_id in self._factories and not replace_existing:
            raise ValueError(f"detector factory already registered: {model_id}")
        self._factories[model_id] = factory

    def load_entry_points(self) -> tuple[str, ...]:
        loaded = []
        for item in entry_points(group=self.ENTRY_POINT_GROUP):
            factory = item.load()
            self.register_factory(item.name, factory, replace_existing=True)
            loaded.append(item.name)
        return tuple(loaded)

    @property
    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._descriptors))

    def descriptor(self, model_id: str) -> DetectorDescriptor:
        return self._descriptors[model_id]

    def create(self, model_id: str) -> DetectorModel:
        descriptor = self.descriptor(model_id)
        factory = self._factories.get(model_id, ParametricDetectorModel)
        return factory(descriptor)

    def load_descriptor(
        self,
        path: str | Path,
        *,
        replace_existing: bool = False,
    ) -> DetectorDescriptor:
        descriptor = load_detector_descriptor(path)
        self.register_descriptor(descriptor, replace_existing=replace_existing)
        return descriptor


def load_detector_descriptor(path: str | Path) -> DetectorDescriptor:
    source = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("detector descriptor must be a mapping")
    if payload.get("schema_version") != "2.0":
        raise ValueError("detector descriptor requires schema_version 2.0")
    data = payload.get("detector", payload)
    responses = []
    for response_data in data["particle_responses"]:
        radiation_type = RadiationType(response_data["radiation_type"])
        if "response_csv" in response_data:
            energies, values = _read_response_csv(
                (source.parent / response_data["response_csv"]).resolve(),
                response_data.get("energy_column", "energy_kev"),
                response_data.get("effective_area_column", "effective_area_m2"),
            )
        else:
            energies = tuple(float(value) for value in response_data["energies_kev"])
            values = tuple(float(value) for value in response_data["effective_area_m2"])
        responses.append(
            ParticleResponse(
                radiation_type=radiation_type,
                effective_area_m2=ResponseCurve(energies, values),
            )
        )
    return DetectorDescriptor(
        model_id=str(data["model_id"]),
        display_name=str(data.get("display_name", data["model_id"])),
        family=DetectorFamily(data["family"]),
        directionality=Directionality(data.get("directionality", "omnidirectional")),
        outputs=tuple(
            DetectorOutput(value) for value in data.get("outputs", ["counts", "count_rate"])
        ),
        particle_responses=tuple(responses),
        background_cps=float(data.get("background_cps", 0.0)),
        dead_time_s=float(data.get("dead_time_s", 0.0)),
        dead_time_model=DeadTimeModel(data.get("dead_time_model", "nonparalyzable")),
        maximum_count_rate_cps=(
            float(data["maximum_count_rate_cps"])
            if data.get("maximum_count_rate_cps") is not None
            else None
        ),
        energy_resolution_fwhm_fraction_at_662kev=(
            float(data["energy_resolution_fwhm_fraction_at_662kev"])
            if data.get("energy_resolution_fwhm_fraction_at_662kev") is not None
            else None
        ),
        energy_bin_edges_kev=tuple(float(value) for value in data.get("energy_bin_edges_kev", ())),
        dose_conversion_usv_h_per_count_kev=float(
            data.get("dose_conversion_usv_h_per_count_kev", 0.0)
        ),
        field_of_view_half_angle_deg=float(data.get("field_of_view_half_angle_deg", 180.0)),
        off_axis_leakage_fraction=float(data.get("off_axis_leakage_fraction", 1.0)),
        angular_power=float(data.get("angular_power", 1.0)),
        response_data_status=str(data.get("response_data_status", "")),
        response_provenance=data.get("response_provenance"),
        metadata=data.get("metadata"),
    )


def _read_response_csv(
    path: Path,
    energy_column: str,
    efficiency_column: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    energies, values = [], []
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            energies.append(float(row[energy_column]))
            values.append(float(row[efficiency_column]))
    if not energies:
        raise ValueError(f"empty detector response CSV: {path}")
    order = sorted(range(len(energies)), key=energies.__getitem__)
    return (
        tuple(energies[index] for index in order),
        tuple(values[index] for index in order),
    )


class CallbackDetectorModel:
    """Adapter for custom simulation code, hardware SDKs, or ROS callbacks."""

    def __init__(
        self,
        descriptor: DetectorDescriptor,
        callback: Callable[[MeasurementRequest], DetectorReading],
    ) -> None:
        self.descriptor = descriptor
        self._callback = callback

    def measure(self, request: MeasurementRequest) -> DetectorReading:
        reading = self._callback(request)
        if reading.detector_id != request.pose.detector_id:
            reading = replace(reading, detector_id=request.pose.detector_id)
        return reading


class ExternalDetectorReadingBuffer:
    """Thread-safe latest-value bridge for ROS 2, serial, TCP, or vendor SDKs."""

    def __init__(self, maximum_age_s: float = 2.0) -> None:
        if maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s must be positive")
        self.maximum_age_s = maximum_age_s
        self._lock = threading.Lock()
        self._reading: DetectorReading | None = None
        self._received_at_s = 0.0

    def ingest(self, reading: DetectorReading) -> None:
        with self._lock:
            self._reading = reading
            self._received_at_s = time.monotonic()

    def ingest_mapping(self, payload: Mapping[str, object]) -> None:
        self.ingest(
            DetectorReading(
                detector_id=str(payload["detector_id"]),
                model_id=str(payload["model_id"]),
                integration_time_s=float(payload["integration_time_s"]),
                expected_count_rate_cps=float(payload.get("expected_count_rate_cps", 0.0)),
                observed_counts=int(payload["observed_counts"]),
                spectrum_counts=tuple(int(value) for value in payload.get("spectrum_counts", ())),
                energy_bin_edges_kev=tuple(
                    float(value) for value in payload.get("energy_bin_edges_kev", ())
                ),
                dose_rate_usv_h=float(payload.get("dose_rate_usv_h", 0.0)),
                estimated_direction_world=(
                    tuple(float(value) for value in payload["estimated_direction_world"])
                    if payload.get("estimated_direction_world") is not None
                    else None
                ),
                saturated=bool(payload.get("saturated", False)),
                metadata=payload.get("metadata"),
            )
        )

    def ingest_json(self, value: str) -> None:
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("external detector JSON must be an object")
        self.ingest_mapping(payload)

    def latest(self, detector_id: str | None = None) -> DetectorReading:
        with self._lock:
            reading = self._reading
            age = time.monotonic() - self._received_at_s
        if reading is None:
            raise RuntimeError("external detector has not produced a reading")
        if age > self.maximum_age_s:
            raise RuntimeError(f"external detector reading is stale ({age:.3f} s)")
        return replace(reading, detector_id=detector_id) if detector_id else reading


class ExternalDetectorModel:
    def __init__(
        self,
        descriptor: DetectorDescriptor,
        buffer: ExternalDetectorReadingBuffer,
    ) -> None:
        self.descriptor = descriptor
        self.buffer = buffer

    def measure(self, request: MeasurementRequest) -> DetectorReading:
        return self.buffer.latest(request.pose.detector_id)
