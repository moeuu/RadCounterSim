"""Executable radiation runtime backed by a live USD stage and Embree."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from radcounter.core.models.radiation import MaterialSpec
from radcounter.core.radiation.backend import PathLengthBatch
from radcounter.core.radiation.materials import MaterialTable, interpolate_attenuation_m_inv
from radcounter.core.radiation.sampling import sample_surface_triangles
from radcounter.core.radiation.scatter import (
    PhotonBuildupModel,
    PrimaryOnlyPhotonModel,
    ReferenceCalibratedBuildupModel,
    load_buildup_data,
)
from radcounter.core.scene import SurfaceActivityMap, VolumeActivityMap

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RuntimeLine:
    energy_keV: float
    yield_per_decay: float


@dataclass(frozen=True, slots=True)
class RuntimeDetector:
    detector_id: str
    energy_bin_edges_keV: FloatArray
    response_energy_keV: FloatArray
    effective_area_m2_per_bin: FloatArray
    background_cps_per_bin: FloatArray
    dead_time_s: float

    def __post_init__(self) -> None:
        bin_count = len(self.energy_bin_edges_keV) - 1
        if bin_count < 1 or np.any(np.diff(self.energy_bin_edges_keV) <= 0.0):
            raise ValueError("runtime detector bin edges must be strictly increasing")
        if len(self.response_energy_keV) < 2 or np.any(np.diff(self.response_energy_keV) <= 0.0):
            raise ValueError("runtime detector response energies must be strictly increasing")
        if self.effective_area_m2_per_bin.shape != (len(self.response_energy_keV), bin_count):
            raise ValueError("runtime detector effective-area response dimensions are invalid")
        if self.background_cps_per_bin.shape != (bin_count,):
            raise ValueError("runtime detector background must match output bins")
        if np.any(self.effective_area_m2_per_bin < 0.0) or np.any(
            self.background_cps_per_bin < 0.0
        ):
            raise ValueError("runtime detector response/background must be nonnegative")

    @property
    def energy_bin_count(self) -> int:
        return len(self.energy_bin_edges_keV) - 1

    def effective_area_m2(self, energies_keV: FloatArray) -> FloatArray:
        energies = np.asarray(energies_keV, dtype=np.float64)
        if np.any(energies < self.response_energy_keV[0]) or np.any(
            energies > self.response_energy_keV[-1]
        ):
            raise ValueError(
                f"detector {self.detector_id} response does not cover all incident energies"
            )
        return np.column_stack(
            [
                np.interp(
                    energies,
                    self.response_energy_keV,
                    self.effective_area_m2_per_bin[:, bin_index],
                )
                for bin_index in range(self.energy_bin_count)
            ]
        )


@dataclass(frozen=True, slots=True)
class RuntimeSource:
    prim_path: str
    isotope_id: str
    source_type: str
    positions_m: FloatArray
    activity_bq: FloatArray
    element_indices: IntArray
    hidden_from_estimator: bool

    def __post_init__(self) -> None:
        positions = np.asarray(self.positions_m, dtype=np.float64)
        activity = np.asarray(self.activity_bq, dtype=np.float64)
        elements = np.asarray(self.element_indices, dtype=np.int64)
        if not self.prim_path.startswith("/") or not self.isotope_id:
            raise ValueError("runtime source requires an absolute prim path and isotope ID")
        if self.source_type not in {"point", "surface", "volume"}:
            raise ValueError(f"unsupported runtime source type: {self.source_type!r}")
        if positions.ndim != 2 or positions.shape[1:] != (3,) or len(positions) == 0:
            raise ValueError("runtime source positions must have nonempty shape (N, 3)")
        if activity.shape != (len(positions),) or elements.shape != (len(positions),):
            raise ValueError("runtime source activity/elements must match its sample count")
        if (
            not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(activity))
            or np.any(activity < 0.0)
            or np.any(elements < -1)
        ):
            raise ValueError("runtime source samples must be finite and nonnegative")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "activity_bq", activity)
        object.__setattr__(self, "element_indices", elements)

    @property
    def total_activity_bq(self) -> float:
        return float(np.sum(self.activity_bq))


@dataclass(frozen=True, slots=True)
class DetectorLocation:
    prim_path: str
    detector_id: str
    position_m: FloatArray


@dataclass(frozen=True, slots=True)
class MeasurementRecord:
    detector_path: str
    detector_id: str
    duration_s: float
    energy_bin_edges_keV: FloatArray
    expected_rate_cps_per_bin: FloatArray
    counts_per_bin: IntArray
    buildup_model_name: str = "primary_only"

    @property
    def expected_rate_cps(self) -> float:
        return float(np.sum(self.expected_rate_cps_per_bin))

    @property
    def counts(self) -> int:
        return int(np.sum(self.counts_per_bin))

    @property
    def measured_rate_cps(self) -> float:
        return self.counts / self.duration_s

    def as_dict(self) -> dict[str, object]:
        return {
            "detector_path": self.detector_path,
            "detector_id": self.detector_id,
            "duration_s": self.duration_s,
            "expected_rate_cps": self.expected_rate_cps,
            "counts": self.counts,
            "measured_rate_cps": self.measured_rate_cps,
            "energy_bin_edges_keV": self.energy_bin_edges_keV.tolist(),
            "expected_rate_cps_per_bin": self.expected_rate_cps_per_bin.tolist(),
            "counts_per_bin": self.counts_per_bin.tolist(),
            "buildup_model_name": self.buildup_model_name,
        }


@dataclass(frozen=True, slots=True)
class RuntimeSpectrumPrediction:
    """Auditable primary, corrected, background, and electronics components."""

    total_cps_per_bin: FloatArray
    primary_cps_per_bin: FloatArray
    corrected_source_cps_per_bin: FloatArray
    background_cps_per_bin: FloatArray
    live_fraction: float
    buildup_model_name: str


@dataclass(frozen=True, slots=True)
class RuntimePublicResponse:
    """Truth-independent source templates evaluated through the live scene."""

    detector_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    source_positions_world_m: FloatArray
    source_rate_cps_per_bq: FloatArray
    background_rate_cps: FloatArray
    dead_time_s: FloatArray
    template_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        detector_count = len(self.detector_paths)
        source_count = len(self.source_paths)
        positions = np.asarray(self.source_positions_world_m, dtype=np.float64)
        response = np.asarray(self.source_rate_cps_per_bq, dtype=np.float64)
        background = np.asarray(self.background_rate_cps, dtype=np.float64)
        dead_time = np.asarray(self.dead_time_s, dtype=np.float64)
        if (
            positions.shape != (source_count, 3)
            or response.shape != (detector_count, source_count)
            or background.shape != (detector_count,)
            or dead_time.shape != (detector_count,)
            or len(self.template_kinds) != source_count
            or len(set(self.detector_paths)) != detector_count
            or len(set(self.source_paths)) != source_count
            or not np.all(np.isfinite(positions))
            or not np.all(np.isfinite(response))
            or not np.all(np.isfinite(background))
            or not np.all(np.isfinite(dead_time))
            or np.any(response < 0.0)
            or np.any(background < 0.0)
            or np.any(dead_time < 0.0)
        ):
            raise ValueError("public response dimensions or values are invalid")
        object.__setattr__(self, "source_positions_world_m", positions)
        object.__setattr__(self, "source_rate_cps_per_bq", response)
        object.__setattr__(self, "background_rate_cps", background)
        object.__setattr__(self, "dead_time_s", dead_time)


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    materials: Mapping[str, tuple[FloatArray, FloatArray]]
    isotopes: Mapping[str, tuple[RuntimeLine, ...]]
    detectors: Mapping[str, RuntimeDetector]
    duration_s: float
    minimum_distance_m: float
    seed: int
    buildup_model: PhotonBuildupModel = PrimaryOnlyPhotonModel()

    @classmethod
    def from_json(cls, path: str | Path) -> RuntimeConfiguration:
        config_path = Path(path).expanduser().resolve()
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 2:
            raise ValueError("radiation runtime configuration requires schema_version 2")
        materials = {
            material_id: (
                np.asarray(spec["energy_keV"], dtype=np.float64),
                np.asarray(spec["mu_m_inv"], dtype=np.float64),
            )
            for material_id, spec in payload["materials"].items()
        }
        isotopes = {
            isotope_id: tuple(
                RuntimeLine(float(line["energy_keV"]), float(line["yield_per_decay"]))
                for line in spec["lines"]
            )
            for isotope_id, spec in payload["isotopes"].items()
        }
        detectors = {
            detector_id: RuntimeDetector(
                detector_id=detector_id,
                energy_bin_edges_keV=np.asarray(spec["energy_bin_edges_keV"], dtype=np.float64),
                response_energy_keV=np.asarray(spec["response_energy_keV"], dtype=np.float64),
                effective_area_m2_per_bin=np.asarray(
                    spec["effective_area_m2_per_bin"], dtype=np.float64
                ),
                background_cps_per_bin=np.asarray(spec["background_cps_per_bin"], dtype=np.float64),
                dead_time_s=float(spec["dead_time_s"]),
            )
            for detector_id, spec in payload["detectors"].items()
        }
        measurement = payload["measurement"]
        buildup_payload = payload.get("photon_buildup")
        if not isinstance(buildup_payload, Mapping):
            raise ValueError("radiation runtime configuration requires photon_buildup")
        mode = str(buildup_payload.get("mode", ""))
        if mode == "primary_only":
            buildup_model: PhotonBuildupModel = PrimaryOnlyPhotonModel()
        elif mode == "reference_calibrated_optical_depth_buildup":
            data_path = Path(str(buildup_payload.get("data_path", "")))
            if not data_path.is_absolute():
                data_path = (config_path.parent / data_path).resolve()
            expected_file_sha256 = str(buildup_payload.get("file_sha256", ""))
            if not expected_file_sha256:
                raise ValueError("reference buildup configuration requires file_sha256")
            actual_file_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
            if actual_file_sha256 != expected_file_sha256:
                raise ValueError("buildup data file SHA256 mismatch")
            required_status = (
                None
                if payload.get("physics_data_class") == "synthetic_validation_only"
                else "reference_calibrated_correction"
            )
            loaded_buildup = load_buildup_data(
                data_path,
                required_status=required_status,
            )
            material_table = MaterialTable(
                tuple(
                    MaterialSpec(material_id, energy, attenuation)
                    for material_id, (energy, attenuation) in materials.items()
                )
            )
            buildup_model = ReferenceCalibratedBuildupModel(
                material_table,
                loaded_buildup.surfaces,
            )
        else:
            raise ValueError(f"unsupported photon_buildup mode: {mode!r}")
        return cls(
            materials=materials,
            isotopes=isotopes,
            detectors=detectors,
            duration_s=float(measurement["duration_s"]),
            minimum_distance_m=float(measurement["minimum_distance_m"]),
            seed=int(measurement["seed"]),
            buildup_model=buildup_model,
        )


@dataclass(slots=True)
class _GeometryRecord:
    prim_path: str
    geometry_id: int
    local_vertices: FloatArray
    local_triangles: IntArray
    world_transform: FloatArray
    world_bounds: tuple[FloatArray, FloatArray]
    material_signature: tuple[str, str, float, float]


@dataclass(frozen=True, slots=True)
class _TransportDefinition:
    prim_path: str
    local_vertices: FloatArray
    local_triangles: IntArray
    world_transform: FloatArray
    material_signature: tuple[str, str, float, float]


@dataclass(slots=True)
class _CachedRayPaths:
    origins: FloatArray
    targets: FloatArray
    path_lengths: FloatArray


def _ensure_material_columns(paths: FloatArray, material_count: int) -> FloatArray:
    """Pad native path lengths for configured materials absent from the current stage."""

    values = np.asarray(paths, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("native path lengths must be a two-dimensional array")
    if values.shape[1] > material_count:
        raise ValueError(
            f"native scene returned {values.shape[1]} material columns for "
            f"{material_count} configured materials"
        )
    if values.shape[1] == material_count:
        return values
    return np.pad(values, ((0, 0), (0, material_count - values.shape[1])))


class NativeStageTransport:
    """Mirror attenuation geometry from USD into the native Embree scene."""

    def __init__(self, stage: Any, config: RuntimeConfiguration) -> None:
        native = importlib.import_module("_radcounter_embree")
        self._stage = stage
        self._config = config
        self._material_ids = tuple(config.materials)
        self._material_index = {name: index for index, name in enumerate(self._material_ids)}
        self._scene = native.EmbreeTransportScene()
        self._geometry: dict[str, _GeometryRecord] = {}
        self._ray_cache: dict[bytes, _CachedRayPaths] = {}
        self._statistics = {
            "native_trace_calls": 0,
            "traced_rays": 0,
            "cache_hits": 0,
            "selectively_patched_rays": 0,
        }
        self._build()

    @property
    def geometry_paths(self) -> tuple[str, ...]:
        return tuple(self._geometry)

    @property
    def statistics(self) -> Mapping[str, int]:
        return dict(self._statistics)

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValueOpinion():
            return default
        value = attr.Get()
        return default if value is None else value

    @staticmethod
    def _world_transform(prim: Any) -> FloatArray:
        from pxr import UsdGeom

        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), dtype=np.float64)
        transform = np.ascontiguousarray(matrix.T)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"invalid world transform for {prim.GetPath()}")
        determinant = float(np.linalg.det(transform[:3, :3]))
        if abs(determinant) <= 1.0e-15:
            raise ValueError(f"singular world transform for {prim.GetPath()}")
        return transform

    @staticmethod
    def _world_bounds(vertices: FloatArray, transform: FloatArray) -> tuple[FloatArray, FloatArray]:
        homogeneous = np.column_stack((vertices, np.ones(len(vertices), dtype=np.float64)))
        world = (transform @ homogeneous.T).T[:, :3]
        return world.min(axis=0), world.max(axis=0)

    @staticmethod
    def _union_bounds(
        first: tuple[FloatArray, FloatArray],
        second: tuple[FloatArray, FloatArray],
    ) -> tuple[FloatArray, FloatArray]:
        return np.minimum(first[0], second[0]), np.maximum(first[1], second[1])

    @staticmethod
    def _segments_intersect_bounds(
        origins: FloatArray,
        targets: FloatArray,
        bounds: tuple[FloatArray, FloatArray],
    ) -> NDArray[np.bool_]:
        direction = targets - origins
        lower, upper = bounds
        entry = np.zeros(len(origins), dtype=np.float64)
        exit_ = np.ones(len(origins), dtype=np.float64)
        valid = np.ones(len(origins), dtype=np.bool_)
        for axis in range(3):
            parallel = np.abs(direction[:, axis]) < 1.0e-14
            valid &= ~(
                parallel & ((origins[:, axis] < lower[axis]) | (origins[:, axis] > upper[axis]))
            )
            nonparallel = ~parallel
            inverse = np.zeros(len(origins), dtype=np.float64)
            inverse[nonparallel] = 1.0 / direction[nonparallel, axis]
            first = (lower[axis] - origins[:, axis]) * inverse
            second = (upper[axis] - origins[:, axis]) * inverse
            entry = np.maximum(entry, np.where(nonparallel, np.minimum(first, second), 0.0))
            exit_ = np.minimum(exit_, np.where(nonparallel, np.maximum(first, second), 1.0))
        return valid & (exit_ >= entry)

    @staticmethod
    def _triangulate_mesh(prim: Any) -> tuple[FloatArray, IntArray]:
        from pxr import UsdGeom

        mesh = UsdGeom.Mesh(prim)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int64)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 3:
            raise ValueError(f"USD mesh {prim.GetPath()} has invalid or empty points")
        if not np.all(np.isfinite(vertices)):
            raise ValueError(f"USD mesh {prim.GetPath()} contains non-finite points")
        if counts.ndim != 1 or len(counts) == 0 or np.any(counts < 3):
            raise ValueError(
                f"USD mesh {prim.GetPath()} contains a face with fewer than 3 vertices"
            )
        if int(np.sum(counts)) != len(indices):
            raise ValueError(f"USD mesh {prim.GetPath()} face counts and indices disagree")
        if np.any(indices < 0) or np.any(indices >= len(vertices)):
            raise ValueError(f"USD mesh {prim.GetPath()} has an out-of-range face index")
        triangles: list[tuple[int, int, int]] = []
        offset = 0
        for count in counts:
            face = indices[offset : offset + int(count)]
            triangles.extend(
                (int(face[0]), int(face[i]), int(face[i + 1])) for i in range(1, len(face) - 1)
            )
            offset += int(count)
        triangle_array = np.asarray(triangles, dtype=np.int64)
        points = vertices[triangle_array]
        doubled_area = np.linalg.norm(
            np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1
        )
        if np.any(doubled_area <= 1.0e-15):
            raise ValueError(f"USD mesh {prim.GetPath()} contains a degenerate triangle")
        return np.ascontiguousarray(vertices), np.ascontiguousarray(triangle_array)

    @staticmethod
    def _cube_mesh(prim: Any) -> tuple[FloatArray, IntArray]:
        from pxr import UsdGeom

        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
        if not np.isfinite(half) or half <= 0.0:
            raise ValueError(f"USD cube {prim.GetPath()} requires a positive finite size")
        vertices = np.asarray(
            [
                (-half, -half, -half),
                (half, -half, -half),
                (half, half, -half),
                (-half, half, -half),
                (-half, -half, half),
                (half, -half, half),
                (half, half, half),
                (-half, half, half),
            ],
            dtype=np.float64,
        )
        triangles = np.asarray(
            [
                (0, 2, 1),
                (0, 3, 2),
                (4, 5, 6),
                (4, 6, 7),
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ],
            dtype=np.int64,
        )
        return vertices, triangles

    @staticmethod
    def _sphere_mesh_values(
        radius_m: float, *, radial_segments: int = 64, polar_segments: int = 32
    ) -> tuple[FloatArray, IntArray]:
        """Create a deterministic outward-wound tessellation of a USD sphere."""

        if not np.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("sphere radius must be positive and finite")
        if radial_segments < 8 or polar_segments < 4:
            raise ValueError("sphere tessellation requires >=8 radial and >=4 polar segments")
        vertices: list[tuple[float, float, float]] = [(0.0, 0.0, radius_m)]
        for polar_index in range(1, polar_segments):
            theta = math.pi * polar_index / polar_segments
            radial = radius_m * math.sin(theta)
            z = radius_m * math.cos(theta)
            for radial_index in range(radial_segments):
                phi = 2.0 * math.pi * radial_index / radial_segments
                vertices.append((radial * math.cos(phi), radial * math.sin(phi), z))
        bottom_index = len(vertices)
        vertices.append((0.0, 0.0, -radius_m))

        triangles: list[tuple[int, int, int]] = []
        first_ring = 1
        for radial_index in range(radial_segments):
            next_index = (radial_index + 1) % radial_segments
            triangles.append((0, first_ring + radial_index, first_ring + next_index))
        for polar_index in range(polar_segments - 2):
            upper = 1 + polar_index * radial_segments
            lower = upper + radial_segments
            for radial_index in range(radial_segments):
                next_index = (radial_index + 1) % radial_segments
                triangles.append((upper + radial_index, lower + radial_index, lower + next_index))
                triangles.append((upper + radial_index, lower + next_index, upper + next_index))
        last_ring = 1 + (polar_segments - 2) * radial_segments
        for radial_index in range(radial_segments):
            next_index = (radial_index + 1) % radial_segments
            triangles.append((bottom_index, last_ring + next_index, last_ring + radial_index))
        return np.asarray(vertices, dtype=np.float64), np.asarray(triangles, dtype=np.int64)

    @staticmethod
    def _cylinder_mesh_values(
        radius_m: float,
        height_m: float,
        *,
        radial_segments: int = 64,
        axis: str = "Z",
    ) -> tuple[FloatArray, IntArray]:
        """Create a closed, outward-wound tessellation of a USD cylinder."""

        if not np.isfinite(radius_m) or radius_m <= 0.0:
            raise ValueError("cylinder radius must be positive and finite")
        if not np.isfinite(height_m) or height_m <= 0.0:
            raise ValueError("cylinder height must be positive and finite")
        if radial_segments < 8:
            raise ValueError("cylinder tessellation requires at least 8 radial segments")
        axis = axis.upper()
        if axis not in {"X", "Y", "Z"}:
            raise ValueError("cylinder axis must be X, Y, or Z")
        half_height = height_m / 2.0
        vertices: list[tuple[float, float, float]] = []
        for z in (-half_height, half_height):
            for radial_index in range(radial_segments):
                phi = 2.0 * math.pi * radial_index / radial_segments
                vertices.append((radius_m * math.cos(phi), radius_m * math.sin(phi), z))
        bottom_center = len(vertices)
        top_center = bottom_center + 1
        vertices.extend(((0.0, 0.0, -half_height), (0.0, 0.0, half_height)))
        triangles: list[tuple[int, int, int]] = []
        for radial_index in range(radial_segments):
            next_index = (radial_index + 1) % radial_segments
            bottom = radial_index
            bottom_next = next_index
            top = radial_segments + radial_index
            top_next = radial_segments + next_index
            triangles.extend(
                (
                    (bottom, bottom_next, top_next),
                    (bottom, top_next, top),
                    (bottom_center, bottom_next, bottom),
                    (top_center, top, top_next),
                )
            )
        vertex_array = np.asarray(vertices, dtype=np.float64)
        if axis == "X":
            vertex_array = vertex_array[:, (2, 0, 1)]
        elif axis == "Y":
            vertex_array = vertex_array[:, (1, 2, 0)]
        return np.ascontiguousarray(vertex_array), np.asarray(triangles, dtype=np.int64)

    @classmethod
    def _mesh_for_prim(cls, prim: Any) -> tuple[FloatArray, IntArray]:
        from pxr import UsdGeom

        if prim.IsA(UsdGeom.Mesh):
            return cls._triangulate_mesh(prim)
        elif prim.IsA(UsdGeom.Cube):
            return cls._cube_mesh(prim)
        elif prim.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
            radial_segments = int(cls._attribute(prim, "rad:transport:radialSegments", 64))
            polar_segments = int(cls._attribute(prim, "rad:transport:polarSegments", 32))
            return cls._sphere_mesh_values(
                radius, radial_segments=radial_segments, polar_segments=polar_segments
            )
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get() or 1.0)
            height = float(cylinder.GetHeightAttr().Get() or 2.0)
            axis = str(cylinder.GetAxisAttr().Get() or "Z")
            radial_segments = int(cls._attribute(prim, "rad:transport:radialSegments", 64))
            return cls._cylinder_mesh_values(
                radius, height, radial_segments=radial_segments, axis=axis
            )
        raise ValueError(
            f"radiation material prim {prim.GetPath()} has unsupported USD type "
            f"{prim.GetTypeName()!r}; convert it to Mesh, Cube, Sphere, or Cylinder"
        )

    def _definition(self, prim: Any) -> _TransportDefinition | None:
        from pxr import UsdGeom

        material_attribute = prim.GetAttribute("rad:material:id")
        if not material_attribute or not material_attribute.HasAuthoredValueOpinion():
            return None
        material_id = str(material_attribute.Get() or "")
        if not material_id:
            raise ValueError(f"radiation material prim {prim.GetPath()} has an empty material ID")
        if material_id not in self._material_index:
            raise ValueError(
                f"radiation material prim {prim.GetPath()} references unconfigured "
                f"material {material_id!r}"
            )
        container_only = bool(self._attribute(prim, "rad:material:containerOnly", False))
        if container_only:
            if prim.IsA(UsdGeom.Gprim):
                raise ValueError(
                    f"geometry prim {prim.GetPath()} cannot be marked material:containerOnly"
                )
            tagged_descendant = False
            descendants = list(prim.GetChildren())
            while descendants:
                descendant = descendants.pop()
                descendant_material = descendant.GetAttribute("rad:material:id")
                if (
                    descendant_material
                    and descendant_material.HasAuthoredValueOpinion()
                    and str(descendant_material.Get() or "")
                ):
                    tagged_descendant = True
                    break
                descendants.extend(descendant.GetChildren())
            if not tagged_descendant:
                raise ValueError(
                    f"material container {prim.GetPath()} has no tagged geometry descendant"
                )
            return None
        vertices, triangles = self._mesh_for_prim(prim)
        transform = self._world_transform(prim)
        if float(np.linalg.det(transform[:3, :3])) < 0.0:
            triangles = np.ascontiguousarray(triangles[:, (0, 2, 1)])
        mode = str(self._attribute(prim, "rad:material:mode", "solid"))
        thickness = float(self._attribute(prim, "rad:material:thicknessM", 0.0))
        grazing_floor = float(self._attribute(prim, "rad:material:minGrazingCosine", 0.05))
        if mode not in {"solid", "thin_sheet"}:
            raise ValueError(f"invalid radiation material mode {mode!r} on {prim.GetPath()}")
        if mode == "thin_sheet" and (not np.isfinite(thickness) or thickness <= 0.0):
            raise ValueError(f"thin-sheet prim {prim.GetPath()} requires positive thicknessM")
        if not np.isfinite(grazing_floor) or not 0.0 < grazing_floor <= 1.0:
            raise ValueError(
                f"prim {prim.GetPath()} requires minGrazingCosine in the interval (0, 1]"
            )
        return _TransportDefinition(
            str(prim.GetPath()),
            np.ascontiguousarray(vertices),
            np.ascontiguousarray(triangles),
            transform,
            (material_id, mode, thickness, grazing_floor),
        )

    def _add_definition(self, definition: _TransportDefinition) -> None:
        material_id, mode, thickness, grazing_floor = definition.material_signature
        geometry_id = self._scene.add_triangle_mesh(
            definition.local_vertices,
            definition.local_triangles,
            self._material_index[material_id],
            mode,
            thickness,
            grazing_floor,
        )
        self._scene.update_instance_transform(geometry_id, definition.world_transform)
        self._geometry[definition.prim_path] = _GeometryRecord(
            definition.prim_path,
            geometry_id,
            definition.local_vertices,
            definition.local_triangles,
            definition.world_transform,
            self._world_bounds(definition.local_vertices, definition.world_transform),
            definition.material_signature,
        )

    def _build(self) -> None:
        definitions = tuple(
            definition
            for prim in self._stage.Traverse()
            if (definition := self._definition(prim)) is not None
        )
        for definition in definitions:
            self._add_definition(definition)
        self._scene.commit()

    def synchronize_transforms(self) -> tuple[str, ...]:
        changed: list[str] = []
        changed_bounds: list[tuple[FloatArray, FloatArray]] = []
        definitions = {
            definition.prim_path: definition
            for prim in self._stage.Traverse()
            if (definition := self._definition(prim)) is not None
        }
        for path, record in tuple(self._geometry.items()):
            if path not in definitions:
                self._scene.remove_geometry(record.geometry_id)
                changed_bounds.append(record.world_bounds)
                del self._geometry[path]
                changed.append(path)
        for path, definition in definitions.items():
            record = self._geometry.get(path)
            if record is None:
                self._add_definition(definition)
                changed.append(path)
                changed_bounds.append(self._geometry[path].world_bounds)
                continue
            geometry_changed = (
                definition.local_vertices.shape != record.local_vertices.shape
                or definition.local_triangles.shape != record.local_triangles.shape
                or not np.allclose(
                    definition.local_vertices, record.local_vertices, rtol=0.0, atol=1.0e-12
                )
                or not np.array_equal(definition.local_triangles, record.local_triangles)
            )
            if definition.material_signature != record.material_signature or geometry_changed:
                old_bounds = record.world_bounds
                self._scene.remove_geometry(record.geometry_id)
                del self._geometry[path]
                self._add_definition(definition)
                changed_bounds.append(
                    self._union_bounds(old_bounds, self._geometry[path].world_bounds)
                )
                changed.append(path)
                continue
            transform = definition.world_transform
            if not np.allclose(transform, record.world_transform, rtol=0.0, atol=1.0e-10):
                old_bounds = record.world_bounds
                self._scene.update_instance_transform(record.geometry_id, transform)
                record.world_transform = transform
                record.world_bounds = self._world_bounds(record.local_vertices, transform)
                changed_bounds.append(self._union_bounds(old_bounds, record.world_bounds))
                changed.append(path)
        if changed:
            self._scene.commit()
            self._patch_cached_rays(changed_bounds)
        return tuple(changed)

    @staticmethod
    def _cache_key(origins: FloatArray, targets: FloatArray) -> bytes:
        digest = hashlib.blake2b(digest_size=20)
        digest.update(np.asarray(origins.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(origins, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(targets, dtype=np.float64).tobytes())
        return digest.digest()

    def _trace_paths(self, origins: FloatArray, targets: FloatArray) -> FloatArray:
        self._statistics["native_trace_calls"] += 1
        self._statistics["traced_rays"] += len(origins)
        return np.asarray(self._scene.trace_path_lengths(origins, targets), dtype=np.float64)

    def _patch_cached_rays(self, changed_bounds: Sequence[tuple[FloatArray, FloatArray]]) -> None:
        for cached in self._ray_cache.values():
            affected = np.zeros(len(cached.origins), dtype=np.bool_)
            for bounds in changed_bounds:
                affected |= self._segments_intersect_bounds(cached.origins, cached.targets, bounds)
            if np.any(affected):
                cached.path_lengths[affected] = self._trace_paths(
                    cached.origins[affected], cached.targets[affected]
                )
                self._statistics["selectively_patched_rays"] += int(np.count_nonzero(affected))

    def path_lengths(self, origins: FloatArray, targets: FloatArray) -> FloatArray:
        origins = np.ascontiguousarray(origins, dtype=np.float64)
        targets = np.ascontiguousarray(targets, dtype=np.float64)
        key = self._cache_key(origins, targets)
        cached = self._ray_cache.get(key)
        if cached is not None:
            self._statistics["cache_hits"] += 1
            return cached.path_lengths
        paths = self._trace_paths(origins, targets)
        if paths.ndim == 1:
            paths = paths.reshape(len(origins), -1)
        paths = _ensure_material_columns(paths, len(self._material_ids))
        self._ray_cache[key] = _CachedRayPaths(origins.copy(), targets.copy(), paths)
        while len(self._ray_cache) > 4096:
            self._ray_cache.pop(next(iter(self._ray_cache)))
        return paths

    def trace_path_lengths(self, origins: FloatArray, targets: FloatArray) -> PathLengthBatch:
        """Expose the same cached material paths used by transmission."""

        paths = self.path_lengths(origins, targets)
        return PathLengthBatch(
            self._material_ids,
            paths,
            np.zeros(len(paths), dtype=np.bool_),
        )

    def transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        paths = self.path_lengths(origins, targets)
        attenuation = np.empty((len(self._material_ids), len(energies_keV)), dtype=np.float64)
        for material_index, material_id in enumerate(self._material_ids):
            grid_energy, grid_mu = self._config.materials[material_id]
            attenuation[material_index] = interpolate_attenuation_m_inv(
                energies_keV,
                grid_energy,
                grid_mu,
                material_id=material_id,
            )
        return np.exp(-(paths @ attenuation))

    def trace_transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        return self.transmission(origins, targets, energies_keV)


class VacuumTransport:
    """Explicit no-attenuation transport for portable scene/content tests only."""

    def synchronize_transforms(self) -> tuple[str, ...]:
        return ()

    def transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        return np.ones((len(origins), len(energies_keV)), dtype=np.float64)

    def trace_path_lengths(self, origins: FloatArray, targets: FloatArray) -> PathLengthBatch:
        if np.asarray(origins).shape != np.asarray(targets).shape:
            raise ValueError("vacuum ray endpoints must have matching shapes")
        return PathLengthBatch(
            (),
            np.zeros((len(origins), 0), dtype=np.float64),
            np.zeros(len(origins), dtype=np.bool_),
        )

    def trace_transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        return self.transmission(origins, targets, energies_keV)


class IsaacRadiationSimulation:
    """Truth-side measurement runtime; estimators receive only its observations."""

    def __init__(
        self,
        stage: Any,
        configuration: RuntimeConfiguration,
        *,
        transport: NativeStageTransport | VacuumTransport | None = None,
        require_native: bool = True,
    ) -> None:
        self.stage = stage
        self.configuration = configuration
        self.rng = np.random.default_rng(configuration.seed)
        if transport is None:
            transport = (
                NativeStageTransport(stage, configuration) if require_native else VacuumTransport()
            )
        self.transport = transport
        self.sources: tuple[RuntimeSource, ...] = ()
        self.detectors: tuple[DetectorLocation, ...] = ()
        self.refresh_scene_state()

    @classmethod
    def from_config(
        cls,
        stage: Any,
        config_path: str | Path,
        *,
        require_native: bool = True,
    ) -> IsaacRadiationSimulation:
        return cls(
            stage, RuntimeConfiguration.from_json(config_path), require_native=require_native
        )

    @staticmethod
    def _attribute(prim: Any, name: str, default: object = None) -> object:
        attr = prim.GetAttribute(name)
        if not attr or not attr.HasAuthoredValueOpinion():
            return default
        value = attr.Get()
        return default if value is None else value

    @staticmethod
    def _world_position(prim: Any) -> FloatArray:
        from pxr import Gf, UsdGeom

        value = (
            UsdGeom.XformCache().GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(0.0, 0.0, 0.0))
        )
        return np.asarray(value, dtype=np.float64)

    @staticmethod
    def _world_mesh(prim: Any) -> tuple[FloatArray, IntArray]:
        from pxr import Gf, UsdGeom

        vertices, triangles = NativeStageTransport._triangulate_mesh(prim)
        matrix = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        world = np.asarray(
            [matrix.Transform(Gf.Vec3d(*map(float, point))) for point in vertices],
            dtype=np.float64,
        )
        return world, triangles

    def _source_rng(self, prim_path: str) -> np.random.Generator:
        digest = hashlib.blake2b(prim_path.encode("utf-8"), digest_size=8).digest()
        path_seed = int.from_bytes(digest, "little", signed=False)
        seed = (int(self.configuration.seed) ^ path_seed) % np.iinfo(np.int64).max
        return np.random.default_rng(seed)

    def _activity_base_directory(self) -> Path:
        real_path = str(self.stage.GetRootLayer().realPath or "")
        if not real_path:
            raise ValueError("activity-map sources require a file-backed USD root layer")
        return Path(real_path).resolve().parent

    def _read_surface_source(
        self,
        prim: Any,
        isotope_id: str,
        hidden: bool,
        detector_positions_m: FloatArray,
    ) -> RuntimeSource:
        vertices, triangles = self._world_mesh(prim)
        uri = str(self._attribute(prim, "rad:source:activityMapUri", ""))
        digest = str(self._attribute(prim, "rad:source:activityMapSha256", ""))
        if not digest:
            raise ValueError(f"surface source {prim.GetPath()} requires activityMapSha256")
        activity_map = SurfaceActivityMap.load(
            uri,
            base_directory=self._activity_base_directory(),
            expected_sha256=digest,
        )
        expected_triangles = np.arange(len(triangles), dtype=np.int64)
        if not np.array_equal(np.sort(activity_map.triangle_indices), expected_triangles):
            raise ValueError(
                f"surface source {prim.GetPath()} activity map must cover exactly the "
                "triangles of the visible source mesh"
            )
        selected = triangles[activity_map.triangle_indices]
        mode = str(self._attribute(prim, "rad:source:samplingMode", "centroid"))
        if mode not in {"centroid", "stratified", "adaptive"}:
            raise ValueError(f"surface source {prim.GetPath()} has invalid samplingMode {mode!r}")
        samples_per_triangle = int(self._attribute(prim, "rad:source:samplesPerTriangle", 4))
        maximum_samples = int(self._attribute(prim, "rad:source:maxSamplesPerTriangle", 16))
        sampled = sample_surface_triangles(
            vertices_world_m=vertices,
            triangles=selected,
            activity_bq_per_triangle=activity_map.activity_bq,
            isotope_index=0,
            source_id_index=0,
            mode=mode,  # type: ignore[arg-type]
            samples_per_triangle=samples_per_triangle,
            detector_positions_world_m=(
                detector_positions_m if len(detector_positions_m) else None
            ),
            maximum_samples_per_triangle=maximum_samples,
            rng=self._source_rng(str(prim.GetPath())),
        )
        element_indices = activity_map.triangle_indices[sampled.triangle_index]
        return RuntimeSource(
            prim_path=str(prim.GetPath()),
            isotope_id=isotope_id,
            source_type="surface",
            positions_m=sampled.positions_world_m,
            activity_bq=sampled.activity_bq,
            element_indices=element_indices,
            hidden_from_estimator=hidden,
        )

    @staticmethod
    def _halton(count: int, base: int) -> FloatArray:
        values = np.empty(count, dtype=np.float64)
        for row in range(count):
            index = row + 1
            factor = 1.0
            value = 0.0
            while index:
                factor /= base
                value += factor * (index % base)
                index //= base
            values[row] = value
        return values

    def _read_volume_source(self, prim: Any, isotope_id: str, hidden: bool) -> RuntimeSource:
        from pxr import Gf, UsdGeom

        distribution = str(self._attribute(prim, "rad:source:volumeDistribution", ""))
        if distribution not in {"uniform", "voxel_map"}:
            raise ValueError(
                f"volume source {prim.GetPath()} requires volumeDistribution="
                "'uniform' or 'voxel_map'"
            )
        count = int(self._attribute(prim, "rad:source:sampleCount", 0))
        if distribution == "uniform" and count < 1:
            raise ValueError(f"uniform volume source {prim.GetPath()} requires sampleCount >= 1")
        u = self._halton(count, 2) if distribution == "uniform" else np.empty(0)
        v = self._halton(count, 3) if distribution == "uniform" else np.empty(0)
        w = self._halton(count, 5) if distribution == "uniform" else np.empty(0)
        primitive_parameters: tuple[str, tuple[float, ...]]
        if prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
            primitive_parameters = ("cube", (half,))
            uniform_local = np.column_stack(
                (
                    (2.0 * u - 1.0) * half,
                    (2.0 * v - 1.0) * half,
                    (2.0 * w - 1.0) * half,
                )
            )
        elif prim.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
            primitive_parameters = ("sphere", (radius,))
            radial = radius * np.cbrt(u)
            cosine = 1.0 - 2.0 * v
            sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
            azimuth = 2.0 * math.pi * w
            uniform_local = np.column_stack(
                (radial * sine * np.cos(azimuth), radial * sine * np.sin(azimuth), radial * cosine)
            )
        elif prim.IsA(UsdGeom.Cylinder):
            cylinder = UsdGeom.Cylinder(prim)
            radius = float(cylinder.GetRadiusAttr().Get() or 1.0)
            half_height = float(cylinder.GetHeightAttr().Get() or 2.0) / 2.0
            axis = str(cylinder.GetAxisAttr().Get() or "Z").upper()
            primitive_parameters = ("cylinder", (radius, half_height, float("XYZ".index(axis))))
            radial = radius * np.sqrt(u)
            azimuth = 2.0 * math.pi * v
            uniform_local = np.column_stack(
                (radial * np.cos(azimuth), radial * np.sin(azimuth), (2.0 * w - 1.0) * half_height)
            )
            if axis == "X":
                uniform_local = uniform_local[:, (2, 0, 1)]
            elif axis == "Y":
                uniform_local = uniform_local[:, (1, 2, 0)]
        else:
            raise ValueError(
                f"volume source {prim.GetPath()} has unsupported USD type "
                f"{prim.GetTypeName()!r}; use Cube, Sphere, or Cylinder"
            )

        if distribution == "uniform":
            local = uniform_local
            total_activity = float(self._attribute(prim, "rad:source:activityBq", -1.0))
            if not np.isfinite(total_activity) or total_activity < 0.0:
                raise ValueError(
                    f"uniform volume source {prim.GetPath()} requires nonnegative activityBq"
                )
            activity = np.full(count, total_activity / count, dtype=np.float64)
        else:
            uri = str(self._attribute(prim, "rad:source:activityMapUri", ""))
            digest = str(self._attribute(prim, "rad:source:activityMapSha256", ""))
            volume_map = VolumeActivityMap.load(
                uri,
                base_directory=self._activity_base_directory(),
                expected_sha256=digest,
            )
            local = volume_map.voxel_centers_local_m
            activity = volume_map.activity_bq_per_voxel
            if not self._volume_points_inside(local, primitive_parameters):
                raise ValueError(
                    f"volume source {prim.GetPath()} contains voxel centers outside its visible "
                    "USD primitive"
                )
            count = len(local)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        positions = np.asarray(
            [transform.Transform(Gf.Vec3d(*point)) for point in local], dtype=np.float64
        )
        return RuntimeSource(
            prim_path=str(prim.GetPath()),
            isotope_id=isotope_id,
            source_type="volume",
            positions_m=positions,
            activity_bq=activity,
            element_indices=np.arange(count, dtype=np.int64),
            hidden_from_estimator=hidden,
        )

    @staticmethod
    def _volume_points_inside(
        points: FloatArray, primitive_parameters: tuple[str, tuple[float, ...]]
    ) -> bool:
        shape, parameters = primitive_parameters
        points = np.asarray(points, dtype=np.float64)
        tolerance = 1.0e-10
        if shape == "cube":
            return bool(np.all(np.abs(points) <= parameters[0] + tolerance))
        if shape == "sphere":
            return bool(np.all(np.linalg.norm(points, axis=1) <= parameters[0] + tolerance))
        radius, half_height, axis_index_value = parameters
        axis_index = int(axis_index_value)
        axial = np.abs(points[:, axis_index])
        radial_axes = [index for index in range(3) if index != axis_index]
        radial = np.linalg.norm(points[:, radial_axes], axis=1)
        return bool(
            np.all(axial <= half_height + tolerance) and np.all(radial <= radius + tolerance)
        )

    def refresh_scene_state(self) -> None:
        sources: list[RuntimeSource] = []
        detectors: list[DetectorLocation] = []
        for prim in self.stage.Traverse():
            role = str(self._attribute(prim, "rad:role", ""))
            detector_id = str(self._attribute(prim, "rad:detector:id", ""))
            if role in {"detector", "detector_station"} and detector_id:
                if detector_id not in self.configuration.detectors:
                    raise ValueError(
                        f"detector prim {prim.GetPath()} references unconfigured detector "
                        f"{detector_id!r}"
                    )
                detectors.append(
                    DetectorLocation(str(prim.GetPath()), detector_id, self._world_position(prim))
                )
        detector_positions = np.asarray(
            [detector.position_m for detector in detectors], dtype=np.float64
        ).reshape(len(detectors), 3)
        for prim in self.stage.Traverse():
            source_type = str(self._attribute(prim, "rad:source:type", ""))
            role = str(self._attribute(prim, "rad:role", ""))
            if not source_type:
                if role in {"source", "contaminated_surface"}:
                    raise ValueError(f"source prim {prim.GetPath()} is missing rad:source:type")
                continue
            if source_type not in {"point", "surface", "volume"}:
                raise ValueError(f"source prim {prim.GetPath()} has invalid type {source_type!r}")
            if not bool(self._attribute(prim, "rad:source:enabled", True)):
                continue
            isotope_id = str(self._attribute(prim, "rad:source:isotopeId", ""))
            if isotope_id not in self.configuration.isotopes:
                raise ValueError(
                    f"source prim {prim.GetPath()} references unconfigured isotope {isotope_id!r}"
                )
            hidden = bool(self._attribute(prim, "rad:source:hiddenFromEstimator", False))
            if source_type == "point":
                activity = float(self._attribute(prim, "rad:source:activityBq", -1.0))
                if not np.isfinite(activity) or activity < 0.0:
                    raise ValueError(
                        f"point source {prim.GetPath()} requires nonnegative activityBq"
                    )
                sources.append(
                    RuntimeSource(
                        prim_path=str(prim.GetPath()),
                        isotope_id=isotope_id,
                        source_type="point",
                        positions_m=self._world_position(prim).reshape(1, 3),
                        activity_bq=np.asarray([activity]),
                        element_indices=np.asarray([-1], dtype=np.int64),
                        hidden_from_estimator=hidden,
                    )
                )
            elif source_type == "surface":
                sources.append(
                    self._read_surface_source(prim, isotope_id, hidden, detector_positions)
                )
            else:
                sources.append(self._read_volume_source(prim, isotope_id, hidden))
        self.sources = tuple(sources)
        self.detectors = tuple(detectors)

    @property
    def belief_source_paths(self) -> tuple[str, ...]:
        return tuple(
            source.prim_path for source in self.sources if not source.hidden_from_estimator
        )

    def _linear_source_spectra(
        self,
        source: RuntimeSource,
        activity_bq: FloatArray,
        location: DetectorLocation,
        detector: RuntimeDetector,
    ) -> tuple[FloatArray, FloatArray]:
        """Return primary and buildup-corrected spectra before electronics."""

        activity = np.asarray(activity_bq, dtype=np.float64)
        if activity.shape != source.activity_bq.shape or np.any(activity < 0.0):
            raise ValueError("source template activity must match source samples")
        lines = self.configuration.isotopes[source.isotope_id]
        energies = np.asarray([line.energy_keV for line in lines], dtype=np.float64)
        yields = np.asarray([line.yield_per_decay for line in lines], dtype=np.float64)
        response_m2 = detector.effective_area_m2(energies)
        targets = np.repeat(location.position_m.reshape(1, 3), len(source.positions_m), axis=0)
        displacement = targets - source.positions_m
        distance = np.maximum(
            np.linalg.norm(displacement, axis=1),
            self.configuration.minimum_distance_m,
        )
        transmission = self.transport.transmission(source.positions_m, targets, energies)
        paths = self.transport.trace_path_lengths(source.positions_m, targets)
        buildup = self.configuration.buildup_model.factors(paths, energies)
        if buildup.shape != transmission.shape:
            raise ValueError("runtime buildup and transport dimensions do not match")
        primary_incident_line_rate = (
            activity[:, None]
            * transmission
            * yields[None, :]
            / (4.0 * math.pi * distance[:, None] ** 2)
        )
        primary = np.sum(primary_incident_line_rate @ response_m2, axis=0)
        corrected = np.sum((primary_incident_line_rate * buildup) @ response_m2, axis=0)
        return primary, corrected

    def _public_template_activity(
        self,
        source: RuntimeSource,
    ) -> tuple[FloatArray, FloatArray, str]:
        """Build a one-becquerel prior from public geometry, never activity values."""

        if source.source_type == "point":
            weights = np.full(len(source.positions_m), 1.0 / len(source.positions_m))
            return weights, np.average(source.positions_m, axis=0, weights=weights), "point"
        if source.source_type == "surface":
            prim = self.stage.GetPrimAtPath(source.prim_path)
            if not prim or not prim.IsValid():
                raise ValueError(f"public source prim is unavailable: {source.prim_path}")
            vertices, triangles = self._world_mesh(prim)
            points = vertices[triangles]
            triangle_areas = 0.5 * np.linalg.norm(
                np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
                axis=1,
            )
            if np.any(source.element_indices < 0) or np.any(
                source.element_indices >= len(triangle_areas)
            ):
                raise ValueError("surface template sample indexes are invalid")
            sample_counts = np.bincount(
                source.element_indices,
                minlength=len(triangle_areas),
            )
            if np.any(sample_counts == 0):
                raise ValueError("surface template must sample every visible triangle")
            weights = triangle_areas[source.element_indices] / sample_counts[source.element_indices]
            weights /= np.sum(weights)
            return (
                weights,
                np.average(source.positions_m, axis=0, weights=weights),
                "uniform_area_on_visible_surface",
            )
        prim = self.stage.GetPrimAtPath(source.prim_path)
        distribution = str(self._attribute(prim, "rad:source:volumeDistribution", ""))
        if distribution != "uniform":
            raise ValueError(
                "voxel-map activity support is not a public estimator template; "
                "author a separate public volume prior"
            )
        weights = np.full(len(source.positions_m), 1.0 / len(source.positions_m))
        return (
            weights,
            np.average(source.positions_m, axis=0, weights=weights),
            "uniform_volume_samples",
        )

    def public_source_response(
        self,
        *,
        source_paths: Iterable[str] | None = None,
        detector_paths: Iterable[str] | None = None,
    ) -> RuntimePublicResponse:
        """Evaluate public one-becquerel source templates in the mutable scene."""

        requested_sources = set(self.belief_source_paths if source_paths is None else source_paths)
        selected_sources = [
            source for source in self.sources if source.prim_path in requested_sources
        ]
        if {source.prim_path for source in selected_sources} != requested_sources:
            raise ValueError("one or more public response source paths are unavailable")
        if any(source.hidden_from_estimator for source in selected_sources):
            raise ValueError("hidden sources cannot be used as public response candidates")
        requested_detectors = (
            {location.prim_path for location in self.detectors}
            if detector_paths is None
            else set(detector_paths)
        )
        selected_detectors = [
            location for location in self.detectors if location.prim_path in requested_detectors
        ]
        if {location.prim_path for location in selected_detectors} != requested_detectors:
            raise ValueError("one or more public response detector paths are unavailable")
        if not selected_sources or not selected_detectors:
            raise ValueError("public response requires at least one source and detector")

        templates = [self._public_template_activity(source) for source in selected_sources]
        response = np.empty((len(selected_detectors), len(selected_sources)), dtype=np.float64)
        background = np.empty(len(selected_detectors), dtype=np.float64)
        dead_time = np.empty(len(selected_detectors), dtype=np.float64)
        for detector_index, location in enumerate(selected_detectors):
            detector = self.configuration.detectors[location.detector_id]
            background[detector_index] = float(np.sum(detector.background_cps_per_bin))
            dead_time[detector_index] = detector.dead_time_s
            for source_index, source in enumerate(selected_sources):
                _, corrected = self._linear_source_spectra(
                    source,
                    templates[source_index][0],
                    location,
                    detector,
                )
                response[detector_index, source_index] = float(np.sum(corrected))
        return RuntimePublicResponse(
            detector_paths=tuple(location.prim_path for location in selected_detectors),
            source_paths=tuple(source.prim_path for source in selected_sources),
            source_positions_world_m=np.asarray(
                [template[1] for template in templates], dtype=np.float64
            ),
            source_rate_cps_per_bq=response,
            background_rate_cps=background,
            dead_time_s=dead_time,
            template_kinds=tuple(template[2] for template in templates),
        )

    def synchronize(self) -> tuple[str, ...]:
        changed = self.transport.synchronize_transforms()
        self.refresh_scene_state()
        return changed

    def expected_spectrum_components(
        self,
        *,
        detector_paths: Iterable[str] | None = None,
        source_paths: Iterable[str] | None = None,
    ) -> dict[str, RuntimeSpectrumPrediction]:
        """Return primary, buildup-corrected, background, and dead-time components."""

        detector_filter = None if detector_paths is None else set(detector_paths)
        source_filter = None if source_paths is None else set(source_paths)
        selected_sources = [
            source
            for source in self.sources
            if source_filter is None or source.prim_path in source_filter
        ]
        spectra: dict[str, RuntimeSpectrumPrediction] = {}
        for location in self.detectors:
            if detector_filter is not None and location.prim_path not in detector_filter:
                continue
            detector = self.configuration.detectors[location.detector_id]
            primary_rate = np.zeros(detector.energy_bin_count, dtype=np.float64)
            corrected_source_rate = np.zeros_like(primary_rate)
            for source in selected_sources:
                primary, corrected = self._linear_source_spectra(
                    source,
                    source.activity_bq,
                    location,
                    detector,
                )
                primary_rate += primary
                corrected_source_rate += corrected
            background = detector.background_cps_per_bin.copy()
            total_rate = corrected_source_rate + background
            live_fraction = 1.0
            if detector.dead_time_s > 0:
                live_fraction = 1.0 / (1.0 + np.sum(total_rate) * detector.dead_time_s)
                total_rate *= live_fraction
                primary_rate *= live_fraction
                corrected_source_rate *= live_fraction
                background *= live_fraction
            spectra[location.prim_path] = RuntimeSpectrumPrediction(
                total_rate,
                primary_rate,
                corrected_source_rate,
                background,
                live_fraction,
                self.configuration.buildup_model.model_name,
            )
        return spectra

    def expected_spectra(
        self,
        *,
        detector_paths: Iterable[str] | None = None,
        source_paths: Iterable[str] | None = None,
    ) -> dict[str, FloatArray]:
        return {
            path: prediction.total_cps_per_bin.copy()
            for path, prediction in self.expected_spectrum_components(
                detector_paths=detector_paths,
                source_paths=source_paths,
            ).items()
        }

    def expected_rates(
        self,
        *,
        detector_paths: Iterable[str] | None = None,
        source_paths: Iterable[str] | None = None,
    ) -> dict[str, float]:
        """Return total rates derived from the canonical binned calculation."""

        return {
            path: float(np.sum(rate))
            for path, rate in self.expected_spectra(
                detector_paths=detector_paths,
                source_paths=source_paths,
            ).items()
        }

    def measure(
        self,
        *,
        duration_s: float | None = None,
        detector_paths: Sequence[str] | None = None,
        source_paths: Sequence[str] | None = None,
    ) -> tuple[MeasurementRecord, ...]:
        duration = self.configuration.duration_s if duration_s is None else float(duration_s)
        if duration <= 0:
            raise ValueError("duration_s must be positive")
        spectra = self.expected_spectra(detector_paths=detector_paths, source_paths=source_paths)
        by_path = {detector.prim_path: detector for detector in self.detectors}
        return tuple(
            MeasurementRecord(
                detector_path=path,
                detector_id=by_path[path].detector_id,
                duration_s=duration,
                energy_bin_edges_keV=self.configuration.detectors[
                    by_path[path].detector_id
                ].energy_bin_edges_keV.copy(),
                expected_rate_cps_per_bin=rate.copy(),
                counts_per_bin=self.rng.poisson(rate * duration).astype(np.int64),
                buildup_model_name=self.configuration.buildup_model.model_name,
            )
            for path, rate in spectra.items()
        )

    def dose_proxy_map(
        self, positions_m: FloatArray, detector_id: str = "gamma-counter"
    ) -> FloatArray:
        if detector_id not in self.configuration.detectors:
            raise KeyError(f"unknown detector ID: {detector_id!r}")
        original = self.detectors
        try:
            self.detectors = tuple(
                DetectorLocation(
                    f"/DoseMap/{index}", detector_id, np.asarray(position, dtype=np.float64)
                )
                for index, position in enumerate(np.asarray(positions_m, dtype=np.float64))
            )
            predictions = self.expected_spectrum_components()
            return np.asarray(
                [
                    float(np.sum(predictions[f"/DoseMap/{index}"].corrected_source_cps_per_bin))
                    for index in range(len(positions_m))
                ]
            )
        finally:
            self.detectors = original


def measurement_payload(records: Sequence[MeasurementRecord]) -> dict[str, object]:
    return {"measurements": [record.as_dict() for record in records]}
