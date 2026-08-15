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

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class RuntimeLine:
    energy_keV: float
    yield_per_decay: float


@dataclass(frozen=True, slots=True)
class RuntimeDetector:
    detector_id: str
    efficiency_energy_keV: FloatArray
    intrinsic_efficiency: FloatArray
    background_cps: float
    dead_time_s: float

    def efficiency(self, energies_keV: FloatArray) -> FloatArray:
        return np.interp(
            energies_keV,
            self.efficiency_energy_keV,
            self.intrinsic_efficiency,
            left=self.intrinsic_efficiency[0],
            right=self.intrinsic_efficiency[-1],
        )


@dataclass(frozen=True, slots=True)
class RuntimeSource:
    prim_path: str
    isotope_id: str
    positions_m: FloatArray
    activity_bq: FloatArray
    hidden_from_estimator: bool


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
    expected_rate_cps: float
    counts: int

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
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfiguration:
    materials: Mapping[str, tuple[FloatArray, FloatArray]]
    isotopes: Mapping[str, tuple[RuntimeLine, ...]]
    detectors: Mapping[str, RuntimeDetector]
    duration_s: float
    minimum_distance_m: float
    seed: int

    @classmethod
    def from_json(cls, path: str | Path) -> RuntimeConfiguration:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
                efficiency_energy_keV=np.asarray(spec["efficiency_energy_keV"], dtype=np.float64),
                intrinsic_efficiency=np.asarray(spec["intrinsic_efficiency"], dtype=np.float64),
                background_cps=float(spec["background_cps"]),
                dead_time_s=float(spec["dead_time_s"]),
            )
            for detector_id, spec in payload["detectors"].items()
        }
        measurement = payload["measurement"]
        return cls(
            materials=materials,
            isotopes=isotopes,
            detectors=detectors,
            duration_s=float(measurement["duration_s"]),
            minimum_distance_m=float(measurement["minimum_distance_m"]),
            seed=int(measurement["seed"]),
        )


@dataclass(slots=True)
class _GeometryRecord:
    prim_path: str
    geometry_id: int
    local_vertices: FloatArray
    world_transform: FloatArray
    world_bounds: tuple[FloatArray, FloatArray]
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
        return np.ascontiguousarray(matrix.T)

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
        triangles: list[tuple[int, int, int]] = []
        offset = 0
        for count in counts:
            face = indices[offset : offset + int(count)]
            triangles.extend(
                (int(face[0]), int(face[i]), int(face[i + 1])) for i in range(1, len(face) - 1)
            )
            offset += int(count)
        return np.ascontiguousarray(vertices), np.asarray(triangles, dtype=np.int64)

    @staticmethod
    def _cube_mesh(prim: Any) -> tuple[FloatArray, IntArray]:
        from pxr import UsdGeom

        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
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

    def _add_prim(self, prim: Any) -> None:
        from pxr import UsdGeom

        material_id = str(self._attribute(prim, "rad:material:id", ""))
        if material_id not in self._material_index:
            return
        if prim.IsA(UsdGeom.Mesh):
            vertices, triangles = self._triangulate_mesh(prim)
        elif prim.IsA(UsdGeom.Cube):
            vertices, triangles = self._cube_mesh(prim)
        else:
            return
        mode = str(self._attribute(prim, "rad:material:mode", "solid"))
        thickness = float(self._attribute(prim, "rad:material:thicknessM", 0.0))
        grazing_floor = float(self._attribute(prim, "rad:material:minGrazingCosine", 0.05))
        geometry_id = self._scene.add_triangle_mesh(
            vertices,
            triangles,
            self._material_index[material_id],
            mode,
            thickness,
            grazing_floor,
        )
        transform = self._world_transform(prim)
        self._scene.update_instance_transform(geometry_id, transform)
        path = str(prim.GetPath())
        self._geometry[path] = _GeometryRecord(
            path,
            geometry_id,
            vertices,
            transform,
            self._world_bounds(vertices, transform),
            (material_id, mode, thickness, grazing_floor),
        )

    def _build(self) -> None:
        for prim in self._stage.Traverse():
            self._add_prim(prim)
        self._scene.commit()

    def synchronize_transforms(self) -> tuple[str, ...]:
        changed: list[str] = []
        changed_bounds: list[tuple[FloatArray, FloatArray]] = []
        for path, record in tuple(self._geometry.items()):
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                self._scene.remove_geometry(record.geometry_id)
                changed_bounds.append(record.world_bounds)
                del self._geometry[path]
                changed.append(path)
                continue
            signature = (
                str(self._attribute(prim, "rad:material:id", "")),
                str(self._attribute(prim, "rad:material:mode", "solid")),
                float(self._attribute(prim, "rad:material:thicknessM", 0.0)),
                float(self._attribute(prim, "rad:material:minGrazingCosine", 0.05)),
            )
            if signature != record.material_signature:
                old_bounds = record.world_bounds
                self._scene.remove_geometry(record.geometry_id)
                del self._geometry[path]
                self._add_prim(prim)
                changed_bounds.append(
                    self._union_bounds(old_bounds, self._geometry[path].world_bounds)
                )
                changed.append(path)
                continue
            transform = self._world_transform(prim)
            if not np.allclose(transform, record.world_transform, rtol=0.0, atol=1.0e-10):
                old_bounds = record.world_bounds
                self._scene.update_instance_transform(record.geometry_id, transform)
                record.world_transform = transform
                record.world_bounds = self._world_bounds(record.local_vertices, transform)
                changed_bounds.append(self._union_bounds(old_bounds, record.world_bounds))
                changed.append(path)
        for prim in self._stage.Traverse():
            path = str(prim.GetPath())
            material_id = str(self._attribute(prim, "rad:material:id", ""))
            if path not in self._geometry and material_id in self._material_index:
                self._add_prim(prim)
                if path in self._geometry:
                    changed.append(path)
                    changed_bounds.append(self._geometry[path].world_bounds)
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

    def transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        paths = self.path_lengths(origins, targets)
        attenuation = np.empty((len(self._material_ids), len(energies_keV)), dtype=np.float64)
        for material_index, material_id in enumerate(self._material_ids):
            grid_energy, grid_mu = self._config.materials[material_id]
            attenuation[material_index] = np.interp(
                energies_keV,
                grid_energy,
                grid_mu,
                left=grid_mu[0],
                right=grid_mu[-1],
            )
        return np.exp(-(paths @ attenuation))


class VacuumTransport:
    """Explicit no-attenuation transport for portable scene/content tests only."""

    def synchronize_transforms(self) -> tuple[str, ...]:
        return ()

    def transmission(
        self, origins: FloatArray, targets: FloatArray, energies_keV: FloatArray
    ) -> FloatArray:
        return np.ones((len(origins), len(energies_keV)), dtype=np.float64)


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

    def _asset_path(self, uri: str) -> Path:
        root_layer = Path(self.stage.GetRootLayer().realPath).resolve()
        return (root_layer.parent / uri).resolve()

    def _read_surface_source(self, prim: Any, isotope_id: str, hidden: bool) -> RuntimeSource:
        vertices, triangles = self._world_mesh(prim)
        uri = str(self._attribute(prim, "rad:source:activityMapUri", ""))
        digest = str(self._attribute(prim, "rad:source:activityMapSha256", ""))
        path = self._asset_path(uri)
        import hashlib

        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest and actual_digest != digest:
            raise ValueError(f"activity-map digest mismatch for {path}")
        with np.load(path, allow_pickle=False) as payload:
            triangle_indices = np.asarray(payload["triangle_indices"], dtype=np.int64)
            activity = np.asarray(payload["activity_bq"], dtype=np.float64)
        selected = triangles[triangle_indices]
        positions = vertices[selected].mean(axis=1)
        return RuntimeSource(str(prim.GetPath()), isotope_id, positions, activity, hidden)

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

        count = max(1, int(self._attribute(prim, "rad:source:sampleCount", 128)))
        u = self._halton(count, 2)
        v = self._halton(count, 3)
        w = self._halton(count, 5)
        if prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0) / 2.0
            local = np.column_stack(
                ((2.0 * u - 1.0) * half, (2.0 * v - 1.0) * half, (2.0 * w - 1.0) * half)
            )
        elif prim.IsA(UsdGeom.Sphere):
            radius = float(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
            radial = radius * np.cbrt(u)
            cosine = 1.0 - 2.0 * v
            sine = np.sqrt(np.maximum(0.0, 1.0 - cosine**2))
            azimuth = 2.0 * math.pi * w
            local = np.column_stack(
                (radial * sine * np.cos(azimuth), radial * sine * np.sin(azimuth), radial * cosine)
            )
        elif prim.IsA(UsdGeom.Cylinder):
            radius = float(UsdGeom.Cylinder(prim).GetRadiusAttr().Get() or 1.0)
            half_height = float(UsdGeom.Cylinder(prim).GetHeightAttr().Get() or 2.0) / 2.0
            radial = radius * np.sqrt(u)
            azimuth = 2.0 * math.pi * v
            local = np.column_stack(
                (radial * np.cos(azimuth), radial * np.sin(azimuth), (2.0 * w - 1.0) * half_height)
            )
        else:
            size = np.asarray(
                self._attribute(prim, "rad:source:volumeSizeM", (1.0, 1.0, 1.0)), dtype=np.float64
            )
            local = (np.column_stack((u, v, w)) - 0.5) * size
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        positions = np.asarray(
            [transform.Transform(Gf.Vec3d(*point)) for point in local], dtype=np.float64
        )
        total_activity = float(self._attribute(prim, "rad:source:activityBq", 0.0))
        return RuntimeSource(
            str(prim.GetPath()),
            isotope_id,
            positions,
            np.full(count, total_activity / count, dtype=np.float64),
            hidden,
        )

    def refresh_scene_state(self) -> None:
        sources: list[RuntimeSource] = []
        detectors: list[DetectorLocation] = []
        for prim in self.stage.Traverse():
            if not bool(self._attribute(prim, "rad:source:enabled", True)):
                continue
            source_type = str(self._attribute(prim, "rad:source:type", ""))
            isotope_id = str(self._attribute(prim, "rad:source:isotopeId", ""))
            hidden = bool(self._attribute(prim, "rad:source:hiddenFromEstimator", False))
            if source_type == "point" and isotope_id:
                sources.append(
                    RuntimeSource(
                        str(prim.GetPath()),
                        isotope_id,
                        self._world_position(prim).reshape(1, 3),
                        np.asarray([float(self._attribute(prim, "rad:source:activityBq", 0.0))]),
                        hidden,
                    )
                )
            elif source_type == "surface" and isotope_id:
                sources.append(self._read_surface_source(prim, isotope_id, hidden))
            elif source_type == "volume" and isotope_id:
                sources.append(self._read_volume_source(prim, isotope_id, hidden))
            role = str(self._attribute(prim, "rad:role", ""))
            detector_id = str(self._attribute(prim, "rad:detector:id", ""))
            if role in {"detector", "detector_station"} and detector_id:
                detectors.append(
                    DetectorLocation(str(prim.GetPath()), detector_id, self._world_position(prim))
                )
        self.sources = tuple(sources)
        self.detectors = tuple(detectors)

    @property
    def belief_source_paths(self) -> tuple[str, ...]:
        return tuple(
            source.prim_path for source in self.sources if not source.hidden_from_estimator
        )

    def synchronize(self) -> tuple[str, ...]:
        changed = self.transport.synchronize_transforms()
        self.refresh_scene_state()
        return changed

    def expected_rates(
        self,
        *,
        detector_paths: Iterable[str] | None = None,
        source_paths: Iterable[str] | None = None,
    ) -> dict[str, float]:
        detector_filter = None if detector_paths is None else set(detector_paths)
        source_filter = None if source_paths is None else set(source_paths)
        selected_sources = [
            source
            for source in self.sources
            if source_filter is None or source.prim_path in source_filter
        ]
        rates: dict[str, float] = {}
        for location in self.detectors:
            if detector_filter is not None and location.prim_path not in detector_filter:
                continue
            detector = self.configuration.detectors[location.detector_id]
            total_rate = detector.background_cps
            for source in selected_sources:
                lines = self.configuration.isotopes[source.isotope_id]
                energies = np.asarray([line.energy_keV for line in lines], dtype=np.float64)
                yields = np.asarray([line.yield_per_decay for line in lines], dtype=np.float64)
                targets = np.repeat(
                    location.position_m.reshape(1, 3), len(source.positions_m), axis=0
                )
                displacement = targets - source.positions_m
                distance = np.maximum(
                    np.linalg.norm(displacement, axis=1), self.configuration.minimum_distance_m
                )
                transmission = self.transport.transmission(source.positions_m, targets, energies)
                line_response = transmission @ (yields * detector.efficiency(energies))
                total_rate += float(
                    np.sum(source.activity_bq * line_response / (4.0 * math.pi * distance**2))
                )
            if detector.dead_time_s > 0:
                total_rate = total_rate / (1.0 + total_rate * detector.dead_time_s)
            rates[location.prim_path] = total_rate
        return rates

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
        rates = self.expected_rates(detector_paths=detector_paths, source_paths=source_paths)
        by_path = {detector.prim_path: detector for detector in self.detectors}
        return tuple(
            MeasurementRecord(
                detector_path=path,
                detector_id=by_path[path].detector_id,
                duration_s=duration,
                expected_rate_cps=rate,
                counts=int(self.rng.poisson(rate * duration)),
            )
            for path, rate in rates.items()
        )

    def dose_proxy_map(
        self, positions_m: FloatArray, detector_id: str = "gamma-counter"
    ) -> FloatArray:
        detector = self.configuration.detectors[detector_id]
        original = self.detectors
        try:
            self.detectors = tuple(
                DetectorLocation(
                    f"/DoseMap/{index}", detector_id, np.asarray(position, dtype=np.float64)
                )
                for index, position in enumerate(np.asarray(positions_m, dtype=np.float64))
            )
            rates = self.expected_rates()
            return np.asarray(
                [
                    rates[f"/DoseMap/{index}"] - detector.background_cps
                    for index in range(len(positions_m))
                ]
            )
        finally:
            self.detectors = original


def measurement_payload(records: Sequence[MeasurementRecord]) -> dict[str, object]:
    return {"measurements": [record.as_dict() for record in records]}
