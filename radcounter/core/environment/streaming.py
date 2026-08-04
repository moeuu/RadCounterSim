"""Out-of-core tiling, LOD generation, and working-set selection.

The conversion step may consume the source reader's normal memory footprint,
but the generated cache is independently loadable per tile and per LOD. Runtime
code never needs to materialize the complete normalized environment.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .models import EnvironmentStreamingConfig

_INDEX_SCHEMA = "radcounter.environment-streaming.v1"


@dataclass(frozen=True)
class TileLod:
    level: int
    uri: str
    triangle_count: int
    vertex_count: int


@dataclass(frozen=True)
class EnvironmentTile:
    tile_id: str
    grid: tuple[int, int, int]
    bounds_min_m: tuple[float, float, float]
    bounds_max_m: tuple[float, float, float]
    lods: tuple[TileLod, ...]

    @property
    def center_m(self) -> np.ndarray:
        return (
            np.asarray(self.bounds_min_m, dtype=np.float64)
            + np.asarray(self.bounds_max_m, dtype=np.float64)
        ) * 0.5

    def lod(self, level: int) -> TileLod:
        by_level = {item.level: item for item in self.lods}
        if level in by_level:
            return by_level[level]
        available = sorted(by_level)
        return by_level[min(available, key=lambda value: abs(value - level))]


@dataclass(frozen=True)
class EnvironmentStreamingIndex:
    environment_id: str
    source_digest: str
    tile_size_m: float
    config: Mapping[str, Any]
    tiles: tuple[EnvironmentTile, ...]
    schema: str = _INDEX_SCHEMA

    def save(self, path: str | Path) -> Path:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": self.schema,
            "environment_id": self.environment_id,
            "source_digest": self.source_digest,
            "tile_size_m": self.tile_size_m,
            "config": dict(self.config),
            "tiles": [
                {
                    "tile_id": tile.tile_id,
                    "grid": list(tile.grid),
                    "bounds_min_m": list(tile.bounds_min_m),
                    "bounds_max_m": list(tile.bounds_max_m),
                    "lods": [asdict(lod) for lod in tile.lods],
                }
                for tile in self.tiles
            ],
        }
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: str | Path) -> EnvironmentStreamingIndex:
        source = Path(path).expanduser().resolve()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != _INDEX_SCHEMA:
            raise ValueError(f"Unsupported streaming index: {payload.get('schema')!r}")
        tiles = []
        for item in payload["tiles"]:
            lods = tuple(TileLod(**lod) for lod in item["lods"])
            resolved_lods = tuple(
                TileLod(
                    level=lod.level,
                    uri=str((source.parent / lod.uri).resolve())
                    if not Path(lod.uri).is_absolute()
                    else lod.uri,
                    triangle_count=lod.triangle_count,
                    vertex_count=lod.vertex_count,
                )
                for lod in lods
            )
            tiles.append(
                EnvironmentTile(
                    tile_id=item["tile_id"],
                    grid=tuple(item["grid"]),
                    bounds_min_m=tuple(item["bounds_min_m"]),
                    bounds_max_m=tuple(item["bounds_max_m"]),
                    lods=resolved_lods,
                )
            )
        return cls(
            environment_id=payload["environment_id"],
            source_digest=payload["source_digest"],
            tile_size_m=float(payload["tile_size_m"]),
            config=payload["config"],
            tiles=tuple(tiles),
        )

    def intersecting_segments(
        self,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
        padding_m: float = 0.0,
    ) -> set[str]:
        origins = np.atleast_2d(np.asarray(origins_m, dtype=np.float64))
        targets = np.atleast_2d(np.asarray(targets_m, dtype=np.float64))
        if origins.shape != targets.shape or origins.shape[1] != 3:
            raise ValueError("origins_m and targets_m must both have shape (N, 3)")
        selected: set[str] = set()
        for tile in self.tiles:
            lower = np.asarray(tile.bounds_min_m) - padding_m
            upper = np.asarray(tile.bounds_max_m) + padding_m
            if any(
                _segment_intersects_aabb(a, b, lower, upper)
                for a, b in zip(origins, targets, strict=True)
            ):
                selected.add(tile.tile_id)
        return selected


@dataclass(frozen=True)
class TileMesh:
    name: str
    vertices_m: np.ndarray
    triangles: np.ndarray
    metadata: Mapping[str, Any]


def load_tile_lod(uri: str | Path) -> tuple[TileMesh, ...]:
    with np.load(Path(uri), allow_pickle=False) as archive:
        count = int(np.asarray(archive["mesh_count"]).reshape(-1)[0])
        meshes = []
        for index in range(count):
            raw_metadata = str(np.asarray(archive[f"mesh_{index}_metadata"]).item())
            meshes.append(
                TileMesh(
                    name=str(np.asarray(archive[f"mesh_{index}_name"]).item()),
                    vertices_m=np.asarray(archive[f"mesh_{index}_vertices"], dtype=np.float64),
                    triangles=np.asarray(archive[f"mesh_{index}_triangles"], dtype=np.int64),
                    metadata=json.loads(raw_metadata),
                )
            )
    return tuple(meshes)


class LargeEnvironmentBuilder:
    """Builds independently loadable 3-D tiles and deterministic visual LODs."""

    def build(
        self,
        scene: Any,
        config: EnvironmentStreamingConfig,
        output_directory: str | Path,
    ) -> Path:
        if not config.enabled:
            raise ValueError("Environment streaming must be enabled")
        output = Path(output_directory).expanduser().resolve()
        tile_directory = output / "tiles"
        tile_directory.mkdir(parents=True, exist_ok=True)
        buckets: dict[tuple[int, int, int], list[TileMesh]] = {}

        for mesh_index, mesh in enumerate(scene.meshes):
            vertices = _mesh_vertices(mesh)
            triangles = _mesh_triangles(mesh)
            if len(triangles) == 0:
                continue
            centroids = vertices[triangles].mean(axis=1)
            keys = np.floor(centroids / config.tile_size_m).astype(np.int64)
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            for local_key_index, key_array in enumerate(unique_keys):
                selected = triangles[inverse == local_key_index]
                compact_vertices, compact_triangles = _compact_mesh(vertices, selected)
                key = tuple(int(value) for value in key_array)
                buckets.setdefault(key, []).append(
                    TileMesh(
                        name=f"{_mesh_name(mesh, mesh_index)}_{len(buckets.get(key, [])):03d}",
                        vertices_m=compact_vertices,
                        triangles=compact_triangles,
                        metadata=_mesh_metadata(mesh),
                    )
                )

        tiles: list[EnvironmentTile] = []
        for grid in sorted(buckets):
            shards = _pack_shards(buckets[grid], config.max_tile_triangles)
            for shard_index, shard in enumerate(shards):
                tile_id = f"x{grid[0]:+07d}_y{grid[1]:+07d}_z{grid[2]:+07d}_s{shard_index:03d}"
                lods = []
                for level, cluster_m in enumerate(config.lod_vertex_cluster_m):
                    lod_meshes = (
                        shard
                        if level == 0 or cluster_m <= 0
                        else tuple(_cluster_mesh(mesh, cluster_m) for mesh in shard)
                    )
                    relative = Path("tiles") / f"{tile_id}.lod{level}.npz"
                    _write_tile(output / relative, lod_meshes)
                    lods.append(
                        TileLod(
                            level=level,
                            uri=relative.as_posix(),
                            triangle_count=sum(len(mesh.triangles) for mesh in lod_meshes),
                            vertex_count=sum(len(mesh.vertices_m) for mesh in lod_meshes),
                        )
                    )
                lower = np.asarray(grid, dtype=np.float64) * config.tile_size_m
                upper = lower + config.tile_size_m
                tiles.append(
                    EnvironmentTile(
                        tile_id=tile_id,
                        grid=grid,
                        bounds_min_m=tuple(float(value) for value in lower),
                        bounds_max_m=tuple(float(value) for value in upper),
                        lods=tuple(lods),
                    )
                )

        environment_id = getattr(scene, "environment_id", None)
        if environment_id is None:
            environment_id = scene.id
        source_digest = getattr(scene, "source_sha256", None)
        if source_digest is None:
            source_digest = scene.source_digest
        index = EnvironmentStreamingIndex(
            environment_id=str(environment_id),
            source_digest=str(source_digest),
            tile_size_m=config.tile_size_m,
            config=config.model_dump(mode="json"),
            tiles=tuple(tiles),
        )
        return index.save(output / "streaming-index.json")


class WorkingSetSelector:
    """Selects a bounded tile/LOD working set around robots and sensors."""

    def __init__(self, index: EnvironmentStreamingIndex) -> None:
        self.index = index
        self.config = EnvironmentStreamingConfig.model_validate(index.config)

    def select(
        self,
        focus_points_m: Sequence[Sequence[float]],
        *,
        loaded: Mapping[str, int] | None = None,
        pinned_lod0: Iterable[str] = (),
    ) -> dict[str, int]:
        points = np.asarray(focus_points_m, dtype=np.float64)
        if points.size == 0:
            points = np.asarray([self.config.initial_focus_world_m], dtype=np.float64)
        points = np.atleast_2d(points)
        loaded = loaded or {}
        forced = set(pinned_lod0)
        ranked: list[tuple[int, float, str, int, int]] = []

        for tile in self.index.tiles:
            distance = min(_point_aabb_distance(point, tile) for point in points)
            keep_radius = self.config.visual_radius_m
            if tile.tile_id in loaded:
                keep_radius += self.config.unload_hysteresis_m
            if tile.tile_id in forced:
                lod = 0
                priority = 0
            elif distance <= self.config.physics_radius_m:
                lod = 0
                priority = 1
            elif distance <= keep_radius:
                lod = self._lod_for_distance(distance)
                priority = 2
            else:
                continue
            info = tile.lod(lod)
            ranked.append((priority, distance, tile.tile_id, info.level, info.triangle_count))

        ranked.sort()
        selected: dict[str, int] = {}
        triangles = 0
        for priority, _distance, tile_id, lod, count in ranked:
            is_forced = priority == 0
            if not is_forced and len(selected) >= self.config.max_loaded_tiles:
                continue
            if not is_forced and triangles + count > self.config.max_loaded_triangles:
                continue
            selected[tile_id] = lod
            triangles += count
        return selected

    def _lod_for_distance(self, distance_m: float) -> int:
        level = 0
        for threshold in self.config.lod_distances_m:
            if distance_m >= threshold:
                level += 1
        return min(level, len(self.config.lod_vertex_cluster_m) - 1)


def _mesh_vertices(mesh: Any) -> np.ndarray:
    for name in ("vertices_m", "vertices"):
        if hasattr(mesh, name):
            value = np.asarray(getattr(mesh, name), dtype=np.float64)
            if value.ndim == 2 and value.shape[1] == 3:
                return value
    raise ValueError("Environment mesh does not expose Nx3 vertices")


def _mesh_triangles(mesh: Any) -> np.ndarray:
    for name in ("triangles", "faces"):
        if hasattr(mesh, name):
            value = np.asarray(getattr(mesh, name), dtype=np.int64)
            if value.ndim == 2 and value.shape[1] == 3:
                return value
    raise ValueError("Environment mesh does not expose Mx3 triangles")


def _mesh_name(mesh: Any, fallback: int) -> str:
    return str(getattr(mesh, "name", f"mesh_{fallback:04d}"))


def _mesh_metadata(mesh: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    aliases = {
        "material_id": "material_id",
        "visual": "visual",
        "visual_enabled": "visual",
        "collision": "collision",
        "collision_enabled": "collision",
        "radiation_enabled": "radiation_enabled",
        "source_node": "source_node",
    }
    for name, output_name in aliases.items():
        if hasattr(mesh, name):
            value = getattr(mesh, name)
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[output_name] = value
    extra = getattr(mesh, "metadata", None)
    if isinstance(extra, Mapping):
        metadata.update(
            {
                str(key): value
                for key, value in extra.items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
        )
    return metadata


def _compact_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    used, inverse = np.unique(triangles.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape((-1, 3)).astype(np.int64)


def _pack_shards(
    meshes: Sequence[TileMesh],
    maximum_triangles: int,
) -> tuple[tuple[TileMesh, ...], ...]:
    pieces: list[TileMesh] = []
    for mesh in meshes:
        for start in range(0, len(mesh.triangles), maximum_triangles):
            vertices, triangles = _compact_mesh(
                mesh.vertices_m,
                mesh.triangles[start : start + maximum_triangles],
            )
            pieces.append(TileMesh(mesh.name, vertices, triangles, mesh.metadata))
    shards: list[list[TileMesh]] = []
    current: list[TileMesh] = []
    count = 0
    for piece in pieces:
        piece_count = len(piece.triangles)
        if current and count + piece_count > maximum_triangles:
            shards.append(current)
            current = []
            count = 0
        current.append(piece)
        count += piece_count
    if current:
        shards.append(current)
    return tuple(tuple(shard) for shard in shards)


def _cluster_mesh(mesh: TileMesh, cell_m: float) -> TileMesh:
    if len(mesh.vertices_m) == 0:
        return mesh
    keys = np.floor(mesh.vertices_m / cell_m + 0.5).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    clustered = np.zeros((int(inverse.max()) + 1, 3), dtype=np.float64)
    counts = np.bincount(inverse)
    np.add.at(clustered, inverse, mesh.vertices_m)
    clustered /= counts[:, None]
    triangles = inverse[mesh.triangles]
    valid = (
        (triangles[:, 0] != triangles[:, 1])
        & (triangles[:, 1] != triangles[:, 2])
        & (triangles[:, 0] != triangles[:, 2])
    )
    triangles = triangles[valid]
    if len(triangles):
        canonical = np.sort(triangles, axis=1)
        _, keep = np.unique(canonical, axis=0, return_index=True)
        triangles = triangles[np.sort(keep)]
        clustered, triangles = _compact_mesh(clustered, triangles)
    return TileMesh(mesh.name, clustered, triangles, mesh.metadata)


def _write_tile(path: Path, meshes: Sequence[TileMesh]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"mesh_count": np.asarray([len(meshes)], dtype=np.int32)}
    for index, mesh in enumerate(meshes):
        payload[f"mesh_{index}_name"] = np.asarray(mesh.name)
        payload[f"mesh_{index}_vertices"] = np.asarray(mesh.vertices_m, dtype=np.float32)
        payload[f"mesh_{index}_triangles"] = np.asarray(mesh.triangles, dtype=np.int32)
        payload[f"mesh_{index}_metadata"] = np.asarray(
            json.dumps(dict(mesh.metadata), sort_keys=True)
        )
    np.savez_compressed(path, **payload)


def _point_aabb_distance(point: np.ndarray, tile: EnvironmentTile) -> float:
    lower = np.asarray(tile.bounds_min_m)
    upper = np.asarray(tile.bounds_max_m)
    delta = np.maximum(np.maximum(lower - point, point - upper), 0.0)
    return float(np.linalg.norm(delta))


def _segment_intersects_aabb(
    origin: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> bool:
    direction = target - origin
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        if math.isclose(float(direction[axis]), 0.0, abs_tol=1e-12):
            if origin[axis] < lower[axis] or origin[axis] > upper[axis]:
                return False
            continue
        inverse = 1.0 / direction[axis]
        near = (lower[axis] - origin[axis]) * inverse
        far = (upper[axis] - origin[axis]) * inverse
        if near > far:
            near, far = far, near
        t_min = max(t_min, float(near))
        t_max = min(t_max, float(far))
        if t_min > t_max:
            return False
    return True
