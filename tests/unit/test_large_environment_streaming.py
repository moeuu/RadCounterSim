from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radcounter.core.environment.models import EnvironmentStreamingConfig
from radcounter.core.environment.streaming import (
    EnvironmentStreamingIndex,
    LargeEnvironmentBuilder,
    WorkingSetSelector,
    load_tile_lod,
)


@dataclass(frozen=True)
class _Mesh:
    name: str
    vertices: np.ndarray
    faces: np.ndarray
    material_id: str = "concrete"
    collision: bool = True


@dataclass(frozen=True)
class _Scene:
    id: str
    source_digest: str
    meshes: tuple[_Mesh, ...]


def _scene() -> _Scene:
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [20.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [20.0, 1.0, 0.0],
        ]
    )
    return _Scene(
        id="site",
        source_digest="abc",
        meshes=(_Mesh("floor", vertices, np.asarray([[0, 1, 2], [3, 4, 5]])),),
    )


def test_builder_creates_independent_tile_lods(tmp_path: Path) -> None:
    config = EnvironmentStreamingConfig(
        enabled=True,
        tile_size_m=10.0,
        lod_distances_m=(5.0,),
        lod_vertex_cluster_m=(0.0, 0.5),
    )
    path = LargeEnvironmentBuilder().build(_scene(), config, tmp_path)
    index = EnvironmentStreamingIndex.load(path)
    assert len(index.tiles) == 2
    assert all(len(tile.lods) == 2 for tile in index.tiles)
    assert len(load_tile_lod(index.tiles[0].lod(0).uri)) == 1


def test_selector_pins_radiation_segment_at_lod_zero(tmp_path: Path) -> None:
    config = EnvironmentStreamingConfig(
        enabled=True,
        tile_size_m=10.0,
        physics_radius_m=2.0,
        visual_radius_m=3.0,
        lod_distances_m=(2.0,),
        lod_vertex_cluster_m=(0.0, 0.5),
        max_loaded_tiles=1,
        max_loaded_triangles=1,
    )
    path = LargeEnvironmentBuilder().build(_scene(), config, tmp_path)
    index = EnvironmentStreamingIndex.load(path)
    pins = index.intersecting_segments(
        np.asarray([[-1.0, 0.5, 0.0]]),
        np.asarray([[25.0, 0.5, 0.0]]),
    )
    selected = WorkingSetSelector(index).select([[0.0, 0.0, 0.0]], pinned_lod0=pins)
    assert pins <= selected.keys()
    assert set(selected.values()) == {0}
