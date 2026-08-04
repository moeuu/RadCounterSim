"""Rate-limited USD payload streaming around active robots and radiation rays."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from radcounter.core.environment.streaming import (
    EnvironmentStreamingIndex,
    WorkingSetSelector,
)


class LargeEnvironmentRuntime:
    """Maintains a bounded payload working set on the Kit main thread."""

    def __init__(self, stage, environment_prim_path: str = "/World/Environment") -> None:
        self._stage = stage
        self._root_path = environment_prim_path
        root = stage.GetPrimAtPath(environment_prim_path)
        index_asset = root.GetAttribute("rad:stream:indexUri").Get()
        index_uri = getattr(index_asset, "path", str(index_asset))
        if not Path(index_uri).is_absolute():
            layer_path = Path(stage.GetRootLayer().realPath)
            index_uri = str((layer_path.parent / index_uri).resolve())
        self.index = EnvironmentStreamingIndex.load(index_uri)
        self.selector = WorkingSetSelector(self.index)
        self._tile_by_id = {tile.tile_id: tile for tile in self.index.tiles}
        self._prim_by_id = {}
        for prim in root.GetChildren():
            tile_id = prim.GetAttribute("rad:stream:tileId").Get()
            if tile_id:
                self._prim_by_id[str(tile_id)] = prim
        self._loaded: dict[str, int] = {}
        self._focus_points: list[tuple[float, float, float]] = []
        self._radiation_pins: set[str] = set()
        self._subscription = None
        self._frame = 0

    @classmethod
    def discover(cls, stage) -> LargeEnvironmentRuntime | None:
        prim = stage.GetPrimAtPath("/World/Environment")
        if not prim or not prim.IsValid():
            return None
        enabled = prim.GetAttribute("rad:stream:enabled").Get()
        return cls(stage) if enabled else None

    def start(self) -> None:
        import omni.kit.app

        if self._subscription is None:
            stream = omni.kit.app.get_app().get_update_event_stream()
            self._subscription = stream.create_subscription_to_pop(
                self._on_update,
                name="radcounter-large-environment-streaming",
            )

    def stop(self) -> None:
        self._subscription = None

    def set_focus_points(self, points_m: Sequence[Sequence[float]]) -> None:
        self._focus_points = [tuple(float(value) for value in point) for point in points_m]

    def pin_radiation_segments(
        self,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
    ) -> None:
        padding = self.selector.config.radiation_segment_padding_m
        self._radiation_pins = self.index.intersecting_segments(
            origins_m,
            targets_m,
            padding_m=padding,
        )

    def clear_radiation_pins(self) -> None:
        self._radiation_pins.clear()

    @property
    def loaded_tiles(self) -> dict[str, int]:
        return dict(self._loaded)

    @property
    def statistics(self) -> dict[str, int]:
        triangles = sum(
            self._tile_by_id[tile_id].lod(level).triangle_count
            for tile_id, level in self._loaded.items()
        )
        return {
            "loaded_tiles": len(self._loaded),
            "loaded_triangles": triangles,
            "radiation_pinned_tiles": len(self._radiation_pins),
        }

    def update(self) -> None:
        focus = self._focus_points or self._discover_focus_points()
        desired = self.selector.select(
            focus,
            loaded=self._loaded,
            pinned_lod0=self._radiation_pins,
        )
        removals = sorted(set(self._loaded) - set(desired))
        changes = sorted(
            tile_id for tile_id, lod in desired.items() if self._loaded.get(tile_id) != lod
        )
        budget = self.selector.config.max_changes_per_update
        for tile_id in removals[:budget]:
            self._unload(tile_id)
            budget -= 1
        for tile_id in changes[: max(0, budget)]:
            self._load(tile_id, desired[tile_id])

    def _on_update(self, _event) -> None:
        self._frame += 1
        if self._frame % self.selector.config.update_interval_frames == 0:
            self.update()

    def _load(self, tile_id: str, lod: int) -> None:
        from pxr import Sdf

        prim = self._prim_by_id[tile_id]
        if tile_id in self._loaded:
            self._stage.Unload(prim.GetPath())
            prim.GetPayloads().ClearPayloads()
        asset = prim.GetAttribute(f"rad:stream:lod{lod}Uri").Get()
        uri = getattr(asset, "path", str(asset))
        prim.GetPayloads().AddPayload(Sdf.Payload(uri, "/EnvironmentTile"))
        self._stage.Load(prim.GetPath())
        self._loaded[tile_id] = lod

    def _unload(self, tile_id: str) -> None:
        prim = self._prim_by_id[tile_id]
        self._stage.Unload(prim.GetPath())
        prim.GetPayloads().ClearPayloads()
        self._loaded.pop(tile_id, None)

    def _discover_focus_points(self) -> list[tuple[float, float, float]]:
        from pxr import UsdGeom

        points = []
        for prim in self._stage.Traverse():
            is_focus = bool(prim.GetAttribute("rad:stream:focus").Get())
            is_robot = bool(prim.GetAttribute("rad:robot:id").Get())
            if not (is_focus or is_robot):
                continue
            transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            translation = transform.ExtractTranslation()
            points.append((translation[0], translation[1], translation[2]))
        return points


class RadiationTileLease:
    """Pins every tile intersecting finite source-detector segments."""

    def __init__(
        self,
        runtime: LargeEnvironmentRuntime,
        origins_m: np.ndarray,
        targets_m: np.ndarray,
    ) -> None:
        self.runtime = runtime
        self.origins_m = origins_m
        self.targets_m = targets_m

    def __enter__(self) -> LargeEnvironmentRuntime:
        self.runtime.pin_radiation_segments(self.origins_m, self.targets_m)
        self.runtime.update()
        return self.runtime

    def __exit__(self, *_exc_info: object) -> None:
        self.runtime.clear_radiation_pins()
