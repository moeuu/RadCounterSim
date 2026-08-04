"""Classify USD path changes into radiation cache invalidation revisions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from radcounter.core.models.state import RevisionState


@dataclass(frozen=True)
class RevisionDelta:
    geometry: bool = False
    material: bool = False
    source_pose: bool = False
    source_activity: bool = False
    detector: bool = False
    registry_refresh: bool = False

    def apply(self, revision: RevisionState) -> None:
        """Apply each invalidation exactly once to mutable revision state."""

        if self.source_pose:
            revision.bump_source_pose(geometry_changed=self.geometry)
        elif self.geometry:
            revision.bump_geometry()
        if self.material:
            revision.bump_material()
        if self.source_activity:
            revision.bump_source_activity()
        if self.detector:
            revision.bump_detector()


def _split_property_path(path: str) -> tuple[str, str]:
    prim_path, separator, property_name = path.rpartition(".")
    return (prim_path, property_name) if separator else (path, "")


def _touches_known_prim(path: str, known_prim_paths: frozenset[str]) -> bool:
    return any(
        path == known
        or path.startswith(f"{known}/")
        or known.startswith(f"{path}/")
        for known in known_prim_paths
    )


def classify_stage_changes(
    changed_info_paths: Iterable[str],
    resynced_paths: Iterable[str],
    *,
    source_prim_paths: frozenset[str],
    geometry_prim_paths: frozenset[str],
    detector_prim_paths: frozenset[str] = frozenset(),
) -> RevisionDelta:
    """Conservatively classify one USD ``ObjectsChanged`` notice."""

    geometry = material = source_pose = source_activity = detector = False
    resynced = tuple(str(path) for path in resynced_paths)
    if resynced:
        geometry = True
        source_activity = True
    for path in resynced:
        prim_path, _ = _split_property_path(path)
        source_pose |= _touches_known_prim(prim_path, source_prim_paths)
        detector |= _touches_known_prim(prim_path, detector_prim_paths)
    for path_value in changed_info_paths:
        prim_path, property_name = _split_property_path(str(path_value))
        if not property_name:
            geometry = True
            continue
        if property_name == "rad:role":
            geometry = source_activity = detector = True
        elif property_name.startswith("rad:material:"):
            material = True
        elif property_name.startswith(("rad:source:", "rad:decon:")):
            source_activity = True
        elif property_name.startswith("rad:detector:"):
            detector = True
        elif property_name.startswith("rad:manipulation:") or property_name.startswith(
            "rad:shield:"
        ):
            geometry = True
        elif property_name.startswith("xformOp:") or property_name == "xformOpOrder":
            source_pose |= _touches_known_prim(prim_path, source_prim_paths)
            geometry |= _touches_known_prim(prim_path, geometry_prim_paths)
            detector |= _touches_known_prim(prim_path, detector_prim_paths)
        elif property_name in {
            "points",
            "faceVertexCounts",
            "faceVertexIndices",
            "extent",
            "orientation",
            "subdivisionScheme",
        }:
            geometry = True
    registry_refresh = bool(resynced) or any(
        ".rad:" in str(path) for path in changed_info_paths
    )
    return RevisionDelta(
        geometry=geometry,
        material=material,
        source_pose=source_pose,
        source_activity=source_activity,
        detector=detector,
        registry_refresh=registry_refresh,
    )
