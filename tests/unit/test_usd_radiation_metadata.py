from pathlib import Path

import numpy as np
import pytest

from radcounter.core.models.state import RevisionState
from radcounter.core.scene import (
    ActivityMapIntegrityError,
    RadiationSceneSnapshot,
    RevisionDelta,
    SourceDescriptor,
    SourceType,
    SurfaceActivityMap,
    UsdRadiationAttributes,
    classify_stage_changes,
)


def _activity_map() -> SurfaceActivityMap:
    return SurfaceActivityMap(
        triangle_indices=np.array([2, 5], dtype=np.int64),
        activity_bq=np.array([100.0, 50.0]),
        cumulative_treatment_exposure=np.zeros(2),
        last_treated_step=np.full(2, -1, dtype=np.int64),
    )


def test_activity_map_round_trip_digest_and_treatment(tmp_path: Path) -> None:
    activity_map = _activity_map()
    path = tmp_path / "floor_activity.npz"
    digest = activity_map.save(path)
    loaded = SurfaceActivityMap.load(
        path.name,
        base_directory=tmp_path,
        expected_sha256=digest,
    )
    removed = loaded.apply_exposure(
        np.array([5]),
        np.array([2.0]),
        rate_constant_s_inv=0.5,
        efficiency=np.array([0.8]),
        simulation_step=12,
    )
    assert removed[0] > 0.0
    assert loaded.activity_bq[1] == pytest.approx(50.0 - removed[0])
    assert loaded.cumulative_treatment_exposure[1] == pytest.approx(2.0)
    assert loaded.last_treated_step[1] == 12


def test_activity_map_rejects_digest_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "map.npz"
    _activity_map().save(path)
    with pytest.raises(ActivityMapIntegrityError, match="SHA256 mismatch"):
        SurfaceActivityMap.load(
            str(path),
            base_directory=tmp_path,
            expected_sha256="0" * 64,
        )


def test_hidden_source_is_excluded_from_estimator_view() -> None:
    visible = SourceDescriptor("/World/Visible", SourceType.POINT, "Cs-137", 10.0)
    hidden = SourceDescriptor(
        "/World/Hidden",
        SourceType.POINT,
        "Co-60",
        20.0,
        hidden_from_estimator=True,
    )
    snapshot = RadiationSceneSnapshot(
        sources={visible.prim_path: visible, hidden.prim_path: hidden}
    )
    assert tuple(snapshot.estimator_visible_sources) == ("/World/Visible",)


def test_stage_change_classification_updates_independent_revisions() -> None:
    delta = classify_stage_changes(
        [
            "/World/Source.xformOp:translate",
            "/World/Shield.rad:material:id",
            "/World/Detector.rad:detector:id",
        ],
        [],
        source_prim_paths=frozenset({"/World/Source"}),
        geometry_prim_paths=frozenset({"/World/Shield"}),
        detector_prim_paths=frozenset({"/World/Detector"}),
    )
    revision = RevisionState()
    delta.apply(revision)
    assert revision.source_pose_revision == 1
    assert revision.geometry_revision == 0
    assert revision.material_revision == 1
    assert revision.detector_revision == 1
    assert revision.source_activity_revision == 0


def test_resync_is_conservative_and_metadata_names_are_canonical() -> None:
    delta = classify_stage_changes(
        [],
        ["/World/RemovedShield"],
        source_prim_paths=frozenset(),
        geometry_prim_paths=frozenset({"/World/RemovedShield"}),
    )
    assert delta == RevisionDelta(
        geometry=True,
        source_activity=True,
        registry_refresh=True,
    )
    assert UsdRadiationAttributes.SOURCE_ACTIVITY_MAP_URI == "rad:source:activityMapUri"
    assert UsdRadiationAttributes.MANIPULATION_REMOVABLE == "rad:manipulation:removable"
