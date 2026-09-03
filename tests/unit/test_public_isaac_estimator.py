from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import radcounter

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))
extension_namespace = str(EXTENSION / "radcounter")
if extension_namespace not in radcounter.__path__:
    radcounter.__path__.append(extension_namespace)

from radcounter.isaac.workflow import IsaacPublicPoissonEstimator  # noqa: E402


class _PublicSimulation:
    def public_source_response(self, *, source_paths: object = None) -> object:
        del source_paths
        return SimpleNamespace(
            detector_paths=("/Detector/A", "/Detector/B"),
            source_paths=("/Public/Surface",),
            source_positions_world_m=np.asarray(((1.0, 2.0, 3.0),)),
            source_rate_cps_per_bq=np.asarray(((1.0e-5,), (2.0e-5,))),
            background_rate_cps=np.zeros(2),
            dead_time_s=np.zeros(2),
            template_kinds=("uniform_area_on_visible_surface",),
        )


def _measurement(path: str, counts: int) -> object:
    return SimpleNamespace(
        detector_path=path,
        detector_id="gamma-counter",
        duration_s=10.0,
        counts=counts,
        measured_rate_cps=counts / 10.0,
    )


def test_public_isaac_estimator_uses_only_response_and_public_counts() -> None:
    estimator = IsaacPublicPoissonEstimator(_PublicSimulation())
    belief = estimator(
        (_measurement("/Detector/A", 100), _measurement("/Detector/B", 200)),
        None,
    )
    np.testing.assert_allclose(belief.source_strength_bq, [1.0e6], rtol=1.0e-6)
    assert belief.basis_ids == ("/Public/Surface",)
    assert belief.covariance.shape == (1, 1)
    assert estimator.last_audit is not None
    assert estimator.last_audit["template_kinds"] == ["uniform_area_on_visible_surface"]


def test_public_isaac_estimator_rejects_missing_detector_measurement() -> None:
    estimator = IsaacPublicPoissonEstimator(_PublicSimulation())
    with pytest.raises(ValueError, match="exactly match"):
        estimator((_measurement("/Detector/A", 100),), None)
