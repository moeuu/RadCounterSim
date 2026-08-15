# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import radcounter

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
sys.path.insert(0, str(EXTENSION))
extension_namespace = str(EXTENSION / "radcounter")
if extension_namespace not in radcounter.__path__:
    radcounter.__path__.append(extension_namespace)

from radcounter.isaac.robot.physx_telemetry import PhysxManipulationTelemetry


class FakeArticulation:
    def __init__(self) -> None:
        self.names = {"joint_a": 1, "joint_b": 3}
        metadata = type("Metadata", (), {"joint_indices": self.names})()
        self._articulation_view = type("ArticulationView", (), {"_metadata": metadata})()
        self.sample_index = 0

    def get_dof_index(self, name: str) -> int:
        return self.names.get(name, -1)

    def get_measured_joint_efforts(self) -> np.ndarray:
        self.sample_index += 1
        scale = float(self.sample_index)
        return np.asarray((0.0, -2.0 * scale, 0.0, 3.0 * scale))

    def get_measured_joint_forces(self) -> np.ndarray:
        reactions = np.zeros((5, 6), dtype=np.float64)
        reactions[2, :3] = (3.0, 4.0, 0.0)
        reactions[2, 3:] = (0.0, 0.0, 2.0)
        reactions[4, :3] = (0.0, 0.0, 12.0)
        reactions[4, 3:] = (0.0, 5.0, 12.0)
        return reactions


class FakeContactView:
    def get_contact_force_matrix(self, *, dt: float) -> np.ndarray:
        assert dt == 0.02
        return np.asarray((((0.0, 0.0, 9.0),),))

    def get_net_contact_forces(self, *, dt: float) -> np.ndarray:
        assert dt == 0.02
        return np.asarray(((0.0, 0.0, 9.0),))


def test_physx_telemetry_accumulates_effort_reaction_and_contact() -> None:
    telemetry = PhysxManipulationTelemetry(
        FakeArticulation(),
        joint_names=("joint_a", "joint_b"),
        rigid_contact_view=FakeContactView(),
        physics_dt_s=0.02,
    )
    telemetry.sample()
    telemetry.sample()
    summary = telemetry.summary()
    assert summary.samples == 2
    assert summary.peak_abs_joint_effort_nm == (4.0, 6.0)
    assert np.allclose(summary.rms_joint_effort_nm, (np.sqrt(10.0), np.sqrt(22.5)))
    assert summary.peak_joint_reaction_force_n == (5.0, 12.0)
    assert summary.peak_joint_reaction_torque_nm == (2.0, 13.0)
    assert summary.peak_rigid_contact_force_n == 9.0
    assert summary.contact_samples == 2


def test_physx_telemetry_rejects_unknown_joint() -> None:
    try:
        PhysxManipulationTelemetry(FakeArticulation(), joint_names=("missing",))
    except ValueError as error:
        assert "must exist" in str(error)
    else:
        raise AssertionError("unknown joint should be rejected")
