"""PhysX joint-effort, joint-reaction, and rigid-contact telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


def _numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PhysxManipulationTelemetrySummary:
    samples: int
    joint_names: tuple[str, ...]
    peak_abs_joint_effort_nm: tuple[float, ...]
    rms_joint_effort_nm: tuple[float, ...]
    peak_joint_reaction_force_n: tuple[float, ...]
    peak_joint_reaction_torque_nm: tuple[float, ...]
    peak_rigid_contact_force_n: float
    contact_samples: int

    @property
    def overall_peak_abs_joint_effort_nm(self) -> float:
        return max(self.peak_abs_joint_effort_nm, default=0.0)

    @property
    def overall_peak_joint_reaction_force_n(self) -> float:
        return max(self.peak_joint_reaction_force_n, default=0.0)

    @property
    def overall_peak_joint_reaction_torque_nm(self) -> float:
        return max(self.peak_joint_reaction_torque_nm, default=0.0)


class PhysxManipulationTelemetry:
    """Accumulate solver measurements without coupling to a particular robot wrapper."""

    def __init__(
        self,
        articulation: Any,
        *,
        joint_names: Sequence[str],
        rigid_contact_view: Any | None = None,
        physics_dt_s: float = 1.0 / 60.0,
        contact_threshold_n: float = 0.1,
    ) -> None:
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if contact_threshold_n < 0.0:
            raise ValueError("contact_threshold_n must be nonnegative")
        self.articulation = articulation
        self.joint_names = tuple(joint_names)
        if not self.joint_names:
            raise ValueError("at least one joint name is required")
        self.joint_indices = np.asarray(
            [articulation.get_dof_index(name) for name in self.joint_names],
            dtype=np.int64,
        )
        if np.any(self.joint_indices < 0):
            raise ValueError("all requested joints must exist in the articulation")
        # The measured-effort vector is indexed by DOF, while the 6-D reaction
        # matrix is indexed by incoming *joint*.  Robot assets contain fixed
        # joints that have no DOF, so using dof_index + 1 for reactions reads
        # unrelated rows and can produce nonsensical forces.  Isaac's API
        # explicitly exposes the required name-to-joint-index map in metadata.
        view = getattr(articulation, "_articulation_view", None)
        metadata = getattr(view, "_metadata", None)
        joint_index_map = getattr(metadata, "joint_indices", None)
        if joint_index_map is None:
            raise ValueError("articulation joint metadata is required for reaction telemetry")
        try:
            self.reaction_rows = np.asarray(
                [int(joint_index_map[name]) + 1 for name in self.joint_names],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(f"joint metadata is missing {error.args[0]}") from error
        self.rigid_contact_view = rigid_contact_view
        self.physics_dt_s = float(physics_dt_s)
        self.contact_threshold_n = float(contact_threshold_n)
        count = len(self.joint_names)
        self._samples = 0
        self._effort_square_sum = np.zeros(count, dtype=np.float64)
        self._peak_abs_effort = np.zeros(count, dtype=np.float64)
        self._peak_reaction_force = np.zeros(count, dtype=np.float64)
        self._peak_reaction_torque = np.zeros(count, dtype=np.float64)
        self._peak_contact_force = 0.0
        self._contact_samples = 0

    def sample(self) -> None:
        efforts = _numpy(self.articulation.get_measured_joint_efforts()).reshape(-1)
        reaction = _numpy(self.articulation.get_measured_joint_forces())
        if efforts.size <= int(self.joint_indices.max()):
            raise RuntimeError("PhysX joint-effort vector does not cover the requested joints")
        if reaction.ndim != 2 or reaction.shape[1] != 6:
            raise RuntimeError("PhysX measured joint reactions must have shape (N, 6)")
        if reaction.shape[0] <= int(self.reaction_rows.max()):
            raise RuntimeError("PhysX joint-reaction matrix does not cover the requested joints")

        selected_efforts = efforts[self.joint_indices]
        selected_reaction = reaction[self.reaction_rows]
        reaction_force = np.linalg.norm(selected_reaction[:, :3], axis=1)
        reaction_torque = np.linalg.norm(selected_reaction[:, 3:], axis=1)
        self._samples += 1
        self._effort_square_sum += selected_efforts * selected_efforts
        self._peak_abs_effort = np.maximum(self._peak_abs_effort, np.abs(selected_efforts))
        self._peak_reaction_force = np.maximum(self._peak_reaction_force, reaction_force)
        self._peak_reaction_torque = np.maximum(self._peak_reaction_torque, reaction_torque)

        if self.rigid_contact_view is None:
            return
        # A contact-filtered view must be read through the pair-wise force
        # matrix. ``get_net_contact_forces`` also includes contacts with
        # unfiltered bodies, which makes a floor-only audit count nearby props.
        pairwise = getattr(self.rigid_contact_view, "get_contact_force_matrix", None)
        if pairwise is None:
            contact_vectors = _numpy(
                self.rigid_contact_view.get_net_contact_forces(dt=self.physics_dt_s)
            ).reshape(-1, 3)
        else:
            contact_vectors = _numpy(pairwise(dt=self.physics_dt_s)).reshape(-1, 3)
        contact_force = float(np.max(np.linalg.norm(contact_vectors, axis=1), initial=0.0))
        self._peak_contact_force = max(self._peak_contact_force, contact_force)
        if contact_force >= self.contact_threshold_n:
            self._contact_samples += 1

    def summary(self) -> PhysxManipulationTelemetrySummary:
        divisor = max(self._samples, 1)
        rms = np.sqrt(self._effort_square_sum / divisor)
        return PhysxManipulationTelemetrySummary(
            samples=self._samples,
            joint_names=self.joint_names,
            peak_abs_joint_effort_nm=tuple(map(float, self._peak_abs_effort)),
            rms_joint_effort_nm=tuple(map(float, rms)),
            peak_joint_reaction_force_n=tuple(map(float, self._peak_reaction_force)),
            peak_joint_reaction_torque_nm=tuple(map(float, self._peak_reaction_torque)),
            peak_rigid_contact_force_n=float(self._peak_contact_force),
            contact_samples=self._contact_samples,
        )
