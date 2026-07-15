"""Atomic countermeasure resource accounting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field


@dataclass
class ResourceState:
    """Available and consumed resource units."""

    available: dict[str, float] = field(default_factory=dict)
    consumed: dict[str, float] = field(default_factory=dict)
    remaining_measurement_time_s: float = float("inf")
    remaining_work_time_s: float = float("inf")
    remaining_robot_runtime_s: dict[str, float] = field(default_factory=dict)
    remaining_shield_units: dict[str, int] = field(default_factory=dict)
    remaining_decon_media: float = float("inf")
    remaining_countermeasure_count: int = 2**31 - 1

    def __post_init__(self) -> None:
        scalar_values = (
            self.remaining_measurement_time_s,
            self.remaining_work_time_s,
            self.remaining_decon_media,
            float(self.remaining_countermeasure_count),
        )
        if any(value < 0 for value in scalar_values):
            raise ValueError("remaining mission resources must be nonnegative")
        if any(value < 0 for value in self.remaining_robot_runtime_s.values()):
            raise ValueError("remaining robot runtime must be nonnegative")
        if any(value < 0 for value in self.remaining_shield_units.values()):
            raise ValueError("remaining shield units must be nonnegative")

    def can_afford(self, cost: Mapping[str, float]) -> bool:
        """Return whether every requested resource is available."""

        return all(
            value >= 0 and self.available.get(name, 0.0) >= value for name, value in cost.items()
        )

    def consume(self, cost: Mapping[str, float]) -> None:
        """Atomically consume resources or raise without changing state."""

        if not self.can_afford(cost):
            missing = {
                name: value - self.available.get(name, 0.0)
                for name, value in cost.items()
                if self.available.get(name, 0.0) < value
            }
            raise ValueError(f"insufficient resources: {missing}")
        for name, value in cost.items():
            self.available[name] = self.available.get(name, 0.0) - value
            self.consumed[name] = self.consumed.get(name, 0.0) + value

    def clone(self) -> ResourceState:
        """Return an independent resource state."""

        return ResourceState(
            dict(self.available),
            dict(self.consumed),
            self.remaining_measurement_time_s,
            self.remaining_work_time_s,
            dict(self.remaining_robot_runtime_s),
            dict(self.remaining_shield_units),
            self.remaining_decon_media,
            self.remaining_countermeasure_count,
        )
