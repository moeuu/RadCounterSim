"""GPU-independent adaptive workload policy and bounded tile residency."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from enum import IntEnum
from statistics import fmean


class QualityLevel(IntEnum):
    HIGH = 0
    BALANCED = 1
    CONSERVATIVE = 2
    MINIMAL = 3


@dataclass(frozen=True)
class RuntimeBudget:
    target_fps: float = 20.0
    initial_quality: QualityLevel = QualityLevel.CONSERVATIVE
    sample_window: int = 120
    downgrade_ratio: float = 1.15
    upgrade_ratio: float = 0.72
    decision_cooldown_frames: int = 180
    max_resident_tiles: int = 50


@dataclass(frozen=True)
class WorkloadDecision:
    quality: QualityLevel
    tile_radius: int
    obstacles_per_tile: int
    environment_update_hz: float
    lidar_read_hz: float
    camera_read_hz: float
    radiation_update_hz: float
    draw_lidar_points: bool


_DECISIONS = {
    QualityLevel.HIGH: WorkloadDecision(QualityLevel.HIGH, 3, 6, 4.0, 10.0, 15.0, 10.0, True),
    QualityLevel.BALANCED: WorkloadDecision(QualityLevel.BALANCED, 2, 4, 2.0, 7.5, 10.0, 5.0, True),
    QualityLevel.CONSERVATIVE: WorkloadDecision(
        QualityLevel.CONSERVATIVE, 2, 2, 1.0, 4.0, 5.0, 2.0, False
    ),
    QualityLevel.MINIMAL: WorkloadDecision(QualityLevel.MINIMAL, 1, 1, 0.5, 2.0, 2.0, 1.0, False),
}


class AdaptiveWorkloadGovernor:
    """Hysteretic frame-time controller that never changes simulation truth."""

    def __init__(self, budget: RuntimeBudget) -> None:
        if budget.target_fps <= 0.0:
            raise ValueError("target_fps must be positive")
        self.budget = budget
        self.quality = budget.initial_quality
        self._frame_times: deque[float] = deque(maxlen=budget.sample_window)
        self._frames_since_decision = budget.decision_cooldown_frames
        self.transitions: list[tuple[int, QualityLevel, QualityLevel, float]] = []
        self.frame_index = 0

    @property
    def decision(self) -> WorkloadDecision:
        return _DECISIONS[self.quality]

    def observe(self, frame_time_s: float) -> WorkloadDecision:
        if frame_time_s <= 0.0 or not math.isfinite(frame_time_s):
            return self.decision
        self.frame_index += 1
        self._frames_since_decision += 1
        self._frame_times.append(frame_time_s)
        if len(self._frame_times) < self.budget.sample_window:
            return self.decision
        if self._frames_since_decision < self.budget.decision_cooldown_frames:
            return self.decision

        mean_s = fmean(self._frame_times)
        target_s = 1.0 / self.budget.target_fps
        previous = self.quality
        if mean_s > target_s * self.budget.downgrade_ratio:
            self.quality = QualityLevel(min(int(self.quality) + 1, int(QualityLevel.MINIMAL)))
        elif mean_s < target_s * self.budget.upgrade_ratio:
            self.quality = QualityLevel(max(int(self.quality) - 1, int(QualityLevel.HIGH)))
        if self.quality is not previous:
            self.transitions.append((self.frame_index, previous, self.quality, mean_s))
            self._frames_since_decision = 0
            self._frame_times.clear()
        return self.decision


class TileResidencyPlanner:
    """Select a bounded set of tiles nearest to one or more moving focus points."""

    def __init__(
        self,
        tile_size_m: float,
        logical_size_m: float,
        max_resident_tiles: int,
    ) -> None:
        if tile_size_m <= 0.0 or logical_size_m <= 0.0:
            raise ValueError("tile and logical sizes must be positive")
        if max_resident_tiles <= 0:
            raise ValueError("max_resident_tiles must be positive")
        self.tile_size_m = tile_size_m
        self.logical_size_m = logical_size_m
        self.max_resident_tiles = max_resident_tiles
        self.half_tiles = max(1, math.ceil(0.5 * logical_size_m / tile_size_m))

    @property
    def logical_tile_count(self) -> int:
        return (2 * self.half_tiles) ** 2

    def select(
        self,
        focus_positions_m: Iterable[tuple[float, float]],
        radius_tiles: int,
    ) -> tuple[tuple[int, int], ...]:
        focus = tuple(focus_positions_m)
        if not focus:
            return ()
        candidates: set[tuple[int, int]] = set()
        centers = []
        for x, y in focus:
            center = (math.floor(x / self.tile_size_m), math.floor(y / self.tile_size_m))
            centers.append(center)
            for dx in range(-radius_tiles, radius_tiles + 1):
                for dy in range(-radius_tiles, radius_tiles + 1):
                    key = (center[0] + dx, center[1] + dy)
                    if (
                        -self.half_tiles <= key[0] < self.half_tiles
                        and -self.half_tiles <= key[1] < self.half_tiles
                    ):
                        candidates.add(key)

        ordered = sorted(
            candidates,
            key=lambda key: (
                min((key[0] - cx) ** 2 + (key[1] - cy) ** 2 for cx, cy in centers),
                key,
            ),
        )
        return tuple(ordered[: self.max_resident_tiles])

    def center(self, key: tuple[int, int]) -> tuple[float, float]:
        return (
            (key[0] + 0.5) * self.tile_size_m,
            (key[1] + 0.5) * self.tile_size_m,
        )


class RateGate:
    """Independent wall/simulation-time rate limits for expensive subsystems."""

    def __init__(self) -> None:
        self._last: dict[str, float] = {}

    def due(self, name: str, now_s: float, frequency_hz: float) -> bool:
        if frequency_hz <= 0.0:
            return False
        interval = 1.0 / frequency_hz
        previous = self._last.get(name)
        if previous is not None and now_s - previous + 1e-12 < interval:
            return False
        self._last[name] = now_s
        return True
