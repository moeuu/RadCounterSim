"""Lifecycle-safe subscription of simulator services to PhysX step events."""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PhysicsLoopStatus:
    running: bool
    tick_count: int
    physics_time_s: float
    callback_thread_id: int | None
    last_error: str | None


class IsaacPhysicsStepLoop:
    """Invoke command/treatment services on the actual Isaac physics callback."""

    def __init__(
        self,
        callbacks: Iterable[Callable[[float], None] | Any],
        *,
        radiation_simulation: Any | None = None,
        synchronize_every_ticks: int = 1,
    ) -> None:
        if synchronize_every_ticks < 1:
            raise ValueError("synchronize_every_ticks must be positive")
        self.callbacks = tuple(callbacks)
        self.radiation_simulation = radiation_simulation
        self.synchronize_every_ticks = synchronize_every_ticks
        self._interface = None
        self._subscription = None
        self._tick_count = 0
        self._physics_time_s = 0.0
        self._callback_thread_id: int | None = None
        self._last_error: str | None = None
        self._in_callback = False

    @staticmethod
    def _invoke(callback: Callable[[float], None] | Any, dt_s: float) -> None:
        if callable(callback):
            callback(dt_s)
        elif hasattr(callback, "update"):
            callback.update(dt_s)
        else:
            raise TypeError("physics callback must be callable or expose update(dt_s)")

    def _on_physics_step(self, dt_s: float) -> None:
        if self._in_callback:
            return
        self._in_callback = True
        try:
            self._callback_thread_id = threading.get_ident()
            value = float(dt_s)
            self._tick_count += 1
            self._physics_time_s += value
            for callback in self.callbacks:
                self._invoke(callback, value)
            if (
                self.radiation_simulation is not None
                and self._tick_count % self.synchronize_every_ticks == 0
            ):
                self.radiation_simulation.synchronize()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._in_callback = False

    def start(self) -> None:
        if self._subscription is not None:
            return
        import omni.physx

        self._interface = omni.physx.get_physx_interface()
        self._subscription = self._interface.subscribe_physics_step_events(self._on_physics_step)

    def stop(self) -> None:
        subscription = self._subscription
        interface = self._interface
        self._subscription = None
        self._interface = None
        if subscription is None:
            return
        if hasattr(subscription, "unsubscribe"):
            subscription.unsubscribe()
        elif interface is not None and hasattr(interface, "unsubscribe_physics_step_events"):
            interface.unsubscribe_physics_step_events(subscription)

    @property
    def status(self) -> PhysicsLoopStatus:
        return PhysicsLoopStatus(
            running=self._subscription is not None,
            tick_count=self._tick_count,
            physics_time_s=self._physics_time_s,
            callback_thread_id=self._callback_thread_id,
            last_error=self._last_error,
        )

    def __enter__(self) -> IsaacPhysicsStepLoop:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
