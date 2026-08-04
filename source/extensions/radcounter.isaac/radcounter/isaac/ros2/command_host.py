"""Thread-safe ROS command host executed from the Isaac physics thread."""

from __future__ import annotations

import json
import queue
from dataclasses import dataclass

from std_msgs.msg import String


@dataclass(slots=True)
class _ActiveTreatment:
    task_id: str
    remaining_s: float
    step: int = 0
    removed_activity_bq: float = 0.0


class IsaacRos2CommandHost:
    """Queue ROS callbacks and apply all USD/PhysX changes on update()."""

    def __init__(self, node, controller, radiation_simulation, decontaminator=None) -> None:
        self.node = node
        self.controller = controller
        self.radiation_simulation = radiation_simulation
        self.decontaminator = decontaminator
        self._commands: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        self._canceled: set[str] = set()
        self._active_treatment: _ActiveTreatment | None = None
        self._publisher = node.create_publisher(String, "/radcounter/isaac/result", 10)
        self._subscription = node.create_subscription(
            String, "/radcounter/isaac/command", self._enqueue, 10
        )

    def _enqueue(self, message: String) -> None:
        payload = json.loads(message.data)
        if payload.get("command") == "cancel":
            self._canceled.add(str(payload.get("task_id", "")))
            return
        self._commands.put(payload)

    def _publish(self, task_id: str, success: bool, message: str, **values: object) -> None:
        result = String()
        result.data = json.dumps(
            {"task_id": task_id, "success": success, "message": message, **values},
            separators=(",", ":"),
        )
        self._publisher.publish(result)

    def _execute(self, command: dict[str, object]) -> None:
        task_id = str(command.get("task_id", ""))
        if task_id in self._canceled:
            self._canceled.discard(task_id)
            self._publish(task_id, False, "task canceled before physical execution")
            return
        operation = str(command.get("command", ""))
        try:
            if operation == "grasp":
                report = self.controller.grasp(str(command["object_path"]))
                self._publish(task_id, report.success, report.message)
            elif operation == "release":
                report = self.controller.release()
                self._publish(task_id, report.success, report.message)
            elif operation == "remove":
                report = self.controller.remove_to_disposal_zone(
                    str(command["object_path"]),
                    str(command["disposal_zone_path"]),
                )
                self._publish(task_id, report.success, report.message)
            elif operation == "synchronize_radiation":
                changed = self.radiation_simulation.synchronize()
                self._publish(
                    task_id, True, "radiation scene synchronized", changed_paths=list(changed)
                )
            elif operation == "measure":
                records = self.radiation_simulation.measure(
                    duration_s=float(command.get("duration_s", 2.0))
                )
                self._publish(
                    task_id,
                    True,
                    "measurement complete",
                    measurements=[item.as_dict() for item in records],
                )
            elif operation == "decontaminate":
                if self.decontaminator is None:
                    self._publish(task_id, False, "decontaminator is not configured")
                elif self._active_treatment is not None:
                    self._publish(task_id, False, "another treatment is active")
                else:
                    self._active_treatment = _ActiveTreatment(task_id, float(command["duration_s"]))
            else:
                self._publish(task_id, False, f"unsupported Isaac command: {operation}")
        except Exception as exc:
            self._publish(task_id, False, f"{type(exc).__name__}: {exc}")

    def update(self, physics_dt_s: float) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                break
            self._execute(command)
        treatment = self._active_treatment
        if treatment is None:
            return
        if treatment.task_id in self._canceled:
            self._canceled.discard(treatment.task_id)
            self._publish(treatment.task_id, False, "decontamination canceled")
            self._active_treatment = None
            return
        treatment.step += 1
        result = self.decontaminator.tick(physics_dt_s, treatment.step)
        treatment.removed_activity_bq += result.removed_activity_bq
        treatment.remaining_s -= physics_dt_s
        if treatment.remaining_s <= 0:
            digest = self.decontaminator.flush()
            self.radiation_simulation.refresh_scene_state()
            self._publish(
                treatment.task_id,
                True,
                "decontamination completed",
                removed_activity_bq=treatment.removed_activity_bq,
                activity_map_sha256=digest,
            )
            self._active_treatment = None
