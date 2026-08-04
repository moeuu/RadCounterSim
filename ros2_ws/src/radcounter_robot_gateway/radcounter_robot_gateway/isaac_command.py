"""Correlated JSON command channel to the Isaac physics thread."""

from __future__ import annotations

import json

from rclpy.task import Future
from std_msgs.msg import String


class IsaacCommandClient:
    def __init__(self, node, command_topic: str, result_topic: str) -> None:
        self._node = node
        self._publisher = node.create_publisher(String, command_topic, 10)
        self._subscription = node.create_subscription(String, result_topic, self._on_result, 10)
        self._pending: dict[str, Future] = {}

    def _on_result(self, message: String) -> None:
        payload = json.loads(message.data)
        future = self._pending.pop(str(payload.get("task_id", "")), None)
        if future is not None and not future.done():
            future.set_result(payload)

    async def execute(self, task_id: str, command: str, **parameters: object) -> dict[str, object]:
        if task_id in self._pending:
            raise RuntimeError(f"task {task_id} already has a pending Isaac command")
        future = Future(executor=self._node.executor)
        self._pending[task_id] = future
        message = String()
        message.data = json.dumps(
            {"task_id": task_id, "command": command, **parameters}, separators=(",", ":")
        )
        self._publisher.publish(message)
        return await future

    def cancel(self, task_id: str) -> None:
        self._pending.pop(task_id, None)
        message = String()
        message.data = json.dumps({"task_id": task_id, "command": "cancel"}, separators=(",", ":"))
        self._publisher.publish(message)
