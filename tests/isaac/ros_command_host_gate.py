"""Cross-runtime gate: ROS DDS command -> Isaac main thread -> Embree measurement."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source/extensions/radcounter.isaac")]


class _RecordingSimulation:
    def __init__(self, simulation) -> None:
        self.simulation = simulation
        self.measurement_thread_id: int | None = None

    def measure(self, **kwargs):
        self.measurement_thread_id = threading.get_ident()
        return self.simulation.measure(**kwargs)

    def synchronize(self):
        return self.simulation.synchronize()

    def refresh_scene_state(self) -> None:
        self.simulation.refresh_scene_state()


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    executor = None
    spin_thread = None
    nodes = []
    try:
        import omni.usd
        import rclpy
        from radcounter.isaac.ros2.command_host import IsaacRos2CommandHost
        from radcounter.isaac.runtime.simulation import IsaacRadiationSimulation
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from std_msgs.msg import String

        context = omni.usd.get_context()
        assert context.open_stage(str(ROOT / "assets/environments/radcounter_vertical_slice.usda"))
        for _ in range(6):
            app.update()
        simulation = _RecordingSimulation(
            IsaacRadiationSimulation.from_config(
                context.get_stage(),
                ROOT / "configs/scenarios/vertical_slice.runtime.json",
            )
        )
        rclpy.init()
        host_node = Node("radcounter_isaac_command_host_gate")
        client_node = Node("radcounter_isaac_command_client_gate")
        nodes = [host_node, client_node]
        host = IsaacRos2CommandHost(host_node, object(), simulation)
        results: list[dict[str, object]] = []
        client_node.create_subscription(
            String,
            "/radcounter/isaac/result",
            lambda message: results.append(json.loads(message.data)),
            10,
        )
        publisher = client_node.create_publisher(String, "/radcounter/isaac/command", 10)
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(host_node)
        executor.add_node(client_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()
        time.sleep(0.1)
        command = String()
        command.data = json.dumps(
            {"task_id": "isaac-ros-measure-1", "command": "measure", "duration_s": 0.2}
        )
        publisher.publish(command)
        main_thread_id = threading.get_ident()
        deadline = time.monotonic() + 5.0
        while not results and time.monotonic() < deadline:
            app.update()
            host.update(1.0 / 60.0)
            time.sleep(0.005)
        assert results, "Isaac command result was not returned over DDS"
        result = results[0]
        assert result["success"] is True, result
        assert len(result["measurements"]) == 5
        assert simulation.measurement_thread_id == main_thread_id
        print(
            json.dumps(
                {
                    "success": True,
                    "measurement_count": len(result["measurements"]),
                    "executed_on_isaac_main_thread": True,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if executor is not None:
            executor.shutdown(timeout_sec=2.0)
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        for node in nodes:
            node.destroy_node()
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except ImportError:
            pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
