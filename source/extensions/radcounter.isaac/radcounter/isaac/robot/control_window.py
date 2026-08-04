"""Small in-simulator status and fallback control window."""

from __future__ import annotations

from radcounter.core.robots.control import TwistCommand
from radcounter.core.robots.input import ControlSource


class RobotControlWindow:
    def __init__(self, router, robot_id: str, command_port: int) -> None:
        import omni.ui as ui

        self.router = router
        self.robot_id = robot_id
        self.window = ui.Window("RadCounterSim Robot Control", width=430, height=330)
        with self.window.frame, ui.VStack(spacing=8):
            ui.Label(
                "W/S forward  A/D turn  Q/E strafe",
                style={"font_size": 18},
            )
            ui.Label("Gamepad: left stick move, right stick turn")
            ui.Label("SPACE: emergency stop  ENTER: clear stop")
            ui.Label(f"Command TCP: 127.0.0.1:{command_port}")
            self.source_label = ui.Label("Active source: none")
            self.pose_label = ui.Label("Pose: waiting")
            with ui.HStack(spacing=6):
                ui.Button("Forward", clicked_fn=lambda: self._twist(0.6, 0.0))
                ui.Button("Left", clicked_fn=lambda: self._twist(0.0, 0.8))
                ui.Button("Stop", clicked_fn=self._stop)
                ui.Button("Right", clicked_fn=lambda: self._twist(0.0, -0.8))
                ui.Button("Reverse", clicked_fn=lambda: self._twist(-0.6, 0.0))
            with ui.HStack(spacing=6):
                ui.Button("Auto ON", clicked_fn=lambda: self._auto(True))
                ui.Button("Auto OFF", clicked_fn=lambda: self._auto(False))
                ui.Button("E-STOP", clicked_fn=self._emergency)
                ui.Button("Clear E-STOP", clicked_fn=self._clear_emergency)

    def update(self, position_m: tuple[float, float, float]) -> None:
        source = self.router.active_source(self.robot_id)
        self.source_label.text = f"Active source: {source.value if source else 'none'}"
        self.pose_label.text = (
            f"Pose: x={position_m[0]:.3f} y={position_m[1]:.3f} z={position_m[2]:.3f}"
        )

    def _twist(self, linear: float, angular: float) -> None:
        self.router.publish(
            self.robot_id,
            TwistCommand(linear_x_m_s=linear, angular_z_rad_s=angular),
            source=ControlSource.GUI,
            ttl_s=0.6,
        )

    def _stop(self) -> None:
        self.router.publish(
            self.robot_id,
            TwistCommand(),
            source=ControlSource.GUI,
            ttl_s=1.0,
        )

    def _auto(self, enabled: bool) -> None:
        self.router.set_auto_enabled(self.robot_id, enabled)

    def _emergency(self) -> None:
        self.router.mux.emergency_stop(self.robot_id, "GUI emergency stop")

    def _clear_emergency(self) -> None:
        self.router.mux.clear_emergency_stop(self.robot_id)
