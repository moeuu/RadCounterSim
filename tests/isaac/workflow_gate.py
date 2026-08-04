"""Actual Isaac closed-loop and physics-callback integration gate."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
for path in (ROOT, EXTENSION, ROOT / "build/native/python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class _Stepper:
    def __init__(self, app) -> None:
        self.app = app

    def step(self, *, render: bool = False) -> None:
        del render
        self.app.update()


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    loop = None
    try:
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom
        from radcounter.isaac.planning import IsaacActionCandidateGenerator
        from radcounter.isaac.robot import (
            IsaacPhysicsRobotController,
            PhysicsControllerConfig,
        )
        from radcounter.isaac.runtime import (
            IsaacPhysicsStepLoop,
            IsaacRadiationSimulation,
        )
        from radcounter.isaac.workflow import IsaacWorkflowServices, PublicMeasurement

        from radcounter.core.actions import ResourceState
        from radcounter.core.models import BeliefState, RevisionState
        from radcounter.core.planning import (
            DeterministicFeasibilityChecker,
            OpenLoopPlanner,
        )
        from radcounter.core.workflow import (
            ClosedLoopCoordinator,
            TaskMetrics,
            TerminationConfig,
            WorkflowState,
        )

        context = omni.usd.get_context()
        assert context.open_stage(str(ROOT / "assets/environments/radcounter_vertical_slice.usda"))
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        for _ in range(12):
            app.update()

        simulation = IsaacRadiationSimulation.from_config(
            stage,
            ROOT / "configs/scenarios/vertical_slice.runtime.json",
        )
        controller = IsaacPhysicsRobotController(
            stage,
            "/World/CountermeasureRobot",
            "/World/CountermeasureRobot/DeconTool",
            _Stepper(app),
            config=PhysicsControllerConfig(
                grasp_distance_m=0.24,
                maximum_navigation_steps=2200,
                maximum_settle_steps=180,
            ),
        )
        generator = IsaacActionCandidateGenerator(stage, simulation, controller=controller)
        initial_belief = BeliefState(
            ("/World/ContaminatedFloor",),
            np.asarray([2.4e7]),
            np.asarray([[4.0e12]]),
            RevisionState(),
        )
        resources = ResourceState(
            remaining_work_time_s=300.0,
            remaining_robot_runtime_s={"/World/CountermeasureRobot": 300.0},
            remaining_shield_units={"lead": 1},
            remaining_countermeasure_count=2,
        )
        checker = DeterministicFeasibilityChecker()
        shield_candidates = [
            item
            for item in generator.generate_shield_actions(initial_belief)
            if checker.evaluate(item, resources).feasible
        ]
        assert shield_candidates, "the live scene generated no feasible shield action"
        selected = shield_candidates[0]
        shield = stage.GetPrimAtPath(selected.action.target_prim_path)
        before_pose = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(shield).Transform(Gf.Vec3d())
        )

        estimator_inputs: list[tuple[PublicMeasurement, ...]] = []

        def estimator(measurement, previous):
            del previous
            assert all(isinstance(item, PublicMeasurement) for item in measurement)
            assert all(not hasattr(item, "expected_rate_cps") for item in measurement)
            estimator_inputs.append(measurement)
            return initial_belief

        services = IsaacWorkflowServices(
            simulation,
            generator,
            estimator,
            controller=controller,
            resources=resources,
            task_evaluator=lambda belief: TaskMetrics(0.0, 0.0),
        )
        coordinator = ClosedLoopCoordinator(
            services,
            OpenLoopPlanner((selected.action.action_id,)),
            TerminationConfig(1, 0.0, 0.0),
        )

        callback_ticks: list[float] = []
        loop = IsaacPhysicsStepLoop(
            [lambda dt: callback_ticks.append(float(dt))],
            radiation_simulation=simulation,
            synchronize_every_ticks=4,
        )
        loop.start()
        for _ in range(6):
            app.update()
        assert loop.status.tick_count > 0
        assert loop.status.last_error is None

        result = asyncio.run(coordinator.run_episode())
        assert result.terminal_state == WorkflowState.COMPLETE, result
        assert result.completed_cycles == 1
        assert estimator_inputs and services.last_verification
        assert services.last_diagnosis is not None
        assert services.last_action is not None
        assert services.last_action.action_id == selected.action.action_id
        after_pose = np.asarray(
            UsdGeom.XformCache().GetLocalToWorldTransform(shield).Transform(Gf.Vec3d())
        )
        displacement = float(np.linalg.norm(after_pose - before_pose))
        assert displacement > 0.25, (before_pose, after_pose)
        assert simulation.transport.statistics["native_trace_calls"] > 0
        print(
            json.dumps(
                {
                    "success": True,
                    "terminal_state": str(result.terminal_state),
                    "action_id": selected.action.action_id,
                    "shield_displacement_m": displacement,
                    "initial_measurements": len(services.last_measurement),
                    "verification_measurements": len(services.last_verification),
                    "physics_callback_ticks": loop.status.tick_count,
                    "truth_fields_exposed_to_estimator": False,
                }
            ),
            flush=True,
        )
        return 0
    finally:
        if loop is not None:
            loop.stop()
        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
