from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import radcounter
from radcounter.core.actions import ResourceState
from radcounter.core.models.actions import ActionStatus, ActionType
from radcounter.core.planning import DeterministicFeasibilityChecker
from radcounter.core.planning.models import FeasibilityFacts

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "source/extensions/radcounter.isaac"
if str(EXTENSION) not in sys.path:
    sys.path.insert(0, str(EXTENSION))
extension_namespace = str(EXTENSION / "radcounter")
if extension_namespace not in radcounter.__path__:
    radcounter.__path__.append(extension_namespace)

from radcounter.isaac.planning import (  # noqa: E402
    IsaacActionCandidateGenerator,
    SceneCandidateConfig,
)
from radcounter.isaac.workflow.services import (  # noqa: E402
    IsaacWorkflowServices,
    _ExecutionReport,
)


class _Attribute:
    def __init__(self, value: object) -> None:
        self.value = value

    def __bool__(self) -> bool:
        return True

    def HasAuthoredValueOpinion(self) -> bool:  # noqa: N802
        return True

    def IsValid(self) -> bool:  # noqa: N802
        return True

    def Get(self) -> object:  # noqa: N802
        return self.value

    def Set(self, value: object) -> None:  # noqa: N802
        self.value = value


class _MissingAttribute:
    def __bool__(self) -> bool:
        return False

    def IsValid(self) -> bool:  # noqa: N802
        return False


class _ShieldPrim:
    path = "/World/LeadShield"

    def __init__(self) -> None:
        self.attributes = {
            "rad:role": _Attribute("shield"),
            "rad:material:id": _Attribute("lead"),
            "rad:shield:movable": _Attribute(True),
            "rad:shield:resourceUnits": _Attribute(1),
            "rad:shield:deployed": _Attribute(False),
            "rad:shield:placementFraction": _Attribute(0.0),
            "rad:manipulation:graspFrame": _Attribute("ShieldGraspFrame"),
        }

    def __bool__(self) -> bool:
        return True

    def GetPath(self) -> str:  # noqa: N802
        return self.path

    def GetAttribute(self, name: str) -> _Attribute | _MissingAttribute:  # noqa: N802
        return self.attributes.get(name, _MissingAttribute())

    def IsValid(self) -> bool:  # noqa: N802
        return True


class _Stage:
    def __init__(self, shield: _ShieldPrim) -> None:
        self.shield = shield

    def Traverse(self) -> tuple[_ShieldPrim, ...]:  # noqa: N802
        return (self.shield,)

    def GetPrimAtPath(self, path: str) -> _ShieldPrim | None:  # noqa: N802
        return self.shield if path == self.shield.path else None


class _Simulation:
    def __init__(self) -> None:
        self.synchronize_calls = 0

    def synchronize(self) -> tuple[str, ...]:
        self.synchronize_calls += 1
        return ("/World/LeadShield",)


def _generator(stage: _Stage, simulation: _Simulation) -> IsaacActionCandidateGenerator:
    generator = object.__new__(IsaacActionCandidateGenerator)
    generator.stage = stage
    generator.simulation = simulation
    generator.config = SceneCandidateConfig(shield_line_fractions=(0.35, 0.65))
    generator.probe = SimpleNamespace(
        world_position=lambda value: (
            np.asarray((0.0, 0.0, 0.6), dtype=np.float64)
            if isinstance(value, str)
            else np.asarray((0.0, 0.0, 0.0), dtype=np.float64)
        )
    )
    generator._public_source_samples = lambda belief: (  # type: ignore[method-assign]
        np.asarray(((0.0, 0.0, 0.5),), dtype=np.float64),
        np.asarray((1.0,), dtype=np.float64),
    )
    generator._task_position = lambda: np.asarray(  # type: ignore[method-assign]
        (10.0, 0.0, 0.5), dtype=np.float64
    )
    generator._robot_position = lambda path: np.asarray(  # type: ignore[method-assign]
        (-1.0, 0.0, 0.0), dtype=np.float64
    )
    generator._center = lambda prim: np.asarray(  # type: ignore[method-assign]
        (0.0, 0.0, 0.5), dtype=np.float64
    )
    generator._base_for_end_effector = (  # type: ignore[method-assign]
        lambda target, **kwargs: np.asarray(target, dtype=np.float64)
    )
    generator._facts = lambda **kwargs: FeasibilityFacts()  # type: ignore[method-assign]
    generator._route = lambda *args, **kwargs: []  # type: ignore[method-assign]
    generator._shield_reduction = lambda shield: 0.1  # type: ignore[method-assign]
    generator._candidate = (  # type: ignore[method-assign]
        lambda action, belief, facts, **kwargs: SimpleNamespace(action=action, feasibility=facts)
    )
    return generator


def test_successful_shield_deployment_transitions_stable_candidates_to_move() -> None:
    shield = _ShieldPrim()
    stage = _Stage(shield)
    simulation = _Simulation()
    generator = _generator(stage, simulation)
    resources = ResourceState(
        remaining_work_time_s=300.0,
        remaining_robot_runtime_s={"/World/CountermeasureRobot": 300.0},
        remaining_shield_units={"lead": 1},
        remaining_countermeasure_count=4,
    )
    services = IsaacWorkflowServices(
        simulation,
        generator,
        lambda measurement, previous: previous,
        resources=resources,
    )
    services._physical_execute = (  # type: ignore[method-assign]
        lambda action: _ExecutionReport(True, "physical placement completed", {})
    )

    initial = generator._shield_candidates(None)  # type: ignore[arg-type]
    place_35 = initial[0].action
    assert place_35.action_type == ActionType.PLACE_SHIELD
    assert place_35.parameters["shield_units"] == 1
    assert place_35.parameters["placement_fraction"] == 0.35
    assert place_35.parameters["deployment_state"] == "available"

    placed = asyncio.run(services.execute(place_35))

    assert placed.status == ActionStatus.COMPLETED
    assert shield.attributes["rad:shield:deployed"].Get() is True
    assert shield.attributes["rad:shield:placementFraction"].Get() == 0.35
    assert resources.remaining_shield_units["lead"] == 0
    assert placed.public_details["deployment_state"] == "deployed"
    assert placed.public_details["placement_fraction"] == 0.35

    deployed = generator._shield_candidates(None)  # type: ignore[arg-type]
    move_35 = deployed[0].action
    move_65_candidate = deployed[1]
    move_65 = move_65_candidate.action
    assert move_35.action_id == place_35.action_id
    assert all(candidate.action.action_type == ActionType.MOVE_SHIELD for candidate in deployed)
    assert all(candidate.action.parameters["shield_units"] == 0 for candidate in deployed)
    assert DeterministicFeasibilityChecker().evaluate(move_65_candidate, resources).feasible
    assert all(
        candidate.action.parameters["deployment_state"] == "deployed" for candidate in deployed
    )

    moved = asyncio.run(services.execute(move_65))

    assert moved.status == ActionStatus.COMPLETED
    assert shield.attributes["rad:shield:placementFraction"].Get() == 0.65
    assert resources.remaining_shield_units["lead"] == 0
    assert simulation.synchronize_calls == 2


def test_failed_shield_placement_does_not_commit_metadata_or_inventory() -> None:
    shield = _ShieldPrim()
    stage = _Stage(shield)
    simulation = _Simulation()
    generator = _generator(stage, simulation)
    resources = ResourceState(
        remaining_shield_units={"lead": 2},
        remaining_countermeasure_count=2,
    )
    services = IsaacWorkflowServices(
        simulation,
        generator,
        lambda measurement, previous: previous,
        resources=resources,
    )
    services._physical_execute = (  # type: ignore[method-assign]
        lambda action: _ExecutionReport(False, "physical placement failed", {})
    )
    action = generator._shield_candidates(None)[0].action  # type: ignore[arg-type]

    result = asyncio.run(services.execute(action))

    assert result.status == ActionStatus.FAILED
    assert shield.attributes["rad:shield:deployed"].Get() is False
    assert shield.attributes["rad:shield:placementFraction"].Get() == 0.0
    assert resources.remaining_shield_units["lead"] == 2
    assert simulation.synchronize_calls == 0
