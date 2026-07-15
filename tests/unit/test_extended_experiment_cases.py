from pathlib import Path

import pytest
import yaml

from radcounter.core.experiments.extended_cases import EXTENDED_CASES


@pytest.mark.parametrize("case_id", sorted(EXTENDED_CASES))
def test_extended_case_meets_acceptance_and_artifact_contract(case_id: str) -> None:
    result = EXTENDED_CASES[case_id]().run(
        seed=42,
        planner_id="closed_loop_residual",
    )
    assert result.metrics["passed"] is True
    assert set(result.tables) == {"measurements", "estimates", "actions", "resources"}
    assert result.maps


def test_closed_loop_reference_config_separates_truth_and_belief() -> None:
    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (root / "configs/scenarios/closed_loop_reference.yaml").read_text(encoding="utf-8")
    )
    assert config["schema_version"] == "1.0.0"
    assert config["truth"]["access_policy"] == "simulator_only"
    assert config["belief"]["access_policy"] == "planner_estimator_only"
    assert config["planning"]["verification"]["required_after_every_action"] is True
