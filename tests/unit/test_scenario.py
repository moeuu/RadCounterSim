from pathlib import Path

from radcounter.core.models import load_scenario


def test_sample_scenario_validates() -> None:
    root = Path(__file__).resolve().parents[2]
    scenario = load_scenario(root / "configs/scenarios/analytic_free_space.yaml")
    assert scenario.scenario_id == "analytic_free_space"
    assert len(scenario.measurement_poses) == 2
