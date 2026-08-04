from pathlib import Path

import yaml


def test_real_robot_assets_and_motion_gate_are_declared() -> None:
    root = Path(__file__).resolve().parents[2]
    module = root / "source/extensions/radcounter.isaac/radcounter/isaac/robot/real_robots.py"
    runner = root / "scripts/run_real_robot_validation.py"
    gate = root / "tests/isaac/real_robot_gate.py"
    assert module.is_file()
    assert runner.is_file()
    assert gate.is_file()
    source = module.read_text(encoding="utf-8")
    assert "RidgebackFranka/ridgeback_franka.usd" in source
    assert "NVIDIA/NovaCarter/nova_carter.usd" in source
    assert "ArticulationAction" in source
    assert "ContactDrivenDecontaminator" not in source
    assert "FixedJoint.Define" in source


def test_robot_configs_select_real_isaac_controllers() -> None:
    root = Path(__file__).resolve().parents[2]
    countermeasure = yaml.safe_load(
        (root / "configs/robots/countermeasure_robot.yaml").read_text(encoding="utf-8")
    )
    measurement = yaml.safe_load(
        (root / "configs/robots/measurement_robot.yaml").read_text(encoding="utf-8")
    )
    assert countermeasure["reference_model_id"] == "mhi-meister"
    assert countermeasure["controller"] == "mhi_meister_dual_arm_crawler"
    assert countermeasure["arm_dofs"] == 14
    assert countermeasure["geometry_fidelity"] == "reference_procedural"
    assert measurement["reference_model_id"] == "irobot-packbot-fukushima"
    assert measurement["controller"] == "packbot_differential_crawler"
    assert measurement["wheel_dof_names"] == ["left_track_joint", "right_track_joint"]
