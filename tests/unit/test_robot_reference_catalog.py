from pathlib import Path

import yaml

from radcounter.core.robots import REAL_ROBOT_REFERENCES, RobotGeometryFidelity


def test_bundled_real_robot_catalog_is_traceable() -> None:
    expected = {
        "irobot-packbot-fukushima",
        "flyability-elios3-rad",
        "mhi-meister",
        "hitachi-ge-arounder",
    }
    assert set(REAL_ROBOT_REFERENCES) == expected
    for reference in REAL_ROBOT_REFERENCES.values():
        assert reference.source_urls
        assert all(url.startswith("https://") for url in reference.source_urls)
        assert all(value > 0 for value in reference.dimensions_m)
        assert reference.mechanisms
        assert reference.config().geometry_fidelity is RobotGeometryFidelity.REFERENCE_PROCEDURAL


def test_standard_fleet_examples_do_not_use_untraceable_robot_proxies() -> None:
    root = Path(__file__).resolve().parents[2]
    for name in ("fleet.example.yaml", "multimodal_fleet.example.yaml"):
        payload = yaml.safe_load((root / "configs/robots" / name).read_text(encoding="utf-8"))
        for robot in payload["robots"]:
            reference = robot["reference"]
            assert reference["model_id"] in REAL_ROBOT_REFERENCES
            assert reference["geometry_fidelity"] == "reference_procedural"
            assert robot["uri"] != "benchmark.usd"


def test_endurance_scene_uses_packbot_and_elios_reference_builders() -> None:
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/isaac_large_fleet_endurance.py").read_text(encoding="utf-8")
    assert "spawn_reference_robot" in source
    assert "irobot-packbot-fukushima" in source
    assert "flyability-elios3-rad" in source
    assert "benchmark.usd" not in source
