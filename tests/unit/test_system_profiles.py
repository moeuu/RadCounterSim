import json
from pathlib import Path

import pytest

from radcounter.core.system_profiles import (
    default_catalog_path,
    load_active_selection,
    load_system_catalog,
    resolve_system_selection,
    save_active_selection,
)
from scripts.convert_solidworks_to_usd import _conversion_command, discover_isaac_root


def test_repository_catalog_resolves_default_decommissioning_profile() -> None:
    catalog_path, catalog = load_system_catalog()
    assert catalog_path == default_catalog_path()
    assert catalog.default_profile == "vertical-slice"

    selection = resolve_system_selection(profile_id="vertical-slice")
    assert selection.environment_ready
    assert not selection.configurable
    assert selection.environment_config.environment_id == "vertical-slice-external"
    assert selection.robot_set.kind == "decommissioning"
    assert [item.descriptor.model_id for item in selection.detectors] == ["nai_tl"]


def test_fukushima_profile_is_pinned_to_expected_conversion_target() -> None:
    selection = resolve_system_selection(profile_id="fukushima-packbot")
    assert selection.configurable
    assert selection.environment_config.format.value == "usd"
    assert selection.environment_config.coordinate_system.up_axis.value == "auto"
    assert selection.environment_config.translation_world_m == (2.2, 0.0, 18.0)
    assert selection.spawn_anchor("ground-primary").translation_m == (0.0, -1.0, 0.0)
    assert selection.spawn_anchor("decon-surface").translation_m == (2.18, 0.0, 1.15)
    assert selection.environment_source_path is not None
    assert selection.environment_source_path.as_posix().endswith(
        ".cache/external/fukushima_daiichi_solidworks/export/Building.usdc"
    )
    assert [path.name for path in selection.environment_preparation_scripts] == [
        "fetch_fukushima_daiichi_cad.py",
        "convert_solidworks_to_usd.py",
    ]
    assert selection.robot_set.kind == "reference"
    assert {item.robot_id for item in selection.robot_set.reference_robots} == {
        "packbot",
        "elios3",
    }
    assert {item.spawn_anchor for item in selection.robot_set.reference_robots} == {
        "ground-primary",
        "aerial-primary",
    }
    assert {item.placement.detector_id for item in selection.detectors} == {
        "packbot-gm",
        "elios-imager",
    }


def test_component_overrides_are_independent_and_validate_robot_dependencies() -> None:
    selection = resolve_system_selection(
        profile_id="vertical-slice",
        robot_set_id="fukushima-response-reference",
        detector_set_id="fukushima-survey",
    )
    assert selection.configurable
    assert selection.environment_id == "vertical-slice"
    assert selection.robot_set_id == "fukushima-response-reference"

    with pytest.raises(ValueError, match="requires robots not present"):
        resolve_system_selection(
            profile_id="vertical-slice",
            robot_set_id="no-robots",
            detector_set_id="fukushima-survey",
        )


def test_fleet_paths_are_rebased_from_the_fleet_descriptor() -> None:
    selection = resolve_system_selection(
        profile_id="vertical-slice",
        robot_set_id="packbot-articulated",
        detector_set_id="no-detectors",
    )
    assert selection.robot_fleet is not None
    robot = selection.robot_fleet.robots[0]
    assert Path(robot.uri).is_absolute()
    assert Path(robot.uri).name == "packbot_fukushima.urdf"
    assert Path(robot.uri).is_file()


def test_active_selection_round_trip_uses_explicit_state_file(tmp_path: Path) -> None:
    state = tmp_path / "selection.json"
    selected = resolve_system_selection(profile_id="vertical-slice-packbot")
    save_active_selection(
        state,
        catalog_path=selected.catalog_path,
        profile_id=selected.profile_id,
    )
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert payload["profile_id"] == "vertical-slice-packbot"

    loaded = load_active_selection(state)
    assert loaded.profile_id == "vertical-slice-packbot"
    assert loaded.robot_set_id == "fukushima-response-reference"
    assert loaded.detector_set_id == "fukushima-survey"


def test_linux_solidworks_converter_uses_minimal_headless_kit(tmp_path: Path) -> None:
    isaac = tmp_path / "isaacsim" / "6.0.1-standalone"
    (isaac / "kit/apps").mkdir(parents=True)
    (isaac / "kit/kit").touch()
    hoops = isaac / "extscache/omni.kit.converter.hoops-510.3.0"
    core = isaac / "extscache/omni.kit.converter.hoops_core-511.3.2"
    launch = hoops / "omni/kit/converter/hoops/process/launch_hoops_app.py"
    launch.parent.mkdir(parents=True)
    launch.touch()
    core.mkdir(parents=True)

    assert discover_isaac_root(isaac) == isaac.resolve()
    command, hoops_name, core_name = _conversion_command(
        isaac,
        source=tmp_path / "Building.SLDASM",
        output=tmp_path / "Building.usdc",
        config=tmp_path / "options.json",
    )
    assert "omni.app.empty.kit" in command[1]
    assert "omni.kit.converter.hoops_core" in command
    assert "--no-window" in command
    assert "Building.SLDASM" in " ".join(command)
    assert hoops_name == hoops.name
    assert core_name == core.name


def test_operator_gui_has_separate_explicit_system_and_llm_controls() -> None:
    dashboard = (
        Path(__file__).resolve().parents[2]
        / "source/extensions/radcounter.isaac/radcounter/isaac/ui/dashboard.py"
    ).read_text(encoding="utf-8")
    assert "SYSTEM CONFIGURATION / 構成" in dashboard
    assert '"Environment"' in dashboard
    assert '"Robot"' in dashboard
    assert '"Detector"' in dashboard
    assert "選択した構成を適用 / Apply" in dashboard
    assert "NATURAL LANGUAGE COMMAND / ROBOT LLM" in dashboard
    assert "構成変更は下の選択欄" in dashboard
    assert dashboard.index("NATURAL LANGUAGE COMMAND / ROBOT LLM") < dashboard.index(
        "self._build_system_selector()"
    )
    assert "save_active_selection(" in dashboard


def test_decommissioning_assets_use_environment_ground_anchors() -> None:
    compositor = (
        Path(__file__).resolve().parents[2]
        / "source/extensions/radcounter.isaac/radcounter/isaac/system_profile.py"
    ).read_text(encoding="utf-8")
    assert '(config.countermeasure_root, "ground-primary")' in compositor
    assert '(config.measurement_root, "ground-secondary")' in compositor
    assert '"rad:spawn:environmentId"' in compositor
