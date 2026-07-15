"""uv-installed command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from radcounter.core.logging import JsonlEventLogger, build_manifest, write_manifest
from radcounter.core.models import (
    DetectorSpec,
    EmissionLine,
    IsotopeSpec,
    PointSourceState,
    load_scenario,
)
from radcounter.core.radiation import AnalyticTransportBackend, MaterialTable, RadiationForwardModel
from radcounter.core.radiation.rng import SeedManager
from radcounter.core.sensors import OmnidirectionalCounter


def validate_main(argv: list[str] | None = None) -> int:
    """Validate one scenario before starting an external simulator."""

    parser = argparse.ArgumentParser(description="Validate a RadCounterSim scenario")
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args(argv)
    scenario = load_scenario(args.scenario)
    print(json.dumps({"valid": True, "scenario_id": scenario.scenario_id}, sort_keys=True))
    return 0


def headless_main(argv: list[str] | None = None) -> int:
    """Run the initial analytic point-source measurement scenario."""

    parser = argparse.ArgumentParser(description="Run a RadCounterSim analytic scenario")
    parser.add_argument("scenario", type=Path)
    args = parser.parse_args(argv)
    scenario_path = args.scenario.resolve()
    config_bytes = scenario_path.read_bytes()
    scenario = load_scenario(scenario_path)
    output = (
        Path(scenario.runtime.output_directory)
        / scenario.scenario_id
        / f"seed-{scenario.runtime.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(
        output / "manifest.json",
        build_manifest(
            repository_root=Path(__file__).resolve().parents[2],
            config_bytes=config_bytes,
            run_seed=scenario.runtime.seed,
        ),
    )
    (output / "resolved_config.json").write_text(
        scenario.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    logger = JsonlEventLogger(output / "events.jsonl")
    isotopes = {
        item.isotope_id: IsotopeSpec(
            item.isotope_id,
            tuple(
                EmissionLine(line.energy_keV, line.photons_per_decay)
                for line in item.emission_lines
            ),
        )
        for item in scenario.isotopes
    }
    sources = [
        PointSourceState(
            item.source_id,
            np.asarray(item.position_world_m),
            item.activity_bq,
            item.isotope_id,
            item.enabled,
        )
        for item in scenario.point_sources
    ]
    detector_cfg = scenario.detector
    detector = DetectorSpec(
        detector_cfg.detector_id,
        np.asarray(detector_cfg.energy_bin_edges_keV),
        np.asarray(detector_cfg.efficiency_energy_keV),
        np.asarray(detector_cfg.intrinsic_efficiency),
        np.asarray(detector_cfg.background_cps_per_bin),
        detector_cfg.dead_time_s,
    )
    transport = AnalyticTransportBackend()
    transport.build_scene([], MaterialTable(()))
    model = RadiationForwardModel(
        transport, minimum_distance_m=scenario.radiation.minimum_distance_m
    )
    sensor = OmnidirectionalCounter(
        detector, SeedManager(scenario.runtime.seed).generator(f"detector/{detector.detector_id}")
    )
    records: list[dict[str, object]] = []
    sim_time_s = 0.0
    for index, pose in enumerate(scenario.measurement_poses):
        position = np.asarray(pose.position_world_m, dtype=np.float64)
        prediction = model.predict_point_count_rate(position[None, :], sources, isotopes, detector)
        measurement_id = sensor.start_measurement(
            timestamp_sim_s=sim_time_s,
            duration_s=pose.duration_s,
            position_world_m=position,
            orientation_world_wxyz=np.asarray(pose.orientation_world_wxyz),
            expected_rate_cps_per_bin=prediction.count_rate_cps_per_bin[0],
            scene_revision=transport.commit_updates(),
        )
        sim_time_s += pose.duration_s
        sensor.update(sim_time_s)
        measurement = sensor.get_latest()
        if measurement is None:
            raise RuntimeError("measurement did not finalize")
        record = {
            "measurement_id": measurement_id,
            "pose_index": index,
            "position_world_m": position.tolist(),
            "duration_s": pose.duration_s,
            "expected_rate_cps_per_bin": prediction.count_rate_cps_per_bin[0].tolist(),
            "counts_per_bin": measurement.counts_per_bin.tolist(),
        }
        records.append(record)
        logger.log("measurement_completed", sim_time_s=sim_time_s, details=record)
    (output / "measurements.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    logger.log("episode_completed", sim_time_s=sim_time_s, details={"measurements": len(records)})
    print(json.dumps({"status": "completed", "output": str(output)}, sort_keys=True))
    return 0
