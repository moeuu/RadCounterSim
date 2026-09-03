"""Isaac/Embree vertical-slice execution gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source/extensions/radcounter.isaac"))


def main() -> int:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import omni.usd
        from radcounter.isaac.physics import Pose3D, UsdSceneStateEditor
        from radcounter.isaac.planning import IsaacActionCandidateGenerator
        from radcounter.isaac.robot import DisposalDisposition, disposal_configuration
        from radcounter.isaac.runtime.simulation import IsaacRadiationSimulation

        from radcounter.core.experiments import EvidenceClass

        context = omni.usd.get_context()
        stage_path = ROOT / "assets/environments/radcounter_vertical_slice.usda"
        if not context.open_stage(str(stage_path)):
            raise AssertionError("vertical-slice USD did not open")
        for _ in range(8):
            app.update()
        stage = context.get_stage()
        disposal = disposal_configuration(stage, "/World/DisposalZone")
        assert disposal.disposition is DisposalDisposition.SHIELDED_STORAGE
        assert disposal.storage_prim_path == "/World/DisposalStorage"
        assert len(disposal.storage_geometry_paths) == 4
        simulation = IsaacRadiationSimulation.from_config(
            stage,
            ROOT / "configs/scenarios/vertical_slice.runtime.json",
        )
        generator = IsaacActionCandidateGenerator(stage, simulation)
        np.testing.assert_allclose(
            generator._center(stage.GetPrimAtPath("/World/LeadShield")),
            (4.8, 2.8, 0.9),
        )
        np.testing.assert_allclose(
            generator._center(stage.GetPrimAtPath("/World/DisposalZone")),
            (-4.6, 2.6, 0.02),
        )
        np.testing.assert_allclose(
            generator._grasp_from_root(
                stage.GetPrimAtPath("/World/HiddenContaminatedDrum"), "GraspFrame"
            ),
            (0.0, 0.0, 0.05),
        )
        protected = "/World/DetectorStations/Protected"
        hidden = "/World/HiddenContaminatedDrum"
        before_prediction = simulation.expected_spectrum_components(
            detector_paths=[protected], source_paths=[hidden]
        )[protected]
        before = float(np.sum(before_prediction.total_cps_per_bin))
        assert before_prediction.buildup_model_name == "primary_only"
        np.testing.assert_allclose(
            before_prediction.corrected_source_cps_per_bin,
            before_prediction.primary_cps_per_bin,
        )
        np.testing.assert_allclose(
            before_prediction.total_cps_per_bin,
            before_prediction.corrected_source_cps_per_bin
            + before_prediction.background_cps_per_bin,
        )
        edit_record = UsdSceneStateEditor().set_geometry_pose(
            "/World/LeadShield",
            Pose3D(
                np.asarray([1.2, 0.0, 0.9]),
                np.asarray([0.0, 0.0, 0.0, 1.0]),
            ),
        )
        changed = simulation.synchronize()
        after = simulation.expected_rates(detector_paths=[protected], source_paths=[hidden])[
            protected
        ]
        records = simulation.measure(detector_paths=[protected])
        surface = next(
            source
            for source in simulation.sources
            if source.prim_path == "/World/ContaminatedFloor"
        )
        assert len(simulation.sources) == 2
        assert surface.source_type == "surface"
        assert len(surface.positions_m) == 994
        assert np.array_equal(surface.element_indices, np.arange(994))
        assert np.isclose(surface.total_activity_bq, 8.0e5)
        assert hidden not in simulation.belief_source_paths
        public_response = simulation.public_source_response()
        assert public_response.source_paths == ("/World/ContaminatedFloor",)
        assert public_response.source_rate_cps_per_bq.shape == (4, 1)
        assert np.all(public_response.source_rate_cps_per_bq > 0.0)
        assert public_response.template_kinds == ("uniform_area_on_visible_surface",)
        try:
            simulation.public_source_response(source_paths=[hidden])
        except ValueError as error:
            assert "hidden sources" in str(error)
        else:
            raise AssertionError("hidden source was exposed as an estimator candidate")
        assert edit_record.evidence_class is EvidenceClass.KINEMATIC_SCENE_EDIT
        assert edit_record.revision == 1
        shield = stage.GetPrimAtPath("/World/LeadShield")
        assert shield.GetAttribute("rad:evidence:class").Get() == "kinematic_scene_edit"
        assert any(path.startswith("/World/LeadShield/") for path in changed), changed
        assert after < before * 0.25, (before, after)
        assert records[0].counts >= 0
        print(
            json.dumps(
                {
                    "before_cps": before,
                    "after_cps": after,
                    "changed": changed,
                    "public_response_shape": public_response.source_rate_cps_per_bq.shape,
                    "storage_geometry": disposal.storage_geometry_paths,
                }
            )
        )
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
