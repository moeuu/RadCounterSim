"""Isaac USD/Embree primitive synchronization and fail-closed geometry gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "source/extensions/radcounter.isaac")]


def _tag(prim: object, sdf: object, material_id: str = "lead") -> None:
    prim.CreateAttribute("rad:material:id", sdf.ValueTypeNames.String, custom=True).Set(
        material_id
    )


def main() -> int:
    from pxr import Gf, Sdf, Usd, UsdGeom
    from radcounter.isaac.runtime.simulation import NativeStageTransport, RuntimeConfiguration

    stage = Usd.Stage.CreateInMemory()
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    sphere = UsdGeom.Sphere.Define(stage, "/World/Sphere")
    sphere.CreateRadiusAttr(1.0)
    _tag(sphere.GetPrim(), Sdf)
    cylinder = UsdGeom.Cylinder.Define(stage, "/World/Cylinder")
    cylinder.CreateRadiusAttr(1.0)
    cylinder.CreateHeightAttr(2.0)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(0.0, 3.0, 0.0))
    _tag(cylinder.GetPrim(), Sdf)

    container = UsdGeom.Xform.Define(stage, "/World/Container")
    container_prim = container.GetPrim()
    _tag(container_prim, Sdf)
    container_prim.CreateAttribute(
        "rad:material:containerOnly", Sdf.ValueTypeNames.Bool, custom=True
    ).Set(True)
    child = UsdGeom.Cube.Define(stage, "/World/Container/Child")
    child.CreateSizeAttr(0.5)
    child.AddTranslateOp().Set(Gf.Vec3d(0.0, -3.0, 0.0))
    _tag(child.GetPrim(), Sdf)

    configuration = RuntimeConfiguration(
        materials={"lead": (np.asarray((100.0, 1000.0)), np.asarray((2.0, 1.0)))},
        isotopes={},
        detectors={},
        duration_s=1.0,
        minimum_distance_m=0.01,
        seed=1,
    )
    transport = NativeStageTransport(stage, configuration)
    assert set(transport.geometry_paths) == {
        "/World/Sphere",
        "/World/Cylinder",
        "/World/Container/Child",
    }
    paths = transport.path_lengths(
        np.asarray(((-2.0, 0.0, 0.0), (-2.0, 3.0, 0.0), (0.0, 0.0, 0.0))),
        np.asarray(((2.0, 0.0, 0.0), (2.0, 3.0, 0.0), (2.0, 0.0, 0.0))),
    )
    np.testing.assert_allclose(paths[:, 0], (2.0, 2.0, 1.0), atol=4.0e-4)

    sphere.GetRadiusAttr().Set(0.5)
    changed = transport.synchronize_transforms()
    assert changed == ("/World/Sphere",)
    resized = transport.path_lengths(
        np.asarray(((-2.0, 0.0, 0.0),)), np.asarray(((2.0, 0.0, 0.0),))
    )[0, 0]
    assert np.isclose(resized, 1.0, atol=4.0e-4)

    sphere.AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 0.5))
    changed = transport.synchronize_transforms()
    assert changed == ("/World/Sphere",)
    scaled = transport.path_lengths(
        np.asarray(((-2.0, 0.0, 0.0),)), np.asarray(((2.0, 0.0, 0.0),))
    )[0, 0]
    assert np.isclose(scaled, 2.0, atol=5.0e-4)

    cone = UsdGeom.Cone.Define(stage, "/World/UnsupportedCone")
    _tag(cone.GetPrim(), Sdf)
    try:
        transport.synchronize_transforms()
    except ValueError as error:
        assert "unsupported USD type" in str(error)
    else:
        raise AssertionError("tagged unsupported geometry did not fail closed")
    assert "/World/UnsupportedCone" not in transport.geometry_paths

    print(
        json.dumps(
            {
                "geometry_paths": transport.geometry_paths,
                "sphere_path_m": scaled,
                "unsupported_geometry_rejected": True,
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
