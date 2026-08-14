"""Calibrated facility, robot, and HDRI lighting authoring."""

from __future__ import annotations

import re

from radcounter.core.rendering import FacilityLightingConfig, LightType


def _identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


class FacilityLightingAuthor:
    def __init__(self, stage, root_path: str = "/World/RadCounterLighting") -> None:
        self.stage = stage
        self.root_path = root_path

    def author(self, config: FacilityLightingConfig) -> tuple[str, ...]:
        from pxr import Gf, Sdf, UsdGeom, UsdLux

        if not config.enabled:
            return ()
        root = self.stage.DefinePrim(self.root_path, "Xform")
        root.CreateAttribute("rad:lighting:calibrationId", Sdf.ValueTypeNames.String).Set(
            config.calibration_id
        )
        paths = []
        schemas = {
            LightType.DOME: UsdLux.DomeLight,
            LightType.RECT: UsdLux.RectLight,
            LightType.DISK: UsdLux.DiskLight,
            LightType.SPHERE: UsdLux.SphereLight,
            LightType.DISTANT: UsdLux.DistantLight,
        }
        for item in config.lights:
            if not item.enabled:
                continue
            path = item.prim_path or f"{self.root_path}/{_identifier(item.id)}"
            light = schemas[item.light_type].Define(self.stage, path)
            prim = light.GetPrim()
            prim.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(item.intensity)
            prim.CreateAttribute("inputs:exposure", Sdf.ValueTypeNames.Float).Set(
                item.exposure + config.exposure_compensation
            )
            prim.CreateAttribute("inputs:color", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*item.color_rgb)
            )
            if item.color_temperature_k is not None:
                prim.CreateAttribute("inputs:enableColorTemperature", Sdf.ValueTypeNames.Bool).Set(
                    True
                )
                prim.CreateAttribute("inputs:colorTemperature", Sdf.ValueTypeNames.Float).Set(
                    item.color_temperature_k
                )
            if item.light_type is LightType.RECT:
                prim.CreateAttribute("inputs:width", Sdf.ValueTypeNames.Float).Set(item.width_m)
                prim.CreateAttribute("inputs:height", Sdf.ValueTypeNames.Float).Set(item.height_m)
            elif item.light_type in {LightType.DISK, LightType.SPHERE}:
                prim.CreateAttribute("inputs:radius", Sdf.ValueTypeNames.Float).Set(item.radius_m)
            elif item.light_type is LightType.DISTANT:
                prim.CreateAttribute("inputs:angle", Sdf.ValueTypeNames.Float).Set(item.angle_deg)
            if item.light_type is LightType.DOME and item.hdri_uri:
                prim.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(
                    Sdf.AssetPath(item.hdri_uri)
                )
                prim.CreateAttribute("inputs:texture:format", Sdf.ValueTypeNames.Token).Set(
                    "latlong"
                )
            xformable = UsdGeom.Xformable(prim)
            xformable.ClearXformOpOrder()
            xformable.AddTranslateOp().Set(Gf.Vec3d(*item.translation_m))
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(*item.rotation_rpy_deg))
            paths.append(path)
        return tuple(paths)
