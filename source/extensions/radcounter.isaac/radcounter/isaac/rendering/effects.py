"""Bounded visual effects for nuclear decontamination environments."""

from __future__ import annotations

import numpy as np

from radcounter.core.rendering import EnvironmentEffectsConfig, RenderBudget


class FacilityEffectsAuthor:
    def __init__(
        self,
        stage,
        config: EnvironmentEffectsConfig,
        budget: RenderBudget,
        root_path: str = "/World/RadCounterEffects",
    ) -> None:
        self.stage = stage
        self.config = config
        self.budget = budget
        self.root_path = root_path
        self._rng = np.random.default_rng(config.random_seed)

    def author_static(self) -> dict[str, int | bool]:
        import carb.settings
        from pxr import Gf, Sdf, UsdGeom

        root = self.stage.DefinePrim(self.root_path, "Scope")
        root.CreateAttribute("rad:effects:bounded", Sdf.ValueTypeNames.Bool).Set(True)
        settings = carb.settings.get_settings()
        fog_enabled = (
            self.config.enabled
            and self.config.fog_density > 0.0
            and self.budget.volumetrics_enabled
        )
        settings.set_bool("/rtx/raytracing/fog/enabled", fog_enabled)
        settings.set_float("/rtx/raytracing/fog/density", self.config.fog_density)
        settings.set_float_array(
            "/rtx/raytracing/fog/color", list(self.config.fog_color_rgb)
        )
        dust_count = int(self.config.dust_particle_count * self.budget.effect_particle_scale)
        if dust_count:
            low = np.asarray(self.config.bounds_min_m)
            high = np.asarray(self.config.bounds_max_m)
            positions = self._rng.uniform(low, high, size=(dust_count, 3))
            sizes = self._rng.uniform(*self.config.dust_particle_size_m, size=dust_count)
            points = UsdGeom.Points.Define(self.stage, f"{self.root_path}/Dust")
            points.CreatePointsAttr([Gf.Vec3f(*value) for value in positions])
            points.CreateWidthsAttr(sizes.astype(np.float32).tolist())
            points.GetPrim().CreateAttribute(
                "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
            ).Set([Gf.Vec3f(0.52, 0.49, 0.42)])
            points.GetPrim().CreateAttribute(
                "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray
            ).Set([0.32])
        return {"fog": fog_enabled, "dust_particles": dust_count}

    def update_spray(
        self,
        origin_m: tuple[float, float, float],
        direction: tuple[float, float, float],
        *,
        reach_m: float = 2.5,
        cone_radius_m: float = 0.35,
    ) -> int:
        from pxr import Gf, Sdf, UsdGeom

        count = int(self.config.spray_particle_count * self.budget.effect_particle_scale)
        path = f"{self.root_path}/WaterSpray"
        points = UsdGeom.Points.Define(self.stage, path)
        if not count:
            points.CreatePointsAttr([])
            return 0
        axis = np.asarray(direction, dtype=np.float64)
        norm = float(np.linalg.norm(axis))
        if norm == 0.0:
            raise ValueError("spray direction cannot be zero")
        axis /= norm
        reference = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(axis, reference))) > 0.9:
            reference = np.asarray([0.0, 1.0, 0.0])
        side_a = np.cross(axis, reference)
        side_a /= np.linalg.norm(side_a)
        side_b = np.cross(axis, side_a)
        t = self._rng.uniform(0.0, 1.0, size=count)
        radius = cone_radius_m * t * np.sqrt(self._rng.uniform(size=count))
        angle = self._rng.uniform(0.0, 2.0 * np.pi, size=count)
        positions = (
            np.asarray(origin_m)
            + t[:, None] * reach_m * axis
            + (radius * np.cos(angle))[:, None] * side_a
            + (radius * np.sin(angle))[:, None] * side_b
        )
        positions[:, 2] -= 0.5 * 9.81 * (0.18 * t) ** 2
        points.CreatePointsAttr([Gf.Vec3f(*value) for value in positions])
        points.CreateWidthsAttr(self._rng.uniform(0.002, 0.012, size=count).tolist())
        points.GetPrim().CreateAttribute(
            "primvars:displayColor", Sdf.ValueTypeNames.Color3fArray
        ).Set([Gf.Vec3f(0.58, 0.72, 0.78)])
        points.GetPrim().CreateAttribute(
            "primvars:displayOpacity", Sdf.ValueTypeNames.FloatArray
        ).Set([0.68])
        points.GetPrim().CreateAttribute("rad:effect:type", Sdf.ValueTypeNames.String).Set(
            "decontamination_water_spray"
        )
        return count

    def author_wet_trace(
        self,
        centerline_m: np.ndarray,
        *,
        width_m: float,
        material=None,
        z_offset_m: float = 0.002,
    ) -> str:
        from pxr import Gf, UsdGeom, UsdShade

        values = np.asarray(centerline_m, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
            raise ValueError("wet trace centerline must have shape (N, 3), N >= 2")
        tangent = np.gradient(values, axis=0)
        side = np.column_stack((-tangent[:, 1], tangent[:, 0], np.zeros(len(values))))
        norm = np.linalg.norm(side, axis=1)
        side[norm > 0.0] /= norm[norm > 0.0, None]
        vertices = np.empty((len(values) * 2, 3), dtype=np.float64)
        vertices[0::2] = values - side * width_m * 0.5
        vertices[1::2] = values + side * width_m * 0.5
        vertices[:, 2] += z_offset_m
        faces = []
        for index in range(len(values) - 1):
            start = index * 2
            faces.extend((start, start + 1, start + 3, start + 2))
        path = f"{self.root_path}/WetTraces/trace_{len(values)}_{abs(hash(values.tobytes())) % 100000}"
        mesh = UsdGeom.Mesh.Define(self.stage, path)
        mesh.CreatePointsAttr([Gf.Vec3f(*value) for value in vertices])
        mesh.CreateFaceVertexCountsAttr([4] * (len(values) - 1))
        mesh.CreateFaceVertexIndicesAttr(faces)
        mesh.CreateSubdivisionSchemeAttr("none")
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return path
