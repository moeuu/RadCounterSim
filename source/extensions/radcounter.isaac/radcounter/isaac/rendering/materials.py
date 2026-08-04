"""USD Preview Surface material authoring and conservative binding."""

from __future__ import annotations

import re
from collections.abc import Iterable

from radcounter.core.rendering import MaterialBindingRuleConfig, PbrMaterialConfig


def _identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


class NuclearPbrMaterialLibrary:
    def __init__(self, stage, root_path: str = "/World/Looks/RadCounter") -> None:
        self.stage = stage
        self.root_path = root_path
        self.materials: dict[str, object] = {}

    def author(self, specs: Iterable[PbrMaterialConfig]) -> dict[str, object]:
        from pxr import Gf, Sdf, UsdShade

        self.stage.DefinePrim(self.root_path, "Scope")
        for spec in specs:
            path = f"{self.root_path}/{_identifier(spec.id)}"
            material = UsdShade.Material.Define(self.stage, path)
            shader = UsdShade.Shader.Define(self.stage, f"{path}/PreviewSurface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*spec.base_color_srgb)
            )
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(spec.roughness)
            shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(spec.metallic)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(spec.opacity)
            shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(spec.ior)
            shader.CreateInput("clearcoat", Sdf.ValueTypeNames.Float).Set(spec.clearcoat)
            shader.CreateInput("clearcoatRoughness", Sdf.ValueTypeNames.Float).Set(
                spec.clearcoat_roughness
            )
            shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
                Gf.Vec3f(*spec.emissive_color)
            )
            st_reader = self._st_reader(path)
            texture_inputs = {
                "base_color_uri": ("diffuseColor", "rgb", "sRGB"),
                "normal_uri": ("normal", "rgb", "raw"),
                "roughness_uri": ("roughness", "r", "raw"),
                "metallic_uri": ("metallic", "r", "raw"),
                "opacity_uri": ("opacity", "r", "raw"),
            }
            for field, (input_name, output_name, color_space) in texture_inputs.items():
                uri = getattr(spec.textures, field)
                if uri:
                    texture = self._texture(path, field, uri, color_space, st_reader)
                    shader.GetInput(input_name).ConnectToSource(
                        texture.ConnectableAPI(), output_name
                    )
            material.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface"
            )
            material.GetPrim().CreateAttribute(
                "rad:material:id", Sdf.ValueTypeNames.String
            ).Set(spec.id)
            self.materials[spec.id] = material
        return dict(self.materials)

    def bind(
        self,
        root_path: str,
        rules: Iterable[MaterialBindingRuleConfig],
        *,
        preserve_existing: bool = True,
    ) -> dict[str, int]:
        from pxr import UsdGeom, UsdShade

        rule_list = tuple(rules)
        counts = {material_id: 0 for material_id in self.materials}
        for prim in self.stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith(root_path) or not prim.IsA(UsdGeom.Gprim):
                continue
            relation = prim.GetRelationship("material:binding")
            has_binding = bool(relation and relation.GetTargets())
            material_attr = prim.GetAttribute("rad:material:id")
            material_id = material_attr.Get() if material_attr and material_attr.HasValue() else ""
            candidates = rule_list or tuple(
                MaterialBindingRuleConfig(pattern=value, material_id=value)
                for value in self.materials
            )
            for rule in candidates:
                if rule.match == "prim_path":
                    value = path
                elif rule.match == "prim_name":
                    value = prim.GetName()
                else:
                    value = str(material_id)
                if not re.fullmatch(rule.pattern, value) and value != rule.pattern:
                    continue
                material = self.materials.get(rule.material_id)
                if material is None:
                    raise KeyError(f"Unknown PBR material in binding rule: {rule.material_id}")
                if has_binding and preserve_existing and not rule.override_existing:
                    break
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
                counts[rule.material_id] += 1
                break
        return counts

    def material(self, material_id: str):
        return self.materials[material_id]

    def _st_reader(self, material_path: str):
        from pxr import Sdf, UsdShade

        reader = UsdShade.Shader.Define(self.stage, f"{material_path}/PrimvarReader_st")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        return reader

    def _texture(self, material_path: str, field: str, uri: str, color_space: str, st_reader):
        from pxr import Sdf, UsdShade

        texture = UsdShade.Shader.Define(
            self.stage, f"{material_path}/Texture_{_identifier(field)}"
        )
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(uri))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(color_space)
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            st_reader.ConnectableAPI(), "result"
        )
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        texture.CreateOutput("r", Sdf.ValueTypeNames.Float)
        return texture
