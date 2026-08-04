"""Configuration models for facility digital-twin rendering.

These models are Isaac Sim independent.  Geometry used for radiation and
collision remains owned by :mod:`radcounter.core.environment`; this module
describes the visual twin and sensor image-formation layer only.
"""

from __future__ import annotations

import os
import urllib.parse
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DigitalTwinSourceType(StrEnum):
    PHOTOGRAMMETRY = "photogrammetry"
    LIDAR = "lidar"
    CAD = "cad"
    BIM = "bim"
    HYBRID = "hybrid"


class RenderMode(StrEnum):
    AUTO = "auto"
    RTX_REALTIME = "rtx_realtime"
    PATH_TRACING = "path_tracing"
    STORM = "storm"


class RenderPurpose(StrEnum):
    INTERACTIVE = "interactive"
    SENSOR = "sensor"
    CAPTURE = "capture"


class RenderQualityTier(StrEnum):
    STRONG = "strong"
    BALANCED = "balanced"
    WEAK = "weak"
    FALLBACK = "fallback"


class RenderProductKind(StrEnum):
    RGB = "rgb"
    DEPTH = "depth"
    NORMALS = "normals"
    SEMANTIC = "semantic"
    INSTANCE = "instance"
    LIDAR = "lidar"
    RADIATION = "radiation"


class LightType(StrEnum):
    DOME = "dome"
    RECT = "rect"
    DISK = "disk"
    SPHERE = "sphere"
    DISTANT = "distant"


class DigitalTwinAssetConfig(_StrictModel):
    visual_uri: str | None = None
    physics_manifest_uri: str | None = None
    source_type: DigitalTwinSourceType = DigitalTwinSourceType.HYBRID
    prim_path: str = "/World/DigitalTwin"
    source_meters_per_unit: float | None = Field(default=None, gt=0.0)
    source_up_axis: str = Field(default="auto", pattern="^(auto|X|Y|Z)$")
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale: float = Field(default=1.0, gt=0.0)
    preserve_source_materials: bool = True
    instanceable_patterns: tuple[str, ...] = ()
    texture_search_paths: tuple[str, ...] = ()
    cache_directory: str = ".cache/radcounter/digital-twin"

    @model_validator(mode="after")
    def require_visual_or_manifest(self) -> DigitalTwinAssetConfig:
        if self.visual_uri is None and self.physics_manifest_uri is None:
            raise ValueError("visual_uri or physics_manifest_uri is required")
        return self


class PbrTextureSetConfig(_StrictModel):
    base_color_uri: str | None = None
    normal_uri: str | None = None
    roughness_uri: str | None = None
    metallic_uri: str | None = None
    opacity_uri: str | None = None
    occlusion_uri: str | None = None


class PbrMaterialConfig(_StrictModel):
    id: str
    display_name: str | None = None
    base_color_srgb: tuple[float, float, float] = (0.5, 0.5, 0.5)
    roughness: float = Field(default=0.6, ge=0.0, le=1.0)
    metallic: float = Field(default=0.0, ge=0.0, le=1.0)
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    ior: float = Field(default=1.5, ge=1.0, le=3.0)
    clearcoat: float = Field(default=0.0, ge=0.0, le=1.0)
    clearcoat_roughness: float = Field(default=0.1, ge=0.0, le=1.0)
    emissive_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    textures: PbrTextureSetConfig = PbrTextureSetConfig()


class MaterialBindingRuleConfig(_StrictModel):
    pattern: str
    material_id: str
    match: str = Field(default="material_id", pattern="^(material_id|prim_path|prim_name)$")
    override_existing: bool = False


class FacilityLightConfig(_StrictModel):
    id: str
    light_type: LightType
    prim_path: str | None = None
    enabled: bool = True
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = Field(default=1000.0, ge=0.0)
    exposure: float = 0.0
    color_temperature_k: float | None = Field(default=None, ge=1000.0, le=20000.0)
    width_m: float = Field(default=1.0, gt=0.0)
    height_m: float = Field(default=1.0, gt=0.0)
    radius_m: float = Field(default=0.1, gt=0.0)
    angle_deg: float = Field(default=0.53, gt=0.0, le=180.0)
    hdri_uri: str | None = None


class FacilityLightingConfig(_StrictModel):
    enabled: bool = True
    calibration_id: str = "uncalibrated"
    exposure_compensation: float = 0.0
    lights: tuple[FacilityLightConfig, ...] = ()


class EnvironmentEffectsConfig(_StrictModel):
    enabled: bool = True
    bounds_min_m: tuple[float, float, float] = (-10.0, -10.0, 0.0)
    bounds_max_m: tuple[float, float, float] = (10.0, 10.0, 8.0)
    fog_density: float = Field(default=0.0, ge=0.0, le=1.0)
    fog_color_rgb: tuple[float, float, float] = (0.65, 0.68, 0.66)
    dust_particle_count: int = Field(default=0, ge=0, le=1_000_000)
    dust_particle_size_m: tuple[float, float] = (0.001, 0.008)
    steam_particle_count: int = Field(default=0, ge=0, le=250_000)
    spray_particle_count: int = Field(default=3000, ge=0, le=250_000)
    wet_trace_enabled: bool = True
    contaminated_water_tint_rgb: tuple[float, float, float] = (0.42, 0.55, 0.36)
    random_seed: int = 20260715


class HighDoseCameraConfig(_StrictModel):
    enabled: bool = True
    onset_dose_rate_gy_h: float = Field(default=0.1, gt=0.0)
    transient_hits_per_mpix_s_per_gy_h: float = Field(default=90.0, ge=0.0)
    permanent_hot_pixel_fraction_per_gy: float = Field(default=2.0e-6, ge=0.0)
    max_permanent_hot_pixel_fraction: float = Field(default=0.01, ge=0.0, le=0.2)
    streak_decay_px: float = Field(default=22.0, gt=0.0)
    bloom_gain: float = Field(default=0.35, ge=0.0)
    desaturation_half_dose_rate_gy_h: float = Field(default=20.0, gt=0.0)
    read_noise_std: float = Field(default=0.006, ge=0.0)
    drop_rate_per_s: float = Field(default=0.015, ge=0.0)
    max_transient_hits_per_frame: int = Field(default=5000, ge=0)
    random_seed: int = 20260715


class RenderProductConfig(_StrictModel):
    id: str
    kind: RenderProductKind
    sensor_prim_path: str | None = None
    resolution_px: tuple[int, int] = (1280, 720)
    update_rate_hz: float = Field(default=10.0, gt=0.0)
    enabled: bool = True
    annotator: str | None = None
    apply_high_dose_effects: bool = False

    @model_validator(mode="after")
    def require_sensor_path(self) -> RenderProductConfig:
        if self.kind not in {RenderProductKind.RADIATION} and self.sensor_prim_path is None:
            raise ValueError(f"sensor_prim_path is required for {self.kind.value}")
        return self


class RendererPolicyConfig(_StrictModel):
    mode: RenderMode = RenderMode.AUTO
    allow_path_tracing_for_capture: bool = True
    forced_tier: RenderQualityTier | None = None
    target_frame_rate_hz: float = Field(default=30.0, gt=0.0)
    base_resolution_px: tuple[int, int] = (1280, 720)
    adaptive: bool = True
    slow_frame_hysteresis: int = Field(default=45, ge=1)
    fast_frame_hysteresis: int = Field(default=240, ge=1)


class DigitalTwinRenderingConfig(_StrictModel):
    schema_version: str = "1.0"
    asset: DigitalTwinAssetConfig
    material_library_uri: str | None = None
    materials: tuple[PbrMaterialConfig, ...] = ()
    material_bindings: tuple[MaterialBindingRuleConfig, ...] = ()
    lighting: FacilityLightingConfig = FacilityLightingConfig()
    effects: EnvironmentEffectsConfig = EnvironmentEffectsConfig()
    camera_radiation: HighDoseCameraConfig = HighDoseCameraConfig()
    renderer: RendererPolicyConfig = RendererPolicyConfig()
    render_products: tuple[RenderProductConfig, ...] = ()
    base_directory: str = "."


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _is_relative_file_uri(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return not parsed.scheme and not Path(value).is_absolute()


def _resolve_texture_set(raw: dict[str, Any], base: Path) -> dict[str, Any]:
    result = dict(raw)
    textures = dict(result.get("textures", {}))
    for key, value in textures.items():
        if value and _is_relative_file_uri(value):
            textures[key] = str((base / value).resolve())
    result["textures"] = textures
    return result


def load_digital_twin_config(path: str | Path) -> DigitalTwinRenderingConfig:
    """Load a rendering descriptor and resolve material assets relative to it."""

    descriptor = Path(path).expanduser().resolve()
    payload = _expand(yaml.safe_load(descriptor.read_text(encoding="utf-8")))
    raw = dict(payload.get("digital_twin", payload))
    raw["base_directory"] = str(descriptor.parent)

    asset = dict(raw.get("asset", {}))
    for key in ("visual_uri", "physics_manifest_uri", "cache_directory"):
        value = asset.get(key)
        if value and _is_relative_file_uri(value):
            asset[key] = str((descriptor.parent / value).resolve())
    asset["texture_search_paths"] = [
        str((descriptor.parent / value).resolve()) if _is_relative_file_uri(value) else value
        for value in asset.get("texture_search_paths", [])
    ]
    raw["asset"] = asset

    lighting = dict(raw.get("lighting", {}))
    lights = []
    for light in lighting.get("lights", []):
        resolved = dict(light)
        hdri = resolved.get("hdri_uri")
        if hdri and _is_relative_file_uri(hdri):
            resolved["hdri_uri"] = str((descriptor.parent / hdri).resolve())
        lights.append(resolved)
    lighting["lights"] = lights
    raw["lighting"] = lighting

    inline_materials = [
        _resolve_texture_set(item, descriptor.parent) for item in raw.get("materials", [])
    ]
    library_uri = raw.get("material_library_uri")
    if library_uri:
        library_path = Path(library_uri)
        if not library_path.is_absolute():
            library_path = descriptor.parent / library_path
        library_path = library_path.resolve()
        library_payload = _expand(yaml.safe_load(library_path.read_text(encoding="utf-8")))
        library_items = library_payload.get("materials", library_payload)
        inline_materials = [
            _resolve_texture_set(item, library_path.parent) for item in library_items
        ] + inline_materials
        raw["material_library_uri"] = str(library_path)
    raw["materials"] = inline_materials
    return DigitalTwinRenderingConfig.model_validate(raw)
