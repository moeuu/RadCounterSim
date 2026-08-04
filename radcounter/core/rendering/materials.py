"""Default physically based material parameters for nuclear facilities."""

from radcounter.core.rendering.models import PbrMaterialConfig


def nuclear_facility_materials() -> tuple[PbrMaterialConfig, ...]:
    return (
        PbrMaterialConfig(
            id="concrete",
            display_name="Aged concrete",
            base_color_srgb=(0.46, 0.45, 0.41),
            roughness=0.88,
        ),
        PbrMaterialConfig(
            id="epoxy_floor",
            display_name="Industrial epoxy floor",
            base_color_srgb=(0.22, 0.31, 0.29),
            roughness=0.34,
            clearcoat=0.18,
        ),
        PbrMaterialConfig(
            id="wet_epoxy",
            display_name="Wet epoxy floor",
            base_color_srgb=(0.12, 0.19, 0.18),
            roughness=0.08,
            clearcoat=0.85,
            clearcoat_roughness=0.03,
        ),
        PbrMaterialConfig(
            id="stainless_steel",
            display_name="Brushed stainless steel",
            base_color_srgb=(0.58, 0.61, 0.62),
            roughness=0.28,
            metallic=0.96,
        ),
        PbrMaterialConfig(
            id="painted_steel",
            display_name="Painted structural steel",
            base_color_srgb=(0.31, 0.37, 0.34),
            roughness=0.47,
            metallic=0.08,
        ),
        PbrMaterialConfig(
            id="lead",
            display_name="Oxidized lead shielding",
            base_color_srgb=(0.22, 0.24, 0.27),
            roughness=0.67,
            metallic=0.72,
        ),
        PbrMaterialConfig(
            id="rust",
            display_name="Iron oxide corrosion",
            base_color_srgb=(0.39, 0.14, 0.055),
            roughness=0.92,
            metallic=0.08,
        ),
        PbrMaterialConfig(
            id="dust",
            display_name="Settled concrete dust",
            base_color_srgb=(0.57, 0.54, 0.47),
            roughness=0.98,
        ),
        PbrMaterialConfig(
            id="rubber",
            display_name="Robot tire rubber",
            base_color_srgb=(0.025, 0.027, 0.025),
            roughness=0.82,
        ),
        PbrMaterialConfig(
            id="water",
            display_name="Contaminated process water",
            base_color_srgb=(0.18, 0.29, 0.24),
            roughness=0.04,
            opacity=0.58,
            ior=1.333,
            clearcoat=1.0,
            clearcoat_roughness=0.02,
        ),
    )
