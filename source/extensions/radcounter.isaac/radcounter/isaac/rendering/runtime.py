"""End-to-end assembly of an OceanSim-grade nuclear visual twin."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from radcounter.core.rendering import (
    DigitalTwinRenderingConfig,
    RenderPurpose,
    nuclear_facility_materials,
)
from radcounter.isaac.rendering.effects import FacilityEffectsAuthor
from radcounter.isaac.rendering.ingestion import DigitalTwinIngestor, IngestionReport
from radcounter.isaac.rendering.lighting import FacilityLightingAuthor
from radcounter.isaac.rendering.materials import NuclearPbrMaterialLibrary
from radcounter.isaac.rendering.products import IsaacRenderProductManager
from radcounter.isaac.rendering.renderer import IsaacRenderController


@dataclass(frozen=True)
class DigitalTwinRuntimeReport:
    ingestion: IngestionReport
    renderer_tier: str
    renderer_mode: str
    material_bindings: dict[str, int]
    light_paths: tuple[str, ...]
    effects: dict[str, int | bool]
    render_products: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class NuclearDigitalTwinRuntime:
    def __init__(
        self,
        stage,
        config: DigitalTwinRenderingConfig,
        *,
        purpose: RenderPurpose = RenderPurpose.INTERACTIVE,
    ) -> None:
        self.stage = stage
        self.config = config
        self.renderer = IsaacRenderController(config.renderer, purpose=purpose)
        self.materials = NuclearPbrMaterialLibrary(stage)
        self.effects: FacilityEffectsAuthor | None = None
        self.products: IsaacRenderProductManager | None = None

    async def load(self) -> DigitalTwinRuntimeReport:
        budget = self.renderer.apply()
        ingestion = await DigitalTwinIngestor(
            self.stage, base_directory=self.config.base_directory
        ).mount(self.config.asset)
        material_specs = self.config.materials or nuclear_facility_materials()
        self.materials.author(material_specs)
        bindings = self.materials.bind(
            self.config.asset.prim_path,
            self.config.material_bindings,
            preserve_existing=self.config.asset.preserve_source_materials,
        )
        lights = FacilityLightingAuthor(self.stage).author(self.config.lighting)
        self.effects = FacilityEffectsAuthor(self.stage, self.config.effects, budget)
        effect_report = self.effects.author_static()
        self.products = IsaacRenderProductManager(
            self.config.render_products,
            budget,
            self.config.camera_radiation,
        )
        product_specs = self.products.create()
        return DigitalTwinRuntimeReport(
            ingestion=ingestion,
            renderer_tier=budget.tier.value,
            renderer_mode=budget.mode.value,
            material_bindings=bindings,
            light_paths=lights,
            effects=effect_report,
            render_products=tuple(product_specs),
        )

    def observe_frame_time(self, frame_time_ms: float) -> bool:
        changed = self.renderer.observe_frame_time(frame_time_ms)
        if changed is None:
            return False
        if self.products is not None:
            self.products.budget = changed
        if self.effects is not None:
            self.effects.budget = changed
        return True

    def close(self) -> None:
        if self.products is not None:
            self.products.close()
