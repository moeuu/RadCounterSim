"""Isaac/Kit renderer settings controlled by a bounded GPU policy."""

from __future__ import annotations

from radcounter.core.rendering import (
    GpuCapabilities,
    RenderBudget,
    RendererPolicyConfig,
    RenderPurpose,
    RenderQualityTier,
    budget_for_tier,
    choose_render_budget,
)


class IsaacRenderController:
    _ORDER = (
        RenderQualityTier.FALLBACK,
        RenderQualityTier.WEAK,
        RenderQualityTier.BALANCED,
        RenderQualityTier.STRONG,
    )

    def __init__(
        self,
        config: RendererPolicyConfig,
        *,
        purpose: RenderPurpose = RenderPurpose.INTERACTIVE,
        capabilities: GpuCapabilities | None = None,
    ) -> None:
        self.config = config
        self.purpose = purpose
        self.capabilities = capabilities
        self.budget = choose_render_budget(config, purpose, capabilities)
        self._ewma_ms: float | None = None
        self._slow_frames = 0
        self._fast_frames = 0

    def apply(self) -> RenderBudget:
        import carb.settings

        settings = carb.settings.get_settings()
        renderer_name = {
            "path_tracing": "PathTracing",
            "rtx_realtime": "RaytracedLighting",
            "storm": "Storm",
        }.get(self.budget.mode.value, "RaytracedLighting")
        settings.set_string("/rtx/rendermode", renderer_name)
        settings.set_int("/rtx/pathtracing/spp", self.budget.path_samples_per_pixel)
        settings.set_int("/rtx/pathtracing/maxBounces", self.budget.max_bounces)
        settings.set_bool(
            "/rtx/pathtracing/optixDenoiser/enabled", self.budget.denoiser_enabled
        )
        settings.set_bool("/rtx-transient/resourcemanager/texturestreaming/enabled", True)
        settings.set_int(
            "/rtx-transient/resourcemanager/texturestreaming/memoryBudget",
            self.budget.texture_budget_mb,
        )
        settings.set_int("/rtx/hydra/textureMemoryBudget", self.budget.texture_budget_mb)
        settings.set_int("/rtx/raytracing/shadow/quality", self.budget.shadow_quality)
        settings.set_bool("/rtx/raytracing/fog/enabled", self.budget.volumetrics_enabled)
        return self.budget

    def observe_frame_time(self, frame_time_ms: float) -> RenderBudget | None:
        """Adapt only visual quality after sustained overload or headroom."""

        if not self.config.adaptive or frame_time_ms <= 0.0:
            return None
        alpha = 0.05
        self._ewma_ms = (
            frame_time_ms
            if self._ewma_ms is None
            else alpha * frame_time_ms + (1.0 - alpha) * self._ewma_ms
        )
        target_ms = 1000.0 / self.config.target_frame_rate_hz
        if self._ewma_ms > target_ms * 1.2:
            self._slow_frames += 1
            self._fast_frames = 0
        elif self._ewma_ms < target_ms * 0.72:
            self._fast_frames += 1
            self._slow_frames = 0
        else:
            self._slow_frames = self._fast_frames = 0
        index = self._ORDER.index(self.budget.tier)
        if self._slow_frames >= self.config.slow_frame_hysteresis and index > 0:
            self._set_tier(self._ORDER[index - 1])
            return self.apply()
        if self._fast_frames >= self.config.fast_frame_hysteresis and index < len(self._ORDER) - 1:
            self._set_tier(self._ORDER[index + 1])
            return self.apply()
        return None

    def _set_tier(self, tier: RenderQualityTier) -> None:
        self.budget = budget_for_tier(tier, self.config, self.purpose)
        self._slow_frames = self._fast_frames = 0
