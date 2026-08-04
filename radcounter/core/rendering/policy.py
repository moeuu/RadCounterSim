"""GPU-aware visual quality policy with bounded adaptive degradation."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass

from radcounter.core.rendering.models import (
    RenderMode,
    RendererPolicyConfig,
    RenderPurpose,
    RenderQualityTier,
)


@dataclass(frozen=True)
class GpuCapabilities:
    name: str
    vram_gb: float
    rtx_available: bool
    gpu_count: int = 1
    free_vram_gb: float | None = None


@dataclass(frozen=True)
class RenderBudget:
    tier: RenderQualityTier
    mode: RenderMode
    resolution_scale: float
    path_samples_per_pixel: int
    max_bounces: int
    denoiser_enabled: bool
    texture_budget_mb: int
    lod_bias: int
    max_visual_tiles: int
    volumetrics_enabled: bool
    effect_particle_scale: float
    shadow_quality: int
    render_product_rate_scale: float


_RTX_NAME = re.compile(
    r"(RTX|A2\b|A10\b|A16\b|A30\b|A40\b|A5000|A6000|L4\b|L20\b|L40\b|ORIN)",
    re.IGNORECASE,
)


def probe_gpu_capabilities() -> GpuCapabilities:
    """Probe NVIDIA hardware without importing a CUDA Python package."""

    override = os.environ.get("RADCOUNTER_GPU_VRAM_GB")
    if override is not None:
        name = os.environ.get("RADCOUNTER_GPU_NAME", "environment override")
        rtx = os.environ.get("RADCOUNTER_GPU_RTX", "1") not in {"0", "false", "False"}
        free = float(os.environ.get("RADCOUNTER_GPU_FREE_VRAM_GB", override))
        return GpuCapabilities(
            name=name,
            vram_gb=float(override),
            rtx_available=rtx,
            free_vram_gb=free,
        )
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return GpuCapabilities(name="unavailable", vram_gb=0.0, rtx_available=False, gpu_count=0)
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    parsed: list[tuple[str, float, float]] = []
    for row in rows:
        values = [value.strip() for value in row.rsplit(",", 2)]
        if len(values) == 3:
            parsed.append((values[0], float(values[1]) / 1024.0, float(values[2]) / 1024.0))
    if not parsed:
        return GpuCapabilities(name="unavailable", vram_gb=0.0, rtx_available=False, gpu_count=0)
    name, memory, free = max(parsed, key=lambda item: item[1])
    forced_rtx = os.environ.get("RADCOUNTER_GPU_RTX")
    rtx = bool(_RTX_NAME.search(name)) if forced_rtx is None else forced_rtx != "0"
    return GpuCapabilities(
        name=name,
        vram_gb=memory,
        rtx_available=rtx,
        gpu_count=len(parsed),
        free_vram_gb=free,
    )


def select_quality_tier(capabilities: GpuCapabilities) -> RenderQualityTier:
    forced = os.environ.get("RADCOUNTER_RENDER_TIER")
    if forced:
        return RenderQualityTier(forced.lower())
    available = min(
        capabilities.vram_gb,
        capabilities.free_vram_gb
        if capabilities.free_vram_gb is not None
        else capabilities.vram_gb,
    )
    if capabilities.rtx_available and available >= 16.0:
        return RenderQualityTier.STRONG
    if capabilities.rtx_available and available >= 8.0:
        return RenderQualityTier.BALANCED
    if capabilities.rtx_available and available >= 3.5:
        return RenderQualityTier.WEAK
    return RenderQualityTier.FALLBACK


def budget_for_tier(
    tier: RenderQualityTier,
    config: RendererPolicyConfig,
    purpose: RenderPurpose,
) -> RenderBudget:
    presets = {
        RenderQualityTier.STRONG: dict(
            resolution_scale=1.0,
            path_samples_per_pixel=64,
            max_bounces=6,
            denoiser_enabled=True,
            texture_budget_mb=8192,
            lod_bias=0,
            max_visual_tiles=180,
            volumetrics_enabled=True,
            effect_particle_scale=1.0,
            shadow_quality=3,
            render_product_rate_scale=1.0,
        ),
        RenderQualityTier.BALANCED: dict(
            resolution_scale=0.85,
            path_samples_per_pixel=16,
            max_bounces=3,
            denoiser_enabled=True,
            texture_budget_mb=4096,
            lod_bias=0,
            max_visual_tiles=100,
            volumetrics_enabled=True,
            effect_particle_scale=0.5,
            shadow_quality=2,
            render_product_rate_scale=0.75,
        ),
        RenderQualityTier.WEAK: dict(
            resolution_scale=0.65,
            path_samples_per_pixel=1,
            max_bounces=1,
            denoiser_enabled=True,
            texture_budget_mb=1536,
            lod_bias=1,
            max_visual_tiles=48,
            volumetrics_enabled=False,
            effect_particle_scale=0.2,
            shadow_quality=1,
            render_product_rate_scale=0.5,
        ),
        RenderQualityTier.FALLBACK: dict(
            resolution_scale=0.5,
            path_samples_per_pixel=1,
            max_bounces=1,
            denoiser_enabled=False,
            texture_budget_mb=512,
            lod_bias=2,
            max_visual_tiles=16,
            volumetrics_enabled=False,
            effect_particle_scale=0.05,
            shadow_quality=0,
            render_product_rate_scale=0.25,
        ),
    }
    if config.mode is not RenderMode.AUTO:
        mode = config.mode
    elif purpose is RenderPurpose.CAPTURE and config.allow_path_tracing_for_capture and tier in {
        RenderQualityTier.STRONG,
        RenderQualityTier.BALANCED,
    }:
        mode = RenderMode.PATH_TRACING
    elif tier is RenderQualityTier.FALLBACK:
        mode = RenderMode.STORM
    else:
        mode = RenderMode.RTX_REALTIME
    return RenderBudget(tier=tier, mode=mode, **presets[tier])


def choose_render_budget(
    config: RendererPolicyConfig,
    purpose: RenderPurpose = RenderPurpose.INTERACTIVE,
    capabilities: GpuCapabilities | None = None,
) -> RenderBudget:
    capabilities = capabilities or probe_gpu_capabilities()
    tier = config.forced_tier or select_quality_tier(capabilities)
    if not capabilities.rtx_available and config.mode is RenderMode.AUTO:
        tier = RenderQualityTier.FALLBACK
    return budget_for_tier(tier, config, purpose)
