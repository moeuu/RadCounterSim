import numpy as np
import pytest

from radcounter.core.rendering import (
    GpuCapabilities,
    HighDoseCameraConfig,
    HighDoseCameraModel,
    RendererPolicyConfig,
    RenderMode,
    RenderPurpose,
    RenderQualityTier,
    choose_render_budget,
    compare_depth,
    compare_rgb,
)


def test_strong_gpu_uses_path_tracing_only_for_capture() -> None:
    gpu = GpuCapabilities("RTX test", 24.0, True)
    config = RendererPolicyConfig()
    interactive = choose_render_budget(config, RenderPurpose.INTERACTIVE, gpu)
    capture = choose_render_budget(config, RenderPurpose.CAPTURE, gpu)
    assert interactive.tier is RenderQualityTier.STRONG
    assert interactive.mode is RenderMode.RTX_REALTIME
    assert capture.mode is RenderMode.PATH_TRACING
    assert capture.path_samples_per_pixel == 64


def test_weak_gpu_disables_volumetrics_and_reduces_products() -> None:
    budget = choose_render_budget(
        RendererPolicyConfig(),
        RenderPurpose.INTERACTIVE,
        GpuCapabilities("RTX test", 4.0, True),
    )
    assert budget.tier is RenderQualityTier.WEAK
    assert not budget.volumetrics_enabled
    assert budget.resolution_scale < 1.0
    assert budget.render_product_rate_scale < 1.0


def test_busy_strong_gpu_uses_free_memory_for_tier_selection() -> None:
    budget = choose_render_budget(
        RendererPolicyConfig(),
        RenderPurpose.INTERACTIVE,
        GpuCapabilities("RTX test", 24.0, True, free_vram_gb=3.0),
    )
    assert budget.tier is RenderQualityTier.FALLBACK
    assert budget.mode is RenderMode.STORM


def test_high_dose_camera_is_deterministic_and_accumulates_damage() -> None:
    config = HighDoseCameraConfig(
        random_seed=7,
        drop_rate_per_s=0.0,
        permanent_hot_pixel_fraction_per_gy=0.01,
        max_permanent_hot_pixel_fraction=0.1,
    )
    left = HighDoseCameraModel(config)
    right = HighDoseCameraModel(config)
    source = np.full((32, 48, 3), 80, dtype=np.uint8)
    result_left = left.process(source, dose_rate_gy_h=3600.0, exposure_s=1.0)
    result_right = right.process(source, dose_rate_gy_h=3600.0, exposure_s=1.0)
    assert np.array_equal(result_left.image, result_right.image)
    assert result_left.diagnostics["permanent_hot_pixels"] > 0
    assert left.cumulative_dose_gy == pytest.approx(1.0)


def test_rgb_and_depth_comparison_are_identity_for_equal_inputs() -> None:
    rgb = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
    rgb_result = compare_rgb(rgb, rgb)
    assert rgb_result.mae == 0.0
    assert rgb_result.psnr_db == float("inf")
    assert rgb_result.global_ssim == pytest.approx(1.0)
    depth = np.full((4, 4), 2.0)
    depth_result = compare_depth(depth, depth)
    assert depth_result.valid_fraction == 1.0
    assert depth_result.rmse_m == 0.0
    assert depth_result.delta_1_25 == 1.0
