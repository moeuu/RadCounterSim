from radcounter.core.performance import (
    AdaptiveWorkloadGovernor,
    GpuTier,
    HardwareInfo,
    QualityLevel,
    RateGate,
    RuntimeBudget,
    TileResidencyPlanner,
    classify_gpu_tier,
    profile_path_for_hardware,
)


def test_governor_degrades_and_recovers_with_hysteresis() -> None:
    governor = AdaptiveWorkloadGovernor(
        RuntimeBudget(
            target_fps=20.0,
            initial_quality=QualityLevel.BALANCED,
            sample_window=3,
            decision_cooldown_frames=0,
        )
    )
    for _ in range(3):
        governor.observe(0.1)
    assert governor.quality is QualityLevel.CONSERVATIVE
    for _ in range(3):
        governor.observe(0.01)
    assert governor.quality is QualityLevel.BALANCED


def test_tile_residency_is_bounded_for_multiple_robots() -> None:
    planner = TileResidencyPlanner(40.0, 4000.0, max_resident_tiles=12)
    selected = planner.select(((0.0, 0.0), (1200.0, -900.0)), radius_tiles=3)
    assert len(selected) == 12
    assert planner.logical_tile_count == 10000
    assert (0, 0) in selected


def test_rate_gate_keeps_subsystems_independent() -> None:
    gate = RateGate()
    assert gate.due("camera", 0.0, 2.0)
    assert not gate.due("camera", 0.4, 2.0)
    assert gate.due("camera", 0.5, 2.0)
    assert gate.due("lidar", 0.4, 1.0)


def test_gpu_tiers_and_profile_selection(tmp_path) -> None:
    assert classify_gpu_tier(3072) is GpuTier.CPU_FALLBACK
    assert classify_gpu_tier(6144) is GpuTier.WEAK
    assert classify_gpu_tier(12288) is GpuTier.BALANCED
    assert classify_gpu_tier(24576) is GpuTier.STRONG
    hardware = HardwareInfo("gpu", 6144, "driver", GpuTier.WEAK, "test")
    assert profile_path_for_hardware(hardware, tmp_path).name == "weak_gpu.yaml"
