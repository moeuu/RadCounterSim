"""Startup hardware detection and deterministic performance-tier selection."""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class GpuTier(StrEnum):
    CPU_FALLBACK = "cpu_fallback"
    WEAK = "weak_gpu"
    BALANCED = "balanced_gpu"
    STRONG = "strong_gpu"


@dataclass(frozen=True)
class HardwareInfo:
    gpu_name: str
    vram_mb: int
    driver_version: str
    tier: GpuTier
    detection_source: str


def classify_gpu_tier(vram_mb: int, gpu_name: str = "") -> GpuTier:
    """Classify by usable VRAM; model names are retained only for reporting."""

    del gpu_name
    if vram_mb < 4096:
        return GpuTier.CPU_FALLBACK
    if vram_mb < 8192:
        return GpuTier.WEAK
    if vram_mb < 16384:
        return GpuTier.BALANCED
    return GpuTier.STRONG


def detect_hardware() -> HardwareInfo:
    override = os.environ.get("RADCOUNTER_GPU_TIER")
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        first = completed.stdout.strip().splitlines()[0]
        name, memory, driver = (part.strip() for part in first.split(",", 2))
        vram_mb = int(float(memory))
        tier = GpuTier(override) if override else classify_gpu_tier(vram_mb, name)
        return HardwareInfo(name, vram_mb, driver, tier, "nvidia-smi")
    except (FileNotFoundError, subprocess.SubprocessError, ValueError, IndexError):
        tier = GpuTier(override) if override else GpuTier.CPU_FALLBACK
        return HardwareInfo("unavailable", 0, "unavailable", tier, "fallback")


def profile_path_for_hardware(
    hardware: HardwareInfo,
    repository_root: str | Path,
) -> Path:
    return (
        Path(repository_root).expanduser().resolve()
        / "configs"
        / "performance"
        / f"{hardware.tier.value}.yaml"
    )
