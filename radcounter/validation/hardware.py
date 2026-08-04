"""Hardware identity and endurance evidence gates."""

from __future__ import annotations

import platform
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GpuEvidence:
    index: int
    name: str
    memory_total_mib: int
    driver_version: str


@dataclass(frozen=True)
class HardwareGateResult:
    passed: bool
    qualified_evidence: bool
    evidence_kind: str
    required_vram_class: str
    required_gpu_count: int
    actual_os: str
    actual_gpus: tuple[GpuEvidence, ...]
    endurance_duration_s: float | None
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["actual_gpus"] = [asdict(gpu) for gpu in self.actual_gpus]
        return payload


def parse_nvidia_smi_csv(value: str) -> tuple[GpuEvidence, ...]:
    gpus = []
    for line in value.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi row: {line}")
        memory_text = fields[2].replace("MiB", "").strip()
        gpus.append(
            GpuEvidence(
                index=int(fields[0]),
                name=fields[1],
                memory_total_mib=int(memory_text),
                driver_version=fields[3],
            )
        )
    return tuple(gpus)


def detect_nvidia_gpus() -> tuple[GpuEvidence, ...]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_nvidia_smi_csv(completed.stdout)


def _vram_range(vram_class: str) -> tuple[int, int | None]:
    ranges = {
        "4gb": (3500, 5500),
        "8gb": (7000, 9500),
        "12gb": (10500, 13500),
        "16gb": (14000, 19500),
        "24gb": (21500, 27500),
        "32gb+": (30000, None),
    }
    try:
        return ranges[vram_class.lower()]
    except KeyError as error:
        raise ValueError(f"unknown VRAM class: {vram_class}") from error


def _find_number(value: object, names: Sequence[str]) -> float | None:
    if isinstance(value, Mapping):
        for name in names:
            item = value.get(name)
            if isinstance(item, int | float):
                return float(item)
        for item in value.values():
            found = _find_number(item, names)
            if found is not None:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_number(item, names)
            if found is not None:
                return found
    return None


def evaluate_hardware_gate(
    *,
    required_vram_class: str,
    endurance_metrics: Mapping[str, object] | None,
    expected_os: str | None = None,
    driver_pattern: str | None = None,
    required_duration_s: float = 600.0,
    minimum_gpu_count: int = 1,
    gpus: Sequence[GpuEvidence] | None = None,
) -> HardwareGateResult:
    if minimum_gpu_count < 1:
        raise ValueError("minimum_gpu_count must be positive")
    actual_gpus = tuple(gpus) if gpus is not None else detect_nvidia_gpus()
    actual_os = platform.system().lower()
    lower, upper = _vram_range(required_vram_class)
    matching = [
        gpu
        for gpu in actual_gpus
        if gpu.memory_total_mib >= lower
        and (upper is None or gpu.memory_total_mib <= upper)
    ]
    reasons = []
    if len(matching) < minimum_gpu_count:
        memories = ", ".join(str(gpu.memory_total_mib) for gpu in actual_gpus) or "none"
        reasons.append(
            f"requires {minimum_gpu_count} physical {required_vram_class} GPU(s); "
            f"matched {len(matching)}, detected memory MiB: {memories}"
        )
    if expected_os and actual_os != expected_os.lower():
        reasons.append(f"OS {actual_os} does not match required {expected_os.lower()}")
    if (
        driver_pattern
        and matching
        and not any(
            re.fullmatch(driver_pattern, gpu.driver_version) for gpu in matching
        )
    ):
        reasons.append(f"no matching GPU driver satisfies pattern {driver_pattern}")
    duration = (
        _find_number(
            endurance_metrics,
            ("actual_duration_s", "duration_s", "elapsed_s", "wall_time_s"),
        )
        if endurance_metrics is not None
        else None
    )
    if duration is None:
        reasons.append("endurance metrics contain no measured duration")
    elif duration < required_duration_s:
        reasons.append(
            f"endurance duration {duration:.3f}s is below {required_duration_s:.3f}s"
        )
    passed_marker = (
        endurance_metrics.get("passed") if endurance_metrics is not None else None
    )
    if passed_marker is False:
        reasons.append("endurance workload reported failure")
    qualified = len(matching) >= minimum_gpu_count and not any(
        reason.startswith(("OS ", "no matching GPU driver"))
        for reason in reasons
    )
    return HardwareGateResult(
        passed=not reasons,
        qualified_evidence=qualified and duration is not None,
        evidence_kind="physical_gpu" if qualified else "workload_profile_only",
        required_vram_class=required_vram_class,
        required_gpu_count=minimum_gpu_count,
        actual_os=actual_os,
        actual_gpus=actual_gpus,
        endurance_duration_s=duration,
        reasons=tuple(reasons),
    )
