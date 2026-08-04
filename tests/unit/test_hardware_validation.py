from radcounter.validation.hardware import (
    GpuEvidence,
    evaluate_hardware_gate,
    parse_nvidia_smi_csv,
)


def test_nvidia_inventory_parser_preserves_driver_and_vram() -> None:
    gpus = parse_nvidia_smi_csv("0, NVIDIA RTX A2000, 4096 MiB, 550.54.15\n")
    assert gpus == (
        GpuEvidence(
            index=0,
            name="NVIDIA RTX A2000",
            memory_total_mib=4096,
            driver_version="550.54.15",
        ),
    )


def test_physical_4gb_gpu_and_600_second_metrics_qualify() -> None:
    result = evaluate_hardware_gate(
        required_vram_class="4gb",
        expected_os=None,
        driver_pattern=r"550\..+",
        endurance_metrics={"passed": True, "actual_duration_s": 600.5},
        gpus=(GpuEvidence(0, "RTX A2000", 4096, "550.54.15"),),
    )
    assert result.passed
    assert result.qualified_evidence
    assert result.evidence_kind == "physical_gpu"


def test_large_gpu_profile_is_not_accepted_as_8gb_physical_evidence() -> None:
    result = evaluate_hardware_gate(
        required_vram_class="8gb",
        endurance_metrics={"passed": True, "actual_duration_s": 601.0},
        gpus=(GpuEvidence(0, "RTX 5090", 32607, "580.159.03"),),
    )
    assert not result.passed
    assert not result.qualified_evidence
    assert result.evidence_kind == "workload_profile_only"


def test_dual_gpu_gate_rejects_a_single_matching_card() -> None:
    result = evaluate_hardware_gate(
        required_vram_class="8gb",
        minimum_gpu_count=2,
        endurance_metrics={"passed": True, "actual_duration_s": 601.0},
        gpus=(GpuEvidence(0, "RTX 4000", 8192, "580.10"),),
    )
    assert not result.passed
    assert result.required_gpu_count == 2
