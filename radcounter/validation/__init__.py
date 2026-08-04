"""Evidence-producing validation gates for external simulators and hardware."""

from .aerial import (
    MultirotorDynamics,
    MultirotorSpec,
    MultirotorState,
    RotorDynamicsGateResult,
    run_rotor_dynamics_gate,
)
from .detector_hil import (
    DetectorFrameCodec,
    DetectorHILSession,
    HILFrame,
    HILMetrics,
    SerialLineTransport,
    SocketLineTransport,
)
from .hardware import (
    GpuEvidence,
    HardwareGateResult,
    detect_nvidia_gpus,
    evaluate_hardware_gate,
)
from .physics import (
    PhysicsValidationResult,
    load_validation_dataset,
    validate_physics_dataset,
)

__all__ = [
    "DetectorFrameCodec",
    "DetectorHILSession",
    "GpuEvidence",
    "HILFrame",
    "HILMetrics",
    "HardwareGateResult",
    "MultirotorDynamics",
    "MultirotorSpec",
    "MultirotorState",
    "PhysicsValidationResult",
    "RotorDynamicsGateResult",
    "SerialLineTransport",
    "SocketLineTransport",
    "detect_nvidia_gpus",
    "evaluate_hardware_gate",
    "load_validation_dataset",
    "run_rotor_dynamics_gate",
    "validate_physics_dataset",
]
