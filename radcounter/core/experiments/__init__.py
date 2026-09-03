"""Headless experiment automation and artifact output."""

from radcounter.core.experiments.artifacts import CaseResult, RunArtifactWriter, RunDirectory
from radcounter.core.experiments.atomic_bundle import (
    EvidenceClass,
    ExperimentArtifactBundle,
    ExperimentCaseRecord,
    sha256_file,
)
from radcounter.core.experiments.cases import AnalyticRadiationValidationCase
from radcounter.core.experiments.paper import (
    EvaluationDataMode,
    PaperEvaluationError,
    PaperPhysicsEvidence,
    collect_physical_paper_evaluation,
    summarize_physical_robot_artifact,
    validate_paper_physics_evidence,
    write_aggregate_metrics,
)
from radcounter.core.experiments.runner import BatchRun, BatchRunner, ExperimentCase

__all__ = [
    "AnalyticRadiationValidationCase",
    "BatchRun",
    "BatchRunner",
    "CaseResult",
    "ExperimentCase",
    "EvidenceClass",
    "EvaluationDataMode",
    "ExperimentArtifactBundle",
    "ExperimentCaseRecord",
    "PaperEvaluationError",
    "PaperPhysicsEvidence",
    "RunArtifactWriter",
    "RunDirectory",
    "collect_physical_paper_evaluation",
    "sha256_file",
    "summarize_physical_robot_artifact",
    "validate_paper_physics_evidence",
    "write_aggregate_metrics",
]
