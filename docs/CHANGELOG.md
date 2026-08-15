# Changelog

## Natural-language application control

- Added English command interpretation through a bundled,
  loopback-only llama.cpp sidecar and Qwen3-4B GGUF model.
- Added schema-constrained plans, live candidate resolution, deterministic
  feasibility validation, physical-action confirmation, and JSONL audits.
- Added bounded public-result workflow conditions, multi-pass irregular-surface
  decontamination, multi-target station/surface expansion, and stateful shield
  placement-to-reposition transitions without duplicate inventory consumption.
- Added CPU/Vulkan/CUDA runtime packaging, verified model installation, a single app
  launcher, and a Linux desktop entry while keeping Isaac Sim user-installed.

## Unreleased

- Created the uv-managed independent RadCounterSim repository.
- Added Milestone 0 configuration, logging, manifest, CI, and extension
  boundaries.
- Added initial Milestone 1 data models, material interpolation, analytic
  free-space/slab transport, point-source forward model, detector dead time,
  and reproducible Poisson sampling.
- Added activity-conserving surface/volume quadrature, sampled-source forward
  prediction, transfer-matrix caching that excludes activity revision, chunked
  dose maps, explicit scatter plugins, and both rotating-shield sensor modes.
- Added deterministic decontamination, shield placement, object move/removal,
  disposal validation, resource accounting, robot abstraction, actual pose
  uncertainty, waste transfer, and public/truth action-result separation.
- Added grid/surface candidate bases, stacked Poisson inverse problems,
  nonnegative MLE/L1 estimation, smooth graph-TV estimation, connected source
  hypotheses, active-set Fisher covariance, bootstrap, and a static truth-leak
  test for the estimator package.
- Added post-action raw/normalized residuals, decontamination-retention,
  shield-pose, hidden-source, global gain/background, and localization-error
  hypotheses, Poisson/BIC model selection, nominal action preview, and
  truth-independent belief updates.
- Added deterministic feasibility, the complete weighted planning objective,
  typed mission budgets, action candidate groups, OpenLoop/Greedy/Nearest/
  Random/Oracle/ClosedLoopResidual planners, and a pauseable closed-loop
  coordinator with all required termination conditions and immutable snapshots.
- Added ROS 2 Jazzy message/action/service packages and optional adapter
  boundary, uv-managed batch execution, reproducibility manifests, JSONL,
  Parquet/JSON/NPZ outputs, self-contained HTML reports, analytic validation,
  and one-command demo scripts.
- Added manufacturer-asset Ridgeback + Franka and Nova Carter execution with
  Lula IK, articulated base/arm/finger control, PhysX fixed-joint grasping,
  contact-driven decontamination, shield placement, object relocation, and
  disposal validation.
- Passed the local Isaac Sim, Embree, PhysX, and ROS 2 runtime gates. The
  canonical GPU/renderer release manifest remains open.
- Cleared all 163 repository-wide Ruff violations and documented the remaining
  specification gaps in `SIMULATOR_REQUIREMENT_AUDIT.md`.
