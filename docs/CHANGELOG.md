# Changelog

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
- External Isaac Sim, Embree, and ROS 2 acceptance gates remain open.
