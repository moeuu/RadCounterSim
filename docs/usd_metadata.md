# USD radiation metadata

The first implementation uses namespaced custom attributes rather than a
custom USD schema. Required namespaces are `rad:source:*`, `rad:material:*`,
`rad:shield:*`, `rad:decon:*`, and `rad:manipulation:*`.

Per-triangle activity arrays are NPZ sidecars. USD stores only the sidecar URI
and SHA-256. Estimator-hidden truth is represented by
`rad:source:hiddenFromEstimator`; adapter code must omit such sources from any
belief/public descriptor.

Every enabled treatment surface also requires a substrate material ID and a
SHA-256-bound treatment-model asset. The model, rather than duplicate USD or
controller constants, supplies the material-dependent dry-contact and water-jet
response coefficients. Missing assets, digest mismatches, and substrate
mismatches are fatal configuration errors.

Every non-shield movable prop must author
`rad:manipulation:placementReference = "root"`. Planned and executed target
poses then refer to the rigid-body root, while collision and support checks use
the same prop's visible bounds translated from that root. Missing or unsupported
placement references are rejected; visual handles must never silently redefine
the commanded object pose.

Each such prop must also author a finite horizontal
`rad:manipulation:parkingOffsetM` (`double3`, with zero vertical component),
measured from the configured disposal-zone center. This makes temporary
relocation positions part of the scene contract instead of assigning them by
USD traversal order. Authors must choose offsets that keep the later disposal
approach clear and accommodate the visible collision envelope of that prop.

`rad:manipulation:baseStandOffM` is also required for each movable prop and
must be in `(0, 0.95]` m. It declares the verified horizontal separation
between the articulated base frame and the grasp frame. This value is authored
per prop because body clearance and loaded manipulator reach differ; a global
fallback is not used.

The definitive attribute list is preserved in
`docs/specs/RadInterAct_Codex_Implementation_Spec.md`.
