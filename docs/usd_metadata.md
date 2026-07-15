# USD radiation metadata

The first implementation uses namespaced custom attributes rather than a
custom USD schema. Required namespaces are `rad:source:*`, `rad:material:*`,
`rad:shield:*`, `rad:decon:*`, and `rad:manipulation:*`.

Per-triangle activity arrays are NPZ sidecars. USD stores only the sidecar URI
and SHA-256. Estimator-hidden truth is represented by
`rad:source:hiddenFromEstimator`; adapter code must omit such sources from any
belief/public descriptor.

The definitive attribute list is preserved in
`docs/specs/RadCounterSim_Codex_Implementation_Spec.md`.
