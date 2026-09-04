# RadInterAct repository instructions

## Surface sources and decontamination

- Before creating or modifying a surface/planar source, its visualization, or a
  decontamination workflow, read and follow
  `docs/decontamination-authoring-rules.md`.
- The August 6, 2026 irregular-source and wall-decontamination videos named in
  that document are the canonical behavior and appearance references. Do not
  substitute a tidy framed panel, a uniformly filled rectangle, or a small set
  of hand-arranged polygons.
- A change to surface-source generation or decontamination behavior is incomplete
  unless the implementation, regression tests, and
  `docs/decontamination-authoring-rules.md` are updated together. If behavior is
  intentionally unchanged, the rule document need not be edited, but the change
  must retain its invariants and tests.
- Validation must operate on the same irregular activity-bearing geometry shown
  to the user. Never decontaminate a hidden regular proxy while displaying an
  unrelated irregular overlay.
