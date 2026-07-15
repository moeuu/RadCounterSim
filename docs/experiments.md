# Reproducible experiments

All Python dependencies and commands are managed by `uv`.

## Built-in cases

| Case | Acceptance target |
| --- | --- |
| `analytic_radiation_validation` | inverse-square and slab-attenuation identities |
| `decontamination_primitive` | post-action efficiency recovery from Poisson counts |
| `shielding_primitive` | effective shield transmission recovery |
| `movable_contaminated_object` | localization after object motion |
| `hidden_source_residual` | residual diagnosis and omitted-source localization |
| `closed_loop_vs_open_loop` | verify-and-replan benefit over a fixed sequence |
| `resource_constrained_multi_action` | feasible weighted action-subset selection |

Run one case:

```bash
uv run radcounter-experiments \
  --case hidden_source_residual \
  --seed 42 \
  --run-root runs
```

Run the end-to-end comparison demo:

```bash
./scripts/run_demo.sh
```

Check versioned regression bounds:

```bash
uv run python scripts/check_regression.py --seed 42
```

Run portable benchmarks:

```bash
uv run python scripts/benchmark_core.py --output runs/benchmark.json
```

Each experiment creates a timestamped directory containing the resolved config,
manifest, JSONL events, four Parquet tables, NPZ maps, metrics, and a standalone
HTML report.
