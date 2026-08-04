# Experiment artifacts

`ExperimentArtifactBundle` creates one immutable JSON record per case, a CSV summary, and a manifest containing stage/configuration SHA-256 values and runtime versions. Writes use temporary files, `fsync`, and atomic replacement.

Run the shield-pose residual sweep inside Isaac Sim:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_vertical_slice_experiment.py --seeds 20
```

This experiment deliberately separates the nominal shield pose used for prediction from the perturbed Truth pose used for observation. It does not modify or evaluate source-estimation algorithms.
