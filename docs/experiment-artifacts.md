# Experiment artifacts

`ExperimentArtifactBundle` creates one immutable JSON record per case, a CSV summary, and a manifest containing stage/configuration SHA-256 values and runtime versions. Writes use temporary files, `fsync`, and atomic replacement.

Run the shield-pose residual sweep inside Isaac Sim:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
source scripts/host_env.sh
uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python scripts/run_vertical_slice_experiment.py --seeds 20
```

This experiment deliberately separates the nominal shield pose used for prediction from the perturbed Truth pose used for observation. It does not modify or evaluate source-estimation algorithms.

## Physical paper evaluation

Create each full articulated run with an explicit seed, then collect only
`physical_robot_execution` artifacts:

```bash
export OMNI_KIT_ACCEPT_EULA=YES RADCOUNTER_HOST_ENV_NO_ROS=1
source scripts/host_env.sh
export PYTHONPATH="$PWD/build/native/python:$PWD/source/extensions/radcounter.isaac:$PWD${PYTHONPATH:+:$PYTHONPATH}"

uv run --project "$RADCOUNTER_ISAAC_ROOT" --locked python \
  scripts/run_gui_validation.py --headless --no-keep-open --phase-hold-s 0 --seed 11 \
  --artifact artifacts/gui-validation/paper-seed-11-runtime.json

uv run python -c 'import json; from pathlib import Path; p=Path("artifacts/gui-validation/paper-seed-11-runtime.json"); d=json.loads(p.read_text()); assert d.get("success") is True, d.get("error"); assert d.get("execution_runtime")'

uv run python scripts/run_paper_evaluation.py \
  --run-id synthetic-seed-11-runtime \
  --artifact artifacts/gui-validation/paper-seed-11-runtime.json
```

The collector rejects failed or incomplete action sequences, duplicate seeds,
different stage/configuration hashes, missing activity/resource/estimator
balances, non-native transport, all-zero physical action time, and artifacts
from another evidence class. Direct scene-edit artifacts remain valid under
`kinematic_scene_edit`, but this collector never pools or relabels them as
physical robot evidence. It also requires the execution-side Python,
RadInterAct, NumPy, Isaac Sim, Embree, renderer, GPU, memory, and driver
identity, preserves that record in the final manifest, and rejects a pooled set
whose runtime identities differ.
`synthetic_validation` output is suitable for implementation verification only.
`research_evaluation` additionally requires authoritative material and isotope
tables, calibrated detector responses, a reference-calibrated buildup table,
and exact equality between those validated values and the runtime configuration.
No automatic fallback from research inputs to synthetic fixtures is permitted.
One seed is a smoke test; paper statistics require the declared fixed seed set.

The current synthetic smoke output is
`artifacts/paper-evaluation/synthetic-seed-11-runtime/`. It passed the artifact
contract but must not be cited as calibrated detector, material, or treatment
accuracy.
