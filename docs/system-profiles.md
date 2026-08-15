# Switchable environments, robots, and detectors

RadCounterSim keeps environments, robot sets, and detector sets independent in
`configs/system/catalog.yaml`. A profile is only a named combination of those
three components plus a radiation runtime configuration. The active selection
is stored outside the repository by default at
`~/.config/radcountersim/system-selection.json`.

The normal operator path is the **SYSTEM CONFIGURATION / 構成** section in the
Isaac Operations window. Choose a preset, or change Environment, Robot, and
Detector independently, then press **選択した構成を適用 / Apply**. Display names
are shown instead of internal IDs. The GUI validates dependencies, pauses the
timeline, rebuilds the stage, initializes radiation transport, and saves the
successful selection for the next launch. **現在値に戻す** discards unapplied
choices. If a catalog environment has not been downloaded or converted yet,
the same button changes to **環境を準備して適用 / Prepare** and runs its reviewed,
repository-local preparation steps automatically.

The **NATURAL LANGUAGE COMMAND / ROBOT LLM** section is intentionally separate. It accepts
natural-language robot workflow instructions only; environment, robot-set, and
detector-set switching always uses the explicit controls above.

## Common commands

```bash
# Inspect every available component and profile.
uv run radcounter-system list

# Resolve a profile without changing the active selection.
uv run radcounter-system show --profile fukushima-packbot

# Save a selection for later radcounter-app launches.
uv run radcounter-system activate --profile vertical-slice-packbot

# Return to the complete articulated decommissioning workflow.
uv run radcounter-system activate --profile vertical-slice

# Override components independently for one launch.
radcounter-app \
  --profile vertical-slice-packbot \
  --environment vertical-slice \
  --robot-set fukushima-response-reference \
  --detector-set fukushima-survey
```

Environment, robot-set, and detector-set overrides are validated before Isaac
starts. A detector set that requires a robot not present in the selected robot
set fails with a named configuration error rather than mounting to a guessed
prim. The existing `vertical-slice` profile retains the full articulated
decontamination workflow. Other combinations use the generic compositor and do
not silently author the vertical-slice-only room, source, or task geometry.

## Fukushima Daiichi SolidWorks environment

The upstream repository contains SolidWorks `.SLDASM` and `.SLDPRT` files only.
RadCounterSim converts the native top-level assembly directly to USD on Linux
with the HOOPS Exchange converter bundled with Isaac Sim 6.0.1. This preserves
the assembly hierarchy and avoids a Windows or STEP prerequisite.

1. Fetch the pinned upstream source and record its hashes.

   ```bash
   uv run python scripts/fetch_fukushima_daiichi_cad.py
   ```

   This writes the checkout below
   `.cache/external/fukushima_daiichi_solidworks/source` and a
   `source-provenance.json` record. The source is pinned to commit
   `f6541deb6159c5d908a4f028d021e3d2c9f7f8e8`.

2. Convert the top-level assembly with the local headless Isaac converter.

   ```bash
   uv run python scripts/convert_solidworks_to_usd.py
   ```

   The converter runs through `omni.app.empty.kit`, so it does not initialize an
   RTX viewport. It emits a monolithic `Building.usdc`, refuses to overwrite an
   existing result unless `--force` is supplied, and records input/output
   hashes, source revision, converter versions, mesh count, and triangle count
   in `Building.usdc.provenance.json`.

3. Validate and activate the selection. These commands are useful for
   automation; GUI users only need to select and apply the profile.

   ```bash
   uv run radcounter-system check --profile fukushima-packbot
   uv run radcounter-system prepare --profile fukushima-packbot
   uv run radcounter-system activate --profile fukushima-packbot
   radcounter-app
   ```

The normalized manifest and composed USD stage are content-addressed below the
configured environment cache (by default
`~/.cache/radcountersim/environments`). The same imported triangles are used for visible USD
geometry, static PhysX collision, and radiation material geometry. The
Fukushima profile adds no contamination surface or hidden regular proxy. Any
later contamination authoring must still follow
`docs/decontamination-authoring-rules.md` and operate on the visible irregular
activity-bearing geometry.

The upstream CAD model is described as generic and incomplete: its own TODO
list includes pipes, biological shielding, supports, scaffolding, catwalks,
shield plugs, and floor-height adjustments. It must not be presented as an
as-built engineering record or used for safety decisions.

## Adding catalog entries

- An environment entry references the existing environment descriptor contract
  from `docs/environment-import.md`.
- A `fleet` robot set references the generic USD/URDF/Xacro/MJCF fleet contract.
  A `reference` set uses the traceable lightweight nuclear-response models.
- A detector set references built-in model IDs from
  `radcounter.core.sensors.catalog` or a custom YAML descriptor.
- A profile selects one entry from each group and a base runtime JSON file.

Run `uv run radcounter-system show --profile NAME` after edits. It loads every
referenced descriptor, rebases fleet-relative asset paths, validates detector
models and parent relationships, and reports whether the environment source is
ready without starting Isaac Sim.

In a configurable composition, the selected detector set replaces detector
roles already authored by the environment. Existing detector geometry remains
visible but is marked inactive in the generated cache stage, so measurements
cannot accidentally combine the selected instruments with hidden defaults.
