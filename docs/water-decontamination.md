# Water decontamination

Water washing is modeled independently from dry contact tools. The product
runtime uses the articulated tool's finite nozzle ray, standoff, spray cone,
incidence angle, flow, pressure, and surface speed. The ray intersection,
spray footprint, activity update, and visualization all use the same irregular,
activity-bearing visible triangle mesh. If the nozzle ray passes through a
clean hole or outside that mesh, treatment is rejected; no regular plane proxy
is substituted.

For each contacted surface cell, the removal fraction is:

    eta = 1 - exp(-k_w * q_i / area_i * pressure_factor
                  * incidence_factor * speed_factor * washability_i)

`k_w` and the washability distribution are loaded from the SHA-256-bound
material treatment model named by the surface. Flow, pressure, cone geometry,
standoff limits, collection efficiency, and tank capacities remain tool or
resource properties. The repository's concrete model is a synthetic validation
fixture and cannot support a quantitative material-removal claim.

Removed activity is divided into recovered wastewater, downstream
redeposition, and in-scene runoff. The following activity balance is
maintained:

    initial activity
      = remaining surface activity
      + captured wastewater activity
      + runoff activity

Recovered wastewater is authored as a point source inside the visible shielded
storage. Activity leaving the treated triangles is authored as a separate
in-scene runoff source. It therefore remains part of later detector calculations
instead of disappearing from the radiation state.

The resource state separately tracks clean-water inventory, wastewater tank
capacity, retained surface water, and discharged water. If collection is
required, spraying is limited or stopped before the wastewater tank overflows.
The workflow dispatch key contains both the visible surface path and the exact
method name. It rejects an unregistered or mismatched implementation instead of
silently using the dry-contact model. Before articulated motion begins, the
declared action limits reserve clean water and wastewater capacity; after a
successful run the ledger consumes the actual applied and recovered volumes.
Setting `SceneCandidateConfig.enable_water_jet_candidates` exposes a separate
water candidate with explicit volume requirements, while the default paper
scenario continues to select dry contact deliberately.

Runoff on the portable grid model is moved one cell in the configured surface
direction. In the product triangle model it is moved only to nearby visible
activity faces; otherwise it enters the explicit runoff source. This is a
calibratable reduced-order process model intended for comparing tool paths,
resource use, residual activity, and detector consequences.

The surface visualization combines the red/yellow/green activity map with a
blue wetness component. Use the Isaac GUI validation scene with:

~~~bash
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  /home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
  scripts/isaac_surface_decon_validation.py --method water
~~~

Tool and resource parameters are in `configs/decontamination/water_jet.yaml`.
Material response parameters are in the referenced treatment-model file.
