# Water decontamination

Water washing is modeled independently from dry contact tools. The model uses
the finite nozzle ray, standoff, spray cone, incidence angle, flow, pressure,
surface speed, and a material-specific washability field.

For each contacted surface cell, the removal fraction is:

    eta = 1 - exp(-k_w * q_i / area_i * pressure_factor
                  * incidence_factor * speed_factor * washability_i)

Removed activity is divided into recovered wastewater, downstream
redeposition, and environmental discharge. The following activity balance is
maintained:

    initial activity
      = remaining surface activity
      + captured wastewater activity
      + discharged activity

The resource state separately tracks clean-water inventory, wastewater tank
capacity, retained surface water, and discharged water. If collection is
required, spraying is limited or stopped before the wastewater tank overflows.

Runoff is moved one cell in the configured surface direction. Activity that
leaves the source grid is counted as environmental discharge. This is a
reduced-order process model, not CFD. It is intended for planning, residual
diagnosis, resource comparison, and closed-loop robot experiments.

The surface visualization combines the red/yellow/green activity map with a
blue wetness component. Use the Isaac GUI validation scene with:

~~~bash
ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  /home/moeu/.local/isaacsim/6.0.1-uv/.venv/bin/python \
  scripts/isaac_surface_decon_validation.py --method water
~~~

The reference parameters are in
configs/decontamination/water_jet.yaml. Material-specific washability maps can
be supplied as one value per source cell.
