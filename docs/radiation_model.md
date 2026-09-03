# Radiation model and units

All public names carry units. Position and path length are metres, activity is
becquerels, energy is keV, count rate is counts/s, measurement duration is
seconds, and dose rate is Sv/h.

For source sample `s`, detector pose `d`, emission line `e`, and detector output
bin `b`, the direct expected count-rate contribution is

```text
Phi_sde = A_s Y_e / (4 pi r_sd^2) * T_sd(E_e)
lambda_sdeb = Phi_sde * R_db(E_e)
T(E) = exp(-sum_m mu_m(E) length_m)
```

`Phi` is incident fluence rate in particles/(m2 s). `R_db` is the calibrated or
explicitly classified detector effective-area response in m2, including energy
redistribution into output bin `b`; therefore `lambda` is counts/s. Integral
counters and spectrometers use this same response-matrix path. A dimensionless
efficiency without its associated active area is not accepted.

The inverse-square distance is clamped at a configured positive `r_min_m`.
The analytic backend supports free space and infinite planar slabs. It is a
separately selected validation backend, not an automatic replacement for scene
ray tracing.

Photon output records both primary and corrected source components. The default
model is explicitly `primary_only`. A corrected run must load a SHA-bound,
reference-calibrated buildup surface for every traversed material and requested
energy/optical-depth range. The correction is applied to incident photon fluence
before the detector effective-area response. Missing material coverage and any
attempted extrapolation are fatal.

The broad detector catalog receives `IncidentParticleFluence` values produced by
one shared path-length provider. Detector models no longer calculate inverse
square loss or inspect separate shield panels. Gamma and X-ray contributions use
the photon attenuation/buildup path above; neutrons use versioned material
removal coefficients; alpha and beta use versioned open-medium and solid range
kernels. These latter kernels are reduced-order models and their data status is
recorded. `particle_transport.synthetic.yaml` is only a software-validation
fixture.

Detector electronics apply background, the configured paralyzable or
non-paralyzable dead-time model, saturation, angular response, energy
redistribution, and Poisson sampling after transport.

Paper experiments must load physics inputs through
`validate_research_evaluation_data`. This entry point verifies content hashes,
requires authoritative material and isotope data plus the requested detector
evidence class, rejects duplicate identifiers, and refuses any combination that
would extrapolate attenuation or detector response beyond its stated energy
range. Synthetic validation data cannot satisfy the default research gate.
