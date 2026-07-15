# Radiation model and units

All public names carry units. Position and path length are metres, activity is
becquerels, energy is keV, count rate is counts/s, measurement duration is
seconds, and dose rate is Sv/h.

For source sample `s`, detector pose `d`, and emission line `e`, the direct
expected count-rate contribution is

```text
lambda_sde = A_s Y_e / (4 pi r_sd^2) * T_sd(E_e) * epsilon_d(E_e)
T(E) = exp(-sum_m mu_m(E) length_m)
```

The inverse-square distance is clamped at a configured positive `r_min_m`.
The analytic backend supports free space and infinite planar slabs. It is a
validation fallback, not a replacement for scene ray tracing.

Detector output applies background, a non-paralyzable dead-time model, and
Poisson sampling. A named scatter model must always be recorded; the initial
implementation uses `NoScatterModel`.
