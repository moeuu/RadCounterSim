"""Built-in detector families commonly used in radiation work."""

from __future__ import annotations

import numpy as np

from .universal import (
    DeadTimeModel,
    DetectorDescriptor,
    DetectorFamily,
    DetectorOutput,
    Directionality,
    ParticleResponse,
    RadiationType,
    ResponseCurve,
)

GAMMA_ENERGIES = (30.0, 60.0, 122.0, 356.0, 662.0, 1173.0, 1332.0, 2615.0)
SPECTRUM_BINS = tuple(float(value) for value in np.linspace(0.0, 3000.0, 121))
H100_GAMMA_ENERGIES = (50.0, 122.0, 356.0, 661.657, 1173.0, 1332.0, 2615.0, 3000.0)
H100_SYNTHETIC_EFFECTIVE_AREA_M2 = (
    4.0e-5,
    6.8e-5,
    6.4e-5,
    5.6e-5,
    4.0e-5,
    3.7e-5,
    2.2e-5,
    1.9e-5,
)


def _curve(
    energies_kev: tuple[float, ...],
    relative_efficiency: tuple[float, ...],
    active_area_m2: float,
) -> ResponseCurve:
    return ResponseCurve(
        energies_kev,
        tuple(active_area_m2 * value for value in relative_efficiency),
    )


def _gamma(values: tuple[float, ...], active_area_m2: float) -> tuple[ParticleResponse, ...]:
    return (
        ParticleResponse(
            RadiationType.GAMMA,
            _curve(GAMMA_ENERGIES, values, active_area_m2),
        ),
    )


def popular_detector_catalog() -> dict[str, DetectorDescriptor]:
    gamma_survey = (0.80, 0.72, 0.55, 0.28, 0.16, 0.09, 0.08, 0.04)
    gamma_spectroscopy = (0.65, 0.58, 0.48, 0.30, 0.22, 0.14, 0.12, 0.07)
    neutron_energies = (0.000025, 0.001, 1.0, 100.0, 1000.0, 10000.0)
    neutron_efficiency = (0.75, 0.70, 0.42, 0.20, 0.12, 0.06)
    alpha_energies = (1000.0, 3000.0, 5500.0, 8000.0)
    alpha_efficiency = (0.15, 0.35, 0.42, 0.38)
    beta_energies = (50.0, 200.0, 500.0, 1000.0, 2500.0)
    beta_efficiency = (0.05, 0.22, 0.38, 0.44, 0.46)

    return {
        "gm_tube": DetectorDescriptor(
            "gm_tube",
            "Geiger-Mueller survey meter",
            DetectorFamily.GEIGER_MULLER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            _gamma(gamma_survey, 8.0e-4),
            background_cps=0.25,
            dead_time_s=190e-6,
            dead_time_model=DeadTimeModel.NONPARALYZABLE,
            maximum_count_rate_cps=5_000.0,
        ),
        "ion_chamber": DetectorDescriptor(
            "ion_chamber",
            "Pressurized ionization chamber",
            DetectorFamily.ION_CHAMBER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNT_RATE, DetectorOutput.DOSE_RATE),
            _gamma((0.92, 0.95, 0.98, 1.0, 1.0, 1.0, 1.0, 0.98), 2.0e-3),
            background_cps=0.02,
            dead_time_model=DeadTimeModel.NONE,
            maximum_count_rate_cps=5.0e8,
            dose_conversion_usv_h_per_count_kev=2.2e-8,
        ),
        "nai_tl": DetectorDescriptor(
            "nai_tl",
            "NaI(Tl) scintillation spectrometer",
            DetectorFamily.SCINTILLATOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE, DetectorOutput.SPECTRUM),
            _gamma(gamma_spectroscopy, 2.0e-3),
            background_cps=1.2,
            dead_time_s=2.0e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.07,
            energy_bin_edges_kev=SPECTRUM_BINS,
        ),
        "csi_tl": DetectorDescriptor(
            "csi_tl",
            "CsI(Tl) scintillation detector",
            DetectorFamily.SCINTILLATOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.SPECTRUM),
            _gamma((0.82, 0.75, 0.62, 0.39, 0.27, 0.17, 0.15, 0.09), 1.5e-3),
            background_cps=1.0,
            dead_time_s=3.0e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.085,
            energy_bin_edges_kev=SPECTRUM_BINS,
        ),
        "plastic_scintillator": DetectorDescriptor(
            "plastic_scintillator",
            "Plastic scintillation portal detector",
            DetectorFamily.SCINTILLATOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            _gamma((0.20, 0.18, 0.16, 0.13, 0.10, 0.08, 0.07, 0.05), 5.0e-2),
            background_cps=25.0,
            dead_time_s=20e-9,
            maximum_count_rate_cps=2.0e7,
        ),
        "hpge": DetectorDescriptor(
            "hpge",
            "HPGe gamma spectrometer",
            DetectorFamily.SEMICONDUCTOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.SPECTRUM),
            _gamma((0.58, 0.52, 0.43, 0.27, 0.20, 0.13, 0.12, 0.07), 2.5e-3),
            background_cps=0.15,
            dead_time_s=5e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.003,
            energy_bin_edges_kev=SPECTRUM_BINS,
        ),
        "czt": DetectorDescriptor(
            "czt",
            "CZT room-temperature spectrometer",
            DetectorFamily.SEMICONDUCTOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.SPECTRUM),
            _gamma((0.74, 0.65, 0.50, 0.25, 0.14, 0.07, 0.06, 0.025), 4.0e-4),
            background_cps=0.2,
            dead_time_s=1e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.025,
            energy_bin_edges_kev=SPECTRUM_BINS,
        ),
        "h3d_h100_omni": DetectorDescriptor(
            "h3d_h100_omni",
            "H3D H100 omnidirectional gamma monitor mode",
            DetectorFamily.GAMMA_IMAGER,
            Directionality.OMNIDIRECTIONAL,
            (
                DetectorOutput.COUNTS,
                DetectorOutput.COUNT_RATE,
                DetectorOutput.SPECTRUM,
                DetectorOutput.DOSE_RATE,
            ),
            (
                ParticleResponse(
                    RadiationType.GAMMA,
                    ResponseCurve(
                        H100_GAMMA_ENERGIES,
                        H100_SYNTHETIC_EFFECTIVE_AREA_M2,
                    ),
                ),
            ),
            background_cps=0.20,
            dead_time_s=1.0e-6,
            dead_time_model=DeadTimeModel.NONPARALYZABLE,
            maximum_count_rate_cps=400_000.0,
            energy_resolution_fwhm_fraction_at_662kev=0.011,
            energy_bin_edges_kev=SPECTRUM_BINS,
            dose_conversion_usv_h_per_count_kev=3.70e-5,
            response_data_status="synthetic_validation_only",
            response_provenance={
                "source": "H3D H100 manufacturer specifications plus synthetic response",
                "calibration_method": (
                    "synthetic CZT effective-area curve; not an H3D calibration dataset"
                ),
                "manufacturer_specification": "https://h3dgamma.com/H100Specs.pdf",
            },
            metadata={
                "manufacturer": "H3D, Inc.",
                "product": "H100 Gamma-Ray Imaging Spectrometer",
                "product_url": "https://h3dgamma.com/h100.php",
                "manufacturer_specification_url": "https://h3dgamma.com/H100Specs.pdf",
                "radiation_fov_sr": 12.566370614359172,
                "energy_range_min_kev": 50.0,
                "energy_range_max_kev": 3000.0,
                "czt_crystal_volume_cm3": 6.0,
                "mass_kg": 3.2,
                "enclosure_rating": "IP67",
                "operating_mode": "omnidirectional scalar monitoring",
            },
        ),
        "cdte": DetectorDescriptor(
            "cdte",
            "CdTe compact spectrometer",
            DetectorFamily.SEMICONDUCTOR,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.SPECTRUM),
            _gamma((0.78, 0.68, 0.49, 0.22, 0.11, 0.05, 0.04, 0.015), 2.5e-4),
            background_cps=0.15,
            dead_time_s=1.5e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.035,
            energy_bin_edges_kev=SPECTRUM_BINS,
        ),
        "proportional_counter": DetectorDescriptor(
            "proportional_counter",
            "Gas proportional counter",
            DetectorFamily.PROPORTIONAL_COUNTER,
            Directionality.COSINE,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            (
                ParticleResponse(
                    RadiationType.ALPHA,
                    _curve(alpha_energies, alpha_efficiency, 1.5e-2),
                ),
                ParticleResponse(
                    RadiationType.BETA,
                    _curve(beta_energies, beta_efficiency, 1.5e-2),
                ),
            ),
            background_cps=0.08,
            dead_time_s=10e-6,
            field_of_view_half_angle_deg=90.0,
            off_axis_leakage_fraction=0.0,
        ),
        "pancake_probe": DetectorDescriptor(
            "pancake_probe",
            "Alpha/beta pancake contamination probe",
            DetectorFamily.SURFACE_CONTAMINATION,
            Directionality.COSINE,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            (
                ParticleResponse(
                    RadiationType.ALPHA,
                    _curve(alpha_energies, alpha_efficiency, 1.55e-3),
                ),
                ParticleResponse(
                    RadiationType.BETA,
                    _curve(beta_energies, beta_efficiency, 1.55e-3),
                ),
                ParticleResponse(
                    RadiationType.GAMMA,
                    _curve(GAMMA_ENERGIES, (0.03,) * len(GAMMA_ENERGIES), 1.55e-3),
                ),
            ),
            background_cps=0.15,
            dead_time_s=100e-6,
            field_of_view_half_angle_deg=90.0,
            off_axis_leakage_fraction=0.0,
        ),
        "he3_neutron": DetectorDescriptor(
            "he3_neutron",
            "Moderated He-3 neutron counter",
            DetectorFamily.NEUTRON_COUNTER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            (
                ParticleResponse(
                    RadiationType.NEUTRON,
                    _curve(neutron_energies, neutron_efficiency, 1.0e-2),
                ),
            ),
            background_cps=0.02,
            dead_time_s=5e-6,
        ),
        "li6_zns_neutron": DetectorDescriptor(
            "li6_zns_neutron",
            "Li-6/ZnS neutron scintillator",
            DetectorFamily.NEUTRON_COUNTER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE),
            (
                ParticleResponse(
                    RadiationType.NEUTRON,
                    _curve(neutron_energies, neutron_efficiency, 8.0e-3),
                ),
            ),
            background_cps=0.03,
            dead_time_s=1e-6,
        ),
        "neutron_rem_meter": DetectorDescriptor(
            "neutron_rem_meter",
            "Neutron rem meter",
            DetectorFamily.NEUTRON_COUNTER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNT_RATE, DetectorOutput.DOSE_RATE),
            (
                ParticleResponse(
                    RadiationType.NEUTRON,
                    _curve(neutron_energies, neutron_efficiency, 2.0e-2),
                ),
            ),
            background_cps=0.01,
            dose_conversion_usv_h_per_count_kev=4.0e-8,
        ),
        "electronic_dosimeter": DetectorDescriptor(
            "electronic_dosimeter",
            "Electronic personal dosimeter",
            DetectorFamily.PERSONAL_DOSIMETER,
            Directionality.OMNIDIRECTIONAL,
            (DetectorOutput.COUNT_RATE, DetectorOutput.DOSE_RATE),
            _gamma((0.62, 0.72, 0.86, 0.96, 1.0, 1.0, 1.0, 0.95), 2.0e-4),
            background_cps=0.02,
            maximum_count_rate_cps=1.0e7,
            dose_conversion_usv_h_per_count_kev=2.2e-8,
        ),
        "collimated_nai": DetectorDescriptor(
            "collimated_nai",
            "Lead-collimated NaI directional detector",
            DetectorFamily.SCINTILLATOR,
            Directionality.COLLIMATED,
            (DetectorOutput.COUNTS, DetectorOutput.COUNT_RATE, DetectorOutput.DIRECTION),
            _gamma(gamma_spectroscopy, 2.0e-3),
            background_cps=0.3,
            dead_time_s=2e-6,
            field_of_view_half_angle_deg=22.0,
            off_axis_leakage_fraction=0.008,
        ),
        "rotating_shield": DetectorDescriptor(
            "rotating_shield",
            "Rotating shield directional counter",
            DetectorFamily.SCINTILLATOR,
            Directionality.ROTATING_COLLIMATOR,
            (DetectorOutput.COUNTS, DetectorOutput.DIRECTION),
            _gamma(gamma_survey, 1.2e-3),
            background_cps=0.3,
            dead_time_s=3e-6,
            field_of_view_half_angle_deg=35.0,
            off_axis_leakage_fraction=0.03,
        ),
        "coded_aperture_czt": DetectorDescriptor(
            "coded_aperture_czt",
            "CZT coded-aperture gamma camera",
            DetectorFamily.GAMMA_IMAGER,
            Directionality.CODED_APERTURE,
            (
                DetectorOutput.COUNTS,
                DetectorOutput.SPECTRUM,
                DetectorOutput.DIRECTION,
                DetectorOutput.IMAGE,
            ),
            _gamma(gamma_spectroscopy, 1.6e-3),
            background_cps=0.5,
            dead_time_s=1e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.025,
            energy_bin_edges_kev=SPECTRUM_BINS,
            field_of_view_half_angle_deg=45.0,
            off_axis_leakage_fraction=0.015,
        ),
        "compton_camera": DetectorDescriptor(
            "compton_camera",
            "Compton gamma camera",
            DetectorFamily.GAMMA_IMAGER,
            Directionality.COMPTON,
            (
                DetectorOutput.COUNTS,
                DetectorOutput.SPECTRUM,
                DetectorOutput.DIRECTION,
                DetectorOutput.IMAGE,
            ),
            _gamma((0.01, 0.02, 0.04, 0.09, 0.13, 0.16, 0.17, 0.20), 8.0e-3),
            background_cps=0.8,
            dead_time_s=0.5e-6,
            energy_resolution_fwhm_fraction_at_662kev=0.04,
            energy_bin_edges_kev=SPECTRUM_BINS,
            field_of_view_half_angle_deg=70.0,
            off_axis_leakage_fraction=0.05,
        ),
    }
