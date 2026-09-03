# Detector arrays and custom detectors

DetectorArray can contain any number and combination of detector models. Each
instance has an independent position and forward direction. One measurement
call returns synchronized readings for every detector. The call requires a
detector-keyed map of already transported `IncidentParticleFluence` values.
This is deliberate: detector code cannot create a private source, inverse-square,
or shielding calculation.

Built-in families include GM tubes, pressurized ion chambers, NaI(Tl), CsI(Tl),
plastic scintillators, HPGe, CZT, CdTe, gas proportional counters, alpha/beta
pancake probes, He-3 and Li-6/ZnS neutron counters, neutron rem meters,
electronic dosimeters, collimated NaI detectors, rotating shields, coded
aperture cameras, and Compton cameras.

## H3D H100 omnidirectional monitoring profile

`h3d_h100_omni` is the detector profile used by the high-wall decontamination
video.  It is grounded in the real H3D H100 Gamma-Ray Imaging Spectrometer.
The manufacturer's [product page](https://h3dgamma.com/h100.php) describes an
omnidirectional instrument for nuclear power and radioactive-waste work.  Its
[published specification](https://h3dgamma.com/H100Specs.pdf) gives a 4-pi
(360-degree) radiation field of view, a 50 keV to 3 MeV spectroscopy range,
6 cm3 of CZT, less than 1.1% FWHM at 662 keV, a 3.2 kg mass, and an IP67
enclosure.

The simulator deliberately uses only scalar counts, count rate, spectrum, and
dose trend from that all-direction sensing capability; it does not use the
H100's directional image output in this scene.  The physical envelope and
published specifications are manufacturer-backed.  The checked-in
effective-area and count-to-dose curves are a deterministic synthetic
surrogate, not H3D calibration data, and therefore retain
`response_data_status: synthetic_validation_only`.  Use a measured response
file before treating the output as instrument-calibrated evidence.

During the video every active Cs-137 surface cell is transported to the fixed
measurement-rover pose using its current activity, the checked-in 661.657 keV
emission yield, and inverse-square fluence.  The decontamination controller
updates those same cells.  Consequently the HUD and CSV measurement decline
comes from the changing simulated source, not from a separately animated
overlay value.

The video also draws a deliberately illustrative subset of straight gamma
paths and moving photon markers from the visible wall cells to the H100.  Their
visible count and opacity scale with the remaining total activity, making the
radiation field diminish alongside decontamination.  These graphics carry
`rad:visualization:transportCoupled = false`: they explain the relationship but
are not transport samples and never feed the detector result.  The detector
calculation continues to integrate every live activity-bearing wall cell.

## Rotating physical shielding

`RotatingShieldConfiguration` loads the detector response, posture program,
encoder uncertainty, posture error, and scene binding together. In
`physical_geometry` mode, `IsaacPhysicalRotatingShield` writes the actual angle
to the configured scalar USD rotation operation, requires at least one
material-tagged attenuation mesh beneath that prim, synchronizes the native
scene, and requests the detector spectrum from `IsaacRadiationSimulation`.
Missing paths, mismatched detector responses, absent geometry updates, and
unavailable sources fail explicitly.

The checked-in physical example is
`configs/detectors/rotating_shield_counter.physical.synthetic.yaml`. Its data
classification is synthetic and it is not a calibrated angular-response
dataset. `configs/detectors/rotating_shield_counter.yaml` remains a distinct
response-mask approximation; it is never substituted for physical geometry.

## Custom response table

Copy configs/detectors/custom_detector.example.yaml and provide a CSV
effective-area curve in m2 for each supported particle type. Curves never
extrapolate. The descriptor also records whether the values are synthetic or
experimentally calibrated and controls dead-time behavior, spectrum bins,
output types, and angular response.

~~~python
from radcounter.core.sensors.plugins import DetectorRegistry

registry = DetectorRegistry()
descriptor = registry.load_descriptor("configs/detectors/my_detector.yaml")
model = registry.create(descriptor.model_id)
~~~

## Python plugin

For a custom simulation or vendor SDK, implement the DetectorModel protocol or
wrap a callback with CallbackDetectorModel. Packages may publish factories
through the radcounter.detectors Python entry-point group. Factories receive
the validated DetectorDescriptor.

## ROS 2, serial, and network hardware

ExternalDetectorReadingBuffer accepts DetectorReading objects or bounded JSON
mappings from any I/O thread. ExternalDetectorModel reads the latest value on
the simulation measurement thread and rejects stale data. ROS subscribers,
serial drivers, TCP clients, and vendor SDK callbacks therefore only need to
convert their message into the common reading schema.

The simulator and hardware paths expose the same counts, count rate, spectrum,
dose rate, direction, saturation, and metadata fields.
