# Detector arrays and custom detectors

DetectorArray can contain any number and combination of detector models. Each
instance has an independent position and forward direction. One measurement
call returns synchronized readings for every detector.

Built-in families include GM tubes, pressurized ion chambers, NaI(Tl), CsI(Tl),
plastic scintillators, HPGe, CZT, CdTe, gas proportional counters, alpha/beta
pancake probes, He-3 and Li-6/ZnS neutron counters, neutron rem meters,
electronic dosimeters, collimated NaI detectors, rotating shields, coded
aperture cameras, and Compton cameras.

## Custom response table

Copy configs/detectors/custom_detector.example.yaml and provide a CSV response
curve. The descriptor controls particle sensitivity, energy efficiency,
dead-time behavior, spectrum bins, output types, and angular response.

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
