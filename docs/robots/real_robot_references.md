# Bundled real-robot references

RadCounterSim's bundled robots must be traceable to a real machine. A bundled
sample may use manufacturer CAD, licensed CAD, or an original procedural model
derived from published dimensions and mechanisms. It may not use an anonymous
box or generic quadrotor while presenting it as task-valid evidence.

| Simulator role | Reference machine | Geometry | Primary source |
|---|---|---|---|
| Ground radiation survey and CBRN reconnaissance | iRobot PackBot used at Fukushima Daiichi | `reference_procedural` | [TEPCO deployment specifications](https://www.tepco.co.jp/en/nu/fukushima-np/f1-roadmap/images/11042801a-e.pdf) |
| Indoor aerial LiDAR and dose mapping | Flyability Elios 3 RAD with Mirion RDS-32 | `reference_procedural` | [Flyability/Mirion release](https://www.flyability.com/news/flyability-launches-a-radiation-survey-meter-payload-for-indoor-inspection-drone-elios-3-in-partnership-with-mirion-technologies) |
| Shield handling, obstacle removal, sampling and dry decontamination | Mitsubishi Heavy Industries MEISTeR | `reference_procedural` | [MHI product specification](https://www.mhi.com/business/products-services/energy-environment/nuclear-power-generation/robot-mechatronics/meister) |
| High-pressure-water decontamination and recovery | Hitachi-GE/IRID Arounder-type machine | `reference_procedural` | [Hitachi-GE release](https://www.hitachi-hgne.co.jp/news/2013/20130308.html) |

`reference_procedural` means that dimensions, locomotion, major joints, tools,
and sensor hardpoints follow primary documentation, but surfaces and inertial
parameters are not manufacturer CAD. USD prims carry
`rad:robot:referenceModelId`, `rad:robot:sourceUrl`,
`rad:robot:geometryFidelity`, and `rad:robot:notManufacturerCAD` attributes so
downstream experiments can reject an unsuitable fidelity level.

Users may still import arbitrary USD, URDF, Xacro, MJCF, or CAD-converted
robots. Those user models should set `reference.geometry_fidelity` to
`manufacturer_asset`, `licensed_cad`, or `user_supplied` as appropriate.
