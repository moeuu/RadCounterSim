"""Fail-closed radiological disposition for physically delivered objects."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class DisposalDisposition(StrEnum):
    """The only supported reasons for changing a delivered source state."""

    SHIELDED_STORAGE = "shielded_storage"
    OUT_OF_EVALUATION_DOMAIN = "out_of_evaluation_domain"


@dataclass(frozen=True, slots=True)
class DisposalConfiguration:
    zone_path: str
    accepts_class: str
    disposition: DisposalDisposition
    storage_prim_path: str | None
    storage_geometry_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DisposalStateChange:
    disposition: DisposalDisposition
    source_present: bool | None
    storage_prim_path: str | None
    storage_geometry_paths: tuple[str, ...]


def _required_value(prim: Any, name: str) -> Any:
    attribute = prim.GetAttribute(name)
    if not attribute or not attribute.HasAuthoredValueOpinion():
        raise ValueError(f"{prim.GetPath()} requires authored {name}")
    value = attribute.Get()
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{prim.GetPath()} has empty {name}")
    return value


def disposal_configuration(stage: Any, zone_path: str) -> DisposalConfiguration:
    """Validate a disposal zone before any source or manipulation state changes."""

    from pxr import Usd, UsdGeom

    zone = stage.GetPrimAtPath(zone_path)
    if not zone or not zone.IsValid() or not zone.IsActive():
        raise ValueError(f"disposal zone does not exist or is inactive: {zone_path}")
    if str(_required_value(zone, "rad:role")) != "disposal_zone":
        raise ValueError(f"{zone_path} is not tagged as a disposal_zone")
    accepts_class = str(_required_value(zone, "rad:disposal:acceptsClass"))
    raw_disposition = str(_required_value(zone, "rad:disposal:disposition"))
    try:
        disposition = DisposalDisposition(raw_disposition)
    except ValueError as error:
        raise ValueError(
            f"unsupported disposal disposition on {zone_path}: {raw_disposition!r}"
        ) from error

    storage_path_attribute = zone.GetAttribute("rad:disposal:storagePrimPath")
    storage_path = (
        str(storage_path_attribute.Get())
        if storage_path_attribute and storage_path_attribute.HasAuthoredValueOpinion()
        else None
    )
    outside_attribute = zone.GetAttribute("rad:disposal:outsideEvaluationDomain")
    outside = bool(outside_attribute.Get()) if outside_attribute else False

    storage_geometry_paths: tuple[str, ...] = ()
    if disposition is DisposalDisposition.SHIELDED_STORAGE:
        if outside:
            raise ValueError("shielded storage cannot be outside the evaluation domain")
        if not storage_path or not storage_path.startswith("/"):
            raise ValueError(f"{zone_path} requires an absolute storagePrimPath")
        storage = stage.GetPrimAtPath(storage_path)
        if not storage or not storage.IsValid() or not storage.IsActive():
            raise ValueError(f"shielded-storage geometry does not exist: {storage_path}")
        tagged_geometry: list[str] = []
        for prim in Usd.PrimRange(storage):
            if not prim.IsActive() or not prim.IsA(UsdGeom.Gprim):
                continue
            material = prim.GetAttribute("rad:material:id")
            if material and material.HasAuthoredValueOpinion() and str(material.Get()).strip():
                tagged_geometry.append(str(prim.GetPath()))
        if not tagged_geometry:
            raise ValueError(
                f"shielded storage {storage_path} has no visible material-tagged geometry"
            )
        storage_geometry_paths = tuple(tagged_geometry)
    else:
        if storage_path:
            raise ValueError("out-of-domain disposal cannot reference storage geometry")
        if not outside_attribute or not outside_attribute.HasAuthoredValueOpinion() or not outside:
            raise ValueError(
                f"{zone_path} must explicitly set outsideEvaluationDomain=true"
            )

    return DisposalConfiguration(
        zone_path,
        accepts_class,
        disposition,
        storage_path,
        storage_geometry_paths,
    )

def apply_disposal_state(
    stage: Any,
    target: Any,
    configuration: DisposalConfiguration,
) -> DisposalStateChange:
    """Apply validated source-presence semantics after physical containment."""

    from pxr import Sdf

    if not target or not target.IsValid() or not target.IsActive():
        raise ValueError("disposal target does not exist or is inactive")
    target_class = str(_required_value(target, "rad:manipulation:disposalClass"))
    if target_class != configuration.accepts_class:
        raise ValueError(
            f"disposal class {target_class!r} is not accepted by {configuration.zone_path}"
        )

    source_type = target.GetAttribute("rad:source:type")
    source_enabled = target.GetAttribute("rad:source:enabled")
    source_present: bool | None = None
    if source_type and source_type.HasAuthoredValueOpinion():
        if not source_enabled or not source_enabled.HasAuthoredValueOpinion():
            raise ValueError(f"source target {target.GetPath()} requires rad:source:enabled")
        source_present = bool(source_enabled.Get())
        if configuration.disposition is DisposalDisposition.SHIELDED_STORAGE:
            if not source_present:
                raise ValueError("a source entering shielded storage must remain present")
        else:
            source_enabled.Set(False)
            source_present = False

    def set_attribute(name: str, value_type: Any, value: Any) -> None:
        attribute = target.GetAttribute(name)
        if not attribute:
            attribute = target.CreateAttribute(name, value_type, custom=True)
        attribute.Set(value)

    set_attribute("rad:disposal:disposed", Sdf.ValueTypeNames.Bool, True)
    set_attribute(
        "rad:disposal:disposition",
        Sdf.ValueTypeNames.String,
        configuration.disposition.value,
    )
    contained = configuration.disposition is DisposalDisposition.SHIELDED_STORAGE
    set_attribute("rad:source:contained", Sdf.ValueTypeNames.Bool, contained)
    set_attribute(
        "rad:source:containmentPrimPath",
        Sdf.ValueTypeNames.String,
        configuration.storage_prim_path or "",
    )
    for name in ("rad:manipulation:movable", "rad:manipulation:removable"):
        attribute = target.GetAttribute(name)
        if attribute:
            attribute.Set(False)

    return DisposalStateChange(
        configuration.disposition,
        source_present,
        configuration.storage_prim_path,
        configuration.storage_geometry_paths,
    )
