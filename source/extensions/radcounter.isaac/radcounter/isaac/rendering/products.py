"""Independent RGB, depth, LiDAR, semantic, and radiation products."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from radcounter.core.rendering import (
    HighDoseCameraConfig,
    HighDoseCameraModel,
    RenderBudget,
    RenderProductConfig,
    RenderProductKind,
)


@dataclass(frozen=True)
class RenderProductFrame:
    product_id: str
    kind: RenderProductKind
    timestamp_s: float
    data: object
    dropped: bool = False
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass
class _RuntimeProduct:
    config: RenderProductConfig
    render_product: object | None = None
    annotator: object | None = None
    provider: Callable[[], object] | None = None
    camera_model: HighDoseCameraModel | None = None
    last_capture_s: float = float("-inf")


class IsaacRenderProductManager:
    _ANNOTATORS = {
        RenderProductKind.RGB: "rgb",
        RenderProductKind.DEPTH: "distance_to_camera",
        RenderProductKind.NORMALS: "normals",
        RenderProductKind.SEMANTIC: "semantic_segmentation",
        RenderProductKind.INSTANCE: "instance_segmentation_fast",
    }

    def __init__(
        self,
        specs: tuple[RenderProductConfig, ...],
        budget: RenderBudget,
        camera_config: HighDoseCameraConfig,
    ) -> None:
        self.specs = specs
        self.budget = budget
        self.camera_config = camera_config
        self._products: dict[str, _RuntimeProduct] = {}
        self._providers: dict[str, Callable[[], object]] = {}

    def register_provider(self, product_id: str, provider: Callable[[], object]) -> None:
        self._providers[product_id] = provider
        if product_id in self._products:
            self._products[product_id].provider = provider

    def create(self) -> Mapping[str, RenderProductConfig]:
        camera_specs = [
            item
            for item in self.specs
            if item.enabled and item.kind in self._ANNOTATORS
        ]
        rep = None
        if camera_specs:
            import omni.kit.app

            manager = omni.kit.app.get_app().get_extension_manager()
            manager.set_extension_enabled_immediate("omni.replicator.core", True)
            import omni.replicator.core as rep

        for spec in self.specs:
            if not spec.enabled:
                continue
            if spec.id in self._products:
                raise ValueError(f"duplicate render product id: {spec.id}")
            runtime = _RuntimeProduct(config=spec, provider=self._providers.get(spec.id))
            if spec.kind in self._ANNOTATORS:
                assert rep is not None and spec.sensor_prim_path is not None
                width = max(1, round(spec.resolution_px[0] * self.budget.resolution_scale))
                height = max(1, round(spec.resolution_px[1] * self.budget.resolution_scale))
                runtime.render_product = rep.create.render_product(
                    spec.sensor_prim_path, (width, height)
                )
                annotator_name = spec.annotator or self._ANNOTATORS[spec.kind]
                runtime.annotator = rep.AnnotatorRegistry.get_annotator(annotator_name)
                runtime.annotator.attach(runtime.render_product)
                if spec.kind is RenderProductKind.RGB and spec.apply_high_dose_effects:
                    runtime.camera_model = HighDoseCameraModel(self.camera_config)
            self._products[spec.id] = runtime
        return {key: value.config for key, value in self._products.items()}

    def due(self, product_id: str, now_s: float | None = None) -> bool:
        runtime = self._products[product_id]
        now = time.monotonic() if now_s is None else now_s
        effective_rate = runtime.config.update_rate_hz * self.budget.render_product_rate_scale
        return now - runtime.last_capture_s >= 1.0 / effective_rate

    def read(
        self,
        product_id: str,
        *,
        dose_rate_gy_h: float = 0.0,
        now_s: float | None = None,
    ) -> RenderProductFrame:
        runtime = self._products[product_id]
        timestamp = time.monotonic() if now_s is None else now_s
        if runtime.provider is not None:
            data = runtime.provider()
        elif runtime.annotator is not None:
            data = runtime.annotator.get_data()
        else:
            raise RuntimeError(
                f"render product {product_id!r} requires a registered LiDAR/radiation provider"
            )
        runtime.last_capture_s = timestamp
        dropped = False
        diagnostics: Mapping[str, float] = {}
        if runtime.camera_model is not None:
            raw = data.get("data") if isinstance(data, dict) and "data" in data else data
            exposure = 1.0 / runtime.config.update_rate_hz
            result = runtime.camera_model.process(
                np.asarray(raw), dose_rate_gy_h=dose_rate_gy_h, exposure_s=exposure
            )
            data = result.image
            dropped = result.dropped
            diagnostics = result.diagnostics
        return RenderProductFrame(
            product_id=product_id,
            kind=runtime.config.kind,
            timestamp_s=timestamp,
            data=data,
            dropped=dropped,
            diagnostics=diagnostics,
        )

    def capture_due(
        self,
        *,
        dose_rate_by_product: Mapping[str, float] | None = None,
        now_s: float | None = None,
    ) -> dict[str, RenderProductFrame]:
        now = time.monotonic() if now_s is None else now_s
        dose = dose_rate_by_product or {}
        return {
            product_id: self.read(
                product_id,
                dose_rate_gy_h=dose.get(product_id, 0.0),
                now_s=now,
            )
            for product_id in self._products
            if self.due(product_id, now)
        }

    @staticmethod
    def write_bundle(frames: Mapping[str, RenderProductFrame], path: str | Path) -> Path:
        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            key: np.asarray(frame.data)
            for key, frame in frames.items()
            if isinstance(frame.data, (np.ndarray, list, tuple))
        }
        metadata = {
            key: {
                "kind": frame.kind.value,
                "timestamp_s": frame.timestamp_s,
                "dropped": frame.dropped,
                "diagnostics": dict(frame.diagnostics),
            }
            for key, frame in frames.items()
        }
        np.savez_compressed(destination, **arrays, metadata_json=json.dumps(metadata))
        return destination

    def close(self) -> None:
        for runtime in self._products.values():
            if runtime.annotator is not None and runtime.render_product is not None:
                runtime.annotator.detach(runtime.render_product)
        self._products.clear()
