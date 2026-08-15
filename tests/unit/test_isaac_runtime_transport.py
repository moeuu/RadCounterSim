import importlib
from pathlib import Path

import numpy as np
import pytest

import radcounter

ROOT = Path(__file__).resolve().parents[2]
EXTENSION_NAMESPACE = str(ROOT / "source/extensions/radcounter.isaac/radcounter")
if EXTENSION_NAMESPACE not in radcounter.__path__:
    radcounter.__path__.append(EXTENSION_NAMESPACE)

_ensure_material_columns = importlib.import_module(
    "radcounter.isaac.runtime.simulation"
)._ensure_material_columns


def test_absent_configured_materials_are_zero_padded() -> None:
    paths = _ensure_material_columns(np.asarray([[0.4, 0.2]]), material_count=3)
    np.testing.assert_allclose(paths, [[0.4, 0.2, 0.0]])


def test_native_scene_cannot_return_more_materials_than_configured() -> None:
    with pytest.raises(ValueError, match="4 material columns"):
        _ensure_material_columns(np.ones((2, 4)), material_count=3)
