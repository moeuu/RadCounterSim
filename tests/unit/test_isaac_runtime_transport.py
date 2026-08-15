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
_usd_row_matrix_values = importlib.import_module(
    "radcounter.isaac.usd.environment"
)._usd_row_matrix_values


def test_absent_configured_materials_are_zero_padded() -> None:
    paths = _ensure_material_columns(np.asarray([[0.4, 0.2]]), material_count=3)
    np.testing.assert_allclose(paths, [[0.4, 0.2, 0.0]])


def test_native_scene_cannot_return_more_materials_than_configured() -> None:
    with pytest.raises(ValueError, match="4 material columns"):
        _ensure_material_columns(np.ones((2, 4)), material_count=3)


def test_native_usd_reference_transposes_column_vector_transform() -> None:
    matrix = np.asarray(
        [
            [0.0, -0.001, 0.0, -35.0],
            [0.001, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.001, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    rows = np.asarray(_usd_row_matrix_values(matrix))
    np.testing.assert_allclose(rows, matrix.T)
    np.testing.assert_allclose(rows[3, :3], (-35.0, 2.0, 3.0))
