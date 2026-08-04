from pathlib import Path

import numpy as np

from radcounter.core.environment.importers import load_pcd_xyz_rgb


def test_binary_pcd_accepts_duplicate_padding_fields_and_trailing_reserve(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pcl-padded.pcd"
    header = """# .PCD v0.7
VERSION 0.7
FIELDS _ x y z _ rgb
SIZE 1 4 4 4 1 4
TYPE U F F F U F
COUNT 4 1 1 1 4 1
WIDTH 1
HEIGHT 1
POINTS 1
DATA binary
""".encode("ascii")
    dtype = np.dtype(
        [
            ("padding_0", "u1", (4,)),
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("padding_1", "u1", (4,)),
            ("rgb", "<f4"),
        ]
    )
    record = np.zeros(1, dtype=dtype)
    record["x"], record["y"], record["z"] = 1.0, -2.0, 3.5
    packed = np.asarray([(10 << 16) | (20 << 8) | 30], dtype="<u4")
    record["rgb"] = packed.view("<f4")
    path.write_bytes(header + record.tobytes() + b"\0" * 17)

    points, colors = load_pcd_xyz_rgb(path)

    assert np.array_equal(points, [[1.0, -2.0, 3.5]])
    assert colors is not None
    assert np.allclose(colors, [[10 / 255, 20 / 255, 30 / 255]])
