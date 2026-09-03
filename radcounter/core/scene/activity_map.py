"""Integrity-checked NPZ sidecars for per-triangle surface activity."""

from __future__ import annotations

import hashlib
import hmac
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class ActivityMapIntegrityError(ValueError):
    """Raised when an activity-map sidecar is malformed or has the wrong digest."""


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA256 digest of one file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_asset_uri(uri: str, base_directory: str | Path) -> Path:
    """Resolve a local path or ``file://`` URI relative to a USD layer."""

    if not uri:
        raise ActivityMapIntegrityError("activity-map URI must not be empty")
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file"}:
        raise ActivityMapIntegrityError(f"unsupported activity-map URI scheme: {parsed.scheme}")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise ActivityMapIntegrityError("remote file URIs are not supported")
    raw_path = unquote(parsed.path) if parsed.scheme == "file" else uri
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path(base_directory).expanduser() / path
    return path.resolve()


@dataclass
class SurfaceActivityMap:
    """Mutable per-triangle truth state persisted outside a USD layer."""

    triangle_indices: IntArray
    activity_bq: FloatArray
    cumulative_treatment_exposure: FloatArray
    last_treated_step: IntArray
    verified_contact_dwell_s: FloatArray
    source_path: Path | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        triangle_indices = np.asarray(self.triangle_indices, dtype=np.int64)
        activity_bq = np.asarray(self.activity_bq, dtype=np.float64)
        exposure = np.asarray(self.cumulative_treatment_exposure, dtype=np.float64)
        last_step = np.asarray(self.last_treated_step, dtype=np.int64)
        verified_dwell = np.asarray(self.verified_contact_dwell_s, dtype=np.float64)
        arrays = (triangle_indices, activity_bq, exposure, last_step, verified_dwell)
        if any(array.ndim != 1 for array in arrays):
            raise ActivityMapIntegrityError("activity-map arrays must be one-dimensional")
        if any(array.shape != triangle_indices.shape for array in arrays[1:]):
            raise ActivityMapIntegrityError("activity-map arrays must have equal lengths")
        if np.any(triangle_indices < 0) or len(np.unique(triangle_indices)) != len(
            triangle_indices
        ):
            raise ActivityMapIntegrityError("triangle indices must be unique and nonnegative")
        if (
            np.any(activity_bq < 0.0)
            or np.any(exposure < 0.0)
            or np.any(verified_dwell < 0.0)
            or not np.all(np.isfinite(activity_bq))
            or not np.all(np.isfinite(exposure))
            or not np.all(np.isfinite(verified_dwell))
        ):
            raise ActivityMapIntegrityError("activity and exposure must be finite and nonnegative")
        if np.any(last_step < -1):
            raise ActivityMapIntegrityError("last_treated_step values must be at least -1")
        self.triangle_indices = triangle_indices
        self.activity_bq = activity_bq
        self.cumulative_treatment_exposure = exposure
        self.last_treated_step = last_step
        self.verified_contact_dwell_s = verified_dwell
        if self.sha256 is not None:
            normalized = self.sha256.lower()
            if len(normalized) != 64 or any(
                character not in "0123456789abcdef" for character in normalized
            ):
                raise ActivityMapIntegrityError("sha256 must contain 64 hexadecimal characters")
            self.sha256 = normalized

    @classmethod
    def load(
        cls,
        uri: str,
        *,
        base_directory: str | Path,
        expected_sha256: str | None = None,
    ) -> SurfaceActivityMap:
        """Load and validate a sidecar without permitting pickled objects."""

        path = resolve_asset_uri(uri, base_directory)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if expected_sha256 is not None and not hmac.compare_digest(
            actual_sha256, expected_sha256.lower()
        ):
            raise ActivityMapIntegrityError(
                "activity-map SHA256 mismatch for "
                f"{path}: expected {expected_sha256}, got {actual_sha256}"
            )
        with np.load(path, allow_pickle=False) as archive:
            required = {"triangle_indices", "activity_bq"}
            missing = required.difference(archive.files)
            if missing:
                raise ActivityMapIntegrityError(
                    f"activity-map sidecar is missing arrays: {', '.join(sorted(missing))}"
                )
            triangle_indices = np.asarray(archive["triangle_indices"], dtype=np.int64)
            activity_bq = np.asarray(archive["activity_bq"], dtype=np.float64)
            exposure = (
                np.asarray(archive["cumulative_treatment_exposure"], dtype=np.float64)
                if "cumulative_treatment_exposure" in archive.files
                else np.zeros_like(activity_bq)
            )
            last_step = (
                np.asarray(archive["last_treated_step"], dtype=np.int64)
                if "last_treated_step" in archive.files
                else np.full(triangle_indices.shape, -1, dtype=np.int64)
            )
            verified_dwell = (
                np.asarray(archive["verified_contact_dwell_s"], dtype=np.float64)
                if "verified_contact_dwell_s" in archive.files
                else np.zeros_like(activity_bq)
            )
        return cls(
            triangle_indices,
            activity_bq,
            exposure,
            last_step,
            verified_dwell,
            source_path=path,
            sha256=actual_sha256,
        )

    def save(self, path: str | Path | None = None) -> str:
        """Atomically save the sidecar and return its new SHA256 digest."""

        destination = self.source_path if path is None else Path(path).expanduser().resolve()
        if destination is None:
            raise ValueError("an activity-map destination path is required")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                np.savez_compressed(
                    stream,
                    triangle_indices=self.triangle_indices,
                    activity_bq=self.activity_bq,
                    cumulative_treatment_exposure=self.cumulative_treatment_exposure,
                    last_treated_step=self.last_treated_step,
                    verified_contact_dwell_s=self.verified_contact_dwell_s,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        self.source_path = destination
        self.sha256 = sha256_file(destination)
        return self.sha256

    def apply_exposure(
        self,
        triangle_indices: IntArray,
        exposure_increment_s: FloatArray,
        *,
        rate_constant_s_inv: float,
        efficiency: FloatArray | float = 1.0,
        simulation_step: int,
    ) -> FloatArray:
        """Apply incremental decontamination exposure and return removed activity."""

        requested = np.asarray(triangle_indices, dtype=np.int64)
        increment = np.asarray(exposure_increment_s, dtype=np.float64)
        if requested.ndim != 1 or increment.shape != requested.shape:
            raise ValueError("triangle indices and exposure increments must be equal vectors")
        if rate_constant_s_inv <= 0.0 or np.any(increment < 0.0) or simulation_step < 0:
            raise ValueError("decontamination rate, exposure, or simulation step is invalid")
        efficiency_array = np.broadcast_to(
            np.asarray(efficiency, dtype=np.float64), requested.shape
        )
        if (
            np.any(efficiency_array < 0.0)
            or np.any(efficiency_array > 1.0)
            or not np.all(np.isfinite(efficiency_array))
        ):
            raise ValueError("decontamination efficiency must be finite and in [0, 1]")
        slot_by_triangle = {
            int(triangle_index): slot
            for slot, triangle_index in enumerate(self.triangle_indices.tolist())
        }
        try:
            slots = np.asarray(
                [slot_by_triangle[int(triangle_index)] for triangle_index in requested],
                dtype=np.int64,
            )
        except KeyError as error:
            raise KeyError(f"triangle {error.args[0]} is absent from the activity map") from error
        actual_fraction = 1.0 - np.exp(
            -rate_constant_s_inv * increment * efficiency_array
        )
        removed = self.activity_bq[slots] * actual_fraction
        self.activity_bq[slots] -= removed
        self.cumulative_treatment_exposure[slots] += increment
        self.last_treated_step[slots] = simulation_step
        return removed


@dataclass(frozen=True)
class VolumeActivityMap:
    """Integrity-checked voxel centers and activities in source-local coordinates."""

    voxel_centers_local_m: FloatArray
    activity_bq_per_voxel: FloatArray
    source_path: Path
    sha256: str

    def __post_init__(self) -> None:
        centers = np.asarray(self.voxel_centers_local_m, dtype=np.float64)
        activity = np.asarray(self.activity_bq_per_voxel, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1:] != (3,) or len(centers) == 0:
            raise ActivityMapIntegrityError("voxel centers must have nonempty shape (N, 3)")
        if activity.shape != (len(centers),):
            raise ActivityMapIntegrityError("voxel activity must have shape (N,)")
        if (
            not np.all(np.isfinite(centers))
            or not np.all(np.isfinite(activity))
            or np.any(activity < 0.0)
        ):
            raise ActivityMapIntegrityError("voxel centers/activity must be finite and nonnegative")
        if len(np.unique(centers, axis=0)) != len(centers):
            raise ActivityMapIntegrityError("voxel centers must be unique")
        normalized_digest = self.sha256.lower()
        if len(normalized_digest) != 64 or any(
            character not in "0123456789abcdef" for character in normalized_digest
        ):
            raise ActivityMapIntegrityError("sha256 must contain 64 hexadecimal characters")
        object.__setattr__(self, "voxel_centers_local_m", centers)
        object.__setattr__(self, "activity_bq_per_voxel", activity)
        object.__setattr__(self, "source_path", self.source_path.resolve())
        object.__setattr__(self, "sha256", normalized_digest)

    @classmethod
    def load(
        cls,
        uri: str,
        *,
        base_directory: str | Path,
        expected_sha256: str,
    ) -> VolumeActivityMap:
        """Load a voxel map whose digest is mandatory for scene consistency."""

        if not expected_sha256:
            raise ActivityMapIntegrityError("volume activity-map SHA256 is required")
        path = resolve_asset_uri(uri, base_directory)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_sha256 = sha256_file(path)
        if not hmac.compare_digest(actual_sha256, expected_sha256.lower()):
            raise ActivityMapIntegrityError(
                "volume activity-map SHA256 mismatch for "
                f"{path}: expected {expected_sha256}, got {actual_sha256}"
            )
        with np.load(path, allow_pickle=False) as archive:
            required = {"voxel_centers_local_m", "activity_bq_per_voxel"}
            missing = required.difference(archive.files)
            if missing:
                raise ActivityMapIntegrityError(
                    f"volume activity map is missing arrays: {', '.join(sorted(missing))}"
                )
            centers = np.asarray(archive["voxel_centers_local_m"], dtype=np.float64)
            activity = np.asarray(archive["activity_bq_per_voxel"], dtype=np.float64)
        return cls(centers, activity, path, actual_sha256)
