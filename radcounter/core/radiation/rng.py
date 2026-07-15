"""Deterministic hierarchical random streams."""

from __future__ import annotations

import hashlib

import numpy as np


class SeedManager:
    """Derive stable component RNGs from one run seed and string path."""

    def __init__(self, run_seed: int) -> None:
        if run_seed < 0:
            raise ValueError("run_seed must be nonnegative")
        self._run_seed = run_seed

    def generator(self, component_path: str) -> np.random.Generator:
        """Return a repeatable generator independent of request order."""

        digest = hashlib.sha256(component_path.encode("utf-8")).digest()
        words = np.frombuffer(digest[:16], dtype=np.uint32).tolist()
        return np.random.default_rng(np.random.SeedSequence([self._run_seed, *words]))
