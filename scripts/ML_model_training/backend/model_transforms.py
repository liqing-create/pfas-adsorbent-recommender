"""Stable, pickle-safe sklearn transformation helpers."""

from __future__ import annotations

import numpy as np


def as_float_array(values):
    """Return model-ready numeric values while preserving native missing values."""
    if hasattr(values, "to_numpy"):
        return values.to_numpy(dtype=float)
    return np.asarray(values, dtype=float)
