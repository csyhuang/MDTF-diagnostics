"""Vertical target levels.

falwa works in pseudoheight,

    z = -H * ln(p / P_GROUND)    <=>    p = P_GROUND * exp(-z / H)

so a grid evenly spaced in pseudoheight is just a particular list of pressures
and needs no new machinery. The constants match ``falwa.constant``
(``P_GROUND`` = 1000 hPa, ``SCALE_HEIGHT`` = 7000 m), which is what the POD's
own ``convert_hPa_to_pseudoheight`` uses.

The stored coordinate stays ``plev`` in Pa on purpose: ``settings.jsonc``
declares the Z axis as ``air_pressure``/Pa and the preprocessor expects a
pressure axis, so writing a height coordinate here would break the varlist.
falwa converts to pseudoheight itself; the values simply happen to be uniform
in z.

Interpolation is done in log(p), which is exactly linear in pseudoheight, so
the scheme is the natural one for this grid rather than a coincidence.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

#: Reference pressure, hPa. ``falwa.constant.P_GROUND``.
P_GROUND_HPA = 1000.0

#: Scale height, m. ``falwa.constant.SCALE_HEIGHT``.
SCALE_HEIGHT_M = 7000.0


def pseudoheight_to_pressure(
    z_m: float,
    scale_height: float = SCALE_HEIGHT_M,
    p_ground_hpa: float = P_GROUND_HPA,
) -> float:
    """Pressure in **Pa** at pseudoheight *z_m* (metres)."""
    return p_ground_hpa * math.exp(-z_m / scale_height) * 100.0


def pressure_to_pseudoheight(
    p_pa: float,
    scale_height: float = SCALE_HEIGHT_M,
    p_ground_hpa: float = P_GROUND_HPA,
) -> float:
    """Pseudoheight in **metres** at pressure *p_pa* (Pa)."""
    return -scale_height * math.log((p_pa / 100.0) / p_ground_hpa)


def pseudoheight_levels(
    z_top_km: float,
    dz_km: float,
    scale_height: float = SCALE_HEIGHT_M,
    p_ground_hpa: float = P_GROUND_HPA,
) -> List[float]:
    """Pressures in **Pa** for levels evenly spaced in pseudoheight.

    Levels run from z = 0 to *z_top_km* inclusive at *dz_km* spacing, so
    ``z_top_km=41, dz_km=1`` yields 42 levels from 1000.0000 to 2.8594 hPa.

    ``z_top_km`` is bounded by the model top. For this dataset the top is
    2.838 hPa, which is z = 41.05 km, so 41 km is the last level with real data
    above it -- asking for 42 km produces an all-missing level.
    """
    if dz_km <= 0:
        raise ValueError(f"dz_km must be positive, got {dz_km}")
    if z_top_km < 0:
        raise ValueError(f"z_top_km must be non-negative, got {z_top_km}")

    levels = []
    n = int(math.floor(z_top_km / dz_km + 1e-9)) + 1
    for i in range(n):
        z_km = i * dz_km
        levels.append(
            pseudoheight_to_pressure(z_km * 1000.0, scale_height, p_ground_hpa)
        )
    return levels


def parse_level_list(text: str) -> List[float]:
    """Parse an explicit level list, e.g. ``"100000, 92500, 85000"`` (Pa).

    Accepts commas and/or whitespace as separators, so a value copied out of a
    CDL file or typed by hand both work.
    """
    tokens = [t for t in text.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("empty level list")
    try:
        levels = [float(t) for t in tokens]
    except ValueError as exc:
        raise ValueError(f"could not parse level list {text!r}: {exc}") from exc
    if any(p <= 0 for p in levels):
        raise ValueError("pressures must be positive (Pa)")
    return levels


def format_levels_cdl(levels: Sequence[float]) -> str:
    """Comma-separated levels for the ``data:`` section of the vertical-grid CDL."""
    return ", ".join(f"{p:.6f}" for p in levels)


def format_level_table(
    levels: Iterable[float],
    scale_height: float = SCALE_HEIGHT_M,
    p_ground_hpa: float = P_GROUND_HPA,
) -> str:
    """Human-readable table of level index, pseudoheight and pressure."""
    rows = ["  idx      z (km)     p (hPa)        p (Pa)",
            "  ---   ---------   ---------   -----------"]
    for i, p_pa in enumerate(levels):
        z_km = pressure_to_pseudoheight(p_pa, scale_height, p_ground_hpa) / 1000.0
        if z_km == 0.0:
            z_km = 0.0   # normalise -0.0, which log() returns at p == p_ground
        rows.append(f"  {i:3d}   {z_km:9.4f}   {p_pa / 100.0:9.4f}   {p_pa:11.4f}")
    return "\n".join(rows)


def levels_above_model_top(
    levels: Iterable[float], model_top_hpa: float
) -> List[float]:
    """Requested levels (Pa) that sit above *model_top_hpa*, i.e. have no data.

    Those levels come back entirely missing and then propagate NaN into any
    column-wise diagnostic, so the caller should warn loudly.
    """
    return [p for p in levels if p / 100.0 < model_top_hpa]
