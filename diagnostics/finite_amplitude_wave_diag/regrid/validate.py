"""Cheap checks that catch the failures that matter.

Run these on the first chunk before committing hours to the full dataset. Each
check corresponds to a bug that actually occurred during development, so a
clean report here is meaningful rather than decorative.
"""

from __future__ import annotations

import logging
import math
import os
from typing import List, Optional

from .config import RegridConfig
from .levels import pressure_to_pseudoheight
from .nco import (
    NCOError,
    data_variable_names,
    dimension_names,
    ncdump_header,
    read_values,
    require_tools,
    run,
)

log = logging.getLogger(__name__)


class ValidationReport:
    """Collected pass/fail results for one file."""

    def __init__(self, path: str):
        self.path = path
        self.checks: List[tuple] = []   # (name, ok, detail)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.checks)

    def __str__(self) -> str:
        lines = [f"validation: {os.path.basename(self.path)}"]
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""))
        lines.append(f"  => {'all checks passed' if self.ok else 'FAILURES PRESENT'}")
        return "\n".join(lines)


def validate_file(cfg: RegridConfig, path: str) -> ValidationReport:
    """Check one finished output file."""
    require_tools("ncdump", "ncwa", "ncks")
    if not os.path.isfile(path):
        raise NCOError(f"not found: {path}")

    report = ValidationReport(path)
    header = ncdump_header(path)

    var = _check_single_data_variable(report, path, header)
    _check_dimensions(cfg, report, header)
    _check_latitudes(cfg, report, path)
    _check_levels(cfg, report, path)
    if var:
        _check_level_means(report, path, var)
    return report


def _check_single_data_variable(
    report: ValidationReport, path: str, header: str
) -> Optional[str]:
    """Exactly one data variable, so the catalog builder cannot pick the wrong one."""
    names = data_variable_names(path, header)
    ok = len(names) == 1
    report.add(
        "single data variable",
        ok,
        f"found {names}" if not ok else names[0],
    )
    return names[0] if names else None


def _check_dimensions(cfg: RegridConfig, report: ValidationReport, header: str) -> None:
    """Axes must be lat/lon/plev -- ncol or ilev means a stage did not run."""
    dims = set(dimension_names("", header))
    for bad, why in (
        ("ncol", "horizontal remap did not run"),
        ("ilev", "hybrid interface levels were not dropped"),
    ):
        absent = bad not in dims
        # Only explain on failure: printing the reason next to a PASS reads as
        # though the failure had occurred.
        report.add(f"no '{bad}' dimension", absent, "" if absent else why)

    has_plev = "plev" in dims
    report.add(
        "plev dimension present",
        has_plev,
        "" if has_plev else "vertical interpolation did not run (check --vrt_ntp)",
    )
    for expected in ("lat", "lon"):
        report.add(f"'{expected}' dimension present", expected in dims)


def _check_latitudes(cfg: RegridConfig, report: ValidationReport, path: str) -> None:
    """Cell-centred grid: expect about -89.5 to 89.5, not -90 to 90."""
    try:
        first = read_values(path, "lat", "lat", 0, 0)[0]
        last = read_values(path, "lat", "lat", cfg.nlat - 1, cfg.nlat - 1)[0]
    except NCOError as exc:
        report.add("latitude range", False, str(exc))
        return
    ok = -90.0 <= first < -85.0 and 85.0 < last <= 90.0
    report.add("latitude range", ok, f"{first:.4f} to {last:.4f}")


def _check_levels(cfg: RegridConfig, report: ValidationReport, path: str) -> None:
    """Level count and spacing match the requested vertical grid."""
    expected = cfg.levels
    try:
        levels = read_values(path, "plev", fmt="%.6f")
    except NCOError as exc:
        report.add("level values", False, str(exc))
        return

    count_ok = len(levels) == len(expected)
    report.add(
        "level count",
        count_ok,
        f"{len(levels)} (expected {len(expected)})",
    )
    if not count_ok:
        return

    max_err = max(abs(a - b) for a, b in zip(levels, expected))
    report.add(
        "level values match target",
        max_err < 1e-3,
        f"max |diff| = {max_err:.2e} Pa",
    )

    # Uniform in pseudoheight is the property we actually asked for; checking it
    # directly catches a mis-specified level list that still has the right count.
    if cfg.plev is None and len(levels) > 2:
        z = [pressure_to_pseudoheight(p, cfg.pseudo_h, cfg.pseudo_p0) for p in levels]
        spacing = [abs(z[i + 1] - z[i]) for i in range(len(z) - 1)]
        target = cfg.dz_km * 1000.0
        worst = max(abs(s - target) for s in spacing)
        report.add(
            "pseudoheight spacing uniform",
            worst < 1.0,
            f"{min(spacing):.6f}-{max(spacing):.6f} m (target {target:.0f})",
        )


def _check_level_means(report: ValidationReport, path: str, var: str) -> None:
    """Per-level mean of the data variable.

    An all-missing level means the vertical target went above the model top or
    the extrapolation mode discarded the column. This is also where the
    renormalization bug showed up: without ``-r``, below-ground cells came back
    as zeros and dragged the lowest level's mean far below anything physical.
    """
    tmp = path + ".chk.nc"
    try:
        run(["ncwa", "-O", "-h", "-y", "avg", "-a", "time,lat,lon", "-v", var,
             path, tmp])
        means = read_values(tmp, var, fmt="%.6f")
    except NCOError as exc:
        report.add("per-level means", False, str(exc))
        return
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass

    bad = [i for i, m in enumerate(means) if math.isnan(m)]
    report.add(
        "no all-missing levels",
        not bad,
        f"levels {bad} are entirely missing" if bad else f"{len(means)} levels",
    )
    finite = [m for m in means if not math.isnan(m)]
    if finite:
        report.add(
            "level means finite",
            all(math.isfinite(m) for m in finite),
            f"{min(finite):.3f} to {max(finite):.3f}",
        )
