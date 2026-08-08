"""Emit an intake-ESM catalog for the regridded output.

The pipeline already knows every path it wrote, so it can build the catalog
directly instead of leaving it to be hand-maintained. That closes the failure
mode that bit this project once already: ``time_range`` was derived from the
filename token ``21600`` (= 06:00), but that token is the averaging interval's
*end*, while the time coordinate sits at the interval midpoint (03:00). Here
the range is read from the ``time`` variable itself, so it cannot drift from
the data.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import re
from datetime import date
from typing import Dict, Optional, Sequence, Tuple

from .config import RegridConfig
from .nco import (
    NCOError,
    data_variable_names,
    ncdump_header,
    read_values,
    variable_attribute,
)

log = logging.getLogger(__name__)

#: Column order required by tools/catalog_builder/parsers.py::catalog_keys.
CATALOG_KEYS = [
    "activity_id", "assoc_files", "institution_id", "member_id", "realm",
    "variable_id", "table_id", "source_id", "source_type", "cell_methods",
    "cell_measures", "experiment_id", "variant_label", "grid_label", "units",
    "time_range", "chunk_freq", "standard_name", "long_name", "frequency",
    "file_name", "path",
]

#: Fallback metadata for the CAM names this POD needs. Used only when the file
#: itself does not carry the attribute -- native CAM output has no
#: standard_name, so in practice this supplies it.
CAM_VARIABLE_METADATA: Dict[str, Dict[str, str]] = {
    "T": {"standard_name": "air_temperature", "units": "K",
          "long_name": "Temperature"},
    "U": {"standard_name": "eastward_wind", "units": "m s-1",
          "long_name": "Zonal wind"},
    "V": {"standard_name": "northward_wind", "units": "m s-1",
          "long_name": "Meridional wind"},
    "PS": {"standard_name": "surface_air_pressure", "units": "Pa",
           "long_name": "Surface pressure"},
    "Z3": {"standard_name": "geopotential_height", "units": "m",
           "long_name": "Geopotential Height (above sea level)"},
    "Q": {"standard_name": "specific_humidity", "units": "kg kg-1",
          "long_name": "Specific humidity"},
}


# ---------------------------------------------------------------------------
# noleap calendar arithmetic
#
# The dataset uses `days since 2000-01-01` on a noleap (365-day) calendar.
# Implementing the conversion here keeps the package free of a cftime
# dependency; noleap is simple enough that this is a few lines rather than a
# reimplementation of a calendar library.
# ---------------------------------------------------------------------------

_MONTH_LEN = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
_MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_DAYS_PER_YEAR = 365


def _noleap_to_day_number(year: int, month: int, day: int) -> int:
    if not 1 <= month <= 12:
        raise ValueError(f"month out of range: {month}")
    if not 1 <= day <= _MONTH_LEN[month - 1]:
        raise ValueError(f"day out of range for month {month}: {day}")
    return year * _DAYS_PER_YEAR + _MONTH_START[month - 1] + (day - 1)


def _noleap_from_day_number(n: int) -> Tuple[int, int, int]:
    year, remainder = divmod(n, _DAYS_PER_YEAR)
    month = 12
    for i in range(12):
        if remainder < _MONTH_START[i] + _MONTH_LEN[i]:
            month = i + 1
            break
    day = remainder - _MONTH_START[month - 1] + 1
    return year, month, day


_RE_TIME_UNITS = re.compile(
    r"time:units\s*=\s*\"(\w+)\s+since\s+(\d+)-(\d+)-(\d+)"
)


def _time_reference(header: str) -> Tuple[str, int]:
    """``(unit, reference day number)`` parsed from the time:units attribute."""
    m = _RE_TIME_UNITS.search(header)
    if not m:
        raise NCOError("could not parse time:units")
    unit = m.group(1).lower()
    if unit not in ("days", "day"):
        raise NCOError(f"unsupported time unit {unit!r}; expected days")
    ref = _noleap_to_day_number(int(m.group(2)), int(m.group(3)), int(m.group(4)))
    return unit, ref


def _format_stamp(days_since_ref: float, ref_day: int) -> str:
    """``YYYYMMDD:HHMMSS`` for an offset in days from the reference date."""
    whole = int(days_since_ref // 1)
    frac = days_since_ref - whole
    year, month, day = _noleap_from_day_number(ref_day + whole)

    seconds = int(round(frac * 86400.0))
    # Rounding can push a value onto the next day; carry it rather than
    # emitting an impossible 24:00:00.
    if seconds >= 86400:
        seconds -= 86400
        year, month, day = _noleap_from_day_number(ref_day + whole + 1)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{year:04d}{month:02d}{day:02d}:{hours:02d}{minutes:02d}{secs:02d}"


def time_range_of(path: str, header: Optional[str] = None) -> str:
    """``time_range`` for a file, read from its own time coordinate."""
    header = ncdump_header(path) if header is None else header
    _, ref_day = _time_reference(header)
    values = read_values(path, "time", fmt="%.10f")
    return f"{_format_stamp(min(values), ref_day)}-{_format_stamp(max(values), ref_day)}"


# ---------------------------------------------------------------------------
# Catalog construction
# ---------------------------------------------------------------------------

def _row_for_file(cfg: RegridConfig, path: str, frequency: str) -> Dict[str, str]:
    """One catalog row, with metadata read from the file wherever possible."""
    header = ncdump_header(path)
    names = data_variable_names(path, header)
    if len(names) != 1:
        raise NCOError(
            f"{path} has {len(names)} data variables {names}; expected exactly "
            f"one so variable_id is unambiguous"
        )
    var = names[0]
    fallback = CAM_VARIABLE_METADATA.get(var, {})

    def attr(name: str, default: str = "") -> str:
        value = variable_attribute(path, var, name, header)
        if value:
            return value
        return fallback.get(name, default)

    row = dict.fromkeys(CATALOG_KEYS, "")
    row.update({
        "activity_id": "CESM",
        "institution_id": "NCAR",
        "member_id": "001",
        "realm": "atmos",
        "variable_id": var,
        "source_id": "CAM",
        "cell_methods": attr("cell_methods", "time: point"),
        "experiment_id": cfg.case,
        "variant_label": "001",
        "grid_label": "gr",   # 'gr' = regridded, as opposed to 'gn' native
        "units": attr("units"),
        "time_range": time_range_of(path, header),
        "standard_name": attr("standard_name"),
        "long_name": attr("long_name"),
        "frequency": frequency,
        "file_name": os.path.basename(path),
        "path": os.path.abspath(path),
    })
    if not row["standard_name"]:
        raise NCOError(
            f"no standard_name for {var} in {path} and no fallback known. "
            f"The framework queries the catalog on standard_name, so a blank "
            f"here means the POD will find nothing. Add {var} to "
            f"CAM_VARIABLE_METADATA."
        )
    return row


def _catalog_json(csv_name: str, description: str) -> Dict:
    """The intake-ESM header describing the CSV."""
    return {
        "esmcat_version": "0.0.1",
        "attributes": [
            {"column_name": key, "vocabulary": ""} for key in CATALOG_KEYS
        ],
        "assets": {
            "column_name": "path",
            "format": "netcdf",
            "format_column_name": None,
        },
        "aggregation_control": {
            "variable_column_name": "variable_id",
            "groupby_attrs": [
                "activity_id", "institution_id", "experiment_id",
                "frequency", "member_id", "realm",
            ],
            "aggregations": [
                {"type": "union", "attribute_name": "variable_id", "options": {}}
            ],
        },
        "id": csv_name,
        "description": description,
        "title": None,
        "last_updated": date.today().isoformat(),
        "catalog_file": csv_name,
    }


def write_catalog(
    cfg: RegridConfig,
    output_json: str,
    files: Optional[Sequence[str]] = None,
    frequency: str = "6hr",
    description: Optional[str] = None,
) -> Tuple[str, str]:
    """Write ``.csv`` + ``.json`` describing the regridded output.

    *files* defaults to every ``.nc`` in ``cfg.out_dir`` whose name starts with
    the case name. Returns ``(csv_path, json_path)``.

    ``frequency`` must be a string the framework's ``DateFrequency`` parser can
    read. Use ``6hr`` even for instantaneous data: ``6hrPt`` appears in the
    catalog documentation but raises ``ValueError`` in
    ``src/util/datelabel.py``.
    """
    if files is None:
        if not os.path.isdir(cfg.out_dir):
            raise NCOError(f"output directory not found: {cfg.out_dir}")
        files = sorted(
            os.path.join(cfg.out_dir, name)
            for name in os.listdir(cfg.out_dir)
            if name.startswith(cfg.case) and name.endswith(".nc")
        )
    files = [f for f in files if os.path.isfile(f) and os.path.getsize(f) > 0]
    if not files:
        raise NCOError(f"no output files found in {cfg.out_dir}")

    rows = []
    for path in files:
        try:
            rows.append(_row_for_file(cfg, path, frequency))
        except NCOError as exc:
            raise NCOError(f"building a catalog row for {path}: {exc}") from exc

    json_path = os.path.abspath(output_json)
    csv_path = os.path.splitext(json_path)[0] + ".csv"

    os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CATALOG_KEYS)
        writer.writeheader()
        writer.writerows(rows)

    if description is None:
        description = (
            f"{cfg.case}: CAM ne120pg3 output remapped to a {cfg.nlat}x{cfg.nlon} "
            f"lat-lon grid on {len(cfg.levels)} pressure levels evenly spaced in "
            f"pseudoheight"
        )
    with open(json_path, "w") as fh:
        json.dump(_catalog_json(os.path.basename(csv_path), description), fh, indent=2)
        fh.write("\n")

    variables = sorted({row["variable_id"] for row in rows})
    log.info(
        "wrote %s (%d rows, variables: %s)", csv_path, len(rows), ", ".join(variables)
    )
    log.info("wrote %s", json_path)
    return csv_path, json_path
