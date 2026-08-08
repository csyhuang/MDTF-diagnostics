"""Configuration for the regridding pipeline.

Every field can be set three ways, in increasing precedence: the default here,
an environment variable of the same (upper-case) name, or a command-line flag.
The environment layer exists so the shell script's documented invocations, e.g.
``DATA_DIR=... STREAM=h8a python -m regrid regrid T``, keep working unchanged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import List, Optional

from .levels import (
    P_GROUND_HPA,
    SCALE_HEIGHT_M,
    parse_level_list,
    pseudoheight_levels,
)


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value == "" else value


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


@dataclass
class RegridConfig:
    """Paths, grids and algorithm choices for one regridding run."""

    # -- inputs -------------------------------------------------------------
    case: str = "mdtf_timeslice_v4.ne120L58.001"
    data_dir: str = "/nas/winds-data/csyhuang/mdtf_data"
    out_dir: str = "/nas/winds-data/csyhuang/mdtf_data/regridded"
    work_dir: str = "/nas/winds-data/csyhuang/mdtf_data/work"

    #: History stream for the 3-D fields. ``h7i`` is instantaneous (6hrPt),
    #: ``h8a`` is 6-hour means. Use h7i for science: local wave activity is a
    #: nonlinear functional of the instantaneous PV field, so pre-averaging the
    #: input fields biases the diagnostic.
    stream: str = "h7i"
    date_range: str = "2001010121600-2002123121600"

    #: Surface pressure file. PS is published only in the h7i stream, never in
    #: h8a, so this default does not follow ``stream``.
    ps_file: Optional[str] = None

    # -- horizontal target --------------------------------------------------
    nlat: int = 181
    nlon: int = 360

    # -- vertical target ----------------------------------------------------
    pseudo_h: float = SCALE_HEIGHT_M
    pseudo_p0: float = P_GROUND_HPA
    z_top_km: float = 41.0
    dz_km: float = 1.0
    #: Explicit pressure list in Pa; overrides the pseudoheight grid when set.
    plev: Optional[List[float]] = None

    # -- algorithm ----------------------------------------------------------
    #: pg3 is a finite-volume grid, so a conservative scheme is the physically
    #: correct choice -- NOT the bilinear appropriate to the np4 spectral grid.
    #: ``traave`` is TempestRemap's FV->FV map, purpose-built for cubed spheres.
    #: Avoid the ESMF-backed aliases (aave/conserve/esmfaave): ESMF_RegridWeightGen
    #: segfaults on this grid pair in the conda-forge osx-arm64 build.
    #: ``ncoaave`` works as a fallback but is markedly slower.
    algo: str = "traave"

    #: Below-ground handling. ncremap's default ``nrs_ngh`` fabricates values by
    #: copying the nearest level; ``mss_val`` leaves them missing. Keep
    #: ``mss_val``: the POD's DataPreprocessor detects NaN, Poisson-fills it,
    #: and saves masks so the figures can mark the filled regions. Fabricated
    #: values would bypass that and be plotted as real data.
    vrt_xtr: str = "mss_val"

    #: Interpolate in log(p). ncremap does *no* vertical interpolation unless
    #: this is set -- naming a target grid alone is not enough.
    vrt_ntp: str = "log"

    #: Renormalization threshold for the horizontal remap. NOT OPTIONAL for this
    #: dataset. Vertical interpolation leaves cells missing below ground; a
    #: conservative remap then averages missing together with valid neighbours
    #: and, without renormalization, counts the missing part as zero. On a
    #: one-day test that silently produced 2599 cells at 1000 hPa holding
    #: 0 < T < 150 K. With 0.0 the same field has a minimum of 228 K and no
    #: corrupted cells.
    rnr_thr: float = 0.0

    #: Timesteps per output file. 124 = one 31-day month at 6-hourly.
    chunk: int = 124

    #: Expected ncol for ne120pg3: 6 * 120^2 * 3^2. np4 would be 777602.
    ncol_expected: int = 777600

    # -- behaviour ----------------------------------------------------------
    dry_run: bool = False

    def __post_init__(self) -> None:
        if self.ps_file is None:
            self.ps_file = os.path.join(
                self.data_dir, f"{self.case}.cam.h7i.PS.{self.date_range}.nc"
            )
        if self.chunk < 1:
            raise ValueError(f"chunk must be >= 1, got {self.chunk}")
        if self.nlat < 2 or self.nlon < 2:
            raise ValueError(f"implausible target grid {self.nlat}x{self.nlon}")

    # -- construction -------------------------------------------------------
    @classmethod
    def from_env(cls, **overrides) -> "RegridConfig":
        """Build a config from defaults, then environment, then *overrides*.

        Environment names are the upper-case field names, matching the shell
        script's variables (``DATA_DIR``, ``STREAM``, ``ALGO``, ...), with
        ``RANGE`` accepted as an alias for ``date_range``.
        """
        cfg = cls(
            case=_env("CASE", cls.case),
            data_dir=_env("DATA_DIR", cls.data_dir),
            out_dir=_env("OUT_DIR", cls.out_dir),
            work_dir=_env("WORK_DIR", cls.work_dir),
            stream=_env("STREAM", cls.stream),
            date_range=_env("RANGE", _env("DATE_RANGE", cls.date_range)),
            ps_file=os.environ.get("PS_FILE") or None,
            nlat=_env_int("NLAT", cls.nlat),
            nlon=_env_int("NLON", cls.nlon),
            pseudo_h=_env_float("PSEUDO_H", cls.pseudo_h),
            pseudo_p0=_env_float("PSEUDO_P0", cls.pseudo_p0),
            z_top_km=_env_float("Z_TOP_KM", cls.z_top_km),
            dz_km=_env_float("DZ_KM", cls.dz_km),
            plev=parse_level_list(os.environ["PLEV"]) if os.environ.get("PLEV") else None,
            algo=_env("ALGO", cls.algo),
            vrt_xtr=_env("VRT_XTR", cls.vrt_xtr),
            vrt_ntp=_env("VRT_NTP", cls.vrt_ntp),
            rnr_thr=_env_float("RNR_THR", cls.rnr_thr),
            chunk=_env_int("CHUNK", cls.chunk),
        )
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in {f.name for f in fields(cls)}:
                raise TypeError(f"unknown config field {key!r}")
            setattr(cfg, key, value)
        cfg.__post_init__()
        return cfg

    # -- derived values -----------------------------------------------------
    @property
    def levels(self) -> List[float]:
        """Target pressure levels in Pa."""
        if self.plev is not None:
            return list(self.plev)
        return pseudoheight_levels(
            self.z_top_km, self.dz_km, self.pseudo_h, self.pseudo_p0
        )

    @property
    def scrip_file(self) -> str:
        return os.path.join(self.work_dir, "ne120pg3_scrip.nc")

    @property
    def exodus_cs_file(self) -> str:
        return os.path.join(self.work_dir, "ne120.g")

    @property
    def exodus_pg_file(self) -> str:
        return os.path.join(self.work_dir, "ne120pg3.g")

    @property
    def dst_grid_file(self) -> str:
        return os.path.join(self.work_dir, f"{self.nlat}x{self.nlon}_RLL.nc")

    @property
    def map_file(self) -> str:
        # The weights depend on the destination grid as well as the algorithm,
        # so both are in the name -- changing NLAT/NLON must not silently reuse
        # a stale map file.
        return os.path.join(
            self.work_dir,
            f"map_ne120pg3_to_{self.nlat}x{self.nlon}_{self.algo}.nc",
        )

    @property
    def vrt_file(self) -> str:
        return os.path.join(self.work_dir, "vrt_plev.nc")

    def source_file(self, var: str) -> str:
        """Path to the raw native-grid file for *var*."""
        return os.path.join(
            self.data_dir, f"{self.case}.cam.{self.stream}.{var}.{self.date_range}.nc"
        )

    def output_file(self, var: str, i0: int, i1: int) -> str:
        """Path to the finished output file for one chunk."""
        return os.path.join(self.out_dir, f"{self.case}.{var}.{i0:06d}-{i1:06d}.nc")

    def describe(self) -> str:
        """Multi-line summary of the settings that affect the science."""
        levels = self.levels
        return "\n".join([
            f"  case         {self.case}",
            f"  stream       {self.stream}  ({'instantaneous' if self.stream == 'h7i' else '6-hour means'})",
            f"  data_dir     {self.data_dir}",
            f"  out_dir      {self.out_dir}",
            f"  work_dir     {self.work_dir}",
            f"  ps_file      {self.ps_file}",
            f"  grid         {self.nlat} x {self.nlon}  (cell-centred)",
            f"  levels       {len(levels)}  ({levels[0] / 100.0:.4f} -> {levels[-1] / 100.0:.4f} hPa)",
            f"  algo         {self.algo}",
            f"  vrt_ntp      {self.vrt_ntp}",
            f"  vrt_xtr      {self.vrt_xtr}",
            f"  rnr_thr      {self.rnr_thr}",
            f"  chunk        {self.chunk}",
        ])
