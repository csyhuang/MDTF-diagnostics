#!/usr/bin/env python3
"""Build the ERA5 digest that the POD compares model output against.

This is the ``obs-compute`` mode: an **offline, one-off** job spanning decades
of reanalysis, producing four small netCDF files. It must never run as part of
``./mdtf`` -- at run time the POD is in ``obs-read`` mode and simply loads what
this wrote.

    python -u make_era5_digest.py \\
        --era5-root /nas/winds-data/csyhuang/ERA5-from-2019 \\
                    /nas/winds-data2/csyhuang/ERA5 \\
        --start-year 1991 --end-year 2020 \\
        --output-dir ../../inputdata/obs_data/finite_amplitude_wave_diag

What comes out, per season: the 30-year mean of the six diagnostics, and their
interannual standard deviation. The mean alone cannot say whether a model
difference is meaningful; sigma gives it a yardstick. That matters especially
here, because the model record is short -- a two-year seasonal mean differs
from a 30-year observed mean by roughly sigma/sqrt(2) from sampling alone,
before any question of bias.

Three accumulation choices, none of them arbitrary:

* **Means are pooled, not averaged over per-year means.** Sums and counts are
  carried across all years and divided once at the end. With 29 February
  discarded every year holds the same number of timesteps, so the two agree --
  but pooling also survives a missing month, which averaging does not.

* **Covariance is pooled from raw moments.** cov(pooled) is not the mean of
  per-year covariances: the latter removes each year's own mean and so throws
  away interannual covariance entirely. Sum(a), Sum(b), Sum(ab) and n are
  accumulated and combined once at the end.

* **Sigma is across years, of the seasonal mean.** Not the spread of
  instantaneous fields, which would be far larger and answer a different
  question.

Resumable: state is checkpointed after every year, so a job killed at year 20
resumes at year 21 rather than starting over. At roughly 1500 timesteps per
season-year this is a many-hour run.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import finite_amplitude_wave_diag_zonal_mean_rework as fawd   # noqa: E402

#: Fields carried through as (mean, sigma) pairs.
MEAN_FIELDS = ("zonal_mean_u", "uref", "zonal_mean_lwa", "lwa_baro", "u_baro")

#: (height, lat) rather than (lat, lon).
YZ_FIELDS = ("zonal_mean_u", "uref", "zonal_mean_lwa")


def compute_year_statistics(season_result) -> Dict[str, np.ndarray]:
    """Per-year summary statistics for one season.

    What is stored is chosen so that pooled quantities over **any subset of
    years** can be reconstructed exactly afterwards, rather than approximated:
    ``n``, ``mean`` and ``variance`` are jointly sufficient for the pooled mean
    and pooled variance,

        s2_pooled = [ sum_y (n_y - 1) s2_y + sum_y n_y (xbar_y - xbar)^2 ]
                    / (N - 1)

    and ``m3``/``m4`` carry the shape information for skewness and kurtosis.

    Raw moments (sum x, sum x^2, ...) would also be additive, but sum x^2 for
    LWA reaches ~1e8 while the variance is ~1e-5 of that, so forming the
    difference later cancels most of the significant digits. Computing each
    year's central moments here, in float64, avoids that entirely at the same
    storage cost.

    ``variance`` uses ddof=1 (a sample of timesteps); ``m3`` and ``m4`` are
    population central moments (divided by n), which is the convention the
    skewness and kurtosis formulas below assume:

        m2       = variance * (n-1)/n          # population second moment
        skewness = m3 / m2**1.5                 # == m3/variance**1.5 * (n/(n-1))**1.5
        kurtosis = m4 / m2**2 - 3               # excess kurtosis

    Returns a flat dict of arrays and scalars.
    """
    results = season_result.results
    stats: Dict[str, np.ndarray] = {}
    n = len(results)
    stats["n_timesteps"] = np.int32(n)

    for field in MEAN_FIELDS:
        block = np.stack([getattr(r, field) for r in results],
                         axis=0).astype(np.float64)
        mean = block.mean(axis=0)
        deviation = block - mean
        stats[f"{field}_mean"] = mean.astype(np.float32)
        stats[f"{field}_variance"] = block.var(axis=0, ddof=1).astype(np.float32) \
            if n > 1 else np.zeros_like(mean, dtype=np.float32)
        stats[f"{field}_m3"] = (deviation ** 3).mean(axis=0).astype(np.float32)
        stats[f"{field}_m4"] = (deviation ** 4).mean(axis=0).astype(np.float32)
        # Cheap QC: a corrupted year shows up here before it contaminates a
        # 30-year mean that nobody re-derives.
        stats[f"{field}_min"] = np.float32(np.nanmin(block))
        stats[f"{field}_max"] = np.float32(np.nanmax(block))
        stats[f"{field}_nan_count"] = np.int32(np.isnan(block).sum())
        del block, deviation

    a = np.stack([r.lwa_baro for r in results], axis=0).astype(np.float64)
    b = np.stack([r.u_baro for r in results], axis=0).astype(np.float64)
    stats["covariance_lwa_u_baro"] = (
        ((a - a.mean(axis=0)) * (b - b.mean(axis=0))).sum(axis=0) / (n - 1.0)
    ).astype(np.float32) if n > 1 else np.zeros(a.shape[1:], dtype=np.float32)
    return stats


def write_year_statistics(year: int, per_season: Dict[str, Dict[str, np.ndarray]],
                          coords: Dict[str, np.ndarray], output_path: str,
                          provenance: Dict[str, str]) -> None:
    """Write one year's per-season statistics.

    A `season` dimension rather than season-prefixed variable names, so the
    file can be indexed as ``ds.sel(season="DJF")`` and concatenated across
    years with ``open_mfdataset``. Seasons with no data are written as NaN with
    ``n_timesteps = 0`` rather than omitted, so every year has the same shape.
    """
    seasons = [s for s, _ in fawd.SEASON_TO_MONTHS]
    present = next(iter(per_season.values()))

    data_vars: Dict[str, tuple] = {}
    for key, sample in present.items():
        if np.isscalar(sample) or np.ndim(sample) == 0:
            dims, shape = ("season",), (len(seasons),)
        else:
            base = key.rsplit("_", 1)[0]
            spatial = ("height", "lat") if base in YZ_FIELDS else ("lat", "lon")
            dims, shape = ("season",) + spatial, (len(seasons),) + sample.shape
        dtype = np.int32 if "count" in key or key == "n_timesteps" else np.float32
        block = np.full(shape, 0 if dtype is np.int32 else np.nan, dtype=dtype)
        for i, season in enumerate(seasons):
            if season in per_season and key in per_season[season]:
                block[i] = per_season[season][key]
        data_vars[key] = (dims, block)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"season": seasons, "height": coords["height"],
                "lat": coords["lat"], "lon": coords["lon"]})
    dataset.attrs.update({
        "title": f"ERA5 finite-amplitude wave activity, per-season statistics for {year}",
        "year": year,
        "purpose": "Sufficient statistics, kept so that pooled quantities over "
                   "any subset of years can be recomputed exactly without "
                   "re-running falwa over the whole record. NOT part of the "
                   "shipped digest.",
        "pooling_variance": "s2 = [sum_y (n_y-1) s2_y + sum_y n_y (xbar_y - "
                            "xbar)^2] / (N-1)",
        "m2": "variance * (n-1)/n   (population second central moment)",
        "skewness": "m3 / m2**1.5",
        "kurtosis": "m4 / m2**2 - 3   (excess kurtosis)",
        "moment_convention": "variance uses ddof=1; m3 and m4 are population "
                             "central moments (divided by n)",
        **provenance,
    })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()


class SeasonAccumulator:
    """Running totals for one season across years.

    Holds O(1) memory in the number of years for the pooled quantities, plus
    one seasonal mean per year for the interannual spread -- about 1.2 MiB a
    year, so 35 MiB over three decades.
    """

    def __init__(self, season: str):
        self.season = season
        self.sums: Dict[str, np.ndarray] = {}
        self.n_timesteps = 0
        self.per_year_means: List[Dict[str, np.ndarray]] = []
        self.years: List[int] = []
        # Raw moments for the pooled covariance of lwa_baro and u_baro.
        self.sum_a: Optional[np.ndarray] = None
        self.sum_b: Optional[np.ndarray] = None
        self.sum_ab: Optional[np.ndarray] = None
        self.per_year_cov: List[np.ndarray] = []
        self.kmax: Optional[int] = None
        self.dz: Optional[float] = None
        self.lat: Optional[np.ndarray] = None
        self.lon: Optional[np.ndarray] = None
        self.height: Optional[np.ndarray] = None

    def add_year(self, year: int, season_result) -> None:
        results = season_result.results
        stacked = {f: np.stack([getattr(r, f) for r in results], axis=0)
                   for f in MEAN_FIELDS}

        for field, block in stacked.items():
            total = block.sum(axis=0, dtype=np.float64)
            self.sums[field] = total if field not in self.sums \
                else self.sums[field] + total
        self.n_timesteps += len(results)

        a = stacked["lwa_baro"].astype(np.float64)
        b = stacked["u_baro"].astype(np.float64)
        for name, value in (("sum_a", a.sum(axis=0)), ("sum_b", b.sum(axis=0)),
                            ("sum_ab", (a * b).sum(axis=0))):
            current = getattr(self, name)
            setattr(self, name, value if current is None else current + value)

        self.per_year_means.append(
            {f: block.mean(axis=0) for f, block in stacked.items()})
        # This year's own covariance, computed here while the timesteps are in
        # hand. It cannot be recovered later from the pooled moments, and the
        # product of the two yearly means is a different quantity entirely.
        n_year = float(a.shape[0])
        self.per_year_cov.append(
            ((a * b).sum(axis=0) - a.sum(axis=0) * b.sum(axis=0) / n_year)
            / (n_year - 1.0))
        self.years.append(year)

        self.kmax = season_result.kmax
        self.dz = season_result.dz
        self.height = np.asarray(season_result.analysis_height_array)

    def finalize(self) -> Dict[str, np.ndarray]:
        """Pooled means, interannual sigma, and the pooled covariance."""
        if self.n_timesteps == 0:
            raise ValueError(f"{self.season}: nothing accumulated")

        out: Dict[str, np.ndarray] = {
            f: (self.sums[f] / self.n_timesteps).astype(np.float32)
            for f in MEAN_FIELDS}

        n = float(self.n_timesteps)
        covariance = (self.sum_ab - self.sum_a * self.sum_b / n) / (n - 1.0)
        out["covariance_lwa_u_baro"] = covariance.astype(np.float32)

        # ddof=1: sigma of a sample of years, not of a population.
        n_years = len(self.per_year_means)
        for field in MEAN_FIELDS:
            block = np.stack([m[field] for m in self.per_year_means], axis=0)
            out[f"{field}_sigma"] = block.std(axis=0, ddof=1).astype(np.float32) \
                if n_years > 1 else np.zeros_like(block[0], dtype=np.float32)

        # Sigma of the covariance: the spread of the per-year covariances that
        # add_year computed while each year's timesteps were in hand. It cannot
        # be recovered from the pooled moments, and the product of the two
        # yearly means is a different quantity. A second moment of a second
        # moment, so noisier at 30 samples than the others -- included for
        # completeness rather than because it is well constrained.
        if n_years > 1:
            out["covariance_lwa_u_baro_sigma"] = np.std(
                np.stack(self.per_year_cov, axis=0), axis=0, ddof=1
            ).astype(np.float32)
        else:
            out["covariance_lwa_u_baro_sigma"] = np.zeros_like(
                out["covariance_lwa_u_baro"])
        return out


def write_season(accumulator: SeasonAccumulator, fields: Dict[str, np.ndarray],
                 output_path: str, args, provenance: Dict[str, str]) -> None:
    """Write one season's digest."""
    data_vars = {}
    for name, values in fields.items():
        base = name.replace("_sigma", "")
        dims = ("height", "lat") if base in YZ_FIELDS else ("lat", "lon")
        data_vars[name] = (dims, values)

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={"height": accumulator.height,
                "lat": accumulator.lat, "lon": accumulator.lon})
    dataset["height"].attrs.update(units="m", long_name="pseudoheight")
    dataset["lat"].attrs.update(units="degrees_north")
    dataset["lon"].attrs.update(units="degrees_east")

    for name in fields:
        if name.endswith("_sigma"):
            dataset[name].attrs["long_name"] = (
                f"interannual standard deviation of {name[:-6]} "
                f"across {len(accumulator.years)} years")
        dataset[name].attrs["units"] = "m s-1"

    dataset.attrs.update({
        "title": f"ERA5 finite-amplitude wave activity climatology, {accumulator.season}",
        "season": accumulator.season,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "n_years": len(accumulator.years),
        "years": ",".join(str(y) for y in accumulator.years),
        "n_timesteps_pooled": accumulator.n_timesteps,
        # kmax and dz are recorded so the POD can warn when they differ from
        # the model's. See the Cautions section of the POD documentation:
        # uref comes from a column-wide inversion, so a different kmax changes
        # it at every height, not only near the top.
        "kmax": accumulator.kmax,
        "dz": accumulator.dz,
        "analysis_grid_top_m": float(accumulator.height[-1]),
        "leap_day_discarded": "yes -- 29 February removed so every year has "
                              "equal weight and the calendar matches the "
                              "model's noleap",
        "season_definition": "climatological: months pooled within each year "
                             "(DJF = Jan+Feb+Dec of the same year)",
        "below_ground_treatment": "ERA5 pressure-level fields are extrapolated "
                                  "below the surface by ECMWF; no gridfill was "
                                  "applied. The model path Poisson-fills "
                                  "genuinely missing values. Read the lowest "
                                  "one or two levels with that in mind.",
        **provenance,
    })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    dataset.to_netcdf(output_path)
    dataset.close()
    size_mb = os.path.getsize(output_path) / 1024**2
    print(f"  wrote {output_path} ({size_mb:.2f} MB)")


def checkpoint_path(output_dir: str) -> str:
    return os.path.join(output_dir, ".digest_state.pkl")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era5-root", required=True, nargs="+", metavar="DIR",
                        help="one or more directories holding the ERA5 files. "
                             "Both the flat layout ({year}_{month}_{var}.nc) "
                             "and the year-foldered one "
                             "({year}/{year}_{month}_{var}.nc) are recognised, "
                             "so archives split across disks can be given "
                             "together. Where two overlap on a year, the "
                             "earlier directory wins.")
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--output-dir", required=True,
                        help="destination for era5_lwa_climatology_<SEASON>.nc")
    parser.add_argument("--variables", nargs=3, default=["u", "v", "t"],
                        metavar=("U", "V", "T"))
    parser.add_argument("--kmax", type=int, default=None,
                        help="force the analysis-grid depth; default is the "
                             "deepest the data supports (49 for ERA5)")
    parser.add_argument("--work-dir", default=None,
                        help="scratch for per-season checkpoints (default: output-dir)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any existing checkpoint and start over")
    parser.add_argument("--stats-dir", default=None,
                        help="destination for the per-year statistics files "
                             "(default: output-dir). These are a local compute "
                             "cache, ~11 MB per year, NOT part of the shipped "
                             "digest -- keep them out of the obs_data tar.")
    parser.add_argument("--no-year-stats", action="store_true",
                        help="skip the per-year statistics files")
    parser.add_argument("--check-only", action="store_true",
                        help="run the preflight file check and exit")
    parser.add_argument("--months", nargs="+", type=int, default=None,
                        metavar="M",
                        help="months to read, default all twelve. A partial "
                             "year is for testing; a climatology built from "
                             "one is not meaningful.")
    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = args.work_dir or args.output_dir
    state_file = checkpoint_path(args.output_dir)

    # Preflight. A run spanning decades should not discover at hour 20 that one
    # year is short a month; the whole file list is cheap to check up front.
    months = list(range(1, 13)) if args.months is None else args.months
    years = list(range(args.start_year, args.end_year + 1))
    missing = []
    for year in years:
        for month in months:
            for variable in args.variables:
                if fawd.resolve_era5_path(args.era5_root, year, month,
                                          variable) is None:
                    missing.append(f"{year}_{month:02d}_{variable}.nc")
    if missing:
        print(f"PREFLIGHT FAILED: {len(missing)} of "
              f"{len(years) * len(months) * len(args.variables)} files not found "
              f"under {args.era5_root}")
        for name in missing[:20]:
            print(f"    {name}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
        return 1
    print(f"Preflight OK: {len(years) * len(months) * len(args.variables)} files "
          f"found for {years[0]}-{years[-1]}")
    if args.check_only:
        return 0

    accumulators: Dict[str, SeasonAccumulator] = {}
    done_years: set = set()
    if os.path.isfile(state_file) and not args.no_resume:
        with open(state_file, "rb") as handle:
            accumulators, done_years = pickle.load(handle)
        print(f"Resuming: {len(done_years)} year(s) already accumulated "
              f"({min(done_years)}-{max(done_years)})")
    else:
        accumulators = {season: SeasonAccumulator(season)
                        for season, _ in fawd.SEASON_TO_MONTHS}

    import falwa
    provenance = {
        "source": "ECMWF ERA5 reanalysis, pressure levels",
        "falwa_version": getattr(falwa, "__version__", "unknown"),
        "produced_by": "make_era5_digest.py",
        "era5_root": " ".join(args.era5_root),
    }

    started = time.time()
    for year in range(args.start_year, args.end_year + 1):
        if year in done_years:
            print(f"{year}: already accumulated, skipping")
            continue

        year_started = time.time()
        print(f"\n=== {year} ===", flush=True)
        ctx = fawd.load_obs_case(
            args.era5_root, year, wk_dir=work_dir,
            variables=args.variables, months=args.months,
            kmax_override=args.kmax)

        year_statistics: Dict[str, Dict[str, np.ndarray]] = {}
        coords: Dict[str, np.ndarray] = {}
        for season, months in fawd.SEASON_TO_MONTHS:
            result = fawd.process_season(
                ctx, season, months,
                checkpoint_every=0,          # this script does its own
                progress_every=0)
            if result.skipped:
                print(f"  {season}: no data, skipped")
                continue
            accumulator = accumulators[season]
            if accumulator.lat is None:
                accumulator.lat = ctx.original_grid[ctx.lat_name].values
                accumulator.lon = ctx.original_grid[ctx.lon_name].values
            accumulator.add_year(year, result)
            year_statistics[season] = compute_year_statistics(result)
            coords = {"height": np.asarray(result.analysis_height_array),
                      "lat": ctx.original_grid[ctx.lat_name].values,
                      "lon": ctx.original_grid[ctx.lon_name].values}
            print(f"  {season}: {result.n_time} timesteps "
                  f"(pooled {accumulator.n_timesteps})", flush=True)
            del result

        if year_statistics and not args.no_year_stats:
            stats_path = os.path.join(args.stats_dir or args.output_dir,
                                      f"era5_stats_{year}.nc")
            write_year_statistics(year, year_statistics, coords,
                                  stats_path, provenance)
            print(f"  per-year statistics -> {os.path.basename(stats_path)} "
                  f"({os.path.getsize(stats_path) / 1024**2:.1f} MB)")
        del year_statistics

        ctx.model_dataset.close()
        done_years.add(year)
        with open(state_file, "wb") as handle:
            pickle.dump((accumulators, done_years), handle)
        print(f"{year}: done in {time.time() - year_started:.0f}s "
              f"(checkpointed)", flush=True)

    print(f"\n=== accumulation finished in "
          f"{(time.time() - started) / 3600:.1f} h ===")

    for season, accumulator in accumulators.items():
        if accumulator.n_timesteps == 0:
            print(f"{season}: nothing accumulated, skipping")
            continue
        print(f"{season}: {len(accumulator.years)} years, "
              f"{accumulator.n_timesteps} timesteps, kmax={accumulator.kmax}")
        fields = accumulator.finalize()
        write_season(
            accumulator, fields,
            os.path.join(args.output_dir,
                         f"era5_lwa_climatology_{season}.nc"),
            args, provenance)

    print("\nDone. Remember the README.txt and licence file before packaging.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
