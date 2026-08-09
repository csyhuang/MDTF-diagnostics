# Finite-amplitude Rossby wave POD -- streaming implementation
# ================================================================================
# Computes the same diagnostics as finite_amplitude_wave_diag_zonal_mean.py but
# one timestep at a time, so that neither memory nor scratch disk scales with
# the length of the season.
#
# Why:
#
#   * falwa's QGDataset holds the input twice -- once as the merged xarray
#     Dataset it keeps a reference to, and again as one QGField per timestep,
#     each with its own copy of u/v/t. Each QGField then accumulates
#     interpolated u/v/theta, qgpv, avort and lwa on top, roughly nine 3-D
#     fields, about 190 MiB at 42x181x360. A 720-step season is >130 GiB.
#
#   * The bigger cost was upstream: DataPreprocessor wrote the whole gridfilled,
#     interpolated season to disk twice, as gridfill_{U,V,T}.nc and again as
#     intermediate_<SEASON>.nc. Measured on a 4-timestep slice and scaled, that
#     is ~88 GiB of scratch per 720-step season.
#
# Here each timestep is read, gridfilled, interpolated, passed through QGField,
# reduced to the five diagnostics that are actually kept, and discarded. Those
# five total 1.17 MiB per timestep -- 0.82 GiB for a whole 720-step season --
# so they are accumulated in memory and written once at the end.
#
# Scratch files are therefore not needed to bound memory. They are still
# written, periodically, for a different reason: a multi-thousand-timestep run
# that dies at step 2000 should not start again from zero. Checkpoints go to
# $WORK_DIR/model/netCDF as the MDTF guidelines require (doc/sphinx/
# dev_guidelines.rst:100).
#
# ================================================================================
# THIS FILE IS A MODULE, NOT THE DRIVER.
#
# The POD's driver is finite_amplitude_wave_diag_zonal_mean.ipynb, which imports
# from here and displays the figures inline. Everything below is written as
# functions taking an explicit CaseContext rather than reading module globals,
# so that the notebook, the test harness and `python -m` all drive the same code
# rather than three copies of it.
#
#   python finite_amplitude_wave_diag_zonal_mean_rework.py   # run standalone
#
# from the notebook:
#   ctx = load_case()
#   season = process_season(ctx, "DJF", [1, 2, 12])
#   plot_season(ctx, season, plot_dir)
# ================================================================================
#   - PI: Clare S. Y. Huang. The University of Chicago. csyhuang@uchicago.edu.
#   - Other contributors: Christopher Polster (JGU Mainz), Noboru Nakamura (UChicago)
#   The MDTF framework is distributed under the LGPLv3 license (see LICENSE.txt).
# ================================================================================
from __future__ import annotations

import gc
import os
from collections import namedtuple
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr
import yaml
from falwa.oopinterface import QGFieldNH18

from finite_amplitude_wave_diag_utils import gridfill_each_level, infer_vertical_grid, \
    normalize_orientation, drop_leap_day, save_seasonal_diagnostics, \
    LatLonMapPlotter, HeightLatPlotter

#: Must match the frequency requested in settings.jsonc.
FREQUENCY = "6hr"

#: Pseudoheight spacing requested when the input is NOT already on an evenly
#: spaced pseudoheight grid. Ignored when it is: falwa then takes dz from the
#: data. See infer_vertical_grid.
TARGET_DZ = 1000.0

#: Write a checkpoint every this many timesteps. Sized by how much recomputation
#: is acceptable after a crash, not by memory -- the accumulated diagnostics are
#: ~1.2 MiB per timestep. 0 disables checkpointing.
CHECKPOINT_EVERY = int(os.environ.get("FAWD_CHECKPOINT_EVERY", "100"))

#: Seasons and the calendar months belonging to each.
SEASON_TO_MONTHS = [
    ("DJF", [1, 2, 12]), ("MAM", [3, 4, 5]),
    ("JJA", [6, 7, 8]), ("SON", [9, 10, 11])]

#: Regular analysis grid defined by the developer. falwa works on this grid;
#: results are mapped back onto the input grid before averaging.
XLON = np.arange(0, 360, 1.0)
YLAT = np.arange(-90, 91, 1.0)

#: The five diagnostics retained per timestep.
TimestepResult = namedtuple(
    "TimestepResult",
    ["uref", "zonal_mean_u", "zonal_mean_lwa", "lwa_baro", "u_baro"])

SeasonalAverage = namedtuple(
    "SeasonalAverage", [
        "zonal_mean_u", "uref", "zonal_mean_lwa",
        "lwa_baro", "u_baro", "covariance_lwa_u_baro"])


@dataclass
class CaseContext:
    """Everything resolved from the framework hand-off, passed explicitly.

    Held in one object rather than as module globals so that the notebook can
    build it, inspect it, and hand it to the same functions the standalone run
    uses.
    """
    wk_dir: str
    casename: str
    catalog_file: str
    model_dataset: xr.Dataset
    u_var_name: str
    v_var_name: str
    t_var_name: str
    time_coord_name: str
    plev_name: str
    lat_name: str
    lon_name: str
    original_grid: Dict[str, xr.DataArray]
    firstyr: int
    lastyr: int
    catalog: object = None       # the intake datastore, handy in a notebook

    #: "model" or "obs". Selects the output subdirectory, both of which the
    #: framework creates under POD_WORK_DIR. Everything else in the pipeline is
    #: source-agnostic: process_season and plot_season never inspect this.
    model_or_obs: str = "model"

    #: Override the kmax that infer_vertical_grid would derive. Left None the
    #: analysis grid reaches as high as the data supports -- 42 levels for this
    #: model case, 49 for ERA5. See the Cautions section of the POD docs: that
    #: asymmetry is deliberate but it is not a like-for-like comparison.
    kmax_override: Optional[int] = None

    @property
    def plot_dir(self) -> str:
        return os.path.join(self.wk_dir, self.model_or_obs, "PS") + os.sep

    @property
    def netcdf_dir(self) -> str:
        return os.path.join(self.wk_dir, self.model_or_obs, "netCDF")

    def title_for(self, season: str) -> str:
        return f"{self.casename} ({self.firstyr}-{self.lastyr}) {season}"


@dataclass
class SeasonResult:
    """One season's per-timestep diagnostics and their average."""
    season: str
    n_time: int
    results: List[TimestepResult]
    seasonal_average: Optional[SeasonalAverage]
    yz_mask: Optional[np.ndarray]
    xy_mask: Optional[np.ndarray]
    analysis_height_array: Optional[np.ndarray]
    dz: float
    kmax: int
    on_even_grid: bool
    skipped: bool = False


# ================================================================================
# 1) Framework hand-off
# ================================================================================

def load_case(wk_dir: Optional[str] = None,
              case_env_file: Optional[str] = None,
              frequency: str = FREQUENCY,
              drop_feb29: bool = True) -> CaseContext:
    """Resolve the framework hand-off into a CaseContext.

    The framework passes the POD its inputs through case_info.yml, whose
    location is in the `case_env_file` environment variable. That file holds the
    postprocessed data catalog and, per case, the model's own names for each
    variable and coordinate. See doc/sphinx/ref_envvars.rst and
    case_info_example.yml.
    """
    wk_dir = wk_dir or os.environ["WORK_DIR"]
    case_env_file = case_env_file or os.environ["case_env_file"]
    if not os.path.isfile(case_env_file):
        raise FileNotFoundError(f"case environment file not found: {case_env_file}")

    with open(case_env_file, "r") as stream:
        case_info = yaml.safe_load(stream)

    catalog_file = case_info["CATALOG_FILE"]
    case_list = case_info["CASE_LIST"]
    casename = list(case_list.keys())[0]
    case_attrs = case_list[casename]
    if len(case_list) > 1:
        print(f"WARNING: {len(case_list)} cases supplied; this POD analyses one. "
              f"Using {casename}.")

    u_var_name = case_attrs.get("ua_var", "ua")
    v_var_name = case_attrs.get("va_var", "va")
    t_var_name = case_attrs.get("ta_var", "ta")
    time_coord_name = case_attrs.get("time_coord", "time")
    plev_name = case_attrs.get("plev_coord", "plev")
    lat_name = case_attrs.get("lat_coord", "lat")
    lon_name = case_attrs.get("lon_coord", "lon")

    # Imported here rather than at module scope: only the model path needs a
    # data catalog, and the ERA5 digest job should not have to install
    # intake-esm to read files it locates itself.
    import intake

    catalog = intake.open_esm_datastore(catalog_file)
    subset = catalog.search(
        variable_id=[u_var_name, v_var_name, t_var_name], frequency=frequency)
    if subset.df.empty:
        raise ValueError(
            f"No assets in {catalog_file} for variables "
            f"{[u_var_name, v_var_name, t_var_name]} at frequency '{frequency}'. "
            f"Available variable_id: {sorted(catalog.df['variable_id'].unique())}")

    dataset_dict = subset.to_dataset_dict(
        progressbar=False,
        xarray_open_kwargs={"decode_times": True, "use_cftime": True})
    model_dataset = dataset_dict[list(dataset_dict)[0]]

    missing = [v for v in (u_var_name, v_var_name, t_var_name)
               if v not in model_dataset]
    if missing:
        raise KeyError(f"{missing} absent from the catalog query result. "
                       f"Found {list(model_dataset.data_vars)}.")

    firstyr = model_dataset.coords[time_coord_name].values[0].year
    lastyr = model_dataset.coords[time_coord_name].values[-1].year

    if model_dataset[plev_name].units == "Pa":
        # True division, not //. Floor division silently truncates any level
        # that is not a whole number of hPa, and on a grid evenly spaced in
        # pseudoheight that is most of them.
        print("plev is in Pa; converting to hPa.")
        model_dataset = model_dataset.assign_coords(
            {plev_name: model_dataset[plev_name] / 100})
        model_dataset[plev_name].attrs["units"] = "hPa"

    # falwa requires latitude ascending and pressure descending. Done once,
    # here, so every downstream step sees a single orientation -- and so the
    # coordinate and the data can never be flipped independently of each other.
    model_dataset = normalize_orientation(model_dataset, lat_name, plev_name)

    # Equalise the calendars: the model is noleap, reanalysis is not. With
    # 29 February removed every year holds the same number of timesteps, so a
    # per-year mean and a pooled mean agree.
    if drop_feb29:
        model_dataset = drop_leap_day(model_dataset, time_coord_name)

    return CaseContext(
        wk_dir=wk_dir, casename=casename, catalog_file=catalog_file,
        model_dataset=model_dataset,
        u_var_name=u_var_name, v_var_name=v_var_name, t_var_name=t_var_name,
        time_coord_name=time_coord_name, plev_name=plev_name,
        lat_name=lat_name, lon_name=lon_name,
        original_grid={
            time_coord_name: model_dataset.coords[time_coord_name],
            plev_name: model_dataset.coords[plev_name],
            lat_name: model_dataset.coords[lat_name],
            lon_name: model_dataset.coords[lon_name]},
        firstyr=firstyr, lastyr=lastyr, catalog=catalog)


#: Layouts ERA5 is stored in, tried in this order under each root. Archives
#: accumulated over decades are rarely uniform -- one span may sit flat in a
#: directory while an older span is foldered by year -- and the caller should
#: not have to say which is which.
ERA5_PATH_PATTERNS = (
    "{year:04d}_{month:02d}_{variable}.nc",                    # flat
    os.path.join("{year:04d}",
                 "{year:04d}_{month:02d}_{variable}.nc"),      # year/ subdir
)


def resolve_era5_path(roots: Sequence[str], year: int, month: int,
                      variable: str) -> Optional[str]:
    """First existing file for (year, month, variable) across roots and layouts.

    Roots are searched in the order given, and within each root the layouts in
    ERA5_PATH_PATTERNS. Where two archives overlap -- a boundary year present
    in both -- the earlier root wins, so precedence is explicit rather than
    accidental.
    """
    for root in roots:
        for pattern in ERA5_PATH_PATTERNS:
            candidate = os.path.join(
                root, pattern.format(year=year, month=month, variable=variable))
            if os.path.isfile(candidate):
                return candidate
    return None

#: ERA5 dimension names (grib_to_netcdf vintage) -> the POD's names. Newer CDS
#: downloads use valid_time/pressure_level instead; both are handled.
ERA5_RENAME = {
    "longitude": "lon", "latitude": "lat",
    "level": "plev", "pressure_level": "plev",
    "valid_time": "time",
}


def load_obs_case(era5_root, year: int,
                  wk_dir: Optional[str] = None,
                  casename: str = "ERA5",
                  variables: Sequence[str] = ("u", "v", "t"),
                  months: Optional[Sequence[int]] = None,
                  drop_feb29: bool = True,
                  kmax_override: Optional[int] = None) -> CaseContext:
    """Load one year of ERA5 into a CaseContext.

    Returns the same object type as :func:`load_case`, so process_season,
    plot_season and save_diagnostics work on reanalysis unchanged -- the only
    thing that differs between model and observations is how the data is found.

    One year at a time, because the digest spans decades and a whole-record
    open_mfdataset over ~1000 files buys nothing when the work is per-season
    anyway.

    Args:
        era5_root: directory, or list of directories, holding the ERA5\n            files. Both the flat and year-foldered layouts are recognised;\n            see ERA5_PATH_PATTERNS. Earlier roots win where they overlap.
        year: calendar year to load
        wk_dir: output root; defaults to $WORK_DIR
        casename: label used in figure titles
        variables: ERA5 variable names for u, v, t in that order
        months: months to load, default all twelve. A partial year is legal --
            useful for a test slice, and for a record that starts or ends
            mid-year -- but every requested month must be present, since a
            silently short season would bias the climatology it feeds.
        drop_feb29: discard the leap day so every year weighs the same
        kmax_override: force a specific analysis-grid depth

    Returns:
        CaseContext with model_or_obs="obs".
    """
    wk_dir = wk_dir or os.environ.get("WORK_DIR", ".")
    u_name, v_name, t_name = variables

    months = list(range(1, 13)) if months is None else list(months)
    roots = [era5_root] if isinstance(era5_root, str) else list(era5_root)

    merged = []
    resolved_roots = set()
    for variable in variables:
        paths, missing = [], []
        for month in months:
            found = resolve_era5_path(roots, year, month, variable)
            if found is None:
                missing.append(f"{year}_{month:02d}_{variable}.nc")
            else:
                paths.append(found)
                resolved_roots.add(
                    next(r for r in roots if found.startswith(r)))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} ERA5 file(s) not found for {year} {variable} "
                f"under {roots}: {', '.join(missing[:4])}"
                f"{' ...' if len(missing) > 4 else ''}")
        # decode_times is needed for the season selection; the fields are
        # stored as packed shorts and xarray unpacks them via scale_factor /
        # add_offset on read.
        merged.append(xr.open_mfdataset(
            paths, combine="by_coords",
            decode_times=True, use_cftime=True))

    dataset = xr.merge(merged, join="exact")
    dataset = dataset.rename(
        {k: v for k, v in ERA5_RENAME.items() if k in dataset.dims
         or k in dataset.coords})

    for name, expected in ((u_name, "u"), (v_name, "v"), (t_name, "t")):
        if name not in dataset:
            raise KeyError(f"{expected} variable {name!r} not in the ERA5 files; "
                           f"found {list(dataset.data_vars)}")

    # ERA5 levels are millibars, i.e. hPa already; label them so that the
    # Pa-to-hPa conversion in the model path is not triggered by accident.
    dataset["plev"].attrs["units"] = "hPa"

    # falwa needs latitude ascending and pressure descending; ERA5 is stored
    # the other way round on both axes.
    dataset = normalize_orientation(dataset, "lat", "plev")
    if drop_feb29:
        dataset = drop_leap_day(dataset, "time")

    if len(resolved_roots) > 1:
        print(f"{year}: files drawn from {len(resolved_roots)} archives "
              f"{sorted(resolved_roots)}")

    return CaseContext(
        wk_dir=wk_dir, casename=casename,
        catalog_file=",".join(sorted(resolved_roots)),
        model_dataset=dataset,
        u_var_name=u_name, v_var_name=v_name, t_var_name=t_name,
        time_coord_name="time", plev_name="plev",
        lat_name="lat", lon_name="lon",
        original_grid={
            "time": dataset.coords["time"],
            "plev": dataset.coords["plev"],
            "lat": dataset.coords["lat"],
            "lon": dataset.coords["lon"]},
        firstyr=year, lastyr=year,
        model_or_obs="obs", kmax_override=kmax_override)


# ================================================================================
# 2) Per-timestep computation
# ================================================================================

def masks_for_timestep(u_slice: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Missing-value masks for one timestep, in the same sense as the original.

    Returns ``(yz, xy)`` where *yz* is (plev, lat) -- any longitude missing at
    that level and latitude -- and *xy* is (lat, lon) -- any level above the
    lowest missing at that column. OR-ing these across timesteps reproduces what
    DataPreprocessor._do_save_mask computes in one pass over the season.
    """
    missing = np.isnan(u_slice)                     # (plev, lat, lon)
    return missing.any(axis=-1), missing[1:, :, :].any(axis=0)


def prepare_timestep(ctx: CaseContext, ds_t: xr.Dataset):
    """Gridfill and interpolate one timestep onto the analysis grid.

    Returns (u, v, t) as (plev, ylat, xlon) arrays. Gridfill runs on the input
    grid, before interpolation, exactly as in the original -- filling afterwards
    would spread missing values first and then fill the smeared result.
    """
    filled = {
        name: xr.apply_ufunc(
            gridfill_each_level, ds_t[name],
            input_core_dims=((ctx.lat_name, ctx.lon_name),),
            output_core_dims=((ctx.lat_name, ctx.lon_name),),
            vectorize=True, dask="forbidden")
        for name in (ctx.u_var_name, ctx.v_var_name, ctx.t_var_name)}

    interpolated = xr.Dataset(filled).interp(
        coords={ctx.lat_name: YLAT, ctx.lon_name: XLON},
        method="linear", kwargs={"fill_value": "extrapolate"})

    return (interpolated[ctx.u_var_name].values,
            interpolated[ctx.v_var_name].values,
            interpolated[ctx.t_var_name].values)


def orient_for_qgfield(u, v, t, plev):
    """Match QGField's expectations: plev descending, ylat ascending.

    QGDataset does this internally; doing it here keeps the two paths
    equivalent. YLAT is ascending by construction, so only plev can need
    flipping.
    """
    if plev[0] < plev[-1]:
        plev = plev[::-1]
        u, v, t = u[::-1], v[::-1], t[::-1]
    return u, v, t, plev


def compute_one_timestep(u, v, t, plev, dz, kmax, on_even_grid) -> TimestepResult:
    """Run falwa on a single timestep and keep only the five diagnostics.

    The QGField object -- and the ~190 MiB of 3-D fields it accumulates -- is
    released before returning, so peak memory is one timestep regardless of how
    long the season is.
    """
    qgfield = QGFieldNH18(
        XLON, YLAT, plev, u, v, t,
        dz=dz, kmax=kmax,
        data_on_evenly_spaced_pseudoheight_grid=on_even_grid)
    qgfield.interpolate_fields(return_named_tuple=False)
    qgfield.compute_reference_states(return_named_tuple=False)
    qgfield.compute_lwa_and_barotropic_fluxes(return_named_tuple=False)

    result = TimestepResult(
        uref=np.array(qgfield.uref),                             # (kmax, nlat)
        zonal_mean_u=np.array(qgfield.interpolated_u).mean(axis=-1),
        zonal_mean_lwa=np.array(qgfield.lwa).mean(axis=-1),
        lwa_baro=np.array(qgfield.lwa_baro),                     # (nlat, nlon)
        u_baro=np.array(qgfield.u_baro))
    del qgfield
    return result


# ================================================================================
# 3) Post-processing
# ================================================================================

def interp_to_original_grid(ctx: CaseContext, field: np.ndarray,
                            dims: str) -> np.ndarray:
    """Map a single field off the analysis grid back onto the input grid."""
    if dims == "yz":                 # (kmax, ylat) -> (kmax, lat)
        da = xr.DataArray(field, dims=("height", "ylat"), coords={"ylat": YLAT})
        return da.interp(ylat=ctx.original_grid[ctx.lat_name].values).values
    if dims == "xy":                 # (ylat, xlon) -> (lat, lon)
        da = xr.DataArray(field, dims=("ylat", "xlon"),
                          coords={"ylat": YLAT, "xlon": XLON})
        return da.interp(ylat=ctx.original_grid[ctx.lat_name].values,
                         xlon=ctx.original_grid[ctx.lon_name].values).values
    raise ValueError(dims)


def result_to_original_grid(ctx: CaseContext,
                            result: TimestepResult) -> TimestepResult:
    """Map one timestep's diagnostics onto the input grid.

    Done per timestep, before any time averaging, because the covariance is not
    linear: interpolating the covariance is not the same as the covariance of
    the interpolated fields, and the original driver does the latter. For the
    linear quantities the order is immaterial, so matching here keeps the two
    implementations comparable across every diagnostic rather than five of six.
    """
    return TimestepResult(
        uref=interp_to_original_grid(ctx, result.uref, "yz"),
        zonal_mean_u=interp_to_original_grid(ctx, result.zonal_mean_u, "yz"),
        zonal_mean_lwa=interp_to_original_grid(ctx, result.zonal_mean_lwa, "yz"),
        lwa_baro=interp_to_original_grid(ctx, result.lwa_baro, "xy"),
        u_baro=interp_to_original_grid(ctx, result.u_baro, "xy"))


def calculate_covariance(lwa_baro: np.ndarray, u_baro: np.ndarray) -> np.ndarray:
    """Temporal covariance of LWA and U at each grid point, Bessel-corrected."""
    n_time = lwa_baro.shape[0]
    if n_time < 2:
        raise ValueError(f"covariance needs at least 2 timesteps, got {n_time}")
    a_anomaly = lwa_baro - lwa_baro.mean(axis=0)
    b_anomaly = u_baro - u_baro.mean(axis=0)
    return (a_anomaly * b_anomaly).sum(axis=0) / (n_time - 1)


def time_average_processing(results: Sequence[TimestepResult]) -> SeasonalAverage:
    stacked = {f: np.stack([getattr(r, f) for r in results], axis=0)
               for f in TimestepResult._fields}
    return SeasonalAverage(
        zonal_mean_u=stacked["zonal_mean_u"].mean(axis=0),
        uref=stacked["uref"].mean(axis=0),
        zonal_mean_lwa=stacked["zonal_mean_lwa"].mean(axis=0),
        lwa_baro=stacked["lwa_baro"].mean(axis=0),
        u_baro=stacked["u_baro"].mean(axis=0),
        covariance_lwa_u_baro=calculate_covariance(
            stacked["lwa_baro"], stacked["u_baro"]))


# ================================================================================
# 4) Checkpointing
# ================================================================================

def checkpoint_path(ctx: CaseContext, season: str) -> str:
    return os.path.join(ctx.netcdf_dir, f"checkpoint_{season}.nc")


def save_checkpoint(ctx: CaseContext, season: str,
                    results: Sequence[TimestepResult],
                    yz_mask: np.ndarray, xy_mask: np.ndarray,
                    checkpoint_every: int = CHECKPOINT_EVERY) -> None:
    """Persist progress so a killed run resumes near where it stopped."""
    if checkpoint_every <= 0 or not results:
        return
    path = checkpoint_path(ctx, season)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stacked = {f: np.stack([getattr(r, f) for r in results], axis=0)
               for f in TimestepResult._fields}
    dataset = xr.Dataset(
        data_vars={
            "uref": (("step", "height", "lat"), stacked["uref"]),
            "zonal_mean_u": (("step", "height", "lat"), stacked["zonal_mean_u"]),
            "zonal_mean_lwa": (("step", "height", "lat"), stacked["zonal_mean_lwa"]),
            "lwa_baro": (("step", "lat", "lon"), stacked["lwa_baro"]),
            "u_baro": (("step", "lat", "lon"), stacked["u_baro"]),
            "yz_mask": (("plev", "lat"), yz_mask.astype("i1")),
            "xy_mask": (("lat", "lon"), xy_mask.astype("i1"))},
        attrs={"n_done": len(results), "season": season,
               "casename": ctx.casename})
    tmp = path + ".tmp"
    dataset.to_netcdf(tmp)
    dataset.close()
    os.replace(tmp, path)   # atomic: a killed write never leaves a half file


def load_checkpoint(ctx: CaseContext, season: str,
                    checkpoint_every: int = CHECKPOINT_EVERY):
    """Return (results, yz_mask, xy_mask) from a checkpoint, or None."""
    path = checkpoint_path(ctx, season)
    if checkpoint_every <= 0 or not os.path.isfile(path):
        return None
    try:
        with xr.open_dataset(path) as ds:
            n = int(ds.attrs["n_done"])
            results = [
                TimestepResult(
                    uref=ds["uref"].values[i],
                    zonal_mean_u=ds["zonal_mean_u"].values[i],
                    zonal_mean_lwa=ds["zonal_mean_lwa"].values[i],
                    lwa_baro=ds["lwa_baro"].values[i],
                    u_baro=ds["u_baro"].values[i])
                for i in range(n)]
            yz = ds["yz_mask"].values.astype(bool)
            xy = ds["xy_mask"].values.astype(bool)
        print(f"{season}: resuming from checkpoint at timestep {n}")
        return results, yz, xy
    except Exception as exc:      # a corrupt checkpoint must not be fatal
        print(f"{season}: checkpoint unreadable ({exc}); starting from scratch")
        return None


# ================================================================================
# 5) Season driver
# ================================================================================

def process_season(ctx: CaseContext, season: str, months: Sequence[int],
                   checkpoint_every: int = CHECKPOINT_EVERY,
                   progress: Optional[Callable[[str], None]] = print,
                   progress_every: int = 20) -> SeasonResult:
    """Compute every timestep of one season and return its diagnostics.

    Seasons with fewer than two timesteps are returned with ``skipped=True``:
    partial-year input is legitimate, and the covariance needs at least two.
    """
    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    season_dataset = ctx.model_dataset.where(
        ctx.model_dataset[ctx.time_coord_name].dt.month.isin(months), drop=True)
    n_time = season_dataset[ctx.time_coord_name].size
    say(f"{season}: {n_time} timesteps selected")

    plev_hpa = season_dataset[ctx.plev_name].values
    on_even_grid, dz, kmax = infer_vertical_grid(plev_hpa, default_dz=TARGET_DZ)
    if ctx.kmax_override is not None and ctx.kmax_override != kmax:
        say(f"{season}: kmax {kmax} -> {ctx.kmax_override} (override); "
            f"falwa will interpolate rather than pass through")
        kmax, on_even_grid = ctx.kmax_override, False

    if n_time < 2:
        say(f"WARNING: {season} has {n_time} timestep(s); skipping this season.")
        season_dataset.close()
        return SeasonResult(season, n_time, [], None, None, None, None,
                            dz, kmax, on_even_grid, skipped=True)

    say(f"{season}: vertical grid "
        f"{'already evenly spaced in pseudoheight, falwa interpolation SKIPPED' if on_even_grid else 'needs interpolation'}"
        f"; dz = {dz:.1f} m, kmax = {kmax}")

    resumed = load_checkpoint(ctx, season, checkpoint_every)
    if resumed is None:
        results, yz_mask, xy_mask = [], None, None
    else:
        results, yz_mask, xy_mask = resumed

    for step in range(len(results), n_time):
        ds_t = season_dataset.isel({ctx.time_coord_name: step})[
            [ctx.u_var_name, ctx.v_var_name, ctx.t_var_name]].load()

        yz_t, xy_t = masks_for_timestep(ds_t[ctx.u_var_name].values)
        yz_mask = yz_t if yz_mask is None else (yz_mask | yz_t)
        xy_mask = xy_t if xy_mask is None else (xy_mask | xy_t)

        u, v, t = prepare_timestep(ctx, ds_t)
        ds_t.close()
        u, v, t, plev_oriented = orient_for_qgfield(u, v, t, plev_hpa)

        results.append(result_to_original_grid(ctx, compute_one_timestep(
            u, v, t, plev_oriented, dz, kmax, on_even_grid)))
        del u, v, t

        if progress_every and ((step + 1) % progress_every == 0 or step + 1 == n_time):
            say(f"    {season}: {step + 1}/{n_time} timesteps")
        if checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            save_checkpoint(ctx, season, results, yz_mask, xy_mask, checkpoint_every)
            gc.collect()

    save_checkpoint(ctx, season, results, yz_mask, xy_mask, checkpoint_every)
    season_dataset.close()

    return SeasonResult(
        season=season, n_time=n_time, results=results,
        seasonal_average=time_average_processing(results),
        yz_mask=yz_mask, xy_mask=xy_mask,
        analysis_height_array=np.arange(kmax) * dz,
        dz=dz, kmax=kmax, on_even_grid=on_even_grid)


# ================================================================================
# 6) Plotting
# ================================================================================

def plot_season(ctx: CaseContext, season_result: SeasonResult,
                plot_dir: Optional[str] = None,
                save: bool = True) -> Dict[str, object]:
    """Draw the seven figures for one season.

    Returns a dict of name -> matplotlib Figure so a notebook can display them
    inline; when *save* is true they are also written as EPS for the framework
    to convert, which is what the generated webpage links to.
    """
    if season_result.skipped:
        return {}
    plot_dir = plot_dir if plot_dir is not None else ctx.plot_dir
    if save:
        os.makedirs(plot_dir, exist_ok=True)

    average = season_result.seasonal_average
    season = season_result.season
    title_str = ctx.title_for(season)
    xy_mask = season_result.xy_mask
    yland, xland = (np.where(xy_mask) if xy_mask is not None else ([], []))
    cmap = "jet"
    figures: Dict[str, object] = {}

    height_lat_plotter = HeightLatPlotter(
        figsize=(4, 4), title_str=title_str,
        xgrid=ctx.original_grid[ctx.lat_name],
        ygrid=season_result.analysis_height_array, cmap=cmap, xlim=[-80, 80])
    for name, field, label in (
            ("zonal_mean_u", average.zonal_mean_u, "zonal mean U"),
            ("zonal_mean_lwa", average.zonal_mean_lwa, "zonal mean LWA"),
            ("zonal_mean_uref", average.uref, "zonal mean Uref"),
            ("zonal_mean_delta_u", average.zonal_mean_u - average.uref,
             r"zonal mean $\Delta$ U")):
        figures[name] = height_lat_plotter.plot_and_save_variable(
            variable=field, cmap=cmap, var_title_str=label,
            save_path=f"{plot_dir}{season}_{name}.eps" if save else None,
            num_level=30)

    lat_lon_plotter = LatLonMapPlotter(
        figsize=(6, 3), title_str=title_str,
        xgrid=ctx.original_grid[ctx.lon_name],
        ygrid=ctx.original_grid[ctx.lat_name], cmap=cmap,
        xland=xland, yland=yland,
        lon_range=np.arange(-180, 181, 60), lat_range=np.arange(-90, 91, 30))
    for name, field, label, this_cmap in (
            ("u_baro", average.u_baro, "U baro", cmap),
            ("lwa_baro", average.lwa_baro, "LWA baro", cmap),
            ("u_lwa_covariance", average.covariance_lwa_u_baro,
             "Covariance between LWA and U(baro)", "Purples_r")):
        figures[name] = lat_lon_plotter.plot_and_save_variable(
            variable=field, cmap=this_cmap, var_title_str=label,
            save_path=f"{plot_dir}{season}_{name}.eps" if save else None,
            num_level=30)

    return figures


def save_diagnostics(ctx: CaseContext, season_result: SeasonResult,
                     output_path: Optional[str] = None) -> Optional[str]:
    """Write one season's mean diagnostics to netCDF."""
    if season_result.skipped:
        return None
    output_path = output_path or os.path.join(
        ctx.netcdf_dir, f"diagnostics_{season_result.season}.nc")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_seasonal_diagnostics(
        seasonal_average_data=season_result.seasonal_average,
        analysis_height_array=season_result.analysis_height_array,
        lat_coord=ctx.original_grid[ctx.lat_name],
        lon_coord=ctx.original_grid[ctx.lon_name],
        output_path=output_path)
    return output_path


# ================================================================================
# 7) Standalone entry point
# ================================================================================

def main() -> int:
    import matplotlib
    matplotlib.use("Agg")   # only when running headless; a notebook sets its own

    ctx = load_case()
    print(f"""
    wk_dir   = {ctx.wk_dir}
    casename = {ctx.casename}
    variables = {ctx.u_var_name}, {ctx.v_var_name}, {ctx.t_var_name}
    years    = {ctx.firstyr}-{ctx.lastyr}
    checkpoint every = {CHECKPOINT_EVERY} timestep(s)
    """)

    for season, months in SEASON_TO_MONTHS:
        print(f"\nseason: {season}")
        season_result = process_season(ctx, season, months)
        if season_result.skipped:
            continue
        save_diagnostics(ctx, season_result)
        plot_season(ctx, season_result)
        print(f"{season}: figures written to {ctx.plot_dir}")
        del season_result
        gc.collect()

    ctx.model_dataset.close()
    print("POD Finite-amplitude wave diagnostic (zonal mean, streaming) "
          "finished successfully!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
