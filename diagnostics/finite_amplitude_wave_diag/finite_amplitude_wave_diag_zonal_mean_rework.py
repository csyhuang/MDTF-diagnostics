# Finite-amplitude Rossby wave POD -- streaming rework
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
# This file exists alongside the original so the two can be compared on the
# same input. See compare_implementations.py.
# ================================================================================
#   - PI: Clare S. Y. Huang. The University of Chicago. csyhuang@uchicago.edu.
#   - Other contributors: Christopher Polster (JGU Mainz), Noboru Nakamura (UChicago)
#   The MDTF framework is distributed under the LGPLv3 license (see LICENSE.txt).
# ================================================================================
import gc
import os
from collections import namedtuple
from typing import Dict, List, Optional

import matplotlib
import intake
import numpy as np
import xarray as xr
import yaml
from falwa.oopinterface import QGFieldNH18

from finite_amplitude_wave_diag_utils import gridfill_each_level, infer_vertical_grid, \
    save_seasonal_diagnostics, LatLonMapPlotter, HeightLatPlotter

matplotlib.use('Agg')  # non-X windows backend; the framework always runs headless

frequency = "6hr"  # must match the frequency requested in settings.jsonc

# Pseudoheight spacing requested when the input is NOT already on an evenly
# spaced pseudoheight grid. Ignored when it is: falwa then takes dz from the
# data. See infer_vertical_grid.
TARGET_DZ = 1000.0

# Write a checkpoint every this many timesteps. Sized by how much recomputation
# is acceptable after a crash, not by memory -- the accumulated diagnostics are
# ~1.2 MiB per timestep. Set to 0 to disable checkpointing entirely.
CHECKPOINT_EVERY = int(os.environ.get("FAWD_CHECKPOINT_EVERY", "100"))

# *** Regular analysis grid defined by developer ***
xlon = np.arange(0, 360, 1.0)
ylat = np.arange(-90, 91, 1.0)

# ================================================================================
# 1) Framework hand-off. Identical to the original driver; see its comments and
#    doc/sphinx/ref_envvars.rst.
# ================================================================================
wk_dir = os.environ["WORK_DIR"]
case_env_file = os.environ["case_env_file"]
assert os.path.isfile(case_env_file), f"case environment file not found: {case_env_file}"
with open(case_env_file, 'r') as stream:
    case_info = yaml.safe_load(stream)

cat_def_file = case_info['CATALOG_FILE']
case_list = case_info['CASE_LIST']
casename = list(case_list.keys())[0]
case_attrs = case_list[casename]
if len(case_list) > 1:
    print(f"WARNING: {len(case_list)} cases supplied; this POD analyses one. Using {casename}.")

u_var_name = case_attrs.get('ua_var', 'ua')
v_var_name = case_attrs.get('va_var', 'va')
t_var_name = case_attrs.get('ta_var', 'ta')
time_coord_name = case_attrs.get('time_coord', 'time')
plev_name = case_attrs.get('plev_coord', 'plev')
lat_name = case_attrs.get('lat_coord', 'lat')
lon_name = case_attrs.get('lon_coord', 'lon')

cat = intake.open_esm_datastore(cat_def_file)
cat_subset = cat.search(
    variable_id=[u_var_name, v_var_name, t_var_name], frequency=frequency)
if cat_subset.df.empty:
    raise ValueError(
        f"No assets in {cat_def_file} for variables "
        f"{[u_var_name, v_var_name, t_var_name]} at frequency '{frequency}'. "
        f"Available variable_id: {sorted(cat.df['variable_id'].unique())}")

dataset_dict = cat_subset.to_dataset_dict(
    progressbar=False,
    xarray_open_kwargs={"decode_times": True, "use_cftime": True})
model_dataset = dataset_dict[list(dataset_dict)[0]]

missing_vars = [v for v in (u_var_name, v_var_name, t_var_name) if v not in model_dataset]
if missing_vars:
    raise KeyError(f"{missing_vars} absent from the catalog query result. "
                   f"Found {list(model_dataset.data_vars)}.")

firstyr = model_dataset.coords[time_coord_name].values[0].year
lastyr = model_dataset.coords[time_coord_name].values[-1].year
if model_dataset[plev_name].units == 'Pa':
    print("model_dataset[plev_name].units == 'Pa'. Convert it to hPa.")
    model_dataset = model_dataset.assign_coords({plev_name: model_dataset[plev_name] / 100})
    model_dataset[plev_name].attrs["units"] = 'hPa'

original_grid = {
    time_coord_name: model_dataset.coords[time_coord_name],
    plev_name: model_dataset.coords[plev_name],
    lat_name: model_dataset.coords[lat_name],
    lon_name: model_dataset.coords[lon_name]}

print(f"""
    wk_dir = {wk_dir}
    casename = {casename}
    variables = {u_var_name}, {v_var_name}, {t_var_name}
    firstyr, lastyr = {firstyr}, {lastyr}
    checkpoint every = {CHECKPOINT_EVERY} timestep(s)
    """)


# ================================================================================
# 2) Per-timestep computation
# ================================================================================

#: The five diagnostics retained per timestep, and their dimensions once the
#: analysis grid is mapped back onto the input grid.
TimestepResult = namedtuple(
    "TimestepResult",
    ["uref", "zonal_mean_u", "zonal_mean_lwa", "lwa_baro", "u_baro"])


def masks_for_timestep(u_slice: np.ndarray):
    """Missing-value masks for one timestep, in the same sense as the original.

    Returns ``(yz, xy)`` where *yz* is (plev, lat) -- any longitude missing at
    that level and latitude -- and *xy* is (lat, lon) -- any level above the
    lowest missing at that column. OR-ing these across timesteps reproduces
    what DataPreprocessor._do_save_mask computes in one pass over the season.
    """
    missing = np.isnan(u_slice)                     # (plev, lat, lon)
    yz = missing.any(axis=-1)                       # (plev, lat)
    xy = missing[1:, :, :].any(axis=0)              # (lat, lon), skipping level 0
    return yz, xy


def prepare_timestep(ds_t: xr.Dataset):
    """Gridfill and interpolate one timestep onto the analysis grid.

    Returns (u, v, t) as (plev, ylat, xlon) numpy arrays. Gridfill runs on the
    input grid, before interpolation, exactly as in the original -- filling
    after interpolation would spread missing values first and then fill the
    smeared result.
    """
    filled = {}
    for name in (u_var_name, v_var_name, t_var_name):
        filled[name] = xr.apply_ufunc(
            gridfill_each_level,
            ds_t[name],
            input_core_dims=((lat_name, lon_name),),
            output_core_dims=((lat_name, lon_name),),
            vectorize=True, dask="forbidden")

    interpolated = xr.Dataset(filled).interp(
        coords={lat_name: ylat, lon_name: xlon},
        method="linear",
        kwargs={"fill_value": "extrapolate"})

    return (interpolated[u_var_name].values,
            interpolated[v_var_name].values,
            interpolated[t_var_name].values)


def compute_one_timestep(u, v, t, plev, dz, kmax, on_even_grid) -> TimestepResult:
    """Run falwa on a single timestep and keep only the five diagnostics.

    The QGField object -- and the ~190 MiB of 3-D fields it accumulates -- is
    released before returning, so peak memory is one timestep regardless of how
    long the season is.
    """
    qgfield = QGFieldNH18(
        xlon, ylat, plev, u, v, t,
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


def orient_for_qgfield(u, v, t, plev):
    """Match QGField's expectations: plev descending, ylat ascending.

    QGDataset does this internally; doing it here keeps the two paths
    equivalent. ylat is the developer-defined analysis grid and is ascending by
    construction, so only plev can need flipping.
    """
    if plev[0] < plev[-1]:      # ascending pressure -> flip to descending
        plev = plev[::-1]
        u, v, t = u[::-1], v[::-1], t[::-1]
    return u, v, t, plev


# ================================================================================
# 3) Post-processing, identical in intent to the original driver
# ================================================================================

def calculate_covariance(lwa_baro: np.ndarray, u_baro: np.ndarray) -> np.ndarray:
    """Temporal covariance of LWA and U at each grid point, Bessel-corrected."""
    n_time = lwa_baro.shape[0]
    if n_time < 2:
        raise ValueError(f"covariance needs at least 2 timesteps, got {n_time}")
    a_anomaly = lwa_baro - lwa_baro.mean(axis=0)
    b_anomaly = u_baro - u_baro.mean(axis=0)
    return (a_anomaly * b_anomaly).sum(axis=0) / (n_time - 1)


SeasonalAverage = namedtuple(
    "SeasonalAverage", [
        "zonal_mean_u", "uref", "zonal_mean_lwa",
        "lwa_baro", "u_baro", "covariance_lwa_u_baro"])


def time_average_processing(results: List[TimestepResult]) -> SeasonalAverage:
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


def result_to_original_grid(result: TimestepResult) -> TimestepResult:
    """Map one timestep's diagnostics off the analysis grid onto the input grid.

    Done per timestep, before any time averaging, because the covariance is not
    linear: interpolating the covariance is not the same as the covariance of
    the interpolated fields, and the original driver does the latter. For the
    linear quantities the order is immaterial, so matching here keeps the two
    implementations comparable across every diagnostic rather than five of six.

    The cost is five small interpolations per timestep -- the fields are 2-D.
    """
    return TimestepResult(
        uref=interp_to_original_grid(result.uref, "yz"),
        zonal_mean_u=interp_to_original_grid(result.zonal_mean_u, "yz"),
        zonal_mean_lwa=interp_to_original_grid(result.zonal_mean_lwa, "yz"),
        lwa_baro=interp_to_original_grid(result.lwa_baro, "xy"),
        u_baro=interp_to_original_grid(result.u_baro, "xy"))


def interp_to_original_grid(field: np.ndarray, dims: str) -> np.ndarray:
    """Map a single field off the analysis grid back onto the input grid."""
    if dims == "yz":                 # (kmax, ylat) -> (kmax, lat)
        da = xr.DataArray(field, dims=("height", "ylat"),
                          coords={"ylat": ylat})
        return da.interp(ylat=original_grid[lat_name].values).values
    if dims == "xy":                 # (ylat, xlon) -> (lat, lon)
        da = xr.DataArray(field, dims=("ylat", "xlon"),
                          coords={"ylat": ylat, "xlon": xlon})
        return da.interp(ylat=original_grid[lat_name].values,
                         xlon=original_grid[lon_name].values).values
    raise ValueError(dims)


# ================================================================================
# 4) Checkpointing
# ================================================================================

def checkpoint_path(season: str) -> str:
    return os.path.join(wk_dir, "model", "netCDF", f"checkpoint_{season}.nc")


def save_checkpoint(season: str, results: List[TimestepResult],
                    yz_mask: np.ndarray, xy_mask: np.ndarray) -> None:
    """Persist progress so a killed run resumes near where it stopped."""
    if CHECKPOINT_EVERY <= 0 or not results:
        return
    path = checkpoint_path(season)
    stacked = {f: np.stack([getattr(r, f) for r in results], axis=0)
               for f in TimestepResult._fields}
    ds = xr.Dataset(
        data_vars={
            "uref": (("step", "height", "lat"), stacked["uref"]),
            "zonal_mean_u": (("step", "height", "lat"), stacked["zonal_mean_u"]),
            "zonal_mean_lwa": (("step", "height", "lat"), stacked["zonal_mean_lwa"]),
            "lwa_baro": (("step", "lat", "lon"), stacked["lwa_baro"]),
            "u_baro": (("step", "lat", "lon"), stacked["u_baro"]),
            "yz_mask": (("plev", "lat"), yz_mask.astype("i1")),
            "xy_mask": (("lat", "lon"), xy_mask.astype("i1"))},
        attrs={"n_done": len(results), "season": season, "casename": casename})
    tmp = path + ".tmp"
    ds.to_netcdf(tmp)
    ds.close()
    os.replace(tmp, path)   # atomic: a killed write never leaves a half file


def load_checkpoint(season: str):
    """Return (results, yz_mask, xy_mask) from a checkpoint, or None."""
    path = checkpoint_path(season)
    if CHECKPOINT_EVERY <= 0 or not os.path.isfile(path):
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
# 5) Plotting -- unchanged from the original driver
# ================================================================================

def plot_and_save_figure(seasonal_average_data, analysis_height_array, plot_dir,
                         title_str, season, xy_mask=None, yz_mask=None):
    if xy_mask is None:
        yland, xland = [], []
    else:
        yland, xland = np.where(xy_mask)
    lon_range = np.arange(-180, 181, 60)
    lat_range = np.arange(-90, 91, 30)
    cmap = "jet"

    height_lat_plotter = HeightLatPlotter(
        figsize=(4, 4), title_str=title_str, xgrid=original_grid[lat_name],
        ygrid=analysis_height_array, cmap=cmap, xlim=[-80, 80])
    height_lat_plotter.plot_and_save_variable(
        variable=seasonal_average_data.zonal_mean_u, cmap=cmap,
        var_title_str='zonal mean U',
        save_path=f"{plot_dir}{season}_zonal_mean_u.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(
        variable=seasonal_average_data.zonal_mean_lwa, cmap=cmap,
        var_title_str='zonal mean LWA',
        save_path=f"{plot_dir}{season}_zonal_mean_lwa.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(
        variable=seasonal_average_data.uref, cmap=cmap,
        var_title_str='zonal mean Uref',
        save_path=f"{plot_dir}{season}_zonal_mean_uref.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(
        variable=seasonal_average_data.zonal_mean_u - seasonal_average_data.uref,
        cmap=cmap, var_title_str=r'zonal mean $\Delta$ U',
        save_path=f"{plot_dir}{season}_zonal_mean_delta_u.eps", num_level=30)

    lat_lon_plotter = LatLonMapPlotter(
        figsize=(6, 3), title_str=title_str, xgrid=original_grid[lon_name],
        ygrid=original_grid[lat_name], cmap=cmap, xland=xland, yland=yland,
        lon_range=lon_range, lat_range=lat_range)
    lat_lon_plotter.plot_and_save_variable(
        variable=seasonal_average_data.u_baro, cmap=cmap, var_title_str='U baro',
        save_path=f"{plot_dir}{season}_u_baro.eps", num_level=30)
    lat_lon_plotter.plot_and_save_variable(
        variable=seasonal_average_data.lwa_baro, cmap=cmap, var_title_str='LWA baro',
        save_path=f"{plot_dir}{season}_lwa_baro.eps", num_level=30)
    lat_lon_plotter.plot_and_save_variable(
        variable=seasonal_average_data.covariance_lwa_u_baro, cmap="Purples_r",
        var_title_str='Covariance between LWA and U(baro)',
        save_path=f"{plot_dir}{season}_u_lwa_covariance.eps", num_level=30)


# ================================================================================
# 6) Main loop
# ================================================================================
model_or_obs: str = "model"
season_to_months = [
    ("DJF", [1, 2, 12]), ("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]), ("SON", [9, 10, 11])]

plot_dir = f"{wk_dir}/{model_or_obs}/PS/"
os.makedirs(os.path.join(wk_dir, model_or_obs, "netCDF"), exist_ok=True)

for season, selected_months in season_to_months:
    print(f"\nseason: {season}")
    season_dataset = model_dataset.where(
        model_dataset[time_coord_name].dt.month.isin(selected_months), drop=True)
    n_time = season_dataset[time_coord_name].size
    print(f"{season}: {n_time} timesteps selected")

    if n_time < 2:
        print(f"WARNING: {season} has {n_time} timestep(s); skipping this season.")
        season_dataset.close()
        continue

    resumed = load_checkpoint(season)
    if resumed is None:
        results, yz_mask, xy_mask = [], None, None
    else:
        results, yz_mask, xy_mask = resumed

    # Vertical grid, decided once per season from the input levels.
    plev_hpa = season_dataset[plev_name].values
    on_even_grid, dz, kmax = infer_vertical_grid(plev_hpa, default_dz=TARGET_DZ)
    print(f"    vertical grid: "
          f"{'evenly spaced in pseudoheight, interpolation SKIPPED' if on_even_grid else 'interpolating'}"
          f"; dz = {dz:.1f} m, kmax = {kmax}")

    for step in range(len(results), n_time):
        ds_t = season_dataset.isel({time_coord_name: step})[
            [u_var_name, v_var_name, t_var_name]].load()

        yz_t, xy_t = masks_for_timestep(ds_t[u_var_name].values)
        yz_mask = yz_t if yz_mask is None else (yz_mask | yz_t)
        xy_mask = xy_t if xy_mask is None else (xy_mask | xy_t)

        u, v, t = prepare_timestep(ds_t)
        ds_t.close()
        u, v, t, plev_oriented = orient_for_qgfield(u, v, t, plev_hpa)

        results.append(result_to_original_grid(compute_one_timestep(
            u, v, t, plev_oriented, dz, kmax, on_even_grid)))
        del u, v, t

        if (step + 1) % 20 == 0 or step + 1 == n_time:
            print(f"    {season}: {step + 1}/{n_time} timesteps")
        if CHECKPOINT_EVERY > 0 and (step + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint(season, results, yz_mask, xy_mask)
            gc.collect()

    save_checkpoint(season, results, yz_mask, xy_mask)
    season_dataset.close()

    # --- seasonal averages, then back onto the input grid --------------------
    analysis_height_array = np.arange(kmax) * dz
    # results are already on the input grid, so the covariance is computed
    # there too -- matching the original, and correct because covariance does
    # not commute with interpolation.
    seasonal_avg = time_average_processing(results)

    save_seasonal_diagnostics(
        seasonal_average_data=seasonal_avg,
        analysis_height_array=analysis_height_array,
        lat_coord=original_grid[lat_name], lon_coord=original_grid[lon_name],
        output_path=f"{wk_dir}/{model_or_obs}/netCDF/diagnostics_{season}.nc")

    title_string = f"{casename} ({firstyr}-{lastyr}) {season}"
    plot_and_save_figure(
        seasonal_average_data=seasonal_avg,
        analysis_height_array=analysis_height_array,
        plot_dir=plot_dir, title_str=title_string, season=season,
        xy_mask=xy_mask, yz_mask=yz_mask)
    print(f"{season}: figures written to {plot_dir}")

    del results, seasonal_avg
    gc.collect()

model_dataset.close()
print("POD Finite-amplitude wave diagnostic (zonal mean, streaming) finished successfully!")
