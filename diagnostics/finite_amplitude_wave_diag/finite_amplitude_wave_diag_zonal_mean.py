# Finite-amplitude Rossby wave POD
# ================================================================================
# Calculate finite-amplitude wave diagnostics that quantifies wave-mean flow
# interactions.
#
# Last update: 03/18/2024
# ================================================================================
#   Version & Contact info
# 
#   - Version/revision information: version 1 (09/07/2023)
#   - PI: Clare S. Y. Huang. The University of Chicago. csyhuang@uchicago.edu.
#   - Developer/point of contact (name, affiliation, email): (same as PI)
#   - Other contributors: Christopher Polster (JGU Mainz), Noboru Nakamura (UChicago)
# ================================================================================
#   Open source copyright agreement
# 
#   The MDTF framework is distributed under the LGPLv3 license (see LICENSE.txt).
# ================================================================================
#   Functionality (not written yet)
# ================================================================================
#   Required programming language and libraries (not written yet)
# ================================================================================
#   Required model output variables (not written yet)
# ================================================================================
#   References (not written yet)
# ================================================================================
import os
import gc
from collections import namedtuple
import matplotlib
from finite_amplitude_wave_diag_utils import infer_vertical_grid, DataPreprocessor, LatLonMapPlotter, \
    HeightLatPlotter

# Commands to load third-party libraries. Any code you don't include that's
# not part of your language's standard library should be listed in the
# settings.jsonc file.
from typing import Dict
import intake
import numpy as np
import xarray as xr  # python library we use to read netcdf files
import yaml
from falwa.xarrayinterface import QGDataset
from falwa.oopinterface import QGFieldNH18

matplotlib.use('Agg')  # non-X windows backend; the framework always runs headless

frequency = "6hr"  # must match the frequency requested in settings.jsonc

# Timesteps handed to falwa at once. QGDataset builds one QGField per timestep
# and each retains roughly nine full 3-D fields, about 190 MiB at 42x181x360,
# so a whole 6-hourly season in one call would need over 100 GiB. Results are
# unaffected by this number -- timesteps are independent -- so lower it if
# memory is tight, raise it if there is headroom.
TIME_BATCH_SIZE = 40

# Pseudoheight spacing requested when the input is NOT already on an evenly
# spaced pseudoheight grid. Ignored when it is: falwa then takes dz from the
# data. See infer_vertical_grid.
TARGET_DZ = 1000.0

# *** Regular analysis grid defined by developer ***
xlon = np.arange(0, 360, 1.0)
ylat = np.arange(-90, 91, 1.0)

# 1) Loading model data files:
#
# The framework hands the POD its inputs through case_info.yml, whose location
# is given by the `case_env_file` environment variable. That file holds the
# path of the postprocessed data catalog plus, for each case, the model's own
# names for every requested variable and coordinate. This follows
# diagnostics/example_multicase and doc/sphinx/ref_envvars.rst.
#
# Reading os.environ['CASENAME'] directly, and building input paths by hand,
# are both pre-v4.0 patterns. The hand-built path this POD used before,
# <DATADIR>/<frequency>/<CASENAME>.<var>.<frequency>.nc, could never have
# matched: DATADIR is the POD's own work directory (WORK_DIR/<pod_name>) while
# the preprocessor writes to the case directory (WORK_DIR/<case_name>/<freq>/).
# Compare src/util/path_utils.py:109 with :158, and src/varlist_util.py:607.
wk_dir = os.environ["WORK_DIR"]
case_env_file = os.environ["case_env_file"]
assert os.path.isfile(case_env_file), f"case environment file not found: {case_env_file}"
with open(case_env_file, 'r') as stream:
    case_info = yaml.safe_load(stream)

cat_def_file = case_info['CATALOG_FILE']
case_list = case_info['CASE_LIST']

# This POD analyses one case at a time.
casename = list(case_list.keys())[0]
case_attrs = case_list[casename]
if len(case_list) > 1:
    print(f"WARNING: {len(case_list)} cases supplied; this POD analyses one. Using {casename}.")

# *** Coordinates of input dataset ***
# Taken from the framework rather than hardcoded, so that a convention whose
# names differ from the POD's still resolves.
u_var_name = case_attrs.get('ua_var', 'ua')
v_var_name = case_attrs.get('va_var', 'va')
t_var_name = case_attrs.get('ta_var', 'ta')
time_coord_name = case_attrs.get('time_coord', 'time')
plev_name = case_attrs.get('plev_coord', 'plev')
lat_name = case_attrs.get('lat_coord', 'lat')
lon_name = case_attrs.get('lon_coord', 'lon')

print(
    f"""
    wk_dir = {wk_dir}
    catalog = {cat_def_file}
    casename = {casename}
    variables = {u_var_name}, {v_var_name}, {t_var_name}
    coords = {time_coord_name}, {plev_name}, {lat_name}, {lon_name}
    """)

# 2) Doing computations:
cat = intake.open_esm_datastore(cat_def_file)
cat_subset = cat.search(
    variable_id=[u_var_name, v_var_name, t_var_name], frequency=frequency)
if cat_subset.df.empty:
    raise ValueError(
        f"No assets in {cat_def_file} for variables "
        f"{[u_var_name, v_var_name, t_var_name]} at frequency '{frequency}'. "
        f"Available variable_id: {sorted(cat.df['variable_id'].unique())}; "
        f"frequency: {sorted(cat.df['frequency'].unique())}")

dataset_dict = cat_subset.to_dataset_dict(
    progressbar=False,
    xarray_open_kwargs={"decode_times": True, "use_cftime": True})
model_dataset = dataset_dict[list(dataset_dict)[0]]

missing_vars = [v for v in (u_var_name, v_var_name, t_var_name) if v not in model_dataset]
if missing_vars:
    raise KeyError(
        f"{missing_vars} absent from the dataset returned by the catalog query. "
        f"Found {list(model_dataset.data_vars)}. If the catalog splits these "
        f"across groups, check the groupby_attrs in {cat_def_file}.")

firstyr = model_dataset.coords[time_coord_name].values[0].year
lastyr = model_dataset.coords[time_coord_name].values[-1].year
if model_dataset[plev_name].units == 'Pa':  # Pa shall be divided by 100 to become hPa
    print("model_dataset[plev_name].units == 'Pa'. Convert it to hPa.")
    # True division, not //. Floor division silently truncates any level that is
    # not a whole number of hPa, and on a grid evenly spaced in pseudoheight that
    # is most of them: 380.504 Pa and 329.851 Pa both collapse onto 3 hPa,
    # leaving a non-monotonic vertical coordinate. It also biases plev.min(),
    # and hence kmax below, towards levels that hold no data.
    model_dataset = model_dataset.assign_coords({plev_name: model_dataset[plev_name] / 100})
    model_dataset[plev_name].attrs["units"] = 'hPa'
print(f"""
    Use xlon: {xlon}
    Use ylat: {ylat}
    firstyr, lastyr = {firstyr}, {lastyr}
    """)

# === 2.0) Save original grid ===
original_grid = {
    time_coord_name: model_dataset.coords[time_coord_name],
    plev_name: model_dataset.coords[plev_name],
    lat_name: model_dataset.coords[lat_name],
    lon_name: model_dataset.coords[lon_name]}


def compute_batch(batch_dataset: xr.Dataset, dz, kmax, on_even_grid: bool):
    """Run the falwa diagnostics on one block of timesteps."""
    qgds = QGDataset(
        batch_dataset,
        var_names={"u": u_var_name, "v": v_var_name, "t": t_var_name},
        qgfield=QGFieldNH18,
        qgfield_kwargs={
            "dz": dz,
            "kmax": kmax,
            "data_on_evenly_spaced_pseudoheight_grid": on_even_grid})
    # Compute reference states and LWA
    qgds.interpolate_fields(return_dataset=False)
    qgds.compute_reference_states(return_dataset=False)
    qgds.compute_lwa_and_barotropic_fluxes(return_dataset=False)
    output_dataset = xr.Dataset(data_vars={
        'uref': qgds.uref,
        'zonal_mean_u': qgds.interpolated_u.mean(axis=-1),
        'zonal_mean_lwa': qgds.lwa.mean(axis=-1),
        'lwa_baro': qgds.lwa_baro,
        'u_baro': qgds.u_baro}).interp(coords={
        "xlon": (lon_name, original_grid[lon_name].data),
        "ylat": (lat_name, original_grid[lat_name].data)})
    # Materialise before the QGField objects behind it are released.
    output_dataset = output_dataset.compute()
    del qgds
    gc.collect()
    return output_dataset


def compute_from_sampled_data(gridfilled_dataset: xr.Dataset):

    # === 2.3) VERTICAL RESOLUTION ===
    # If the input already sits on an evenly spaced pseudoheight grid, hand it
    # to falwa as-is: it then takes dz and kmax from the data and skips the
    # vertical interpolation. Otherwise fall back to interpolating onto a
    # TARGET_DZ grid reaching as high as the data does.
    on_even_grid, dz, kmax = infer_vertical_grid(
        gridfilled_dataset[plev_name].values, default_dz=TARGET_DZ)
    print(
        f"""
        Vertical grid: {'already evenly spaced in pseudoheight' if on_even_grid else 'needs interpolation'}
        dz = {dz} m, kmax = {kmax}
        vertical interpolation in falwa: {'SKIPPED' if on_even_grid else 'performed'}
        """)

    # === 2.4) WAVE ACTIVITY COMPUTATION: Compute Uref, FAWA, barotropic components of u and LWA ===
    # Processed in blocks of timesteps. QGDataset holds one QGField per
    # timestep, and each retains ~9 full 3-D fields, so a whole season at
    # 6-hourly resolution would need well over 100 GiB at once. The quantities
    # kept afterwards are all 2-D and cost about 1 MiB per timestep, so
    # batching bounds the peak without changing any result: every timestep is
    # still processed independently, exactly as before.
    n_time = gridfilled_dataset[time_coord_name].size
    batches = []
    for start in range(0, n_time, TIME_BATCH_SIZE):
        stop = min(start + TIME_BATCH_SIZE, n_time)
        print(f"  computing timesteps {start}-{stop - 1} of {n_time}")
        batches.append(compute_batch(
            gridfilled_dataset.isel({time_coord_name: slice(start, stop)}),
            dz=dz, kmax=kmax, on_even_grid=on_even_grid))

    output_dataset = batches[0] if len(batches) == 1 \
        else xr.concat(batches, dim=time_coord_name)
    gridfilled_dataset.close()
    return output_dataset


def calculate_covariance(lwa_baro, u_baro):
    """
    Calculate the temporal covariance of LWA and U at each grid point.

    Args:
        lwa_baro: dataset.lwa_baro, dimension (time, lat, lon)
        u_baro: dataset.u_baro, dimension (time, lat, lon)
    Returns:
        cov_map in dimension of (lat, lon)

    Note:
        This used to call np.cov(m, y, rowvar=False), which treats every grid
        point as a separate variable and so builds a (2N, 2N) matrix before
        np.diagonal throws almost all of it away. On a 181x360 grid that is
        130320^2 float64 = 127 GiB allocated to keep 0.5 MiB. The elementwise
        form below gives the same numbers -- Bessel-corrected, matching
        np.cov's default ddof=1 -- in O(N*T).
    """
    a = np.asarray(lwa_baro.data)
    b = np.asarray(u_baro.data)
    n_time = a.shape[0]
    if n_time < 2:
        raise ValueError(
            f"covariance needs at least 2 timesteps, got {n_time}")
    a_anomaly = a - a.mean(axis=0)
    b_anomaly = b - b.mean(axis=0)
    cov_map = (a_anomaly * b_anomaly).sum(axis=0) / (n_time - 1)
    return cov_map


def time_average_processing(dataset: xr.Dataset):
    SeasonalAverage = namedtuple(
        "SeasonalAverage", [
            "zonal_mean_u",
            "uref",
            "zonal_mean_lwa",
            "lwa_baro",
            "u_baro",
            "covariance_lwa_u_baro"])

    seasonal_avg_zonal_mean_u = dataset.zonal_mean_u.mean(axis=0)
    seasonal_avg_zonal_mean_lwa = dataset.zonal_mean_lwa.mean(axis=0)
    seasonal_avg_uref = dataset.uref.mean(axis=0)
    seasonal_avg_lwa_baro = dataset.lwa_baro.mean(axis=0)
    seasonal_avg_u_baro = dataset.u_baro.mean(axis=0)
    seasonal_covariance_lwa_u_baro = calculate_covariance(lwa_baro=dataset.lwa_baro, u_baro=dataset.u_baro)
    seasonal_avg_data = SeasonalAverage(
        seasonal_avg_zonal_mean_u, seasonal_avg_uref, seasonal_avg_zonal_mean_lwa,
        seasonal_avg_lwa_baro, seasonal_avg_u_baro, seasonal_covariance_lwa_u_baro)
    return seasonal_avg_data


def plot_and_save_figure(seasonal_average_data, analysis_height_array, plot_dir, title_str, season,
                         xy_mask=None, yz_mask=None):
    if xy_mask is None:
        xy_mask = np.zeros_like(seasonal_average_data.u_baro)
        yland, xland = [], []
    else:
        yland, xland = np.where(xy_mask)
    if yz_mask is None:
        yz_mask = np.zeros_like(seasonal_average_data.zonal_mean_u)
    lon_range = np.arange(-180, 181, 60)
    lat_range = np.arange(-90, 91, 30)

    cmap = "jet"

    height_lat_plotter = HeightLatPlotter(figsize=(4, 4), title_str=title_str, xgrid=original_grid[lat_name],
                                          ygrid=analysis_height_array, cmap=cmap, xlim=[-80, 80])
    height_lat_plotter.plot_and_save_variable(variable=seasonal_average_data.zonal_mean_u, cmap=cmap,
                                              var_title_str='zonal mean U',
                                              save_path=f"{plot_dir}{season}_zonal_mean_u.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(variable=seasonal_average_data.zonal_mean_lwa, cmap=cmap,
                                              var_title_str='zonal mean LWA',
                                              save_path=f"{plot_dir}{season}_zonal_mean_lwa.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(variable=seasonal_average_data.uref, cmap=cmap,
                                              var_title_str='zonal mean Uref',
                                              save_path=f"{plot_dir}{season}_zonal_mean_uref.eps", num_level=30)
    height_lat_plotter.plot_and_save_variable(variable=seasonal_average_data.zonal_mean_u - seasonal_average_data.uref,
                                              cmap=cmap, var_title_str=r'zonal mean $\Delta$ U',
                                              save_path=f"{plot_dir}{season}_zonal_mean_delta_u.eps", num_level=30)

    # Use encapsulated class to plot
    lat_lon_plotter = LatLonMapPlotter(figsize=(6, 3), title_str=title_str, xgrid=original_grid[lon_name],
                                       ygrid=original_grid[lat_name], cmap=cmap, xland=xland, yland=yland,
                                       lon_range=lon_range, lat_range=lat_range)
    lat_lon_plotter.plot_and_save_variable(variable=seasonal_average_data.u_baro, cmap=cmap, var_title_str='U baro',
                                           save_path=f"{plot_dir}{season}_u_baro.eps", num_level=30)
    lat_lon_plotter.plot_and_save_variable(variable=seasonal_average_data.lwa_baro, cmap=cmap, var_title_str='LWA baro',
                                           save_path=f"{plot_dir}{season}_lwa_baro.eps", num_level=30)
    lat_lon_plotter.plot_and_save_variable(variable=seasonal_average_data.covariance_lwa_u_baro, cmap="Purples_r",
                                           var_title_str='Covariance between LWA and U(baro)',
                                           save_path=f"{plot_dir}{season}_u_lwa_covariance.eps", num_level=30)


# === 3) Saving output data ===
# Diagnostics should write output data to disk to a) make relevant results
# available to the user for further use or b) to pass large amounts of data
# between stages of a calculation run as different sub-scripts. Data can be in
# any format (as long as it's documented) and should be written to the
# directory <WK_DIR>/model/netCDF (created by the framework).

# *** MAIN PROCESS: Produce data by season, daily ***
model_or_obs: str = "model"  # It can be "model" or "obs"
season_to_months = [
    ("DJF", [1, 2, 12]), ("MAM", [3, 4, 5]), ("JJA", [6, 7, 8]), ("SON", [9, 10, 11])]
intermediate_output_paths: Dict[str, str] = {
    item[0]: f"{wk_dir}/{model_or_obs}/intermediate_{item[0]}.nc" for item in season_to_months}

for season, selected_months in season_to_months:
    print(f"season: {season}")
    # Construct data preprocessor
    data_preprocessor = DataPreprocessor(
        wk_dir=wk_dir, xlon=xlon, ylat=ylat, u_var_name=u_var_name, v_var_name=v_var_name, t_var_name=t_var_name,
        plev_name=plev_name, lat_name=lat_name, lon_name=lon_name, time_coord_name=time_coord_name)

    plot_dir = f"{wk_dir}/{model_or_obs}/PS/"

    # Select the season. Every timestep in it is used -- at 6-hourly input that
    # is all four samples a day.
    #
    # This previously ended in .groupby("time.day").first(). time.day is the
    # day of the *month*, so grouping a multi-month, multi-year season by it
    # collapsed to at most 31 samples, all drawn from whichever month came
    # first: on a two-year record, "DJF" was 720 timesteps reduced to 31, every
    # one of them 03:00 in January of year one. It also replaced the time
    # dimension with a day dimension. Both are gone.
    #
    # LWA is a nonlinear functional of the instantaneous field, so the average
    # of the diagnostic is not the diagnostic of the average. Keeping every
    # timestep and averaging the results afterwards is the faithful order, and
    # it is what the h7i (instantaneous) input was chosen for.
    sampled_dataset = model_dataset.where(
        model_dataset[time_coord_name].dt.month.isin(selected_months), drop=True)
    print(f"{season}: {sampled_dataset[time_coord_name].size} timesteps selected "
          f"(all samples in the season, no temporal subsampling)")
    preprocessed_output_path = intermediate_output_paths[season]  # TODO set it
    print(f"Start preparing intermediate data in the directory: {preprocessed_output_path}")
    data_preprocessor.output_preprocess_data(
        sampled_dataset=sampled_dataset, output_path=preprocessed_output_path)
    print(f"Finished preparing intermediate data in the directory: {preprocessed_output_path}")
    intermediate_dataset = xr.open_mfdataset(preprocessed_output_path)
    print(f"Start computing FAWA diagnostics from sampled data.")
    fawa_diagnostics_dataset = compute_from_sampled_data(intermediate_dataset)
    analysis_height_array = fawa_diagnostics_dataset.coords['height'].data
    seasonal_avg_data = time_average_processing(fawa_diagnostics_dataset)
    print(
        f"""
        Finished computing FAWA diagnostics from sampled data.
        fawa_diagnostics_dataset: {fawa_diagnostics_dataset}
        seasonal_avg_data: {seasonal_avg_data}
        """)

    # === 4) Saving output plots ===
    #
    # Plots should be saved in EPS or PS format at <WK_DIR>/<model or obs>/PS
    # (created by the framework). Plots can be given any filename, but should have
    # the extension ".eps" or ".ps". To make the webpage output, the framework will
    # convert these to bitmaps with the same name but extension ".png".

    # Define a python function to make the plot, since we'll be doing it twice and
    # we don't want to repeat ourselves.

    # set an informative title using info about the analysis set in env vars
    title_string = f"{casename} ({firstyr}-{lastyr}) {season}"
    # Plot the model data:
    plot_and_save_figure(
        seasonal_average_data=seasonal_avg_data,
        analysis_height_array=analysis_height_array,
        plot_dir=plot_dir,
        title_str=title_string,
        season=season,
        xy_mask=data_preprocessor.xy_mask,
        yz_mask=data_preprocessor.yz_mask)
    print(f"Finishing outputting figures to {plot_dir}.")

    # Close xarray datasets
    sampled_dataset.close()
    intermediate_dataset.close()
    fawa_diagnostics_dataset.close()
    gc.collect()
print("Finish the whole process")
model_dataset.close()

# 6) Cleaning up:
#
# In addition to your language's normal housekeeping, don't forget to delete any
# temporary/scratch files you created in step 4).
# os.system(f"rm -f {wk_dir}/model/gridfill_*.nc")
# os.system(f"rm -f {wk_dir}/model/intermediate_*.nc")

# 7) Error/Exception-Handling Example ########################################
# nonexistent_file_path = "{DATADIR}/mon/nonexistent_file.nc".format(**os.environ)
# try:
#     nonexistent_dataset = xr.open_dataset(nonexistent_file_path)
# except IOError as error:
#     print(error)
#     print("This message is printed by the example POD because exception-handling is working!")

# 8) Confirm POD executed sucessfully ########################################
print("POD Finite-amplitude wave diagnostic (zonal mean) finished successfully!")

