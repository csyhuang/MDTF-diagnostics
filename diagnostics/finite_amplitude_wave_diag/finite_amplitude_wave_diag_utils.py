from typing import Optional

import gridfill
import matplotlib.pyplot as plt  # python library we use to make plots
import numpy as np
import xarray as xr
from falwa.constant import P_GROUND, SCALE_HEIGHT
from matplotlib import gridspec
from cartopy import crs as ccrs

# from diagnostics.finite_amplitude_wave_diag.finite_amplitude_wave_diag_zonal_mean import plev_name, lat_name, lon_name, \
#     sampled_dataset


def gridfill_each_level(lat_lon_field, itermax=1000, verbose=False):
    """
    Fill missing values in lat-lon grids with values derived by solving Poisson's equation
    using a relaxation scheme.

    Args:
        lat_lon_field(np.ndarray): 2D array to apply gridfill on
        itermax(int): maximum iteration for poisson solver
        verbose(bool): verbose level of poisson solver

    Returns:
        A 2D array of the same dimension with all nan filled.
    """
    if np.isnan(lat_lon_field).sum() == 0:
        return lat_lon_field

    lat_lon_filled, converged = gridfill.fill(
        grids=np.ma.masked_invalid(lat_lon_field), xdim=1, ydim=0, eps=0.01,
        cyclic=True, itermax=itermax, verbose=verbose)

    return lat_lon_filled


class LatLonMapPlotter(object):
    def __init__(self, figsize, title_str, xgrid, ygrid, cmap, xland, yland, lon_range, lat_range):
        self._figsize = figsize
        self._title_str = title_str
        self._xgrid = xgrid
        self._ygrid = ygrid
        self._cmap = cmap
        self._xland = xland
        self._yland = yland
        self._lon_range = lon_range
        self._lat_range = lat_range

    def plot_and_save_variable(self, variable, cmap, var_title_str, save_path=None, num_level=30):
        fig = plt.figure(figsize=self._figsize)
        spec = gridspec.GridSpec(
            ncols=1, nrows=1, wspace=0.3, hspace=0.3)
        ax = fig.add_subplot(spec[0], projection=ccrs.PlateCarree())
        ax.coastlines(color='black', alpha=0.7)
        ax.set_aspect('auto', adjustable=None)
        main_fig = ax.contourf(
            self._xgrid, self._ygrid,
            variable,
            num_level,
            cmap=cmap)
        ax.scatter(self._xgrid[self._xland], self._ygrid[self._yland], s=1, c='gray')
        ax.set_xticks(self._lon_range, crs=ccrs.PlateCarree())
        ax.set_yticks(self._lat_range, crs=ccrs.PlateCarree())
        fig.colorbar(main_fig, ax=ax)
        ax.set_title(f"{self._title_str}\n{var_title_str}")
        # save_path=None: used by the notebook driver, which displays the figure
        # inline instead of writing EPS for the framework to convert.
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        # Returning the figure lets a caller display or further annotate it.
        # plt.show() is deliberately NOT called here: under nbconvert it would
        # close the figure before the notebook could render it, and headless it
        # is a no-op that only costs time.
        return fig


class HeightLatPlotter(object):
    def __init__(self, figsize, title_str, xgrid, ygrid, cmap, xlim):
        self._figsize = figsize
        self._title_str = title_str
        self._xgrid = xgrid
        self._ygrid = ygrid
        self._cmap = cmap
        self._xlim = xlim  # [-80, 80]

    def plot_and_save_variable(self, variable, cmap, var_title_str, save_path=None, num_level=30):
        fig = plt.figure(figsize=self._figsize)
        spec = gridspec.GridSpec(ncols=1, nrows=1)
        ax = fig.add_subplot(spec[0])
        # *** Zonal mean U ***
        main_fig = ax.contourf(
            self._xgrid,
            self._ygrid,
            variable,
            num_level,
            cmap=cmap if cmap else self._cmap)
        fig.colorbar(main_fig, ax=ax)
        ax.set_title(f"{self._title_str}\n{var_title_str}")
        ax.set_xlim(self._xlim)
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')
        return fig


def convert_pseudoheight_to_hPa(height_array):
    """
    Args:
        height_array(np.array): pseudoheight in [m]

    Returns:
        np.array which contains pressure levels in [hPa]
    """
    p_array = P_GROUND * np.exp(- height_array / SCALE_HEIGHT)
    return p_array


def convert_hPa_to_pseudoheight(p_array):
    """
    Args:
        height_array(np.array): pseudoheight in [m]

    Returns:
        np.array which contains pressure levels in [hPa]
    """
    height_array = - SCALE_HEIGHT * np.log(p_array / P_GROUND)
    return height_array


def save_seasonal_diagnostics(seasonal_average_data, analysis_height_array,
                              lat_coord, lon_coord, output_path):
    """Write the seasonal-mean diagnostics to netCDF.

    The POD's primary product is figures, but the numbers behind them are worth
    keeping: they are what a user would reanalyse, and what lets two
    implementations be compared quantitatively rather than by eyeballing plots.

    Args:
        seasonal_average_data: namedtuple with the six seasonal-mean fields.
        analysis_height_array: pseudoheight coordinate, metres.
        lat_coord, lon_coord: coordinates of the grid the fields sit on.
        output_path: destination .nc path.
    """
    height = np.asarray(analysis_height_array)
    lat = np.asarray(lat_coord)
    lon = np.asarray(lon_coord)

    def _values(field):
        return np.asarray(getattr(field, "values", field))

    dataset = xr.Dataset(
        data_vars={
            "zonal_mean_u": (("height", "lat"), _values(seasonal_average_data.zonal_mean_u)),
            "uref": (("height", "lat"), _values(seasonal_average_data.uref)),
            "zonal_mean_lwa": (("height", "lat"), _values(seasonal_average_data.zonal_mean_lwa)),
            "lwa_baro": (("lat", "lon"), _values(seasonal_average_data.lwa_baro)),
            "u_baro": (("lat", "lon"), _values(seasonal_average_data.u_baro)),
            "covariance_lwa_u_baro": (
                ("lat", "lon"), _values(seasonal_average_data.covariance_lwa_u_baro)),
        },
        coords={"height": height, "lat": lat, "lon": lon})
    dataset["height"].attrs.update(units="m", long_name="pseudoheight")
    dataset["zonal_mean_u"].attrs.update(units="m s-1")
    dataset["uref"].attrs.update(units="m s-1", long_name="reference state zonal wind")
    dataset["zonal_mean_lwa"].attrs.update(units="m s-1", long_name="local wave activity")
    dataset.to_netcdf(output_path)
    dataset.close()
    print(f"Saved seasonal diagnostics to {output_path}")


def normalize_orientation(dataset, lat_name, plev_name, verbose=True):
    """Put a dataset into the axis order falwa requires.

    ``QGField`` expects **latitude ascending** (south to north) and **pressure
    descending** (i.e. pseudoheight ascending). Reanalysis is commonly stored
    the other way round -- ERA5 from the CDS is latitude-descending, and its
    pressure levels ascend from 1 hPa -- so the input has to be reoriented
    before anything else touches it.

    This mirrors the treatment in falwa's own
    ``notebooks/nh2018_science/demo_script_for_nh2018.ipynb``, with one
    difference: doing it on the xarray Dataset flips the coordinate and the
    data together and cannot desynchronise them. Flipping a coordinate array
    and the data separately is easy to get half-right, and an inverted column
    or a hemisphere-flipped field produces plausible-looking output rather than
    an error.

    Applied once at load time, so every downstream step sees one orientation.

    Args:
        dataset (xr.Dataset): input, modified by reindexing (not in place).
        lat_name (str): name of the latitude coordinate.
        plev_name (str): name of the pressure coordinate.
        verbose (bool): report when a flip happens.

    Returns:
        xr.Dataset with latitude ascending and pressure descending.
    """
    if lat_name in dataset.coords:
        lat = dataset[lat_name].values
        if lat.size > 1 and lat[1] < lat[0]:
            if verbose:
                print(f"Flip {lat_name}: descending -> ascending (falwa requires "
                      f"south-to-north)")
            dataset = dataset.isel({lat_name: slice(None, None, -1)})

    if plev_name in dataset.coords:
        plev = dataset[plev_name].values
        if plev.size > 1 and plev[1] > plev[0]:
            if verbose:
                print(f"Flip {plev_name}: ascending -> descending (falwa requires "
                      f"ascending pseudoheight)")
            dataset = dataset.isel({plev_name: slice(None, None, -1)})

    return dataset


def drop_leap_day(dataset, time_coord_name, verbose=True):
    """Remove 29 February, so every year contributes the same sample count.

    The model runs on a ``noleap`` calendar; reanalysis does not. Dropping the
    leap day makes the two calendars agree and, more usefully, makes every year
    -- and every DJF winter -- contain exactly the same number of timesteps. A
    per-year mean and a pooled mean then coincide, which removes the weighting
    question rather than answering it.

    The cost is small: over 1991-2020 there are 8 leap years, so at 6-hourly
    sampling this discards 32 of 43832 timesteps, 0.073%.

    A no-op on calendars that have no leap day.
    """
    time = dataset[time_coord_name]
    try:
        is_leap_day = (time.dt.month == 2) & (time.dt.day == 29)
    except (AttributeError, TypeError):
        return dataset                      # non-datetime axis; nothing to do

    n_dropped = int(is_leap_day.sum())
    if n_dropped == 0:
        return dataset
    if verbose:
        print(f"Dropping {n_dropped} timestep(s) on 29 February so every year "
              f"has equal weight")
    return dataset.sel({time_coord_name: ~is_leap_day})


def infer_vertical_grid(plev_hPa, default_dz=1000.0, tol=1.0):
    """Decide whether the input already sits on an evenly spaced pseudoheight grid.

    falwa's analysis grid is uniform in pseudoheight, z = -H ln(p/P_GROUND).
    When the input pressure levels already form such a grid, `QGField` can take
    them as they are and skip the vertical interpolation entirely -- pass
    ``data_on_evenly_spaced_pseudoheight_grid=True`` and it derives ``dz`` and
    ``kmax`` from the data itself. Skipping that step avoids resampling fields
    that are already where they need to be, which would otherwise smooth the
    profile for no reason.

    Two conditions must hold, both checked here:

    1. The levels are equally spaced in pseudoheight, to within *tol* metres.
    2. The lowest level is the ground, z = 0 (p = P_GROUND = 1000 hPa). falwa's
       own grid starts there, and parts of the reference-state solver assume a
       column beginning at zero, so a uniform grid that starts elsewhere is not
       interchangeable with it.

    Args:
        plev_hPa: pressure levels in hPa, ordered from high to low pressure.
        default_dz: pseudoheight spacing to request when interpolation is needed.
        tol: tolerance in metres for calling the spacing uniform.

    Returns:
        (data_on_evenly_spaced_pseudoheight_grid, dz, kmax)
    """
    height = convert_hPa_to_pseudoheight(np.asarray(plev_hPa, dtype=float))

    hmax = float(np.nanmax(height))
    fallback = (False, default_dz, int(hmax // default_dz) + 1)

    if height.size < 3:
        return fallback
    # falwa needs ascending pseudoheight, i.e. descending pressure. Anything
    # else is not a candidate for the pass-through branch.
    spacing = np.diff(height)
    if np.any(spacing <= 0):
        return fallback

    uniform = float(np.max(np.abs(spacing - spacing[0]))) <= tol
    starts_at_ground = abs(float(height[0])) <= tol
    if uniform and starts_at_ground:
        return True, float(spacing[0]), int(height.size)
    return fallback


class DataPreprocessor:
    def __init__(
            self, wk_dir, xlon, ylat, u_var_name, v_var_name, t_var_name, plev_name, lat_name, lon_name, time_coord_name):

        self._wk_dir = wk_dir
        self._xlon: np.array = xlon  # user input
        self._ylat: np.array = ylat  # user input
        self._u_var_name: str = u_var_name
        self._v_var_name: str = v_var_name
        self._t_var_name: str = t_var_name
        self._plev_name: str = plev_name
        self._lat_name: str = lat_name
        self._lon_name: str = lon_name
        self._original_plev = None
        self._original_lat = None
        self._original_lon = None
        self._original_time_coord = None
        self._time_coord_name: str = time_coord_name
        self._new_time_coord_name: str = "day"
        self._sampled_dataset = None  # Shall be xarray. Set type later
        self._gridfill_needed: Optional[bool] = None
        self._yz_mask = None
        self._xy_mask = None

    @property
    def xy_mask(self):
        return self._xy_mask

    @property
    def yz_mask(self):
        return self._yz_mask

    def _save_original_coordinates(self, dataset):
        self._original_plev = dataset.coords[self._plev_name]
        self._original_lat = dataset.coords[self._lat_name]
        self._original_lon = dataset.coords[self._lon_name]

    def _check_if_gridfill_is_needed(self, sampled_dataset):
        num_of_nan = sampled_dataset[self._u_var_name].isnull().sum().values
        if num_of_nan > 0:
            self._gridfill_needed = True
            self._do_save_mask(sampled_dataset)
        else:
            self._gridfill_needed = False

    def _do_save_mask(self, dataset):
        self._yz_mask = dataset[self._u_var_name]\
            .to_masked_array().mask.sum(axis=0).sum(axis=-1).astype(bool)
        self._xy_mask = dataset[self._u_var_name]\
            .to_masked_array().mask[:, 1:, :, :].sum(axis=0).sum(axis=0).astype(bool)

    def _save_preprocessed_data(self, dataset, output_path):
        dataset.to_netcdf(output_path)
        dataset.close()
        print(f"Finished outputing preprocessed dataset: {output_path}")

    def _interpolate_onto_regular_grid(self, dataset):
        dataset = dataset.interp(
            coords={self._lat_name: self._ylat, self._lon_name: self._xlon},
            method="linear",
            kwargs={"fill_value": "extrapolate"})
        return dataset

    def _implement_gridfill(self, dataset: xr.Dataset) -> xr.Dataset:
        """Fill NaN horizontally at each level, returning the filled dataset.

        Always returns an xr.Dataset. It previously returned either a Dataset
        (nothing to fill) or a path glob (something filled), and the caller
        assumed the latter, so on NaN-free input xr.open_mfdataset was handed a
        Dataset, iterated it into variable names, and failed trying to open a
        file called 'T'. NaN-free input is not exotic -- it is what pressure-
        level reanalysis, or a regrid that extrapolates below ground instead of
        masking, both produce.
        """
        if not self._gridfill_needed:
            print("No NaN values detected. Gridfill not needed. Bypass DataPreprocessor._implement_gridfill.")
            return dataset
        # *** Implement gridfill procedure ***
        print(f"self._gridfill_needed = True. Do gridfill with poisson solver.")
        args_tuple = [self._u_var_name, self._v_var_name, self._t_var_name]
        gridfill_file_path = self._wk_dir + "/model/netCDF/gridfill_{var}.nc"
        for var_name in args_tuple:
            field_at_all_level = xr.apply_ufunc(
                gridfill_each_level,
                *[dataset[var_name]],
                input_core_dims=((self._lat_name, self._lon_name),),
                output_core_dims=((self._lat_name, self._lon_name),),
                # dask="parallelized", NOT "allowed". With "allowed" the dask
                # array is handed straight to np.vectorize, which does not
                # iterate it correctly once there is a loop dimension over
                # (time, plev): the solver silently returns wrong values --
                # ~15 m/s errors on a synthetic test, and the largest
                # discrepancies at the level with the most missing data, which
                # is the 1000 hPa level where nearly half the grid is below
                # ground. Input from the framework is always dask-backed
                # (intake's to_dataset_dict), so this path was always taken.
                # "parallelized" reproduces the numpy result exactly, and
                # raises rather than guessing if lat/lon are ever chunked.
                vectorize=True, dask="parallelized",
                output_dtypes=[dataset[var_name].dtype])
            field_at_all_level.to_netcdf(gridfill_file_path.format(var=var_name))
            field_at_all_level.close()
            print(f"Finished outputing {var_name} to {gridfill_file_path.format(var=var_name)}")
        load_gridfill_path = gridfill_file_path.format(var="*")
        return xr.open_mfdataset(load_gridfill_path)

    def output_preprocess_data(self, sampled_dataset, output_path):
        """
        Main procedure executed by this class
        """
        self._save_original_coordinates(sampled_dataset)
        self._check_if_gridfill_is_needed(sampled_dataset)
        gridfilled_dataset = self._implement_gridfill(sampled_dataset)
        dataset = self._interpolate_onto_regular_grid(gridfilled_dataset)  # Interpolate onto regular grid
        if self._gridfill_needed:
            # Only close what we opened. When no filling was required the
            # object above *is* sampled_dataset, and closing it would drop the
            # caller's file handles for every remaining season.
            gridfilled_dataset.close()
        self._save_preprocessed_data(dataset, output_path)  # Save preprocessed data
        dataset.close()



