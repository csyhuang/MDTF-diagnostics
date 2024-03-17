"""
Compute daily average from sample files. This is working on OTC
"""
import os
import xarray as xr                # python library we use to read netcdf files
import matplotlib.pyplot as plt    # python library we use to make plots
from hn2016_falwa.xarrayinterface import QGDataset

u_path = f"{os.environ['HOME']}/GitHub/mdtf/inputdata/model/GFDL-CM4/data/atmos_inst/ts/hourly/1yr/atmos_inst.1984010100-1984123123.ua.nc"
v_path = f"{os.environ['HOME']}/GitHub/mdtf/inputdata/model/GFDL-CM4/data/atmos_inst/ts/hourly/1yr/atmos_inst.1984010100-1984123123.va.nc"
t_path = f"{os.environ['HOME']}/GitHub/mdtf/inputdata/model/GFDL-CM4/data/atmos_inst/ts/hourly/1yr/atmos_inst.1984010100-1984123123.ta.nc"
wk_dir = f"{os.environ['HOME']}/GitHub/mdtf/wkdir/"
u_daily_mean_path = f"{wk_dir}u_daily_mean.nc"
v_daily_mean_path = f"{wk_dir}v_daily_mean.nc"
t_daily_mean_path = f"{wk_dir}t_daily_mean.nc"
u_daily_mean_3steps_path = f"{wk_dir}u_daily_mean_3steps.nc"
v_daily_mean_3steps_path = f"{wk_dir}v_daily_mean_3steps.nc"
t_daily_mean_3steps_path = f"{wk_dir}t_daily_mean_3steps.nc"


def output_daily_avg(input_path, output_file, varname="ua"):
    input_file = xr.open_dataset(input_path)  # xarray.Dataset
    print(input_file)
    print("================")
    daily_resampled = input_file[varname].resample(time="1D").mean()
    print(daily_resampled)
    print("================")
    print(f"daily_resampled.shape:\n{daily_resampled.shape}")
    output_path = f"{wk_dir}{output_file}"
    print(f"Start outputing file: {output_path}")
    daily_resampled.to_netcdf(output_path)
    print(f"Finished outputing file: {output_path}")
    return output_path


if __name__ == '__main__':
    var_names = {
        'plev': 'level',
        'ylat': 'lat',
        'xlon': 'lon',
        'u': 'ua',
        'v': 'va',
        't': 'ta'}
    # u_output_path = output_daily_avg(u_path, "u_daily_mean.nc", varname="ua")
    # v_output_path = output_daily_avg(v_path, "v_daily_mean.nc", varname="va")
    # t_output_path = output_daily_avg(t_path, "t_daily_mean.nc", varname="ta")
    data_u = xr.open_dataset(u_daily_mean_path).isel(time=[0, 1, 2]).to_netcdf(u_daily_mean_3steps_path)
    data_v = xr.open_dataset(v_daily_mean_path).isel(time=[0, 1, 2]).to_netcdf(v_daily_mean_3steps_path)
    data_t = xr.open_dataset(t_daily_mean_path).isel(time=[0, 1, 2]).to_netcdf(t_daily_mean_3steps_path)

    print("Finished full procedures")