"""
Extract uvT data for 1 time step
"""
import os
import xarray as xr                # python library we use to read netcdf files


wk_dir = f"{os.environ['HOME']}/GitHub/mdtf/wkdir/"
data_dir = f"{os.environ['HOME']}/GitHub/mdtf/inputdata/model/GFDL-CM4/data/atmos_inst/ts/hourly/1yr/"
u_path = f"{data_dir}atmos_inst.1984010100-1984123123.ua.nc"
v_path = f"{data_dir}atmos_inst.1984010100-1984123123.va.nc"
t_path = f"{data_dir}atmos_inst.1984010100-1984123123.ta.nc"

print("Start outputting 1 timestamp")
tstep = 1000
xr.open_dataset(u_path).isel(time=tstep).to_netcdf(f"{data_dir}atmos_inst_t{tstep}_u.nc")
xr.open_dataset(v_path).isel(time=tstep).to_netcdf(f"{data_dir}atmos_inst_t{tstep}_v.nc")
xr.open_dataset(t_path).isel(time=tstep).to_netcdf(f"{data_dir}atmos_inst_t{tstep}_t.nc")
print("Finish outputting 1 timestamp")


