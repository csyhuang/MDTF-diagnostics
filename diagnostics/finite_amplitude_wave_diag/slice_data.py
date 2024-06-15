import os
import gc
import socket
from collections import namedtuple
import matplotlib.pyplot as plt
from finite_amplitude_wave_diag_utils import convert_hPa_to_pseudoheight, DataPreprocessor, LatLonMapPlotter, \
    HeightLatPlotter

# Commands to load third-party libraries. Any code you don't include that's
# not part of your language's standard library should be listed in the
# settings.jsonc file.
from typing import Dict
import numpy as np
import xarray as xr  # python library we use to read netcdf files
from falwa.xarrayinterface import QGDataset
from falwa.oopinterface import QGFieldNH18
from falwa.constant import P_GROUND, SCALE_HEIGHT


wk_dir = "/home/clare/GitHub/mdtf/scratch"
data_dir = "/home/clare/GitHub/mdtf/inputdata/model/GFDL-CM4/data/atmos/ts/6hr/1yr/"
uvt_path = f"{data_dir}atmos.2000010100-2000123123.[uvt]a.nc"
casename = "GFDL-CM4"

dataset = xr.open_mfdataset(uvt_path).isel(time=0)
for var in ["ua", "va", "ta"]:
    long_name = dataset.variables[var].attrs['long_name']
    fig = plt.contourf(dataset.coords['lat'], dataset.coords['level'], dataset.variables[var].mean(axis=-1),
                       40, cmap="jet")
    plt.colorbar()
    ax = fig.axes
    ax.invert_yaxis()
    plt.title(f"{casename}\nzonal mean {long_name} at timestep=0")
    plt.xlabel("latitude")
    plt.ylabel("pressure [hPa]")
    plt.savefig(f"check_{var}.png")
    plt.show()
print("From here")