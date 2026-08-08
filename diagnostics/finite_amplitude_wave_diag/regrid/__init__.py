"""Regrid native-grid CAM ne120pg3 output to a regular lat-lon pressure grid.

The MDTF preprocessor assumes X/Y axes and a pressure Z axis, but the raw files
are ``VAR(time, lev, ncol)`` on an unstructured cubed-sphere mesh with
hybrid-sigma levels. Both a horizontal remap and a vertical interpolation are
therefore required before the finite_amplitude_wave_diag POD can read them.

This package wraps that pipeline. It is a port of ``regrid_ne120_to_latlon.sh``
and issues the same NCO commands in the same order; the shell script remains in
the POD directory as the reference implementation.

The module deliberately depends on nothing outside the standard library. Every
netCDF operation shells out to NCO, which has to be on ``PATH`` regardless, so
the package runs unmodified in the MDTF framework environment, in the
``mdtf_regrid`` environment, or in a bare Python 3.9+ interpreter.

Typical use::

    python -m regrid setup
    python -m regrid regrid T U V
    python -m regrid catalog
    python -m regrid validate /path/to/output.nc

or from Python::

    from regrid import RegridConfig, setup, regrid_variable
    cfg = RegridConfig.from_env(data_dir="/data", out_dir="/data/regridded")
    setup(cfg)
    regrid_variable(cfg, "T")
"""

from .config import RegridConfig
from .levels import (
    P_GROUND_HPA,
    SCALE_HEIGHT_M,
    format_level_table,
    parse_level_list,
    pressure_to_pseudoheight,
    pseudoheight_levels,
    pseudoheight_to_pressure,
)
from .mesh import setup
from .nco import NCOError, ToolNotFoundError
from .pipeline import regrid_variable
from .validate import validate_file
from .catalog import write_catalog

__all__ = [
    "RegridConfig",
    "NCOError",
    "ToolNotFoundError",
    "setup",
    "regrid_variable",
    "validate_file",
    "write_catalog",
    "pseudoheight_levels",
    "pseudoheight_to_pressure",
    "pressure_to_pseudoheight",
    "format_level_table",
    "parse_level_list",
    "P_GROUND_HPA",
    "SCALE_HEIGHT_M",
]

__version__ = "0.1.0"
