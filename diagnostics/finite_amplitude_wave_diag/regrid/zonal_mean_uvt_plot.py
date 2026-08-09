#!/usr/bin/env python3
"""Zonal-mean cross-sections of regridded U, V and T, as a single PDF.

A visual counterpart to the automated checks in ``regrid validate``. Those
confirm the output is structurally sound; this confirms it is *physically*
sound, which is the part no assertion catches. On a correct regrid you should
see the subtropical jets near 200 hPa, a zonal-mean meridional wind of
essentially zero, a cold point near 100 hPa in the tropics, and gaps along the
bottom wherever the surface rises above the target level.

Standalone by design. It does not import the ``regrid`` package -- it reads
everything it needs, including the regridding provenance, from the netCDF files
themselves, so it works on any regridded output regardless of how it was
produced. It is also the only script here that needs matplotlib, xarray and
netCDF4; the pipeline itself is standard library only, and requiring a plotting
stack to run a regrid on a compute node would be a poor trade.

Usage::

    python zonal_mean_uvt_plot.py /path/to/regridded
    python zonal_mean_uvt_plot.py /path/to/regridded -o check.pdf -t 0
    python zonal_mean_uvt_plot.py file_U.nc file_V.nc file_T.nc

Run it from an environment with the plotting stack, e.g. the POD's own runtime
environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

# Pseudoheight constants, matching falwa.constant. Duplicated here rather than
# imported so the script stays independent of the regrid package.
SCALE_HEIGHT_M = 7000.0
P_GROUND_PA = 100000.0

#: (variable, long name, units, colormap, diverging). Diverging fields get a
#: symmetric scale centred on zero so the sign structure stays readable.
PANELS: List[Tuple[str, str, str, str, bool]] = [
    ("U", "Zonal wind", "m s$^{-1}$", "RdBu_r", True),
    ("V", "Meridional wind", "m s$^{-1}$", "RdBu_r", True),
    ("T", "Temperature", "K", "viridis", False),
]

#: Pressure ticks in hPa, chosen to read naturally on a log axis.
PRESSURE_TICKS = [1000, 700, 500, 300, 200, 100, 50, 20, 10, 5, 3]

#: Pseudoheight ticks in km for the right-hand axis.
PSEUDOHEIGHT_TICKS = [0, 5, 10, 15, 20, 25, 30, 35, 40]

PROVENANCE_ATTRS = (
    "regrid_source_stream",
    "regrid_source_file",
    "regrid_vrt_xtr",
    "regrid_algo",
)


def resolve_inputs(paths: Sequence[str], variables: Sequence[str]) -> Dict[str, str]:
    """Map each variable to a file.

    *paths* is either a single directory to search or an explicit list of files.
    A file is matched to a variable by the ``.{VAR}.`` token in its name, and
    failing that by whether the variable is a data variable inside it.
    """
    if len(paths) == 1 and os.path.isdir(paths[0]):
        candidates = sorted(glob.glob(os.path.join(paths[0], "*.nc")))
        if not candidates:
            raise SystemExit(f"no .nc files in {paths[0]}")
    else:
        candidates = list(paths)
        for path in candidates:
            if not os.path.isfile(path):
                raise SystemExit(f"not found: {path}")

    found: Dict[str, str] = {}
    for var in variables:
        for path in candidates:
            if f".{var}." in os.path.basename(path):
                found[var] = path
                break
    missing = [v for v in variables if v not in found]
    if missing:
        raise SystemExit(
            f"no file found for {', '.join(missing)} among "
            f"{len(candidates)} candidate(s). Expected a '.{missing[0]}.' token "
            f"in the filename."
        )
    return found


def build_figure(inputs: Dict[str, str], timestep: int):
    """Draw the panels. Returns ``(figure, header)``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import xarray as xr
    from matplotlib.ticker import FixedLocator, NullFormatter

    panels = [p for p in PANELS if p[0] in inputs]
    fig, axes = plt.subplots(
        len(panels), 1, figsize=(8.5, 4.35 * len(panels)), constrained_layout=True
    )
    if len(panels) == 1:
        axes = [axes]

    header: Dict[str, str] = {}
    for ax, (var, long_name, units, cmap, diverging) in zip(axes, panels):
        path = inputs[var]
        ds = xr.open_dataset(path, decode_times=True)

        if var not in ds:
            raise SystemExit(f"{path} has no variable {var} (found {list(ds.data_vars)})")
        if timestep >= ds.sizes.get("time", 0):
            raise SystemExit(
                f"{path} has {ds.sizes.get('time', 0)} timestep(s); "
                f"cannot select index {timestep}"
            )

        if not header:
            header["case"] = os.path.basename(path).split(f".{var}.")[0]
            header["time"] = str(ds["time"].values[timestep])
            header["nlat"] = str(ds.sizes.get("lat", "?"))
            header["nlon"] = str(ds.sizes.get("lon", "?"))
            header["nlev"] = str(ds.sizes.get("plev", "?"))
            for key in PROVENANCE_ATTRS:
                header[key] = ds.attrs.get(key, "?")

        da = ds[var].isel(time=timestep)
        # skipna=True: below-ground cells are NaN by design (vrt_xtr=mss_val),
        # so each level averages the longitudes that actually exist there.
        zm = da.mean(dim="lon", skipna=True)
        valid_pct = float(da.notnull().mean(dim=("lat", "lon")).values[0] * 100.0)

        lat = zm["lat"].values
        plev_hpa = zm["plev"].values / 100.0
        field = zm.values

        if diverging:
            lim = float(np.nanmax(np.abs(field)))
            levels = np.linspace(-lim, lim, 25)
        else:
            levels = np.linspace(float(np.nanmin(field)), float(np.nanmax(field)), 25)

        cf = ax.contourf(lat, plev_hpa, field, levels=levels, cmap=cmap, extend="both")
        cs = ax.contour(lat, plev_hpa, field, levels=levels[::4], colors="k",
                        linewidths=0.4, alpha=0.55)
        ax.clabel(cs, fmt="%.0f", fontsize=6, inline=True)

        # Log pressure: the levels are uniform in pseudoheight,
        # z = -H ln(p/p0), which is linear in log(p). On a linear pressure axis
        # roughly three quarters of them pile into the bottom fifth.
        ax.set_yscale("log")
        ax.set_ylim(plev_hpa.max(), plev_hpa.min())   # highest pressure at bottom
        ax.yaxis.set_major_locator(FixedLocator(PRESSURE_TICKS))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_yticklabels([str(t) for t in PRESSURE_TICKS])
        ax.set_ylabel("pressure (hPa)")

        ax.set_xlim(-90, 90)
        ax.set_xticks(np.arange(-90, 91, 30))
        ax.set_xticklabels(["90S", "60S", "30S", "EQ", "30N", "60N", "90N"])
        ax.set_xlabel("latitude")
        ax.grid(alpha=0.15, linewidth=0.4)

        # Right-hand axis in pseudoheight, the coordinate falwa works in.
        rax = ax.twinx()
        rax.set_yscale("log")
        rax.set_ylim(ax.get_ylim())
        rax.yaxis.set_major_locator(FixedLocator([
            P_GROUND_PA * np.exp(-z * 1000.0 / SCALE_HEIGHT_M) / 100.0
            for z in PSEUDOHEIGHT_TICKS
        ]))
        rax.set_yticklabels([str(z) for z in PSEUDOHEIGHT_TICKS])
        rax.yaxis.set_minor_formatter(NullFormatter())
        rax.set_ylabel("pseudoheight (km)")

        cb = fig.colorbar(cf, ax=[ax, rax], pad=0.09, aspect=28)
        cb.set_label(f"{var} ({units})" if units else var)

        plain_units = units.replace("$", "").replace("^{-1}", "-1")
        ax.set_title(
            f"{var} — {long_name}   "
            f"[{np.nanmin(field):.1f}, {np.nanmax(field):.1f}] {plain_units}   "
            f"({valid_pct:.0f}% of the {plev_hpa[0]:.0f} hPa level above ground)",
            fontsize=10, loc="left",
        )
        ds.close()

    fig.suptitle(
        f"Zonal-mean cross-sections, timestep {timestep}\n"
        f"{header['case']}   {header['time']}",
        fontsize=13,
    )
    return fig, header


def caption_for(header: Dict[str, str]) -> str:
    """Self-describing footer, so a forwarded PDF still says how it was made."""
    return (
        f"Regridded to {header['nlat']}x{header['nlon']} cell-centred lat-lon on "
        f"{header['nlev']} pressure levels evenly spaced in pseudoheight.\n"
        f"Source stream {header['regrid_source_stream']} "
        f"({header['regrid_source_file']}); horizontal remap "
        f"{header['regrid_algo']}; below-ground extrapolation "
        f"{header['regrid_vrt_xtr']} (cells below the surface are left missing, "
        f"not fabricated).\n"
        f"Zonal means skip missing values, so the lowest levels average fewer "
        f"longitudes than the levels above them."
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Needs matplotlib, xarray and netCDF4.",
    )
    parser.add_argument(
        "inputs", nargs="+", metavar="PATH",
        help="a directory of regridded files, or explicit file paths",
    )
    parser.add_argument(
        "-o", "--output",
        help="output PDF (default: zonal_mean_check.pdf beside the input)",
    )
    parser.add_argument(
        "-t", "--timestep", type=int, default=0,
        help="time index to plot (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--variables", nargs="+", default=[p[0] for p in PANELS],
        help="variables to plot (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            f"plotting needs matplotlib, xarray and netCDF4, which are not "
            f"importable here ({exc}). Run this from an environment that has "
            f"them, e.g. the POD's own runtime environment."
        )

    inputs = resolve_inputs(args.inputs, args.variables)

    output = args.output
    if output is None:
        base = args.inputs[0] if os.path.isdir(args.inputs[0]) \
            else os.path.dirname(os.path.abspath(args.inputs[0]))
        output = os.path.join(base, "zonal_mean_check.pdf")

    fig, header = build_figure(inputs, args.timestep)
    fig.text(0.01, -0.012, caption_for(header), fontsize=7.0,
             va="top", ha="left", wrap=True)

    os.makedirs(os.path.dirname(os.path.abspath(output)) or ".", exist_ok=True)
    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        info = pdf.infodict()
        info["Title"] = "Regridding check: zonal-mean cross-sections"
        info["Subject"] = header["case"]
    plt.close(fig)

    print(f"wrote {output} ({os.path.getsize(output):,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
