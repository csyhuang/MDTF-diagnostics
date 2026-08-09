#!/usr/bin/env python3
"""Plot the ERA5 digest for inspection before it is packaged and shipped.

The digest is checked by ``make_era5_digest.py`` for shape and finiteness, but
no assertion catches a field that is structurally perfect and physically wrong
-- an inverted column, a flipped hemisphere, a season mislabelled. Those are
obvious in a picture and invisible in a summary.

Seasons are laid out **side by side** rather than one page each, because the
most informative check is the contrast between them: in DJF the northern
subtropical jet should dominate, in JJA the southern one. A latitude flip
survives every numerical test and reverses exactly that.

    python plot_era5_digest.py <digest-dir> -o digest_check.pdf

Needs matplotlib, cartopy, xarray. Run from the POD's runtime environment.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional

SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]

#: (variable, kind, colormap, diverging, label)
PANELS = [
    ("zonal_mean_u",          "yz", "RdBu_r", True,  "zonal-mean U"),
    ("uref",                  "yz", "RdBu_r", True,  "reference state Uref"),
    ("zonal_mean_lwa",        "yz", "viridis", False, "zonal-mean LWA"),
    ("u_baro",                "xy", "RdBu_r", True,  "barotropic U"),
    ("lwa_baro",              "xy", "viridis", False, "barotropic LWA"),
    ("covariance_lwa_u_baro", "xy", "PuOr_r", True,  "cov(LWA, U) barotropic"),
]


def load_digest(digest_dir: str) -> Dict[str, "xr.Dataset"]:
    import xarray as xr
    found = {}
    for season in SEASON_ORDER:
        matches = glob.glob(os.path.join(
            digest_dir, f"*climatology_{season}.nc"))
        if matches:
            found[season] = xr.open_dataset(matches[0])
    if not found:
        raise SystemExit(f"no digest files in {digest_dir}")
    return found


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("digest_dir")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--sigma", action="store_true",
                        help="plot the interannual sigma instead of the mean")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from cartopy import crs as ccrs
    from matplotlib.backends.backend_pdf import PdfPages

    digest = load_digest(args.digest_dir)
    seasons = [s for s in SEASON_ORDER if s in digest]
    suffix = "_sigma" if args.sigma else ""
    output = args.output or os.path.join(
        args.digest_dir, f"digest_check{suffix}.pdf")

    n_rows, n_cols = len(PANELS), len(seasons)
    fig = plt.figure(figsize=(5.2 * n_cols + 1.2, 3.1 * n_rows))
    fig.suptitle(
        f"ERA5 digest {'interannual sigma' if args.sigma else 'climatology'}"
        f"  --  {', '.join(seasons)}", fontsize=14, y=0.997)

    for row, (name, kind, cmap, diverging, label) in enumerate(PANELS):
        variable = name + suffix
        # Common scale across seasons, so the comparison between them is
        # meaningful rather than each panel being self-normalised.
        blocks = [digest[s][variable].values for s in seasons
                  if variable in digest[s]]
        if not blocks:
            continue
        finite = np.concatenate([b[np.isfinite(b)].ravel() for b in blocks])
        if diverging and not args.sigma:
            limit = np.percentile(np.abs(finite), 99.5)
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = 0.0 if not diverging else np.percentile(finite, 0.5), \
                np.percentile(finite, 99.5)

        for col, season in enumerate(seasons):
            dataset = digest[season]
            if variable not in dataset:
                continue
            index = row * n_cols + col + 1
            values = dataset[variable].values

            if kind == "yz":
                ax = fig.add_subplot(n_rows, n_cols, index)
                y = dataset["height"].values / 1000.0
                mesh = ax.contourf(dataset["lat"].values, y, values,
                                   levels=21, cmap=cmap, vmin=vmin, vmax=vmax,
                                   extend="both")
                if diverging and not args.sigma:
                    ax.contour(dataset["lat"].values, y, values, levels=[0],
                               colors="k", linewidths=0.7)
                ax.set_xlim(-80, 80)
                ax.set_xticks(np.arange(-80, 81, 40))
                ax.set_xticklabels(["80S", "40S", "EQ", "40N", "80N"])
                ax.set_ylabel("pseudoheight (km)" if col == 0 else "")
            else:
                ax = fig.add_subplot(n_rows, n_cols, index,
                                     projection=ccrs.PlateCarree())
                ax.coastlines(color="black", alpha=0.6, linewidth=0.5)
                ax.set_aspect("auto")
                mesh = ax.contourf(dataset["lon"].values, dataset["lat"].values,
                                   values, levels=21, cmap=cmap,
                                   vmin=vmin, vmax=vmax, extend="both",
                                   transform=ccrs.PlateCarree())
                ax.set_xticks(np.arange(-180, 181, 90), crs=ccrs.PlateCarree())
                ax.set_yticks(np.arange(-90, 91, 45), crs=ccrs.PlateCarree())

            fig.colorbar(mesh, ax=ax, pad=0.02, aspect=18)
            n_years = dataset.attrs.get("n_years", "?")
            ax.set_title(f"{season}  {label}"
                         + (f"  ({n_years} yr)" if col == 0 else ""),
                         fontsize=9)

    fig.tight_layout(rect=[0, 0.01, 1, 0.985])

    first = digest[seasons[0]]
    caption = (
        f"Source: {first.attrs.get('source', '?')} | "
        f"years {first.attrs.get('start_year','?')}-{first.attrs.get('end_year','?')} "
        f"({first.attrs.get('n_years','?')}) | "
        f"kmax {first.attrs.get('kmax','?')}, dz {first.attrs.get('dz','?')} m | "
        f"falwa {first.attrs.get('falwa_version','?')}\n"
        f"What to look for: DJF should show the stronger jet and larger LWA in "
        f"the northern hemisphere, JJA in the southern. A latitude flip passes "
        f"every numerical check and reverses exactly this.\n"
        f"Barotropic LWA should be positive-definite and largest in the "
        f"storm-track latitudes; cov(LWA, U) predominantly negative where wave "
        f"activity decelerates the flow."
    )
    fig.text(0.005, 0.002, caption, fontsize=7.5, va="bottom", ha="left")

    with PdfPages(output) as pdf:
        pdf.savefig(fig, bbox_inches="tight")
        pdf.infodict()["Title"] = "ERA5 finite-amplitude wave activity digest check"
    plt.close(fig)
    for dataset in digest.values():
        dataset.close()

    print(f"wrote {output} ({os.path.getsize(output) / 1024**2:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
