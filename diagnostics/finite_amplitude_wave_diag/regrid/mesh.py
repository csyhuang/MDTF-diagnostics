"""One-time setup: source mesh, target grid, remap weights, vertical grid.

All four artifacts are reused by every variable and every chunk, so this runs
once per (grid, algorithm) combination. Each step is skipped if its output
already exists; delete the file to force a rebuild.
"""

from __future__ import annotations

import logging
import os
import shutil

from .config import RegridConfig
from .levels import format_levels_cdl
from .nco import (
    NCOError,
    read_values,
    require_tools,
    run,
    run_tolerant,
    scrip_grid_size,
)

log = logging.getLogger(__name__)

#: Cell-centre agreement tolerance in degrees. TempestRemap and CAM use
#: slightly different cell-centre conventions, leaving a residual of ~8e-4 deg
#: (~80 m on 28 km cells) even when the mesh is correct. A genuinely wrong
#: orientation or element ordering is off by whole degrees, so this separates
#: the two cleanly. Conservative remapping uses cell corners, not centres, so
#: the residual does not affect the result.
MESH_LATITUDE_TOL_DEG = 0.01

#: Number of cells compared in the mesh-vs-data coordinate check.
MESH_CHECK_NCELLS = 200


def setup(cfg: RegridConfig) -> None:
    """Build every artifact the per-variable pipeline depends on."""
    require_tools("ncremap", "ncgen", "ncdump", "GenerateCSMesh", "GenerateVolumetricMesh")

    if not cfg.dry_run:
        os.makedirs(cfg.work_dir, exist_ok=True)
        os.makedirs(cfg.out_dir, exist_ok=True)

    _build_source_mesh(cfg)
    _check_source_mesh(cfg)
    _build_target_grid(cfg)
    _build_weights(cfg)
    _build_vertical_grid(cfg)
    log.info("setup complete")


# ---------------------------------------------------------------------------
# Source mesh
# ---------------------------------------------------------------------------

def _build_source_mesh(cfg: RegridConfig) -> None:
    """Generate the ne120pg3 SCRIP mesh.

    The data is on the pg3 *physics* grid, not np4: ncol = 6*120^2*3^2 = 777600
    (np4 would be 777602) and the files carry ``fv_nphys = 3``. Generating the
    mesh is more reliable than hunting for a shipped SCRIP file, and it is why
    the ``ne120np4_pentagons`` file named in the circulated remapping notes is
    the wrong source for this dataset.
    """
    if os.path.isfile(cfg.scrip_file):
        log.info("reusing %s", cfg.scrip_file)
        return

    log.info("generating ne120pg3 SCRIP mesh")
    run_tolerant(
        cfg.exodus_cs_file,
        ["GenerateCSMesh", "--res", "120", "--alt", "--file", cfg.exodus_cs_file],
        cfg.dry_run,
    )
    run_tolerant(
        cfg.exodus_pg_file,
        ["GenerateVolumetricMesh",
         "--in", cfg.exodus_cs_file,
         "--out", cfg.exodus_pg_file,
         "--np", "3", "--uniform"],
        cfg.dry_run,
    )
    # The subcommand name varies by tempest-remap version.
    converter = (
        "ConvertMeshToSCRIP"
        if shutil.which("ConvertMeshToSCRIP")
        else "ConvertExodusToSCRIP"
    )
    run_tolerant(
        cfg.scrip_file,
        [converter, "--in", cfg.exodus_pg_file, "--out", cfg.scrip_file],
        cfg.dry_run,
    )


def _check_source_mesh(cfg: RegridConfig) -> None:
    """Verify the mesh actually matches the data.

    Two checks, in order of importance:

    1. ``grid_size`` against the data's ``ncol``. If these differ, nothing
       downstream is meaningful.
    2. Cell centres against the data's own ``lat``. Cell *count* alone does not
       prove a match -- cube-sphere orientation (``--alt``) and element
       ordering both change coordinates while leaving the count identical. A
       mismatch here means every remapped field would be silently scrambled,
       which is the failure mode most likely to survive to publication.
    """
    if cfg.dry_run or not os.path.isfile(cfg.scrip_file):
        return

    size = scrip_grid_size(cfg.scrip_file)
    if size != cfg.ncol_expected:
        raise NCOError(
            f"mesh grid_size={size} but the data has ncol={cfg.ncol_expected} "
            f"-- wrong mesh. Delete {cfg.scrip_file} and check the --res and "
            f"--np arguments."
        )
    log.info("mesh grid_size = %d, matches data", size)

    if not os.path.isfile(cfg.ps_file):
        log.warning(
            "%s not found, skipping the mesh-vs-data coordinate check", cfg.ps_file
        )
        return

    n = MESH_CHECK_NCELLS
    try:
        mesh_lat = read_values(
            cfg.scrip_file, "grid_center_lat", "grid_size", 0, n - 1, fmt="%.6f"
        )
        data_lat = read_values(cfg.ps_file, "lat", "ncol", 0, n - 1, fmt="%.6f")
    except NCOError as exc:
        log.warning("could not run the mesh-vs-data coordinate check: %s", exc)
        return

    pairs = min(len(mesh_lat), len(data_lat))
    if pairs == 0:
        log.warning("mesh-vs-data coordinate check read no values, skipping")
        return

    max_diff = max(abs(mesh_lat[i] - data_lat[i]) for i in range(pairs))
    log.info(
        "mesh vs data latitude, first %d cells: max |diff| = %.6f deg",
        pairs, max_diff,
    )
    if max_diff > MESH_LATITUDE_TOL_DEG:
        raise NCOError(
            f"mesh cell centres do not match the data's lat (max |diff| = "
            f"{max_diff:.4f} deg > {MESH_LATITUDE_TOL_DEG} deg) -- wrong "
            f"orientation or ordering.\n"
            f"Try toggling the --alt flag on GenerateCSMesh, then delete "
            f"{cfg.scrip_file} and rerun setup."
        )
    log.info("mesh cell centres match the data")


# ---------------------------------------------------------------------------
# Target grid and weights
# ---------------------------------------------------------------------------

def _build_target_grid(cfg: RegridConfig) -> None:
    """Generate the destination lat-lon grid.

    This is a *cell-centred* grid: 181 latitudes from -89.503 to 89.503 at
    0.9945 deg spacing, with no row exactly at either pole. That does not match
    the POD driver's ``ylat = np.arange(-90, 91, 1.0)`` point for point; falwa
    interpolates onto its own analysis grid, so the offset is absorbed there.
    If you ever need centres exactly on -90..90 (e.g. to compare against a CAM
    FV grid), append ``#lat_typ=cap`` to the -G argument and regenerate the
    weights -- the map file depends on the destination grid.
    """
    if os.path.isfile(cfg.dst_grid_file):
        log.info("reusing %s", cfg.dst_grid_file)
        return
    log.info("generating %dx%d target grid", cfg.nlat, cfg.nlon)
    run(
        ["ncremap", "-g", cfg.dst_grid_file, "-G", f"latlon={cfg.nlat},{cfg.nlon}"],
        cfg.dry_run,
    )


def _build_weights(cfg: RegridConfig) -> None:
    """Generate the remap weights. The expensive artifact; built once."""
    if os.path.isfile(cfg.map_file):
        log.info("reusing %s", cfg.map_file)
        return
    log.info("generating %s weights (slow)", cfg.algo)
    run(
        ["ncremap",
         "-a", cfg.algo,
         "-s", cfg.scrip_file,
         "-g", cfg.dst_grid_file,
         "-m", cfg.map_file],
        cfg.dry_run,
    )


# ---------------------------------------------------------------------------
# Vertical grid
# ---------------------------------------------------------------------------

_VRT_CDL = """netcdf vrt_plev {{
dimensions:
    plev = {nlev} ;
variables:
    double plev(plev) ;
        plev:units = "Pa" ;
        plev:long_name = "pressure" ;
        plev:axis = "Z" ;
        plev:positive = "down" ;
data:
    plev = {levels} ;
}}
"""


def _build_vertical_grid(cfg: RegridConfig) -> None:
    """Write the target pressure grid via ncgen, so no template file is needed."""
    if os.path.isfile(cfg.vrt_file):
        log.info("reusing %s", cfg.vrt_file)
        return

    levels = cfg.levels
    cdl_path = os.path.join(cfg.work_dir, "vrt_plev.cdl")
    log.info("generating vertical grid: %d levels", len(levels))

    if cfg.dry_run:
        print(f"  + write {cdl_path} ({len(levels)} levels)")
        print(f"  + ncgen -o {cfg.vrt_file} {cdl_path}")
        return

    with open(cdl_path, "w") as fh:
        fh.write(_VRT_CDL.format(nlev=len(levels), levels=format_levels_cdl(levels)))
    run(["ncgen", "-o", cfg.vrt_file, cdl_path], cfg.dry_run)
