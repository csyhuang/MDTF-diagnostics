"""Per-variable regridding, chunk by chunk.

Vertical interpolation runs *before* the horizontal remap so the hybrid
coefficients are applied on the grid they were defined on, and so pressure is
never derived from remapped topography. The intermediate is larger this way;
chunking caps it.

(ADF makes the opposite choice -- horizontal first, carrying PS to the target
grid -- which is defensible for monthly climatologies but interpolates along
terrain-following surfaces across topography. For 6-hourly instantaneous fields
over ne120 orography, vertical-first is the more conservative order.)
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

from .config import RegridConfig
from .levels import levels_above_model_top
from .nco import (
    NCOError,
    global_attribute,
    n_time,
    read_values,
    require_tools,
    run,
)

log = logging.getLogger(__name__)

#: Variables that exist only to describe the hybrid-sigma coordinate. Dropped
#: once the data is on pressure levels.
_HYBRID_VARS = ("hyam", "hybm", "hyai", "hybi", "ilev", "P0", "PS")

#: Artifacts ncremap adds during the horizontal remap.
_REMAP_ARTIFACTS = ("area", "gw", "lat_bnds", "lon_bnds")

#: Global attribute recording which history stream a chunk came from.
_STREAM_ATTR = "regrid_source_stream"


def chunk_bounds(n_steps: int, chunk: int) -> List[Tuple[int, int]]:
    """Inclusive ``(first, last)`` time indices for each chunk."""
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    bounds = []
    for i0 in range(0, n_steps, chunk):
        bounds.append((i0, min(i0 + chunk - 1, n_steps - 1)))
    return bounds


def regrid_variable(cfg: RegridConfig, var: str) -> List[str]:
    """Regrid one variable end to end. Returns the output paths.

    Completed chunks are never redone, so a killed run resumes where it left
    off. Files are published with an atomic rename, so an interrupted chunk can
    never be mistaken for a finished one.
    """
    require_tools("ncremap", "ncks", "ncap2", "ncrename", "ncatted", "ncdump")

    src = cfg.source_file(var)
    if not os.path.isfile(src):
        raise NCOError(f"input not found: {src}")
    if not os.path.isfile(cfg.ps_file):
        raise NCOError(f"PS not found: {cfg.ps_file}")
    # Under --dry-run these are advisory: the whole point of a preview is to
    # inspect the plan before committing to setup, which takes minutes and
    # writes hundreds of MB. A real run still refuses to start without them.
    for artifact in (cfg.map_file, cfg.vrt_file):
        if os.path.isfile(artifact):
            continue
        if cfg.dry_run:
            log.warning("%s missing -- `setup` has not been run yet", artifact)
        else:
            raise NCOError(f"{artifact} missing -- run `setup` first")

    if not cfg.dry_run:
        os.makedirs(cfg.out_dir, exist_ok=True)
        os.makedirs(cfg.work_dir, exist_ok=True)

    n_steps = _check_time_alignment(cfg, var, src)
    _warn_levels_above_model_top(cfg, src)

    bounds = chunk_bounds(n_steps, cfg.chunk)
    log.info("%s: %d timesteps, %d chunk(s) of %d", var, n_steps, len(bounds), cfg.chunk)

    outputs = []
    for i0, i1 in bounds:
        out = cfg.output_file(var, i0, i1)
        outputs.append(out)
        if _is_complete(cfg, out, var, i0, i1):
            continue
        _regrid_chunk(cfg, var, src, i0, i1, out)
        log.info("%s chunk %06d-%06d -> %s", var, i0, i1, out)

    log.info("%s complete", var)
    return outputs


# ---------------------------------------------------------------------------
# Restart logic
# ---------------------------------------------------------------------------

def _is_complete(cfg: RegridConfig, out: str, var: str, i0: int, i1: int) -> bool:
    """True if *out* is a finished chunk from the stream we are asked for.

    A chunk left over from a different history stream is not a valid restart
    point: h7i and h8a share the time coordinate, so the file looks entirely
    plausible while holding instantaneous data where means were wanted, or the
    reverse. Rebuild it rather than trusting the name.
    """
    if not (os.path.isfile(out) and os.path.getsize(out) > 0):
        return False

    stream = global_attribute(out, _STREAM_ATTR)
    if stream is None:
        log.warning(
            "%s chunk %06d-%06d: existing file has no %s attribute (written by "
            "an older version?) -- rebuilding to be sure of its provenance",
            var, i0, i1, _STREAM_ATTR,
        )
        return False
    if stream != cfg.stream:
        log.warning(
            "%s chunk %06d-%06d: existing file came from stream %s but %s was "
            "requested -- rebuilding",
            var, i0, i1, stream, cfg.stream,
        )
        return False

    log.info("%s chunk %06d-%06d: exists (stream %s), skipping", var, i0, i1, stream)
    return True


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _check_time_alignment(cfg: RegridConfig, var: str, src: str) -> int:
    """Confirm the variable and PS have the same number of timesteps.

    h7i and h8a share the time coordinate exactly, so index-aligned slicing
    across streams is safe -- but only if the step counts agree. A mismatch
    means the PS slice appended to each chunk would be for the wrong times,
    which would corrupt the vertical interpolation without any error.

    This reads the files even under --dry-run: reading is free, and a dry run
    that invented its own step count would print chunk boundaries that do not
    match what a real run would do, which defeats the point of previewing.
    """
    nt = n_time(src)
    nt_ps = n_time(cfg.ps_file)
    if nt != nt_ps:
        raise NCOError(
            f"time mismatch: {var} has {nt} steps but PS has {nt_ps}. "
            f"Index-aligned slicing would pair the wrong times."
        )
    return nt


def _warn_levels_above_model_top(cfg: RegridConfig, src: str) -> None:
    """Warn about target levels with no data above them.

    ``hybm[0] = 0`` makes the top level pure pressure, so ``lev[0]`` is the
    model top exactly, everywhere, independent of PS. Levels above it come back
    100% missing and then propagate NaN into any column-wise diagnostic.

    Runs under --dry-run too, so a preview surfaces this before you commit
    hours to a vertical grid that reaches above the data.
    """
    try:
        top_hpa = read_values(src, "lev", "lev", 0, 0, fmt="%f")[0]
    except NCOError:
        log.warning("could not read lev[0]; skipping the model-top check")
        return

    above = levels_above_model_top(cfg.levels, top_hpa)
    if above:
        log.warning(
            "model top is %.4f hPa but these target levels are above it: %s hPa "
            "-- they will be entirely missing",
            top_hpa,
            " ".join(f"{p / 100.0:.4g}" for p in above),
        )
    else:
        log.info("all target levels are at or below the model top (%.4f hPa)", top_hpa)


# ---------------------------------------------------------------------------
# The pipeline itself
# ---------------------------------------------------------------------------

def _regrid_chunk(
    cfg: RegridConfig, var: str, src: str, i0: int, i1: int, out: str
) -> None:
    """Run the eight-step pipeline for a single chunk."""
    log.info("%s chunk %06d-%06d", var, i0, i1)
    stem = os.path.join(cfg.work_dir, f"{var}.{i0:06d}-{i1:06d}")
    sub, p0, vrt, thin, hrz, cln = (
        f"{stem}.sub.nc", f"{stem}.p0.nc", f"{stem}.vrt.nc",
        f"{stem}.thin.nc", f"{stem}.hrz.nc", f"{stem}.cln.nc",
    )

    try:
        # 1. Time subset of the 3-D field. Brings hyam/hybm/hyai/hybi/lev along.
        run(["ncks", "-O", "-h", "-d", f"time,{i0},{i1}", src, sub], cfg.dry_run)

        # 2. Append the matching PS slice. PS lives in its own file and carries
        #    no hybrid coefficients, so it must be merged into the 3-D file
        #    rather than the other way round.
        run(["ncks", "-A", "-h", "-d", f"time,{i0},{i1}", "-v", "PS",
             cfg.ps_file, sub], cfg.dry_run)

        # 3. P0 is absent from every file in this dataset. lev = 1000*(A+B) in
        #    hPa implies P0 = 100000 Pa.
        run(["ncap2", "-O", "-h", "-s",
             'P0=100000.0; P0@units="Pa"; P0@long_name="reference pressure"',
             sub, p0], cfg.dry_run)

        # 4. Hybrid-sigma -> pressure, still on the native mesh.
        run(["ncremap",
             f"--vrt_out={cfg.vrt_file}",
             f"--vrt_ntp={cfg.vrt_ntp}",
             f"--vrt_xtr={cfg.vrt_xtr}",
             "--ps_nm=PS", "--vrt_nm=plev",
             "-i", p0, "-o", vrt], cfg.dry_run)

        # 5. Drop the hybrid machinery now that we are on pressure levels.
        #    Doing this before the horizontal remap avoids regridding PS
        #    pointlessly, and sidesteps an NCO warning about PS carrying a NaN
        #    _FillValue, which cannot be compared arithmetically. The ^(...)$
        #    regex form means absent variables are skipped rather than erroring.
        run(["ncks", "-O", "-h", "-x", "-v",
             f"^({'|'.join(_HYBRID_VARS)})$", vrt, thin], cfg.dry_run)

        # 6. Unstructured -> lat-lon, with renormalization. See RegridConfig.rnr_thr:
        #    this flag is not optional for this dataset.
        run(["ncremap", "-m", cfg.map_file, "-r", str(cfg.rnr_thr),
             "-i", thin, "-o", hrz], cfg.dry_run)

        # 7. Drop the artifacts ncremap adds, leaving exactly one data variable
        #    plus time bounds -- what the catalog builder needs to identify
        #    variable_id unambiguously. cell_measures is deleted too, or it
        #    would dangle a reference to the removed 'area'.
        #    -C is required: lat_bnds/lon_bnds are CF-associated with lat/lon,
        #    and without it NCO re-adds them to the extraction list and then
        #    refuses the exclusion outright.
        run(["ncks", "-O", "-h", "-C", "-x", "-v",
             f"^({'|'.join(_REMAP_ARTIFACTS)})$", hrz, cln], cfg.dry_run)
        run(["ncatted", "-O", "-h", "-a", "cell_measures,,d,,", cln], cfg.dry_run)

        # 8. time_bounds -> time_bnds. The catalog builder's exclusion list
        #    (tools/catalog_builder/parsers.py) contains 'time_bnds' but not
        #    'time_bounds', so without this rename it would pick the bounds
        #    variable as variable_id. The leading '.' marks the rename optional,
        #    making this a no-op when the variable is already correctly named.
        run(["ncrename", "-O", "-h", "-v", ".time_bounds,time_bnds", cln], cfg.dry_run)
        run(["ncatted", "-O", "-h", "-a", "bounds,time,o,c,time_bnds", cln], cfg.dry_run)

        # 9. Stamp the provenance the output filename does not carry. Names are
        #    {case}.{var}.{chunk}.nc with no stream token, so without this a
        #    chunk regridded from h8a and one from h7i are indistinguishable on
        #    disk -- and the skip-if-exists check below would silently accept
        #    the wrong one when you switch streams.
        run(["ncatted", "-O", "-h",
             "-a", f"{_STREAM_ATTR},global,o,c,{cfg.stream}",
             "-a", f"regrid_source_file,global,o,c,{os.path.basename(src)}",
             "-a", f"regrid_vrt_xtr,global,o,c,{cfg.vrt_xtr}",
             "-a", f"regrid_algo,global,o,c,{cfg.algo}",
             cln], cfg.dry_run)

        # 10. Publish atomically. os.replace is atomic within a filesystem, so a
        #    killed job never leaves a partial file that the skip-if-exists
        #    check would mistake for a finished one.
        if cfg.dry_run:
            print(f"  + mv {cln} {out}")
        else:
            os.replace(cln, out)
    finally:
        _cleanup(cfg, [sub, p0, vrt, thin, hrz, cln])


def _cleanup(cfg: RegridConfig, paths: List[str]) -> None:
    """Remove intermediates. Runs even on failure, so a crash leaves no debris.

    Intermediates for one chunk of a 3-D field are tens of GiB, so leaving them
    behind after an error would fill the disk before anyone noticed.
    """
    if cfg.dry_run:
        return
    for path in paths:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except OSError as exc:
            log.warning("could not remove %s: %s", path, exc)
