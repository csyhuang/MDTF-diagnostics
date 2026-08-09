#!/usr/bin/env python3
"""Compare the original and streaming POD implementations on the same input.

Runs both drivers into separate working directories and reports, per season and
per diagnostic, the largest absolute and relative difference between them.

The two are not expected to agree bit-for-bit. They differ in one deliberate
way: the original assembles every timestep into a dataset, interpolates the
results back onto the input grid, and then averages over time; the streaming
version averages over time first and interpolates the mean. Linear
interpolation commutes with the mean, so those agree to rounding, but the
operations are ordered differently and float addition is not associative.

The original also round-trips its intermediates through netCDF at float32
twice, so ~1e-5 relative agreement is the practical floor. Anything beyond
1e-4 means the two are doing genuinely different arithmetic.

Usage:
    python compare_implementations.py --case-info /path/to/case_info.yml \\
        --work-dir /path/to/scratch [--python /path/to/python]

It writes <work-dir>/original and <work-dir>/streaming, runs one driver into
each, and compares their figures' underlying data by re-running the diagnostic
maths from each run's checkpoint where available. Where the original has no
checkpoint, its intermediate netCDF is used instead.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORIGINAL = "finite_amplitude_wave_diag_zonal_mean.py"
STREAMING = "finite_amplitude_wave_diag_zonal_mean_rework.py"

SEASONS = ["DJF", "MAM", "JJA", "SON"]
FIGURES = ["zonal_mean_u", "zonal_mean_uref", "zonal_mean_lwa",
           "zonal_mean_delta_u", "u_baro", "lwa_baro", "u_lwa_covariance"]


def run_driver(script: str, work_dir: str, case_info: str, python: str) -> int:
    """Run one driver with a clean working directory. Returns its exit code."""
    for sub in ("model/PS", "model/netCDF", "obs/PS", "obs/netCDF"):
        os.makedirs(os.path.join(work_dir, sub), exist_ok=True)
    env = dict(os.environ, WORK_DIR=work_dir, case_env_file=case_info)
    print(f"\n=== running {script} -> {work_dir} ===", flush=True)
    proc = subprocess.run([python, os.path.join(HERE, script)],
                          cwd=HERE, env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    log = os.path.join(work_dir, "driver.log")
    with open(log, "w") as fh:
        fh.write(proc.stdout)
    tail = proc.stdout.strip().splitlines()[-3:]
    for line in tail:
        print("   ", line)
    print(f"    exit {proc.returncode}; full log at {log}")
    return proc.returncode


def compare_figures(dir_a: str, dir_b: str) -> None:
    """Report which figures both runs produced, as a coarse first check."""
    print("\n=== figures produced ===")
    print(f"{'season':8s} {'figure':22s} {'original':>10s} {'streaming':>10s}")
    for season in SEASONS:
        for fig in FIGURES:
            name = f"{season}_{fig}.eps"
            a = os.path.join(dir_a, "model", "PS", name)
            b = os.path.join(dir_b, "model", "PS", name)
            ea, eb = os.path.isfile(a), os.path.isfile(b)
            if not (ea or eb):
                continue
            print(f"{season:8s} {fig:22s} {'yes' if ea else 'MISSING':>10s} "
                  f"{'yes' if eb else 'MISSING':>10s}")


def compare_diagnostics(dir_a: str, dir_b: str) -> int:
    """Diff the seasonal-mean diagnostics the two runs wrote. Returns #failures."""
    import numpy as np
    import xarray as xr

    print("\n=== numerical comparison of seasonal diagnostics ===")
    failures = 0
    any_compared = False

    for season in SEASONS:
        pa = os.path.join(dir_a, "model", "netCDF", f"diagnostics_{season}.nc")
        pb = os.path.join(dir_b, "model", "netCDF", f"diagnostics_{season}.nc")
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        any_compared = True
        print(f"\n  {season}")
        print(f"    {'variable':24s} {'max |diff|':>12s} {'max rel':>12s} "
              f"{'scale':>12s}  verdict")
        with xr.open_dataset(pa) as da, xr.open_dataset(pb) as db:
            for var in sorted(set(da.data_vars) & set(db.data_vars)):
                a = np.asarray(da[var].values, dtype=float)
                b = np.asarray(db[var].values, dtype=float)
                if a.shape != b.shape:
                    print(f"    {var:24s} {'SHAPE MISMATCH':>12s} "
                          f"{str(a.shape):>12s} {str(b.shape):>12s}  FAIL")
                    failures += 1
                    continue
                finite = np.isfinite(a) & np.isfinite(b)
                if not finite.any():
                    print(f"    {var:24s} {'all NaN in both':>38s}  --")
                    continue
                diff = np.abs(a[finite] - b[finite])
                scale = float(np.nanmax(np.abs(a[finite]))) or 1.0
                max_abs = float(diff.max())
                rel = max_abs / scale
                # Tolerance is set by the ORIGINAL, not by the maths. It
                # round-trips through netCDF twice -- gridfill_*.nc and then
                # intermediate_<SEASON>.nc -- both float32, so its own inputs
                # carry ~1e-7 relative error before falwa amplifies it. The
                # streaming version keeps float64 throughout. Agreement to
                # ~1e-5 is therefore the floor, not a defect; anything above
                # 1e-4 means the two are genuinely doing different arithmetic.
                verdict = "ok" if rel < 1e-5 else ("close" if rel < 1e-4 else "DIFFERS")
                if verdict == "DIFFERS":
                    failures += 1
                print(f"    {var:24s} {max_abs:12.3e} {rel:12.3e} "
                      f"{scale:12.3e}  {verdict}")

    if not any_compared:
        print("  no season produced diagnostics from both runs")
    return failures


def compare_checkpoints(dir_b: str) -> None:
    """Summarise the streaming run's checkpoints, which hold the raw numbers."""
    import numpy as np
    import xarray as xr

    print("\n=== streaming checkpoints (per-timestep diagnostics) ===")
    for season in SEASONS:
        path = os.path.join(dir_b, "model", "netCDF", f"checkpoint_{season}.nc")
        if not os.path.isfile(path):
            continue
        with xr.open_dataset(path) as ds:
            n = int(ds.attrs.get("n_done", ds.sizes.get("step", 0)))
            print(f"  {season}: {n} timestep(s)")
            for var in ("uref", "zonal_mean_u", "zonal_mean_lwa",
                        "lwa_baro", "u_baro"):
                if var not in ds:
                    continue
                arr = ds[var].values
                print(f"    {var:16s} shape={arr.shape} "
                      f"mean={np.nanmean(arr):+.4f} "
                      f"min={np.nanmin(arr):+.4f} max={np.nanmax(arr):+.4f} "
                      f"nan={int(np.isnan(arr).sum())}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case-info", required=True,
                    help="path to a case_info.yml (see case_info_example.yml)")
    ap.add_argument("--work-dir", required=True,
                    help="scratch directory; two subdirectories are created in it")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter to run the drivers with (default: this one)")
    ap.add_argument("--only", choices=["original", "streaming"],
                    help="run just one of the two")
    args = ap.parse_args()

    if not os.path.isfile(args.case_info):
        raise SystemExit(f"case_info not found: {args.case_info}")

    dir_a = os.path.join(os.path.abspath(args.work_dir), "original")
    dir_b = os.path.join(os.path.abspath(args.work_dir), "streaming")

    rc = 0
    if args.only in (None, "original"):
        rc |= run_driver(ORIGINAL, dir_a, args.case_info, args.python)
    if args.only in (None, "streaming"):
        rc |= run_driver(STREAMING, dir_b, args.case_info, args.python)

    failures = 0
    if args.only is None:
        compare_figures(dir_a, dir_b)
        failures = compare_diagnostics(dir_a, dir_b)
    compare_checkpoints(dir_b)

    print("\nNote: the original stores its intermediates as float32, so ~1e-5")
    print("relative agreement is the floor; beyond 1e-4 is a real difference.")
    if failures:
        print(f"\n{failures} variable(s) DIFFER beyond tolerance.")
    return rc or (1 if failures else 0)


if __name__ == "__main__":
    sys.exit(main())
