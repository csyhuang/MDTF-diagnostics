#!/usr/bin/env python3
"""Detect and repair the 3-hour time shift in regridded output.

Background
----------
Until the fix in pipeline.py, step 2 of the regrid merged surface pressure from
the standalone PS file with::

    ncks -A -h -d time,i0,i1 -v PS <ps_file> <target>

``ncks -A`` brings the donor's coordinate variables along with the variable
being appended. The h7i files label a timestep differently depending on which
file it is in:

    T / U / V   time = 365.2500   (the END of time_bnds = [365.00, 365.25])
    PS          time = 365.1250   (the MIDPOINT of the same bounds)

so the append silently rewrote every output's time coordinate three hours
early.

**The data is not wrong.** The two PS sources are bit-identical at every index,
over identical bounds, so the surface pressure was always paired with the right
temperature field. Only the timestamp is wrong. But it propagates into the
catalog's ``time_range``, and at a month boundary it can move a timestep into
the neighbouring season -- the step labelled 1 March 00:00 becomes 28 February
21:00.

Detection
---------
A shifted file has ``time`` equal to the midpoint of its own ``time_bnds``; a
correct one has it equal to the upper bound. That is checked per file rather
than assumed from a date or a filename, so this is safe to run over a directory
holding a mixture of both.

    python fix_time_shift.py <dir-or-files>          # report only
    python fix_time_shift.py <dir-or-files> --apply  # repair in place
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from typing import List, Optional, Tuple

TOLERANCE = 1e-6          # days; the shift is 0.125, so this is ample


def _read(path: str, variable: str, extra: Optional[List[str]] = None) -> List[float]:
    cmd = ["ncks", "-H", "-C", "-s", "%.10f\n", "-v", variable]
    if extra:
        cmd += extra
    cmd.append(path)
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True).stdout
    return [float(x) for x in out.split() if x.strip()]


def classify(path: str) -> Tuple[str, float]:
    """Return (verdict, shift_in_days) for one file."""
    try:
        time = _read(path, "time", ["-d", "time,0,0"])
        bnds = _read(path, "time_bnds", ["-d", "time,0,0"])
        if not bnds:
            bnds = _read(path, "time_bounds", ["-d", "time,0,0"])
    except Exception as exc:                       # noqa: BLE001
        return f"unreadable ({exc})", 0.0
    if not time or len(bnds) < 2:
        return "no time_bnds -- cannot tell", 0.0

    t, lower, upper = time[0], bnds[0], bnds[1]
    midpoint = 0.5 * (lower + upper)
    if abs(t - upper) < TOLERANCE:
        return "OK", 0.0
    if abs(t - midpoint) < TOLERANCE:
        return "SHIFTED", upper - midpoint
    return f"unexpected (time={t}, bnds=[{lower}, {upper}])", 0.0


def repair(path: str, shift: float) -> bool:
    """Add *shift* days to the time coordinate, in place, atomically."""
    tmp = path + ".fixtime.tmp"
    result = subprocess.run(
        ["ncap2", "-O", "-h", "-s", f"time=time+{shift:.10f}", path, tmp],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not os.path.isfile(tmp):
        print(f"    FAILED: {result.stderr.strip()[:200]}")
        if os.path.isfile(tmp):
            os.unlink(tmp)
        return False
    os.replace(tmp, path)
    return True


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help="directories or .nc files")
    parser.add_argument("--apply", action="store_true",
                        help="repair; without this the run only reports")
    args = parser.parse_args(argv)

    files: List[str] = []
    for path in args.paths:
        if os.path.isdir(path):
            files.extend(sorted(glob.glob(os.path.join(path, "*.nc"))))
        else:
            files.append(path)
    if not files:
        raise SystemExit("no .nc files found")

    shifted, ok, other = [], 0, []
    for path in files:
        verdict, shift = classify(path)
        if verdict == "SHIFTED":
            shifted.append((path, shift))
        elif verdict == "OK":
            ok += 1
        else:
            other.append((path, verdict))

    print(f"{len(files)} file(s): {ok} OK, {len(shifted)} shifted, "
          f"{len(other)} indeterminate")
    for path, verdict in other:
        print(f"  ?  {os.path.basename(path)}: {verdict}")
    for path, shift in shifted:
        print(f"  !  {os.path.basename(path)}  (needs +{shift * 24:.1f} h)")

    if not shifted:
        return 0
    if not args.apply:
        print("\nRe-run with --apply to repair. Each file is rewritten by ncap2 "
              "and replaced atomically.")
        return 1

    print()
    failed = 0
    for path, shift in shifted:
        print(f"  repairing {os.path.basename(path)} ...", flush=True)
        if repair(path, shift):
            verdict, _ = classify(path)
            if verdict != "OK":
                print(f"    still {verdict} after repair")
                failed += 1
        else:
            failed += 1
    print(f"\nrepaired {len(shifted) - failed} of {len(shifted)}")
    print("Regenerate the catalog afterwards: python -m regrid catalog -o ...")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
