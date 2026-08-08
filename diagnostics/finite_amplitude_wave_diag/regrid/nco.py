"""Thin subprocess layer over the NCO / TempestRemap command-line tools.

Everything that touches a netCDF file goes through here. Keeping the actual
command strings in one place means the pipeline modules read as a recipe, and
means the ``--dry-run`` and logging behaviour is uniform without each call site
having to remember it.

No third-party imports: metadata is read back by parsing ``ncdump -h`` and
``ncks -H``, exactly as the shell script did, so the package has no dependency
beyond the tools it is wrapping.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from typing import List, Optional, Sequence

log = logging.getLogger(__name__)


class NCOError(RuntimeError):
    """A wrapped command failed."""


class ToolNotFoundError(NCOError):
    """A required executable is not on PATH."""


def require_tools(*tools: str) -> None:
    """Raise :class:`ToolNotFoundError` listing every missing executable.

    Reporting all of them at once matters: the usual cause is a forgotten
    ``conda activate``, and a one-at-a-time failure would make that look like
    several unrelated problems.
    """
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        raise ToolNotFoundError(
            f"not on PATH: {' '.join(missing)} "
            f"(did you `conda activate mdtf_regrid`?)"
        )


def run(
    cmd: Sequence[str],
    dry_run: bool = False,
    capture: bool = False,
) -> Optional[str]:
    """Run *cmd*, or print it under *dry_run*.

    Returns stdout when *capture* is set, otherwise ``None``. Raises
    :class:`NCOError` on a non-zero exit.
    """
    cmd = [str(c) for c in cmd]
    if dry_run:
        print("  + " + " ".join(cmd))
        return None

    log.debug("running: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise NCOError(
            f"{cmd[0]} exited {proc.returncode}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr: {stderr[-2000:]}"
        )
    return proc.stdout if capture else None


def run_tolerant(expect: str, cmd: Sequence[str], dry_run: bool = False) -> None:
    """Run *cmd*, treating the existence of *expect* as the completion test.

    TempestRemap's mesh generators abort during HDF5 teardown on macOS
    (SIGABRT, exit 134) *after* writing a correct output file. Checking the
    artifact is a better test than the exit status here. A genuine failure
    still trips, because then the file is absent or empty.
    """
    cmd = [str(c) for c in cmd]
    if dry_run:
        print("  + " + " ".join(cmd))
        return

    proc = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if not (os.path.isfile(expect) and os.path.getsize(expect) > 0):
        stderr = (proc.stderr or "").strip()
        raise NCOError(
            f"{cmd[0]} exited {proc.returncode} and did not create {expect}\n"
            f"  command: {' '.join(cmd)}\n"
            f"  stderr: {stderr[-2000:]}"
        )
    if proc.returncode != 0:
        log.info(
            "%s exited %d but wrote %s (known teardown bug, continuing)",
            cmd[0], proc.returncode, expect,
        )


# ---------------------------------------------------------------------------
# Metadata readers
# ---------------------------------------------------------------------------

def ncdump_header(path: str) -> str:
    """``ncdump -h`` output for *path*."""
    out = run(["ncdump", "-h", path], capture=True)
    return out or ""


_RE_TIME_UNLIMITED = re.compile(
    r"time\s*=\s*UNLIMITED\s*;\s*//\s*\((\d+)\s+currently\)"
)
_RE_TIME_FIXED = re.compile(r"^\s*time\s*=\s*(\d+)\s*;", re.MULTILINE)


def n_time(path: str, header: Optional[str] = None) -> int:
    """Number of timesteps in *path*.

    Handles both the unlimited form (``time = UNLIMITED ; // (2920 currently)``)
    and a fixed ``time`` dimension, which is what some NCO intermediates end up
    with after a subset.
    """
    header = ncdump_header(path) if header is None else header
    m = _RE_TIME_UNLIMITED.search(header)
    if m:
        return int(m.group(1))
    m = _RE_TIME_FIXED.search(header)
    if m:
        return int(m.group(1))
    raise NCOError(f"could not determine the number of timesteps in {path}")


_RE_GRID_SIZE = re.compile(r"^\s*grid_size\s*=\s*(\d+)\s*;", re.MULTILINE)


def scrip_grid_size(path: str) -> int:
    """``grid_size`` of a SCRIP mesh file."""
    header = ncdump_header(path)
    m = _RE_GRID_SIZE.search(header)
    if not m:
        raise NCOError(f"no grid_size dimension in {path} -- not a SCRIP file?")
    return int(m.group(1))


def read_values(
    path: str,
    var: str,
    dim: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None,
    fmt: str = "%.10f",
) -> List[float]:
    """Read numeric values of *var*, optionally hyperslabbed along *dim*.

    ``-C`` suppresses NCO's habit of pulling in CF-associated coordinates,
    which would otherwise interleave other variables' values into the output.
    """
    cmd = ["ncks", "-H", "-C", "-s", fmt + "\n", "-v", var]
    if dim is not None:
        if start is None:
            raise ValueError("start is required when dim is given")
        stop = start if end is None else end
        cmd += ["-d", f"{dim},{start},{stop}"]
    cmd.append(path)

    out = run(cmd, capture=True) or ""
    values = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            # NCO sometimes emits a trailing blank or a warning line; skip it
            # rather than failing the whole read.
            continue
    if not values:
        raise NCOError(f"no values read for {var} from {path}")
    return values


def dimension_names(path: str, header: Optional[str] = None) -> List[str]:
    """Names declared in the ``dimensions:`` block."""
    header = ncdump_header(path) if header is None else header
    block = re.search(r"^dimensions:\n(.*?)^variables:", header, re.MULTILINE | re.DOTALL)
    if not block:
        return []
    return re.findall(r"^\s*(\w+)\s*=", block.group(1), re.MULTILINE)


def data_variable_names(path: str, header: Optional[str] = None) -> List[str]:
    """Names of the actual data variables in a finished output file.

    A variable qualifies if it has a ``time`` dimension and is neither a
    coordinate variable (name identical to a dimension, which is how ``time``
    itself would otherwise slip in) nor a bounds variable. A clean output file
    yields exactly one name, which is what lets the catalog builder identify
    ``variable_id`` unambiguously.
    """
    header = ncdump_header(path) if header is None else header
    dims = set(dimension_names(path, header))
    names = []
    for m in re.finditer(
        r"^\s*\w+\s+(\w+)\s*\(\s*time\s*[,)]", header, re.MULTILINE
    ):
        name = m.group(1)
        if name in dims:
            continue
        if name.endswith("_bnds") or name.endswith("_bounds"):
            continue
        names.append(name)
    return names


def global_attribute(path: str, attr: str, header: Optional[str] = None) -> Optional[str]:
    """Value of a global attribute, or ``None`` if absent."""
    header = ncdump_header(path) if header is None else header
    m = re.search(rf'^\s*:{re.escape(attr)}\s*=\s*"(.*?)"\s*;', header, re.MULTILINE)
    return m.group(1) if m else None


def variable_attribute(
    path: str, var: str, attr: str, header: Optional[str] = None
) -> Optional[str]:
    """Value of a variable attribute, or ``None`` if absent."""
    header = ncdump_header(path) if header is None else header
    m = re.search(
        rf'^\s*{re.escape(var)}:{re.escape(attr)}\s*=\s*"(.*?)"\s*;',
        header,
        re.MULTILINE,
    )
    return m.group(1) if m else None
