"""Command-line interface.

    python -m regrid setup
    python -m regrid regrid T U V
    python -m regrid levels
    python -m regrid catalog -o esm_catalog_regridded.json
    python -m regrid validate output.nc

Run from the POD directory, or add it to ``PYTHONPATH``. Every setting can also
come from the environment, so the shell script's documented invocations carry
over::

    DATA_DIR=/data STREAM=h8a RANGE=TEST1D python -m regrid regrid T
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from .catalog import write_catalog
from .config import RegridConfig
from .levels import format_level_table, parse_level_list
from .mesh import setup
from .nco import NCOError
from .pipeline import regrid_variable
from .validate import validate_file


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand. Defaults come from the environment."""
    group = parser.add_argument_group("paths")
    group.add_argument("--case", help="case name (default: %(default)s)")
    group.add_argument("--data-dir", help="directory holding the raw native-grid files")
    group.add_argument("--out-dir", help="directory for the regridded output")
    group.add_argument("--work-dir", help="directory for meshes, weights, intermediates")
    group.add_argument("--ps-file", help="surface pressure file (default: the h7i PS file)")

    group = parser.add_argument_group("input selection")
    group.add_argument("--stream", help="history stream: h7i (instantaneous) or h8a (6-hour means)")
    group.add_argument("--date-range", help="date token in the raw filenames")

    group = parser.add_argument_group("target grid")
    group.add_argument("--nlat", type=int, help="target latitude count")
    group.add_argument("--nlon", type=int, help="target longitude count")
    group.add_argument("--z-top-km", type=float, help="highest pseudoheight level, km")
    group.add_argument("--dz-km", type=float, help="pseudoheight spacing, km")
    group.add_argument(
        "--plev",
        help="explicit pressure levels in Pa, comma- or space-separated; "
             "overrides the pseudoheight grid",
    )

    group = parser.add_argument_group("algorithm")
    group.add_argument("--algo", help="remap algorithm (traave, ncoaave, ...)")
    group.add_argument("--vrt-ntp", help="vertical interpolation: log or lin")
    group.add_argument("--vrt-xtr", help="below-ground handling: mss_val or nrs_ngh")
    group.add_argument("--rnr-thr", type=float, help="renormalization threshold; see docs before changing")
    group.add_argument("-c", "--chunk", type=int, help="timesteps per output file")


def _config_from_args(args: argparse.Namespace) -> RegridConfig:
    overrides = {
        key: getattr(args, key, None)
        for key in (
            "case", "data_dir", "out_dir", "work_dir", "ps_file", "stream",
            "date_range", "nlat", "nlon", "z_top_km", "dz_km", "algo",
            "vrt_ntp", "vrt_xtr", "rnr_thr", "chunk",
        )
    }
    overrides["dry_run"] = getattr(args, "dry_run", False)
    if getattr(args, "plev", None):
        overrides["plev"] = parse_level_list(args.plev)
    return RegridConfig.from_env(**overrides)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m regrid",
        description=(
            "Regrid native-grid CAM ne120pg3 output to a regular lat-lon grid on "
            "pressure levels, for the finite_amplitude_wave_diag POD."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Requires NCO, ESMF and tempest-remap on PATH:\n"
            "  conda create -n mdtf_regrid -c conda-forge nco esmf tempest-remap\n"
            "  conda activate mdtf_regrid"
        ),
    )
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="print commands without running them")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    parser.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")

    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="build mesh, target grid, weights, vertical grid")
    _add_config_arguments(p)

    p = sub.add_parser("regrid", help="regrid one or more variables")
    p.add_argument("variables", nargs="+", metavar="VAR",
                   help="CAM variable names, e.g. T U V")
    _add_config_arguments(p)

    p = sub.add_parser("levels", help="print the target levels and exit")
    _add_config_arguments(p)

    p = sub.add_parser("catalog", help="write an intake-ESM catalog for the output")
    p.add_argument("-o", "--output", required=True,
                   help="path to the catalog .json to write (.csv written alongside)")
    p.add_argument("--frequency", default="6hr",
                   help="catalog frequency string; use 6hr, never 6hrPt "
                        "(default: %(default)s)")
    _add_config_arguments(p)

    p = sub.add_parser("validate", help="check a finished output file")
    p.add_argument("files", nargs="+", metavar="FILE")
    _add_config_arguments(p)

    p = sub.add_parser("config", help="print the resolved configuration and exit")
    _add_config_arguments(p)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.INFO
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
    )

    try:
        cfg = _config_from_args(args)

        if args.command == "config":
            print(cfg.describe())
            return 0

        if args.command == "levels":
            levels = cfg.levels
            print(f"{len(levels)} levels "
                  f"(H = {cfg.pseudo_h:.0f} m, p0 = {cfg.pseudo_p0:.0f} hPa)\n")
            print(format_level_table(levels, cfg.pseudo_h, cfg.pseudo_p0))
            return 0

        if args.command == "setup":
            setup(cfg)
            return 0

        if args.command == "regrid":
            for var in args.variables:
                regrid_variable(cfg, var)
            return 0

        if args.command == "catalog":
            write_catalog(cfg, args.output, frequency=args.frequency)
            return 0

        if args.command == "validate":
            failures = 0
            for path in args.files:
                report = validate_file(cfg, path)
                print(report)
                failures += 0 if report.ok else 1
            return 1 if failures else 0

    except NCOError as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("interrupted")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
