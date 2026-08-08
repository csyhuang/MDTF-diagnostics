#!/usr/bin/env bash
#
# Regrid native-grid CAM ne120pg3 output to a regular lat-lon grid on pressure
# levels, for use with the finite_amplitude_wave_diag POD.
#
# The MDTF preprocessor assumes X/Y axes and a pressure Z axis; the raw files are
# VAR(time, lev, ncol) on an unstructured mesh with hybrid-sigma levels, so both
# a horizontal remap and a vertical interpolation are required before the POD can
# read them.
#
# Usage:
#   ./regrid_ne120_to_latlon.sh setup                 # one-time: mesh, grid, weights
#   ./regrid_ne120_to_latlon.sh regrid T              # regrid one variable
#   ./regrid_ne120_to_latlon.sh validate <file.nc>    # sanity-check an output file
#
# Options (before the subcommand):
#   -n | --dry-run    print commands without running them
#   -c | --chunk N    timesteps per output file (default 124)
#   -h | --help
#
# Requires: NCO (ncremap, ncks, ncap2, ncrename, ncatted, ncwa, ncgen), ESMF,
# and tempest-remap. All are in the mdtf_regrid conda environment:
#   conda create -n mdtf_regrid -c conda-forge nco esmf tempest-remap
#
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration -- edit these, or override any of them from the environment,
# e.g.  DATA_DIR=/path/to/test RANGE=TEST1D ./regrid_ne120_to_latlon.sh regrid T
# ---------------------------------------------------------------------------
CASE="${CASE:-mdtf_timeslice_v4.ne120L58.001}"
DATA_DIR="${DATA_DIR:-/nas/winds-data/csyhuang/mdtf_data}"
OUT_DIR="${OUT_DIR:-/nas/winds-data/csyhuang/mdtf_data/regridded}"
WORK_DIR="${WORK_DIR:-/nas/winds-data/csyhuang/mdtf_data/work}"

# Stream to read 3D fields from. h7i = instantaneous (6hrPt), h8a = 6-hour means.
# Use h7i for science: local wave activity is a nonlinear functional of the
# instantaneous PV field, so pre-averaging the fields biases the diagnostic.
STREAM="${STREAM:-h7i}"
RANGE="${RANGE:-2001010121600-2002123121600}"

# PS always comes from h7i -- it is not published in the h8a stream.
PS_FILE="${PS_FILE:-${DATA_DIR}/${CASE}.cam.h7i.PS.${RANGE}.nc}"

# Target horizontal grid. 181x360 matches the xlon/ylat the POD driver defines.
NLAT="${NLAT:-181}"
NLON="${NLON:-360}"

# Vertical target: levels evenly spaced in PSEUDOHEIGHT, which is the coordinate
# falwa works in. Pseudoheight depends only on pressure,
#
#     z = -H * ln(p / P_GROUND)   <=>   p = P_GROUND * exp(-z / H)
#
# so an evenly spaced pseudoheight grid is just a particular list of pressures,
# and no new machinery is needed. Constants match falwa.constant (P_GROUND =
# 1000 hPa, SCALE_HEIGHT = 7000 m) via the POD's own convert_hPa_to_pseudoheight.
#
# The stored coordinate stays 'plev' in Pa on purpose: settings.jsonc declares
# the Z axis as air_pressure/Pa and the preprocessor expects a pressure axis, so
# writing a height coordinate here would break the varlist. falwa converts to
# pseudoheight itself. The values simply happen to be uniform in z.
#
# Interpolation is done in log(p) (VRT_NTP above), which is exactly linear in
# pseudoheight -- the natural scheme for this grid rather than a coincidence.
#
# Z_TOP_KM=41 is set by the model top: 2.838 hPa maps to z = 41.05 km, so 41 km
# (2.859 hPa) is the last level with real data above it. Asking for 42 km would
# produce an all-missing level.
PSEUDO_H="${PSEUDO_H:-7000.0}"      # SCALE_HEIGHT, m
PSEUDO_P0="${PSEUDO_P0:-1000.0}"    # P_GROUND, hPa
Z_TOP_KM="${Z_TOP_KM:-41}"
DZ_KM="${DZ_KM:-1}"

# Set PLEV directly to override the pseudoheight grid with an explicit list
# (e.g. ERA-Interim's 37 pressure levels).
if [[ -z "${PLEV:-}" ]]; then
    PLEV=$(awk -v h="$PSEUDO_H" -v p0="$PSEUDO_P0" -v zt="$Z_TOP_KM" -v dz="$DZ_KM" 'BEGIN {
        s = "";
        for (z = 0; z <= zt + 1e-9; z += dz) {
            p = p0 * exp(-z * 1000.0 / h) * 100.0;   # hPa -> Pa
            s = s (s == "" ? "" : ", ") sprintf("%.6f", p);
        }
        print s;
    }')
fi

# Remap algorithm. pg3 is a finite-volume grid, so a conservative scheme is the
# physically correct choice -- NOT the bilinear used for the np4 spectral grid.
#
# traave is TempestRemap's FV->FV conservative map, which is purpose-built for
# cubed-sphere grids. Do not use the ESMF-backed aliases (aave/conserve/esmfaave):
# ESMF_RegridWeightGen segfaults on this grid pair in the conda-forge osx-arm64
# build. ncoaave (NCO's own conservative scheme) is a working fallback but is
# markedly slower.
ALGO="${ALGO:-traave}"

# What to do with target levels below ground (e.g. 1000 hPa over Tibet).
# ncremap's default is nrs_ngh, which fabricates below-ground values by copying
# the nearest level; mss_val leaves them missing, which is honest. Switch only
# if falwa cannot cope with NaN in those columns.
VRT_XTR="${VRT_XTR:-mss_val}"

# Interpolation in log(p), the correct choice for a pressure coordinate.
# ncremap's default is no interpolation at all, so this must be set explicitly.
VRT_NTP="${VRT_NTP:-log}"

# Renormalization threshold for the horizontal remap. THIS IS NOT OPTIONAL for
# this dataset. Vertical interpolation leaves cells missing below ground; a
# conservative remap then averages missing together with valid neighbours and,
# without renormalization, treats the missing part as zero. On a one-day test
# that silently produced 2599 cells at 1000 hPa holding 0 < T < 150 K -- values
# that are wrong but not obviously wrong. With -r 0.0 the same field has a
# minimum of 228 K and no corrupted cells. 0.0 means: renormalize by the valid
# weight sum wherever any valid source data exists, leave fully-underground
# cells missing.
RNR_THR="${RNR_THR:-0.0}"

# Timesteps per output file. 124 = one 31-day month at 6-hourly.
CHUNK="${CHUNK:-124}"

# ---------------------------------------------------------------------------
# Derived paths -- generally no need to edit
# ---------------------------------------------------------------------------
SCRIP_FL="${WORK_DIR}/ne120pg3_scrip.nc"
DST_GRD="${WORK_DIR}/${NLAT}x${NLON}_RLL.nc"
MAP_FL="${WORK_DIR}/map_ne120pg3_to_${NLAT}x${NLON}_${ALGO}.nc"
VRT_FL="${WORK_DIR}/vrt_plev.nc"
NCOL_EXPECTED=777600

DRY_RUN=0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# Run a command, or just print it under --dry-run.
run() {
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  + %s\n' "$*" >&2
    else
        "$@"
    fi
}

# Run a command that is known to write a valid file and then exit non-zero.
# tempest-remap's mesh generators abort (SIGABRT, exit 134) during HDF5 teardown
# on macOS after the output has been written correctly. Verifying the artifact is
# a better completion test than the exit status here; a genuine failure still
# trips the "not created" check.
run_tolerant() {
    local expect="$1"; shift
    if [[ $DRY_RUN -eq 1 ]]; then
        printf '  + %s\n' "$*" >&2
        return 0
    fi
    local rc=0
    "$@" || rc=$?
    [[ -s "$expect" ]] || die "$1 failed (exit $rc) and did not create $expect"
    (( rc != 0 )) && log "note: $1 exited $rc but wrote $expect (known teardown bug)"
    return 0
}

require_tools() {
    local missing=()
    for t in "$@"; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done
    [[ ${#missing[@]} -eq 0 ]] || die "not on PATH: ${missing[*]} (conda activate mdtf_regrid?)"
}

# Number of timesteps in a file, read from the unlimited dimension.
n_time() {
    ncdump -h "$1" | sed -n 's/.*time = UNLIMITED ; \/\/ (\([0-9]*\) currently).*/\1/p'
}

# ---------------------------------------------------------------------------
# setup: build the mesh, target grid, weights and vertical grid. Run once; all
# four artifacts are reused by every variable and every chunk.
# ---------------------------------------------------------------------------
cmd_setup() {
    require_tools ncremap ncgen ncdump GenerateCSMesh GenerateVolumetricMesh
    run mkdir -p "$WORK_DIR" "$OUT_DIR"

    # -- source mesh -------------------------------------------------------
    # The data is on the pg3 physics grid, not np4: ncol = 6*120^2*3^2 = 777600
    # (np4 would be 777602), and the files carry fv_nphys = 3. Generating the
    # mesh is more reliable than hunting for a shipped SCRIP file.
    if [[ ! -f "$SCRIP_FL" ]]; then
        log "generating ne120pg3 SCRIP mesh"
        run_tolerant "${WORK_DIR}/ne120.g" \
            GenerateCSMesh --res 120 --alt --file "${WORK_DIR}/ne120.g"
        run_tolerant "${WORK_DIR}/ne120pg3.g" \
            GenerateVolumetricMesh --in "${WORK_DIR}/ne120.g" \
            --out "${WORK_DIR}/ne120pg3.g" --np 3 --uniform
        # subcommand name varies by tempest-remap version; try both
        if command -v ConvertMeshToSCRIP >/dev/null 2>&1; then
            run_tolerant "$SCRIP_FL" ConvertMeshToSCRIP --in "${WORK_DIR}/ne120pg3.g" --out "$SCRIP_FL"
        else
            run_tolerant "$SCRIP_FL" ConvertExodusToSCRIP --in "${WORK_DIR}/ne120pg3.g" --out "$SCRIP_FL"
        fi
    else
        log "reusing $SCRIP_FL"
    fi

    # The single most valuable check in this script: if grid_size is not 777600
    # the mesh does not match the data and everything downstream is silently wrong.
    if [[ $DRY_RUN -eq 0 && -f "$SCRIP_FL" ]]; then
        local gs
        gs=$(ncdump -h "$SCRIP_FL" | sed -n 's/.*grid_size = \([0-9]*\).*/\1/p' | head -1)
        [[ "$gs" == "$NCOL_EXPECTED" ]] \
            || die "mesh grid_size=$gs but data ncol=$NCOL_EXPECTED -- wrong mesh"
        log "mesh grid_size = $gs, matches data"

        # Cell count alone does not prove the mesh matches: cube-sphere
        # orientation (GenerateCSMesh --alt) and element ordering both change the
        # coordinates while leaving the count identical. Comparing the mesh's cell
        # centres against the data's own lat/lon settles it -- a mismatch here
        # means every remapped field would be silently scrambled.
        if [[ -f "$PS_FILE" ]]; then
            local n=200
            ncks -H -C -s '%.6f\n' -v grid_center_lat -d grid_size,0,$((n - 1)) \
                "$SCRIP_FL" > "${WORK_DIR}/.mesh_lat" 2>/dev/null || true
            ncks -H -C -s '%.6f\n' -v lat -d ncol,0,$((n - 1)) \
                "$PS_FILE" > "${WORK_DIR}/.data_lat" 2>/dev/null || true
            if [[ -s "${WORK_DIR}/.mesh_lat" && -s "${WORK_DIR}/.data_lat" ]]; then
                local maxdiff
                maxdiff=$(paste "${WORK_DIR}/.mesh_lat" "${WORK_DIR}/.data_lat" \
                    | awk 'NF==2 {d=$1-$2; if(d<0)d=-d; if(d>m)m=d} END {printf "%.6f", m+0}')
                # Tolerance 0.01 deg: TempestRemap and CAM use slightly different
                # cell-centre conventions, giving a residual of ~8e-4 deg (~80 m on
                # 28 km cells) even when the mesh is correct. A genuinely wrong
                # orientation or ordering is off by degrees, so this separates
                # cleanly. Conservative remapping uses corners, not centres.
                log "mesh vs data latitude, first $n cells: max |diff| = $maxdiff deg"
                awk -v m="$maxdiff" 'BEGIN {exit !(m > 0.01)}' \
                    && die "mesh cell centres do not match the data's lat -- wrong
       orientation or ordering. Try toggling the --alt flag on GenerateCSMesh,
       then delete $SCRIP_FL and rerun setup."
                log "mesh cell centres match the data"
            fi
            rm -f "${WORK_DIR}/.mesh_lat" "${WORK_DIR}/.data_lat"
        else
            log "WARNING: $PS_FILE not found, skipping mesh-vs-data coordinate check"
        fi
    fi

    # -- target grid -------------------------------------------------------
    # This is a cell-centred grid: 181 latitudes running -89.503 .. 89.503 at
    # 0.9945 deg spacing, with no row exactly at either pole. That does not
    # match the POD driver's ylat = np.arange(-90, 91, 1.0) point-for-point;
    # falwa interpolates onto its own analysis grid, so the offset is absorbed
    # there. If you ever need cell centres exactly on -90 .. 90 (e.g. to compare
    # against a CAM FV grid), append '#lat_typ=cap' below and regenerate the
    # weights -- the map file depends on the destination grid.
    if [[ ! -f "$DST_GRD" ]]; then
        log "generating ${NLAT}x${NLON} target grid"
        run ncremap -g "$DST_GRD" -G "latlon=${NLAT},${NLON}"
    fi

    # -- weights (the expensive artifact; built once) -----------------------
    if [[ ! -f "$MAP_FL" ]]; then
        log "generating $ALGO weights (slow)"
        run ncremap -a "$ALGO" -s "$SCRIP_FL" -g "$DST_GRD" -m "$MAP_FL"
    fi

    # -- vertical grid -----------------------------------------------------
    # Built with ncgen from CDL so this needs no Python and no template file.
    if [[ ! -f "$VRT_FL" ]]; then
        log "generating vertical grid"
        local nlev
        nlev=$(printf '%s' "$PLEV" | tr -cd ',' | wc -c)
        nlev=$((nlev + 1))
        if [[ $DRY_RUN -eq 0 ]]; then
            cat > "${WORK_DIR}/vrt_plev.cdl" <<EOF
netcdf vrt_plev {
dimensions:
    plev = ${nlev} ;
variables:
    double plev(plev) ;
        plev:units = "Pa" ;
        plev:long_name = "pressure" ;
        plev:axis = "Z" ;
        plev:positive = "down" ;
data:
    plev = ${PLEV} ;
}
EOF
            ncgen -o "$VRT_FL" "${WORK_DIR}/vrt_plev.cdl"
        fi
        log "vertical grid: $nlev levels"
    fi

    log "setup complete"
}

# ---------------------------------------------------------------------------
# regrid: process one variable, chunk by chunk.
#
# Vertical interpolation runs before the horizontal remap so the hybrid
# coefficients are applied on the grid they were defined on, and so pressure is
# never derived from remapped topography. The intermediate is larger this way;
# chunking caps it.
# ---------------------------------------------------------------------------
cmd_regrid() {
    local var="$1"
    require_tools ncremap ncks ncap2 ncrename ncatted ncdump

    local src="${DATA_DIR}/${CASE}.cam.${STREAM}.${var}.${RANGE}.nc"
    [[ -f "$src" ]] || die "input not found: $src"
    [[ -f "$PS_FILE" ]] || die "PS not found: $PS_FILE"
    for f in "$MAP_FL" "$VRT_FL"; do
        [[ -f "$f" ]] || die "$f missing -- run '$0 setup' first"
    done
    run mkdir -p "$OUT_DIR" "$WORK_DIR"

    local nt nt_ps
    nt=$(n_time "$src")
    nt_ps=$(n_time "$PS_FILE")
    [[ "$nt" == "$nt_ps" ]] \
        || die "time mismatch: $var has $nt steps, PS has $nt_ps"
    log "$var: $nt timesteps, chunks of $CHUNK"

    # Guard against requesting levels above the model top. There is no data up
    # there, so those levels come back 100% missing and then propagate NaN into
    # any column-wise diagnostic. hybm[0] = 0 makes the top level pure pressure,
    # so lev[0] is the model top exactly, everywhere.
    if [[ $DRY_RUN -eq 0 ]]; then
        local top_hpa
        top_hpa=$(ncks -H -C -s '%f' -v lev -d lev,0 "$src" 2>/dev/null || true)
        if [[ -n "$top_hpa" ]]; then
            local above
            above=$(awk -v t="$top_hpa" -v lst="$PLEV" 'BEGIN {
                n = split(lst, a, ","); s = "";
                for (i = 1; i <= n; i++) {
                    p = a[i] + 0;
                    if (p / 100.0 < t) s = s sprintf("%s%.4g", (s == "" ? "" : " "), p / 100.0);
                }
                print s;
            }')
            if [[ -n "$above" ]]; then
                log "WARNING: model top is ${top_hpa} hPa but these target levels are above it:"
                log "         ${above} hPa -- they will be entirely missing."
            else
                log "all target levels are at or below the model top (${top_hpa} hPa)"
            fi
        fi
    fi

    local i0 i1 tag out tmp
    for (( i0=0; i0<nt; i0+=CHUNK )); do
        i1=$(( i0 + CHUNK - 1 ))
        (( i1 >= nt )) && i1=$(( nt - 1 ))
        tag=$(printf '%06d-%06d' "$i0" "$i1")
        out="${OUT_DIR}/${CASE}.${var}.${tag}.nc"

        # idempotent: a completed chunk is never redone, so a killed run resumes
        if [[ -s "$out" ]]; then
            log "$var chunk $tag: exists, skipping"
            continue
        fi
        log "$var chunk $tag"
        tmp="${WORK_DIR}/${var}.${tag}"

        # 1. time subset of the 3D field (brings hyam/hybm/hyai/hybi/lev along)
        run ncks -O -h -d "time,${i0},${i1}" "$src" "${tmp}.sub.nc"

        # 2. append the matching PS slice. PS lives in its own file and carries
        #    no hybrid coefficients, so it must be merged into the 3D file
        #    rather than the other way round.
        run ncks -A -h -d "time,${i0},${i1}" -v PS "$PS_FILE" "${tmp}.sub.nc"

        # 3. P0 is absent from every file in this dataset. lev = 1000*(A+B) in
        #    hPa implies P0 = 100000 Pa.
        run ncap2 -O -h -s 'P0=100000.0; P0@units="Pa"; P0@long_name="reference pressure"' \
            "${tmp}.sub.nc" "${tmp}.p0.nc"

        # 4. hybrid-sigma -> pressure, still on the native mesh
        run ncremap --vrt_out="$VRT_FL" --vrt_ntp="$VRT_NTP" --vrt_xtr="$VRT_XTR" \
            --ps_nm=PS --vrt_nm=plev \
            -i "${tmp}.p0.nc" -o "${tmp}.vrt.nc"

        # 5. drop the hybrid machinery now that we are on pressure levels. Doing
        #    this before the horizontal remap avoids regridding PS pointlessly,
        #    and sidesteps an NCO warning about PS carrying a NaN _FillValue,
        #    which cannot be compared arithmetically. The ^(...)$ regex form is
        #    used so absent variables are skipped rather than erroring.
        run ncks -O -h -x -v '^(hyam|hybm|hyai|hybi|ilev|P0|PS)$' \
            "${tmp}.vrt.nc" "${tmp}.thin.nc"

        # 6. unstructured -> lat-lon, with renormalization (see RNR_THR above)
        run ncremap -m "$MAP_FL" -r "$RNR_THR" -i "${tmp}.thin.nc" -o "${tmp}.hrz.nc"

        # 7. drop the artifacts ncremap adds, leaving exactly one data variable
        #    plus time bounds -- what the catalog builder needs to identify
        #    variable_id unambiguously. cell_measures is deleted too, or it would
        #    dangle a reference to the removed 'area'.
        #    -C is required: lat_bnds/lon_bnds are CF-associated with lat/lon, and
        #    without it NCO re-adds them to the extraction list and then refuses
        #    the exclusion outright.
        run ncks -O -h -C -x -v '^(area|gw|lat_bnds|lon_bnds)$' \
            "${tmp}.hrz.nc" "${tmp}.cln.nc"
        run ncatted -O -h -a cell_measures,,d,, "${tmp}.cln.nc"

        # 8. time_bounds -> time_bnds. The catalog builder's exclusion list
        #    (tools/catalog_builder/parsers.py) contains 'time_bnds' but not
        #    'time_bounds', so without this rename it picks the bounds variable
        #    as variable_id. The leading '.' marks the rename optional, so this
        #    is a no-op when the variable is already named time_bnds -- an
        #    earlier `ncdump | grep -q` guard here silently never fired, because
        #    grep -q exits on first match and SIGPIPEs ncdump, which `pipefail`
        #    then turns into a false condition.
        run ncrename -O -h -v .time_bounds,time_bnds "${tmp}.cln.nc"
        run ncatted -O -h -a bounds,time,o,c,time_bnds "${tmp}.cln.nc"

        # 8. publish atomically -- a killed job never leaves a partial file that
        #    the skip-if-exists check would mistake for a finished one
        run mv "${tmp}.cln.nc" "$out"
        run rm -f "${tmp}".*.nc
        log "$var chunk $tag -> $out"
    done
    log "$var complete"
}

# ---------------------------------------------------------------------------
# validate: cheap checks that catch the failures that matter, before you spend
# hours on the full dataset.
# ---------------------------------------------------------------------------
cmd_validate() {
    local f="$1"
    require_tools ncdump ncwa ncks
    [[ -f "$f" ]] || die "not found: $f"

    log "header:"
    ncdump -h "$f" | sed -n '/dimensions:/,/^\/\//p'

    local var
    var=$(ncdump -h "$f" | sed -n 's/^\t[a-z]* \([A-Z][A-Za-z0-9_]*\)(time.*/\1/p' | head -1)
    [[ -n "$var" ]] || die "could not identify the data variable in $f"
    log "data variable: $var"

    # Per-level means. An all-missing level means the vertical target went above
    # the model top, or the extrapolation mode discarded the column.
    log "per-level mean of $var (NaN or missing => bad level):"
    ncwa -O -h -y avg -a time,lat,lon -v "$var" "$f" "${f}.chk.nc"
    ncks -H -C -v "$var" "${f}.chk.nc" || true
    rm -f "${f}.chk.nc"

    # Cell-centred grid: expect about -89.5 to 89.5, not -90 to 90.
    log "latitude range (cell-centred, expect ~-89.5 to ~89.5):"
    ncks -H -C -v lat -d lat,0 "$f" || true
    ncks -H -C -v lat -d "lat,$((NLAT - 1))" "$f" || true
}

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

# ---------------------------------------------------------------------------
main() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -n|--dry-run) DRY_RUN=1; shift ;;
            -c|--chunk)   CHUNK="$2"; shift 2 ;;
            -h|--help)    usage; exit 0 ;;
            *)            break ;;
        esac
    done
    [[ $# -ge 1 ]] || { usage; exit 1; }

    case "$1" in
        setup)    cmd_setup ;;
        regrid)   [[ $# -eq 2 ]] || die "usage: $0 regrid <VAR>"; cmd_regrid "$2" ;;
        validate) [[ $# -eq 2 ]] || die "usage: $0 validate <file.nc>"; cmd_validate "$2" ;;
        *)        die "unknown subcommand: $1" ;;
    esac
}

main "$@"
