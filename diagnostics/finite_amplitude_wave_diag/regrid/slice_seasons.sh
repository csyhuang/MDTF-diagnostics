#!/usr/bin/env bash
#
# Slice the first N days of each season out of the native-grid files, so the
# POD can be tested end to end without waiting for the full two-year regrid.
#
# Four disjoint time windows are extracted per variable and concatenated into
# one file. NCO cannot take disjoint hyperslabs on the same dimension in a
# single call, hence the per-window ncks followed by ncrcat.
#
# Usage:
#   ./slice_seasons.sh                              # foreground, logs as it goes
#   nohup ./slice_seasons.sh > /dev/null 2>&1 &     # survives logout
#   tail -f slice_seasons.log
#
# Everything printed is also written to LOG_FILE regardless of how the script
# is redirected, so the nohup form above loses nothing.
#
# Resumable: a window already extracted is not redone, and a variable whose
# final file exists is skipped entirely. Safe to rerun after an interruption.
#
# Environment (all optional):
#   DATA      directory holding the raw native-grid files
#   OUT       destination for the slices        (default: $DATA/subset)
#   CASE      case name
#   RANGE     date token in the raw filenames
#   VARS      variables to slice                (default: "T U V PS")
#   DAYS      days per season                   (default: 3)
#   TAG       date token for the output files   (default: SEASONS<DAYS>D)
#   LOG_FILE  progress log                      (default: ./slice_seasons.log)
#
set -uo pipefail

DATA="${DATA:-/nas/winds-data/csyhuang/mdtf_data}"
OUT="${OUT:-${DATA}/subset}"
CASE="${CASE:-mdtf_timeslice_v4.ne120L58.001}"
RANGE="${RANGE:-2001010121600-2002123121600}"
VARS="${VARS:-T U V PS}"
DAYS="${DAYS:-3}"
TAG="${TAG:-SEASONS${DAYS}D}"
LOG_FILE="${LOG_FILE:-$PWD/slice_seasons.log}"

# Capture our own output so `nohup ... > /dev/null 2>&1 &` still leaves a
# record, including any error the shell itself emits.
if [[ -t 1 ]]; then
    exec > >(tee -a "$LOG_FILE") 2>&1
else
    exec >> "$LOG_FILE" 2>&1
fi

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }

# Day-of-year offset (0-based) of each season's start, on a noleap calendar.
# Jan 1 = 0, Mar 1 = 31+28, Jun 1 = +31+30+31, Sep 1 = +30+31+31.
SEASON_NAMES=(DJF MAM JJA SON)
SEASON_DOY=(0 59 151 243)
STEPS_PER_DAY=4                      # 6-hourly

command -v ncks   >/dev/null || die "ncks not on PATH (conda activate mdtf_regrid?)"
command -v ncrcat >/dev/null || die "ncrcat not on PATH"
mkdir -p "$OUT" || die "cannot create $OUT"

log "=== slicing ${DAYS} day(s) per season ==="
log "source : ${DATA}"
log "output : ${OUT}"
log "tag    : ${TAG}"

# --- sanity-check the time axis before trusting any index ------------------
# The windows below assume the record starts 2001-01-01 03:00 and steps every
# 6 hours. If that is not what the file holds, every index is wrong and the
# slices would silently be the wrong days.
probe="${DATA}/${CASE}.cam.h7i.${VARS%% *}.${RANGE}.nc"
[[ -f "$probe" ]] || die "not found: $probe"
t0=$(ncks -H -C -s '%.4f\n' -v time -d time,0 "$probe" 2>/dev/null | head -1)
t1=$(ncks -H -C -s '%.4f\n' -v time -d time,1 "$probe" 2>/dev/null | head -1)
log "time[0] = ${t0}, time[1] = ${t1} (days since 2000-01-01, noleap)"
if [[ "$t0" != "365.1250" || "$t1" != "365.3750" ]]; then
    log "WARNING: expected 365.1250 and 365.3750, i.e. 2001-01-01 03:00 at 6-hourly."
    log "         The window indices below are derived from that assumption and"
    log "         will select the wrong days if it does not hold. Check the file."
fi

nwin=$(( DAYS * STEPS_PER_DAY ))
windows=()
for k in "${!SEASON_NAMES[@]}"; do
    i0=$(( SEASON_DOY[k] * STEPS_PER_DAY ))
    windows+=( "${i0},$(( i0 + nwin - 1 ))" )
    log "  ${SEASON_NAMES[k]}: time indices ${i0}-$(( i0 + nwin - 1 ))"
done
log "total: $(( ${#windows[@]} * nwin )) timesteps per variable"

# --- the work --------------------------------------------------------------
started=$SECONDS
for VAR in $VARS; do
    final="${OUT}/${CASE}.cam.h7i.${VAR}.${TAG}.nc"
    if [[ -s "$final" ]]; then
        log "${VAR}: ${final##*/} exists, skipping"
        continue
    fi

    src="${DATA}/${CASE}.cam.h7i.${VAR}.${RANGE}.nc"
    [[ -f "$src" ]] || { log "${VAR}: SKIP -- input not found: $src"; continue; }

    var_started=$SECONDS
    parts=()
    for k in "${!windows[@]}"; do
        part="${OUT}/part_${VAR}_${k}.nc"
        parts+=( "$part" )
        if [[ -s "$part" ]]; then
            log "${VAR} ${SEASON_NAMES[k]}: window ${windows[k]} already extracted, skipping"
            continue
        fi
        log "${VAR} ${SEASON_NAMES[k]}: extracting time ${windows[k]} ($(( k + 1 ))/${#windows[@]}) ..."
        step_started=$SECONDS
        # Write to a temporary name and rename on success, so an interrupted
        # ncks cannot leave a partial file that the skip check above trusts.
        if ! ncks -O -d "time,${windows[k]}" "$src" "${part}.tmp"; then
            rm -f "${part}.tmp"
            log "${VAR} ${SEASON_NAMES[k]}: FAILED"
            continue 2
        fi
        mv "${part}.tmp" "$part"
        log "${VAR} ${SEASON_NAMES[k]}: done in $(( SECONDS - step_started ))s, $(du -h "$part" | cut -f1)"
    done

    # Every window must be present before concatenating. Checked by iterating
    # rather than indexing from the end: negative array subscripts need bash
    # 4.3, and macOS still ships 3.2.
    complete=1
    for part in "${parts[@]}"; do
        [[ -s "$part" ]] || complete=0
    done
    if [[ ${#parts[@]} -ne ${#windows[@]} || $complete -ne 1 ]]; then
        log "${VAR}: incomplete, not concatenating"
        continue
    fi

    log "${VAR}: concatenating ${#parts[@]} windows ..."
    if ncrcat -O "${parts[@]}" "${final}.tmp"; then
        mv "${final}.tmp" "$final"
        rm -f "${parts[@]}"
        n=$(ncks -H -C -s '%.4f\n' -v time "$final" 2>/dev/null | grep -c .)
        log "${VAR}: ${final##*/} -- ${n} timesteps, $(du -h "$final" | cut -f1), $(( SECONDS - var_started ))s"
    else
        rm -f "${final}.tmp"
        log "${VAR}: ncrcat FAILED; parts kept for inspection"
    fi
done

# --- verify ----------------------------------------------------------------
log "=== finished in $(( (SECONDS - started) / 60 ))m $(( (SECONDS - started) % 60 ))s ==="
for VAR in $VARS; do
    final="${OUT}/${CASE}.cam.h7i.${VAR}.${TAG}.nc"
    [[ -s "$final" ]] || { log "  ${VAR}: MISSING"; continue; }
    n=$(ncks -H -C -s '%.4f\n' -v time "$final" 2>/dev/null | grep -c .)
    first=$(ncks -H -C -s '%.4f\n' -v time -d time,0 "$final" 2>/dev/null | head -1)
    last=$(ncks -H -C -s '%.4f\n' -v time -d "time,$(( n - 1 ))" "$final" 2>/dev/null | head -1)
    log "  ${VAR}: ${n} timesteps, time ${first} .. ${last}, $(du -h "$final" | cut -f1)"
done

log ""
log "Next: regrid the subset, reusing the existing mesh and weights."
log "  cd \$(dirname \$(dirname \$0))    # the POD directory"
log "  DATA_DIR=${OUT} OUT_DIR=${OUT}/regridded WORK_DIR=${DATA}/work \\"
log "    RANGE=${TAG} python -m regrid regrid T U V"
