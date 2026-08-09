#!/usr/bin/env bash
#
# Supervisor for long regridding runs.
#
# Runs `python -m regrid regrid` for each variable, restarting automatically if
# it dies. Because completed chunks are published with an atomic rename and
# skipped on the next pass, a restart resumes at the first unfinished chunk --
# no work is repeated and no partial file is ever mistaken for a finished one.
#
# The point of this wrapper is to distinguish two kinds of failure that a bare
# retry loop cannot:
#
#   * transient  -- node hiccup, NFS timeout, OOM from something else on the
#                   box. The next attempt makes progress. Retry indefinitely.
#   * permanent  -- disk full, missing input, bad weights. Every attempt fails
#                   identically. Retrying forever just fills the log.
#
# It tells them apart by counting finished output files: a retry that produces
# no new chunk is "no progress", and after MAX_STALLED_RETRIES of those in a
# row the variable is abandoned and the script moves on.
#
# Usage:
#   nohup ./run_supervised.sh > /dev/null 2>&1 &      # survives logout
#   tmux new -s regrid './run_supervised.sh'          # or, if tmux is available
#
#   ./run_supervised.sh status                        # progress summary
#   ./run_supervised.sh stop                          # stop after current chunk
#
# Logs, all under LOG_DIR (default ./regrid_logs):
#
#   run_supervised_output.log   everything this script prints, including
#                               bash-level errors. Written regardless of how
#                               the script is redirected, so
#                               `nohup ... > /dev/null 2>&1 &` loses nothing.
#   regrid_<VAR>.log            full regrid + NCO output for that variable,
#                               appended across restarts.
#
# Environment (all optional):
#   VARS="T U V"          variables to process, in order
#   PARALLEL=1            run the variables concurrently (see the note below)
#   RETRY_DELAY=60        seconds to wait after a failure
#   MAX_STALLED_RETRIES=3 consecutive no-progress failures before giving up
#   LOG_DIR=./regrid_logs where logs and state are written
#   plus every variable the regrid package itself understands
#   (DATA_DIR, OUT_DIR, WORK_DIR, STREAM, RANGE, CHUNK, ...)
#
set -uo pipefail   # deliberately NOT -e: failures are handled, not fatal

VARS="${VARS:-T U V}"
PARALLEL="${PARALLEL:-0}"
RETRY_DELAY="${RETRY_DELAY:-60}"
MAX_STALLED_RETRIES="${MAX_STALLED_RETRIES:-3}"
LOG_DIR="${LOG_DIR:-$PWD/regrid_logs}"
PYTHON="${PYTHON:-python3}"

STOP_FILE="${LOG_DIR}/STOP"
PID_FILE="${LOG_DIR}/supervisor.pid"
OUTPUT_LOG="${LOG_DIR}/run_supervised_output.log"

mkdir -p "$LOG_DIR"

# Unbuffered so `tail -f` shows the line as it happens rather than when a
# block fills -- on a job whose steps take tens of minutes, a stale log looks
# indistinguishable from a hung run.
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

# Where the finished chunks land. Ask the package rather than guessing, so this
# stays correct whatever combination of environment variables is in play.
out_dir() {
    $PYTHON -c "
import sys
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)')
from regrid import RegridConfig
print(RegridConfig.from_env().out_dir)" 2>/dev/null
}

case_name() {
    $PYTHON -c "
import sys
sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)')
from regrid import RegridConfig
print(RegridConfig.from_env().case)" 2>/dev/null
}

# Number of finished chunks for one variable.
count_done() {
    local var="$1" od cs
    od=$(out_dir); cs=$(case_name)
    [[ -d "$od" ]] || { echo 0; return; }
    find "$od" -maxdepth 1 -name "${cs}.${var}.*.nc" -size +0c 2>/dev/null | wc -l | tr -d ' '
}

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
cmd_status() {
    local od cs
    od=$(out_dir); cs=$(case_name)
    printf 'output dir : %s\n' "$od"
    printf 'case       : %s\n' "$cs"
    if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
        printf 'supervisor : RUNNING (pid %s)\n' "$(cat "$PID_FILE")"
    else
        printf 'supervisor : not running\n'
    fi
    printf '\n%-6s %8s  %s\n' "var" "chunks" "most recent"
    for var in $VARS; do
        local n newest
        n=$(count_done "$var")
        newest=$(find "$od" -maxdepth 1 -name "${cs}.${var}.*.nc" -size +0c 2>/dev/null \
                 | sort | tail -1 | xargs -r basename)
        printf '%-6s %8s  %s\n' "$var" "$n" "${newest:-none}"
    done
    echo
    echo "--- last lines of ${OUTPUT_LOG} ---"
    tail -5 "$OUTPUT_LOG" 2>/dev/null
}

cmd_stop() {
    touch "$STOP_FILE"
    echo "Stop requested. The supervisor will exit after the chunk in flight."
    echo "To stop immediately instead: kill \$(cat $PID_FILE)"
    echo "Either way, rerunning resumes from the last finished chunk."
}

# ---------------------------------------------------------------------------
# The supervised run for one variable
# ---------------------------------------------------------------------------
run_variable() {
    local var="$1"
    local attempt=0 stalled=0 before after

    while :; do
        if [[ -f "$STOP_FILE" ]]; then
            log "$var: stop requested, not starting another attempt"
            return 130
        fi

        attempt=$((attempt + 1))
        before=$(count_done "$var")
        log "$var: attempt $attempt (chunks done: $before)"

        $PYTHON -m regrid regrid "$var" >> "${LOG_DIR}/regrid_${var}.log" 2>&1
        local rc=$?
        after=$(count_done "$var")

        if [[ $rc -eq 0 ]]; then
            log "$var: COMPLETE after $attempt attempt(s), $after chunk(s)"
            return 0
        fi

        # Interrupted on purpose (Ctrl-C, SIGTERM): do not treat as a failure
        # to retry through.
        if [[ $rc -eq 130 || $rc -eq 143 ]]; then
            log "$var: interrupted (exit $rc); stopping"
            return $rc
        fi

        if [[ "$after" -gt "$before" ]]; then
            stalled=0
            log "$var: exit $rc after completing $((after - before)) chunk(s) -- transient, retrying in ${RETRY_DELAY}s"
        else
            stalled=$((stalled + 1))
            log "$var: exit $rc with NO progress ($stalled/$MAX_STALLED_RETRIES) -- see ${LOG_DIR}/regrid_${var}.log"
            if [[ $stalled -ge $MAX_STALLED_RETRIES ]]; then
                log "$var: ABANDONED after $stalled attempts without progress"
                log "$var: last 15 lines of its log:"
                tail -15 "${LOG_DIR}/regrid_${var}.log"
                return 1
            fi
        fi
        sleep "$RETRY_DELAY"
    done
}

# ---------------------------------------------------------------------------
main() {
    case "${1:-run}" in
        status) cmd_status; exit 0 ;;
        stop)   cmd_stop;   exit 0 ;;
        run)    ;;
        *)      echo "usage: $0 [run|status|stop]" >&2; exit 1 ;;
    esac

    # Everything from here on -- supervisor messages, bash-level errors, the
    # trap message -- goes to OUTPUT_LOG, so `nohup ... > /dev/null 2>&1 &`
    # still leaves a complete record. When attached to a terminal it is teed so
    # you can watch it as well.
    if [[ -t 1 ]]; then
        exec > >(tee -a "$OUTPUT_LOG") 2>&1
    else
        exec >> "$OUTPUT_LOG" 2>&1
    fi

    rm -f "$STOP_FILE"
    echo $$ > "$PID_FILE"
    trap 'log "supervisor received a signal; exiting"; rm -f "$PID_FILE"; exit 143' TERM INT

    log "=== supervisor starting: vars='$VARS' parallel=$PARALLEL ==="
    log "output dir: $(out_dir)"
    local start_ts=$SECONDS failed=""

    if [[ "$PARALLEL" == "1" ]]; then
        # Each concurrent variable holds its own set of intermediates, so this
        # multiplies scratch use by the number of variables. Only worth it if
        # the filesystem has both the space and spare throughput -- the work is
        # I/O bound, so on a saturated link this buys nothing.
        local pids=()
        for var in $VARS; do
            run_variable "$var" & pids+=("$!")
            log "$var: supervised in background as pid ${pids[-1]}"
        done
        local i=0
        for pid in "${pids[@]}"; do
            wait "$pid" || failed="$failed $(echo $VARS | cut -d' ' -f$((i + 1)))"
            i=$((i + 1))
        done
    else
        for var in $VARS; do
            run_variable "$var" || failed="$failed $var"
            [[ -f "$STOP_FILE" ]] && { log "stop requested; not starting further variables"; break; }
        done
    fi

    local elapsed=$((SECONDS - start_ts))
    log "=== finished in $((elapsed / 3600))h $(((elapsed % 3600) / 60))m ==="
    for var in $VARS; do log "  $var: $(count_done "$var") chunk(s)"; done
    rm -f "$PID_FILE"

    if [[ -n "$failed" ]]; then
        log "INCOMPLETE:$failed -- rerun this script to resume"
        exit 1
    fi
    log "All variables complete."
}

main "$@"
