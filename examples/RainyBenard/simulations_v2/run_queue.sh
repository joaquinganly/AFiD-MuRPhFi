#!/usr/bin/env bash
# Queue for the simulations_v2 runs.
# Runs every folder in runs/ that has a bou.in and no run.log yet, highest RAYT first,
# NJOBS at a time with NPROC MPI processes each (decomposition NPROC x 1).
# Per run, inside its folder:
#   provenance.txt  start/end time, host, command, afid md5, source commit
#   run.log         solver stdout + stderr, last line "EXIT <code>" (0 = normal end)
# Folders that already have a run.log (finished or interrupted) are never touched.
# To stop starting new runs: touch STOP (in this folder); running jobs finish normally.
#
# Usage (inside tmux, from anywhere):
#   bash ~/AFiD-MuRPhFi/examples/RainyBenard/simulations_v2/run_queue.sh
# Optional environment: NJOBS (2), NPROC (4), AFID (~/AFiD-MuRPhFi/afid), MPIEXEC (mpiexec),
#                       STAGGER (5 s between starts), RUNS_DIR (<this folder>/runs)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNS_DIR="${RUNS_DIR:-$HERE/runs}"
NJOBS="${NJOBS:-2}"
NPROC="${NPROC:-4}"
AFID="${AFID:-$HOME/AFiD-MuRPhFi/afid}"
MPIEXEC="${MPIEXEC:-mpiexec}"
STAGGER="${STAGGER:-5}"      # seconds between job starts
QLOG="$HERE/queue.log"

log() { echo "$(date -Is) $*" | tee -a "$QLOG"; }

[ -x "$AFID" ] || { echo "afid not found or not executable: $AFID"; exit 1; }
command -v "$MPIEXEC" >/dev/null || { echo "launcher not found: $MPIEXEC"; exit 1; }
SRC_COMMIT="${SRC_COMMIT:-$(git -C "$(dirname "$AFID")" rev-parse --short HEAD 2>/dev/null || echo unknown)}"

# Only one queue at a time (a second copy would start the same folders).
exec 9> "$HERE/.queue.lock"
flock -n 9 || { echo "another run_queue.sh is already running (lock $HERE/.queue.lock)"; exit 1; }

# Pending folders: bou.in, humid.in and drizzle.h5 present, no run.log; highest RAYT first.
# (Without drizzle.h5 afid would silently use its blob cold start instead of the drizzle state.)
PENDING=()
while IFS= read -r line; do PENDING+=("$line"); done < <(
    for d in "$RUNS_DIR"/*/; do
        d="${d%/}"
        [ -f "$d/bou.in" ] || continue
        if [ -e "$d/run.log" ]; then
            tail -n 1 "$d/run.log" | grep -q '^EXIT' ||
                echo "$(date -Is) skip $(basename "$d"): run.log without EXIT (interrupted?)" | tee -a "$QLOG" >&2
            continue
        fi
        if [ ! -f "$d/humid.in" ] || [ ! -f "$d/drizzle.h5" ]; then
            echo "$(date -Is) skip $(basename "$d"): humid.in or drizzle.h5 missing" | tee -a "$QLOG" >&2
            continue
        fi
        ra="$(awk '$1 == "RAYT" { getline; print $1; exit }' "$d/bou.in")"
        printf '%s\t%s\n' "$ra" "$d"
    done | sort -t$'\t' -k1,1 -g -r | cut -f2-)

log "queue start: ${#PENDING[@]} pending, NJOBS=$NJOBS NPROC=$NPROC afid=$AFID (commit $SRC_COMMIT)"
log "free disk: $(df -h "$RUNS_DIR" | awk 'NR == 2 { print $4 }')"

run_one() {
    local d="$1" rc
    cd "$d" || return 1
    if [ -e run.log ]; then log "skip  $(basename "$d"): run.log appeared"; return 0; fi
    {
        echo "start    $(date -Is)"
        echo "host     $(hostname)"
        echo "command  $MPIEXEC -n $NPROC --bind-to none $AFID $NPROC 1"
        echo "afid     $AFID (md5 $(md5sum "$AFID" | cut -d' ' -f1), source commit $SRC_COMMIT)"
    } > provenance.txt
    "$MPIEXEC" -n "$NPROC" --bind-to none "$AFID" "$NPROC" 1 > run.log 2>&1
    rc=$?
    echo "EXIT $rc" >> run.log
    echo "end      $(date -Is) exit $rc" >> provenance.txt
    log "done  $(basename "$d") exit $rc"
}

for d in "${PENDING[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$NJOBS" ]; do wait -n; done
    if [ -e "$HERE/STOP" ]; then log "STOP file found: no new runs started"; break; fi
    log "start $(basename "$d")"
    run_one "$d" &
    sleep "$STAGGER"
done
wait
log "queue finished"
