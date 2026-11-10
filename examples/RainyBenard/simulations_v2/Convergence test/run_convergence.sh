#!/usr/bin/env bash
# Convergence test: runs Ra_1e8_Pr_1 and Ra_1e6_Pr_1 in this folder at the same time, NPROC MPI
# processes each (decomposition NPROC x 1, as the source runs). Each restarts from the end state of
# the simulations_v2 training run of the same name and writes a movie frame at every step for
# 10 time units (folders made by make_convergence_runs.py; see each setup.txt).
# Same conventions as simulations_v2/run_queue.sh: per folder provenance.txt and run.log ending in
# "EXIT <code>" (0 = normal end); a folder that already has a run.log or a movie is never touched.
# Expected: about 10.5 GB of movie per run; a few minutes.
#
# Usage, from WSL (not Docker), detached so that closing the terminal does not kill the runs:
#   setsid nohup bash "$HOME/AFiD-MuRPhFi/examples/RainyBenard/simulations_v2/Convergence test/run_convergence.sh" > /dev/null 2>&1 &
# Progress: tail -f "$HOME/AFiD-MuRPhFi/examples/RainyBenard/simulations_v2/Convergence test/convergence.log"
# Optional environment: NPROC (4), AFID (~/AFiD-MuRPhFi/afid), MPIEXEC (mpiexec)
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPROC="${NPROC:-4}"
AFID="${AFID:-$HOME/AFiD-MuRPhFi/afid}"
MPIEXEC="${MPIEXEC:-mpiexec}"
LOG="$HERE/convergence.log"

log() { echo "$(date -Is) $*" | tee -a "$LOG"; }

[ -x "$AFID" ] || { log "afid not found or not executable: $AFID"; exit 1; }
command -v "$MPIEXEC" >/dev/null || { log "launcher not found: $MPIEXEC"; exit 1; }
SRC_COMMIT="$(git -C "$(dirname "$AFID")" rev-parse --short HEAD 2>/dev/null || echo unknown)"
AFID_MD5="$(md5sum "$AFID" | cut -d' ' -f1)"

run_one() {
    local name="$1" d="$HERE/$1" src_md5 rc
    cd "$d" || { log "skip  $name: folder missing"; return 1; }
    if [ -e run.log ]; then log "skip  $name: run.log exists"; return 0; fi
    for f in bou.in humid.in outputdir/continua_master.h5; do
        [ -f "$f" ] || { log "skip  $name: $f missing"; return 1; }
    done
    if [ -e outputdir/flowmov ]; then log "skip  $name: outputdir/flowmov exists"; return 1; fi
    src_md5="$(sed -n 's/.*(md5 \([0-9a-f]*\),.*/\1/p' "$HERE/../runs/$name/provenance.txt" | head -n 1)"
    [ "$src_md5" = "$AFID_MD5" ] || log "note  $name: afid md5 $AFID_MD5 differs from the source run's ($src_md5)"
    {
        echo "start    $(date -Is)"
        echo "host     $(hostname)"
        echo "command  $MPIEXEC -n $NPROC --bind-to none $AFID $NPROC 1"
        echo "afid     $AFID (md5 $AFID_MD5, source commit $SRC_COMMIT)"
    } > provenance.txt
    log "start $name"
    "$MPIEXEC" -n "$NPROC" --bind-to none "$AFID" "$NPROC" 1 > run.log 2>&1
    rc=$?
    echo "EXIT $rc" >> run.log
    echo "end      $(date -Is) exit $rc" >> provenance.txt
    log "done  $name exit $rc"
}

log "convergence test start: NPROC=$NPROC afid=$AFID (md5 $AFID_MD5, commit $SRC_COMMIT); free disk $(df -h "$HERE" | awk 'NR == 2 { print $4 }')"
run_one Ra_1e8_Pr_1 &
sleep 5
run_one Ra_1e6_Pr_1 &
wait
log "convergence test finished"
