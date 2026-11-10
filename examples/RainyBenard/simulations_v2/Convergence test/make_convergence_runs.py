"""Create the convergence-test run folders: restarts of simulations_v2 training runs with a movie
frame at every solver step, to measure how the finite-difference LHS Db/Dt (central differences of
the frames + flux-form advection) converges to Q_d + Q_c as the frame spacing goes from 100 dt to dt.

Runs: Ra 1e6 (steady at t = 1000: a control, Db/Dt ~ 0) and Ra 1e8 (a convective event is under way
at t ~ 995-1000). Per run, <this folder>/Ra_<Ra>_Pr_1/:
  bou.in     the source run's bou.in with only these lines changed:
               NREAD 1, IRESET 1  restart from outputdir/continua_*.h5 with time reset to 0. Needed:
                                  movie frame keys are nint(time / TFRAME) written as 5 digits
                                  (src/h5tools/h5_tools.F90, write_H5_plane); without the reset,
                                  t = 1000.01 and TFRAME = 1e-2 give keys > 99999.
               TMAX 10.0          10 time units = 1000 steps of dt = 1e-2
               TFRAME 1e-2        a frame at every step (same literal as DT; TOUT, SAVE_3D unchanged)
  humid.in   copied unchanged from the source run
  outputdir/continua_*.h5  copied from the source run's outputdir (its end state). The solver
             overwrites them with its own end state when the run finishes; their md5 before the
             run is in setup.txt.
  setup.txt  provenance: source run, md5 of every copied file, changed bou.in lines, time mapping
Movie frame k (key f"{k:05d}") is at t = T0 + k dt of the source run, T0 = time in the source
continua_master.h5. Refuses to touch an existing folder; the source runs are only read.
Usage (inside the Docker container): python "make_convergence_runs.py"
"""
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "runs"
RUNS = ["Ra_1e6_Pr_1", "Ra_1e8_Pr_1"]
TMAX = "10.0"
TFRAME = "1e-2"
CONTINUA = ["continua_master.h5", "continua_vx.h5", "continua_vy.h5", "continua_vz.h5",
            "continua_temp.h5", "continua_pr.h5", "continua_humid.h5"]


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def header_index(lines, header):
    idx = [i for i, l in enumerate(lines) if l.split()[:1] == [header]]
    if len(idx) != 1:
        sys.exit(f"bou.in: expected one '{header}' header, found {len(idx)}")
    return idx[0]


def values(lines, header):
    return lines[header_index(lines, header) + 1].split()


def set_line(lines, header, text):
    lines[header_index(lines, header) + 1] = text + "\n"


targets = [HERE / r for r in RUNS]
existing = [str(d) for d in targets if d.exists()]
if existing:
    sys.exit(f"refusing to overwrite: {existing}")

plans = []                                  # all checks first: no folder is created on an error
for run in RUNS:
    src = SRC / run
    tail = (src / "run.log").read_text(errors="replace").strip().splitlines()[-1:]
    if tail != ["EXIT 0"]:
        sys.exit(f"{src}: run.log does not end with EXIT 0")
    old = (src / "bou.in").read_text()
    lines = old.splitlines(keepends=True)
    idtv = values(lines, "IDTV")            # IDTV DT RESID CFLMAX DTMIN DTMAX
    if idtv[0] != "0" or idtv[1] != TFRAME or idtv[5] != TFRAME:
        sys.exit(f"{src}/bou.in: expected fixed dt (IDTV 0) with DT = DTMAX = {TFRAME}, got {idtv}")
    if values(lines, "NREAD") != ["0", "0"]:
        sys.exit(f"{src}/bou.in: expected NREAD 0 IRESET 0 in the source run")
    nxm, nym, nzm, nsst = (int(x) for x in values(lines, "NXM"))
    with h5py.File(src / "outputdir/continua_master.h5", "r") as f:
        t0 = float(np.ravel(f["time"][()])[0])
        grid = tuple(int(np.ravel(f[k][()])[0]) for k in ("nx", "ny", "nz"))
    if grid != (nxm + 1, nym + 1, nzm + 1):
        sys.exit(f"{src}: continua grid {grid} does not match bou.in {nxm} x {nym} x {nzm}")
    missing = [c for c in CONTINUA if not (src / "outputdir" / c).exists()]
    if missing:
        sys.exit(f"{src}/outputdir: missing {missing}")
    ntst, tout = values(lines, "NTST"), values(lines, "TOUT")
    new = list(lines)
    set_line(new, "NREAD", "1           1")
    set_line(new, "NTST", f"{ntst[0]}     {ntst[1]}             {TMAX}")
    set_line(new, "TOUT", f"{tout[0]}     {TFRAME}     {tout[2]}")
    changed = [f"line {i + 1}: '{a.rstrip()}' -> '{b.rstrip()}'" for i, (a, b) in enumerate(zip(lines, new)) if a != b]
    if len(changed) != 3:
        sys.exit(f"{run}: expected 3 changed bou.in lines, got {changed}")
    plans.append((run, src, old, "".join(new), changed, t0))

me = Path(__file__).resolve()
for run, src, old, text, changed, t0 in plans:
    d = HERE / run
    (d / "outputdir").mkdir(parents=True)
    (d / "bou.in").write_text(text)
    shutil.copy2(src / "humid.in", d / "humid.in")
    copied = []
    for name in ["humid.in"] + [f"outputdir/{c}" for c in CONTINUA]:
        if name != "humid.in":
            shutil.copy2(src / name, d / name)
        a, b = md5(src / name), md5(d / name)
        if a != b:
            sys.exit(f"{d / name}: copy differs from source")
        copied.append(f"  {name}  md5 {a}")
    (d / "setup.txt").write_text("\n".join(
        [f"created {datetime.now(timezone.utc).isoformat(timespec='seconds')} by {me.name} (md5 {md5(me)})",
         "set: convergence test (not for training or testing the model)",
         f"source run: simulations_v2/runs/{run} (bou.in md5 {hashlib.md5(old.encode()).hexdigest()}); "
         f"restart from its end state, t = T0 = {t0!r}",
         "bou.in = source bou.in with only these lines changed:"]
        + [f"  {c}" for c in changed]
        + ["copied unchanged from the source run (outputdir/continua_* are overwritten by the solver at the end of the run):"]
        + copied
        + [f"movie frame k (key k:05d) is at t = T0 + k * {TFRAME} of the source run (time reset to 0 at restart)", ""]))
    print(d)
    for c in changed:
        print("   ", c)
