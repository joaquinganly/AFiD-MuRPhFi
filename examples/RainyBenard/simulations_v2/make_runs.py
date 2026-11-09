"""Create the simulations_v2 run folders: one folder per (Ra, Pr), named from the same values
that go into its bou.in, so name and parameters cannot disagree.

Two sets, same setup:
  train  RUNS       the 8 training runs (default)
  test   TEST_RUNS  held-out runs for testing the trained model only (never used in training):
                    Ra 8.7e6 between the periodic (7.5e6) and intermittent (1e7) regimes, and
                    Ra 2e8 outside the training range (extrapolation)

bou.in   = examples/RainyBenard/bou.in (template) with only these lines changed:
           grid NXM x NYM x NZM (refined grid set equal), NREAD 0, TMAX, RAYT/PRAT,
           IDTV 0 with DT = DTMAX.
humid.in = copied unchanged from examples/RainyBenard/humid.in.
drizzle.h5 is written afterwards into each folder by tools/drizzle.py (initial drizzle
state + 1e-3 noise added by the solver, NREAD 0).
Refuses to touch an existing folder.
Usage: python make_runs.py [train|test]   (inside the Docker container; paths below)"""
import hashlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

RB = Path("/workspace/examples/RainyBenard")
OUT = RB / "simulations_v2/runs"
RUNS = [(1e6, 1.0), (5e6, 1.0), (7.5e6, 1.0), (1e7, 1.0),
        (1.75e7, 1.0), (2.5e7, 1.0), (5e7, 1.0), (1e8, 1.0)]
TEST_RUNS = [(8.7e6, 1.0), (2e8, 1.0)]
SETS = {"train": (RUNS, "train"),
        "test": (TEST_RUNS, "test (held out: for testing the trained model only, not for training)")}
NXM, NYM, NZM = 128, 2048, 1
TMAX, DT = 1000.0, 1e-2


def fmt(x):
    """1e6 -> '1e6', 7.5e6 -> '7.5e6', 1.75e7 -> '1.75e7', 0.01 -> '1e-2'."""
    m, e = f"{x:.2e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"{m}e{int(e)}"


def label(ra, pr):
    return f"Ra_{fmt(ra)}_Pr_{pr:g}"


def set_line(lines, header, values):
    """Replace the line after the header line whose first token is `header`."""
    idx = [i for i, l in enumerate(lines) if l.split()[:1] == [header]]
    if len(idx) != 1:
        sys.exit(f"template: expected one '{header}' header, found {len(idx)}")
    lines[idx[0] + 1] = values + "\n"


which = sys.argv[1] if len(sys.argv) > 1 else "train"
if which not in SETS or len(sys.argv) > 2:
    sys.exit(f"usage: python make_runs.py [{'|'.join(SETS)}]")
runs, set_note = SETS[which]
template = (RB / "bou.in").read_text()
lines = template.splitlines(keepends=True)
if len(lines) != 55:
    sys.exit(f"template bou.in has {len(lines)} lines, afidtools expects 55")
md5 = hashlib.md5(template.encode()).hexdigest()
all_names = [label(ra, pr) for ra, pr in RUNS + TEST_RUNS]
if len(set(all_names)) != len(all_names):
    sys.exit("duplicate folder names (or a run in both sets)")
targets = [OUT / label(ra, pr) for ra, pr in runs]
existing = [str(d) for d in targets if d.exists()]
if existing:
    sys.exit(f"refusing to overwrite: {existing}")

if not (RB / "humid.in").exists():
    sys.exit("examples/RainyBenard/humid.in not found")
bou = []                                   # all bou.in contents first: no folder is created on a template error
for ra, pr in runs:
    new = list(lines)
    set_line(new, "NXM", f"{NXM}      {NYM}    {NZM}       3")
    set_line(new, "MULTIRES", f"0           {NXM}      {NYM}    {NZM}")
    set_line(new, "NREAD", "0           0")
    set_line(new, "NTST", f"1000000     1e8             {TMAX:.1f}")
    set_line(new, "TOUT", "1.0     1.0     6000.0")
    set_line(new, "RAYT", f"{fmt(ra)}     {pr:.1f}     1e7     1.0     0")
    set_line(new, "IDTV", f"0       {fmt(DT)}    1.e-3   1.0     1e-8    {fmt(DT)}")
    bou.append("".join(new))

for (ra, pr), d, text in zip(runs, targets, bou):
    d.mkdir(parents=True)
    (d / "bou.in").write_text(text)
    shutil.copy2(RB / "humid.in", d / "humid.in")
    (d / "setup.txt").write_text(
        f"created {datetime.now(timezone.utc).isoformat(timespec='seconds')} by simulations_v2/make_runs.py\n"
        f"set: {set_note}\n"
        f"bou.in from examples/RainyBenard/bou.in (md5 {md5}); changed: grid {NXM}x{NYM}x{NZM}, "
        f"refined grid = base, NREAD 0, TMAX {TMAX}, TOUT 1 TFRAME 1 SAVE_3D 6000, "
        f"RAYT {fmt(ra)}, PRAT {pr:.1f}, IDTV 0, DT = DTMAX = {fmt(DT)}\n"
        f"humid.in copied unchanged from examples/RainyBenard/humid.in\n"
        f"drizzle.h5 written by tools/drizzle.py (fld set to this folder); drizzle start (NREAD 0)\n")
    print(d)
