"""Checks of the post-processed simulations_v2 data (read-only; summary to stdout).
Per run:
 - files, keys, shapes (nz, ny, nt), dtype, NaN/Inf; Q_c >= 0; b, q bounds
 - domain buoyancy budget: integrating Db/Dt = Q_d + Q_c over the box removes advection (periodic
   in y, w = 0 at the walls), so d<b>/dt must equal <Q_d + Q_c>. d<b>/dt from the frames by central
   differences (frame spacing 1); compared over t in [50, 1000): correlation and relative RMS
   difference, plus the same for the flux-form finite-difference Q stored as 'Q' in
   <run>_fields_Q.npz (it telescopes to d<b>/dt exactly, so it checks the bookkeeping)
 - pointwise: correlation and relative RMS difference between that finite-difference Q and
   Q_d + Q_c, median over frames t >= 50 (how well 1-time-unit frames resolve Db/Dt)
 - where Q_c > 0 (fraction of points), and RMS of Q_d and Q_c over t >= 50.
Usage: python check_postprocess_v2.py [RUN_NAME ...]   (default: every post-processed run)"""
import math
import sys
from pathlib import Path

import numpy as np

RUNS = Path("/workspace/examples/RainyBenard/simulations_v2/runs")


def ra_of(run):
    l = [x.split() for x in (run / "bou.in").read_text().splitlines()]
    i = [k for k, x in enumerate(l) if x and x[0] == "RAYT"][0]
    return float(l[i + 1][0])


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))


def rel_rms(a, ref):
    return float(np.sqrt(np.mean((a - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))


print(f"{'run':<15} {'nt':>5} {'finite':>6} {'Qc>=0':>5} {'b_min':>9} {'q_min':>9}  "
      f"{'budget corr / relRMS':>21}  {'FD-Q budget':>11}  {'pointwise FD-Q vs Qd+Qc corr / relRMS':>37}  "
      f"{'Qc>0 %':>6} {'rms Qd':>8} {'rms Qc':>8}")
names = sys.argv[1:]
for run in sorted((d for d in RUNS.iterdir() if (d / "postprocess.txt").exists() and (not names or d.name in names)),
                  key=ra_of):
    with np.load(run / "Q_d_c.npz") as d:
        Qd, Qc = d["Q_d"], d["Q_c"]
    with np.load(run / f"{run.name}_fields_Q.npz") as d:
        keys = sorted(d.files)
        b, Qfd = d["theta"], d["Q"]
        shapes = {k: d[k].shape for k in ("u", "v", "rel_q")}
    with np.load(run / f"{run.name}_q.npz") as d:
        q_min = float(d["q"].min())
    nz, ny, nt = b.shape
    assert keys == ["Q", "rel_q", "theta", "u", "v"], keys
    assert all(s == b.shape for s in shapes.values()) and Qd.shape == Qc.shape == b.shape
    assert b.dtype == Qd.dtype == np.float64
    finite = all(np.isfinite(x).all() for x in (Qd, Qc, b, Qfd))
    bm = b.mean(axis=(0, 1))
    dbdt = (bm[2:] - bm[:-2]) / 2.0                                   # frames 1 .. nt-2
    src = (Qd + Qc).mean(axis=(0, 1))[1:-1]
    fdm = Qfd.mean(axis=(0, 1))[1:-1]
    t = np.arange(1, nt - 1)
    L = t >= 50
    pc, pr = [], []
    for n in range(50, nt - 1):
        s = Qd[:, :, n] + Qc[:, :, n]
        pc.append(corr(Qfd[:, :, n], s))
        pr.append(rel_rms(Qfd[:, :, n], s))
    late = slice(50, nt)
    print(f"{run.name:<15} {nt:>5} {str(finite):>6} {str(bool((Qc >= 0).all())):>5} {b.min():>9.2e} {q_min:>9.2e}  "
          f"{corr(dbdt[L], src[L]):>9.3f} / {rel_rms(dbdt[L], src[L]):>8.3f}  {rel_rms(fdm[L], dbdt[L]):>11.1e}  "
          f"{np.median(pc):>17.3f} / {np.median(pr):>17.3f}  "
          f"{100 * np.mean(Qc[:, :, late] > 0):>6.1f} {np.sqrt(np.mean(Qd[:, :, late] ** 2)):>8.2e} "
          f"{np.sqrt(np.mean(Qc[:, :, late] ** 2)):>8.2e}", flush=True)
    del Qd, Qc, b, Qfd
