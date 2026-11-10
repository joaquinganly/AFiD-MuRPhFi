"""Checks of the simulations_v2 batch (8 runs, 128 x 2048, dt = 1e-2, drizzle start, t = 0..1000).
Read-only except the outputs in /workspace/tmp/v2_checks/. Ra, Pr from each run's bou.in.

Per run:
 - run.log: normal end ("time greater than tmax", last line EXIT 0); wall time from provenance.txt
 - cfl.out: maximum CFL (AFiD stops a fixed-dt run above CFLMAX = 1)
 - movie_zcut.h5, every frame: number of frames; b min/max and % of cells with b < -0.01;
   q min/max; grid-scale RMS of b along y (y-wavenumbers k >= NY/4, interior rows);
   cells in runs of >= 3 consecutive alternating extrema (+-+, |2nd difference| > 0.01) along y and z
 - means.h5, every frame: KE = <(vxrms^2 + vyrms^2)/2>_z; moist Nusselt number at bottom and top
   (Nu_m = -(dm/dz at the wall) / (Delta m / H), m = b + gamma q, Delta m = 1, one-sided
   difference wall -> first cell centre); Kolmogorov length eta_K = (nu^3 / max_z eps)^(1/4)
 - resolution: max(dz, dy) / (pi * min eta_K over t >= 10) (Groetzbach: <= 1); cells in the
   thinnest boundary layer delta = H / (2 max Nu_m) over t >= 20 vs 0.35 Ra^0.15 (Shishkina et al.)
 - stationarity: relative change of the mean of KE and Nu_m between t in [500, 750) and [750, 1000]
 - decorrelation of b' = b - <b>_y over the last 200 frames: lag (frames) where correlation < 1/e
Outputs: summary.txt, summary.json, ts_<run>.npz, ke_timeseries.png.
Usage: python check_v2_runs.py [OUT_DIR RUN_NAME ...]  (default: /workspace/tmp/v2_checks, all runs;
an existing OUT_DIR is refused, not overwritten)"""
import json
import math
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

V2 = Path("/workspace/examples/RainyBenard/simulations_v2/runs")
OUT = Path("/workspace/tmp/v2_checks")
GAMMA, M_BOT, M_TOP = 2.0, 2.0, 1.0          # m = b + gamma q; b(0)=0, q(0)=1; b(1)=1, q(1)=0


def read_bou(path):
    lines = [l.split() for l in path.read_text().replace("\r", "").splitlines()]
    g = {l[0]: lines[i + 1] for i, l in enumerate(lines) if l and l[0] in ("NXM", "RAYT", "ALX3", "IDTV")}
    return dict(nxm=int(g["NXM"][0]), nym=int(g["NXM"][1]), ylen=float(g["ALX3"][1]),
                ra=float(g["RAYT"][0]), pr=float(g["RAYT"][1]), dt=float(g["IDTV"][1]))


def alt_runs(b, axis):
    a = b if axis == "z" else b.T
    d = np.diff(a, axis=0)
    ext = (d[:-1] * d[1:] < 0) & (np.abs(d[1:] - d[:-1]) > 0.01)
    return int((ext[:-2] & ext[1:-1] & ext[2:]).sum())


def check(run, out):
    p = read_bou(run / "bou.in")
    nu = math.sqrt(p["pr"] / p["ra"])
    dz, dy = 1.0 / p["nxm"], p["ylen"] / p["nym"]
    log = (run / "run.log").read_text(errors="replace")
    prov = (run / "provenance.txt").read_text()
    cfl = np.loadtxt(run / "outputdir/cfl.out")
    r = dict(run=run.name, **p, normal_end="time greater than tmax" in log,
             exit_line=log.strip().splitlines()[-1],
             start=re.search(r"start\s+(\S+)", prov).group(1), end=re.search(r"end\s+(\S+)", prov).group(1),
             cfl_max=float(cfl[:, 3].max()), cfl_max_t=float(cfl[cfl[:, 3].argmax(), 1]))
    fr = []
    with h5py.File(run / "outputdir/flowmov/movie_zcut.h5") as h:
        keys = sorted(h["temp"].keys())
        last = keys[-200:]
        bp = []
        for k in keys:
            b = h["temp"][k][()].T.astype(np.float64)          # (z, y)
            q = h["qhum"][k][()]
            f = b[1:-1]
            F = np.fft.rfft(f - f.mean(axis=1, keepdims=True), axis=1)
            F[:, :p["nym"] // 4] = 0.0
            grid = float(np.sqrt(np.mean(np.fft.irfft(F, n=p["nym"], axis=1) ** 2)))
            fr.append((int(k), b.min(), b.max(), 100 * np.mean(b < -0.01), q.min(), q.max(), grid,
                       alt_runs(b, "y"), alt_runs(b, "z")))
            if k in last:
                bp.append((b - b.mean(axis=1, keepdims=True)).ravel())
    fr = np.array(fr)
    bp = np.array(bp)
    bp -= bp.mean(axis=0, keepdims=True)
    nrm = np.sqrt((bp ** 2).sum(axis=1))
    tau = None
    for lag in range(1, 150):
        c = np.mean([(bp[i] @ bp[i + lag]) / (nrm[i] * nrm[i + lag] + 1e-300) for i in range(len(bp) - lag)])
        if c < 1 / math.e:
            tau = lag
            break
    with h5py.File(run / "outputdir/means.h5") as me:
        mk = sorted(me["epsilon"].keys())
        t = np.array([int(k) for k in mk], float)
        ke = np.array([np.mean(me["vxrms"][k][()] ** 2 + me["vyrms"][k][()] ** 2) / 2 for k in mk])
        eps = np.array([np.max(me["epsilon"][k][()]) for k in mk])
        m = [np.asarray(me["Tbar"][k][()], float) + GAMMA * np.asarray(me["qbar"][k][()], float) for k in mk]
    nub = np.array([-(mm[0] - M_BOT) / (dz / 2) for mm in m])
    nut = np.array([-(M_TOP - mm[-1]) / (dz / 2) for mm in m])
    with np.errstate(divide="ignore"):
        eta = np.where(eps > 0, (nu ** 3 / np.where(eps > 0, eps, 1)) ** 0.25, np.inf)
    late, l1, l2 = t >= 10, (t >= 500) & (t < 750), t >= 750
    nu_max = float(np.max(np.maximum(nub, nut)[t >= 20]))
    rel = lambda x: float(abs(x[l2].mean() - x[l1].mean()) / abs(x[l1].mean()))
    r.update(
        n_frames=len(fr), t_first=int(fr[0, 0]), t_last=int(fr[-1, 0]),
        b_min=float(fr[1:, 1].min()), b_min_t=int(fr[1:, 0][fr[1:, 1].argmin()]), b_max=float(fr[:, 2].max()),
        pct_b_neg_max=float(fr[1:, 3].max()), q_min=float(fr[1:, 4].min()), q_max=float(fr[:, 5].max()),
        grid_rms_y_median=float(np.median(fr[1:, 6])), grid_rms_y_max=float(fr[1:, 6].max()),
        alt_runs_y_max=int(fr[:, 7].max()), alt_runs_z_max=int(fr[:, 8].max()),
        eta_min=float(eta[late].min()), eta_min_t=float(t[late][eta[late].argmin()]),
        groetzbach=float(max(dz, dy) / (math.pi * eta[late].min())),
        nu_bot_late=float(nub[t >= 500].mean()), nu_top_late=float(nut[t >= 500].mean()),
        nu_max=nu_max, bl_cells=float(1 / (2 * nu_max) / dz), shishkina_min=0.35 * p["ra"] ** 0.15,
        ke_change=rel(ke), nu_top_change=rel(nut), decorr_frames=tau)
    np.savez(out / f"ts_{run.name}.npz", t=t, ke=ke, nub=nub, nut=nut, eta=eta, frames=fr)
    return r


if __name__ == "__main__":
    if len(sys.argv) > 1:
        OUT = Path(sys.argv[1])
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwritten (give a new OUT_DIR)")
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[2:]
    runs = sorted((d for d in V2.iterdir() if (d / "bou.in").exists() and (not names or d.name in names)),
                  key=lambda d: read_bou(d / "bou.in")["ra"])
    if names and len(runs) != len(names):
        sys.exit(f"runs not found: {set(names) - {d.name for d in runs}}")
    with Pool(len(runs)) as pool:
        res = pool.starmap(check, [(r, OUT) for r in runs])
    json.dump(res, open(OUT / "summary.json", "w"), indent=2)
    L = ["run              end   EXIT    wall   frames  CFLmax(t)    b_min(t)      %b<-.01  q_min     b_max  "
         "gridRMS_y med/max   +-+ y/z  eta_min  dx/(pi eta)  Nu_m bot/top(t>=500)  BLcells(min)  dKE   dNu_top  decorr"]
    for r in res:
        from datetime import datetime
        wall = (datetime.fromisoformat(r["end"]) - datetime.fromisoformat(r["start"])).total_seconds() / 3600
        L.append(f"{r['run']:<15} {'ok' if r['normal_end'] else 'NO':<5} {r['exit_line']:<7} {wall:5.2f}h "
                 f"{r['n_frames']:>5}  {r['cfl_max']:.2f}({r['cfl_max_t']:.0f})  {r['b_min']:+.2e}({r['b_min_t']}) "
                 f"{r['pct_b_neg_max']:7.3f}  {r['q_min']:+.1e}  {r['b_max']:.2f}  "
                 f"{r['grid_rms_y_median']:.1e}/{r['grid_rms_y_max']:.1e}   {r['alt_runs_y_max']:>3}/{r['alt_runs_z_max']:<3}  "
                 f"{r['eta_min']:.4f}   {r['groetzbach']:.2f}        {r['nu_bot_late']:.2f}/{r['nu_top_late']:.2f}"
                 f"              {r['bl_cells']:5.1f} (>={r['shishkina_min']:.1f})  {100 * r['ke_change']:4.0f}%  "
                 f"{100 * r['nu_top_change']:4.0f}%    {r['decorr_frames']}")
    txt = "\n".join(L)
    (OUT / "summary.txt").write_text(txt + "\n")
    print(txt)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ncol = min(4, len(res))
    nrow = -(-len(res) // ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.75 * ncol, 3.25 * nrow), sharex=True, facecolor="#fcfcfb", squeeze=False)
    for a, r in zip(ax.ravel(), res):
        d = np.load(OUT / f"ts_{r['run']}.npz")
        a.set_facecolor("#fcfcfb")
        ok = d["ke"] > 0
        a.semilogy(d["t"][ok], d["ke"][ok], color="#2a78d6", lw=0.9)
        a.set_title(f"Ra {r['ra']:.3g}, Pr {r['pr']:g}", fontsize=9, loc="left")
        a.grid(True, color="#e4e3df", lw=0.5)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    for a in ax[-1]:
        a.set_xlabel("t (free-fall units)")
    for a in ax[:, 0]:
        a.set_ylabel("domain-mean KE")
    fig.suptitle("simulations_v2: kinetic energy over time (128 x 2048, dt = 1e-2, drizzle start)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT / "ke_timeseries.png", dpi=140, facecolor="#fcfcfb")
    print("saved", OUT / "ke_timeseries.png")
