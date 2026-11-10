"""Convergence of the finite-difference LHS Db/Dt to Q_d + Q_c as the frame spacing goes to dt.

Reads the convergence-test runs in this folder (a movie frame at every solver step, dt = 1e-2;
see make_convergence_runs.py) and, with the functions of simulations_v2/postprocess.py, computes
for centre frames n and frame spacings h = k dt:
  S    = Q_d + Q_c at frame n                    RHS (the training target)
  A    = d(w b)/dz + d(v b)/dy at frame n        flux-form advection (the solver's stencil)
  cd2  = (b[n+k] - b[n-k]) / (2 h)               the stencil of the stored 'Q' key (there h = 1)
  cd4  = (-b[n+2k] + 8 b[n+k] - 8 b[n-k] + b[n-2k]) / (12 h)
  rel  = RMS(cd + A - S) / RMS(S)
For a smooth b(t), rel(cd2) ~ C h^2 and rel(cd4) ~ C' h^4 until another error dominates (the
solver's own time stepping, the kink of Q_c at q = q_s, round-off). Observed order between
successive spacings: p = log(rel(h2) / rel(h1)) / log(h2 / h1), from the medians over centres.
Checks first: run.log ends with EXIT 0; movie keys contiguous 0..K for temp, qhum, vx, vy; DT in
bou.in equals TFRAME; frame 0 equals the source run's last movie frame (the restart state).
Writes nothing; summary to stdout.
Usage (inside the Docker container): python analyse_convergence.py [run names]
"""
import math
import sys
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
import postprocess as pp  # noqa: E402

KS = (1, 2, 4, 5, 10, 20, 25, 50, 100)        # frame spacings in solver steps
CENTRE_STEP = 50
VARS = ("temp", "qhum", "vx", "vy")


def rms(x):
    return math.sqrt(float(np.mean(x * x)))


def bou_dt(path):
    lines = [l.split() for l in Path(path).read_text().replace("\r", "").splitlines()]
    i = [j for j, l in enumerate(lines) if l[:1] == ["IDTV"]]
    if len(i) != 1:
        raise ValueError(f"{path}: expected one IDTV header")
    return float(lines[i[0] + 1][1])


def order(e1, e2, h1, h2):
    return math.log(e2 / e1) / math.log(h2 / h1) if e1 > 0 and e2 > 0 else float("nan")


names = sys.argv[1:] or ["Ra_1e8_Pr_1", "Ra_1e6_Pr_1"]
for name in names:
    run = HERE / name
    tail = (run / "run.log").read_text(errors="replace").strip().splitlines()[-1:]
    if tail != ["EXIT 0"]:
        print(f"{name}: run.log does not end with EXIT 0 ({tail}); skipped", flush=True)
        continue
    p = pp.read_bou(run / "bou.in")
    h = pp.read_humid(run / "humid.in")
    zc, dz, dy = pp.read_grid(run / "outputdir/cordin_info.h5", p["nz"], p["ny"], p["ly"])
    kappa = 1.0 / math.sqrt(p["ra"] * p["pr"])
    al, be, ga, tau = h["alpha"], h["beta"], h["gamma"], h["tau"]
    dt = p["tframe"]
    if bou_dt(run / "bou.in") != dt:
        raise ValueError(f"{name}: DT != TFRAME in bou.in")
    with h5py.File(run / "outputdir/flowmov/movie_zcut.h5", "r") as f:
        keys = sorted(f["temp"].keys())
        nt = len(keys)
        for var in VARS:
            if [int(k) for k in sorted(f[var].keys())] != list(range(nt)):
                raise ValueError(f"{name}: {var} frame keys are not 0 .. {nt - 1}")
        steps = round(10.0 / dt)                                      # TMAX 10 (make_convergence_runs.py)
        if nt - 1 not in (steps, steps + 1):                          # one frame per step, keys = nint(time/dt)
            raise ValueError(f"{name}: {nt} frames, expected {steps + 1} or {steps + 2} (one per step)")
        src_movie = SIM / "runs" / name / "outputdir/flowmov/movie_zcut.h5"
        with h5py.File(src_movie, "r") as g:
            last = sorted(g["temp"].keys())[-1]
            diff = {v: float(np.abs(f[v]["00000"][()] - g[v][last][()]).max()) for v in VARS}
        print(f"\n{name}: Ra {p['ra']:.3g}, {nt} frames (keys 00000..{nt - 1:05d}), dt = TFRAME = {dt:g}; "
              f"frame 0 vs source frame {last}: max |diff| " + ", ".join(f"{v} {d:.1e}" for v, d in diff.items())
              + ("  (identical: OK)" if max(diff.values()) == 0.0 else "  (NOT identical: check the restart)"),
              flush=True)

        cache = {}

        def b_at(m):
            if m not in cache:
                cache[m] = f["temp"][f"{m:05d}"][()].T                 # movie arrays are (ny, nz)
            return cache[m]

        kmax = max(KS)
        centres = list(range(2 * kmax, nt - 2 * kmax, CENTRE_STEP))
        rel2 = {k: [] for k in KS}
        rel4 = {k: [] for k in KS}
        s_rms = []
        for n in centres:
            b0 = b_at(n)
            q = f["qhum"][f"{n:05d}"][()].T
            w = f["vx"][f"{n:05d}"][()].T                             # z-faces incl. both walls
            v = f["vy"][f"{n:05d}"][()].T
            if w.shape != (b0.shape[0] + 1, b0.shape[1]) or v.shape != b0.shape:
                raise ValueError(f"{name}: frame {n}: vx {w.shape}, vy {v.shape}, temp {b0.shape}")
            w[0] = 0.0                                                # as in postprocess.py
            w[-1] = 0.0
            qs = pp.saturation(b0, zc, al, be)
            S = kappa * pp.laplacian(b0, dz, dy, 0.0, be - 1.0) + pp.condensation(q, qs, ga, tau)
            A = pp.flux_divergence(b0, w, v, dz, dy)
            sS = rms(S)
            s_rms.append(sS)
            for k in KS:
                hk = k * dt
                cd2 = (b_at(n + k) - b_at(n - k)) / (2 * hk)
                cd4 = (-b_at(n + 2 * k) + 8 * b_at(n + k) - 8 * b_at(n - k) + b_at(n - 2 * k)) / (12 * hk)
                rel2[k].append(rms(cd2 + A - S) / sS)
                rel4[k].append(rms(cd4 + A - S) / sS)
    print(f"centre frames {centres[0]}..{centres[-1]} every {CENTRE_STEP} ({len(centres)}), "
          f"median RMS(Q_d + Q_c) {np.median(s_rms):.3e}; medians (min, max) over centres", flush=True)
    print("p: order from the medians; p~: median over centres of the per-centre order")
    print(f"{'k':>4} {'h':>6} {'rel cd2':>10} {'(min':>9} {'max)':>9} {'p cd2':>6} {'p~':>6} "
          f"{'rel cd4':>10} {'(min':>9} {'max)':>9} {'p cd4':>6} {'p~':>6}")
    m2 = {k: float(np.median(rel2[k])) for k in KS}
    m4 = {k: float(np.median(rel4[k])) for k in KS}
    for i, k in enumerate(KS):
        nan = float("nan")
        p2 = order(m2[KS[i - 1]], m2[k], KS[i - 1], k) if i else nan
        p4 = order(m4[KS[i - 1]], m4[k], KS[i - 1], k) if i else nan
        q2 = float(np.median([order(a, b, KS[i - 1], k) for a, b in zip(rel2[KS[i - 1]], rel2[k])])) if i else nan
        q4 = float(np.median([order(a, b, KS[i - 1], k) for a, b in zip(rel4[KS[i - 1]], rel4[k])])) if i else nan
        print(f"{k:>4} {k * dt:>6.2f} {m2[k]:>10.3e} {min(rel2[k]):>9.2e} {max(rel2[k]):>9.2e} {p2:>6.2f} {q2:>6.2f} "
              f"{m4[k]:>10.3e} {min(rel4[k]):>9.2e} {max(rel4[k]):>9.2e} {p4:>6.2f} {q4:>6.2f}", flush=True)
