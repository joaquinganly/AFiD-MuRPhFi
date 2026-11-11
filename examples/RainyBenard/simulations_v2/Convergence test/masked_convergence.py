"""Masked convergence test of the Q_c-kink hypothesis (handover doc, section 5.1).

analyse_convergence.py found that at Ra 1e8 the residual of the finite-difference left-hand side,
RMS(cd + A - (Q_d + Q_c)) / RMS(Q_d + Q_c), falls with the frame spacing h = k dt at an observed
order of 1.4-1.8 for the second-order central difference cd2, and that the fourth-order cd4 gives
no gain at small h (tmp/convergence/analyse_convergence.log). Hypothesis: Q_c = (gamma/tau) s H(s),
s = q - q_s, has a kink at s = 0, so d2b/dt2 jumps in a cell where s changes sign. The central
difference's error (h^2/6) d3b/dt3 assumes a smooth b(t) (LeVeque 2007, ch. 1); across the jump
the error is only O(h). If the fraction of such cells grows like h, the domain RMS goes like h^1.5.
Test: the residual of analyse_convergence.py (same run, centre frames and spacings), split into
cells where s has one sign in every frame of the stencil's window (frames n - k .. n + k for cd2,
n - 2k .. n + 2k for cd4) and cells where it changes sign ("crossing" cells). If the hypothesis
holds: the crossing fraction grows ~ h, the crossing cells carry most of the residual, and the
non-crossing cells converge at order ~2 (cd2) and ~4 (cd4) until another error dominates.
Reported per spacing (medians over the centre frames): fraction of crossing cells; contribution
of each subset to the domain RMS, sqrt(sum_subset r^2 / N) / RMS(Q_d + Q_c) (the two add in
quadrature to the total); RMS of r within the crossing cells / RMS(Q_d + Q_c); observed orders
between successive spacings from the medians (as analyse_convergence.py).
Ra 1e6 is not analysed: it is steady and its residual is at round-off (3e-11) at every spacing.
Output: table on stdout; figure tmp/convergence/masked_convergence_<run>.png (refuses to overwrite).
Usage (inside the Docker container): python masked_convergence.py [run name]   (default Ra_1e8_Pr_1)
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

KS = (1, 2, 4, 5, 10, 20, 25, 50, 100)        # frame spacings in solver steps (as analyse_convergence.py)
CENTRE_STEP = 50
VARS = ("temp", "qhum", "vx", "vy")
OUT_DIR = HERE.parents[3] / "tmp" / "convergence"
NAN = float("nan")


def crossing(cum, n, w):
    """Cells whose sign of s is not the same in all frames n - w .. n + w. cum (nt, nz, ny):
    cumulative count over frames of (s > 0), cum[m] = number of frames 0..m with s > 0."""
    c = cum[n + w].astype(np.int32)
    if n - w >= 1:
        c = c - cum[n - w - 1]
    return (c > 0) & (c < 2 * w + 1)


def split(r, mask, s_rms):
    """Contributions of the masked and the other cells to the domain RMS of r, and the RMS of r
    within the masked cells, each divided by s_rms."""
    e_in, e_out = float(np.sum(r[mask] ** 2)), float(np.sum(r[~mask] ** 2))
    within = math.sqrt(e_in / mask.sum()) / s_rms if mask.any() else NAN
    return math.sqrt(e_in / r.size) / s_rms, math.sqrt(e_out / r.size) / s_rms, within


def order(e1, e2, h1, h2):
    return math.log(e2 / e1) / math.log(h2 / h1) if e1 > 0 and e2 > 0 else NAN


def rms(x):
    return math.sqrt(float(np.mean(x * x)))


def main(name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = HERE / name
    out_png = OUT_DIR / f"masked_convergence_{name}.png"
    if out_png.exists():
        sys.exit(f"{out_png} exists; refusing to overwrite")
    tail = (run / "run.log").read_text(errors="replace").strip().splitlines()[-1:]
    if tail != ["EXIT 0"]:
        sys.exit(f"{name}: run.log does not end with EXIT 0 ({tail})")
    p = pp.read_bou(run / "bou.in")
    h = pp.read_humid(run / "humid.in")
    zc, dz, dy = pp.read_grid(run / "outputdir/cordin_info.h5", p["nz"], p["ny"], p["ly"])
    kappa = 1.0 / math.sqrt(p["ra"] * p["pr"])
    al, be, ga, tau = h["alpha"], h["beta"], h["gamma"], h["tau"]
    dt = p["tframe"]
    with h5py.File(run / "outputdir/flowmov/movie_zcut.h5", "r") as f:
        nt = len(f["temp"].keys())
        for var in VARS:
            if [int(k) for k in sorted(f[var].keys())] != list(range(nt)):
                raise ValueError(f"{name}: {var} frame keys are not 0 .. {nt - 1}")
        b = np.empty((nt, p["nz"], p["ny"]))
        pos = np.empty((nt, p["nz"], p["ny"]), dtype=bool)
        for m in range(nt):
            b[m] = f["temp"][f"{m:05d}"][()].T                     # movie arrays are (ny, nz)
            q = f["qhum"][f"{m:05d}"][()].T
            pos[m] = (q - pp.saturation(b[m], zc, al, be)) > 0.0
        cum = np.cumsum(pos, axis=0, dtype=np.int16)
        del pos
        kmax = max(KS)
        centres = list(range(2 * kmax, nt - 2 * kmax, CENTRE_STEP))
        rec = {(st, k): {key: [] for key in ("total", "cross", "non", "within", "frac")}
               for st in ("cd2", "cd4") for k in KS}
        for n in centres:
            q = f["qhum"][f"{n:05d}"][()].T
            w = f["vx"][f"{n:05d}"][()].T                          # z-faces incl. both walls
            v = f["vy"][f"{n:05d}"][()].T
            w[0] = 0.0                                             # as in postprocess.py
            w[-1] = 0.0
            qs = pp.saturation(b[n], zc, al, be)
            S = kappa * pp.laplacian(b[n], dz, dy, 0.0, be - 1.0) + pp.condensation(q, qs, ga, tau)
            A = pp.flux_divergence(b[n], w, v, dz, dy)
            sS = rms(S)
            for k in KS:
                hk = k * dt
                cd2 = (b[n + k] - b[n - k]) / (2 * hk)
                cd4 = (-b[n + 2 * k] + 8 * b[n + k] - 8 * b[n - k] + b[n - 2 * k]) / (12 * hk)
                for st, cd, wdw in (("cd2", cd2, k), ("cd4", cd4, 2 * k)):
                    r = cd + A - S
                    mask = crossing(cum, n, wdw)
                    c_in, c_out, within = split(r, mask, sS)
                    d = rec[(st, k)]
                    d["total"].append(rms(r) / sS)
                    d["cross"].append(c_in)
                    d["non"].append(c_out)
                    d["within"].append(within)
                    d["frac"].append(float(mask.mean()))
    med = {key: {f: float(np.nanmedian(v)) if np.isfinite(v).any() else NAN for f, v in d.items()}
           for key, d in rec.items()}
    print(f"{name}: Ra {p['ra']:.3g}, {nt} frames, dt = TFRAME = {dt:g}; centre frames {centres[0]}..{centres[-1]} "
          f"every {CENTRE_STEP} ({len(centres)}); medians over centres; p_x: observed order between successive spacings")
    print("crossing cells: s = q - q_s changes sign within the stencil window (cd2: n-k..n+k, cd4: n-2k..n+2k)")
    print("total, non, cross: RMS contribution to the domain residual / RMS(Q_d + Q_c) (non^2 + cross^2 = total^2);"
          " within: RMS of r in the crossing cells / RMS(Q_d + Q_c)")
    for st in ("cd2", "cd4"):
        print(f"\n{st}\n{'k':>4} {'h':>6} {'frac':>9} {'total':>10} {'non':>10} {'cross':>10} {'within':>10} "
              f"{'p_frac':>7} {'p_tot':>6} {'p_non':>6} {'p_cr':>6} {'p_with':>6}")
        for i, k in enumerate(KS):
            m = med[(st, k)]
            if i:
                m0, k0 = med[(st, KS[i - 1])], KS[i - 1]
                ps = [order(m0[f], m[f], k0, k) for f in ("frac", "total", "non", "cross", "within")]
            else:
                ps = [NAN] * 5
            print(f"{k:>4} {k * dt:>6.2f} {m['frac']:>9.2e} {m['total']:>10.3e} {m['non']:>10.3e} {m['cross']:>10.3e} "
                  f"{m['within']:>10.3e} " + " ".join(f"{x:>6.2f}" for x in ps), flush=True)

    hs = np.array(KS) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, st, slopes in ((axes[0, 0], "cd2", (1.5, 2.0)), (axes[0, 1], "cd4", (1.5, 4.0))):
        for f, col, lab in (("total", "k", "all cells"), ("non", "tab:blue", "non-crossing cells (contribution)"),
                            ("cross", "tab:red", "crossing cells (contribution)")):
            ax.loglog(hs, [med[(st, k)][f] for k in KS], "o-", color=col, label=lab)
        ref = med[(st, 10)]["total"]
        for sl, ls in zip(slopes, (":", "--")):
            ax.loglog(hs, ref * (hs / (10 * dt)) ** sl, ls, color="0.5", label=f"slope {sl:g}")
        ax.set_title(f"{st}: residual / RMS(Q_d + Q_c)", fontsize=10)
    for st, col in (("cd2", "tab:green"), ("cd4", "tab:purple")):
        axes[1, 0].loglog(hs, [med[(st, k)]["frac"] for k in KS], "o-", color=col, label=f"{st} window")
        axes[1, 1].loglog(hs, [med[(st, k)]["within"] for k in KS], "o-", color=col, label=f"{st}")
    for ax, sl in ((axes[1, 0], 1.0), (axes[1, 1], 1.0)):
        ys = [l.get_ydata()[4] for l in ax.get_lines()]
        ax.loglog(hs, ys[0] * (hs / (10 * dt)) ** sl, ":", color="0.5", label=f"slope {sl:g}")
    axes[1, 0].set_title("fraction of crossing cells", fontsize=10)
    axes[1, 1].set_title("RMS residual within the crossing cells / RMS(Q_d + Q_c)", fontsize=10)
    for ax in axes.ravel():
        ax.set_xlabel("frame spacing h (time units; dt = %g)" % dt)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(f"Masked convergence test, {name} (Ra {p['ra']:.3g}): cells where q - q_s changes sign "
                 f"within the stencil window vs the rest", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    print(f"\nfigure: {out_png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Ra_1e8_Pr_1")
