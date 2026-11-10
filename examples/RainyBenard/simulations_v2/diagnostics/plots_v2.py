"""Figures for the v2 check: mean moist static energy <m>(t) and time-mean Nu(z) over t in [500, 1000]
(means.h5), and b at t = 1000 (movie_zcut.h5), for the 8 training runs (setup.txt without "set: test").
Read-only on the runs; refuses to overwrite m_budget.png or b_t1000.png in /workspace/tmp/v2_checks."""
import math, sys
from pathlib import Path
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
V2 = Path("/workspace/examples/RainyBenard/simulations_v2/runs"); OUT = Path("/workspace/tmp/v2_checks")
SURF, GRID, INK, MUTED = "#fcfcfb", "#e4e3df", "#2b2b29", "#6b6a66"
STEPS = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
RAMP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": MUTED})
def ra_pr(run):
    l = [x.split() for x in (run / "bou.in").read_text().splitlines()]
    i = [k for k, x in enumerate(l) if x and x[0] == "RAYT"][0]
    return float(l[i + 1][0]), float(l[i + 1][1])
runs = sorted((d for d in V2.iterdir() if (d / "bou.in").exists() and "set: test" not in (d / "setup.txt").read_text()), key=lambda d: ra_pr(d)[0])
if len(runs) != len(STEPS):
    sys.exit(f"expected {len(STEPS)} training runs, found {len(runs)}")
for n in ("m_budget.png", "b_t1000.png"):
    if (OUT / n).exists():
        sys.exit(f"{OUT / n} exists; not overwritten")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, gridspec_kw=dict(width_ratios=[1.6, 1]))
for c, run in zip(STEPS, runs):
    ra, pr = ra_pr(run); kap = 1 / math.sqrt(ra * pr)
    with h5py.File(run / "outputdir/means.h5") as me:
        ks = sorted(me["Tbar"].keys()); g = lambda n: np.array([me[n][k][()] for k in ks], float)
        T, q, wT, wq = g("Tbar"), g("qbar"), g("vxT"), g("vxq")
    t = np.array([int(k) for k in ks], float); nz = T.shape[1]; dz = 1 / nz; z = (np.arange(nz) + .5) * dz
    m = T + 2 * q; L = t >= 500
    Nu = ((wT + 2 * wq - kap * np.gradient(m, dz, axis=1)) / kap)[L].mean(axis=0)
    lab = f"Ra {ra:.3g}"
    a1.plot(t, m.mean(axis=1), color=c, lw=1.6, label=lab)
    a2.plot(Nu, z, color=c, lw=2, label=lab)
for a in (a1, a2):
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=.6)
    for s in ("top", "right"): a.spines[s].set_visible(False)
a1.set_xlabel("t (free-fall units)"); a1.set_ylabel("domain mean of m = b + 2q")
a1.set_title("Mean moist static energy", loc="left", fontsize=10)
a1.axvspan(500, 1000, color=GRID, alpha=.35, lw=0); a1.text(505, a1.get_ylim()[1], "averaging window", va="top", color=MUTED, fontsize=8)
a2.set_xlabel("Nu_m(z) = total m flux / (kappa Delta m / H)"); a2.set_ylabel("z")
a2.set_title("Time-mean Nu_m(z), t in [500, 1000]", loc="left", fontsize=10)
a1.legend(frameon=False, ncol=2, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "m_budget.png", dpi=140, facecolor=SURF); plt.close(fig)
fig, ax = plt.subplots(len(runs), 1, figsize=(13, 0.95 * len(runs) + .6), facecolor=SURF)
for a, run in zip(ax, runs):
    ra, _ = ra_pr(run)
    with h5py.File(run / "outputdir/flowmov/movie_zcut.h5") as h:
        b = h["temp"]["01000"][()].T
    im = a.imshow(b, origin="lower", extent=(0, 16, 0, 1), aspect="auto", cmap=RAMP, vmin=0, vmax=1.5)
    a.set_yticks([]); a.set_xticks([] if a is not ax[-1] else range(0, 17, 2))
    a.set_ylabel(f"Ra {ra:.3g}", rotation=0, ha="right", va="center", fontsize=8.5)
    for s in a.spines.values(): s.set_visible(False)
ax[-1].set_xlabel("y")
fig.suptitle("Buoyancy b at t = 1000 (each strip: 0 <= z <= 1, 0 <= y <= 16)", x=.01, ha="left", fontsize=10)
cb = fig.colorbar(im, ax=ax, fraction=.012, pad=.01); cb.set_label("b"); cb.outline.set_visible(False)
fig.savefig(OUT / "b_t1000.png", dpi=130, facecolor=SURF); plt.close(fig)
print("ok")
