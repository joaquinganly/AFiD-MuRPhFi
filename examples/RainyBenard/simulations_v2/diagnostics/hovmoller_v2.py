"""Hovmoller diagrams of vertical velocity w(y, t) at mid-height z = 0.5 (vx face index NXM/2 in
movie_zcut.h5), all frames, the 8 training runs (setup.txt without "set: test"); and the fraction of (y, t >= 50) with |w| > 0.01. Read-only on the runs."""
import sys
from pathlib import Path
from multiprocessing import Pool
import h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
V2 = Path("/workspace/examples/RainyBenard/simulations_v2/runs"); OUT = Path("/workspace/tmp/v2_checks")
SURF, INK, MUTED = "#fcfcfb", "#2b2b29", "#6b6a66"
DIV = LinearSegmentedColormap.from_list("bwr", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
plt.rcParams.update({"font.size": 9, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": MUTED})
def ra_of(run):
    l = [x.split() for x in (run / "bou.in").read_text().splitlines()]
    i = [k for k, x in enumerate(l) if x and x[0] == "RAYT"][0]
    return float(l[i + 1][0])
def hov(run):
    with h5py.File(run / "outputdir/flowmov/movie_zcut.h5") as h:
        ks = sorted(h["vx"].keys()); mid = h["vx"][ks[0]].shape[1] // 2
        w = np.array([h["vx"][k][:, mid] for k in ks])          # (t, y)
    return run.name, w
runs = sorted((d for d in V2.iterdir() if (d / "bou.in").exists() and "set: test" not in (d / "setup.txt").read_text()), key=ra_of)
if len(runs) != 8:
    sys.exit(f"expected 8 training runs, found {len(runs)}")
if (OUT / "w_hovmoller.png").exists():
    sys.exit(f"{OUT / 'w_hovmoller.png'} exists; not overwritten")
with Pool(len(runs)) as p:
    res = dict(p.map(hov, runs))
fig, ax = plt.subplots(2, 4, figsize=(15, 7.2), sharey=True, facecolor=SURF)
for a, run in zip(ax.ravel(), runs):
    w = res[run.name]; lim = np.percentile(np.abs(w[50:]), 99.5)
    act = np.mean(np.abs(w[50:]) > 0.01)
    print(f"{run.name:<15} max|w(z=.5)| {np.abs(w).max():.3f}  99.5% |w| {lim:.3f}  fraction |w|>0.01 (t>=50) {act:.3f}")
    im = a.imshow(w, origin="lower", extent=(0, 16, 0, w.shape[0] - 1), aspect="auto", cmap=DIV, vmin=-lim, vmax=lim,
                  interpolation="nearest")
    a.set_title(f"Ra {ra_of(run):.3g}   (colour limit +-{lim:.2f})", loc="left", fontsize=9)
    for s in a.spines.values(): s.set_visible(False)
    cb = fig.colorbar(im, ax=a, fraction=.04, pad=.02); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
for a in ax[-1]: a.set_xlabel("y")
for a in ax[:, 0]: a.set_ylabel("t (free-fall units)")
fig.suptitle("Vertical velocity w at mid-height z = 0.5 over the whole run (red = up, blue = down)", x=.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .96)); fig.savefig(OUT / "w_hovmoller.png", dpi=120, facecolor=SURF)
print("ok")
