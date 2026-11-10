"""GIFs of the simulations_v2 runs: one GIF per field, the 8 runs stacked (Ra ascending, top to
bottom), every STRIDE-th frame. Read-only on the runs; writes simulations_v2/figures/v2_<field>.gif.
Fields: b (buoyancy, movie_zcut.h5 'temp'), Q_d, Q_c (Q_d_c.npz), Q = Q_d + Q_c.
Colour limits: b fixed [0, 1.5]; Q_d and Q symmetric, per run, 99.5th percentile of |field| over the
sampled frames, rows 2 .. nz-3 (wall rows exceed it and saturate); Q_c per run, 99.5th percentile
of its positive values.
Usage: python make_gifs.py [--prefix P] [--runs NAME ...] FIELD ...   (FIELD in b Q_d Q_c Q;
default prefix v2, default runs: every run in runs/)"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

V2 = Path("/workspace/examples/RainyBenard/simulations_v2")
OUT = V2 / "figures"
STRIDE, FPS = 5, 10
SURF, INK, MUTED = "#fcfcfb", "#2b2b29", "#6b6a66"
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])
REDS = LinearSegmentedColormap.from_list("reds", ["#f7f6f3", "#f4b9a7", "#e34948", "#8f1f1f"])
LABEL = {"b": "buoyancy b", "Q_d": "Q_d = kappa lap(b)", "Q_c": "Q_c = (gamma/tau)(q - q_s) H(q - q_s)",
         "Q": "Q = Q_d + Q_c"}
plt.rcParams.update({"font.size": 8, "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.edgecolor": MUTED})


def ra_of(run):
    l = [x.split() for x in (run / "bou.in").read_text().splitlines()]
    i = [k for k, x in enumerate(l) if x and x[0] == "RAYT"][0]
    return float(l[i + 1][0])


def frames(run, field, idx):
    """(len(idx), nz, ny) float32 for the sampled frame indices."""
    if field == "b":
        with h5py.File(run / "outputdir/flowmov/movie_zcut.h5", "r") as f:
            return np.stack([f["temp"][f"{n:05d}"][()].T for n in idx]).astype(np.float32)
    with np.load(run / "Q_d_c.npz") as d:
        if field == "Q":
            a = d["Q_d"][:, :, idx] + d["Q_c"][:, :, idx]
        else:
            a = d[field][:, :, idx]
    return np.moveaxis(a, -1, 0).astype(np.float32)


def make(field, runs, prefix):
    with h5py.File(runs[0] / "outputdir/flowmov/movie_zcut.h5", "r") as f:
        nt = len(f["temp"].keys())
    idx = np.arange(0, nt, STRIDE)
    data = [frames(r, field, idx) for r in runs]
    H = 1.2 + 0.8 * len(runs)                       # 7.6 in for 8 runs, as before
    fig, ax = plt.subplots(len(runs), 1, figsize=(11, H), facecolor=SURF, squeeze=False)
    ax = ax[:, 0]
    ims = []
    for a, r, d in zip(ax, runs, data):
        if field == "b":
            cmap, lo, hi = "inferno", 0.0, 1.5
        elif field == "Q_c":
            pos = d[d > 0]
            cmap, lo, hi = REDS, 0.0, float(np.percentile(pos, 99.5)) if pos.size else 1.0
        else:
            L = float(np.percentile(np.abs(d[:, 2:-2]), 99.5))
            cmap, lo, hi = DIV, -L, L
        im = a.imshow(d[0], origin="lower", extent=(0, 16, 0, 1), aspect="auto", cmap=cmap, vmin=lo, vmax=hi,
                      interpolation="antialiased")
        ims.append(im)
        a.set_yticks([])
        a.set_xticks([] if a is not ax[-1] else range(0, 17, 2))
        a.set_ylabel(f"Ra {ra_of(r):.3g}", rotation=0, ha="right", va="center")
        for s in a.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=a, fraction=0.025, pad=0.01, ticks=[lo, 0.5 * (lo + hi), hi])
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=6.5)
        cb.ax.set_yticklabels([f"{x:.2g}" for x in (lo, 0.5 * (lo + hi), hi)])
    ax[-1].set_xlabel("y")
    note = {"b": "colour range 0 - 1.5 for all runs",
            "Q_c": "colour range per run (99.5th percentile of positive values)",
            "Q_d": "colour range per run (99.5th percentile of |Q_d| away from the walls; wall rows saturate)",
            "Q": "colour range per run (99.5th percentile of |Q| away from the walls; wall rows saturate)"}[field]
    title = fig.suptitle("", x=0.01, y=1 - 0.076 / H, ha="left", va="top", fontsize=10)
    fig.text(0.01, 1 - 0.418 / H, f"each strip: 0 <= z <= 1 (vertical), 0 <= y <= 16; {note}", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(left=0.1, right=0.97, top=1 - 0.608 / H, bottom=0.456 / H, hspace=0.12)

    def update(i):
        for im, d in zip(ims, data):
            im.set_data(d[i])
        title.set_text(f"{LABEL[field]}    t = {idx[i]:4d}")
        return ims

    update(0)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{prefix}_{field}.gif"
    if path.exists():
        raise FileExistsError(f"{path} exists; not overwritten")
    tmp = OUT / f"{prefix}_{field}.partial.gif"
    FuncAnimation(fig, update, frames=len(idx), blit=False).save(tmp, writer=PillowWriter(fps=FPS), dpi=90,
                                                                  savefig_kwargs={"facecolor": SURF})
    tmp.rename(path)
    update(len(idx) // 2)
    fig.savefig(OUT / f"{prefix}_{field}_t{idx[len(idx) // 2]}.png", dpi=90, facecolor=SURF)
    plt.close(fig)
    print(f"{path} ({path.stat().st_size / 1e6:.1f} MB, {len(idx)} frames)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v2")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("fields", nargs="+", choices=["b", "Q_d", "Q_c", "Q"])
    a = ap.parse_args()
    runs = sorted((d for d in (V2 / "runs").iterdir() if (d / "bou.in").exists() and (not a.runs or d.name in a.runs)),
                  key=ra_of)
    if a.runs and len(runs) != len(a.runs):
        raise SystemExit(f"runs not found: {set(a.runs) - {d.name for d in runs}}")
    for field in a.fields:
        make(field, runs, a.prefix)
