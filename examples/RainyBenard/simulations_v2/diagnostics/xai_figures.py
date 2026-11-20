"""Figures added to the XAI attribution after its runs (agreed with the user, 2026-11-20; changelog.txt). Outputs
go to simulations_v2/v034/xai/, next to those of diagnostics/xai_attribution.py, whose functions are reused; the
figures of xai_attribution.py are kept.

Part npz (redrawn from xai_qd.npz and xai_qc.npz; nothing recomputed):
  Qc_channel_shares_pct.png: the channel-Shapley shares S_ch of xai_attribution.py in per cent, stacked bars per
      run, mean over seeds (Qc_channel_shapley.png showed them on a log axis spanning 40 decades).
  Qd_a_rho_share.png: S_rho of Q_d_a (left panel of Qd_a_channel_attribution.png).
  Qd_a_channel_errors.png: e_0, e_-rho and e_-b of Q_d_a in three panels (right panel of
      Qd_a_channel_attribution.png). e_-ch is the error against the DNS with channel ch at its reference
      (b patch -> bbar, rho = log10 Ra -> rho_bar), the others as in the data.
Part qc_jacobian (computed; read-only on the models):
  J_ch,pq = d Q^_c(i, j) / d x_ch(i + p, j + q), in raw units of channel ch (Q_c per unit of the input), by
  automatic differentiation on the 3 x 3 patch (xai_attribution.Patches), at the explained cells of
  xai_attribution.py (interior sampled cells of the 8 validation-block frames per run) where the DNS condenses
  (Q_c > 0); every Q_c model of v033 and v034 and the exact controls (C1, C2, C4). Per model, seed and run: number
  of cells, median, mean, 5th and 95th percentiles over the cells; "train": the same over the condensing cells of
  the 8 training runs pooled. Outputs qc_jacobian.npz, summary_qc_jacobian.txt, Qc_jacobian_stencils.png (median
  over the pooled training cells, mean over seeds).
Usage (from simulations_v2/, inside the Docker container):
  python diagnostics/xai_figures.py npz | qc_jacobian      (refuses to overwrite existing outputs)
"""
import argparse
import gc
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SIM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("xai_attribution", SIM / "diagnostics" / "xai_attribution.py")
xa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(xa)
ja, T = xa.ja, xa.T
OUT_DIR = xa.OUT_DIR
NPZ_FIGS = ("Qc_channel_shares_pct.png", "Qd_a_rho_share.png", "Qd_a_channel_errors.png")
JAC_OUT = ("qc_jacobian.npz", "summary_qc_jacobian.txt", "Qc_jacobian_stencils.png")
PCT = (5, 50, 95)
TEST_TEX = r"$\dagger\ \mathrm{test}$"


def refuse_existing(paths):
    old = [str(p) for p in paths if Path(p).exists()]
    if old:
        sys.exit("refusing to overwrite: " + ", ".join(old))


# ============================================================
# 1. Figures redrawn from the data files of xai_attribution.py
# ============================================================

def load_results(path):
    """Inverse of xa.save_data: res {(version, config, seed, run): {key: array}}, control {(config, run): {...}}."""
    res, control = {}, {}
    with np.load(path) as d:
        for name in d.files:
            p = name.split("__")
            if len(p) != 5:
                continue
            v, n, s, l, k = p
            if v == "exact":
                control.setdefault((n, l), {})[k] = d[name]
            else:
                res.setdefault((v, n, int(s), l), {})[k] = d[name]
    return res, control


def run_meta(labels):
    return {l: {"ra": float(T.pp.read_bou(T.run_paths(l)["bou"])["ra"])} for l in labels}


def shares_pct(res, control, v, n, labels):
    """(runs, channels) array of 100 S_ch, mean over seeds (exact: the control); None if a run has no entry."""
    if v == "exact":
        rows = [[control[(n, l)]["S_ch"]] if (n, l) in control else [] for l in labels]
    else:
        rows = [[res[(v, n, s, l)]["S_ch"] for s in xa.seeds_of(res, v, n) if (v, n, s, l) in res] for l in labels]
    if not all(rows):
        return None
    return 100.0 * np.array([np.mean(r, axis=0) for r in rows])


def plot_qc_shares_pct(res, control, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    rows, cols = xa.VERSIONS + ("exact",), xa.CONFIGS["qc"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.6 * len(cols), 3.2 * len(rows) + 1.2), sharey=True,
                             squeeze=False, layout="constrained")
    used = set()
    for i, v in enumerate(rows):
        for j, n in enumerate(cols):
            ax = axes[i][j]
            chans = T.CONFIGS[n]["inputs"]
            P = shares_pct(res, control, v, n, labels)
            if P is None:
                ax.axis("off")
                ax.text(0.5, 0.5, r"no exact $Q_c$ ($z$ not an input)" if v == "exact" else "no data", ha="center",
                        va="center", fontsize=10, transform=ax.transAxes)
                continue
            xs = xa.run_axis(ax, labels, runs)
            bottom = np.zeros(len(labels))
            for k, c in enumerate(chans):
                ax.bar(xs, P[:, k], 0.7, bottom=bottom, color=xa.CH_COL[c], edgecolor="w", lw=0.4, zorder=3)
                for x, y0, h in zip(xs, bottom, P[:, k]):
                    if h >= 8.0:
                        ax.text(x, y0 + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7, color="w", zorder=4)
                bottom = bottom + P[:, k]
                used.add(c)
            for x, row in zip(xs, P):
                k = int(np.argmin(row))
                if row[k] < 1.0:
                    ax.text(x, 102, "<0.1" if row[k] < 0.1 else f"{row[k]:.1f}", color=xa.CH_COL[chans[k]],
                            ha="center", va="bottom", fontsize=7, rotation=90)
            ax.set_ylim(0, 118)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_title(f"{v}: {n}", fontsize=11)
        axes[i][0].set_ylabel(r"$100\,S_{ch}\ [\%]$", fontsize=12)
    handles = [Patch(color=xa.CH_COL[c], label=xa.CH_TEX[c]) for c in ("b", "q", "q_minus_qs", "z") if c in used]
    handles += [Line2D([], [], ls="", label=r"above a bar: $\min_{ch}100\,S_{ch}$ if $<1$ (colour of that $ch$)"),
                Patch(color="0.92", label=TEST_TEX)]
    fig.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=10)
    fig.suptitle(r"$S_{ch}=\frac{\sum_c\phi_{ch}^2}{\sum_c\sum_{ch'}\phi_{ch'}^2},\ \phi_{ch}=\sum_{S\subseteq N\setminus\{ch\}}"
                 r"\frac{|S|!\,(n-1-|S|)!}{n!}\left[v(S\cup\{ch\})-v(S)\right]$, $v(S)=\hat Q_c$(channels in $S$ from "
                 r"$\mathbf{x}$, the others at their mean patch); mean over seeds; numbers: $100\,S_{ch}$ where $\geq8$", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def seed_points(ax, xs, res, labels, value):
    for v, o in (("v033", -0.12), ("v034", 0.12)):
        for s in xa.seeds_of(res, v, "Q_d_a"):
            ax.plot(xs + o, [value(res[(v, "Q_d_a", s, l)]) for l in labels], "o", color=xa.VCOL[v], ms=5)


def qd_a_handles():
    return [Line2D([], [], color=xa.VCOL[v], marker="o", ls="", label=f"{v}: " + ja.MODEL_TEX["Q_d_a"] + " (one per seed)")
            for v in xa.VERSIONS] + \
           [Line2D([], [], color="k", ls="--", marker="D", label=r"exact: $\Delta z^2\kappa(\rho)\nabla^2 b/\kappa_c$")]


def plot_qd_a_rho(res, control, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    ir = T.CONFIGS["Q_d_a"]["inputs"].index("log10_Ra")
    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
    xs = xa.run_axis(ax, labels, runs)
    seed_points(ax, xs, res, labels, lambda r: float(r["S_ch"][ir]))
    ax.plot(xs, [float(control[("Q_d_a", l)]["S_ch"][ir]) for l in labels], color="k", ls="--", marker="D", ms=5)
    ax.set_yscale("log")
    ax.set_ylabel(r"$S_\rho=\frac{\sum_c\phi_\rho^2}{\sum_c(\phi_b^2+\phi_\rho^2)}$", fontsize=13)
    fig.legend(handles=qd_a_handles() + [Patch(color="0.92", label=TEST_TEX)], loc="outside lower center", ncol=2,
               fontsize=10)
    fig.suptitle(r"$\phi_\rho=\frac{1}{2}\left[v(\rho)-v(\emptyset)\right]+\frac{1}{2}\left[v(b,\rho)-v(b)\right],"
                 r"\ \rho=\log_{10}\mathrm{Ra}$;  $v(S)=\Psi$(channels in $S$ from $\mathbf{x}$, the others at "
                 r"$\bar{\mathbf{b}},\bar\rho$)", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_a_errors(res, control, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    inputs = T.CONFIGS["Q_d_a"]["inputs"]
    ir, ib = inputs.index("log10_Ra"), inputs.index("b")
    panels = ((None, r"$e_\emptyset$"), (ir, r"$e_{-\rho}:\ \rho\to\bar\rho$"),
              (ib, r"$e_{-b}:\ \mathbf{b}\to\bar{\mathbf{b}}$"))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True, layout="constrained")
    for ax, (key, title) in zip(axes, panels):
        xs = xa.run_axis(ax, labels, runs)
        value = (lambda r, key=key: float(r["e0"]) if key is None else float(r["e_ch"][key]))
        seed_points(ax, xs, res, labels, value)
        ax.plot(xs, [value(control[("Q_d_a", l)]) for l in labels], color="k", ls="--", marker="D", ms=5)
        ax.axhline(1.0, color="0.5", ls="-.", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=13)
    axes[0].set_ylabel(r"$e=\sqrt{\frac{\sum_c\left(\Psi-\Psi_{\mathrm{DNS}}\right)^2}{\sum_c\Psi_{\mathrm{DNS}}^2}}$",
                       fontsize=13)
    handles = qd_a_handles() + [Line2D([], [], color="0.5", ls="-.", label=r"$e=1$: prediction $\Psi=0$"),
                                Patch(color="0.92", label=TEST_TEX)]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle(r"$\Psi=\Delta z^2\hat Q_d/\kappa_c$;  $\bar{\mathbf{b}}$: mean training patch of $b$;  "
                 r"$\bar\rho$: mean $\log_{10}\mathrm{Ra}$ of the training runs", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 2. Jacobian of Q^_c at the condensing cells
# ============================================================

def condensing_jacobian(model, X, norms, inputs, psi_dns):
    """J (n, C, 9), d Q^_c / d x in raw units of each channel, at the explained cells with psi_dns > 0 (in the
    order of xa: row sets, then frame, row, column); patch index m = 3 (p + 1) + (q + 1)."""
    pf = xa.Patches(model)
    (_, sy), fac = norms["target"], norms["fac"]
    scale = np.array([fac * sy / norms["input"][c][1] for c in inputs])[None, :, None]
    row_sets, cols = xa.interior_row_sets(X.shape[1], X.shape[2])
    out = []
    for rows in row_sets:
        wet = xa.cells(psi_dns, rows, cols).reshape(-1) > 0
        if wet.any():
            out.append(pf.grad(xa.patches_of(X, rows, cols).reshape(-1, len(inputs), 9)[wet]) * scale)
    return np.concatenate(out) if out else np.zeros((0, len(inputs), 9))


def describe(J):
    if len(J) == 0:
        nan = np.full(J.shape[1:], np.nan)
        return {"n": 0, "median": nan, "mean": nan, "p5": nan, "p95": nan}
    p5, p50, p95 = np.percentile(J, PCT, axis=0)
    return {"n": len(J), "median": p50, "mean": J.mean(axis=0), "p5": p5, "p95": p95}


def compute(runs):
    stats = {}
    for v in xa.VERSIONS + ("exact",):
        ev = ja.EVS["v034" if v == "exact" else v]
        for n in xa.CONFIGS["qc"]:
            setup = ev.load_config(n)
            if setup is None:
                sys.exit(f"{v} {n} is not trained")
            t = ev.T
            pooled = {}
            for l, run in runs.items():
                if v == "exact":
                    em = xa.exact_model(t, setup, run["gamma_tau"])
                    if em is None:
                        break
                    models = {0: em}
                else:
                    models = setup["models"]
                X = xa.model_inputs(t, setup, run)
                inputs, norms = list(t.INPUTS), xa.norms_of(t, setup, run)
                for s, model in models.items():
                    J = condensing_jacobian(model, X, norms, inputs, run["psi_dns"])
                    stats[(v, n, s, l)] = d = describe(J)
                    if l in T.TRAIN_RUNS:
                        pooled.setdefault(s, []).append(J)
                    print(f"  {v} {n} seed {s} {l:<15} {d['n']:>7d} condensing cells; centre medians "
                          + ", ".join(f"{c} {d['median'][k, 4]:+.4g}" for k, c in enumerate(inputs)), flush=True)
            for s, Js in pooled.items():
                stats[(v, n, s, "train")] = describe(np.concatenate(Js))
            del setup
            xa._PATCH_MODELS.clear()
            tf.keras.backend.clear_session()
            gc.collect()
    return stats


def summary_jacobian(stats, labels):
    L = ["Jacobian of Q^_c at the condensing cells (DNS Q_c > 0) among the explained cells of xai_attribution.py",
         "(interior sampled cells of the 8 validation-block frames per run). J_ch,pq = d Q^_c(i, j) / d x_ch(i + p, j + q)",
         "in raw units of channel ch (Q_c per unit of the input); exact: (gamma/tau) max(q - q_s, 0) through the same code.",
         "Per channel: median of the centre entry J_ch,00 [5th, 95th percentile] and the sum of the 9 medians over the",
         "patch. 'train': the condensing cells of the 8 training runs pooled.", ""]
    for v in xa.VERSIONS + ("exact",):
        for n in xa.CONFIGS["qc"]:
            chans = T.CONFIGS[n]["inputs"]
            for s in sorted({k[2] for k in stats if k[0] == v and k[1] == n}):
                L.append(f"=== {v} {n}" + ("" if v == "exact" else f" seed {s}") + f"   channels {', '.join(chans)}")
                for l in list(labels) + ["train"]:
                    st = stats.get((v, n, s, l))
                    if st is None:
                        continue
                    L.append(f"  {l:<15} n {st['n']:>7d} | " + " | ".join(
                        f"{c} {st['median'][k, 4]:+.4g} [{st['p5'][k, 4]:+.3g}, {st['p95'][k, 4]:+.3g}] "
                        f"sum {np.sum(st['median'][k]):+.4g}" for k, c in enumerate(chans)))
                L.append("")
    return "\n".join(L)


def save_stats(stats, path):
    np.savez_compressed(path, **{f"{v}__{n}__{s}__{l}__{k}": np.asarray(x)
                                 for (v, n, s, l), st in stats.items() for k, x in st.items()})


def panel_value(stats, v, n):
    keys = [k for k in stats if k[0] == v and k[1] == n and k[3] == "train"]
    return np.mean([stats[k]["median"] for k in keys], axis=0) if keys else None


def plot_stencils(stats, path, gamma_tau=20.0):
    rows, cols = xa.CONFIGS["qc"], xa.VERSIONS + ("exact",)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(6.0 * len(cols) + 1.5, 2.6 * len(rows) + 1.4),
                             squeeze=False, layout="constrained")
    cmap = plt.get_cmap("RdBu_r")
    for i, n in enumerate(rows):
        chans = T.CONFIGS[n]["inputs"]
        vals = {v: panel_value(stats, v, n) for v in cols}
        lim = [max([float(np.nanmax(np.r_[np.abs(M[k]), 0.0])) for M in vals.values() if M is not None] + [1e-12])
               for k in range(len(chans))]
        for j, v in enumerate(cols):
            ax = axes[i][j]
            if i == 0:
                ax.set_title(v, fontsize=12)
            M = vals[v]
            if M is None:
                ax.axis("off")
                ax.text(0.5, 0.5, r"no exact $Q_c$ ($z$ not an input)" if v == "exact" else "no data", ha="center",
                        va="center", fontsize=10, transform=ax.transAxes)
                continue
            for k in range(len(chans)):
                ax.imshow(M[k].reshape(3, 3), origin="lower", cmap=cmap, vmin=-lim[k], vmax=lim[k],
                          extent=(4 * k - 0.5, 4 * k + 2.5, -0.5, 2.5))
                for p in range(3):
                    for q in range(3):
                        val = M[k, 3 * p + q]
                        ax.text(4 * k + q, p, f"{val:+.3g}", ha="center", va="center", fontsize=6.5,
                                color="w" if abs(val) > 0.6 * lim[k] else "k")
            ax.set_xlim(-0.5, 4 * len(chans) - 1.5)
            ax.set_ylim(-0.5, 2.5)
            ax.set_xticks([4 * k + 1 for k in range(len(chans))], [xa.CH_TEX[c] for c in chans], fontsize=11)
            ax.set_yticks([0, 1, 2], ["$z-1$", "$z$", "$z+1$"], fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{n}\n" + r"$\hat Q_c(" + r",\,".join(xa.CH_TEX[c].strip("$") for c in chans) + ")$",
                              fontsize=11)
    fig.suptitle(r"$J_{ch,pq}=\mathrm{median}_{c:\,Q_c^{\mathrm{DNS}}>0}\ \frac{\partial\hat Q_c(i,j)}{\partial x_{ch}(i+p,\,j+q)}$"
                 r" (raw units of $x_{ch}$); cells $c$: 8 training runs; mean over seeds; exact: $\gamma/\tau="
                 + f"{gamma_tau:g}" + r"$;"
                 r"  colour: $\pm\max|J_{ch}|$ per row and channel", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 3. Main
# ============================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description="Figures added to the XAI attribution (see the docstring).")
    ap.add_argument("part", choices=("npz", "qc_jacobian"))
    a = ap.parse_args(argv)
    t0 = time.time()
    if a.part == "npz":
        paths = [OUT_DIR / f for f in NPZ_FIGS]
        refuse_existing(paths)
        res_d, ctl_d = load_results(OUT_DIR / "xai_qd.npz")
        res_c, ctl_c = load_results(OUT_DIR / "xai_qc.npz")
        runs = run_meta(sorted({k[3] for k in res_c}))
        plot_qc_shares_pct(res_c, ctl_c, runs, paths[0])
        plot_qd_a_rho(res_d, ctl_d, runs, paths[1])
        plot_qd_a_errors(res_d, ctl_d, runs, paths[2])
    else:
        paths = [OUT_DIR / f for f in JAC_OUT]
        refuse_existing(paths)
        T.configure_gpu()
        labels = T.TRAIN_RUNS + T.TEST_RUNS
        runs = xa.load_part_runs("qc", labels)
        gamma_tau = {r["gamma_tau"] for r in runs.values()}
        if len(gamma_tau) != 1:
            sys.exit(f"gamma/tau differs between runs: {gamma_tau}")
        print(f"Data loaded ({time.time() - t0:.0f} s)", flush=True)
        stats = compute(runs)
        text = summary_jacobian(stats, labels)
        paths[1].write_text(text)
        print("\n" + text)
        save_stats(stats, paths[0])
        plot_stencils(stats, paths[2], gamma_tau.pop())
    print("Written: " + ", ".join(str(p) for p in paths) + f"  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
