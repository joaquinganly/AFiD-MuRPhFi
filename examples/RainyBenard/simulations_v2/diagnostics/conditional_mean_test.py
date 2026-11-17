"""Direct test of the reading of the attribution (diagnostics/jacobian_attribution.py; agreed with the user,
2026-11-17; changelog.txt): v033's Q_d CNNs return, approximately, the conditional mean of psi = Lap b given
the patch mean (and the vertical difference), not psi; v034's return psi. Read-only on the trained models.

Notation. psi = Q_d / kappa = Lap b (DNS, Q_d_c.npz; the target of Q_d_b). Interior cell (i, j), rows
1..NZ-2; x its 3 x 3 patch of raw b (periodic in y). Template coordinates (u_ab = P_a(z) P_b(y), as in
jacobian_attribution.py):
  a00 = u00 . x = (1/3) sum_{p,q} b_{i+p, j+q}                      (3 x the patch mean)
  a10 = u10 . x = (1/sqrt6) sum_q (b_{i+1, j+q} - b_{i-1, j+q})    (vertical difference)
Conditional means E[psi | A] (the regression function: the mean-squared-error-optimal function of A,
Hastie, Tibshirani & Friedman 2009, sec. 2.4), estimated by partitioning (regressogram: bin averages;
Wasserman 2006, Example 5.24) on the training sample (every TRAIN_SUB-th training-block frame):
  z     per run, the row profile <psi>_{y,t}(z_i)
  a, ab per run, E[psi | a00], E[psi | a00, a10]          (what Q_d_a can know: b and Ra)
  w_a, w_ab pooled over the training runs, run r weighted by w_r = 1 / <psi^2>_r, the run weight of the
        training loss of Q_d_b ("v034 stencil CNN.py", run_weights; identical in v033): E_w[psi | A]
        minimises sum_r w_r sum_c (f(A) - psi)^2, the loss of a model that sees b only
  Bins: NB quantile bins of a00; within each, NB quantile bins of a10 (equal training counts: a data-
  dependent partition into "statistically equivalent blocks", Gyorfi 2018, Nonparametric Estimations and
  Predictions, lecture notes, sec. 2.1). A bin with fewer than MIN_COUNT training cells takes the a00-only
  value, then the overall mean.
Metrics, per run, on the interior cells c of all validation-block frames (811..1000) of the 8 training runs:
  e(f)    = sqrt( sum_c (f - psi)^2 / sum_c psi^2 )     relative RMS error
  d(f, g) = sqrt( sum_c (f - g)^2 / sum_c psi^2 )        distance between f and g, same scale
  1 - eta^2(Y | A) = sum_c (Y - Ybar_bin(A))^2 / sum_c (Y - Ybar)^2, bins of the run's partition; the fraction
        of the variance of Y not explained by any function of A (eta^2: the correlation ratio, = 1 iff Y is a
        function of A; Encyclopedia of Mathematics, "Correlation ratio"); also at NB_COARSE: if Y is a
        continuous function of A it falls as the bins shrink.
  for f in the estimators and Y = psi, and for psi^ (every seed of v033 and v034, Q_d_a and Q_d_b).
Reading: v033 = E[psi | A] predicts e(psi^) ~ e(E), d(psi^, E) << e(psi^) and 1 - eta^2(psi^ | A) << 1;
v034 = psi predicts e(psi^) << e(E) and 1 - eta^2(psi^ | A) ~ 1 - eta^2(psi | A).
Output: simulations_v2/v034/conditional_mean/ (refuses to overwrite): summary.txt, conditional_mean_data.npz,
        Qd_conditional_mean_errors.png, Qd_conditional_mean_distance.png, Qd_correlation_ratio.png
Usage : python diagnostics/conditional_mean_test.py   (from simulations_v2/, inside the Docker container)
"""
import gc
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SIM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("jacobian_attribution", SIM / "diagnostics" / "jacobian_attribution.py")
ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ja)            # both evaluations, their training scripts, TensorFlow
T = ja.T

TRAIN_SUB = 4                 # every TRAIN_SUB-th training-block frame for the estimators
NB = 48                       # quantile bins of a00, and of a10 within each a00 bin
NB_COARSE = 24                # 1 - eta^2 also with these bins
MIN_COUNT = 20                # fewer training cells in a bin: a00-only value, then the overall mean
QSAMPLE = 4_000_000           # training cells drawn (without replacement) for the pooled bin edges
SEED = 0
OUT_DIR = SIM / "v034" / "conditional_mean"
U00, U10 = ja.TEMPLATES[(0, 0)], ja.TEMPLATES[(1, 0)]
EST = ("z", "a", "ab", "w_a", "w_ab")
BINS = ("a", "ab", "a_c", "ab_c")
MATCH = {"Q_d_a": ("a", "ab"), "Q_d_b": ("w_a", "w_ab")}    # the estimators with a model's information
EST_TEX = {"z": r"$\langle\psi\rangle_{y,t}(z)$", "a": r"$\mathbb{E}[\psi\,|\,a_{00}]$",
           "ab": r"$\mathbb{E}[\psi\,|\,a_{00},a_{10}]$", "w_a": r"$\mathbb{E}_w[\psi\,|\,a_{00}]$",
           "w_ab": r"$\mathbb{E}_w[\psi\,|\,a_{00},a_{10}]$"}
EST_STYLE = {"z": ("0.55", "-", "_"), "a": ("k", "--", "^"), "ab": ("k", "-", "v"),
             "w_a": ("k", "--", "^"), "w_ab": ("k", "-", "v")}
DEF_TEX = (r"$\psi=Q_d/\kappa=\nabla^2 b,\quad a_{00}=\frac{1}{3}\sum_{p,q}b_{i+p,\,j+q},\quad "
           r"a_{10}=\frac{1}{\sqrt{6}}\sum_q\left(b_{i+1,\,j+q}-b_{i-1,\,j+q}\right),\quad "
           r"w_r=1/\langle\psi^2\rangle_r,\quad c\in\mathrm{validation},\ \mathbb{E}\ \mathrm{from\ training}$")
OFFSET_NOTE = Line2D([], [], ls="", label="runs equally spaced; markers offset sideways within a run")


# ============================================================
# 1. Features, partitions, estimators, metrics
# ============================================================

def template_coeff(B, u):
    """sum_m u_m b_{i+p, j+q} at the interior rows of B (n, NZ, NY), periodic in y: (n, NZ - 2, NY)."""
    nz = B.shape[1]
    out = np.zeros((B.shape[0], nz - 2, B.shape[2]))
    for a, p in enumerate((-1, 0, 1)):
        for c, q in enumerate((-1, 0, 1)):
            if u[3 * a + c] != 0.0:
                out += u[3 * a + c] * np.roll(B[:, 1 + p:nz - 1 + p].astype(np.float64), -q, axis=2)
    return out


def features(B):
    """a00 and a10 at the interior cells, flattened in (frame, row, column) order."""
    return template_coeff(B, U00).ravel(), template_coeff(B, U10).ravel()


def bin_index(v, edges):
    """k with edges[k] <= v < edges[k + 1]; values beyond the end edges go to the end bins."""
    return np.searchsorted(edges[1:-1], v, side="right")


def fit_partition(a0, a1, nb):
    """Quantile bins of a0 and, within each, quantile bins of a1 (equal counts of the fitting data)."""
    q = np.linspace(0.0, 1.0, nb + 1)
    e0 = np.quantile(a0, q)
    i0 = bin_index(a0, e0)
    e1 = np.empty((nb, nb + 1))
    for k in range(nb):
        sel = a1[i0 == k]
        e1[k] = np.quantile(sel if sel.size else a1, q)
    return {"nb": nb, "e0": e0, "e1": e1}


def assign(a0, a1, part):
    """Bin of a00 (k0) and of (a00, a10) (k0 nb + k1), int64."""
    nb = part["nb"]
    i0 = bin_index(a0, part["e0"])
    i2 = np.empty(i0.shape, np.int64)
    for k in range(nb):
        m = i0 == k
        i2[m] = k * nb + bin_index(a1[m], part["e1"][k])
    return i0, i2


def run_table(i0, i2, y, nb):
    """Counts and sums of y per bin of both partitions."""
    return {"n1": np.bincount(i0, minlength=nb).astype(np.float64), "s1": np.bincount(i0, weights=y, minlength=nb),
            "n2": np.bincount(i2, minlength=nb * nb).astype(np.float64),
            "s2": np.bincount(i2, weights=y, minlength=nb * nb)}


def fit_estimators(tables, nb, min_count=MIN_COUNT):
    """tables: [(w_r, run_table)]. Bin means sum_r w_r s_r / sum_r w_r n_r of both partitions; a bin with
    fewer than min_count cells (all runs) takes the a00-only mean, an a00 bin the overall mean."""
    n1, n2 = sum(t["n1"] for _, t in tables), sum(t["n2"] for _, t in tables)
    w1, s1 = sum(w * t["n1"] for w, t in tables), sum(w * t["s1"] for w, t in tables)
    w2, s2 = sum(w * t["n2"] for w, t in tables), sum(w * t["s2"] for w, t in tables)
    ok1, ok2 = n1 >= min_count, n2 >= min_count
    m1 = np.where(ok1, s1 / np.where(ok1, w1, 1.0), s1.sum() / w1.sum())
    m2 = np.where(ok2, s2 / np.where(ok2, w2, 1.0), np.repeat(m1, nb))
    return {"m1": m1, "m2": m2, "n1": n1, "n2": n2, "ok2": ok2}


def rel(f, g, norm):
    """sqrt( sum (f - g)^2 / norm )."""
    return math.sqrt(float(np.sum(np.square(np.asarray(f, np.float64) - g))) / norm)


def unexplained(Y, idx, nbins):
    """1 - eta^2(Y | bins) = sum (Y - bin mean)^2 / sum (Y - mean)^2."""
    Y = np.asarray(Y, np.float64)
    n = np.bincount(idx, minlength=nbins)
    m = np.bincount(idx, weights=Y, minlength=nbins) / np.maximum(n, 1)
    return float(np.sum(np.square(Y - m[idx])) / np.sum(np.square(Y - Y.mean())))


def prepare_run(label, Bt, Yt, Bv, Yv, nb=NB, nb_coarse=NB_COARSE, min_count=MIN_COUNT):
    """Per-run estimators fitted on the training sample (Bt, Yt) and everything needed on the validation
    frames (Bv, Yv); B raw b, Y = psi, both (n, NZ, NY)."""
    if nb * nb > np.iinfo(np.int16).max:
        raise ValueError("nb too large for int16 bin indices")
    a0t, a1t = features(Bt)
    yt = Yt[:, 1:-1].astype(np.float64)
    profile = yt.mean(axis=(0, 2))
    yt = yt.ravel()
    part = fit_partition(a0t, a1t, nb)
    est = fit_estimators([(1.0, run_table(*assign(a0t, a1t, part), yt, nb))], nb, min_count)
    a0v, a1v = features(Bv)
    i0v, i2v = assign(a0v, a1v, part)
    i0c, i2c = assign(a0v, a1v, fit_partition(a0t, a1t, nb_coarse))
    shape = Yv[:, 1:-1].shape
    psi = Yv[:, 1:-1].astype(np.float64).ravel()
    return {"label": label, "train": (a0t.astype(np.float32), a1t.astype(np.float32), yt.astype(np.float32)),
            "w": 1.0 / float(np.mean(yt ** 2)), "a_val": (a0v.astype(np.float32), a1v.astype(np.float32)),
            "psi": psi.astype(np.float32), "norm": float(np.sum(psi ** 2)), "part": part, "est": est,
            "bins": {"a": (i0v.astype(np.int16), nb), "ab": (i2v.astype(np.int16), nb * nb),
                     "a_c": (i0c.astype(np.int16), nb_coarse), "ab_c": (i2c.astype(np.int16), nb_coarse ** 2)},
            "c": {"z": np.broadcast_to(profile[None, :, None], shape).ravel().astype(np.float32),
                  "a": est["m1"][i0v].astype(np.float32), "ab": est["m2"][i2v].astype(np.float32)},
            "fallback": {"ab": float(np.mean(~est["ok2"][i2v]))}}


def pool(runs, nb=NB, qsample=QSAMPLE, seed=SEED, min_count=MIN_COUNT):
    """Pooled, run-weighted estimators E_w[psi | a00], E_w[psi | a00, a10] on a partition fitted to a random
    sample of the pooled training cells; adds c["w_a"], c["w_ab"] to every run."""
    rng = np.random.default_rng(seed)
    sizes = [r["train"][0].size for r in runs.values()]
    picks = [rng.choice(s, min(s, max(1, round(qsample * s / sum(sizes)))), replace=False) for s in sizes]
    a0 = np.concatenate([r["train"][0][p] for r, p in zip(runs.values(), picks)]).astype(np.float64)
    a1 = np.concatenate([r["train"][1][p] for r, p in zip(runs.values(), picks)]).astype(np.float64)
    part = fit_partition(a0, a1, nb)
    tables = [(r["w"], run_table(*assign(r["train"][0], r["train"][1], part), r["train"][2].astype(np.float64), nb))
              for r in runs.values()]
    est = fit_estimators(tables, nb, min_count)
    for r in runs.values():
        i0v, i2v = assign(*r["a_val"], part)
        r["c"]["w_a"] = est["m1"][i0v].astype(np.float32)
        r["c"]["w_ab"] = est["m2"][i2v].astype(np.float32)
        r["fallback"]["w_ab"] = float(np.mean(~est["ok2"][i2v]))
    return {"part": part, "est": est, "n_sample": int(sum(len(p) for p in picks)), "n_total": int(sum(sizes))}


def estimator_metrics(r):
    return {"e": {k: rel(r["c"][k], r["psi"], r["norm"]) for k in EST},
            "u": {k: unexplained(r["psi"], *r["bins"][k]) for k in BINS}, "w": r["w"], "fallback": r["fallback"]}


def model_metrics(psi_hat, r):
    return {"e": rel(psi_hat, r["psi"], r["norm"]),
            "d": {k: rel(psi_hat, r["c"][k], r["norm"]) for k in EST if k != "z"},
            "u": {k: unexplained(psi_hat, *r["bins"][k]) for k in BINS}}


def psi_from_prediction(p, setup, kappa):
    """Normalised prediction (n, NZ, NY) -> psi^ at the interior cells: (p sigma_y + mu_y) / kappa for
    target Q_d (Q_d_a), without kappa for target Q_d / kappa (Q_d_b)."""
    my, sy = setup["target_norm"]
    target = setup["cfg"]["target"]
    if target not in ("Q_d", "Q_d_over_kappa"):
        raise ValueError(f"unexpected target {target}")
    kap = kappa if target == "Q_d" else 1.0
    return ((np.asarray(p, np.float64) * sy + my) / kap)[:, 1:-1].ravel()


# ============================================================
# 2. Summary and data file
# ============================================================

def seeds_of(res, v, n):
    return sorted({k[2] for k in res if k[0] == v and k[1] == n})


def summary_text(est, res, runs, pooled_info):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    L = ["Conditional-mean test of the Q_d CNNs (v033, v034). psi = Q_d/kappa = Lap b; interior cells of all",
         "validation-block frames of the 8 training runs. a00 = (1/3) sum_pq b_{i+p,j+q}, a10 = (1/sqrt6) sum_q "
         "(b_{i+1,j+q} - b_{i-1,j+q}).",
         f"Estimators fitted on every {TRAIN_SUB}th training-block frame: z = row profile; a, ab = per-run bin means of "
         "psi given a00, (a00, a10);",
         "w_a, w_ab = pooled over runs, weight w_r = 1/<psi^2>_r (the Q_d_b loss weight). "
         f"Bins: {NB} quantile bins of a00 x {NB} of a10 within each;",
         f"min {MIN_COUNT} training cells per bin, else the a00-only value. 'coarse' = {NB_COARSE} x {NB_COARSE}. "
         f"Pooled edges from {pooled_info} cells.",
         "e(f) = sqrt(sum (f - psi)^2 / sum psi^2); d(f, g) = sqrt(sum (f - g)^2 / sum psi^2); "
         "1 - eta^2(Y|A) = sum (Y - bin mean)^2 / sum (Y - mean)^2.", "",
         "Estimators (validation): relative error e, and 1 - eta^2 of psi itself; fallback = fraction of cells in "
         "2D bins below the minimum count",
         f"{'run':<15} {'w_r':>9} " + " ".join(f"{'e_' + k:>9}" for k in EST) + "  "
         + " ".join(f"{'u_' + k:>9}" for k in BINS) + f" {'fb_ab':>7} {'fb_w_ab':>7}"]
    for l in labels:
        s = est[l]
        L.append(f"{l:<15} {s['w']:9.2e} " + " ".join(f"{s['e'][k]:9.3e}" for k in EST) + "  "
                 + " ".join(f"{s['u'][k]:9.3e}" for k in BINS)
                 + f" {s['fallback']['ab']:7.1e} {s['fallback']['w_ab']:7.1e}")
    for v in ja.VERSIONS:
        for n in ja.QD:
            seeds = seeds_of(res, v, n)
            ka, kab = MATCH[n]
            L += ["", f"=== {v} {n}: mean over seeds {seeds} [min, max]; E = {ka} (a00), {kab} (a00, a10)",
                  f"{'run':<15} {'e(psi^)':>27} {'e(E_ab)':>9} {'d(psi^, E_a)':>27} {'d(psi^, E_ab)':>27} "
                  f"{'1-eta2(psi^|a00)':>27} {'1-eta2(psi^|a00,a10)':>27} {'coarse a, ab':>19}"]
            for l in labels:
                rs = [res[(v, n, s, l)] for s in seeds]

                def col(get):
                    x = np.array([get(q) for q in rs])
                    return f"{x.mean():9.3e} [{x.min():7.1e}, {x.max():7.1e}]"
                L.append(f"{l:<15} {col(lambda q: q['e'])} {est[l]['e'][kab]:9.3e} {col(lambda q: q['d'][ka])} "
                         f"{col(lambda q: q['d'][kab])} {col(lambda q: q['u']['a'])} {col(lambda q: q['u']['ab'])} "
                         f"{np.mean([q['u']['a_c'] for q in rs]):9.3e} {np.mean([q['u']['ab_c'] for q in rs]):9.3e}")
    return "\n".join(L) + "\n"


def save_data(est, res, runs, pooled, path):
    d = {"pooled_e0": pooled["part"]["e0"], "pooled_e1": pooled["part"]["e1"], "pooled_m1": pooled["est"]["m1"],
         "pooled_m2": pooled["est"]["m2"], "pooled_n1": pooled["est"]["n1"], "pooled_n2": pooled["est"]["n2"],
         "nb": np.array(NB), "nb_coarse": np.array(NB_COARSE), "min_count": np.array(MIN_COUNT),
         "train_sub": np.array(TRAIN_SUB)}
    for l, r in runs.items():
        d[f"{l}__e0"], d[f"{l}__e1"] = r["part"]["e0"], r["part"]["e1"]
        d[f"{l}__m1"], d[f"{l}__m2"], d[f"{l}__n2"] = r["est"]["m1"], r["est"]["m2"], r["est"]["n2"]
        d[f"{l}__w"] = np.array(est[l]["w"])
        for k in EST:
            d[f"{l}__e_{k}"] = np.array(est[l]["e"][k])
        for k in BINS:
            d[f"{l}__u_psi_{k}"] = np.array(est[l]["u"][k])
        d[f"{l}__frames_train"] = np.array(r["frames"]["train"])
        d[f"{l}__frames_val"] = np.array(r["frames"]["val"])
    for key, q in res.items():
        tag = "__".join(str(x) for x in key)
        d[f"{tag}__e"] = np.array(q["e"])
        for k, x in q["d"].items():
            d[f"{tag}__d_{k}"] = np.array(x)
        for k, x in q["u"].items():
            d[f"{tag}__u_{k}"] = np.array(x)
    np.savez_compressed(path, **d)


# ============================================================
# 3. Figures (formulas, few words)
# ============================================================

def run_ticks(ax, labels, runs):
    ax.set_xticks(np.arange(len(labels)), [ja.ra_tex(l, runs[l]["ra"]).replace(r"\mathrm{Ra}=", "") for l in labels],
                  rotation=45, fontsize=9)
    ax.set_xlabel(r"$\mathrm{Ra}$", fontsize=12)


def plot_errors(est, res, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), sharey=True, layout="constrained")
    for ax, n in zip(axes, ja.QD):
        for k in ("z",) + MATCH[n]:
            c, ls, mk = EST_STYLE[k]
            ax.plot(xs, [est[l]["e"][k] for l in labels], color=c, ls=ls, marker=mk, lw=1.6, ms=7, label=EST_TEX[k])
        for v, o in (("v033", -0.12), ("v034", 0.12)):
            for i, s in enumerate(seeds_of(res, v, n)):
                ax.plot(xs + o, [res[(v, n, s, l)]["e"] for l in labels], ja.MK[n], color=ja.COL[v], ms=6, ls="",
                        label=rf"$\hat\psi$, {v}" if i == 0 else None)
        ax.set_yscale("log")
        run_ticks(ax, labels, runs)
        ax.set_title(ja.MODEL_TEX[n], fontsize=13)
        ax.grid(True, which="major", ls=":", alpha=0.5)
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h + [OFFSET_NOTE], fontsize=9, loc="best")
    axes[0].set_ylabel(r"$e(f)=\sqrt{\sum_c(f-\psi)^2\,/\,\sum_c\psi^2}$", fontsize=12)
    fig.suptitle(DEF_TEX + "\none marker per seed", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_distance(est, res, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 11.0), layout="constrained")
    for j, n in enumerate(ja.QD):
        kab = MATCH[n][1]
        etex = EST_TEX[kab].strip("$")
        allv = []
        for v in ja.VERSIONS:
            P = np.array([(est[l]["e"][kab], res[(v, n, s, l)]["e"], res[(v, n, s, l)]["d"][kab])
                          for s in seeds_of(res, v, n) for l in labels])
            allv.append(P)
            axes[0][j].plot(P[:, 0], P[:, 1], ja.MK[n], color=ja.COL[v], ls="", ms=6, label=v)
            axes[1][j].plot(P[:, 1], P[:, 2], ja.MK[n], color=ja.COL[v], ls="", ms=6, label=v)
        allv = np.concatenate(allv)
        for ax in axes[:, j]:
            lo, hi = allv.min() / 2.0, allv.max() * 2.0
            ax.plot([lo, hi], [lo, hi], "k:", lw=1.2, label=r"$y=x$")
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.grid(True, which="major", ls=":", alpha=0.5)
        axes[0][j].set_title(ja.MODEL_TEX[n], fontsize=13)
        axes[0][j].set_xlabel(rf"$e\left({etex}\right)$", fontsize=12)
        axes[0][j].set_ylabel(r"$e(\hat\psi)$", fontsize=12)
        axes[1][j].set_xlabel(r"$e(\hat\psi)$", fontsize=12)
        axes[1][j].set_ylabel(rf"$d\left(\hat\psi,\,{etex}\right)$", fontsize=12)
    fig.legend(handles=[Patch(color=ja.COL[v], label=v) for v in ja.VERSIONS]
               + [Line2D([], [], color="k", ls=":", label=r"$y=x$")], loc="outside lower center", ncol=3, fontsize=11)
    fig.suptitle(DEF_TEX + "\n" + r"$e(f)=\sqrt{\sum_c(f-\psi)^2/\sum_c\psi^2},\quad "
                 r"d(f,g)=\sqrt{\sum_c(f-g)^2/\sum_c\psi^2}$; one marker per (seed, run)", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_unexplained(est, res, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    xs = np.arange(len(labels))
    offs = {("v033", "Q_d_a"): -0.27, ("v033", "Q_d_b"): -0.09, ("v034", "Q_d_a"): 0.09, ("v034", "Q_d_b"): 0.27}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), sharey=True, layout="constrained")
    for ax, k, title in zip(axes, ("a", "ab"), (r"$A=a_{00}$", r"$A=(a_{00},\,a_{10})$")):
        ax.plot(xs, [est[l]["u"][k] for l in labels], color="k", lw=1.8, marker="D", ms=6, label=r"$Y=\psi$")
        for (v, n), o in offs.items():
            for i, s in enumerate(seeds_of(res, v, n)):
                ax.plot(xs + o, [res[(v, n, s, l)]["u"][k] for l in labels], ja.MK[n], color=ja.COL[v], ms=5, ls="",
                        label=rf"$Y=\hat\psi$, {v}: " + ja.MODEL_TEX[n] if i == 0 else None)
        ax.set_yscale("log")
        run_ticks(ax, labels, runs)
        ax.set_title(title, fontsize=13)
        ax.grid(True, which="major", ls=":", alpha=0.5)
    axes[0].set_ylabel(r"$1-\eta^2(Y\,|\,A)=\frac{\sum_c\left(Y-\bar Y_{\mathrm{bin}(A)}\right)^2}"
                       r"{\sum_c\left(Y-\bar Y\right)^2}$", fontsize=13)
    h, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=h + [OFFSET_NOTE], loc="outside lower center", ncol=3, fontsize=10)
    fig.suptitle(DEF_TEX + rf"; ${NB}\times{NB}$ bins; one marker per seed", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 4. Main
# ============================================================

def main():
    t_start = time.time()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} exists and is not empty; refusing to overwrite.")
    T.configure_gpu()
    runs = {}
    for label in T.TRAIN_RUNS:
        T.select_config("Q_d_b")                       # input b; target Q_d / kappa = psi
        run = T.load_run(label)
        tr, va = run["parts"]["train"], run["parts"]["val"]
        sub = np.arange(tr.start, tr.stop)[::TRAIN_SUB]
        B, Y = run["fields"][..., 0], run["y"]
        r = prepare_run(label, B[sub], Y[sub], B[va], Y[va])
        r.update(ra=run["ra"], kappa=run["kappa"], xm=run["xm"], params=run["params"],
                 b_val=np.ascontiguousarray(B[va]),
                 frames={"train": [run["frames"]["train"][k - tr.start] for k in sub], "val": run["frames"]["val"]})
        runs[label] = r
        del run, B, Y
        gc.collect()
        print(f"  {label:<15} prepared ({time.time() - t_start:.0f} s)", flush=True)
    pooled = pool(runs)
    for r in runs.values():
        del r["train"], r["a_val"]
    gc.collect()
    est = {l: estimator_metrics(r) for l, r in runs.items()}
    res = {}
    for v in ja.VERSIONS:
        ev = ja.EVS[v]
        for n in ja.QD:
            setup = ev.load_config(n)
            if setup is None:
                sys.exit(f"{v} {n} is not trained")
            for l, r in runs.items():
                X = ja.model_inputs(ev, setup, {"label": l, "b": r["b_val"], "xm": r["xm"], "params": r["params"]})
                for s, model in setup["models"].items():
                    res[(v, n, s, l)] = model_metrics(psi_from_prediction(ja.predict(model, X), setup, r["kappa"]), r)
                    q = res[(v, n, s, l)]
                    print(f"  {v} {n} seed {s} {l:<15} e {q['e']:.3e}  d_ab {q['d'][MATCH[n][1]]:.3e}  "
                          f"1-eta2 ab {q['u']['ab']:.3e}", flush=True)
                del X
            del setup
            ja.tf.keras.backend.clear_session()
            gc.collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = summary_text(est, res, runs, f"{pooled['n_sample']} of the {pooled['n_total']} training")
    (OUT_DIR / "summary.txt").write_text(text)
    print("\n" + text)
    save_data(est, res, runs, pooled, OUT_DIR / "conditional_mean_data.npz")
    plot_errors(est, res, runs, OUT_DIR / "Qd_conditional_mean_errors.png")
    plot_distance(est, res, runs, OUT_DIR / "Qd_conditional_mean_distance.png")
    plot_unexplained(est, res, runs, OUT_DIR / "Qd_correlation_ratio.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return est, res


if __name__ == "__main__":
    main()
