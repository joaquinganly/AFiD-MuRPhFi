"""v033 diagnostics: a strong linear baseline for Q_d, the conditioning of its input, and the Q_c
error against supersaturation (agreed with the user, 2026-11-11; changelog.txt).

 1. Q_d baseline. Linear least squares (numpy.linalg.lstsq: SVD-based LAPACK solver, float64) of
    Q_d / kappa on the 3 x 3 patch of b: the 9 values + intercept in the interior rows; the 6
    values of the wall row and its neighbour + intercept in each wall row. Fitted once on a random
    sample of cells of the 8 training runs' training blocks, with b in float32 as the CNN receives
    it. Scored like the CNN (relative RMS error sqrt(sum (p - y)^2 / sum y^2) over all cells of
    each training run's training and validation blocks and of the test runs' frames), next to the
    v033 evaluation (evaluation/per_run_errors.csv) and the best z-profile. It is the model class
    of the CNN's linear part, solved directly instead of by gradient descent (strong-baseline
    practice: McGreivy & Hakim 2024). Also scored: the exact operator of postprocess.py applied
    to the float32 b ([0 1 0; 1 -4 1; 0 1 0] / dz^2 in the interior; in the wall rows centre
    -5 / dz^2 and intercept 2 b_wall / dz^2 from the ghost value 2 b_wall - b), the floor that
    float32 inputs set.
 2. Conditioning of the Q_d input. Covariance of the 9 normalised b values of an interior 3 x 3
    patch (normalisation of the trained Q_d_b models), per training run (training block, random
    sample of cells): eigenvalues, condition number, and the share (L . v_i)^2 of the exact
    Laplacian direction L = [0 1 0 1 -4 1 0 1 0] / sqrt(20) in each eigenvector v_i. For a linear
    unit trained by gradient descent, learning time is governed by the eigenvalues of this matrix
    (LeCun, Kanter & Solla 1991; LeCun, Bottou, Orr & Mueller 1998).
 3. Q_c. (a) The floor set by float32 inputs: (gamma / tau) max(s, 0), s = q - q_s as the CNN
    receives it, against the stored Q_c (both as in postprocess.py). (b) For Q_c_C2 and Q_c_C4:
    histogram of s and the prediction error per bin of s, on the validation blocks of the training
    runs and all frames of the test runs, every seed. (c) s / q_s in condensing cells: the
    relative accuracy that a model given q instead of q - q_s (Q_c_C1) must reach in
    q_s = exp(alpha (b - beta z)).
Output: simulations_v2/v033/diagnostics/ (refuses to overwrite): summary.txt, Qd_*.png, Qc_*.png.
Usage : python diagnostics/v033_diagnostics.py   (from simulations_v2/, inside the Docker container)
"""
import csv
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v033_evaluate", SIM / "v033 evaluate.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)      # imports the training script as EV.T (loader, StencilPad, losses)
T = EV.T
pp = T.pp

OUT_NAME = "diagnostics"
SEED = 0
FIT_CELLS = 250_000              # interior cells per training run for the fit (drawn with replacement)
FIT_WALL_CELLS = 50_000          # cells per wall row per training run
COND_CELLS = 1_000_000           # interior cells per training run for the patch covariance
CHUNK = 32                       # frames per chunk when scoring the baseline
BATCH = 8
QC_NAMES = ("Q_c_C2", "Q_c_C4")
S_MIN, S_MAX, PER_DECADE = 1e-7, 1e2, 4     # bins of s: |s| from S_MIN to S_MAX, central bin |s| < S_MIN
SHARE_EDGES = (1e-5, 1e-4, 1e-3, 1e-2)
KINDS = ("interior", "bottom", "top")
DIS = {"interior": (-1, 0, 1), "bottom": (0, 1), "top": (-1, 0)}   # patch rows relative to the cell
LAPLACIAN = np.array([0., 1., 0., 1., -4., 1., 0., 1., 0.])       # patch order [di + 1, dj + 1]


# ============================================================
# 1. Q_d: linear baseline and conditioning
# ============================================================

def centre_row(kind):
    return 0 if kind == "bottom" else T.NZ - 1


def sets_of(label):
    return {"train": ["train"], "val": ["val"]} if label in T.TRAIN_RUNS else {"test": ["train", "val"]}


def gather(b, t, i, j, dis):
    """Patch values b[t, i + di, (j + dj) % NY], di in dis, dj in (-1, 0, 1): (n, 3 len(dis))."""
    return np.stack([b[t, i + di, (j + dj) % T.NY] for di in dis for dj in (-1, 0, 1)], axis=1)


def sample_cells(b, y, kind, n_cells, rng):
    """Random cells (with replacement) of b, y (n, NZ, NY): patches (float64) and targets."""
    n = b.shape[0]
    t = rng.integers(0, n, n_cells)
    j = rng.integers(0, T.NY, n_cells)
    i = rng.integers(1, T.NZ - 1, n_cells) if kind == "interior" else np.full(n_cells, centre_row(kind))
    return gather(b, t, i, j, DIS[kind]).astype(np.float64), y[t, i, j].astype(np.float64)


def lstsq(X, y):
    """Least squares with an intercept; returns coefficients, intercept, condition number of [X 1]."""
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, _, _, sv = np.linalg.lstsq(A, y, rcond=None)
    return coef[:-1], float(coef[-1]), float(sv[0] / sv[-1])


def to_grid(coef, dis):
    """Coefficients in patch order -> 3 x 3 [di + 1, dj + 1]; NaN for rows beyond a wall."""
    g = np.full((3, 3), np.nan)
    for k, di in enumerate(dis):
        g[di + 1] = coef[3 * k: 3 * k + 3]
    return g


def exact_fit():
    dz2 = (1.0 / T.NZ) ** 2
    f = {"interior": {"grid": EV.EXACT["interior"] / dz2, "intercept": 0.0}}
    for kind, bw in zip(("bottom", "top"), T.B_WALL):
        f[kind] = {"grid": EV.EXACT[kind] / dz2, "intercept": 2.0 * bw / dz2}
    return f


def predict(b, fit):
    """Q_d / kappa on every cell of b (n, NZ, NY) from 3 x 3 stencils; float64."""
    b = b.astype(np.float64)
    p = np.empty_like(b)
    g, c0 = fit["interior"]["grid"], fit["interior"]["intercept"]
    p[:, 1:-1] = c0
    for di in (-1, 0, 1):
        rows = b[:, 1 + di: T.NZ - 1 + di]
        for dj in (-1, 0, 1):
            p[:, 1:-1] += g[di + 1, dj + 1] * np.roll(rows, -dj, axis=2)
    for kind in ("bottom", "top"):
        g, c0 = fit[kind]["grid"], fit[kind]["intercept"]
        r = centre_row(kind)
        acc = np.full(b[:, r].shape, c0)
        for di in DIS[kind]:
            for dj in (-1, 0, 1):
                acc += g[di + 1, dj + 1] * np.roll(b[:, r + di], -dj, axis=1)
        p[:, r] = acc
    return p


def patch_covariance(b, sb, rng):
    """Covariance of the 9 normalised b values of random interior patches; eigen-decomposition."""
    n = b.shape[0]
    t = rng.integers(0, n, COND_CELLS)
    i = rng.integers(1, T.NZ - 1, COND_CELLS)
    j = rng.integers(0, T.NY, COND_CELLS)
    X = gather(b, t, i, j, DIS["interior"]).astype(np.float64) / sb
    C = np.cov(X, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]                   # descending
    L = LAPLACIAN / np.linalg.norm(LAPLACIAN)
    return {"eig": lam, "share": (V.T @ L) ** 2, "cond": float(lam[0] / lam[-1]),
            "lap_var": float(L @ C @ L), "trace": float(np.trace(C))}


def qd_diagnostics():
    setup = EV.load_config("Q_d_b")                  # selects Q_d_b (input b, target Q_d / kappa)
    if setup is None:
        sys.exit("Q_d_b is not trained")
    sb = setup["input_norm"]["b"][1]
    del setup
    rng = np.random.default_rng(SEED)
    data, samples, cond = {}, {k: [] for k in KINDS}, {}
    for label in T.TRAIN_RUNS + T.TEST_RUNS:
        run = T.load_run(label)
        b, y = run["fields"][..., 0], run["y"]
        data[label] = {"b": b, "y": y, "parts": run["parts"], "ra": run["ra"]}
        if label in T.TRAIN_RUNS:
            tr = run["parts"]["train"]
            for kind in KINDS:
                samples[kind].append(sample_cells(b[tr], y[tr], kind,
                                                  FIT_CELLS if kind == "interior" else FIT_WALL_CELLS, rng))
            cond[label] = patch_covariance(b[tr], sb, rng)
            cond[label]["ra"] = run["ra"]
    fit = {}
    for kind in KINDS:
        X = np.concatenate([s[0] for s in samples[kind]])
        y = np.concatenate([s[1] for s in samples[kind]])
        coef, c0, cnd = lstsq(X, y)
        fit[kind] = {"grid": to_grid(coef, DIS[kind]), "intercept": c0, "cond": cnd}
    del samples
    fits = {"fit": fit, "exact": exact_fit()}
    errors = []
    for label, d in data.items():
        for st, parts in sets_of(label).items():
            acc = {k: [0.0, 0.0] for k in fits}
            for part in parts:
                sl = d["parts"][part]
                for k0 in range(sl.start, sl.stop, CHUNK):
                    kk = slice(k0, min(k0 + CHUNK, sl.stop))
                    yy = d["y"][kk].astype(np.float64)
                    sy2 = float(np.sum(yy ** 2))
                    for name, f in fits.items():
                        acc[name][0] += float(np.sum((predict(d["b"][kk], f) - yy) ** 2))
                        acc[name][1] += sy2
            errors.append({"run": label, "set": st, "ra": d["ra"],
                           **{k: math.sqrt(a / c) for k, (a, c) in acc.items()}})
        print(f"  Q_d baseline {label:<15} done", flush=True)
    return {"fit": fit, "errors": errors, "cond": cond, "sb": sb}


# ============================================================
# 2. Q_c: error against supersaturation
# ============================================================

def qc_sets(label):
    return {"val": ["val"]} if label in T.TRAIN_RUNS else {"test": ["train", "val"]}


def s_edges():
    m = int(round(math.log10(S_MAX / S_MIN))) * PER_DECADE
    pos = np.logspace(math.log10(S_MIN), math.log10(S_MAX), m + 1)
    return np.concatenate([-pos[::-1], pos])


def s_bin(s, edges):
    """Bin index of s; values beyond +-S_MAX go to the outer bins."""
    return np.clip(np.searchsorted(edges, s, side="right") - 1, 0, len(edges) - 2)


def s_centres(edges):
    lo, hi = edges[:-1], edges[1:]
    c = np.sign(lo + hi) * np.sqrt(np.abs(lo * hi))
    c[(lo < 0) & (hi > 0)] = 0.0
    return c


def share_within(sse, thr, edges):
    """Share of sum(sse) in the bins inside (-thr, thr); thr is (the nearest) bin edge."""
    half = len(edges) // 2
    m = int(np.argmin(np.abs(edges[half:] - thr)))
    c = half - 1
    tot = float(sse.sum())
    return float(sse[c - m: c + m + 1].sum()) / tot if tot > 0 else float("nan")


def s_half(sse, edges):
    """Smallest bin edge e with at least half of sum(sse) in |s| < e."""
    half = len(edges) // 2
    pos, c, tot = edges[half:], half - 1, float(sse.sum())
    for m in range(len(pos)):
        if sse[c - m: c + m + 1].sum() >= 0.5 * tot:
            return float(pos[m])
    return float("nan")


def qc_diagnostics(name, with_qs):
    setup = EV.load_config(name)
    if setup is None:
        sys.exit(f"{name} is not trained")
    mean, std = setup["target_norm"]
    chans = T.field_channels()
    i_s, i_b = chans.index("q_minus_qs"), chans.index("b")
    edges = s_edges()
    nb = len(edges) - 1
    out = {"name": name, "seeds": setup["seeds"], "s_norm": setup["input_norm"]["q_minus_qs"],
           "edges": edges, "runs": {}}
    for label in T.TRAIN_RUNS + T.TEST_RUNS:
        run = T.load_run(label)
        h = pp.read_humid(T.run_paths(label)["humid"])
        g = h["gamma"] / h["tau"]
        (st, parts), = qc_sets(label).items()
        raw = {p: run["fields"][run["parts"][p]][..., i_s].copy() for p in parts}
        braw = {p: run["fields"][run["parts"][p]][..., i_b].astype(np.float64) for p in parts} if with_qs else None
        T.apply_input_norm(run, setup["input_norm"])
        r = {"ra": run["ra"], "set": st, "g": g, "counts": np.zeros(nb),
             "sse": {s: np.zeros(nb) for s in setup["seeds"]}, "sy2": 0.0, "floor_sse": 0.0,
             "s_pos": [], "s_rel": []}
        z = run["xm"][None, :, None]
        for part in parts:
            gen = T.RunSnapshots([run], part, setup["target_norm"], BATCH, shuffle=False, seed=0)
            t0 = run["parts"][part].start
            for i in range(len(gen)):
                X, _ = gen[i]
                sl = slice(i * BATCH, i * BATCH + len(X))
                y = run["y"][t0 + sl.start: t0 + sl.stop].astype(np.float64)
                s = raw[part][sl]
                idx = s_bin(s, edges).ravel()
                r["counts"] += np.bincount(idx, minlength=nb)
                r["sy2"] += float(np.sum(y ** 2))
                r["floor_sse"] += float(np.sum((g * np.maximum(s.astype(np.float64), 0.0) - y) ** 2))
                for seed, model in setup["models"].items():
                    p = np.asarray(model.predict_on_batch(X))[..., 0].astype(np.float64) * std + mean
                    r["sse"][seed] += np.bincount(idx, weights=((p - y) ** 2).ravel(), minlength=nb)
                c = s > 0
                r["s_pos"].append(s[c].astype(np.float64))
                if with_qs:
                    qs = np.exp(T.ALPHA * (braw[part][sl] - T.BETA * z))
                    r["s_rel"].append(s[c].astype(np.float64) / qs[c])
        r["s_pos"] = np.concatenate(r["s_pos"])
        r["s_rel"] = np.concatenate(r["s_rel"]) if with_qs else None
        r["floor_rel"] = math.sqrt(r["floor_sse"] / r["sy2"])
        r["rel"] = {s: math.sqrt(float(v.sum()) / r["sy2"]) for s, v in r["sse"].items()}
        r["sse_mean"] = np.mean([r["sse"][s] for s in setup["seeds"]], axis=0)
        out["runs"][label] = r
        print(f"  {name:<7} {label:<15} done", flush=True)
        del run, raw, braw
    return out


# ============================================================
# 3. Summary and figures (Q_d and Q_c separate)
# ============================================================

def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def eval_rel(rows, config, run, st, seed=None):
    v = [float(r["rel_rmse"]) for r in rows if r["config"] == config and r["run"] == run
         and r["set"] == st and (seed is None or int(r["seed"]) == seed)]
    return float(np.mean(v)) if v else float("nan")


def eval_ref(rows, config, run, st):
    v = [float(r["ref_rel_rmse"]) for r in rows if r["config"] == config and r["run"] == run and r["set"] == st]
    return v[0] if v else float("nan")


def pct(x, q):
    return float(np.percentile(x, q)) if x.size else float("nan")


def summary_text(qd, qc, rows):
    dz2 = (1.0 / T.NZ) ** 2
    L = [f"{T.VERSION} diagnostics (relative RMS error = sqrt(sum (p - y)^2 / sum y^2) per run and block)", "",
         "1. Q_d / kappa: linear least squares on the 3 x 3 patch of b, coefficients x dz^2. Exact: interior",
         "   [0 1 0; 1 -4 1; 0 1 0]; wall rows centre -5, intercept 2 b_wall "
         f"({2 * T.B_WALL[0]:g} bottom, {2 * T.B_WALL[1]:g} top). Rows z-1, z, z+1; columns y-1, y, y+1.",
         f"   Fitted on {FIT_CELLS} interior cells and {FIT_WALL_CELLS} cells per wall row per training run "
         f"(training blocks, b in float32)."]
    for kind in KINDS:
        f = qd["fit"][kind]
        g = f["grid"] * dz2
        L.append(f"   {kind}: intercept {f['intercept'] * dz2:+.6f}; condition number of [patch, 1] "
                 f"{f['cond']:.3e}")
        for a in range(3):
            L.append("      " + "  ".join("      nan" if np.isnan(g[a, c]) else f"{g[a, c]:+9.6f}"
                                          for c in range(3)))
    L += ["", "   lstsq: this fit; exact f32: the exact operator on the float32 b; Q_d_a, Q_d_b: CNNs "
          "(evaluation, mean of 3 seeds); z-prof: best z-profile",
          f"   {'run':<15} {'set':<5} {'lstsq':>9} {'exact f32':>10} {'Q_d_a':>7} {'Q_d_b':>7} {'z-prof':>7}"]
    for e in qd["errors"]:
        L.append(f"   {e['run']:<15} {e['set']:<5} {e['fit']:>9.2e} {e['exact']:>10.2e} "
                 f"{eval_rel(rows, 'Q_d_a', e['run'], e['set']):>7.3f} "
                 f"{eval_rel(rows, 'Q_d_b', e['run'], e['set']):>7.3f} "
                 f"{eval_ref(rows, 'Q_d_a', e['run'], e['set']):>7.3f}")
    L += ["", f"2. Conditioning of the Q_d input: covariance of the 9 normalised b values (b / {qd['sb']:.4f}) "
          f"of an interior 3 x 3 patch,",
          f"   {COND_CELLS} random cells per training run (training block). Eigenvalues lambda_1 >= ... >= lambda_9;",
          "   share (L.v_i)^2 of the unit Laplacian direction L in eigenvectors 9, 8, 7 (the three smallest);",
          "   variance of the Laplacian combination L.C.L relative to lambda_1."]
    for label, c in qd["cond"].items():
        L.append(f"   {label:<15} cond {c['cond']:.2e}  L.C.L/lambda_1 {c['lap_var'] / c['eig'][0]:.2e}  "
                 f"shares {c['share'][8]:.3f} {c['share'][7]:.3f} {c['share'][6]:.3f}")
        L.append("      eigenvalues " + " ".join(f"{v:.2e}" for v in c["eig"]))
    L += ["", "3. Q_c (validation blocks of the training runs; test runs: all their frames). s = q - q_s.",
          "   floor: (gamma/tau) max(s, 0) from the float32 s against the stored Q_c; rel: CNN relative RMS error",
          "   (mean of the seeds; 'check' = largest relative difference from the evaluation's per_run_errors.csv);",
          "   share |s| < e: share of the CNN's squared error (mean of the seeds) in cells with |s| < e;",
          "   s_50: smallest bin edge e with half of the squared error in |s| < e."]
    for name, res in qc.items():
        mu, sd = res["s_norm"]
        L += ["", f"   {name}: network input (s - {mu:.4f}) / {sd:.4f}; the kink s = 0 is at {-mu / sd:+.4f} "
                  f"in normalised units; 1e-3 in s is {1e-3 / sd:.2e} there"]
        L.append(f"   {'run':<15} {'set':<5} {'frac s>0':>8} {'med s>0':>9} {'p90 s>0':>9} {'floor':>9} "
                 f"{'rel':>9} {'check':>8} " + " ".join(f"{'<' + format(e, '.0e'):>7}" for e in SHARE_EDGES)
                 + f" {'s_50':>8}")
        for label, r in res["runs"].items():
            check = max(abs(v - eval_rel(rows, name, label, r["set"], s)) / eval_rel(rows, name, label, r["set"], s)
                        for s, v in r["rel"].items())
            L.append(f"   {label:<15} {r['set']:<5} {r['s_pos'].size / r['counts'].sum():>8.4f} "
                     f"{pct(r['s_pos'], 50):>9.2e} {pct(r['s_pos'], 90):>9.2e} {r['floor_rel']:>9.2e} "
                     f"{np.mean(list(r['rel'].values())):>9.3e} {check:>8.1e} "
                     + " ".join(f"{share_within(r['sse_mean'], e, res['edges']):>7.3f}" for e in SHARE_EDGES)
                     + f" {s_half(r['sse_mean'], res['edges']):>8.1e}")
    rel = next((res for res in qc.values() if next(iter(res["runs"].values()))["s_rel"] is not None), None)
    if rel is not None:
        L += ["", "4. s / q_s in condensing cells (s > 0): the relative accuracy needed in q_s by a model given q",
              f"   {'run':<15} {'set':<5} {'p10':>9} {'median':>9} {'p90':>9}"]
        for label, r in rel["runs"].items():
            L.append(f"   {label:<15} {r['set']:<5} {pct(r['s_rel'], 10):>9.2e} {pct(r['s_rel'], 50):>9.2e} "
                     f"{pct(r['s_rel'], 90):>9.2e}")
    return "\n".join(L) + "\n"


def ra_colour(ra):
    return plt.cm.viridis(min(max((math.log10(ra) - 6.0) / 2.2, 0.0), 1.0))


SET_MARK = {"train": "o", "val": "s", "test": "*"}


def plot_qd_stencil(qd, path):
    dz2 = (1.0 / T.NZ) ** 2
    f = qd["fit"]
    panels = [("exact, interior", EV.EXACT["interior"], "intercept 0"),
              ("least squares, interior", f["interior"]["grid"] * dz2,
               f"intercept {f['interior']['intercept'] * dz2:+.4f} (exact 0)"),
              ("least squares, bottom wall row", f["bottom"]["grid"] * dz2,
               f"intercept {f['bottom']['intercept'] * dz2:+.4f} (exact {2 * T.B_WALL[0]:g})"),
              ("least squares, top wall row", f["top"]["grid"] * dz2,
               f"intercept {f['top']['intercept'] * dz2:+.4f} (exact {2 * T.B_WALL[1]:g})")]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8))
    for ax, (title, g, sub) in zip(axes, panels):
        ax.imshow(np.nan_to_num(g), origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        for a in range(3):
            for c in range(3):
                ax.text(c, a, "beyond wall" if np.isnan(g[a, c]) else f"{g[a, c]:.4f}",
                        ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1, 2], ["y-1", "y", "y+1"])
        ax.set_yticks([0, 1, 2], ["z-1", "z", "z+1"])
        ax.set_title(f"{title}\n{sub}", fontsize=9)
    fig.suptitle(f"{T.VERSION}: Q_d / kappa from the 3x3 patch of b by linear least squares "
                 f"(coefficients x dz^2; wall rows: exact centre -5)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_errors(qd, rows, path):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    series = [("linear least squares", "k", lambda e: e["fit"]),
              ("exact operator on float32 b", "0.6", lambda e: e["exact"]),
              ("Q_d_a CNN (mean of 3 seeds)", "tab:blue", lambda e: eval_rel(rows, "Q_d_a", e["run"], e["set"])),
              ("Q_d_b CNN (mean of 3 seeds)", "tab:orange", lambda e: eval_rel(rows, "Q_d_b", e["run"], e["set"])),
              ("best z-profile", "tab:red", lambda e: eval_ref(rows, "Q_d_a", e["run"], e["set"]))]
    off = {"train": -0.03, "val": 0.03, "test": 0.0}
    for lab, col, fn in series:
        for st, mk in SET_MARK.items():
            es = [e for e in qd["errors"] if e["set"] == st]
            if es:
                ax.scatter(np.log10([e["ra"] for e in es]) + off[st], [fn(e) for e in es], color=col, marker=mk,
                           s=90 if mk == "*" else 28, label=f"{lab}" if st == "val" else None)
    for st, mk in SET_MARK.items():
        ax.scatter([], [], color="0.3", marker=mk, label={"train": "training block", "val": "validation block",
                                                          "test": "test run"}[st])
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}$ Ra")
    ax.set_ylabel("relative RMS error")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(f"{T.VERSION}: Q_d, linear least-squares baseline vs CNNs (dotted: error of predicting zero)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_conditioning(qd, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    idx = np.arange(1, 10)
    for label, c in qd["cond"].items():
        col = ra_colour(c["ra"])
        axes[0].semilogy(idx, np.abs(c["eig"]) / c["eig"][0], "o-", color=col, label=label)
        axes[1].semilogy(idx, np.maximum(c["share"], 1e-12), "o-", color=col, label=label)
    axes[0].set_title("eigenvalues of the patch covariance / largest", fontsize=10)
    axes[1].set_title("share of the Laplacian direction in each eigenvector", fontsize=10)
    for ax in axes[:2]:
        ax.set_xlabel("eigenvalue index (descending)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=6)
    ra = np.log10([c["ra"] for c in qd["cond"].values()])
    axes[2].semilogy(ra, [c["cond"] for c in qd["cond"].values()], "ko-", label="condition number lambda_1 / lambda_9")
    axes[2].semilogy(ra, [c["eig"][0] / c["lap_var"] for c in qd["cond"].values()], "s--", color="tab:purple",
                     label="lambda_1 / variance of the Laplacian combination")
    axes[2].set_xlabel(r"$\log_{10}$ Ra")
    axes[2].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[2].legend(fontsize=7)
    axes[2].set_title("per training run (training block)", fontsize=10)
    fig.suptitle(f"{T.VERSION}: conditioning of the Q_d input (9 normalised b values of an interior 3x3 patch)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_bins(res, path):
    edges = res["edges"]
    cen = s_centres(edges)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    g = None
    for label, r in res["runs"].items():
        col, ls = ra_colour(r["ra"]), ("--" if r["set"] == "test" else "-")
        n = r["counts"]
        ok = n > 0
        lab = f"{label} ({'test' if r['set'] == 'test' else 'validation'})"
        axes[0].plot(cen[ok], n[ok] / n.sum(), ls, color=col, label=lab)
        axes[1].plot(cen[ok], np.sqrt(r["sse_mean"][ok] / n[ok]), ls, color=col, label=lab)
        axes[2].plot(cen[ok], r["sse_mean"][ok] / r["sse_mean"].sum(), ls, color=col, label=lab)
        g = r["g"]
    pos = cen > 0
    axes[1].plot(cen[pos], g * cen[pos], "k:", linewidth=1.5, label="target (gamma/tau) s")
    titles = ("fraction of cells per bin of s", "RMS error of the prediction per bin (mean of seeds)",
              "share of the run's squared error per bin")
    for ax, t in zip(axes, titles):
        ax.set_xscale("symlog", linthresh=S_MIN, linscale=0.5)
        ax.set_yscale("log")
        ax.axvline(0.0, color="k", linewidth=0.8)
        ax.set_xlabel("s = q - q_s")
        ax.set_title(t, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
    axes[1].legend(fontsize=6)
    mu, sd = res["s_norm"]
    fig.suptitle(f"{T.VERSION}: {res['name']} ({', '.join(T.CONFIGS[res['name']]['inputs'])} -> Q_c), error against "
                 f"supersaturation; network input (s - {mu:.3f}) / {sd:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_relative(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    bins = np.linspace(-9, 1, 81)
    for label, r in res["runs"].items():
        x = np.log10(r["s_rel"][r["s_rel"] > 0])
        if x.size:
            axes[0].hist(x, bins=bins, density=True, histtype="step", color=ra_colour(r["ra"]),
                         linestyle="--" if r["set"] == "test" else "-", label=label)
    axes[0].set_xlabel(r"$\log_{10}(s / q_s)$ in condensing cells")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=6)
    axes[0].set_title("relative supersaturation (validation blocks; test runs dashed)", fontsize=10)
    runs = list(res["runs"].values())
    ra = np.log10([r["ra"] for r in runs])
    axes[1].semilogy(ra, [pct(r["s_pos"], 50) for r in runs], "o", color="tab:blue", label="median s")
    axes[1].semilogy(ra, [pct(r["s_rel"], 50) for r in runs], "s", color="tab:green", label="median s / q_s")
    axes[1].set_xlabel(r"$\log_{10}$ Ra")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].set_title("condensing cells (s > 0)", fontsize=10)
    fig.suptitle(f"{T.VERSION}: supersaturation in condensing cells", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 4. Main
# ============================================================

def main():
    t_start = time.time()
    out = T.OUT_DIR / OUT_NAME
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite.")
    csv_path = T.OUT_DIR / EV.EVAL_DIR_NAME / "per_run_errors.csv"
    if not csv_path.is_file():
        sys.exit(f"{csv_path} is missing; run 'v033 evaluate.py' first")
    rows = read_rows(csv_path)
    T.configure_gpu()
    print("\n=== Q_d: linear baseline and conditioning", flush=True)
    qd = qd_diagnostics()
    qc = {}
    for name in QC_NAMES:
        print(f"\n=== {name}: error against supersaturation", flush=True)
        qc[name] = qc_diagnostics(name, with_qs=(name == QC_NAMES[0]))
    out.mkdir(parents=True, exist_ok=True)
    text = summary_text(qd, qc, rows)
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    plot_qd_stencil(qd, out / "Qd_linear_baseline_stencil.png")
    plot_qd_errors(qd, rows, out / "Qd_linear_baseline_errors.png")
    plot_qd_conditioning(qd, out / "Qd_input_conditioning.png")
    for name in QC_NAMES:
        plot_qc_bins(qc[name], out / f"Qc_{name.split('_')[-1]}_error_vs_supersaturation.png")
    plot_qc_relative(qc[QC_NAMES[0]], out / "Qc_relative_supersaturation.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return {"qd": qd, "qc": qc}


if __name__ == "__main__":
    main()
