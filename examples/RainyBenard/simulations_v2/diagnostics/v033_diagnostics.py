"""v033 diagnostics: a strong linear baseline for Q_d, the conditioning of its input, and the Q_c
error against supersaturation (agreed with the user, 2026-11-11; changelog.txt).

Notation. For an interior cell (i, j) of one frame, the patch vector of the 9 normalised b values
in the order [di + 1, dj + 1] (rows z-1, z, z+1; columns y-1, y, y+1):
    x = (b_{i-1,j-1}, b_{i-1,j}, ..., b_{i+1,j+1}) / sigma_b,   sigma_b: the Q_d_b input normalisation.
L = (0, 1, 0, 1, -4, 1, 0, 1, 0), so that Lap b = (sigma_b / dz^2) L.x in the interior; L^ = L / sqrt(20).
C = Cov(x) = sum_k lambda_k v_k v_k^T (lambda_1 >= ... >= lambda_9, v_k orthonormal).
s = q - q_s; the exact target Q_c = (gamma/tau) max(s, 0). Errors e_c = Q^_c - Q_c per cell c = (frame, z, y).

 1. Q_d baseline. Linear least squares (numpy.linalg.lstsq: SVD-based LAPACK solver, float64) of
    Q_d / kappa on the 3 x 3 patch of b: the 9 values + intercept in the interior rows; the 6
    values of the wall row and its neighbour + intercept in each wall row. Fitted once on a random
    sample of cells of the 8 training runs' training blocks, with b in float32 as the CNN receives
    it. Scored like the CNN (relative RMS error sqrt(sum (p - y)^2 / sum y^2) over all cells of
    each training run's training and validation blocks and of the test runs' frames), next to the
    v033 evaluation (evaluation/per_run_errors.csv) and the best z-profile. It is the model class
    of the CNN's linear part, solved directly instead of by gradient descent (strong-baseline
    practice: McGreivy & Hakim 2024). Also scored: the exact operator of postprocess.py applied
    to the float32 b, the floor that float32 inputs set.
 2. Conditioning of the Q_d input, per training run (training block, random interior cells):
    lambda_k / lambda_1; the contribution of each eigen-direction to the variance of the
    Laplacian combination, f_k = lambda_k (L^.v_k)^2 / L^T C L^ (sum_k f_k = 1); lambda_1 / L^T C L^;
    std(L.x) = sqrt(L^T C L) (each x_m has std ~ 1). For a linear unit trained by gradient
    descent, learning time is governed by the eigenvalues of C (LeCun, Kanter & Solla 1991;
    LeCun, Bottou, Orr & Mueller 1998). The eigenvectors v_k as 3 x 3 patterns, each matched to the
    nearest template u_ab = P_a(z) P_b(y) built from the orthonormal polynomials of degree 0, 1, 2 on
    three equally spaced points, P_0 = (1,1,1)/sqrt3, P_1 = (-1,0,1)/sqrt2, P_2 = (1,-2,1)/sqrt6 (mean,
    first and second difference); in this basis L = sqrt2 u_20 + sqrt2 u_02 - 4 u_22.
 3. Q_c, for Q_c_C2 and Q_c_C4 on the validation blocks of the training runs and all evaluated
    frames of the test runs, every seed. (a) Floor: (gamma/tau) max(s, 0) from the float32 s
    against the stored Q_c. (b) The learned function: per linear bin B_k of s, the mean and
    standard deviation of Q^_c over the cells and seeds, against the exact (gamma/tau) max(s, 0),
    and the number of cells N_k. (c) Where the error is: F(x) = sum_{|s_c| < x} e_c^2 / sum_c e_c^2,
    and the fraction of condensing cells G(x) = #{c: 0 < s_c < x} / #{c: s_c > 0}. (d) s / q_s in
    condensing cells: the relative accuracy that a model given q instead of s (Q_c_C1) must reach
    in q_s = exp(alpha (b - beta z)). Figures show REP_RUNS; summary.txt lists every run.
Output: simulations_v2/v033/diagnostics_3/ (refuses to overwrite; earlier figures in v033/diagnostics/
        and v033/diagnostics_2/): summary.txt, diagnostics_data.npz, Qd_*.png, Qc_*.png.
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
from matplotlib.lines import Line2D

SIM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v033_evaluate", SIM / "v033 evaluate.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)      # imports the training script as EV.T (loader, StencilPad, losses)
T = EV.T
pp = T.pp

OUT_NAME = "diagnostics_3"
SEED = 0
FIT_CELLS = 250_000              # interior cells per training run for the fit (drawn with replacement)
FIT_WALL_CELLS = 50_000          # cells per wall row per training run
COND_CELLS = 1_000_000           # interior cells per training run for the patch covariance
CHUNK = 32                       # frames per chunk when scoring the baseline
BATCH = 8
QC_NAMES = ("Q_c_C2", "Q_c_C4")
REP_RUNS = ("Ra_1e6_Pr_1", "Ra_1e7_Pr_1", "Ra_1e8_Pr_1", "Ra_1.5e8_Pr_1")   # shown in the Q_c figures
S_MIN, S_MAX, PER_DECADE = 1e-7, 1e2, 4     # log bins of s: |s| from S_MIN to S_MAX, central bin |s| < S_MIN
LIN_LO, LIN_HI, LIN_BINS = -0.005, 0.01, 150   # linear bins of s for the learned-function figure
SHARE_EDGES = (1e-5, 1e-4, 1e-3, 1e-2)
KINDS = ("interior", "bottom", "top")
DIS = {"interior": (-1, 0, 1), "bottom": (0, 1), "top": (-1, 0)}   # patch rows relative to the cell
LAPLACIAN = np.array([0., 1., 0., 1., -4., 1., 0., 1., 0.])       # patch order [di + 1, dj + 1]
GRAM = (np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0), np.array([-1.0, 0.0, 1.0]) / math.sqrt(2.0),
        np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0))   # orthonormal polynomials of degree 0, 1, 2 on 3 points
TEMPLATES = {(a, b): np.outer(GRAM[a], GRAM[b]).ravel() for a in range(3) for b in range(3)}   # patch order


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
    """Covariance C of the 9 normalised b values of random interior patches; its eigen-decomposition,
    the share (L^.v_k)^2 and variance contribution f_k of each eigen-direction, std(L.x)."""
    n = b.shape[0]
    t = rng.integers(0, n, COND_CELLS)
    i = rng.integers(1, T.NZ - 1, COND_CELLS)
    j = rng.integers(0, T.NY, COND_CELLS)
    X = gather(b, t, i, j, DIS["interior"]).astype(np.float64) / sb
    C = np.cov(X, rowvar=False)
    lam, V = np.linalg.eigh(C)
    lam, V = lam[::-1], V[:, ::-1]                   # descending
    V = V * np.where(V[np.argmax(np.abs(V), axis=0), np.arange(9)] < 0.0, -1.0, 1.0)   # sign: largest entry > 0
    L = LAPLACIAN / np.linalg.norm(LAPLACIAN)
    share = (V.T @ L) ** 2
    lap_var = float(L @ C @ L)                       # = sum_k lambda_k share_k
    return {"eig": lam, "V": V, "share": share, "contrib": lam * share / lap_var, "cond": float(lam[0] / lam[-1]),
            "lap_var": lap_var, "lap_std": math.sqrt(float(LAPLACIAN @ C @ LAPLACIAN)),
            "trace": float(np.trace(C))}


def template_match(v):
    """(a, b, |v.u_ab|) for the template u_ab = P_a(z) P_b(y) nearest to the unit vector v."""
    (a, b), u = max(TEMPLATES.items(), key=lambda kv: abs(float(kv[1] @ v)))
    return a, b, abs(float(u @ v))


def derivative_text(a, b, tex=False):
    """d_z^a d_y^b as text: the derivative that the template u_ab differences (P_1: first, P_2: second)."""
    if a == b == 0:
        return "mean"
    parts = [(rf"\partial_{v}" if tex else f"d_{v}") + (f"^{k}" if k > 1 else "") for v, k in (("z", a), ("y", b)) if k]
    return "$" + "".join(parts) + "$" if tex else " ".join(parts)


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


def cum_within(sse, edges):
    """F at every positive edge e_m: share of sum(sse) in the bins inside (-e_m, e_m)."""
    half = len(edges) // 2
    c, tot = half - 1, float(sse.sum())
    return np.array([sse[c - m: c + m + 1].sum() / tot if tot > 0 else np.nan for m in range(half)])


def share_within(sse, thr, edges):
    """Share of sum(sse) in the bins inside (-thr, thr); thr is (the nearest) bin edge."""
    half = len(edges) // 2
    m = int(np.argmin(np.abs(edges[half:] - thr)))
    return float(cum_within(sse, edges)[m])


def s_half(sse, edges):
    """Smallest bin edge e with at least half of sum(sse) in |s| < e."""
    half = len(edges) // 2
    F = cum_within(sse, edges)
    i = np.flatnonzero(F >= 0.5)
    return float(edges[half:][i[0]]) if i.size else float("nan")


def s_equal(r, edges):
    """Smallest positive bin centre c_k at which the RMS error per bin, sqrt(sse_k / N_k), is not
    larger than the target there, (gamma/tau) c_k."""
    cen = s_centres(edges)
    ok = (cen > 0) & (r["counts"] > 0)
    rms = np.sqrt(r["sse_mean"][ok] / r["counts"][ok])
    i = np.flatnonzero(rms <= r["g"] * cen[ok])
    return float(cen[ok][i[0]]) if i.size else float("nan")


def qc_diagnostics(name, with_qs):
    setup = EV.load_config(name)
    if setup is None:
        sys.exit(f"{name} is not trained")
    mean, std = setup["target_norm"]
    chans = T.field_channels()
    i_s, i_b = chans.index("q_minus_qs"), chans.index("b")
    edges = s_edges()
    nb = len(edges) - 1
    width = (LIN_HI - LIN_LO) / LIN_BINS
    ns = len(setup["seeds"])
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
        r = {"ra": run["ra"], "set": st, "g": g, "n_frames": sum(len(raw[p]) for p in parts),
             "counts": np.zeros(nb), "sse": {s: np.zeros(nb) for s in setup["seeds"]}, "sy2": 0.0,
             "floor_sse": 0.0, "s_pos": [], "s_rel": [],
             "lin_count": np.zeros(LIN_BINS), "lin_sum": np.zeros(LIN_BINS), "lin_sumsq": np.zeros(LIN_BINS)}
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
                li = np.floor((s.astype(np.float64).ravel() - LIN_LO) / width).astype(np.int64)
                inr = (li >= 0) & (li < LIN_BINS)
                li = li[inr]
                r["counts"] += np.bincount(idx, minlength=nb)
                r["lin_count"] += np.bincount(li, minlength=LIN_BINS)
                r["sy2"] += float(np.sum(y ** 2))
                r["floor_sse"] += float(np.sum((g * np.maximum(s.astype(np.float64), 0.0) - y) ** 2))
                for seed, model in setup["models"].items():
                    p = np.asarray(model.predict_on_batch(X))[..., 0].astype(np.float64) * std + mean
                    r["sse"][seed] += np.bincount(idx, weights=((p - y) ** 2).ravel(), minlength=nb)
                    pl = p.ravel()[inr]
                    r["lin_sum"] += np.bincount(li, weights=pl, minlength=LIN_BINS)
                    r["lin_sumsq"] += np.bincount(li, weights=pl ** 2, minlength=LIN_BINS)
                c = s > 0
                r["s_pos"].append(s[c].astype(np.float64))
                if with_qs:
                    qs = np.exp(T.ALPHA * (braw[part][sl] - T.BETA * z))
                    r["s_rel"].append(s[c].astype(np.float64) / qs[c])
        r["s_pos"] = np.sort(np.concatenate(r["s_pos"]))
        r["s_rel"] = np.concatenate(r["s_rel"]) if with_qs else None
        r["floor_rel"] = math.sqrt(r["floor_sse"] / r["sy2"])
        r["rel"] = {s: math.sqrt(float(v.sum()) / r["sy2"]) for s, v in r["sse"].items()}
        r["sse_mean"] = np.mean([r["sse"][s] for s in setup["seeds"]], axis=0)
        n = np.maximum(r["lin_count"] * ns, 1.0)
        r["lin_mean"] = r["lin_sum"] / n
        r["lin_std"] = np.sqrt(np.maximum(r["lin_sumsq"] / n - r["lin_mean"] ** 2, 0.0))
        pos = edges[len(edges) // 2:]
        r["F"] = cum_within(r["sse_mean"], edges)
        r["G"] = np.searchsorted(r["s_pos"], pos, side="left") / max(r["s_pos"].size, 1)
        r["s_eq"] = s_equal(r, edges)
        out["runs"][label] = r
        print(f"  {name:<7} {label:<15} done", flush=True)
        del run, raw, braw
    return out


# ============================================================
# 3. Summary, data file and figures (Q_d and Q_c separate)
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
    return float(np.percentile(x, q)) if x is not None and x.size else float("nan")


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
    L += ["", f"2. Conditioning of the Q_d input: C = Cov(x), x = the 9 values b / {qd['sb']:.4f} of an interior "
          f"3 x 3 patch;",
          f"   {COND_CELLS} random cells per training run (training block). C = sum_k lambda_k v_k v_k^T, descending;",
          "   L = (0,1,0,1,-4,1,0,1,0), L^ = L / sqrt(20). share_k = (L^.v_k)^2; f_k = lambda_k share_k / L^T C L^",
          "   (contribution of direction k to Var(L^.x); sum_k f_k = 1); std(L.x) = sqrt(L^T C L) (each x_m: std ~ 1)."]
    L.append("   Templates u_ab = P_a(z) P_b(y), P_0 = (1,1,1)/sqrt3, P_1 = (-1,0,1)/sqrt2, P_2 = (1,-2,1)/sqrt6; L = "
             + " ".join(f"{float(u @ LAPLACIAN):+.4f} u_{a}{b}" for (a, b), u in TEMPLATES.items()
                        if abs(float(u @ LAPLACIAN)) > 1e-12))
    for label, c in qd["cond"].items():
        L.append(f"   {label:<15} lambda_1 {c['eig'][0]:.3f}  lambda_1/lambda_9 {c['cond']:.2e}  "
                 f"lambda_1/L^T C L^ {c['eig'][0] / c['lap_var']:.2e}  std(L.x) {c['lap_std']:.2e}  "
                 f"trace {c['trace']:.3f}")
        L.append("      lambda_k / lambda_1 " + " ".join(f"{v / c['eig'][0]:.1e}" for v in c["eig"]))
        L.append("      share_k             " + " ".join(f"{v:.1e}" for v in c["share"]))
        L.append("      f_k                 " + " ".join(f"{v:.1e}" for v in c["contrib"]))
        L.append("      v_k nearest template (|v_k.u_ab|): " + ", ".join(
            f"{k + 1}: {derivative_text(a, b)} ({dot:.3f})"
            for k, (a, b, dot) in enumerate(template_match(c["V"][:, k]) for k in range(9))))
    L += ["", "3. Q_c (validation blocks of the training runs; test runs: all their evaluated frames). s = q - q_s.",
          "   floor: (gamma/tau) max(s, 0) from the float32 s against the stored Q_c; rel: CNN relative RMS error",
          "   (mean of the seeds; 'check' = largest relative difference from the evaluation's per_run_errors.csv);",
          "   share |s| < e: F(e) = sum_{|s_c| < e} e_c^2 / sum_c e_c^2 (e_c^2 averaged over the seeds);",
          "   s_50: smallest bin edge e with F(e) >= 0.5; s_eq: smallest s > 0 (bin centre) at which the RMS error",
          "   per bin, sqrt(sum_{c in B_k} e_c^2 / N_k), is not larger than the target (gamma/tau) s."]
    for name, res in qc.items():
        mu, sd = res["s_norm"]
        L += ["", f"   {name}: network input (s - {mu:.4f}) / {sd:.4f}; the kink s = 0 is at {-mu / sd:+.4f} "
                  f"in normalised units; 1e-3 in s is {1e-3 / sd:.2e} there"]
        L.append(f"   {'run':<15} {'set':<5} {'frames':>6} {'frac s>0':>8} {'med s>0':>9} {'p90 s>0':>9} "
                 f"{'floor':>9} {'rel':>9} {'check':>8} "
                 + " ".join(f"{'<' + format(e, '.0e'):>7}" for e in SHARE_EDGES) + f" {'s_50':>8} {'s_eq':>8}")
        for label, r in res["runs"].items():
            check = max(abs(v - eval_rel(rows, name, label, r["set"], s)) / eval_rel(rows, name, label, r["set"], s)
                        for s, v in r["rel"].items())
            L.append(f"   {label:<15} {r['set']:<5} {r['n_frames']:>6} {r['s_pos'].size / r['counts'].sum():>8.4f} "
                     f"{pct(r['s_pos'], 50):>9.2e} {pct(r['s_pos'], 90):>9.2e} {r['floor_rel']:>9.2e} "
                     f"{np.mean(list(r['rel'].values())):>9.3e} {check:>8.1e} "
                     + " ".join(f"{share_within(r['sse_mean'], e, res['edges']):>7.3f}" for e in SHARE_EDGES)
                     + f" {s_half(r['sse_mean'], res['edges']):>8.1e} {r['s_eq']:>8.1e}")
    rel = next((res for res in qc.values() if next(iter(res["runs"].values()))["s_rel"] is not None), None)
    if rel is not None:
        L += ["", "4. s / q_s in condensing cells (s > 0): a model given q (Q_c_C1) builds q_s itself; a relative error",
              "   eps = dq_s / q_s gives |Q^_c - Q_c| / Q_c = eps / (s / q_s)",
              f"   {'run':<15} {'set':<5} {'p10':>9} {'median':>9} {'p90':>9}"]
        for label, r in rel["runs"].items():
            L.append(f"   {label:<15} {r['set']:<5} {pct(r['s_rel'], 10):>9.2e} {pct(r['s_rel'], 50):>9.2e} "
                     f"{pct(r['s_rel'], 90):>9.2e}")
    return "\n".join(L) + "\n"


def save_data(qd, qc, path):
    d = {}
    for label, c in qd["cond"].items():
        for k in ("eig", "share", "contrib", "V"):
            d[f"cond__{label}__{k}"] = c[k]
    for name, res in qc.items():
        d[f"{name}__edges"] = res["edges"]
        for label, r in res["runs"].items():
            for k in ("counts", "sse_mean", "lin_count", "lin_mean", "lin_std", "F", "G"):
                d[f"{name}__{label}__{k}"] = r[k]
    d["lin_edges"] = np.linspace(LIN_LO, LIN_HI, LIN_BINS + 1)
    np.savez(path, **d)


def ra_colour(ra):
    return plt.cm.viridis(min(max((math.log10(ra) - 6.0) / 2.2, 0.0), 1.0))


def ra_text(label):
    return label.replace("Ra_", "Ra ").replace("_Pr_1", "")


SET_MARK = {"train": "o", "val": "s", "test": "*"}


def rep_labels(runs):
    labels = [l for l in REP_RUNS if l in runs]
    return labels or list(runs)[:4]


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
    errs = qd["errors"]
    ra_of = {e["run"]: e["ra"] for e in errs}
    runs = sorted(ra_of, key=lambda l: (l in T.TEST_RUNS, ra_of[l]))
    xpos = {l: k for k, l in enumerate(runs)}
    ntrain = sum(l not in T.TEST_RUNS for l in runs)
    models = (("Q_d_a", "tab:blue", -0.25, "o", "Q_d_a CNN (inputs b, log10 Ra; target Q_d): mean of the seeds, bar = range"),
              ("Q_d_b", "tab:orange", 0.0, "o", "Q_d_b CNN (input b; target Q_d / kappa): mean of the seeds, bar = range"),
              (None, "tab:red", 0.25, "D", "best function of z alone (the run's mean profile of the target)"))
    dxs = {"train": -0.07, "val": 0.07, "test": 0.0}
    fig, (ax, axb) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": (3, 2)}, layout="constrained")
    for e in errs:
        x0 = xpos[e["run"]] + dxs[e["set"]]
        face = "white" if e["set"] == "train" else None
        for cfg, col, off, mk, _ in models:
            if cfg is None:
                ax.plot(x0 + off, eval_ref(rows, "Q_d_a", e["run"], e["set"]), mk, ms=6, color=col,
                        mfc=face or col, mew=1.3)
                continue
            vals = [float(r["rel_rmse"]) for r in rows
                    if r["config"] == cfg and r["run"] == e["run"] and r["set"] == e["set"]]
            if vals:
                m = float(np.mean(vals))
                ax.errorbar(x0 + off, m, yerr=[[m - min(vals)], [max(vals) - m]], fmt=mk, ms=6, color=col,
                            mfc=face or col, mew=1.3, capsize=2.5, elinewidth=1.0)
        axb.plot(xpos[e["run"]] + dxs[e["set"]], e["fit"], "o", ms=6, color="k", mfc=face or "k", mew=1.3)
    for a in (ax, axb):
        a.axvline(ntrain - 0.5, color="0.3", lw=1.0)
        a.set_xticks(range(len(runs)), [ra_text(l) for l in runs])
        a.set_xlim(-0.6, len(runs) - 0.4)
        a.grid(True, axis="y", which="both", ls=":", alpha=0.5)
        a.set_ylabel(r"relative RMS error $\sqrt{\sum_c(\hat y_c-y_c)^2/\sum_c y_c^2}$")
    ax.axhline(1.0, color="0.4", ls=":", lw=1.0)
    ax.set_ylim(0.0, 1.1)
    for x, ha, txt in ((ntrain - 0.55, "right", "training runs"), (ntrain - 0.45, "left", "test runs")):
        ax.text(x, 1.05, txt, ha=ha, va="center", fontsize=9)
    ax.set_title("(a) the CNNs and the best function of z alone, per run and block (dotted: predicting "
                 r"$Q_d=0$, error 1). y = $Q_d/\kappa$ for Q_d_b, $Q_d$ otherwise; c runs over every cell of "
                 "the block's frames", fontsize=10, loc="left")
    handles = [Line2D([], [], marker=mk, ls="", color=col, mfc=col, label=lab) for _, col, _, mk, lab in models]
    handles += [Line2D([], [], marker="o", ls="", color="0.25", mfc="white", mew=1.3,
                       label="open: training block (frames 25-748, every 3rd)"),
                Line2D([], [], marker="o", ls="", color="0.25", mfc="0.25",
                       label="filled: validation block (811-1000, every 3rd) or test run (both blocks' frames)")]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.95)
    ratio = [abs(e["fit"] - e["exact"]) / e["exact"] for e in errs if e["exact"] > 0]
    axb.set_yscale("log")
    axb.set_title("(b) linear least squares of $Q_d/\\kappa$ on the 3x3 patch of b (fitted on training blocks; "
                  "the CNN's linear part solved directly).\nIts error equals that of the exact operator on the "
                  f"float32 b to a relative difference of at most {max(ratio):.1e}", fontsize=10, loc="left")
    axb.legend(handles=handles[3:], loc="upper right", fontsize=8)
    fig.suptitle(f"{T.VERSION}: Q_d, CNNs against a linear least-squares baseline and the best z-profile", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_conditioning(qd, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    k = np.arange(1, 10)
    for label, c in qd["cond"].items():
        col = ra_colour(c["ra"])
        axes[0, 0].semilogy(k, np.abs(c["eig"]) / c["eig"][0], "o-", color=col, label=ra_text(label))
        axes[0, 1].plot(k, c["contrib"], "o-", color=col, label=ra_text(label))
    cs = list(qd["cond"].values())
    ra = np.log10([c["ra"] for c in cs])
    axes[1, 0].semilogy(ra, [c["eig"][0] / c["lap_var"] for c in cs], "s-", color="tab:purple")
    axes[1, 1].semilogy(ra, [c["lap_std"] for c in cs], "o-", color="tab:green", label=r"std$(L\cdot x)$")
    axes[1, 1].axhline(1.0, color="0.4", linestyle=":", label=r"std of each input $x_m$ ($\approx 1$)")
    axes[0, 0].set_title(r"(a) eigenvalues of $C=\mathrm{Cov}(x)$: $\lambda_k/\lambda_1$", fontsize=10)
    axes[0, 0].set_ylabel(r"$\lambda_k / \lambda_1$")
    axes[0, 1].set_title(r"(b) $f_k=\lambda_k(\hat L\cdot v_k)^2/\hat L^{T}C\hat L$: share of "
                         r"Var$(\hat L\cdot x)$ from direction $v_k$", fontsize=10)
    axes[0, 1].set_ylabel(r"$f_k$  ($\sum_k f_k = 1$)")
    axes[0, 1].set_ylim(-0.03, 1.03)
    for ax in axes[0]:
        ax.set_xlabel(r"eigen-direction $k$ ($\lambda_1 \geq \dots \geq \lambda_9$)")
        ax.set_xticks(k)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    axes[1, 0].set_title(r"(c) $\lambda_1/\hat L^{T}C\hat L$: how much weaker the Laplacian direction "
                         "is than the dominant one", fontsize=10)
    axes[1, 0].set_ylabel(r"$\lambda_1 / \hat L^{T} C \hat L$")
    axes[1, 1].set_title(r"(d) std$(L\cdot x)=\sqrt{L^{T}CL}$, $L\cdot x = (dz^2/\sigma_b)\,\nabla^2 b$", fontsize=10)
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(r"$\log_{10}$ Ra (training runs, training block)")
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    fig.suptitle(f"{T.VERSION}: conditioning of the Q_d input. x = the 9 values b / {qd['sb']:.4f} of an interior "
                 r"3x3 patch; $L=(0,1,0,1,-4,1,0,1,0)$, $\hat L = L/\sqrt{20}$", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_eigenvectors(qd, path):
    labels = list(qd["cond"])
    labels = [labels[0], labels[-1]] if len(labels) > 1 else labels
    Lh = LAPLACIAN / np.linalg.norm(LAPLACIAN)
    coef = {ab: float(u @ Lh) for ab, u in TEMPLATES.items() if abs(float(u @ Lh)) > 1e-12}
    lap = " ".join(f"{v:+.3f} $u_{{{a}{b}}}$" for (a, b), v in coef.items())
    fig, axes = plt.subplots(len(labels), 10, figsize=(21, 2.45 * len(labels) + 1.9), squeeze=False,
                             layout="constrained")
    for i, label in enumerate(labels):
        c = qd["cond"][label]
        for k in range(10):
            ax = axes[i][k]
            if k < 9:
                v = c["V"][:, k]
                a, b, dot = template_match(v)
                title = (f"$v_{{{k + 1}}}$ ~ {derivative_text(a, b, tex=True)} ($u_{{{a}{b}}}$; |v.u| = {dot:.3f})\n"
                         rf"$\lambda_{{{k + 1}}}/\lambda_1$ = {c['eig'][k] / c['eig'][0]:.1e},  "
                         rf"$f_{{{k + 1}}}$ = {c['contrib'][k]:.3f}")
            else:
                v = Lh
                title = r"$\hat L=L/\sqrt{20}$ (Laplacian)" + "\n" + lap
            g = v.reshape(3, 3)
            ax.imshow(g, origin="lower", cmap="RdBu_r", vmin=-0.9, vmax=0.9)
            for p in range(3):
                for q in range(3):
                    ax.text(q, p, f"{g[p, q]:+.2f}", ha="center", va="center", fontsize=7.5)
            ax.set_xticks([0, 1, 2], ["y-1", "y", "y+1"], fontsize=7)
            ax.set_yticks([0, 1, 2], ["z-1", "z", "z+1"], fontsize=7)
            ax.set_title(title, fontsize=7.5)
        axes[i][0].set_ylabel(f"{ra_text(label)}\n(training block)", fontsize=10)
    fig.suptitle(f"{T.VERSION}: eigenvectors $v_k$ of $C=\\mathrm{{Cov}}(x)$ as 3x3 patterns (x: the 9 values b / "
                 f"{qd['sb']:.4f} of an interior patch; bottom row z-1; sign arbitrary, largest entry positive), in "
                 "order of decreasing variance $\\lambda_k$.\nEach is matched to the nearest template $u_{ab}=P_a(z)P_b(y)$, "
                 "$P_0=(1,1,1)/\\sqrt{3}$, $P_1=(-1,0,1)/\\sqrt{2}$, $P_2=(1,-2,1)/\\sqrt{6}$ (orthonormal polynomials of "
                 "degree 0, 1, 2 on three equally spaced points: mean, first and second difference); |v.u| = 1 is a "
                 "perfect match.\n$f_k=\\lambda_k(\\hat L\\cdot v_k)^2/\\hat L^TC\\hat L$: share of the variance of the "
                 "Laplacian combination $\\hat L\\cdot x$ from $v_k$ ($\\sum_k f_k=1$)", fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_function(res, path):
    labels = rep_labels(res["runs"])
    e = np.linspace(LIN_LO, LIN_HI, LIN_BINS + 1)
    x = 0.5 * (e[1:] + e[:-1])
    xs = np.linspace(LIN_LO, LIN_HI, 601)
    fig = plt.figure(figsize=(4.4 * len(labels), 7.0))
    gs = fig.add_gridspec(2, len(labels), height_ratios=(3, 1))
    for j, label in enumerate(labels):
        r = res["runs"][label]
        ax = fig.add_subplot(gs[0, j])
        axn = fig.add_subplot(gs[1, j], sharex=ax)
        ok = r["lin_count"] > 0
        m, sd = r["lin_mean"], r["lin_std"]
        ax.fill_between(x[ok], (m - sd)[ok], (m + sd)[ok], color="tab:blue", alpha=0.25,
                        label=r"prediction: mean $\pm$ 1 std in the bin")
        ax.plot(x[ok], m[ok], color="tab:blue", linewidth=1.5, label=r"prediction: mean $\hat Q_c$ in the bin")
        ax.plot(xs, r["g"] * np.maximum(xs, 0.0), "k-", linewidth=1.2,
                label=r"exact $Q_c=(\gamma/\tau)\max(s,0)$")
        ax.axvline(0.0, color="0.4", linestyle=":", label=r"$s=0$ ($q=q_s$)")
        ax.set_title(f"{ra_text(label)} ({'test run' if r['set'] == 'test' else 'validation block'}, "
                     f"{r['n_frames']} frames)", fontsize=10)
        ax.set_ylabel(r"$Q_c$")
        ax.grid(True, linestyle=":", alpha=0.5)
        axn.semilogy(x[ok], r["lin_count"][ok], color="0.3", drawstyle="steps-mid")
        axn.axvline(0.0, color="0.4", linestyle=":")
        axn.set_xlabel(r"$s = q - q_s$  (bin width %.0e)" % ((LIN_HI - LIN_LO) / LIN_BINS))
        axn.set_ylabel(r"cells $N_k$")
        axn.grid(True, which="both", linestyle=":", alpha=0.5)
        if j == 0:
            ax.legend(fontsize=7, loc="upper left")
    mu, sd = res["s_norm"]
    fig.suptitle(f"{T.VERSION} {res['name']} (inputs {', '.join(T.CONFIGS[res['name']]['inputs'])}): the learned "
                 f"function. Top: Q^_c of the cells c with s_c in bin B_k, averaged over the cells and the "
                 f"{len(res['seeds'])} seeds; bottom: number of cells N_k per bin.\nNetwork input (s - {mu:.3f}) / "
                 f"{sd:.3f}; cells with s outside [{LIN_LO:g}, {LIN_HI:g}] not shown", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_cumulative(res, path):
    labels = rep_labels(res["runs"])
    pos = res["edges"][len(res["edges"]) // 2:]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label in labels:
        r = res["runs"][label]
        col, ls = ra_colour(r["ra"]), ("--" if r["set"] == "test" else "-")
        lab = f"{ra_text(label)} ({'test' if r['set'] == 'test' else 'validation'})"
        axes[0].semilogx(pos, r["F"], ls, color=col, label=lab)
        axes[1].semilogx(pos, r["G"], ls, color=col, label=lab)
    axes[0].set_ylabel(r"$F(x)=\sum_{|s_c|<x} e_c^2 \,/\, \sum_c e_c^2$")
    axes[0].set_title("(a) share of the squared error in cells with |s| < x (both signs of s)", fontsize=10)
    axes[1].set_ylabel(r"$G(x)=\#\{c: 0<s_c<x\} \,/\, \#\{c: s_c>0\}$")
    axes[1].set_title("(b) fraction of the condensing cells (s > 0) with s < x", fontsize=10)
    for ax in axes:
        ax.axhline(0.5, color="0.5", linestyle=":", linewidth=1)
        ax.set_xlabel("x")
        ax.set_xlim(S_MIN, 1.0)
        ax.set_ylim(0, 1.02)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)
    fig.suptitle(f"{T.VERSION} {res['name']}: where the error is (a) and where the condensing cells are (b); "
                 r"$e_c=\hat Q_c-Q_c$, $e_c^2$ averaged over the seeds", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_relative(res, path):
    labels = rep_labels(res["runs"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bins = np.linspace(-9, 1, 81)
    for label in labels:
        r = res["runs"][label]
        x = np.log10(r["s_rel"][r["s_rel"] > 0])
        if x.size:
            axes[0].hist(x, bins=bins, density=True, histtype="step", color=ra_colour(r["ra"]),
                         linestyle="--" if r["set"] == "test" else "-", label=ra_text(label))
    axes[0].set_xlabel(r"$\log_{10}(s/q_s)$, condensing cells ($s>0$) only")
    axes[0].set_ylabel("probability density")
    axes[0].legend(fontsize=8)
    axes[0].set_title(r"(a) relative supersaturation $s/q_s$ (validation blocks; test run dashed)", fontsize=10)
    runs = [res["runs"][l] for l in labels]
    ra = np.log10([r["ra"] for r in runs])
    axes[1].semilogy(ra, [pct(r["s_pos"], 50) for r in runs], "o", color="tab:blue", label=r"median of $s$")
    axes[1].semilogy(ra, [pct(r["s_rel"], 50) for r in runs], "s", color="tab:green", label=r"median of $s/q_s$")
    axes[1].set_xlabel(r"$\log_{10}$ Ra")
    axes[1].grid(True, which="both", linestyle=":", alpha=0.5)
    axes[1].legend(fontsize=8)
    axes[1].set_title("(b) medians over the condensing cells", fontsize=10)
    fig.suptitle(f"{T.VERSION}: supersaturation in condensing cells. A model given q (Q_c_C1) must build "
                 r"$q_s=e^{\alpha(b-\beta z)}$; a relative error $\epsilon$ in $q_s$ gives "
                 r"$|\hat Q_c-Q_c|/Q_c=\epsilon/(s/q_s)$", fontsize=10)
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
    save_data(qd, qc, out / "diagnostics_data.npz")
    plot_qd_stencil(qd, out / "Qd_linear_baseline_stencil.png")
    plot_qd_errors(qd, rows, out / "Qd_linear_baseline_errors.png")
    plot_qd_conditioning(qd, out / "Qd_input_conditioning.png")
    plot_qd_eigenvectors(qd, out / "Qd_input_eigenvectors.png")
    for name in QC_NAMES:
        tag = name.split("_")[-1]
        plot_qc_function(qc[name], out / f"Qc_{tag}_learned_function.png")
        plot_qc_cumulative(qc[name], out / f"Qc_{tag}_cumulative_error.png")
    plot_qc_relative(qc[QC_NAMES[0]], out / "Qc_relative_supersaturation.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return {"qd": qd, "qc": qc}


if __name__ == "__main__":
    main()
