"""Jacobian, finite-difference probe and Integrated-Gradients attribution of the Q_d CNNs of v033 and
v034 (agreed with the user, 2026-11-16; template basis added 2026-11-17; changelog.txt). Read-only on the
trained models. The first version (no template basis) wrote simulations_v2/v034/jacobian/ (commit 49d44ab).

Notation. Cell (i, j): row i (height z_i), column j (y_j); offsets p, q in {-1, 0, 1}; patch index
m = 3 (p + 1) + (q + 1). psi^ is the model's estimate of Lap b: Q^_d / kappa for Q_d_a (inputs b,
log10 Ra; target Q_d), the output itself for Q_d_b (input b; target Q_d / kappa). Psi = dz^2 psi^.
  J_pq(i, j) = dz^2 d psi^_ij / d b_{i+p, j+q}, by reverse-mode automatic differentiation at the DNS
      frame (tf.GradientTape; Baydin, Pearlmutter, Radul & Siskind 2018, sec. 3.2). Exact operator
      L = [0 1 0; 1 -4 1; 0 1 0]; wall rows: centre -5, nothing beyond the wall.
  D_pq(i, j): the same by central differences with step EPS = 0.01 in normalised b, at the same
      cells and frames ("v0xx evaluate.py", stencil_response; the probe of the evaluation).
  Cells: rows 0 and 1 mod 3, columns 0 mod 3 (evaluate.py, sample_cells). Their 3 x 3
      neighbourhoods do not overlap, so one backward pass of sum_cells psi^ gives every cell's J.
  Taylor basis (interior patches): x~ = Lambda^(-1/2) V^T (b - bbar) / sigma_b, the whitening of
      v034 (pooled covariance of the normalised b of training-block patches; v034/Q_d_b/
      normalization_constants.npz); each column of V is labelled by its nearest template
      u_ab = P_a(z) P_b(y) (as diagnostics/v033_diagnostics.py).
  Integrated gradients (Sundararajan, Taly & Yan 2017, eq. 1) of Psi, baseline: the patch set to
      the mean patch bbar (log10 Ra unchanged), M_STEPS midpoint steps alpha_s = (s - 1/2) / M:
        Jbar_m = (1/M) sum_s J_m(bbar + alpha_s (b - bbar)),   IG_m = (b_m - bbar_m) Jbar_m,
        IG~_k = x~_k sigma_b (Jbar V Lambda^(1/2))_k,   sum_k IG~_k = sum_m IG_m ~ Psi(b) - Psi(bbar)
      (completeness, Sundararajan et al. 2017, Prop. 1; checked). Exact operator: Jbar = L, giving
      IG~ex_k = x~_k sigma_b (L V Lambda^(1/2))_k.
  Shares S_k = sum_c (IG~_k)^2 / sum_c sum_j (IG~_j)^2 over the interior cells c of a run.
  Learned fraction r_k = sum_c IG~_k IG~ex_k / sum_c (IG~ex_k)^2 (= (J.v_k) / (L.v_k) for a linear map).
  Template basis (2026-11-17): the same decomposition in the exact templates u_ab instead of V. In any
      orthonormal basis W, IG~_k = a_k g_k with a_k = w_k . (b - bbar) (amount of pattern k in the patch)
      and g_k = w_k . Jbar (the path-averaged slope along it). The eigenvectors equal the templates only to
      |v . u| >= 0.9985, which gives the exact operator small shares along v_1 and v_2; in the templates
      L = sqrt2 u_20 + sqrt2 u_02 - 4 u_22 exactly, so the exact shares and r_k live on u_20, u_02, u_22
      only (r_k = NaN elsewhere). Also the template coordinates of the AD Jacobian, J . u_ab, averaged over
      the interior cells (exact: 0 0 0 sqrt2 0 sqrt2 0 0 -4).
Data: N_FRAMES evenly spaced validation-block frames (811..1000, every 3rd) of the 8 training runs
and of the 2 test runs; every seed of Q_d_a and Q_d_b of v033 and v034; and a control model (the
exact operator in normalised units: StencilPad + a fixed 3 x 3 convolution) through the same code.
Output: simulations_v2/v034/jacobian_2/ (refuses to overwrite): summary.txt, jacobian_data.npz,
        Qd_jacobian_stencils.png, Qd_jacobian_vs_probe.png, Qd_IG_scale_shares.png and
        Qd_IG_learned_fraction.png (eigenvector basis), Qd_IG_template_shares.png and
        Qd_IG_template_learned_fraction.png (template basis)
Usage : python diagnostics/jacobian_attribution.py   (from simulations_v2/, inside the Docker container)
"""
import gc
import importlib.util
import math
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


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EVS = {"v033": _load("v033_evaluate", SIM / "v033 evaluate.py"),      # each imports its training script
       "v034": _load("v034_evaluate", SIM / "v034 evaluate.py")}
T = EVS["v034"].T                     # run lists, frame blocks and the loader (identical in v033)
sample_cells = EVS["v034"].sample_cells
EXACT = EVS["v034"].EXACT
L3 = EXACT["interior"]

VERSIONS = ("v033", "v034")
QD = ("Q_d_a", "Q_d_b")
N_FRAMES = 8                          # validation-block frames per run (evenly spaced)
M_STEPS = 32                          # midpoint steps of the integrated gradients
CHUNK = 4                             # frames (or path points) per backward pass
PCT = (5, 25, 50, 75, 95)
OUT_DIR = SIM / "v034" / "jacobian_2"      # first version: v034/jacobian/
GRAM = (np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0), np.array([-1.0, 0.0, 1.0]) / math.sqrt(2.0),
        np.array([1.0, -2.0, 1.0]) / math.sqrt(6.0))     # orthonormal polynomials of degree 0, 1, 2 on 3 points
TEMPLATES = {(a, b): np.outer(GRAM[a], GRAM[b]).ravel() for a in range(3) for b in range(3)}
CURVATURE = ((2, 0), (0, 2), (2, 2))  # templates carrying L = sqrt2 u_20 + sqrt2 u_02 - 4 u_22
TEMPLATE_ORDER = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2))
U = np.stack([TEMPLATES[k] for k in TEMPLATE_ORDER], axis=1)   # columns: the exact templates (orthonormal)
TEMPLATE_LABELS = [(a, b, 1.0) for a, b in TEMPLATE_ORDER]
COL = {"v033": "tab:blue", "v034": "tab:red"}
MK = {"Q_d_a": "o", "Q_d_b": "s"}
LS = {"Q_d_a": "-", "Q_d_b": ":"}
MODEL_TEX = {"Q_d_a": r"$\hat Q_d(b,\log_{10}\mathrm{Ra})/\kappa$", "Q_d_b": r"$\widehat{Q_d/\kappa}(b)$"}


# ============================================================
# 1. Taylor basis
# ============================================================

def template_match(v):
    """(a, b, |v.u_ab|) for the template u_ab = P_a(z) P_b(y) nearest to the unit vector v."""
    (a, b), u = max(TEMPLATES.items(), key=lambda kv: abs(float(kv[1] @ v)))
    return a, b, abs(float(u @ v))


def template_tex(a, b):
    if a == b == 0:
        return r"$\bar b$"
    parts = [rf"\partial_{v}" + (f"^{k}" if k > 1 else "") for v, k in (("z", a), ("y", b)) if k]
    return "$" + "".join(parts) + "$"


def basis_from_whitening(W, lam, xbar, b_mean, b_std):
    """V (orthonormal columns, descending eigenvalue, largest entry of each column positive),
    eigenvalues, mean patch bbar in units of b, sigma_b, template labels; from W = Lambda^(-1/2) V^T."""
    W, lam = np.asarray(W, np.float64), np.asarray(lam, np.float64)
    V = W.T * np.sqrt(lam)[None, :]
    if not np.allclose(V.T @ V, np.eye(9), atol=1e-8):
        raise ValueError("W does not give an orthonormal V")
    V = V * np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(9)])[None, :]
    return {"V": V, "lam": lam, "bbar": b_mean + b_std * np.asarray(xbar, np.float64), "sigma": float(b_std),
            "labels": [template_match(V[:, k]) for k in range(9)]}


def load_basis(path):
    with np.load(path) as f:
        norm = {str(c): (float(m), float(s)) for c, m, s in zip(f["inputs"], f["input_mean"], f["input_std"])}
        return basis_from_whitening(f["whiten_W"], f["whiten_eigenvalues"], f["whiten_xbar"], *norm["b"])


def taylor_components(dB, Jbar, basis):
    """dB = b - bbar and Jbar (n, 9), patch order m. Returns x~, IG~ (network) and IG~ex (exact
    operator), each (n, 9) in the order of the basis columns."""
    V, lam, s = basis["V"], basis["lam"], basis["sigma"]
    xt = (dB / s) @ V / np.sqrt(lam)[None, :]
    A = V * np.sqrt(lam)[None, :]
    return xt, xt * s * (Jbar @ A), xt * s * (L3.ravel() @ A)[None, :]


def shares(ig):
    q = np.sum(ig ** 2, axis=0)
    return q / q.sum()


def learned_fraction(ig, ig_ex):
    """r_k; NaN where the exact operator has no component (zero denominator)."""
    den = np.sum(ig_ex ** 2, axis=0)
    num = np.sum(ig * ig_ex, axis=0)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def pattern_components(dB, Jbar, W):
    """In the orthonormal basis W (columns): a = dB W, the amount of each pattern in b - bbar, and
    IG~ = a (Jbar W) for the network and a (L W) for the exact operator (|L.w| < 1e-12 set to 0)."""
    a = dB @ W
    lw = L3.ravel() @ W
    lw[np.abs(lw) < 1e-12] = 0.0
    return a, a * (Jbar @ W), a * lw[None, :]


# ============================================================
# 2. Derivatives of the model at the sampled cells
# ============================================================

def gather_patches(F, rows, cols):
    """F (B, NZ, NY) -> (B, n_rows, n_cols, 3, 3), index [p + 1, q + 1]; NaN beyond a wall."""
    nz, ny = F.shape[1], F.shape[2]
    out = np.full((F.shape[0], len(rows), len(cols), 3, 3), np.nan)
    for a, di in enumerate((-1, 0, 1)):
        r = rows + di
        ok = (r >= 0) & (r < nz)
        for c, dj in enumerate((-1, 0, 1)):
            view = out[..., a, c]                             # basic indexing: a view (B, n_rows, n_cols)
            view[:, ok, :] = F[:, r[ok]][:, :, (cols + dj) % ny]
    return out


def cell_grad(model, X, rows, cols):
    """d/dX of the sum of the (normalised) prediction over the cells rows x cols of every frame of X."""
    mask = np.zeros(X.shape[1:3], np.float64)
    mask[np.ix_(rows, cols)] = 1.0
    m = tf.constant(mask[None, :, :, None])
    Xt = tf.constant(X)
    with tf.GradientTape() as tape:
        tape.watch(Xt)
        y = model(Xt, training=False)
        s = tf.reduce_sum(tf.cast(y, tf.float64) * m)
    return tape.gradient(s, Xt).numpy().astype(np.float64)


def jacobian_at_cells(model, X, ib, scale, chunk=CHUNK):
    """As stencil_response, but by automatic differentiation: [(rows, J)], J (B, n_rows, n_cols, 3, 3)
    = scale * d(prediction)/d(input channel ib), NaN beyond a wall."""
    row_sets, cols = sample_cells(X.shape[1], X.shape[2])
    out = []
    for rows in row_sets:
        J = np.concatenate([gather_patches(cell_grad(model, X[f:f + chunk], rows, cols)[..., ib], rows, cols)
                            for f in range(0, len(X), chunk)])
        out.append((rows, J * scale))
    return out


def predict(model, X, chunk=CHUNK):
    return np.concatenate([np.asarray(model.predict_on_batch(X[f:f + chunk]))[..., 0]
                           for f in range(0, len(X), chunk)]).astype(np.float64)


def ig_at_cells(model, X, ib, scale, sb, xbar_n, steps=M_STEPS, chunk=CHUNK):
    """Integrated gradients at the interior sampled cells. xbar_n: baseline patch (9,) in the model's
    normalised units. Per row set: dB = b - bbar (n, 9) in units of b, Jbar (n, 9) in J units, and
    dPsi = Psi(b) - Psi(bbar) (n,); n = frames x rows x columns."""
    nz, ny = X.shape[1], X.shape[2]
    row_sets, cols = sample_cells(nz, ny)
    alphas = ((np.arange(steps) + 0.5) / steps).astype(np.float32)
    out = []
    for rows in row_sets:
        rows = rows[(rows >= 1) & (rows <= nz - 2)]
        Xb = X.copy()
        for a, di in enumerate((-1, 0, 1)):
            for c, dj in enumerate((-1, 0, 1)):
                Xb[:, (rows + di)[:, None], ((cols + dj) % ny)[None, :], ib] = np.float32(xbar_n[3 * a + c])
        gsum = np.zeros((len(X), len(rows), len(cols), 9))
        for f in range(len(X)):
            D = X[f] - Xb[f]
            for k0 in range(0, steps, chunk):
                al = alphas[k0:k0 + chunk]
                Xa = Xb[f][None] + al[:, None, None, None] * D[None]
                G = gather_patches(cell_grad(model, Xa, rows, cols)[..., ib], rows, cols)
                gsum[f] += G.reshape(len(al), len(rows), len(cols), 9).sum(axis=0)
        dX = gather_patches((X - Xb)[..., ib].astype(np.float64), rows, cols).reshape(len(X), len(rows), len(cols), 9)
        p1, p0 = predict(model, X, chunk), predict(model, Xb, chunk)
        dp = (p1 - p0)[:, rows][:, :, cols]
        out.append({"dB": (sb * dX).reshape(-1, 9), "Jbar": (scale * gsum / steps).reshape(-1, 9),
                    "dPsi": (scale * sb * dp).ravel()})
    return {k: np.concatenate([o[k] for o in out]) for k in ("dB", "Jbar", "dPsi")}


def pct(v):
    return np.percentile(v, PCT)


def analyse(model, X, ib, scale, sb, xbar_n, basis, ev):
    """All quantities of the docstring for one model on one run's frames X (normalised inputs)."""
    acc = {k: [np.zeros((3, 3)), np.zeros((3, 3)), 0] for k in ("interior", "bottom", "top")}
    cJ, sJ, jT = [], [], []
    for rows, J in jacobian_at_cells(model, X, ib, scale):
        for i, r in enumerate(rows):
            blk = J[:, i].reshape(-1, 3, 3)
            kind = "bottom" if r == 0 else "top" if r == X.shape[1] - 1 else "interior"
            a = acc[kind]
            a[0] += np.nan_to_num(blk).sum(0); a[1] += np.nan_to_num(blk ** 2).sum(0); a[2] += len(blk)
        inner = (rows >= 1) & (rows <= X.shape[1] - 2)
        blk = J[:, inner].reshape(-1, 3, 3)                 # same cell order as D below
        cJ.append(blk[:, 1, 1]); sJ.append(blk.sum(axis=(1, 2)))
        jT.append(blk.reshape(-1, 9) @ U)                   # template coordinates of J
    cD, sD = [], []
    Ds = [ev.stencil_response(model, X[f:f + CHUNK], ib, scale) for f in range(0, len(X), CHUNK)]
    for k, (rows, _) in enumerate(Ds[0]):
        D = np.concatenate([d[k][1] for d in Ds])          # all frames of a row set, as J above
        inner = (rows >= 1) & (rows <= X.shape[1] - 2)
        blk = D[:, inner].reshape(-1, 3, 3)
        cD.append(blk[:, 1, 1]); sD.append(blk.sum(axis=(1, 2)))
    cJ, sJ, cD, sD = (np.concatenate(v) for v in (cJ, sJ, cD, sD))
    mean, std = {}, {}
    for kind, (s1, s2, n) in acc.items():
        m = s1 / n
        sd = np.sqrt(np.maximum(s2 / n - m ** 2, 0.0))
        m[np.isnan(EXACT[kind])] = np.nan
        sd[np.isnan(EXACT[kind])] = np.nan
        mean[kind], std[kind] = m, sd
    ig = ig_at_cells(model, X, ib, scale, sb, xbar_n)
    igm = ig["dB"] * ig["Jbar"]
    rms = math.sqrt(float(np.mean(ig["dPsi"] ** 2)))
    resid = igm.sum(axis=1) - ig["dPsi"]
    _, igt, igex = taylor_components(ig["dB"], ig["Jbar"], basis)
    _, igT, igTex = pattern_components(ig["dB"], ig["Jbar"], U)
    jT = np.concatenate(jT)
    return {"mean": mean, "std": std, "n": {k: v[2] for k, v in acc.items()},
            "JT_mean": jT.mean(axis=0), "JT_std": jT.std(axis=0),
            "S_T": shares(igT), "S_T_ex": shares(igTex), "r_T": learned_fraction(igT, igTex),
            "pct": {"centre_J": pct(cJ), "sum_J": pct(sJ), "centre_D": pct(cD), "sum_D": pct(sD)},
            "mean_abs_JD_centre": float(np.mean(np.abs(cJ - cD))),
            "comp_rms": math.sqrt(float(np.mean(resid ** 2))) / rms, "comp_max": float(np.max(np.abs(resid))) / rms,
            "basis_check": float(np.max(np.abs(igt.sum(axis=1) - igm.sum(axis=1)))) / rms,
            "S": shares(igt), "S_ex": shares(igex), "r": learned_fraction(igt, igex), "n_ig": len(igt)}


# ============================================================
# 3. Data, models, control
# ============================================================

def load_selected(label):
    """Raw b on N_FRAMES evenly spaced validation-block frames of a run (the training loader)."""
    T.select_config("Q_d_b")
    run = T.load_run(label)
    val = run["parts"]["val"]
    n = val.stop - val.start
    sel = np.unique(np.linspace(0, n - 1, min(N_FRAMES, n)).round().astype(int))
    out = {"label": label, "b": np.ascontiguousarray(run["fields"][val][sel, ..., 0]),
           "frames": [run["frames"]["val"][i] for i in sel], "xm": run["xm"], "params": run["params"],
           "ra": run["ra"], "kappa": run["kappa"]}
    del run
    gc.collect()
    return out


def model_inputs(ev, setup, run):
    """Normalised inputs (n, NZ, NY, C) of the run's selected frames for the configuration of `setup`,
    built by its training script (apply_input_norm, RunSnapshots)."""
    t = ev.T
    t.select_config(setup["name"])
    if t.field_channels() != ["b"]:
        raise ValueError(f"{setup['name']}: field channels {t.field_channels()}, expected ['b']")
    rd = {"label": run["label"], "fields": run["b"][..., None].copy(), "y": np.zeros_like(run["b"]),
          "xm": run["xm"], "params": run["params"], "parts": {"val": slice(0, len(run["b"]))}}
    t.apply_input_norm(rd, setup["input_norm"])
    return t.RunSnapshots([rd], "val", setup["target_norm"], len(run["b"]), shuffle=False, seed=0)[0][0]


def exact_model(t, input_norm, target_norm):
    """The exact operator in the normalised units of a Q_d_b model: StencilPad (the training
    padding) and a fixed 3 x 3 convolution, kernel L sigma_b / (dz^2 sigma_y), bias -mu_y / sigma_y."""
    t.select_config("Q_d_b")
    inp = tf.keras.layers.Input((t.NZ, t.NY, 1))
    x = t.StencilPad(**t.padding_args(input_norm))(inp)
    conv = tf.keras.layers.Conv2D(1, 3, padding="valid")
    model = tf.keras.Model(inp, conv(x))
    (my, sy), sb = target_norm, input_norm["b"][1]
    conv.set_weights([(L3 * sb / ((1.0 / t.NZ) ** 2 * sy))[:, :, None, None].astype(np.float32),
                      np.array([-my / sy], np.float32)])
    return model


def scale_of(t, setup, run):
    """d(prediction_norm)/d(b_norm) -> J units: sigma_y / sigma_b x dz^2, / kappa for target Q_d."""
    return (setup["target_norm"][1] / setup["input_norm"]["b"][1] * (1.0 / t.NZ) ** 2
            / (run["kappa"] if t.TARGET == "Q_d" else 1.0))


# ============================================================
# 4. Summary and data file
# ============================================================

def seeds_of(res, v, n):
    return sorted({k[2] for k in res if k[0] == v and k[1] == n})


def summary_text(res, control, basis, runs, frames_info):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    L = ["Q_d CNNs of v033 and v034: Jacobian by automatic differentiation (J), central differences with",
         "EPS = 0.01 in normalised b at the same cells (D), integrated gradients (IG) in the Taylor basis.",
         "J_pq = dz^2 d psi^_ij / d b_{i+p, j+q}; psi^ = Q^_d / kappa (Q_d_a) or the output (Q_d_b);",
         "exact interior [0 1 0; 1 -4 1; 0 1 0], wall rows centre -5. Interior means/stds over cells; percentiles",
         f"{PCT} over interior cells. IG: baseline = mean patch bbar, {M_STEPS} midpoint steps; completeness",
         "error = |sum_m IG_m - [Psi(b) - Psi(bbar)]| / RMS(Psi(b) - Psi(bbar)), Psi = dz^2 psi^.",
         "S_k = sum_c IG~_k^2 / sum_c sum_j IG~_j^2; r_k = sum_c IG~_k IG~ex_k / sum_c IG~ex_k^2 (IG~ex: exact operator).",
         f"Frames per run: {frames_info}", "",
         "Taylor basis (v034 whitening, pooled): k, eigenvalue lambda_k, nearest template u_ab, |v_k . u_ab|:"]
    for k, (a, b, d) in enumerate(basis["labels"]):
        L.append(f"  {k + 1}  {basis['lam'][k]:.3e}  u_{a}{b}  {d:.4f}")
    L += ["", "Control (exact operator through the same code):"]
    for label in labels:
        c = control[label]
        L.append(f"  {label:<15} max|J - L| interior {np.nanmax(np.abs(c['mean']['interior'] - L3)):.1e}, "
                 f"bottom {np.nanmax(np.abs(c['mean']['bottom'] - EXACT['bottom'])):.1e}, "
                 f"top {np.nanmax(np.abs(c['mean']['top'] - EXACT['top'])):.1e}; completeness rms {c['comp_rms']:.1e}; "
                 f"max|S - S_ex| {np.max(np.abs(c['S'] - c['S_ex'])):.1e}; max|J.u - L.u| "
                 f"{np.max(np.abs(c['JT_mean'] - L3.ravel() @ U)):.1e}; max|S_T - S_T_ex| "
                 f"{np.max(np.abs(c['S_T'] - c['S_T_ex'])):.1e}")
    kc = [k for k, (a, b, _) in enumerate(basis["labels"]) if (a, b) in CURVATURE]
    kT = [TEMPLATE_ORDER.index(c) for c in CURVATURE]
    for v in VERSIONS:
        for n in QD:
            for s in seeds_of(res, v, n):
                L += ["", f"=== {v} {n} seed {s}"]
                pooled = np.mean([res[(v, n, s, l)]["mean"]["interior"] for l in labels if l in T.TRAIN_RUNS], axis=0)
                L.append("  interior mean J over the training runs' cells (rows p = -1, 0, +1; columns q = -1, 0, +1):")
                for row in pooled:
                    L.append("      " + "  ".join(f"{x:+8.4f}" for x in row))
                for kind in ("bottom", "top"):
                    m = np.mean([res[(v, n, s, l)]["mean"][kind] for l in labels], axis=0)
                    L.append(f"  {kind} wall row mean J: centre {m[1, 1]:+.3f}, sideways {m[1, 0]:+.3f} {m[1, 2]:+.3f}, "
                             f"towards interior {m[2, 1] if kind == 'bottom' else m[0, 1]:+.3f}")
                L.append(f"  {'run':<15} {'J_00 p50 [p5, p95]':>26} {'D_00 p50':>9} {'sum J p50 [p5, p95]':>28} "
                         f"{'sum D p50':>10} {'<|J00-D00|>':>11} {'compl rms':>9} {'compl max':>9} "
                         + " ".join(f"{'r_' + str(k + 1):>8}" for k in kc))
                for l in labels:
                    r = res[(v, n, s, l)]
                    P = r["pct"]
                    L.append(f"  {l:<15} {P['centre_J'][2]:+8.3f} [{P['centre_J'][0]:+7.3f}, {P['centre_J'][4]:+7.3f}] "
                             f"{P['centre_D'][2]:+9.3f} {P['sum_J'][2]:+9.4f} [{P['sum_J'][0]:+8.4f}, {P['sum_J'][4]:+8.4f}] "
                             f"{P['sum_D'][2]:+10.4f} {r['mean_abs_JD_centre']:11.3e} {r['comp_rms']:9.1e} {r['comp_max']:9.1e} "
                             + " ".join(f"{r['r'][k]:+8.1e}" for k in kc))
                L.append("  shares S_k (columns k = 1..9; last row: exact operator on the same cells)")
                for l in labels:
                    r = res[(v, n, s, l)]
                    L.append(f"  {l:<15} " + " ".join(f"{x:8.1e}" for x in r["S"]))
                L.append(f"  {'exact (Ra 1e8)':<15} " + " ".join(f"{x:8.1e}" for x in res[(v, n, s, 'Ra_1e8_Pr_1')]["S_ex"]))
                L.append("  template basis (columns u00 u10 u01 u20 u11 u02 u21 u12 u22): mean J.u_ab over interior cells "
                         "(exact 0 0 0 +1.414 0 +1.414 0 0 -4)")
                for l in labels:
                    L.append(f"  {l:<15} " + " ".join(f"{x:+8.4f}" for x in res[(v, n, s, l)]["JT_mean"]))
                L.append("  template shares S_T (same columns; last row: exact operator) and r_T for u20, u02, u22")
                for l in labels:
                    r = res[(v, n, s, l)]
                    L.append(f"  {l:<15} " + " ".join(f"{x:8.1e}" for x in r["S_T"]) + "   r_T "
                             + " ".join(f"{r['r_T'][k]:+8.2e}" for k in kT))
                L.append(f"  {'exact (Ra 1e8)':<15} " + " ".join(f"{x:8.1e}" for x in res[(v, n, s, 'Ra_1e8_Pr_1')]["S_T_ex"]))
    worst = max(r["comp_rms"] for r in res.values())
    L += ["", f"Largest completeness rms error over all models and runs: {worst:.1e}",
          f"Largest basis check (sum_k IG~_k vs sum_m IG_m) : {max(r['basis_check'] for r in res.values()):.1e}"]
    return "\n".join(L) + "\n"


def save_data(res, control, basis, runs, path):
    d = {"V": basis["V"], "lambda": basis["lam"], "bbar": basis["bbar"], "U": U,
         "template_order": np.array(TEMPLATE_ORDER),
         "templates": np.array([[a, b] for a, b, _ in basis["labels"]]), "percentiles": np.array(PCT)}
    for key, r in list(res.items()) + [(("control", "Q_d_b", 0, l), c) for l, c in control.items()]:
        tag = "__".join(str(x) for x in key)
        for kind in ("interior", "bottom", "top"):
            d[f"{tag}__mean_{kind}"] = r["mean"][kind]
            d[f"{tag}__std_{kind}"] = r["std"][kind]
        for k, v in r["pct"].items():
            d[f"{tag}__pct_{k}"] = v
        for k in ("S", "S_ex", "r", "S_T", "S_T_ex", "r_T", "JT_mean", "JT_std"):
            d[f"{tag}__{k}"] = r[k]
        d[f"{tag}__completeness_rms"] = np.array(r["comp_rms"])
    np.savez_compressed(path, **d)


# ============================================================
# 5. Figures (as few words as possible; formulas)
# ============================================================

def ra_tex(label, ra):
    m, e = f"{ra:.2e}".split("e")
    m, e = float(m), int(e)
    base = rf"10^{{{e}}}" if abs(m - 1.0) < 1e-9 else rf"{m:g}\times10^{{{e}}}"
    return rf"$\mathrm{{Ra}}={base}" + (r"\,^{\dagger}" if label in T.TEST_RUNS else "") + "$"


def plot_stencils(res, runs, show, path):
    rows = [(v, n) for v in VERSIONS for n in QD]
    cols = list(show) + ["exact"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.55 * len(cols) + 1.0, 2.5 * len(rows) + 1.3),
                             squeeze=False, layout="constrained")
    for i, (v, n) in enumerate(rows):
        for j, label in enumerate(cols):
            ax = axes[i][j]
            M = L3 if label == "exact" else np.mean(
                [res[(v, n, s, label)]["mean"]["interior"] for s in seeds_of(res, v, n)], axis=0)
            im = ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
            for a in range(3):
                for c in range(3):
                    val = M[a, c]
                    txt = f"{val:+.1e}" if 0 < abs(val) < 0.01 else f"{val:+.3f}"
                    ax.text(c, a, txt, ha="center", va="center", fontsize=8.5, color="w" if abs(val) > 3 else "k")
            ax.set_xticks([0, 1, 2], ["$-1$", "$0$", "$1$"])
            ax.set_yticks([0, 1, 2], ["$-1$", "$0$", "$1$"])
            if i == 0:
                ax.set_title(r"$\Delta z^2\nabla_h^2$" if label == "exact" else ra_tex(label, runs[label]["ra"]), fontsize=11)
            if i == len(rows) - 1:
                ax.set_xlabel(r"$q$", fontsize=11)
            if j == 0:
                ax.set_ylabel(f"{v}:  " + MODEL_TEX[n] + "\n" + r"$p$", fontsize=11)
    cb = fig.colorbar(im, ax=axes, shrink=0.6)
    cb.set_label(r"$J_{pq}$", fontsize=12)
    fig.suptitle(r"$J_{pq}=\frac{1}{N}\sum_{(i,j)}\Delta z^2\,\frac{\partial\hat\psi_{ij}}{\partial b_{i+p,\,j+q}}$"
                 r"$,\quad \hat\psi\in\{\hat Q_d/\kappa,\ \widehat{Q_d/\kappa}\}$"
                 r"$,\quad {}^{\dagger}\,\mathrm{test}$", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_vs_probe(res, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True, layout="constrained")
    offs = {("v033", "J"): -0.27, ("v033", "D"): -0.09, ("v034", "J"): 0.09, ("v034", "D"): 0.27}
    for jc, n in enumerate(QD):
        for ir, (what, ref) in enumerate((("centre", -4.0), ("sum", 0.0))):
            ax = axes[ir][jc]
            for x, l in zip(xs, labels):
                if l in T.TEST_RUNS:
                    ax.axvspan(x - 0.45, x + 0.45, color="0.92", lw=0)
            ax.axhline(ref, color="k", ls=":", lw=1.2)
            for v in VERSIONS:
                seeds = seeds_of(res, v, n)
                for meth, mk in (("J", "o"), ("D", "x")):
                    P = np.array([[res[(v, n, s, l)]["pct"][f"{what}_{meth}"] for l in labels] for s in seeds])
                    x0 = xs + offs[(v, meth)]
                    lo, hi = P[:, :, 0].min(axis=0), P[:, :, 4].max(axis=0)
                    ax.vlines(x0, lo, hi, color=COL[v], lw=1.0)
                    for k in range(len(seeds)):
                        ax.plot(x0, P[k, :, 2], mk, color=COL[v], ms=5, mfc=COL[v] if mk == "o" else "none", mew=1.2)
            ax.set_ylabel(r"$J_{00},\ D_{00}$" if what == "centre" else r"$\sum_{p,q}J_{pq},\ \sum_{p,q}D_{pq}$", fontsize=12)
            ax.grid(True, axis="y", ls=":", alpha=0.5)
            if ir == 0:
                ax.set_title(MODEL_TEX[n], fontsize=13)
        axes[1][jc].set_xticks(xs, [ra_tex(l, runs[l]["ra"]).replace(r"\mathrm{Ra}=", "") for l in labels],
                               rotation=45, fontsize=9)
        axes[1][jc].set_xlabel(r"$\mathrm{Ra}$", fontsize=12)
    handles = [Line2D([], [], marker="o", ls="", color="0.3", label=r"$J=\Delta z^2\,\partial\hat\psi/\partial b$"),
               Line2D([], [], marker="x", ls="", color="0.3",
                      label=r"$D=\Delta z^2\,[\hat\psi(b+h)-\hat\psi(b-h)]/2h,\ h=10^{-2}\sigma_b$"),
               Line2D([], [], color="0.3", lw=1.0, label=r"$[\min_s P_5,\ \max_s P_{95}]$"),
               Patch(color=COL["v033"], label="v033"), Patch(color=COL["v034"], label="v034"),
               Line2D([], [], color="k", ls=":", label=r"$\Delta z^2\nabla_h^2$"),
               Patch(color="0.92", label=r"$\dagger\ \mathrm{test}$"),
               Line2D([], [], ls="", label="runs equally spaced; markers offset sideways within a run")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(r"$P_{50}$ over cells $(i,j)$, one marker per seed $s$", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_shares(res, runs, basis_labels, path, key="S", title=None):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    k = np.arange(1, 10)
    ticks = [template_tex(a, b) for a, b, _ in basis_labels]
    ncol = 5
    nrow = math.ceil(len(labels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.0 * nrow + 1.2), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, l in zip(axes.ravel(), labels):
        ex = np.mean([res[(v, n, s, l)][key + "_ex"] for v in VERSIONS for n in QD for s in seeds_of(res, v, n)], axis=0)
        ex = np.where(ex > 0, ex, np.nan)                 # exact zeros (template basis) are not drawn
        ax.plot(k, ex, color="k", lw=2.4, ls="--", marker="D", ms=5, zorder=5)
        for v in VERSIONS:
            for n in QD:
                S = np.mean([res[(v, n, s, l)][key] for s in seeds_of(res, v, n)], axis=0)
                ax.plot(k, S, color=COL[v], ls=LS[n], marker=MK[n], ms=4.5, lw=1.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2.0)
        ax.set_xticks(k, ticks, rotation=90, fontsize=11)
        ax.set_title(ra_tex(l, runs[l]["ra"]), fontsize=11)
        ax.grid(True, which="major", ls=":", alpha=0.5)
    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r][0].set_ylabel(r"$S_k$", fontsize=12)
    handles = [Line2D([], [], color="k", lw=2.4, ls="--", marker="D", label=r"$\Delta z^2\nabla_h^2$")]
    handles += [Line2D([], [], color=COL[v], ls=LS[n], marker=MK[n], label=f"{v}: " + MODEL_TEX[n])
                for v in VERSIONS for n in QD]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=10)
    fig.suptitle(title or r"$S_k=\frac{\sum_c(\widetilde{\mathrm{IG}}_k)^2}{\sum_c\sum_j(\widetilde{\mathrm{IG}}_j)^2},\quad"
                 r"\widetilde{\mathrm{IG}}_k=\tilde x_k\int_0^1\frac{\partial\Psi}{\partial\tilde x_k}"
                 r"\left(\bar x+\alpha(x-\bar x)\right)d\alpha,\quad \tilde x=\Lambda^{-1/2}V^{\top}(x-\bar x),"
                 r"\quad\Psi=\Delta z^2\hat\psi$", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


TEMPLATE_SHARES_TEX = (r"$S_k=\frac{\sum_c(a_kg_k)^2}{\sum_c\sum_j(a_jg_j)^2},\quad a_k=\mathbf{u}_k\cdot(\mathbf{x}-\bar{\mathbf{x}}),"
                       r"\quad g_k=\mathbf{u}_k\cdot\int_0^1\nabla\Psi\left(\bar{\mathbf{x}}+\alpha(\mathbf{x}-\bar{\mathbf{x}})\right)d\alpha,"
                       r"\quad\Psi=\Delta z^2\hat\psi$")
TEMPLATE_FRACTION_TEX = (r"$r_k=\frac{\sum_c a_k^2\,g_k\,g_k^{\nabla^2}}{\sum_c a_k^2\,(g_k^{\nabla^2})^2}"
                         r",\quad g^{\nabla^2}=(\sqrt{2},\sqrt{2},-4)$ on $(\mathbf{u}_{20},\mathbf{u}_{02},\mathbf{u}_{22})$, one marker per seed")


def plot_learned_fraction(res, runs, basis_labels, path, key="r", title=None):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    xs = np.arange(len(labels))
    kc = [k for k, (a, b, _) in enumerate(basis_labels) if (a, b) in CURVATURE]
    fig, axes = plt.subplots(1, len(kc), figsize=(5.6 * len(kc), 5.2), sharey=True, squeeze=False, layout="constrained")
    offs = {("v033", "Q_d_a"): -0.27, ("v033", "Q_d_b"): -0.09, ("v034", "Q_d_a"): 0.09, ("v034", "Q_d_b"): 0.27}
    for ax, k in zip(axes[0], kc):
        for x, l in zip(xs, labels):
            if l in T.TEST_RUNS:
                ax.axvspan(x - 0.45, x + 0.45, color="0.92", lw=0)
        ax.axhline(1.0, color="k", ls=":", lw=1.2)
        for (v, n), o in offs.items():
            for s in seeds_of(res, v, n):
                r = np.array([res[(v, n, s, l)][key][k] for l in labels])
                pos = r > 0
                ax.plot(xs[pos] + o, r[pos], MK[n], color=COL[v], ms=5)
                ax.plot(xs[~pos] + o, -r[~pos], MK[n], color=COL[v], ms=5, mfc="none")
        ax.set_yscale("log")
        a, b, _ = basis_labels[k]
        ax.set_title(rf"$k={k + 1}$: " + template_tex(a, b), fontsize=12)
        ax.set_xticks(xs, [ra_tex(l, runs[l]["ra"]).replace(r"\mathrm{Ra}=", "") for l in labels], rotation=45, fontsize=9)
        ax.set_xlabel(r"$\mathrm{Ra}$", fontsize=12)
        ax.grid(True, which="major", ls=":", alpha=0.5)
    axes[0][0].set_ylabel(r"$|r_k|$", fontsize=12)
    handles = [Line2D([], [], marker=MK[n], ls="", color=COL[v], label=f"{v}: " + MODEL_TEX[n])
               for v in VERSIONS for n in QD]
    handles += [Line2D([], [], marker="o", ls="", color="0.3", label=r"$r_k>0$"),
                Line2D([], [], marker="o", ls="", color="0.3", mfc="none", label=r"$r_k<0$"),
                Line2D([], [], color="k", ls=":", label=r"$r_k=1$"), Patch(color="0.92", label=r"$\dagger\ \mathrm{test}$"),
                Line2D([], [], ls="", label="runs equally spaced; markers offset sideways within a run")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(title or r"$r_k=\frac{\sum_c\widetilde{\mathrm{IG}}_k\,\widetilde{\mathrm{IG}}{}^{\nabla^2}_k}"
                 r"{\sum_c(\widetilde{\mathrm{IG}}{}^{\nabla^2}_k)^2}$, one marker per seed", fontsize=13)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def main():
    t_start = time.time()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} exists and is not empty; refusing to overwrite.")
    T.configure_gpu()
    basis = load_basis(SIM / "v034" / "Q_d_b" / "normalization_constants.npz")
    with np.load(SIM / "v034" / "Q_d_a" / "normalization_constants.npz") as fa, \
            np.load(SIM / "v034" / "Q_d_b" / "normalization_constants.npz") as fb:
        if not np.array_equal(fa["whiten_W"], fb["whiten_W"]):
            sys.exit("v034 Q_d_a and Q_d_b whitenings differ")
    runs = {l: load_selected(l) for l in T.TRAIN_RUNS + T.TEST_RUNS}
    frames_info = "; ".join(f"{l}: {r['frames']}" for l, r in runs.items())
    res, control = {}, {}
    for v in VERSIONS:
        ev = EVS[v]
        for n in QD:
            setup = ev.load_config(n)
            if setup is None:
                sys.exit(f"{v} {n} is not trained")
            ib = list(ev.T.INPUTS).index("b")
            mb, sb = setup["input_norm"]["b"]
            xbar_n = (basis["bbar"] - mb) / sb
            for l, run in runs.items():
                X = model_inputs(ev, setup, run)
                sc = scale_of(ev.T, setup, run)
                for s, model in setup["models"].items():
                    t0 = time.time()
                    res[(v, n, s, l)] = analyse(model, X, ib, sc, sb, xbar_n, basis, ev)
                    r = res[(v, n, s, l)]
                    print(f"  {v} {n} seed {s} {l:<15} J00 p50 {r['pct']['centre_J'][2]:+.3f}  "
                          f"D00 p50 {r['pct']['centre_D'][2]:+.3f}  completeness {r['comp_rms']:.1e}  "
                          f"({time.time() - t0:.0f} s)", flush=True)
            if v == "v034" and n == "Q_d_b":
                cm = exact_model(ev.T, setup["input_norm"], setup["target_norm"])
                for l, run in runs.items():
                    control[l] = analyse(cm, model_inputs(ev, setup, run), ib, scale_of(ev.T, setup, run), sb,
                                         xbar_n, basis, ev)
            del setup
            tf.keras.backend.clear_session()
            gc.collect()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = summary_text(res, control, basis, runs, frames_info)
    (OUT_DIR / "summary.txt").write_text(text)
    print("\n" + text)
    save_data(res, control, basis, runs, OUT_DIR / "jacobian_data.npz")
    show = ("Ra_1e6_Pr_1", "Ra_1e7_Pr_1", "Ra_1e8_Pr_1", "Ra_8.7e6_Pr_1", "Ra_1.5e8_Pr_1")
    plot_stencils(res, runs, show, OUT_DIR / "Qd_jacobian_stencils.png")
    plot_vs_probe(res, runs, OUT_DIR / "Qd_jacobian_vs_probe.png")
    plot_shares(res, runs, basis["labels"], OUT_DIR / "Qd_IG_scale_shares.png")
    plot_learned_fraction(res, runs, basis["labels"], OUT_DIR / "Qd_IG_learned_fraction.png")
    plot_shares(res, runs, TEMPLATE_LABELS, OUT_DIR / "Qd_IG_template_shares.png", key="S_T", title=TEMPLATE_SHARES_TEX)
    plot_learned_fraction(res, runs, TEMPLATE_LABELS, OUT_DIR / "Qd_IG_template_learned_fraction.png", key="r_T",
                          title=TEMPLATE_FRACTION_TEX)
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return res, control


if __name__ == "__main__":
    main()
