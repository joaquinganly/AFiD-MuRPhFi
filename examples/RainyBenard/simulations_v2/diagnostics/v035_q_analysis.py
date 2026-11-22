"""Analyses of the v035 network for the whole source term Q = Q_d + Q_c: the analyses run on the v034 Q_d and Q_c
networks, applied to it, and their link to those networks (asked by the user, 2026-11-20; changelog.txt). Read-only on
every model; reuses the tested functions of diagnostics/xai_attribution.py (xa) and jacobian_attribution.py (ja).

Notation. Q = Q_d + Q_c with Q_d = kappa Lap b, Q_c = (gamma/tau) max(s, 0), s = q - q_s, kappa = (Ra Pr)^(-1/2), Pr = 1,
  rho = log10 Ra (Vallis, Parker & Tobias 2019). Q^: a model's prediction in raw units. Models ("tags"):
    Q_a    v035 Q_a: inputs b, s, rho -> Q (seeds 42, 43, 44);
    Q_d_a  v034 Q_d_a: b, rho -> Q_d;   C4  v034 Q_c_C4: b, s -> Q_c;
    sum    Q^_d(Q_d_a seed k) + Q^_c(C4 seed k), the two separately trained networks added, paired by seed k;
    exact  the exact Q of the same inputs, through the same code (control; v035 normalisation).
  Wet cells: DNS Q_c > 0 (the DNS condenses; Q = Q_d + Q_c there); dry cells: the rest (Q = Q_d there).
  Relative error over a set of cells c: e = sqrt(sum_c (Q^_c - Q_c)^2 / sum_c Q_c^2) with the DNS Q; e = 1 is the error
  of predicting 0. Explained cells (parts jac, xai): the interior sampled cells (rows 0 and 1 mod 3 without the wall
  rows, columns 0 mod 3) of the ja.N_FRAMES = 8 evenly spaced validation-block frames per run, as in xa.
Parts (python diagnostics/v035_q_analysis.py <part> [--out NAME], from simulations_v2/ inside the Docker container;
outputs in simulations_v2/v035/analysis/, or in simulations_v2/v035/NAME/ with --out (added 2026-11-22); refuses to
overwrite):
  jac   Jacobian, as jacobian_attribution.py (Q_d) and xai_figures.py qc_jacobian (Q_c): J_ch,pq = d Q^(i, j) /
        d x_ch(i + p, j + q) in raw units, by reverse-mode automatic differentiation (Baydin et al. 2018) on the 3 x 3
        patches of the explained cells, for wet and dry cells separately, per run and pooled over the 8 training runs.
        Reported: the b slope in Laplacian units Jt_b = dz^2 J_b / kappa (exact L = [0 1 0; 1 -4 1; 0 1 0] in every
        cell) and its template coordinates Jt_b . u_ab (exact 0 0 0 sqrt2 0 sqrt2 0 0 -4); the s slope J_s (exact
        gamma/tau = 20 at the centre of a wet cell, 0 elsewhere); the slope to a uniform change of rho,
        J_rho = sum_pq J_rho,pq (exact -(ln 10 / 2) kappa Lap b = -(ln 10 / 2) Q_d), summarised by
        r_rho = sum_c J_rho J_rho^ex / sum_c (J_rho^ex)^2 (1 for the exact Q); medians and 5th / 95th percentiles over
        the cells. Pooled percentiles use every wet cell and every DRY_STRIDE-th dry cell of the training runs.
        Models: Q_a, Q_d_a, C4, sum, exact.
  xai   As xai_attribution.py without its path methods (GradSHAP, IG: their completeness for Q_c is 0.1 - 0.3 even for
        the exact function, handover 7.12): exact Shapley values over the channels (b, s, rho) with references bbar
        (mean patch of the v034 whitening), the mean s patch of the xa pools and rho_bar (mean training log10 Ra), and
        channel ablation e_-ch; exact Shapley values over the 9 templates u_ab of b (s and rho at the cell's own
        values) and template ablation e_-k. Shares S = sum_c phi^2 / sum_c sum phi^2 on all, wet and dry cells.
        Models: Q_a and exact; the v034 Q_d_a template shares are read from v034/xai/xai_qd.npz for comparison.
        Exact Q, analytically (checked): phi_b = (kappa_c + kappa_bar) L.(b - bbar) / (2 dz^2), phi_rho = (kappa_c -
        kappa_bar) L.(b + bbar) / (2 dz^2), phi_s = (gamma/tau) [max(s_c, 0) - max(sbar_c, 0)]; template credits
        kappa_c (L.u_k) a_k / dz^2, a_k = u_k.(b - bbar).
  eval  As "v034 evaluate.py", analyses 1 and 3: e per run on every frame of the training and validation blocks
        (training runs) and of both blocks (test runs), on all, wet and dry cells, for Q_a and sum; reference: the run's
        best function of height alone (its mean profile of Q; McGreivy & Hakim 2024); the domain-mean bias
        beta = (sum Q^ - sum Q) / sum Q_c over the same cells and frames (see beta(); on 2026-11-22 it replaced the
        ratio sum Q^ / sum Q, whose denominator nearly vanishes); the validation loss recomputed against the Keras
        history; the training summary.
  dns   As diagnostics/v034_dns_vs_cnn.py for Q, every frame 0..1000 of the two test runs: domain means <Q>(t), <Q^>(t);
        e on frames 0..24 (drizzle start, never trained on), 25..1000 and all; beta = (mean_t <Q^> - mean_t <Q>) /
        mean_t <Q_c> over frames 25..1000 (see beta()); time-mean profiles <Q>_{y,t}(z) over
        frames 25..1000; joint histograms of (Q, Q^) (seed 42); snapshots of Q, Q^ - Q at t* = argmax_{t >= 25} <Q_c>(t)
        and t_q = argmin_{t >= 25} <Q_c>(t) (the frames of the v034 comparison).
"""
import argparse
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SIM = Path(__file__).resolve().parents[1]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


xa = _load("xai_attribution", SIM / "diagnostics" / "xai_attribution.py")
ja = xa.ja
T34 = ja.EVS["v034"].T                   # v034 training script: Q_d_a, Q_c_C4 and the data loader
T35 = _load("v035_train", SIM / "v035 stencil CNN.py")
OUT_DIR = T35.OUT_DIR / "analysis"
U, L3 = ja.U, ja.L3
L9 = L3.ravel()
CHANS = ("b", "q_minus_qs")              # field channels read from the DNS
Q_INPUTS = ("b", "q_minus_qs", "log10_Ra")
SEEDS = (42, 43, 44)
PCT = (5, 50, 95)
CLASSES = ("all", "wet", "dry")
DRY_STRIDE = 8                           # pooled percentiles: every DRY_STRIDE-th dry cell (all wet cells)
BATCH = 8                                # frames per prediction batch (eval, dns)
SPIN = 25                                # frames 0..SPIN - 1: drizzle start, never used in training
NBIN = 240                               # bins per axis of the joint histograms
SNAP_SEED = 42
DEV_TOL = 1e-3                           # exact control: analytic credits; xa saw up to 6e-4 (float32 patches)
TAG_TEX = {"Q_a": r"$\hat Q(b,s,\rho)$", "Q_d_a": r"$\hat Q_d(b,\rho)$", "C4": r"$\hat Q_c(b,s)$",
           "sum": r"$\hat Q_d+\hat Q_c$", "exact": r"$Q$ exact"}
TAG_COL = {"Q_a": "tab:red", "Q_d_a": "tab:green", "C4": "tab:purple", "sum": "tab:blue", "exact": "k"}
CH_TEX = {"b": r"$b$", "q_minus_qs": r"$s$", "log10_Ra": r"$\rho$"}
CH_COL = {"b": "tab:green", "q_minus_qs": "tab:blue", "log10_Ra": "tab:purple"}
TEST_TEX = r"$\dagger\ \mathrm{test}$"
PART_OUT = {
    "jac": ("jacobian.npz", "summary_jacobian.txt", "Q_jacobian_stencils.png", "Q_jacobian_by_run.png"),
    "xai": ("xai.npz", "summary_xai.txt", "Q_channel_shares_pct.png", "Q_template_shares.png", "Q_ablation_errors.png"),
    "eval": ("per_run_errors.csv", "summary_eval.txt", "Q_errors_by_run.png", "Q_training_curves.png"),
    "dns": ("dns_vs_cnn.npz", "summary_dns.txt", "Q_timeseries.png", "Q_profiles.png", "Q_joint.png", "Q_snapshots.png"),
}


# ============================================================
# 1. Models, inputs, data
# ============================================================

def load_setup(t, name):
    """As load_config of "v034 evaluate.py", for the training module t (v034 or v035): configuration, normalisation,
    models and histories, checked against the training script."""
    d = t.OUT_DIR / name
    if not (d / "summary.json").is_file():
        sys.exit(f"{d}: not trained")
    t.select_config(name)
    cfg = json.loads((d / "config.json").read_text())
    expected = {"inputs": list(t.INPUTS), "target": t.TARGET, "train_runs": t.TRAIN_RUNS,
                "blocks": {"train_frames": list(t.TRAIN_FRAMES), "val_frames": list(t.VAL_FRAMES),
                           "stride": t.STRIDE, "gap_min": t.GAP_MIN}, "grid": [t.NZ, t.NY]}
    differ = [k for k, v in expected.items() if cfg.get(k) != v]
    if differ:
        sys.exit(f"{d}/config.json differs from the training script in {differ}")
    with np.load(d / "normalization_constants.npz") as f:
        input_norm = {str(c): (float(m), float(s)) for c, m, s in zip(f["inputs"], f["input_mean"], f["input_std"])}
        target_norm = (float(f["target_mean"]), float(f["target_std"]))
        whiten = {k: f[k] for k in ("whiten_W", "whiten_xbar") if k in f.files}
    if not (t._same(json.loads(json.dumps(input_norm)), cfg["input_norm"])
            and t._same(list(target_norm), cfg["target_norm"])):
        sys.exit(f"{d}: normalization_constants.npz differs from config.json")
    summary = {s["seed"]: s for s in json.loads((d / "summary.json").read_text())}
    seeds = sorted(summary)
    models = {s: tf.keras.models.load_model(d / f"seed_{s}" / "model.keras", compile=False) for s in seeds}
    histories = {s: json.loads((d / f"seed_{s}" / "history.json").read_text()) for s in seeds}
    return {"name": name, "t": t, "cfg": cfg, "summary": summary, "seeds": seeds, "models": models,
            "histories": histories, "input_norm": input_norm, "target_norm": target_norm, "whiten": whiten}


class ExactQLayer(tf.keras.layers.Layer):
    """Exact Q on a padded input (n, H + 2, W + 2, 3) in the normalised units of Q_a (channels b, s, rho): at every
    centre cell kappa(rho) Lap b + (gamma/tau) max(s, 0), kappa = 10^(-rho/2) (Pr = 1), Lap b = (sigma_b / dz^2) L * b_n
    (dz = dy); output (Q - mu_y) / sigma_y. float64 inside."""

    def __init__(self, input_norm, target_norm, gamma_tau, dz, **kwargs):
        super().__init__(dtype="float64", **kwargs)
        self.nb, self.ns, self.nr = (tuple(float(v) for v in input_norm[c]) for c in Q_INPUTS)
        self.my, self.sy = (float(v) for v in target_norm)
        self.gt, self.dz2 = float(gamma_tau), float(dz) ** 2
        self.kernel = tf.constant(L3[:, :, None, None], tf.float64)

    def call(self, x):
        x = tf.cast(x, tf.float64)
        lap = tf.nn.conv2d(x[..., 0:1], self.kernel, 1, "VALID") * (self.nb[1] / self.dz2)
        c = x[:, 1:-1, 1:-1, :]
        rho = self.nr[0] + self.nr[1] * c[..., 2:3]
        s = self.ns[0] + self.ns[1] * c[..., 1:2]
        q = tf.pow(tf.constant(10.0, tf.float64), -0.5 * rho) * lap + self.gt * tf.maximum(s, tf.constant(0.0, tf.float64))
        return tf.cast((q - self.my) / self.sy, tf.float32)


def exact_q_model(setup, gamma_tau):
    """The exact Q with the training padding of Q_a (StencilPad) and ExactQLayer; xa.Patches drops the padding."""
    t = T35
    t.select_config("Q_a")
    if tuple(t.INPUTS) != Q_INPUTS:
        raise ValueError(f"Q_a inputs {t.INPUTS}, expected {Q_INPUTS}")
    inp = tf.keras.layers.Input((t.NZ, t.NY, 3))
    x = t.StencilPad(**t.padding_args(setup["input_norm"]), dtype="float64", name="stencil_pad")(inp)
    out = ExactQLayer(setup["input_norm"], setup["target_norm"], gamma_tau, 1.0 / t.NZ, name="exact_q")(x)
    return tf.keras.Model(inp, out, name="exact_Q")


def inputs_for(setup, fields, run):
    """Normalised inputs (n, NZ, NY, C) of raw fields (n, NZ, NY, 2: b, s) for the configuration of `setup`, built by
    its training script (xa.model_inputs: apply_input_norm, RunSnapshots)."""
    return xa.model_inputs(setup["t"], setup, {"label": run["label"], "chans": list(CHANS), "fields": fields,
                                               "xm": run["xm"], "params": run["params"]})


def predict_raw(model, setup, X):
    my, sy = setup["target_norm"]
    return my + sy * ja.predict(model, X)


def read_npz_frames(path, key, frames):
    """(n, NZ, NY) of the stored (NZ, NY, nt) array at `frames`; float64."""
    with np.load(path) as f:
        arr = f[key]
    out = np.ascontiguousarray(np.moveaxis(arr[:, :, frames], -1, 0))
    del arr
    return out


def load_explained(labels):
    """As xa.load_part_runs for the channels b, s: raw fields on the explained frames, DNS Q_c, Q_d, Q there (float64),
    and the pool of xa (the same draws, in TRAIN_RUNS + TEST_RUNS order), whose mean patches are the references."""
    xa.set_inputs(T34, CHANS, "Q_c")
    n_train = len(T34.frame_indices(T34.TRAIN_FRAMES))
    rngs = {"train": np.random.default_rng(T34.WHITEN_SEED), "test": np.random.default_rng(T34.WHITEN_SEED + 1)}
    runs = {}
    for label in T34.TRAIN_RUNS + T34.TEST_RUNS:
        ti, ii, jj = xa.pool_draws(rngs["train" if label in T34.TRAIN_RUNS else "test"], n_train, T34.NZ, T34.NY,
                                   T34.WHITEN_CELLS)
        if label not in labels:
            continue
        xa.set_inputs(T34, CHANS, "Q_c")
        run = T34.load_run(label)
        tr, val = run["parts"]["train"], run["parts"]["val"]
        if tr.stop - tr.start != n_train or run["pr"] != 1.0:
            raise ValueError(f"{label}: {tr.stop - tr.start} training frames (expected {n_train}), Pr {run['pr']}")
        n = val.stop - val.start
        sel = np.unique(np.linspace(0, n - 1, min(ja.N_FRAMES, n)).round().astype(int))
        frames = [run["frames"]["val"][i] for i in sel]
        h = T34.pp.read_humid(T34.run_paths(label)["humid"])
        qc = run["y"][val][sel].astype(np.float64)
        qd = read_npz_frames(T34.run_paths(label)["qdc"], "Q_d", frames)
        runs[label] = {"label": label, "chans": list(CHANS), "fields": np.ascontiguousarray(run["fields"][val][sel]),
                       "qc": qc, "qd": qd, "q": qd + qc, "frames": frames, "xm": run["xm"], "params": run["params"],
                       "ra": run["ra"], "kappa": run["kappa"], "gamma_tau": h["gamma"] / h["tau"],
                       "pool": xa.pool_patches(run["fields"][tr], ti, ii, jj), "pool_rows": ii}
        del run
        gc.collect()
        print(f"  loaded {label}", flush=True)
    return runs


def cell_values(F):
    """F (B, NZ, NY) at the explained cells, in the order of xa (row sets, then frame, row, column)."""
    row_sets, cols = xa.interior_row_sets(F.shape[1], F.shape[2])
    return np.concatenate([xa.cells(F, rows, cols).reshape(-1) for rows in row_sets])


def cell_patches(X):
    """X (B, NZ, NY, C) -> (n, C, 9) patches of the explained cells, same order as cell_values."""
    row_sets, cols = xa.interior_row_sets(X.shape[1], X.shape[2])
    return np.concatenate([xa.patches_of(X, rows, cols).reshape(-1, X.shape[3], 9) for rows in row_sets])


def class_masks(wet):
    return {"all": np.ones_like(wet, bool), "wet": wet, "dry": ~wet}


def seed_pairs(setups):
    seeds = [set(setups[k]["seeds"]) for k in ("Q_a", "Q_d_a", "C4")]
    if not all(s == set(SEEDS) for s in seeds):
        sys.exit(f"seeds differ from {SEEDS}: {seeds}")
    return SEEDS


def check_spacing(labels):
    """dz = dy = 1/NZ is assumed (Laplacian units, exact Q, as jacobian_attribution.py): checked against each bou.in."""
    for l in labels:
        p = T34.pp.read_bou(T34.run_paths(l)["bou"])
        dz, dy = 1.0 / p["nz"], p["ly"] / p["ny"]
        if p["nz"] != T35.NZ or abs(dy - dz) > 1e-12 * dz:
            sys.exit(f"{l}: grid {p['nz']} x {p['ny']}, ly {p['ly']}: dz = {dz}, dy = {dy}; dz = dy = 1/NZ is assumed")


def refuse(paths):
    old = [str(p) for p in paths if Path(p).exists()]
    if old:
        sys.exit("refusing to overwrite: " + ", ".join(old))


def ra_sorted(runs):
    return sorted(runs, key=lambda l: runs[l]["ra"])


# ============================================================
# 2. Part jac: Jacobians
# ============================================================

def jacobian_cells(model, setup, X):
    """J (n, C, 9) = d Q^ / d x_ch(i + p, j + q) in raw units at the explained cells (patch index m = 3 (p + 1) + (q + 1))."""
    inputs = list(setup["cfg"]["inputs"])
    sy = setup["target_norm"][1]
    scale = np.array([sy / setup["input_norm"][c][1] for c in inputs])[None, :, None]
    return xa.Patches(model).grad(cell_patches(X)) * scale


def per_cell(J, inputs, run, dz2, qd_cells):
    """Per-cell quantities of the docstring: Jt_b (n, 9), JbT (n, 9), J_s (n, 9), J_rho (n,), J_rho^ex (n,); NaN for a
    channel that is not an input."""
    n = len(J)
    nan9 = np.full((n, 9), np.nan)
    jb = J[:, inputs.index("b")] * (dz2 / run["kappa"]) if "b" in inputs else nan9
    js = J[:, inputs.index("q_minus_qs")] if "q_minus_qs" in inputs else nan9
    jr = J[:, inputs.index("log10_Ra")].sum(axis=1) if "log10_Ra" in inputs else np.full(n, np.nan)
    return {"Jb": jb, "JbT": jb @ U, "Js": js, "Jr": jr, "Jr_ex": -0.5 * math.log(10.0) * qd_cells}


def sum_jacobian(*parts):
    """Jacobian (n, 3, 9) over Q_INPUTS of the sum of models, from their (J (n, C, 9), inputs): d(sum)/dx_ch is the sum
    of the parts' slopes to ch (0 where a part does not take ch)."""
    out = np.zeros((len(parts[0][0]), len(Q_INPUTS), 9))
    for J, inputs in parts:
        for k, c in enumerate(inputs):
            out[:, Q_INPUTS.index(c)] += J[:, k]
    return out


def describe(q, mask):
    """Percentiles PCT over the cells in mask of the per-cell quantities q; r_rho from sums."""
    out = {"n": int(mask.sum())}
    for k in ("Jb", "JbT", "Js"):
        v = q[k][mask]
        out[k] = np.percentile(v, PCT, axis=0) if len(v) and np.isfinite(v).all() else np.full((3, 9), np.nan)
    v = q["Jr"][mask]
    out["Jr"] = np.percentile(v, PCT) if len(v) and np.isfinite(v).all() else np.full(3, np.nan)
    den = float(np.sum(q["Jr_ex"][mask] ** 2))
    out["r_rho"] = float(np.sum(q["Jr"][mask] * q["Jr_ex"][mask])) / den if den > 0 and np.isfinite(v).all() else np.nan
    return out


def jac_compute(setups, runs, gamma_tau):
    """stats[(tag, seed, run or 'train', class)] for every model; exact under seed 0."""
    dz2 = (1.0 / T35.NZ) ** 2
    exact = exact_q_model(setups["Q_a"], gamma_tau)
    pooled = {}
    stats = {}
    for label in ra_sorted(runs):
        run = runs[label]
        t0 = time.time()
        wet = cell_values(run["qc"]) > 0
        qd_c = cell_values(run["qd"])
        masks = class_masks(wet)
        keep = wet | (np.cumsum(~wet) % DRY_STRIDE == 1)
        raw = {}
        for tag in ("Q_a", "Q_d_a", "C4"):
            st = setups[tag]
            X = inputs_for(st, run["fields"], run)
            for s in SEEDS:
                raw[(tag, s)] = (jacobian_cells(st["models"][s], st, X), list(st["cfg"]["inputs"]))
            if tag == "Q_a":
                raw[("exact", 0)] = (jacobian_cells(exact, st, X), list(Q_INPUTS))
            del X
        for s in SEEDS:
            raw[("sum", s)] = (sum_jacobian(raw[("Q_d_a", s)], raw[("C4", s)]), list(Q_INPUTS))
        for (tag, s), (J, inputs) in raw.items():
            q = per_cell(J, inputs, run, dz2, qd_c)
            for cls, m in masks.items():
                stats[(tag, s, label, cls)] = describe(q, m)
            if label in T34.TRAIN_RUNS:
                p = pooled.setdefault((tag, s), {"wet": [], **{k: [] for k in q}})
                for k in q:
                    p[k].append(q[k][keep].astype(np.float32))
                p["wet"].append(wet[keep])
        print(f"  jac {label:<15} {int(wet.sum())} wet / {len(wet)} cells; Q_a 42: wet J_s,00 "
              f"{stats[('Q_a', 42, label, 'wet')]['Js'][1, 4]:+.3f}, dry Jt_b,00 "
              f"{stats[('Q_a', 42, label, 'dry')]['Jb'][1, 4]:+.3f}, r_rho {stats[('Q_a', 42, label, 'all')]['r_rho']:+.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)
        del raw
        gc.collect()
    for (tag, s), p in pooled.items():
        q = {k: np.concatenate(v).astype(np.float64) for k, v in p.items() if k != "wet"}
        wet = np.concatenate(p["wet"])
        for cls, m in class_masks(wet).items():
            stats[(tag, s, "train", cls)] = describe(q, m)
    return stats


def tags_seeds(stats, tag):
    return sorted({k[1] for k in stats if k[0] == tag})


def seed_mean(stats, tag, label, cls, key, i=1):
    """Mean over seeds of the percentile row i (1: median) of stats[...][key]."""
    vals = [stats[(tag, s, label, cls)][key] for s in tags_seeds(stats, tag) if (tag, s, label, cls) in stats]
    return np.mean([v[i] if np.ndim(v) else v for v in vals], axis=0) if vals else None


def jac_summary(stats, runs):
    labels = ra_sorted(runs)
    L = ["Jacobian of the Q networks at the explained cells (xa: interior sampled cells of 8 validation-block frames per",
         "run). Jt_b = dz^2 dQ^/db / kappa (exact L = [0 1 0; 1 -4 1; 0 1 0]); JbT = Jt_b . u_ab, columns u00 u10 u01 u20 u11",
         "u02 u21 u12 u22 (exact 0 0 0 +1.414 0 +1.414 0 0 -4); J_s = dQ^/ds raw (exact 20 at the centre of wet cells, 0",
         "otherwise); J_rho = dQ^/d(log10 Ra) for a uniform change; r_rho = sum J_rho J_rho^ex / sum (J_rho^ex)^2,",
         "J_rho^ex = -(ln 10 / 2) Q_d^DNS (1 for the exact Q). Medians [5th, 95th percentiles] over the cells. 'train':",
         f"the 8 training runs pooled (all wet cells, every {DRY_STRIDE}th dry cell). Rows of a 3x3: p = -1, 0, +1 (z);",
         "columns q = -1, 0, +1 (y). Wet: DNS Q_c > 0.", ""]
    for tag in ("Q_a", "sum", "Q_d_a", "C4", "exact"):
        for s in tags_seeds(stats, tag):
            L.append(f"=== {tag}" + ("" if tag == "exact" else f" seed {s}"))
            for cls in ("dry", "wet"):
                st = stats[(tag, s, "train", cls)]
                L.append(f"  train, {cls} cells (n = {st['n']}):")
                for name, key in (("Jt_b", "Jb"), ("J_s", "Js")):
                    M = st[key][1].reshape(3, 3)
                    if np.isfinite(M).all():
                        L.append(f"    {name} median (sum {M.sum():+.4g}):  " + "  |  ".join(
                            " ".join(f"{x:+9.4g}" for x in row) for row in M))
                if np.isfinite(st["JbT"][1]).all():
                    L.append("    JbT median: " + " ".join(f"{x:+8.4f}" for x in st["JbT"][1]))
                if np.isfinite(st["Jr"][1]):
                    L.append(f"    J_rho median {st['Jr'][1]:+.4g} [{st['Jr'][0]:+.3g}, {st['Jr'][2]:+.3g}], r_rho {st['r_rho']:+.4f}")
            L.append(f"  {'run':<15} {'n dry':>7} {'n wet':>6} | dry: JbT u20 u02 u22 medians      | wet: J_s,00 "
                     f"[p5, p95]        sum_pq med(J_s) | r_rho all")
            for l in labels:
                d, w, a = (stats[(tag, s, l, c)] for c in ("dry", "wet", "all"))
                L.append(f"  {l:<15} {d['n']:>7d} {w['n']:>6d} | " + " ".join(f"{d['JbT'][1][k]:+8.3f}" for k in (3, 5, 8))
                         + f" | {w['Js'][1][4]:+8.3f} [{w['Js'][0][4]:+7.3f}, {w['Js'][2][4]:+7.3f}] {np.sum(w['Js'][1]):+8.3f}"
                         + f" | {a['r_rho']:+8.4f}")
            L.append("")
    return "\n".join(L)


def save_stats(stats, path):
    np.savez_compressed(path, **{f"{t}__{s}__{l}__{c}__{k}": np.asarray(v)
                                 for (t, s, l, c), st in stats.items() for k, v in st.items()})


def stencil_panel(ax, M, lim, fmt="{:+.3g}"):
    ax.imshow(M, origin="lower", cmap="RdBu_r", vmin=-lim, vmax=lim)
    for a in range(3):
        for c in range(3):
            ax.text(c, a, fmt.format(M[a, c]), ha="center", va="center", fontsize=8,
                    color="w" if abs(M[a, c]) > 0.6 * lim else "k")
    ax.set_xticks([0, 1, 2], ["$-1$", "$0$", "$1$"])
    ax.set_yticks([0, 1, 2], ["$-1$", "$0$", "$1$"])


def plot_jac_stencils(stats, path, gamma_tau):
    tags = ("Q_a", "sum", "Q_d_a", "C4", "exact")
    cols = (("Jb", "dry", r"$\frac{\Delta z^2}{\kappa}\frac{\partial\hat Q}{\partial b_{i+p,j+q}}$, dry"),
            ("Jb", "wet", r"$\frac{\Delta z^2}{\kappa}\frac{\partial\hat Q}{\partial b_{i+p,j+q}}$, wet"),
            ("Js", "wet", r"$\frac{\partial\hat Q}{\partial s_{i+p,j+q}}$, wet"),
            ("Js", "dry", r"$\frac{\partial\hat Q}{\partial s_{i+p,j+q}}$, dry"))
    lims = {"Jb": 5.0, "Js": gamma_tau}
    fig, axes = plt.subplots(len(tags), len(cols), figsize=(3.3 * len(cols) + 1.5, 2.7 * len(tags) + 1.2),
                             squeeze=False, layout="constrained")
    for i, tag in enumerate(tags):
        for j, (key, cls, title) in enumerate(cols):
            ax = axes[i][j]
            M = seed_mean(stats, tag, "train", cls, key)
            if M is None or not np.isfinite(M).all():
                ax.axis("off")
                ax.text(0.5, 0.5, "not an input", ha="center", va="center", fontsize=10, transform=ax.transAxes)
            else:
                stencil_panel(ax, M.reshape(3, 3), lims[key])
            if i == 0:
                ax.set_title(title, fontsize=12)
            if i == len(tags) - 1:
                ax.set_xlabel(r"$q$", fontsize=11)
            if j == 0:
                ax.set_ylabel(TAG_TEX[tag] + "\n" + r"$p$", fontsize=11)
    fig.suptitle(r"median over the cells of the 8 training runs (wet: $Q_c^{\mathrm{DNS}}>0$), mean over seeds; exact: "
                 r"$\frac{\Delta z^2}{\kappa}\frac{\partial Q}{\partial b}=[0\,1\,0;\,1\,{-4}\,1;\,0\,1\,0]$, "
                 r"$\frac{\partial Q}{\partial s}=\frac{\gamma}{\tau}H(s)\,\delta_{p0}\delta_{q0}$, $\frac{\gamma}{\tau}="
                 + f"{gamma_tau:g}$; colour " + r"$\pm5$ ($b$), $\pm\gamma/\tau$ ($s$)", fontsize=11)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_jac_by_run(stats, runs, path, gamma_tau):
    labels = ra_sorted(runs)
    panels = ((lambda st: st["dry"]["JbT"][1][3], r"$\mathbf{J}_b\cdot\mathbf{u}_{20}$, dry", math.sqrt(2.0)),
              (lambda st: st["dry"]["JbT"][1][5], r"$\mathbf{J}_b\cdot\mathbf{u}_{02}$, dry", math.sqrt(2.0)),
              (lambda st: st["dry"]["JbT"][1][8], r"$\mathbf{J}_b\cdot\mathbf{u}_{22}$, dry", -4.0),
              (lambda st: st["wet"]["Js"][1][4], r"$\partial\hat Q/\partial s_{i,j}$, wet", gamma_tau),
              (lambda st: float(np.sum(st["wet"]["Js"][1])), r"$\sum_{p,q}\partial\hat Q/\partial s_{i+p,j+q}$, wet", gamma_tau),
              (lambda st: st["all"]["r_rho"], r"$r_\rho$, all cells", 1.0))
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5), layout="constrained")
    offs = {"Q_a": -0.24, "sum": -0.08, "Q_d_a": 0.08, "C4": 0.24}
    for ax, (value, title, ref) in zip(axes.ravel(), panels):
        xs = xa.run_axis(ax, labels, runs)
        ax.axhline(ref, color="k", ls="--", lw=1.2)
        for tag, o in offs.items():
            for s in tags_seeds(stats, tag):
                v = np.array([value({c: stats[(tag, s, l, c)] for c in CLASSES}) for l in labels], float)
                if np.isfinite(v).any():
                    ax.plot(xs + o, v, "o", color=TAG_COL[tag], ms=5)
        ax.set_title(title, fontsize=12)
    axes[0][0].set_ylabel(r"$\mathbf{J}_b=\frac{\Delta z^2}{\kappa}\nabla_{\mathbf{b}}\hat Q$ (median over cells)", fontsize=11)
    axes[1][0].set_ylabel(r"median over cells", fontsize=11)
    handles = [Line2D([], [], marker="o", ls="", color=TAG_COL[t], label=TAG_TEX[t] + " (one per seed)")
               for t in ("Q_a", "sum", "Q_d_a", "C4")]
    handles += [Line2D([], [], color="k", ls="--", label=r"exact $Q$: $\sqrt{2},\sqrt{2},-4$; $\gamma/\tau$; $r_\rho=1$"),
                Patch(color="0.92", label=TEST_TEX)]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=10)
    fig.suptitle(r"$r_\rho=\frac{\sum_c J_\rho J_\rho^{\mathrm{ex}}}{\sum_c(J_\rho^{\mathrm{ex}})^2},\ J_\rho=\frac{\partial\hat Q}"
                 r"{\partial\rho},\ J_\rho^{\mathrm{ex}}=-\frac{\ln10}{2}Q_d^{\mathrm{DNS}},\ \rho=\log_{10}\mathrm{Ra}$;  "
                 r"$\mathbf{u}_{ab}$: templates (mean, $\partial_z$, $\partial_y$, ... on 3 points)", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 3. Part xai: exact Shapley values and ablation
# ============================================================

def xai_one(model, setup, X, run, refs, rho, exact=False):
    t = setup["t"]
    ctx = xa.context(t, setup, run, refs, rho, None)
    inputs, ref_n = ctx["inputs"], ctx["ref_n"]
    ib, i_s = inputs.index("b"), inputs.index("q_minus_qs")
    my, sy = setup["target_norm"]
    pf = xa.Patches(model)

    def psi(P):
        return sy * pf.pred(P) + my

    P = cell_patches(X)
    px = psi(P)
    frame = cell_values(predict_raw(model, setup, X))
    dns = cell_values(run["q"])
    wet = cell_values(run["qc"]) > 0
    phi_c, abl_c, v0_c = xa.channel_shapley(P, ref_n, psi, px)
    phi_t, abl_t, v0_t = xa.template_shapley(P, ib, ref_n[ib], psi, px)
    out = {"patch_check": xa.rel_max(px - frame, px), "eff_ch": xa.rel_max(phi_c.sum(axis=1) - (px - v0_c), px - v0_c),
           "eff_T": xa.rel_max(phi_t.sum(axis=1) - (px - v0_t), px - v0_t)}
    rows = {k: [] for k in ("n", "e0", "e_ch", "S_ch", "S_T", "e_T")}
    for cls, m in class_masks(wet).items():
        d2 = float(np.sum(dns[m] ** 2))

        def err(v):
            return math.sqrt(float(np.sum((v[m] - dns[m]) ** 2)) / d2) if d2 > 0 else float("nan")
        rows["n"].append(int(m.sum()))
        rows["e0"].append(err(px))
        rows["e_ch"].append([err(abl_c[:, k]) for k in range(len(inputs))])
        rows["S_ch"].append(ja.shares(phi_c[m]))
        rows["S_T"].append(ja.shares(phi_t[m]))
        rows["e_T"].append([err(abl_t[:, k]) for k in range(9)])
    out.update({k: np.array(v) for k, v in rows.items()})
    if exact:
        mb, sb = setup["input_norm"]["b"]
        ms, ss = setup["input_norm"]["q_minus_qs"]
        dz2, kc, gt = (1.0 / T35.NZ) ** 2, run["kappa"], run["gamma_tau"]
        kbar = 10.0 ** (-rho / 2.0)
        Lb, Lbbar = (mb + sb * P[:, ib]) @ L9, float(refs["b"] @ L9)
        an = np.zeros_like(phi_c)
        an[:, ib] = 0.5 * (kc + kbar) * (Lb - Lbbar) / dz2
        an[:, inputs.index("log10_Ra")] = 0.5 * (kc - kbar) * (Lb + Lbbar) / dz2
        an[:, i_s] = gt * (np.maximum(ms + ss * P[:, i_s, 4], 0.0) - max(float(refs["q_minus_qs"][4]), 0.0))
        a = sb * (P[:, ib] - ref_n[ib]) @ U
        out["dev_ch"] = xa.rel_max(phi_c - an, px - v0_c)
        out["dev_T"] = xa.rel_max(phi_t - kc / dz2 * xa.L_U * a, px - v0_t)
    return out


def load_npz_results(path, version, config, key):
    """{(seed, run): array} of `key` for (version, config) from a data file of xa.save_data."""
    out = {}
    with np.load(path) as d:
        for name in d.files:
            p = name.split("__")
            if len(p) == 5 and p[0] == version and p[1] == config and p[4] == key:
                out[(int(p[2]), p[3])] = d[name]
    return out


def xai_summary(res, runs, info):
    labels = ra_sorted(runs)
    L = ["Exact Shapley values and ablation of the Q networks at the explained cells (xa). Channels b, s, rho; references",
         f"bbar, the mean s patch (centre {info['s_ref_centre']:+.4f}), rho_bar = {info['rho_bar']:.6f}. Templates of b:",
         "u00 u10 u01 u20 u11 u02 u21 u12 u22 (s, rho at the cell's own values). S: shares sum phi^2 / sum sum phi^2;",
         "e: relative error against the DNS Q; e_-x: x at its reference. Columns per class: all | wet | dry.",
         f"bbar of the v035 whitening vs v034: max diff {info['bbar_check']:.1e}; pool mean b vs bbar {info['pool_check']:.1e}", ""]
    for tag in ("Q_a", "exact"):
        for s in sorted({k[1] for k in res if k[0] == tag}):
            L.append(f"=== {tag}" + ("" if tag == "exact" else f" seed {s}"))
            for l in labels:
                r = res[(tag, s, l)]
                L.append(f"  {l:<15} n {r['n'][0]:>7d} (wet {r['n'][1]:>6d}); patch {r['patch_check']:.1e}, efficiency "
                         f"{r['eff_ch']:.1e} / {r['eff_T']:.1e}" + (f"; analytic dev ch {r['dev_ch']:.1e}, T {r['dev_T']:.1e}"
                                                                   if "dev_ch" in r else ""))
                for i, cls in enumerate(CLASSES):
                    L.append(f"    {cls:<4} e0 {r['e0'][i]:.3e}  e_-b {r['e_ch'][i][0]:.3e} e_-s {r['e_ch'][i][1]:.3e} "
                             f"e_-rho {r['e_ch'][i][2]:.3e} | S_b {r['S_ch'][i][0]:.3e} S_s {r['S_ch'][i][1]:.3e} "
                             f"S_rho {r['S_ch'][i][2]:.3e}")
                    L.append(f"         S_T " + " ".join(f"{x:8.1e}" for x in r["S_T"][i]))
                    L.append(f"         e_-T " + " ".join(f"{x:8.2e}" for x in r["e_T"][i]))
            L.append("")
    return "\n".join(L)


def save_res(res, path):
    np.savez_compressed(path, **{f"{t}__{s}__{l}__{k}": np.asarray(v) for (t, s, l), r in res.items() for k, v in r.items()})


def plot_channel_shares(res, runs, path):
    labels = ra_sorted(runs)
    fig, axes = plt.subplots(2, 3, figsize=(18, 8.5), sharey=True, layout="constrained")
    for i, tag in enumerate(("Q_a", "exact")):
        seeds = sorted({k[1] for k in res if k[0] == tag})
        for j, cls in enumerate(CLASSES):
            ax = axes[i][j]
            xs = xa.run_axis(ax, labels, runs)
            P = 100.0 * np.array([np.mean([res[(tag, s, l)]["S_ch"][j] for s in seeds], axis=0) for l in labels])
            bottom = np.zeros(len(labels))
            for k, c in enumerate(Q_INPUTS):
                ax.bar(xs, P[:, k], 0.7, bottom=bottom, color=CH_COL[c], edgecolor="w", lw=0.4, zorder=3)
                for x, y0, h in zip(xs, bottom, P[:, k]):
                    if h >= 8.0:
                        ax.text(x, y0 + h / 2, f"{h:.0f}", ha="center", va="center", fontsize=7, color="w", zorder=4)
                bottom += P[:, k]
            ax.set_ylim(0, 105)
            ax.set_title(f"{TAG_TEX[tag]}, {cls}", fontsize=12)
        axes[i][0].set_ylabel(r"$100\,S_{ch}\ [\%]$", fontsize=12)
    handles = [Patch(color=CH_COL[c], label=CH_TEX[c]) for c in Q_INPUTS] + [Patch(color="0.92", label=TEST_TEX)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=11)
    fig.suptitle(r"$S_{ch}=\frac{\sum_c\phi_{ch}^2}{\sum_c\sum_{ch'}\phi_{ch'}^2}$, $\phi$: exact Shapley values over "
                 r"$(b,s,\rho)$, $v(S)=\hat Q$(channels in $S$ from $\mathbf{x}$, the others at $\bar{\mathbf{b}},\bar{\mathbf{s}},"
                 r"\bar\rho$); wet: $Q_c^{\mathrm{DNS}}>0$; mean over seeds; numbers where $\geq8$", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_template_shares(res, qd34, runs, path):
    labels = ra_sorted(runs)
    k = np.arange(9)
    ticks = [ja.template_tex(a, b) for a, b in ja.TEMPLATE_ORDER]
    fig, axes = plt.subplots(2, 5, figsize=(24, 9.5), sharey=True, squeeze=False, layout="constrained")
    seeds = sorted({key[1] for key in res if key[0] == "Q_a"})
    for ax, l in zip(axes.ravel(), labels):
        ex = res[("exact", 0, l)]["S_T"][2]
        kc = [ja.TEMPLATE_ORDER.index(c) for c in ja.CURVATURE]        # the exact operator's only templates; the
        ax.plot(k[kc], ex[kc], color="k", ls="none", marker="D", ms=6, zorder=5)   # others are round-off (not drawn)
        for i, (cls, ls) in enumerate((("all", "-"), ("dry", ":"))):
            S = np.mean([res[("Q_a", s, l)]["S_T"][CLASSES.index(cls)] for s in seeds], axis=0)
            ax.plot(k, S, color=TAG_COL["Q_a"], ls=ls, marker="o", ms=4.5, lw=1.4)
        v34 = [qd34[(s, l)] for s in SEEDS if (s, l) in qd34]
        if v34:
            ax.plot(k, np.mean(v34, axis=0), color=TAG_COL["Q_d_a"], ls="-", marker="s", ms=4.5, lw=1.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-9, 2.0)
        ax.set_xticks(k, ticks, rotation=90, fontsize=11)
        ax.set_title(ja.ra_tex(l, runs[l]["ra"]), fontsize=11)
        ax.grid(True, ls=":", alpha=0.5)
    for r in range(2):
        axes[r][0].set_ylabel(r"$S_k$", fontsize=12)
    handles = [Line2D([], [], color="k", ls="none", marker="D", label=r"exact $Q$, dry cells ($=\kappa\nabla^2 b$; $S_k=0$ elsewhere)"),
               Line2D([], [], color=TAG_COL["Q_a"], ls="-", marker="o", label=TAG_TEX["Q_a"] + ", all cells"),
               Line2D([], [], color=TAG_COL["Q_a"], ls=":", marker="o", label=TAG_TEX["Q_a"] + ", dry cells"),
               Line2D([], [], color=TAG_COL["Q_d_a"], ls="-", marker="s", label="v034 " + TAG_TEX["Q_d_a"] + ", all cells")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(xa.SHAPLEY_TEX + r";  mean over seeds", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_ablation(res, runs, path):
    labels = ra_sorted(runs)
    seeds = sorted({k[1] for k in res if k[0] == "Q_a"})
    panels = [("e0", None, r"$e_\emptyset$"), ("e_ch", 0, r"$e_{-b}:\ \mathbf{b}\to\bar{\mathbf{b}}$"),
              ("e_ch", 1, r"$e_{-s}:\ \mathbf{s}\to\bar{\mathbf{s}}$"), ("e_ch", 2, r"$e_{-\rho}:\ \rho\to\bar\rho$"),
              ("e_T", 3, r"$e_{-\mathbf{u}_{20}}$"), ("e_T", 5, r"$e_{-\mathbf{u}_{02}}$"),
              ("e_T", 8, r"$e_{-\mathbf{u}_{22}}$"), ("e_T", 0, r"$e_{-\mathbf{u}_{00}}$")]
    fig, axes = plt.subplots(2, 4, figsize=(22, 9.5), sharey=True, layout="constrained")
    for ax, (key, k, title) in zip(axes.ravel(), panels):
        xs = xa.run_axis(ax, labels, runs)

        def val(r):
            return float(r[key][0] if k is None else r[key][0][k])
        for s in seeds:
            ax.plot(xs, [val(res[("Q_a", s, l)]) for l in labels], "o", color=TAG_COL["Q_a"], ms=5)
        ax.plot(xs, [val(res[("exact", 0, l)]) for l in labels], color="k", ls="--", marker="D", ms=5)
        ax.axhline(1.0, color="0.5", ls="-.", lw=1.0)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=13)
    for r in range(2):
        axes[r][0].set_ylabel(r"$e=\sqrt{\frac{\sum_c(\hat Q-Q^{\mathrm{DNS}})^2}{\sum_c(Q^{\mathrm{DNS}})^2}}$", fontsize=13)
    handles = [Line2D([], [], color=TAG_COL["Q_a"], marker="o", ls="", label=TAG_TEX["Q_a"] + " (one per seed)"),
               Line2D([], [], color="k", ls="--", marker="D", label=r"exact $Q$"),
               Line2D([], [], color="0.5", ls="-.", label=r"$e=1$: prediction 0"), Patch(color="0.92", label=TEST_TEX)]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=10)
    fig.suptitle(r"all explained cells; $e_{-x}$: input $x$ at its reference, the DNS target unchanged; "
                 r"$e_{-\mathbf{u}_k}$: $\mathbf{b}\to\mathbf{b}-a_k\mathbf{u}_k$, $a_k=\mathbf{u}_k\cdot(\mathbf{b}-\bar{\mathbf{b}})$",
                 fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 4. Part eval: errors on every frame of the blocks
# ============================================================

def beta(sum_pred, sum_dns, sum_qc):
    """Domain-mean bias of a predicted source term as a fraction of the mean latent heating [model vs DNS]:
    beta = (sum Q^ - sum Q) / sum Q_c, the three sums over the same cells and frames (means over them give the same
    value). Reference values: 0 for a perfect prediction; beta_0 = -sum Q / sum Q_c for Q^ = 0, which is small because
    the box integral of Q = d<b>/dt nearly vanishes: beta measures the box budget of the prediction, not its skill
    (that is e). The denominator is >= 0 in every cell (Q_c >= 0) and positive over a run's frames; the ratio
    sum Q^ / sum Q that beta replaced (2026-11-22) divided by the nearly vanishing sum Q, and a relative error
    |x - x^| / |x| has no meaning as x -> 0 (Higham 2002, sec. 1.2)."""
    if not sum_qc > 0:
        raise ValueError(f"sum Q_c = {sum_qc}: beta is undefined")
    return (sum_pred - sum_dns) / sum_qc


def eval_run(setups, label):
    """Error sums per (model, seed, set, class) on one run; z-profile reference."""
    xa.set_inputs(T34, CHANS, "Q_c")
    run = T34.load_run(label)
    used = run["frames"]["train"] + run["frames"]["val"]
    q = read_npz_frames(T34.run_paths(label)["qdc"], "Q_d", used)
    q += run["y"].astype(np.float64)
    wet_all = run["y"] > 0
    sets = ({"train": ["train"], "val": ["val"]} if label in T34.TRAIN_RUNS else {"test": ["train", "val"]})
    rec = []
    for st, parts in sets.items():
        idx = np.concatenate([np.arange(run["parts"][p].start, run["parts"][p].stop) for p in parts])
        prof = q[idx].mean(axis=(0, 2))
        acc = {}
        ref = {c: [0.0, 0.0] for c in CLASSES}
        for b0 in range(0, len(idx), BATCH):
            ii = idx[b0:b0 + BATCH]
            y, wet = q[ii], wet_all[ii]
            masks = class_masks(wet)
            for c, m in masks.items():
                ref[c][0] += float(np.sum(((prof[None, :, None] - y) ** 2)[m]))
                ref[c][1] += float(np.sum(y[m] ** 2))
            X = {k: inputs_for(setups[k], run["fields"][ii], run) for k in ("Q_a", "Q_d_a", "C4")}
            for s in SEEDS:
                preds = {"Q_a": predict_raw(setups["Q_a"]["models"][s], setups["Q_a"], X["Q_a"]),
                         "sum": predict_raw(setups["Q_d_a"]["models"][s], setups["Q_d_a"], X["Q_d_a"])
                         + predict_raw(setups["C4"]["models"][s], setups["C4"], X["C4"])}
                for tag, p in preds.items():
                    a = acc.setdefault((tag, s), {c: [0.0, 0.0] for c in CLASSES} | {"sum_p": 0.0, "sum_y": 0.0, "sum_c": 0.0})
                    e2 = (p - y) ** 2
                    for c, m in masks.items():
                        a[c][0] += float(e2[m].sum())
                        a[c][1] += float(np.sum(y[m] ** 2))
                    a["sum_p"] += float(p.sum())
                    a["sum_y"] += float(y.sum())
                    a["sum_c"] += float(np.sum(run["y"][ii], dtype=np.float64))     # run["y"] is Q_c here
        for (tag, s), a in acc.items():
            r = {"model": tag, "seed": s, "run": label, "set": st, "ra": run["ra"], "kappa": run["kappa"],
                 "frac_wet": float(wet_all[idx].mean()), "beta": beta(a["sum_p"], a["sum_y"], a["sum_c"]),
                 "beta_zero": beta(0.0, a["sum_y"], a["sum_c"])}
            for c in CLASSES:
                r[f"rel_mse_{c}"] = a[c][0] / a[c][1]
                r[f"e_{c}"] = math.sqrt(a[c][0] / a[c][1])
                r[f"ref_{c}"] = math.sqrt(ref[c][0] / ref[c][1])
            rec.append(r)
    del run, q
    gc.collect()
    print(f"  eval {label} done", flush=True)
    return rec


def eval_summary(records, loss_check, train_info):
    L = ["v035 Q_a and the sum Q^_d(v034 Q_d_a) + Q^_c(v034 C4), paired by seed. e = sqrt(sum (Q^ - Q)^2 / sum Q^2) per run",
         "and set over every frame of the block(s); ref: the run's mean profile of Q over the same frames; wet: DNS Q_c > 0.",
         "beta = (sum Q^ - sum Q) / sum Q_c over the same cells and frames (0: no bias of the box-integrated source term);",
         "beta_0: beta of Q^ = 0 (reference).", "",
         "1. Validation loss recomputed (mean over the training runs of the relative MSE) vs the Keras history at the best epoch"]
    for s, (a, b) in loss_check.items():
        L.append(f"   Q_a seed {s}: {a:.6e} vs {b:.6e} (relative difference {abs(a - b) / b:.1e})")
    L += ["", "2. Training (per seed): epochs, best epoch, stop, epochs with a learning-rate drop, final lr, relative change",
          "   of the training loss over the last 10 epochs, best validation loss"]
    for s, v in train_info.items():
        L.append(f"   Q_a {s}: {v['epochs']} {v['best_epoch']} {v['stopped']} {v['lr_drops']} {v['final_lr']:.2e} "
                 f"{v['train_change_last10']:+.4f} {v['best_val_loss']:.4e}")
    L += ["", "3. Per run: mean over seeds [min, max]",
          f"   {'run':<15} {'set':<5} {'model':<5} {'e all':>9} {'[min,':>9} {'max]':>9} {'e wet':>9} {'e dry':>9} "
          f"{'ref all':>9} {'beta':>10} {'beta_0':>10} {'frac wet':>8}"]
    for label in T34.TRAIN_RUNS + T34.TEST_RUNS:
        for st in ("train", "val", "test"):
            for tag in ("Q_a", "sum"):
                rs = [r for r in records if r["run"] == label and r["set"] == st and r["model"] == tag]
                if not rs:
                    continue
                e = [r["e_all"] for r in rs]
                L.append(f"   {label:<15} {st:<5} {tag:<5} {np.mean(e):9.3e} {min(e):9.3e} {max(e):9.3e} "
                         f"{np.mean([r['e_wet'] for r in rs]):9.3e} {np.mean([r['e_dry'] for r in rs]):9.3e} "
                         f"{rs[0]['ref_all']:9.3e} {np.mean([r['beta'] for r in rs]):+10.3e} {rs[0]['beta_zero']:+10.3e} "
                         f"{rs[0]['frac_wet']:8.4f}")
    return "\n".join(L) + "\n"


def plot_eval(records, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True, layout="constrained")
    for ax, cls in zip(axes, CLASSES):
        for tag, off in (("Q_a", -0.02), ("sum", 0.02)):
            for st, mk in (("val", "o"), ("test", "*")):
                rs = [r for r in records if r["model"] == tag and r["set"] == st]
                ax.scatter(np.log10([r["ra"] for r in rs]) + off, [r[f"e_{cls}"] for r in rs], marker=mk,
                           color=TAG_COL[tag], s=90 if mk == "*" else 26, alpha=0.85)
        refs = {(r["run"], r["set"]): (r["ra"], r[f"ref_{cls}"]) for r in records if r["set"] in ("val", "test")}
        ax.scatter(np.log10([v[0] for v in refs.values()]), [v[1] for v in refs.values()], marker="_", color="0.25",
                   s=260, linewidths=2)
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}\mathrm{Ra}$", fontsize=12)
        ax.set_title({"all": "all cells", "wet": r"wet: $Q_c^{\mathrm{DNS}}>0$", "dry": r"dry: $Q_c^{\mathrm{DNS}}=0$"}[cls],
                     fontsize=12)
        ax.grid(True, which="both", ls=":", alpha=0.5)
    axes[0].set_ylabel(r"$e=\sqrt{\frac{\sum(\hat Q-Q)^2}{\sum Q^2}}$ (one marker per seed)", fontsize=12)
    handles = [Line2D([], [], marker="o", ls="", color=TAG_COL[t], label=TAG_TEX[t]) for t in ("Q_a", "sum")]
    handles += [Line2D([], [], marker="o", ls="", color="0.4", label="validation block"),
                Line2D([], [], marker="*", ls="", color="0.4", ms=11, label="test run (both blocks)"),
                Line2D([], [], marker="_", ls="", color="0.25", ms=14, mew=2, label=r"best $Q(z)$ profile"),
                Line2D([], [], color="0.5", ls=":", label=r"$e=1$: $\hat Q=0$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=10)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(setup, path):
    fig, (ax, axl) = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    for s, h in sorted(setup["histories"].items()):
        ep = np.arange(1, len(h["loss"]) + 1)
        line, = ax.plot(ep, h["loss"], lw=1.3, label=f"seed {s}, training")
        ax.plot(ep, h["val_loss"], ls="--", color=line.get_color(), lw=1.3, label=f"seed {s}, validation")
        b = setup["summary"][s]["best_epoch"]
        ax.plot(b, h["val_loss"][b - 1], "o", color=line.get_color())
        axl.plot(ep, h.get("lr", h.get("learning_rate", [])), color=line.get_color(), label=f"seed {s}")
    ax.set_yscale("log"); axl.set_yscale("log")
    ax.set_ylabel(r"$\frac{1}{8}\sum_{r}\frac{\sum_c(\hat Q-Q)^2}{\sum_c Q^2}$", fontsize=13)
    axl.set_ylabel(r"$\eta$ (learning rate)", fontsize=12)
    for a in (ax, axl):
        a.set_xlabel("epoch"); a.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=7, ncol=3, title="dot: best validation epoch", title_fontsize=7); axl.legend(fontsize=8)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 5. Part dns: test runs, every frame
# ============================================================

def load_test_frames(label):
    """Every frame of a run: fields (nt, NZ, NY, 2: b, s) float32, Q = Q_d + Q_c float64, Q_c float32. One stored
    (NZ, NY, nt) float64 array is read at a time (peak about 3.5 x one such array)."""
    p = T34.run_paths(label)
    bou = T34.pp.read_bou(p["bou"])
    zc, _, _ = T34.pp.read_grid(p["grid"], bou["nz"], bou["ny"], bou["ly"])
    fields = q = qc = None
    sources = [T34.CHANNELS[c]["source"] for c in CHANS] + [("qdc", "Q_d"), ("qdc", "Q_c")]   # the training loader's keys
    for c, (kind, key) in enumerate(sources):
        with np.load(p[kind]) as f:
            arr = f[key]
        if arr.ndim != 3 or arr.shape[:2] != (T34.NZ, T34.NY) or (fields is not None and arr.shape[2] != len(fields)):
            raise ValueError(f"{label}: {key} has shape {arr.shape}")
        v = np.moveaxis(arr, -1, 0)
        if fields is None:
            fields = np.empty(v.shape + (2,), np.float32)
            q = np.zeros(v.shape, np.float64)
        if c < 2:
            fields[..., c] = v
        else:
            q += v
            if c == 3:
                qc = v.astype(np.float32)
        del arr, v
    return {"label": label, "fields": fields, "q": q, "qc": qc, "xm": zc[:, 0].astype(np.float64),
            "params": {"log10_Ra": float(np.log10(bou["ra"]))}, "ra": bou["ra"], "nt": len(fields)}


def dns_run(setups, label):
    run = load_test_frames(label)
    nt, q = run["nt"], run["q"]
    dm_c = run["qc"].mean(axis=(1, 2), dtype=np.float64)
    frames = {"t_star": SPIN + int(np.argmax(dm_c[SPIN:])), "t_quiet": SPIN + int(np.argmin(dm_c[SPIN:]))}
    lo, hi = np.percentile(q[SPIN::10], [0.01, 99.99])          # every 10th frame from SPIN on
    r = {"label": label, "ra": run["ra"], "frames": frames, "dm": q.mean(axis=(1, 2)), "dm_c": dm_c,
         "prof": q[SPIN:].mean(axis=(0, 2)), "range": (float(lo), float(hi)), "snap": {}, "pred": {}}
    for w, f in frames.items():
        r["snap"][("dns", w)] = q[f].copy()
    r["joint"] = {}
    edges = np.linspace(lo, hi, NBIN + 1)
    for b0 in range(0, nt, BATCH):
        ii = np.arange(b0, min(b0 + BATCH, nt))
        X = {k: inputs_for(setups[k], run["fields"][ii], run) for k in ("Q_a", "Q_d_a", "C4")}
        y = q[ii]
        for s in SEEDS:
            preds = {"Q_a": predict_raw(setups["Q_a"]["models"][s], setups["Q_a"], X["Q_a"]),
                     "sum": predict_raw(setups["Q_d_a"]["models"][s], setups["Q_d_a"], X["Q_d_a"])
                     + predict_raw(setups["C4"]["models"][s], setups["C4"], X["C4"])}
            for tag, p in preds.items():
                a = r["pred"].setdefault((tag, s), {"dm": np.zeros(nt), "prof": np.zeros(T34.NZ), "sse": np.zeros(nt),
                                                    "sy2": np.zeros(nt)})
                a["dm"][ii] = p.mean(axis=(1, 2))
                a["sse"][ii] = ((p - y) ** 2).sum(axis=(1, 2))
                a["sy2"][ii] = (y ** 2).sum(axis=(1, 2))
                late = ii >= SPIN
                if late.any():
                    a["prof"] += p[late].sum(axis=(0, 2)) / ((nt - SPIN) * T34.NY)
                if s == SNAP_SEED:
                    if late.any():
                        h, _, _ = np.histogram2d(y[late].ravel(), p[late].ravel(), bins=(edges, edges))
                        r["joint"][tag] = r["joint"].get(tag, 0) + h
                    for w, f in frames.items():
                        if f in ii:
                            r["snap"][(tag, w)] = p[list(ii).index(f)].copy()
    del run, q
    gc.collect()
    print(f"  dns {label} done; t* {frames['t_star']}, t_q {frames['t_quiet']}", flush=True)
    return r


def dns_errors(a, sl):
    return math.sqrt(float(a["sse"][sl].sum() / a["sy2"][sl].sum()))


def dns_summary(res):
    L = ["Q networks against the DNS on every frame of the test runs. e = sqrt(sum (Q^ - Q)^2 / sum Q^2) over the frames named;",
         "beta = (mean_t <Q^>(t) - mean_t <Q>(t)) / mean_t <Q_c>(t) over frames 25..1000 (0: no bias); beta_0: beta of",
         "Q^ = 0 (reference); <.>: domain mean.", ""]
    for label, r in res.items():
        dm, dm_c = r["dm"][SPIN:].mean(), r["dm_c"][SPIN:].mean()
        L.append(f"=== {label}: t* = {r['frames']['t_star']} (largest <Q_c>), t_q = {r['frames']['t_quiet']} (smallest <Q_c>), "
                 f"frames >= {SPIN}; beta_0 = {beta(0.0, dm, dm_c):+.3e}")
        L.append(f"  {'model':<5} {'seed':>4} {'e all':>9} {'e 0..24':>9} {'e 25..':>9} {'beta':>10}")
        for (tag, s), a in sorted(r["pred"].items()):
            L.append(f"  {tag:<5} {s:>4} {dns_errors(a, slice(None)):9.3e} {dns_errors(a, slice(0, SPIN)):9.3e} "
                     f"{dns_errors(a, slice(SPIN, None)):9.3e} {beta(a['dm'][SPIN:].mean(), dm, dm_c):+10.3e}")
        L.append("")
    return "\n".join(L)


def save_dns(res, path):
    d = {}
    for label, r in res.items():
        d[f"{label}__dm"], d[f"{label}__dm_c"], d[f"{label}__prof"] = r["dm"], r["dm_c"], r["prof"]
        d[f"{label}__frames"] = np.array([r["frames"]["t_star"], r["frames"]["t_quiet"]])
        for (tag, s), a in r["pred"].items():
            for k, v in a.items():
                d[f"{label}__{tag}__{s}__{k}"] = v
        for tag, h in r["joint"].items():
            d[f"{label}__{tag}__joint"] = h
        for (tag, w), f in r["snap"].items():
            d[f"{label}__{tag}__snap_{w}"] = f
    np.savez_compressed(path, **d)


def plot_dns(res, paths):
    labels = list(res)
    # time series
    fig, axes = plt.subplots(len(labels), 1, figsize=(15, 4.2 * len(labels)), squeeze=False, layout="constrained")
    for ax, l in zip(axes[:, 0], labels):
        r = res[l]
        t = np.arange(len(r["dm"]))
        ax.plot(t, r["dm"], color="k", lw=1.6)
        for (tag, s), a in sorted(r["pred"].items()):
            ax.plot(t, a["dm"], color=TAG_COL[tag], lw=0.9, ls=("-", "--", ":")[SEEDS.index(s)])
        ax.axvspan(0, SPIN - 0.5, color="0.92", lw=0)
        for w, ls in (("t_star", "--"), ("t_quiet", ":")):
            ax.axvline(r["frames"][w], color="0.3", ls=ls, lw=0.8)
        ax.set_title(ja.ra_tex(l, r["ra"]), fontsize=12)
        ax.set_ylabel(r"$\langle Q\rangle(t)=\frac{1}{N_zN_y}\sum_{i,j}Q$", fontsize=12)
    axes[-1][0].set_xlabel(r"$t$ (frame)", fontsize=12)
    handles = [Line2D([], [], color="k", lw=1.6, label="DNS"),
               Line2D([], [], color=TAG_COL["Q_a"], label=TAG_TEX["Q_a"]), Line2D([], [], color=TAG_COL["sum"], label=TAG_TEX["sum"]),
               Line2D([], [], color="0.4", ls="-", label="seed 42"), Line2D([], [], color="0.4", ls="--", label="seed 43"),
               Line2D([], [], color="0.4", ls=":", label="seed 44"), Patch(color="0.92", label=r"$t<25$: never trained on"),
               Line2D([], [], color="0.3", ls="--", label=r"$t^*$"), Line2D([], [], color="0.3", ls=":", label=r"$t_q$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=9, fontsize=9)
    fig.savefig(paths["ts"], dpi=200)
    plt.close(fig)
    # profiles
    fig, axes = plt.subplots(1, len(labels), figsize=(6.5 * len(labels), 6), squeeze=False, layout="constrained")
    for ax, l in zip(axes[0], labels):
        r = res[l]
        z = (np.arange(T34.NZ) + 0.5) / T34.NZ
        ax.plot(r["prof"], z, color="k", lw=2)
        for (tag, s), a in sorted(r["pred"].items()):
            ax.plot(a["prof"], z, color=TAG_COL[tag], lw=1, ls=("-", "--", ":")[SEEDS.index(s)])
        ax.axvline(0.0, color="0.6", lw=0.8)
        ax.set_xlabel(r"$\langle Q\rangle_{y,t}(z)$, $t\geq25$", fontsize=12)
        ax.set_ylabel(r"$z$", fontsize=12)
        ax.set_title(ja.ra_tex(l, r["ra"]), fontsize=12)
    fig.legend(handles=handles[:6], loc="outside lower center", ncol=6, fontsize=10)
    fig.savefig(paths["prof"], dpi=200)
    plt.close(fig)
    # joint histograms
    fig, axes = plt.subplots(len(labels), 2, figsize=(12, 5.6 * len(labels)), squeeze=False, layout="constrained")
    for i, l in enumerate(labels):
        r = res[l]
        lo, hi = r["range"]
        for j, tag in enumerate(("Q_a", "sum")):
            ax = axes[i][j]
            h = r["joint"][tag]
            im = ax.imshow(h.T, origin="lower", extent=(lo, hi, lo, hi), norm=LogNorm(vmin=1, vmax=max(h.max(), 2)),
                           cmap="viridis", aspect="auto")
            ax.plot([lo, hi], [lo, hi], color="r", lw=0.8, ls="--")
            ax.set_xlabel(r"$Q$ (DNS)", fontsize=12)
            ax.set_ylabel(TAG_TEX[tag] + f" (seed {SNAP_SEED})", fontsize=12)
            ax.set_title(ja.ra_tex(l, r["ra"]) + r", $t\geq25$", fontsize=11)
            fig.colorbar(im, ax=ax, label="cells")
    fig.suptitle(r"red dashed: $\hat Q=Q$ (perfect prediction); axes: $[P_{0.01},P_{99.99}]$ of the DNS $Q$ (every 10th "
                 r"frame $t\geq25$); cells outside not counted", fontsize=12)
    fig.savefig(paths["joint"], dpi=200)
    plt.close(fig)
    # snapshots
    rows = [(l, w) for l in labels for w in ("t_star", "t_quiet")]
    fig, axes = plt.subplots(len(rows), 4, figsize=(24, 2.2 * len(rows) + 1.5), squeeze=False, layout="constrained")
    for i, (l, w) in enumerate(rows):
        r = res[l]
        dns = r["snap"][("dns", w)]
        lim = float(np.percentile(np.abs(dns), 99.9)) or 1.0
        for j, (key, title) in enumerate(((("dns", w), r"$Q$ (DNS)"), (("Q_a", w), TAG_TEX["Q_a"] + r"$-Q$"),
                                           (("sum", w), TAG_TEX["sum"] + r"$-Q$"), (("Q_a", w), TAG_TEX["Q_a"]))):
            F = r["snap"][key] - (dns if j in (1, 2) else 0.0)
            im = axes[i][j].imshow(F, origin="lower", extent=(0, 16, 0, 1), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
            axes[i][j].set_title(f"{title}, " + ja.ra_tex(l, r["ra"]) + f", $t={r['frames'][w]}$", fontsize=10)
        fig.colorbar(im, ax=axes[i].tolist(), shrink=0.8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$y$", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$z$", fontsize=11)
    fig.suptitle(rf"seed {SNAP_SEED}; one colour scale per row: $\pm P_{{99.9}}(|Q|)$ of the DNS frame; "
                 r"$t^*$: largest $\langle Q_c\rangle$, $t_q$: smallest, $t\geq25$", fontsize=12)
    fig.savefig(paths["snap"], dpi=150)
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

def load_all_setups():
    return {"Q_a": load_setup(T35, "Q_a"), "Q_d_a": load_setup(T34, "Q_d_a"), "C4": load_setup(T34, "Q_c_C4")}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Analyses of the v035 Q networks (see the docstring).")
    global OUT_DIR
    ap.add_argument("part", choices=tuple(PART_OUT))
    ap.add_argument("--out", default=None, help="output folder under simulations_v2/v035/ (default: analysis)")
    a = ap.parse_args(argv)
    if a.out is not None:
        if a.out in ("", ".", "..") or Path(a.out).name != a.out:
            sys.exit(f"--out {a.out!r}: give a plain folder name")
        OUT_DIR = T35.OUT_DIR / a.out
    t_start = time.time()
    paths = [OUT_DIR / f for f in PART_OUT[a.part]]
    refuse(paths)
    T35.configure_gpu()
    setups = load_all_setups()
    seed_pairs(setups)
    labels = T34.TRAIN_RUNS + T34.TEST_RUNS
    check_spacing(labels)
    if a.part in ("jac", "xai"):
        runs = load_explained(labels)
        gts = {r["gamma_tau"] for r in runs.values()}
        if len(gts) != 1:
            sys.exit(f"gamma/tau differs between runs: {gts}")
        gamma_tau = gts.pop()
        print(f"Data loaded ({time.time() - t_start:.0f} s)", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.part == "jac":
        stats = jac_compute(setups, runs, gamma_tau)
        text = jac_summary(stats, runs)
        paths[1].write_text(text)
        print("\n" + text)
        save_stats(stats, paths[0])
        plot_jac_stencils(stats, paths[2], gamma_tau)
        plot_jac_by_run(stats, runs, paths[3], gamma_tau)
    elif a.part == "xai":
        basis = ja.load_basis(SIM / "v034" / "Q_d_b" / "normalization_constants.npz")
        w = setups["Q_a"]["whiten"]
        mb, sb = setups["Q_a"]["input_norm"]["b"]
        bbar_check = float(np.max(np.abs(mb + sb * w["whiten_xbar"] - basis["bbar"])))
        refs, pool_check = xa.channel_references(runs, basis["bbar"])
        rho = xa.rho_bar()
        info = {"rho_bar": rho, "bbar_check": bbar_check, "pool_check": pool_check,
                "s_ref_centre": float(refs["q_minus_qs"][4])}
        exact = exact_q_model(setups["Q_a"], gamma_tau)
        res = {}
        for l in ra_sorted(runs):
            run, t0 = runs[l], time.time()
            X = inputs_for(setups["Q_a"], run["fields"], run)
            for s in SEEDS:
                res[("Q_a", s, l)] = xai_one(setups["Q_a"]["models"][s], setups["Q_a"], X, run, refs, rho)
            res[("exact", 0, l)] = r = xai_one(exact, setups["Q_a"], X, run, refs, rho, exact=True)
            for key, tol in (("patch_check", xa.PATCH_TOL), ("dev_ch", DEV_TOL), ("dev_T", DEV_TOL)):
                if r[key] > tol:
                    sys.exit(f"exact Q {l}: {key} = {r[key]:.2e} > {tol}")
            for s in SEEDS:
                if res[("Q_a", s, l)]["patch_check"] > xa.PATCH_TOL:
                    sys.exit(f"Q_a {s} {l}: patch_check {res[('Q_a', s, l)]['patch_check']:.2e} > {xa.PATCH_TOL}")
            print(f"  xai {l:<15} Q_a 42: e0 {res[('Q_a', 42, l)]['e0'][0]:.3e}, S_ch {res[('Q_a', 42, l)]['S_ch'][0]}; "
                  f"exact dev {r['dev_ch']:.1e} / {r['dev_T']:.1e} ({time.time() - t0:.0f} s)", flush=True)
        text = xai_summary(res, runs, info)
        paths[1].write_text(text)
        print("\n" + text)
        save_res(res, paths[0])
        qd34 = load_npz_results(SIM / "v034" / "xai" / "xai_qd.npz", "v034", "Q_d_a", "S_SH")
        plot_channel_shares(res, runs, paths[2])
        plot_template_shares(res, qd34, runs, paths[3])
        plot_ablation(res, runs, paths[4])
    elif a.part == "eval":
        records = []
        for l in labels:
            records += eval_run(setups, l)
        st = setups["Q_a"]
        loss_check = {}
        for s in SEEDS:
            fixed = float(np.mean([r["rel_mse_all"] for r in records
                                   if r["model"] == "Q_a" and r["seed"] == s and r["set"] == "val"]))
            loss_check[s] = (fixed, st["histories"][s]["val_loss"][st["summary"][s]["best_epoch"] - 1])
        train_info = ja.EVS["v034"].training_summary(st)
        keys = list(records[0])
        with open(paths[0], "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=keys)
            wr.writeheader()
            wr.writerows(records)
        text = eval_summary(records, loss_check, train_info)
        paths[1].write_text(text)
        print("\n" + text)
        plot_eval(records, paths[2])
        plot_training(st, paths[3])
    else:
        res = {l: dns_run(setups, l) for l in T34.TEST_RUNS}
        text = dns_summary(res)
        paths[1].write_text(text)
        print("\n" + text)
        save_dns(res, paths[0])
        plot_dns(res, {"ts": paths[2], "prof": paths[3], "joint": paths[4], "snap": paths[5]})
    print(f"Written: {', '.join(p.name for p in paths)} ({(time.time() - t_start) / 60:.1f} min)")


if __name__ == "__main__":
    main()
