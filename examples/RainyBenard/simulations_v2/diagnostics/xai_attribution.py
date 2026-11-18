"""XAI attribution of the v033 and v034 stencil CNNs (agreed with the user, 2026-11-18; changelog.txt):
GradSHAP (expected gradients), integrated gradients (IG), exact Shapley values and ablation, on the cells
and frames of diagnostics/jacobian_attribution.py, whose functions are reused. Read-only on the models.

Notation. Cell c: an interior sampled cell (rows 0 and 1 mod 3 without the wall rows, columns 0 mod 3) of
the N_FRAMES = 8 validation-block frames per run of jacobian_attribution.py. x: its 3 x 3 input patch, all
channels; the receptive field is exactly 3 x 3, so the output at c depends on x only, and the patches of
the sampled cells do not overlap, so every edit below is made at all of them at once. Psi(x): the
explained output at c. Q_d models: Psi = dz^2 psi^ (psi^ = Q^_d / kappa_c for Q_d_a, the output for
Q_d_b; as jacobian_attribution.py). Q_c models: Psi = Q^_c. Psi_DNS: the same quantity from the DNS
(dz^2 Q_d / kappa = dz^2 Lap b, or Q_c).
Channel references (raw units). b: bbar, the mean patch of the v034 whitening (the IG baseline of
jacobian_attribution.py). q, q - q_s, z: the mean patch over the same cells (the training script's
whitening draws, WHITEN_CELLS interior cells per training run, seed WHITEN_SEED, repeated here; checked:
their mean b patch is bbar). log10 Ra: rho_bar, its mean over the training runs.
Reference pool of a run: the whole input patch at those cells of the run's own training block (test runs:
the same construction on their frames 25..750, seed WHITEN_SEED + 1).
Level 2, which operator inside b (Q_d configurations; log10 Ra kept at the cell's own value):
  a_k = u_k . (b - bbar), with u_k the templates of jacobian_attribution.py (columns of U).
  IG (Sundararajan, Taly & Yan 2017): IG_k = a_k g_k, g_k = u_k . int_0^1 grad_b Psi(bbar + alpha (b - bbar))
      dalpha (ja.ig_at_cells, M_STEPS midpoint steps); sum_k IG_k = Psi(x) - Psi(bbar).
  GradSHAP = expected gradients (Erion et al. 2021; Captum GradientShap without noise): IG with the
      reference x'_j drawn from the run's pool, averaged over K references per cell,
      phi_k = (1/K) sum_j u_k.(b - b'_j) u_k.int_0^1 grad_b Psi(x'_j + alpha (x - x'_j)) dalpha;
      the sum over every channel and patch cell is Psi(x) - (1/K) sum_j Psi(x'_j).
  Shapley values over the 9 templates (Shapley 1953): v(S) = Psi(bbar + sum_{k in S} a_k u_k),
      phi_k = sum_{S in N - k} |S|! (8 - |S|)! / 9! [v(S + k) - v(S)]; sum_k phi_k = Psi(x) - Psi(bbar).
  Ablation: e_-k = sqrt(sum_c (v(N - k) - Psi_DNS)^2 / sum_c Psi_DNS^2) over the run's explained cells;
      e_0 the same with v(N) = Psi(x).
Level 1, which input channel (every configuration): Shapley values over the input channels with v(S) =
  Psi(channels in S from x, the others at their reference patch); channel ablation e_-ch from v(N - ch).
Q_c configurations within the channels: IG (reference: every channel at its reference patch) and GradSHAP
  per channel and patch cell (cell basis: the exact Q_c is pointwise).
Shares: S = sum_c phi^2 / sum_c sum phi^2 over the run's explained cells (as jacobian_attribution.py).
Controls through the same code: the exact operator (Q_d_b: Lap b; Q_d_a: kappa(rho) Lap b, kappa =
  (Ra Pr)^(-1/2) with Pr = 1 and rho = log10 Ra from the input) and the exact Q_c = (gamma/tau) max(q - q_s, 0)
  for C1 (q_s = exp(alpha (b - beta z)) from the inputs) and C2, C4 (q - q_s is an input); C3 has none.
  Analytic values checked. Linear operator, Level 2: IG_k = Shapley_k = (L.u_k) a_k and GradSHAP_k =
  (L.u_k) (1/K) sum_j u_k.(b - b'_j). Q_d_a, Level 1: phi_b = (1 + r)/2 L.(b - bbar), phi_rho = (1 - r)/2
  L.(b + bbar), r = 10^((rho_c - rho_bar)/2). C2, C4: no credit to b or z.
Evaluation on patches: except the IG of the Q_d models (ja.ig_at_cells, full frames), every quantity is
  computed on the patches themselves. The model's layers without the padding (StencilPad), applied to an
  (n, 3, 3, C) input, give the output at the patch centre (receptive field exactly 3 x 3; the padding only
  acts at the walls and across the periodic edge). Checked for every model and run against the full-frame
  prediction (patch_check, max over cells) and the lattice gradient of jacobian_attribution.py on the first
  frames (grad_check, rms: a pre-activation within round-off of zero can flip a ReLU and with it one cell's
  gradient); the run stops if either exceeds PATCH_TOL.
Removal caveat: an ablated or path patch can be a combination absent from the training data (Hooker,
  Erhan, Kindermans & Kim 2019).
Output: simulations_v2/v034/xai/ (per part; refuses to overwrite): summary_<part>.txt, xai_<part>.npz and
  Qd_GradSHAP_template_shares.png, Qd_Shapley_template_shares.png, Qd_ablation_errors.png,
  Qd_a_channel_attribution.png (part qd); Qc_GradSHAP_maps.png, Qc_channel_shapley.png,
  Qc_channel_ablation.png (part qc).
Usage (from simulations_v2/, inside the Docker container):
  python diagnostics/xai_attribution.py qd|qc [--k K] [--hours H]
      Every model result is saved at once to tmp/xai/xai_<part>_checkpoint.pkl; after H hours (default 3.5;
      docker_run allows 4) the run stops with exit code 3 before the next run's models, and the same
      command resumes from the checkpoint. The outputs are written when every result is in.
  python diagnostics/xai_attribution.py pilot VERSION CONFIG RUN [--k K]   (first seed on one run: timing
      and the K/2 check; writes nothing)
"""
import argparse
import gc
import importlib.util
import itertools
import math
import os
import pickle
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
_spec = importlib.util.spec_from_file_location("jacobian_attribution", SIM / "diagnostics" / "jacobian_attribution.py")
ja = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ja)
T = ja.T                                  # v034 training script: run lists, loader, normalisation
U = ja.U                                  # columns: the templates u00 u10 u01 u20 u11 u02 u21 u12 u22
L_U = ja.L3.ravel() @ U                   # exact operator in the templates: (0, 0, 0, sqrt2, 0, sqrt2, 0, 0, -4)

VERSIONS = ja.VERSIONS
CONFIGS = {"qd": ("Q_d_a", "Q_d_b"), "qc": ("Q_c_C1", "Q_c_C2", "Q_c_C3", "Q_c_C4")}
PART_DATA = {"qd": (("b",), "Q_d_over_kappa"), "qc": (("b", "q", "q_minus_qs"), "Q_c")}
K_REF = 8                                 # GradSHAP references per cell (even: the K/2 check)
CHUNK = ja.CHUNK
BATCH = 1 << 17                           # patches per model call
PATCH_TOL = 1e-3                          # largest patch_check / grad_check accepted (layout errors give O(1))
REF_SEED = 2026
OUT_DIR = SIM / "v034" / "xai"
J2_DATA = SIM / "v034" / "jacobian_2" / "jacobian_data.npz"
CKPT_DIR = SIM.parents[2] / "tmp" / "xai"     # checkpoints (repository tmp/, git-ignored)
HAS_EXACT = ("Q_d_a", "Q_d_b", "Q_c_C1", "Q_c_C2", "Q_c_C4")
CH_TEX = {"b": r"$b$", "q": r"$q$", "q_minus_qs": r"$q-q_s$", "z": r"$z$", "log10_Ra": r"$\log_{10}\mathrm{Ra}$"}
CH_COL = {"b": "tab:green", "q": "tab:blue", "q_minus_qs": "tab:blue", "z": "tab:orange", "log10_Ra": "tab:purple"}
VCOL = {"v033": "tab:blue", "v034": "tab:red", "exact": "k"}


# ============================================================
# 1. Patches, Shapley coefficients, paths
# ============================================================

def interior_row_sets(nz, ny):
    row_sets, cols = ja.sample_cells(nz, ny)
    return [r[(r >= 1) & (r <= nz - 2)] for r in row_sets], cols


def cells(F, rows, cols):
    return F[:, rows][:, :, cols]


def patches_of(X, rows, cols):
    """X (B, NZ, NY, C) -> (B, nr, nc, C, 9) float64; index m = 3 (p + 1) + (q + 1)."""
    return np.stack([ja.gather_patches(X[..., ch], rows, cols).reshape(len(X), len(rows), len(cols), 9)
                     for ch in range(X.shape[3])], axis=3)


def patch_model(model):
    """The model on 3 x 3 patches: its layers without the padding (StencilPad) applied to an (n, 3, 3, C)
    input, whose output (n, 1, 1, 1) is the output at the patch centre. The exact controls (padding inside
    call) are used as they are; the centre of their (n, 3, 3, 1) output is read."""
    if isinstance(model, (ExactQd, ExactQc)):
        return model
    if type(model.layers[0]).__name__ != "InputLayer":
        raise ValueError(f"{model.name}: first layer {type(model.layers[0]).__name__}, expected InputLayer")
    x = inp = tf.keras.layers.Input((3, 3, model.inputs[0].shape[-1]))
    for layer in model.layers[1:]:
        if type(layer).__name__ != "StencilPad":
            x = layer(x)
    if tuple(x.shape[1:]) != (1, 1, 1):
        raise ValueError(f"{model.name}: patch model output {x.shape}, expected (None, 1, 1, 1)")
    return tf.keras.Model(inp, x)


_PATCH_MODELS = {}                         # id(model) -> (model, patch model): built once per loaded model


class Patches:
    """A model on flat patches P (n, C, 9), index m = 3 (p + 1) + (q + 1), in batches: the normalised
    prediction at the patch centre (n,) and its gradient with respect to the patch (n, C, 9)."""

    def __init__(self, model, batch=BATCH):
        if id(model) not in _PATCH_MODELS:
            _PATCH_MODELS[id(model)] = (model, patch_model(model))
        self.f, self.batch = _PATCH_MODELS[id(model)][1], batch

    @staticmethod
    def _image(p):
        return tf.constant(np.swapaxes(p, 1, 2).reshape(len(p), 3, 3, p.shape[1]).astype(np.float32))

    @staticmethod
    def _centre(y):
        return tf.cast(y[:, y.shape[1] // 2, y.shape[2] // 2, 0], tf.float64)

    def pred(self, P):
        return np.concatenate([self._centre(self.f(self._image(P[i:i + self.batch]), training=False)).numpy()
                               for i in range(0, len(P), self.batch)])

    def grad(self, P):
        out = []
        for i in range(0, len(P), self.batch):
            x = self._image(P[i:i + self.batch])
            with tf.GradientTape() as tape:
                tape.watch(x)
                s = tf.reduce_sum(self._centre(self.f(x, training=False)))
            g = tape.gradient(s, x).numpy().astype(np.float64)
            out.append(np.swapaxes(g.reshape(len(g), 9, -1), 1, 2))
        return np.concatenate(out)

    def path_grad(self, P, Pr, steps=ja.M_STEPS):
        """Midpoint-rule mean of the gradient over the straight path Pr + alpha (P - Pr), with the path
        points in float32 as in ja.ig_at_cells: (n, C, 9). Pr (n, C, 9) or (C, 9)."""
        P32 = P.astype(np.float32)
        R32 = np.broadcast_to(np.asarray(Pr, np.float32), P32.shape)
        D = P32 - R32
        g = np.zeros(P32.shape)
        for a in ((np.arange(steps) + 0.5) / steps).astype(np.float32):
            g += self.grad(R32 + a * D)
        return g / steps


def shapley_matrix(n):
    """Subsets S (2^n, n) bool and coefficients (2^n, n) with phi = coef.T @ v (Shapley 1953):
    phi_k = sum_{S in N - k} w(|S|) [v(S + k) - v(S)], w(s) = s! (n - 1 - s)! / n!, so v(S) enters
    phi_k with +w(|S| - 1) if k is in S and with -w(|S|) if not."""
    S = np.array(list(itertools.product((False, True), repeat=n)))
    s = S.sum(axis=1)
    w = np.array([math.factorial(m) * math.factorial(n - 1 - m) / math.factorial(n) for m in range(n)])
    return S, np.where(S, w[np.maximum(s - 1, 0)][:, None], -w[np.minimum(s, n - 1)][:, None])


# ============================================================
# 2. The attributions of one model on one run
# ============================================================

def gradshap(pf, Pn, pool_n, idx, osc, ib, sb, psi):
    """Expected gradients with references pool_n[idx[j]], j < K, for the cells' patches Pn (n, C, 9). Per
    cell: phi (cell basis) and phiT (templates of b), each also from the first K/2 references (_h);
    da = mean_j u.(b - b'_j) in b units; psi_ref = mean_j Psi(x'_j)."""
    K = len(idx)
    phi, phi_h = np.zeros(Pn.shape), np.zeros(Pn.shape)
    phiT, phiT_h, da = (np.zeros((len(Pn), 9)) for _ in range(3))
    psi_ref = np.zeros(len(Pn))
    for j in range(K):
        Pr = pool_n[idx[j]]
        G = pf.path_grad(Pn, Pr)
        D = Pn - Pr
        c = osc * D * G
        dT = D[..., ib, :] @ U
        cT = osc * dT * (G[..., ib, :] @ U)
        phi += c
        phiT += cT
        da += sb * dT
        if j < K // 2:
            phi_h += c
            phiT_h += cT
        psi_ref += psi(Pr)
    h = K // 2
    return {"phi": phi / K, "phi_h": phi_h / h, "phiT": phiT / K, "phiT_h": phiT_h / h, "da": da / K,
            "psi_ref": psi_ref / K}


def subset_values(S, coef, patch_of_mask, psi, px):
    """phi = sum_S coef(S) v(S), the values v(N - k) and v(empty), for v(S) = psi(patch_of_mask(S));
    v(N) = px."""
    n = S.shape[1]
    phi = np.zeros(px.shape + (n,))
    vabl = np.zeros(px.shape + (n,))
    v0 = None
    for m, mask in enumerate(S):
        v = px if mask.all() else psi(patch_of_mask(mask))
        phi += coef[m] * v[..., None]
        if mask.sum() == n - 1:
            vabl[..., np.flatnonzero(~mask)[0]] = v
        if not mask.any():
            v0 = v
    return phi, vabl, v0


def template_shapley(Pn, ib, xbar_n, psi, px):
    S, coef = shapley_matrix(9)
    a = (Pn[:, ib] - xbar_n) @ U
    P = Pn.copy()

    def patch(mask):
        P[:, ib] = xbar_n + (a * mask) @ U.T
        return P
    return subset_values(S, coef, patch, psi, px)


def channel_shapley(Pn, ref_n, psi, px):
    S, coef = shapley_matrix(Pn.shape[1])
    return subset_values(S, coef, lambda mask: np.where(mask[:, None], Pn, ref_n), psi, px)


def rel_rms(resid, ref):
    return math.sqrt(float(np.mean(np.square(resid)) / np.mean(np.square(ref))))


def rel_max(resid, ref):
    m = float(np.max(np.abs(resid)))
    return 0.0 if m == 0.0 else m / math.sqrt(float(np.mean(np.square(ref))))


def analyse(model, X, norms, run, ctx, chunk=CHUNK):
    """Every attribution of one model on one run's explained frames. X (B, NZ, NY, C) normalised inputs;
    norms: input_norm, target_norm, fac (Psi = fac x prediction in physical units); run: psi_dns
    (B, NZ, NY); ctx: inputs, ref_n (C, 9) and pool_n (N, C, 9) normalised, ref_idx (one (K, B, nr, nc)
    array per row set), level2 ("templates" or "cells"), bbar, exact (analytic checks of a control)."""
    inputs, ref_n, pool_n = ctx["inputs"], ctx["ref_n"], ctx["pool_n"]
    C, ib = len(inputs), inputs.index("b")
    (my, sy), fac = norms["target"], norms["fac"]
    osc = fac * sy                                        # Psi per unit of normalised prediction
    mb, sb = norms["input"]["b"]
    templ = ctx["level2"] == "templates"
    row_sets, cols = interior_row_sets(X.shape[1], X.shape[2])
    sec = dict.fromkeys(("gradshap", "ig", "shapley", "channels"), 0.0)
    acc = {}

    def add(key, v):
        acc.setdefault(key, []).append(v)

    pf = Patches(model)

    def psi(P):
        return osc * pf.pred(P) + fac * my

    psi_frame = osc * ja.predict(model, X, chunk) + fac * my     # full frames: the patch_check
    for rs, rows in enumerate(row_sets):
        Pn = patches_of(X, rows, cols).reshape(-1, C, 9)          # cells in (frame, row, column) order
        px = psi(Pn)
        add("px", px)
        add("pf", cells(psi_frame, rows, cols).reshape(-1))
        add("dns", cells(run["psi_dns"], rows, cols).reshape(-1))
        add("b_n", Pn[:, ib])
        if rs == 0:                                               # gradient on patches vs the lattice gradient
            nf = min(chunk, len(X))
            lat = patches_of(ja.cell_grad(model, X[:nf], rows, cols), rows, cols).reshape(-1, C, 9)
            grad_check = rel_rms(pf.grad(Pn[:len(lat)]) - lat, lat)     # rms: robust to rare ReLU-kink flips
        t0 = time.time()
        idx = ctx["ref_idx"][rs]
        for k, v in gradshap(pf, Pn, pool_n, idx.reshape(len(idx), -1), osc, ib, sb, psi).items():
            add("g_" + k, v)
        t1 = time.time()
        for k, v in zip(("phi", "abl", "v0"), channel_shapley(Pn, ref_n, psi, px)):
            add("ch_" + k, v)
        t2 = time.time()
        sec["gradshap"] += t1 - t0
        sec["channels"] += t2 - t1
        if templ:
            for k, v in zip(("phi", "abl", "v0"), template_shapley(Pn, ib, ref_n[ib], psi, px)):
                add("sh_" + k, v)
            sec["shapley"] += time.time() - t2
        else:
            add("ig_phi", osc * (Pn - ref_n) * pf.path_grad(Pn, ref_n))
            add("ig_v0", psi(np.broadcast_to(ref_n, Pn.shape)))
            sec["ig"] += time.time() - t2
    c = {k: np.concatenate(v) for k, v in acc.items()}
    px, dns = c["px"], c["dns"]

    def err(v):
        return math.sqrt(float(np.sum((v - dns) ** 2) / np.sum(dns ** 2)))

    d_g, d_ch = px - c["g_psi_ref"], px - c["ch_v0"]
    n = len(px)
    out = {"n_cells": n, "e0": err(px), "e_ch": np.array([err(c["ch_abl"][:, k]) for k in range(C)]),
           "S_ch": ja.shares(c["ch_phi"]), "comp_G": rel_rms(c["g_phi"].sum(axis=(1, 2)) - d_g, d_g),
           "eff_ch": rel_max(c["ch_phi"].sum(axis=1) - d_ch, d_ch), "patch_check": rel_max(px - c["pf"], px),
           "grad_check": grad_check, "seconds": sec}
    if templ:
        t0 = time.time()
        ig = ja.ig_at_cells(model, X, ib, osc / sb, sb, ref_n[ib], chunk=chunk)
        sec["ig"] += time.time() - t0
        a, igT, _ = ja.pattern_components(ig["dB"], ig["Jbar"], U)
        d_sh = px - c["sh_v0"]
        out.update(S_IG=ja.shares(igT), S_G=ja.shares(c["g_phiT"]), S_G_half=ja.shares(c["g_phiT_h"]),
                   S_SH=ja.shares(c["sh_phi"]), e_abl=np.array([err(c["sh_abl"][:, k]) for k in range(9)]),
                   comp_IG=rel_rms((ig["dB"] * ig["Jbar"]).sum(axis=1) - ig["dPsi"], ig["dPsi"]),
                   eff_SH=rel_max(c["sh_phi"].sum(axis=1) - d_sh, d_sh))
    else:
        flat = lambda v: v.reshape(n, -1)
        out.update(S_G=ja.shares(flat(c["g_phi"])).reshape(C, 9), S_G_half=ja.shares(flat(c["g_phi_h"])).reshape(C, 9),
                   S_IG=ja.shares(flat(c["ig_phi"])).reshape(C, 9),
                   comp_IG=rel_rms(c["ig_phi"].sum(axis=(1, 2)) - (px - c["ig_v0"]), px - c["ig_v0"]))
    out["G_half_dev"] = float(np.max(np.abs(out["S_G"] - out["S_G_half"])))
    ex = ctx.get("exact") or {}
    if ex.get("linear_b"):                                # Level 2 of the linear operator: analytic credits
        out.update(dev_IG=rel_max(igT - L_U * a, ig["dPsi"]), dev_SH=rel_max(c["sh_phi"] - L_U * a, d_sh),
                   dev_G=rel_max(c["g_phiT"] - L_U * c["g_da"], d_g))
    if "r" in ex:                                         # Q_d_a, Level 1: phi_b, phi_rho of the exact operator
        r, L9 = ex["r"], ja.L3.ravel()
        Lb, Lbbar = (mb + sb * c["b_n"]) @ L9, float(ctx["bbar"] @ L9)
        an = np.zeros((n, C))
        an[:, ib] = 0.5 * (1.0 + r) * (Lb - Lbbar)
        an[:, inputs.index("log10_Ra")] = 0.5 * (1.0 - r) * (Lb + Lbbar)
        out["dev_ch"] = rel_max(c["ch_phi"] - an, d_ch)
    if ex.get("s_only"):                                  # C2, C4: only q - q_s carries credit
        out["dev_ch"] = rel_max(c["ch_phi"][:, [k for k, ch in enumerate(inputs) if ch != "q_minus_qs"]], d_ch)
    return out


# ============================================================
# 3. Exact controls (normalised units of a configuration)
# ============================================================

class ExactQd(tf.keras.Model):
    """The exact operator: training padding (StencilPad), Lap b = (sigma_b / dz^2) L * b_n in float64, times
    kappa(rho) = 10^(-rho/2) (Pr = 1) for target Q_d (Q_d_a), rho = log10 Ra from the input channel;
    output (y - mu_y) / sigma_y."""

    def __init__(self, t, setup):
        super().__init__()
        t.select_config(setup["name"])
        inputs = list(t.INPUTS)
        self.pad = t.StencilPad(**t.padding_args(setup["input_norm"]), dtype="float64")
        self.ib = inputs.index("b")
        self.ir = inputs.index("log10_Ra") if "log10_Ra" in inputs else None
        self.rho = setup["input_norm"].get("log10_Ra", (0.0, 1.0))
        self.sb = setup["input_norm"]["b"][1]
        self.my, self.sy = setup["target_norm"]
        self.dz2 = (1.0 / t.NZ) ** 2
        self.kappa_in_output = t.CONFIGS[setup["name"]]["target"] == "Q_d"
        self.kernel = tf.constant(ja.L3[:, :, None, None], tf.float64)

    def call(self, x, training=False):
        x = tf.cast(x, tf.float64)
        y = tf.nn.conv2d(self.pad(x)[..., self.ib:self.ib + 1], self.kernel, 1, "VALID") * (self.sb / self.dz2)
        if self.kappa_in_output:
            rho = self.rho[0] + self.rho[1] * x[..., self.ir:self.ir + 1]
            y = y * tf.pow(tf.constant(10.0, tf.float64), -0.5 * rho)
        return tf.cast((y - self.my) / self.sy, tf.float32)


class ExactQc(tf.keras.Model):
    """Exact Q_c = (gamma/tau) max(s, 0), pointwise, in float64: s = q - q_s from the input q_minus_qs (C2,
    C4), or s = q - exp(alpha (b - beta z)) from b, q, z (C1); output (Q_c - mu_y) / sigma_y."""

    def __init__(self, inputs, input_norm, target_norm, gamma_tau, alpha, beta):
        super().__init__()
        self.idx = {c: i for i, c in enumerate(inputs)}
        if "q_minus_qs" not in self.idx and not {"b", "q", "z"} <= set(self.idx):
            raise ValueError(f"no exact Q_c from the inputs {inputs}")
        self.norm = {c: tuple(input_norm[c]) for c in inputs}
        self.my, self.sy = target_norm
        self.gt, self.alpha, self.beta = float(gamma_tau), float(alpha), float(beta)

    def raw(self, x, c):
        m, s = self.norm[c]
        return m + s * x[..., self.idx[c]:self.idx[c] + 1]

    def call(self, x, training=False):
        x = tf.cast(x, tf.float64)
        if "q_minus_qs" in self.idx:
            sat = self.raw(x, "q_minus_qs")
        else:
            sat = self.raw(x, "q") - tf.exp(self.alpha * (self.raw(x, "b") - self.beta * self.raw(x, "z")))
        return tf.cast((self.gt * tf.maximum(sat, 0.0) - self.my) / self.sy, tf.float32)


def exact_model(t, setup, gamma_tau):
    n = setup["name"]
    if n in CONFIGS["qd"]:
        return ExactQd(t, setup)
    t.select_config(n)
    if "q_minus_qs" in t.INPUTS or {"q", "z"} <= set(t.INPUTS):
        return ExactQc(list(t.INPUTS), setup["input_norm"], setup["target_norm"], gamma_tau, t.ALPHA, t.BETA)
    return None


def exact_info(name, ra, rho_bar):
    if name == "Q_d_b":
        return {"linear_b": True}
    if name == "Q_d_a":
        return {"linear_b": True, "r": 10.0 ** ((math.log10(ra) - rho_bar) / 2.0)}
    if name in ("Q_c_C2", "Q_c_C4"):
        return {"s_only": True}
    return {}


# ============================================================
# 4. Data
# ============================================================

def set_inputs(t, inputs, target):
    """As t.select_config, for an input set of no training configuration: the loader reads exactly these."""
    t.INPUTS, t.TARGET = tuple(inputs), target


def psi_factor(target, nz, kappa):
    """Psi = factor x (prediction or DNS value in physical units)."""
    dz2 = (1.0 / nz) ** 2
    return {"Q_d": dz2 / kappa, "Q_d_over_kappa": dz2, "Q_c": 1.0}[target]


def pool_draws(rng, n_train, nz, ny, n):
    """Cells (training-block frame, row, column), drawn as fit_whitening of the training script draws them."""
    return rng.integers(0, n_train, n), rng.integers(1, nz - 1, n), rng.integers(0, ny, n)


def pool_patches(F, ti, ii, jj):
    """F (n_frames, NZ, NY, C) -> (n, C, 9) patches at the cells (ti, ii, jj)."""
    ny = F.shape[2]
    return np.stack([F[ti, ii + m // 3 - 1, (jj + m % 3 - 1) % ny] for m in range(9)], axis=-1)


def load_part_runs(part, labels):
    """Per run in `labels`: the raw channels of the part and Psi_DNS on the explained frames, and the
    reference pool. Every run's draws are made (in TRAIN_RUNS + TEST_RUNS order) whether loaded or not, so
    the training-run pools are the whitening cells for any subset of runs."""
    chans, target = PART_DATA[part]
    set_inputs(T, chans, target)
    n_train = len(T.frame_indices(T.TRAIN_FRAMES))
    rngs = {"train": np.random.default_rng(T.WHITEN_SEED), "test": np.random.default_rng(T.WHITEN_SEED + 1)}
    runs = {}
    for label in T.TRAIN_RUNS + T.TEST_RUNS:
        ti, ii, jj = pool_draws(rngs["train" if label in T.TRAIN_RUNS else "test"], n_train, T.NZ, T.NY,
                                T.WHITEN_CELLS)
        if label not in labels:
            continue
        run = T.load_run(label)
        tr, val = run["parts"]["train"], run["parts"]["val"]
        if tr.stop - tr.start != n_train or run["pr"] != 1.0:
            raise ValueError(f"{label}: {tr.stop - tr.start} training frames (expected {n_train}), Pr {run['pr']} "
                             f"(kappa(rho) = 10^(-rho/2) assumes Pr = 1)")
        n = val.stop - val.start
        sel = np.unique(np.linspace(0, n - 1, min(ja.N_FRAMES, n)).round().astype(int))
        h = T.pp.read_humid(T.run_paths(label)["humid"])
        runs[label] = {"label": label, "chans": list(chans), "fields": np.ascontiguousarray(run["fields"][val][sel]),
                       "psi_dns": psi_factor(target, T.NZ, run["kappa"]) * run["y"][val][sel].astype(np.float64),
                       "frames": [run["frames"]["val"][i] for i in sel], "xm": run["xm"], "params": run["params"],
                       "ra": run["ra"], "kappa": run["kappa"], "gamma_tau": h["gamma"] / h["tau"],
                       "pool": pool_patches(run["fields"][tr], ti, ii, jj), "pool_rows": ii}
        del run
        gc.collect()
    return runs


def channel_references(runs, bbar):
    """Mean patch (raw units) of every channel and of z over the training-run pools (all loaded runs if no
    training run is loaded), b replaced by bbar; and max |mean b patch - bbar|."""
    tr = [runs[l] for l in T.TRAIN_RUNS if l in runs] or list(runs.values())
    n = sum(len(r["pool"]) for r in tr)
    mean = sum(r["pool"].astype(np.float64).sum(axis=0) for r in tr) / n
    refs = {c: mean[i] for i, c in enumerate(tr[0]["chans"])}
    rows = np.concatenate([r["pool_rows"] for r in tr])
    refs["z"] = np.array([tr[0]["xm"][rows + m // 3 - 1].mean() for m in range(9)])
    check = float(np.max(np.abs(refs["b"] - bbar)))
    refs["b"] = np.asarray(bbar, np.float64)
    return refs, check


def rho_bar():
    return float(np.mean([math.log10(T.pp.read_bou(T.run_paths(l)["bou"])["ra"]) for l in T.TRAIN_RUNS]))


def reference_indices(run_no, n_pool, n_frames, k, nz, ny):
    """Pool indices of the K GradSHAP references of every explained cell, one (K, B, nr, nc) array per row
    set; the same for every model."""
    rng = np.random.default_rng([REF_SEED, run_no])
    row_sets, cols = interior_row_sets(nz, ny)
    return [rng.integers(0, n_pool, (k, n_frames, len(r), len(cols))) for r in row_sets]


def model_inputs(t, setup, run):
    """Normalised inputs (n, NZ, NY, C) of the explained frames for the configuration of `setup`, built by
    its training script (apply_input_norm, RunSnapshots), as ja.model_inputs."""
    t.select_config(setup["name"])
    idx = [run["chans"].index(c) for c in t.field_channels()]
    rd = {"label": run["label"], "fields": run["fields"][..., idx].copy(),
          "y": np.zeros(run["fields"].shape[:3], np.float32), "xm": run["xm"], "params": run["params"],
          "parts": {"val": slice(0, len(run["fields"]))}}
    t.apply_input_norm(rd, setup["input_norm"])
    return t.RunSnapshots([rd], "val", setup["target_norm"], len(rd["fields"]), shuffle=False, seed=0)[0][0]


def context(t, setup, run, refs, rho, ref_idx):
    """Normalised channel references and pool of a configuration on a run (as apply_input_norm)."""
    t.select_config(setup["name"])
    inputs, norm = list(t.INPUTS), setup["input_norm"]
    N = len(run["pool"])
    ref_n, pool_n = [], []
    for c in inputs:
        m, s = norm[c]
        if c == "log10_Ra":
            ref_n.append(np.full(9, (rho - m) / s))
            pool_n.append(np.full((N, 9), np.float32((run["params"][c] - m) / s), np.float32))
        elif c == "z":
            ref_n.append((refs["z"] - m) / s)
            z = run["xm"][run["pool_rows"][:, None] + (np.arange(9) // 3 - 1)[None, :]]
            pool_n.append(((z - m) / s).astype(np.float32))
        else:
            ref_n.append((refs[c] - m) / s)
            pool_n.append((run["pool"][:, run["chans"].index(c)] - np.float32(m)) / np.float32(s))
    return {"inputs": inputs, "ref_n": np.stack(ref_n), "pool_n": np.stack(pool_n, axis=1), "ref_idx": ref_idx,
            "level2": "templates" if setup["name"] in CONFIGS["qd"] else "cells", "bbar": refs["b"]}


def norms_of(t, setup, run):
    return {"input": setup["input_norm"], "target": setup["target_norm"],
            "fac": psi_factor(t.CONFIGS[setup["name"]]["target"], t.NZ, run["kappa"])}


# ============================================================
# 5. Summary and data file
# ============================================================

def seeds_of(res, v, n):
    return sorted({k[2] for k in res if k[0] == v and k[1] == n})


def fmt(v, f="8.1e"):
    return " ".join(f"{x:{f}}" for x in np.ravel(v))


def summary_text(part, res, control, runs, info):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    tl = " ".join(f"u{a}{b}" for a, b in ja.TEMPLATE_ORDER)
    worst = lambda key: max(r[key] for r in list(res.values()) + list(control.values()) if key in r)
    L = [f"XAI attribution, part {part} ({', '.join(CONFIGS[part])}): v033, v034 (every seed) and exact controls.",
         "Definitions: docstring of diagnostics/xai_attribution.py. Psi = dz^2 psi^ (Q_d models) or Q^_c (Q_c models),",
         "explained at the interior sampled cells of the frames below; e = sqrt(sum_c (Psi - Psi_DNS)^2 / sum_c Psi_DNS^2)",
         "over a run's cells (e = 1: prediction 0); shares S = sum_c phi^2 / sum_c sum phi^2.",
         f"K = {info['K']} GradSHAP references per cell (pool of the same run), M = {ja.M_STEPS} midpoint steps per path;",
         f"{next(iter(res.values()))['n_cells']} explained cells per run.",
         "Frames per run: " + "; ".join(f"{l}: {runs[l]['frames']}" for l in labels), "", "Checks",
         f"  pool cells = whitening cells: max |mean b patch of the training-run pools - bbar| = {info['pool_check']:.1e} (b units)",
         f"  rho_bar = {info['rho_bar']:.6f} (mean log10 Ra of the training runs)",
         f"  evaluation on patches vs full frames: max |Psi_patch(x) - Psi_frame(x)| / rms(Psi) = {worst('patch_check'):.1e}; "
         f"rms(gradient - lattice gradient) / rms(lattice gradient), first frames, {worst('grad_check'):.1e}",
         f"  completeness, rms(sum of credits - output difference) / rms(output difference): IG max {worst('comp_IG'):.1e}, "
         f"GradSHAP max {worst('comp_G'):.1e}",
         f"  Shapley efficiency (exact up to round-off), max |sum - difference| / rms(difference): channels "
         f"{worst('eff_ch'):.1e}" + (f", templates {worst('eff_SH'):.1e}" if part == "qd" else ""),
         f"  GradSHAP with K/2 vs K references, max over shares |S(K/2) - S(K)|: median "
         f"{np.median([r['G_half_dev'] for r in res.values()]):.1e}, max {worst('G_half_dev'):.1e}"]
    if part == "qd":
        L.append(f"  IG vs v034/jacobian_2 (same models, frames and baseline): max |S_IG - S_T| = {info['ig_regression']:.1e}")
    L += ["", "Controls (exact operator / exact Q_c through the same code). e_0: error against the DNS (float32-input",
          "floor). dev: max |credit - analytic credit| / rms(output difference) (docstring)."]
    for (n, l), r in sorted(control.items(), key=lambda kv: (kv[0][0], runs[kv[0][1]]["ra"])):
        devs = " ".join(f"{k} {r[k]:.1e}" for k in ("dev_IG", "dev_SH", "dev_G", "dev_ch") if k in r)
        L.append(f"  {n:<7} {l:<15} e_0 {r['e0']:.2e}  {devs}")
    for v in VERSIONS + ("exact",):
        for n in CONFIGS[part]:
            seeds = [0] if v == "exact" else seeds_of(res, v, n)
            src = (lambda s, l: control.get((n, l))) if v == "exact" else (lambda s, l: res.get((v, n, s, l)))
            if not any(src(s, l) for s in seeds for l in labels):
                continue
            chans = T.CONFIGS[n]["inputs"]
            for s in seeds:
                L += ["", f"=== {v} {n}" + ("" if v == "exact" else f" seed {s}") + f"   channels {', '.join(chans)}"]
                rows = [(l, src(s, l)) for l in labels if src(s, l)]
                L.append(f"  {'run':<15} e_0 | e_-ch (channels) | channel shares S_ch")
                for l, r in rows:
                    L.append(f"  {l:<15} {r['e0']:8.2e} | {fmt(r['e_ch'], '8.2e')} | {fmt(r['S_ch'])}")
                if part == "qd":
                    for key, what in (("e_abl", "e_-k, k = " + tl), ("S_IG", "IG shares"), ("S_G", "GradSHAP shares"),
                                      ("S_SH", "Shapley shares")):
                        L.append(f"  {what}")
                        for l, r in rows:
                            L.append(f"  {l:<15} {fmt(r[key], '8.2e' if key == 'e_abl' else '8.1e')}")
                else:
                    L.append("  GradSHAP | IG shares per channel: centre cell, sum of the 8 neighbours")
                    for l, r in rows:
                        cell = lambda S: " ".join(f"{S[k, 4]:7.1e} {S[k].sum() - S[k, 4]:7.1e}" for k in range(len(chans)))
                        L.append(f"  {l:<15} {cell(r['S_G'])} | {cell(r['S_IG'])}")
    return "\n".join(L) + "\n"


def save_data(res, control, path):
    d = {"U": U, "template_order": np.array(ja.TEMPLATE_ORDER)}
    items = list(res.items()) + [(("exact", n, 0, l), r) for (n, l), r in control.items()]
    for key, r in items:
        tag = "__".join(str(x) for x in key)
        for k, v in r.items():
            if k != "seconds":
                d[f"{tag}__{k}"] = np.asarray(v)
    np.savez_compressed(path, **d)


# ============================================================
# 6. Figures (formulas, few words)
# ============================================================

GRADSHAP_TEX = (r"$S_k=\frac{\sum_c\phi_k^2}{\sum_c\sum_j\phi_j^2},\quad\phi_k=\frac{1}{K}\sum_{j=1}^{K}"
                r"\mathbf{u}_k\cdot(\mathbf{x}-\mathbf{x}'_j)\;\mathbf{u}_k\cdot\int_0^1\nabla\Psi\left(\mathbf{x}'_j"
                r"+\alpha(\mathbf{x}-\mathbf{x}'_j)\right)d\alpha,\quad\mathbf{x}'_j$: random patches of the same run")
SHAPLEY_TEX = (r"$S_k=\frac{\sum_c\phi_k^2}{\sum_c\sum_j\phi_j^2},\quad\phi_k=\sum_{S\subseteq N\setminus\{k\}}"
               r"\frac{|S|!\,(8-|S|)!}{9!}\left[v(S\cup\{k\})-v(S)\right],\quad v(S)=\Psi\left(\bar{\mathbf{x}}"
               r"+\sum_{j\in S}a_j\mathbf{u}_j\right)$")
ABLATION_TEX = (r"$e_{-k}=\sqrt{\frac{\sum_c\left(\Psi(\mathbf{x}-a_k\mathbf{u}_k)-\Psi_{\mathrm{DNS}}\right)^2}"
                r"{\sum_c\Psi_{\mathrm{DNS}}^2}},\quad a_k=\mathbf{u}_k\cdot(\mathbf{x}-\bar{\mathbf{x}}),"
                r"\quad e_{\emptyset}$: nothing removed")


def run_axis(ax, labels, runs):
    xs = np.arange(len(labels))
    for x, l in zip(xs, labels):
        if l in T.TEST_RUNS:
            ax.axvspan(x - 0.45, x + 0.45, color="0.92", lw=0)
    ax.set_xticks(xs, [ja.ra_tex(l, runs[l]["ra"]).replace(r"\mathrm{Ra}=", "") for l in labels], rotation=45, fontsize=9)
    ax.set_xlabel(r"$\mathrm{Ra}$", fontsize=12)
    ax.grid(True, which="major", ls=":", alpha=0.5)
    return xs


def plot_template_shares(res, control, runs, key, title, path):
    view = {k: {key: r[key], key + "_ex": control[(k[1], k[3])][key]} for k, r in res.items()}
    ja.plot_shares(view, runs, ja.TEMPLATE_LABELS, path, key=key, title=title)


def plot_ablation(res, control, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    x = np.arange(10)
    ticks = [r"$\emptyset$"] + [ja.template_tex(a, b) for a, b, _ in ja.TEMPLATE_LABELS]
    ncol = 5
    nrow = math.ceil(len(labels) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.8 * ncol, 4.0 * nrow + 1.4), sharey=True, squeeze=False,
                             layout="constrained")
    for ax, l in zip(axes.ravel(), labels):
        c = control[("Q_d_b", l)]
        ax.plot(x, np.r_[c["e0"], c["e_abl"]], color="k", lw=2.4, ls="--", marker="D", ms=5, zorder=5)
        for v in VERSIONS:
            for n in CONFIGS["qd"]:
                e = np.mean([np.r_[res[(v, n, s, l)]["e0"], res[(v, n, s, l)]["e_abl"]] for s in seeds_of(res, v, n)], axis=0)
                ax.plot(x, e, color=ja.COL[v], ls=ja.LS[n], marker=ja.MK[n], ms=4.5, lw=1.4)
        ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
        ax.set_yscale("log")
        ax.set_xticks(x, ticks, rotation=90, fontsize=11)
        ax.set_title(ja.ra_tex(l, runs[l]["ra"]), fontsize=11)
        ax.grid(True, which="major", ls=":", alpha=0.5)
    for ax in axes.ravel()[len(labels):]:
        ax.set_visible(False)
    for r in range(nrow):
        axes[r][0].set_ylabel(r"$e_{-k}$", fontsize=12)
    handles = [Line2D([], [], color="k", lw=2.4, ls="--", marker="D", label=r"$\Delta z^2\nabla_h^2$ (exact)")]
    handles += [Line2D([], [], color=ja.COL[v], ls=ja.LS[n], marker=ja.MK[n], label=f"{v}: " + ja.MODEL_TEX[n])
                for v in VERSIONS for n in CONFIGS["qd"]]
    handles.append(Line2D([], [], color="0.5", ls=":", label=r"$e=1$: prediction $\Psi=0$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=10)
    fig.suptitle(ABLATION_TEX + "  (mean over seeds)", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qd_a_channels(res, control, runs, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    ir = T.CONFIGS["Q_d_a"]["inputs"].index("log10_Ra")
    ib = T.CONFIGS["Q_d_a"]["inputs"].index("b")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), layout="constrained")
    xs = run_axis(a1, labels, runs)
    run_axis(a2, labels, runs)
    for v, o in (("v033", -0.12), ("v034", 0.12)):
        for s in seeds_of(res, v, "Q_d_a"):
            a1.plot(xs + o, [res[(v, "Q_d_a", s, l)]["S_ch"][ir] for l in labels], "o", color=VCOL[v], ms=5)
    a1.plot(xs, [control[("Q_d_a", l)]["S_ch"][ir] for l in labels], color="k", ls="--", marker="D", ms=5)
    a1.set_yscale("log")
    a1.set_ylabel(r"$S_\rho=\frac{\sum_c\phi_\rho^2}{\sum_c(\phi_b^2+\phi_\rho^2)}$", fontsize=13)
    for v in VERSIONS + ("exact",):
        get = (lambda l: [control[("Q_d_a", l)]]) if v == "exact" else \
            (lambda l, v=v: [res[(v, "Q_d_a", s, l)] for s in seeds_of(res, v, "Q_d_a")])
        for key, ls in ((None, "-"), (ib, "--"), (ir, ":")):
            e = [np.mean([r["e0"] if key is None else r["e_ch"][key] for r in get(l)]) for l in labels]
            a2.plot(xs, e, color=VCOL[v], ls=ls, marker="o", ms=3.5, lw=1.4)
    a2.axhline(1.0, color="0.5", ls="-.", lw=1.0)
    a2.set_yscale("log")
    a2.set_ylabel(r"$e$", fontsize=13)
    handles = [Line2D([], [], color=VCOL[v], marker="o", ls="", label=f"{v}: " + ja.MODEL_TEX["Q_d_a"]) for v in VERSIONS]
    handles += [Line2D([], [], color="k", ls="--", marker="D", label=r"exact: $\Delta z^2\kappa(\rho)\nabla^2 b/\kappa_c$"),
                Line2D([], [], color="0.3", ls="-", label=r"$e_\emptyset$ (nothing removed)"),
                Line2D([], [], color="0.3", ls="--", label=r"$e_{-b}$ ($b$ patch $\to\bar{\mathbf{x}}$)"),
                Line2D([], [], color="0.3", ls=":", label=r"$e_{-\rho}$ ($\rho\to\bar\rho$)"),
                Line2D([], [], color="0.5", ls="-.", label=r"$e=1$: prediction $\Psi=0$"),
                Patch(color="0.92", label=r"$\dagger\ \mathrm{test}$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=10)
    fig.suptitle(r"$\phi_b=\frac{1}{2}\left[v(b)-v(\emptyset)\right]+\frac{1}{2}\left[v(b,\rho)-v(\rho)\right],"
                 r"\ \phi_\rho=\frac{1}{2}\left[v(\rho)-v(\emptyset)\right]+\frac{1}{2}\left[v(b,\rho)-v(b)\right],"
                 r"\ \rho=\log_{10}\mathrm{Ra}$;  $v(S)=\Psi$(channels in $S$ from $\mathbf{x}$, the others at "
                 r"$\bar{\mathbf{x}},\bar\rho$)", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def qc_entries(res, control, v, n):
    if v == "exact":
        return [r for (nn, l), r in control.items() if nn == n]
    return [r for (vv, nn, s, l), r in res.items() if vv == v and nn == n]


def plot_qc_maps(res, control, path):
    rows, cols = VERSIONS + ("exact",), CONFIGS["qc"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.6 * len(cols), 2.6 * len(rows) + 1.2),
                             squeeze=False, layout="constrained")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.85")
    im = None
    for i, v in enumerate(rows):
        for j, n in enumerate(cols):
            ax = axes[i][j]
            ent = qc_entries(res, control, v, n)
            if not ent:
                ax.axis("off")
                ax.text(0.5, 0.5, r"no exact $Q_c$ ($z$ not an input)", ha="center", va="center", fontsize=10)
                continue
            chans = T.CONFIGS[n]["inputs"]
            S = np.mean([r["S_G"] for r in ent], axis=0)
            img = np.full((3, 4 * len(chans) - 1), np.nan)
            for k in range(len(chans)):
                img[:, 4 * k:4 * k + 3] = S[k].reshape(3, 3)
            im = ax.imshow(np.where(img > 0, img, np.nan), origin="lower", cmap=cmap, norm=LogNorm(1e-6, 1.0))
            for k in range(len(chans)):
                for p in range(3):
                    for q in range(3):
                        if S[k, 3 * p + q] >= 0.01:
                            ax.text(4 * k + q, p, f"{S[k, 3 * p + q]:.2f}", ha="center", va="center", fontsize=7,
                                    color="w" if S[k, 3 * p + q] < 0.3 else "k")
            ax.set_xticks([4 * k + 1 for k in range(len(chans))], [CH_TEX[c] for c in chans], fontsize=11)
            ax.set_yticks([0, 1, 2], ["$z-1$", "$z$", "$z+1$"], fontsize=8)
            ax.set_title(f"{v}: {n}", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label=r"$S_{ch,pq}$ (mean over runs and seeds; grey: 0)")
    fig.suptitle(r"$S_{ch,pq}=\frac{\sum_c\phi_{ch,pq}^2}{\sum_c\sum\phi^2},\ \phi=\frac{1}{K}\sum_j(\mathbf{x}-\mathbf{x}'_j)"
                 r"\odot\int_0^1\nabla\Psi\left(\mathbf{x}'_j+\alpha(\mathbf{x}-\mathbf{x}'_j)\right)d\alpha,\ \Psi=\hat Q_c$; "
                 r"each block: the $3\times3$ patch of one channel", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_channels(res, control, runs, kind, path):
    labels = sorted(runs, key=lambda l: runs[l]["ra"])
    rows, cols = VERSIONS + ("exact",), CONFIGS["qc"]
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(4.6 * len(cols), 3.4 * len(rows) + 1.2), sharey=True,
                             squeeze=False, layout="constrained")
    used = set()
    for i, v in enumerate(rows):
        for j, n in enumerate(cols):
            ax = axes[i][j]
            chans = T.CONFIGS[n]["inputs"]
            if v == "exact":
                get = lambda l: [control[(n, l)]] if (n, l) in control else []
            else:
                get = lambda l, v=v: [res[(v, n, s, l)] for s in seeds_of(res, v, n) if (v, n, s, l) in res]
            if not any(get(l) for l in labels):
                ax.axis("off")
                ax.text(0.5, 0.5, r"no exact $Q_c$ ($z$ not an input)", ha="center", va="center", fontsize=10)
                continue
            xs = run_axis(ax, labels, runs)
            if kind == "ablation":
                ax.plot(xs, [np.mean([r["e0"] for r in get(l)]) for l in labels], color="k", marker="o", ms=3.5, lw=1.4)
                ax.axhline(1.0, color="0.5", ls="-.", lw=1.0)
            for k, c in enumerate(chans):
                key = "S_ch" if kind == "shapley" else "e_ch"
                ax.plot(xs, [np.mean([r[key][k] for r in get(l)]) for l in labels], color=CH_COL[c], ls="--",
                        marker="o", ms=3.5, lw=1.4)
                used.add(c)
            ax.set_yscale("log")
            ax.set_title(f"{v}: {n}", fontsize=11)
        axes[i][0].set_ylabel(r"$S_{ch}$" if kind == "shapley" else r"$e$", fontsize=12)
    handles = [Line2D([], [], color=CH_COL[c], ls="--", marker="o", label=("" if kind == "shapley" else "removed: ")
                      + CH_TEX[c]) for c in ("b", "q", "q_minus_qs", "z") if c in used]
    if kind == "ablation":
        handles += [Line2D([], [], color="k", marker="o", label=r"$e_\emptyset$ (nothing removed)"),
                    Line2D([], [], color="0.5", ls="-.", label=r"$e=1$: prediction $\hat Q_c=0$")]
    handles.append(Patch(color="0.92", label=r"$\dagger\ \mathrm{test}$"))
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=10)
    if kind == "shapley":
        fig.suptitle(r"$S_{ch}=\frac{\sum_c\phi_{ch}^2}{\sum_c\sum_{ch'}\phi_{ch'}^2},\ \phi_{ch}=\sum_{S\subseteq N\setminus\{ch\}}"
                     r"\frac{|S|!\,(n-1-|S|)!}{n!}\left[v(S\cup\{ch\})-v(S)\right]$, $v(S)=\hat Q_c$(channels in $S$ from "
                     r"$\mathbf{x}$, the others at their mean patch); mean over seeds", fontsize=12)
    else:
        fig.suptitle(r"$e_{-ch}=\sqrt{\frac{\sum_c\left(\hat Q_c(\mathbf{x}\ \mathrm{with\ channel}\ ch\ \mathrm{at\ its\ mean\ patch})"
                     r"-Q_c\right)^2}{\sum_c Q_c^2}}$; mean over seeds", fontsize=12)
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 7. Main
# ============================================================

def part_outputs(part):
    figs = {"qd": ("Qd_GradSHAP_template_shares.png", "Qd_Shapley_template_shares.png", "Qd_ablation_errors.png",
                   "Qd_a_channel_attribution.png"),
            "qc": ("Qc_GradSHAP_maps.png", "Qc_channel_shapley.png", "Qc_channel_ablation.png")}[part]
    return [OUT_DIR / f"summary_{part}.txt", OUT_DIR / f"xai_{part}.npz"] + [OUT_DIR / f for f in figs]


def check_patches(r, what):
    if not (r["patch_check"] < PATCH_TOL and r["grad_check"] < PATCH_TOL):
        sys.exit(f"{what}: evaluation on patches differs from the full frame (prediction {r['patch_check']:.1e}, "
                 f"gradient {r['grad_check']:.1e}; tolerance {PATCH_TOL:.0e})")


def load_checkpoint(path, k):
    if not path.exists():
        return {"K": k, "res": {}, "control": {}}
    with open(path, "rb") as f:
        ck = pickle.load(f)
    if ck["K"] != k:
        sys.exit(f"{path} was made with K = {ck['K']}, not {k}")
    return ck


def save_checkpoint(path, ck):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(ck, f)
    os.replace(tmp, path)


def ig_regression(res):
    with np.load(J2_DATA) as d:
        return max(float(np.max(np.abs(r["S_IG"] - d[f"{v}__{n}__{s}__{l}__S_T"]))) for (v, n, s, l), r in res.items())


def main(argv=None):
    ap = argparse.ArgumentParser(description="XAI attribution of the stencil CNNs (see the docstring).")
    ap.add_argument("part", choices=("qd", "qc", "pilot"))
    ap.add_argument("pilot", nargs="*", help="pilot only: VERSION CONFIG RUN")
    ap.add_argument("--k", type=int, default=K_REF, help="GradSHAP references per cell (even)")
    ap.add_argument("--hours", type=float, default=3.5, help="stop (exit code 3, checkpoint kept) after this")
    a = ap.parse_args(argv)
    if a.k < 2 or a.k % 2:
        sys.exit("--k must be even and >= 2 (the K/2 check)")
    t_start = time.time()
    if a.part == "pilot":
        if len(a.pilot) != 3 or a.pilot[0] not in VERSIONS or a.pilot[2] not in T.TRAIN_RUNS + T.TEST_RUNS:
            sys.exit("pilot: give VERSION CONFIG RUN")
        part = next((p for p, cs in CONFIGS.items() if a.pilot[1] in cs), None)
        if part is None:
            sys.exit(f"pilot: unknown configuration {a.pilot[1]}")
        versions, configs, labels = [a.pilot[0]], [a.pilot[1]], [a.pilot[2]]
    else:
        part = a.part
        versions, configs, labels = list(VERSIONS), list(CONFIGS[part]), T.TRAIN_RUNS + T.TEST_RUNS
        if any(p.exists() for p in part_outputs(part)):
            sys.exit(f"outputs of part {part} exist in {OUT_DIR}; refusing to overwrite.")
    T.configure_gpu()
    basis = ja.load_basis(SIM / "v034" / "Q_d_b" / "normalization_constants.npz")
    runs = load_part_runs(part, labels)
    refs, pool_check = channel_references(runs, basis["bbar"])
    rho = rho_bar()
    if len({r["gamma_tau"] for r in runs.values()}) != 1:
        sys.exit("gamma/tau differs between runs")
    order = T.TRAIN_RUNS + T.TEST_RUNS
    idx = {l: reference_indices(order.index(l), len(r["pool"]), len(r["fields"]), a.k, T.NZ, T.NY)
           for l, r in runs.items()}
    print(f"Data loaded ({time.time() - t_start:.0f} s); pool check {pool_check:.1e}; rho_bar {rho:.6f}", flush=True)
    pilot = a.part == "pilot"
    ckpt = CKPT_DIR / f"xai_{part}_checkpoint.pkl"
    if not pilot:
        CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ck = {"K": a.k, "res": {}, "control": {}} if pilot else load_checkpoint(ckpt, a.k)
    res, control = ck["res"], ck["control"]
    if res or control:
        print(f"Resumed from {ckpt}: {len(res)} model results, {len(control)} controls", flush=True)
    for v in versions:
        ev = ja.EVS[v]
        for n in configs:
            setup = ev.load_config(n)
            if setup is None:
                sys.exit(f"{v} {n} is not trained")
            t = ev.T
            for l, run in runs.items():
                todo = [s for s in setup["models"] if (v, n, s, l) not in res]
                ctl = v == "v034" and not pilot and n in HAS_EXACT and (n, l) not in control
                if not todo and not ctl:
                    continue
                if not pilot and time.time() - t_start > a.hours * 3600:
                    print(f"Time budget of {a.hours} h reached: {len(res)} model results and {len(control)} controls "
                          f"saved in {ckpt}; run the same command again to resume.", flush=True)
                    sys.exit(3)
                X = model_inputs(t, setup, run)
                norms = norms_of(t, setup, run)
                ctx = context(t, setup, run, refs, rho, idx[l])
                for s in todo:
                    model = setup["models"][s]
                    t0 = time.time()
                    r = res[(v, n, s, l)] = analyse(model, X, norms, run, ctx)
                    print(f"  {v} {n} seed {s} {l:<15} e_0 {r['e0']:.3e}  comp IG {r['comp_IG']:.1e} G {r['comp_G']:.1e}  "
                          f"K/2 dev {r['G_half_dev']:.1e}  patch {r['patch_check']:.1e} grad {r['grad_check']:.1e}  "
                          f"({time.time() - t0:.0f} s: " + ", ".join(f"{k} {x:.0f}" for k, x in r["seconds"].items())
                          + ")", flush=True)
                    check_patches(r, f"{v} {n} seed {s} {l}")
                    if not pilot:
                        save_checkpoint(ckpt, ck)
                    if pilot:
                        for key in ("S_IG", "S_G", "S_G_half", "S_SH", "S_ch", "e_abl", "e_ch"):
                            if key in r:
                                print(f"    {key:<9} {fmt(r[key])}", flush=True)
                        break
                if ctl:
                    em = exact_model(t, setup, run["gamma_tau"])
                    if em is not None:
                        ctx_e = dict(ctx, exact=exact_info(n, run["ra"], rho))
                        r = control[(n, l)] = analyse(em, X, norms, run, ctx_e)
                        print(f"  exact {n} {l:<15} e_0 {r['e0']:.3e}  patch {r['patch_check']:.1e} "
                              f"grad {r['grad_check']:.1e}  "
                              + " ".join(f"{k} {r[k]:.1e}" for k in ("dev_IG", "dev_SH", "dev_G", "dev_ch") if k in r),
                              flush=True)
                        check_patches(r, f"exact {n} {l}")
                        save_checkpoint(ckpt, ck)
            del setup
            _PATCH_MODELS.clear()
            tf.keras.backend.clear_session()
            gc.collect()
    if a.part == "pilot":
        print(f"Elapsed: {time.time() - t_start:.1f} s")
        return res
    info = {"K": a.k, "pool_check": pool_check, "rho_bar": rho,
            "ig_regression": ig_regression(res) if part == "qd" else None}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    text = summary_text(part, res, control, runs, info)
    (OUT_DIR / f"summary_{part}.txt").write_text(text)
    print("\n" + text)
    save_data(res, control, OUT_DIR / f"xai_{part}.npz")
    if part == "qd":
        plot_template_shares(res, control, runs, "S_G", GRADSHAP_TEX, OUT_DIR / "Qd_GradSHAP_template_shares.png")
        plot_template_shares(res, control, runs, "S_SH", SHAPLEY_TEX, OUT_DIR / "Qd_Shapley_template_shares.png")
        plot_ablation(res, control, runs, OUT_DIR / "Qd_ablation_errors.png")
        plot_qd_a_channels(res, control, runs, OUT_DIR / "Qd_a_channel_attribution.png")
    else:
        plot_qc_maps(res, control, OUT_DIR / "Qc_GradSHAP_maps.png")
        plot_qc_channels(res, control, runs, "shapley", OUT_DIR / "Qc_channel_shapley.png")
        plot_qc_channels(res, control, runs, "ablation", OUT_DIR / "Qc_channel_ablation.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return res, control


if __name__ == "__main__":
    main()
