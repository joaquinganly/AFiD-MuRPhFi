"""Test of diagnostics/xai_attribution.py (TensorFlow on the CPU; plain script, no pytest). On small grids,
synthetic fields and hand-built models:
 1. shapley_matrix: efficiency, an additive game returns its own terms, the two-player formula, the
    a1 a2 game gives (1/2, 1/2), symmetric players get equal values;
 2. pool_draws + pool_patches repeat the whitening draws of the training script (fit_whitening's xbar);
    channel_references (z from the pool rows, b replaced by bbar, the check value);
 3. Patches (the model on 3 x 3 patches, padding dropped) = the full-frame prediction and the lattice
    gradient of jacobian_attribution.py, for a nonlinear model and the v034 whitening layout;
 4. Patches.path_grad with the IG baseline equals ja.ig_at_cells (nonlinear two-channel model); its
    completeness with a random reference within 1e-2 at 256 steps;
 5. ExactQd (Q_d_b) = ja.exact_model; analyse() on it: e_0 ~ 0 against the numpy Laplacian with wall
    ghosts, IG = Shapley = (L.u_k) a_k, GradSHAP = (L.u_k) mean_j u_k.(b - b'_j), completeness, Shapley
    efficiency, e_-k = the analytic value;
 6. ExactQd (Q_d_a, log10 Ra channel) = kappa(rho) Lap b; analyse(): channel Shapley = the analytic
    (1 +- r)/2 formulas, Level 2 shares as for Q_d_b;
 7. ExactQc (C4, C1) = numpy (gamma/tau) max(q - q_s, 0); analyse() on C4: no credit to b, GradSHAP and IG
    shares all on the centre cell of q - q_s; context() and model_inputs() for C1 (z channel);
 8. analyse() on a nonlinear model: Shapley efficiency to round-off;
 9. summary_text, save_data and the seven figures run (mathtext renders); checkpoint round trip, and a
    checkpoint made with another K is refused.
Usage (from simulations_v2/): python tests/test_xai_attribution.py diagnostics/xai_attribution.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # exact-equality tests on the CPU
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("xai_attribution", script)
xa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xa)
ja, T = xa.ja, xa.T
rng = np.random.default_rng(5)
NZ, NY = 10, 18                                    # as test_jacobian_attribution.py
dz = 1.0 / NZ
T.NZ, T.NY = NZ, NY
ROWS, COLS = xa.interior_row_sets(NZ, NY)
U, L_U = xa.U, xa.L_U


def smooth_b(n):
    return (np.linspace(0, 1, NZ)[None, :, None] + 0.05 * rng.normal(size=(n, NZ, NY))).astype(np.float32)


def lap_numpy(b):
    bp = np.concatenate([2 * 0.0 - b[:, :1], b, 2 * 1.0 - b[:, -1:]], axis=1).astype(np.float64)
    return (bp[:, 2:] + bp[:, :-2] + np.roll(bp[:, 1:-1], 1, 2) + np.roll(bp[:, 1:-1], -1, 2) - 4 * bp[:, 1:-1]) / dz ** 2


def at_cells(F):
    return np.concatenate([xa.cells(F, r, COLS).reshape(-1) for r in ROWS])


# ---------- 1. Shapley coefficients ----------
for n in (1, 2, 3, 9):
    S, coef = xa.shapley_matrix(n)
    v = rng.normal(size=len(S))
    phi = coef.T @ v
    assert math.isclose(phi.sum(), v[S.all(axis=1)][0] - v[~S.any(axis=1)][0], abs_tol=1e-12)
    c = rng.normal(size=n)
    assert np.allclose(coef.T @ (S @ c), c)                     # additive game
    assert np.allclose(coef.T @ S.sum(axis=1).astype(float) ** 2, (coef.T @ S.sum(axis=1).astype(float) ** 2)[0])
S, coef = xa.shapley_matrix(2)
val = {(0, 0): 0.0, (0, 1): 1.0, (1, 0): 2.0, (1, 1): 5.0}
assert np.allclose(coef.T @ np.array([val[tuple(int(x) for x in s)] for s in S]), [3.0, 2.0])
assert np.allclose(coef.T @ S.prod(axis=1).astype(float), [0.5, 0.5])   # v = a1 a2 at (1, 1): (1/2, 1/2)
print("1. Shapley coefficients: efficiency, additive game, two-player formula, a1 a2 game, symmetry: OK")

# ---------- 2. pool draws = whitening draws ----------
T.select_config("Q_d_b")
T.WHITEN_CELLS = 400
n_train = 6
runs_w = [{"fields": smooth_b(n_train + 2)[..., None], "parts": {"train": slice(0, n_train)}} for _ in range(3)]
wh = T.fit_whitening(runs_w)
g = np.random.default_rng(T.WHITEN_SEED)
pools = []
for rw in runs_w:
    ti, ii, jj = xa.pool_draws(g, n_train, NZ, NY, T.WHITEN_CELLS)
    pools.append((xa.pool_patches(rw["fields"][rw["parts"]["train"]], ti, ii, jj), ii))
assert np.allclose(np.concatenate([p for p, _ in pools])[:, 0].astype(np.float64).mean(axis=0), wh["xbar"],
                   rtol=0, atol=1e-12)
xm = (np.arange(NZ) + 0.5) * dz
fake = {l: {"pool": p, "pool_rows": rows, "chans": ["b"], "xm": xm} for l, (p, rows) in zip(T.TRAIN_RUNS, pools)}
refs0, check = xa.channel_references(fake, wh["xbar"] + 0.01)
assert abs(check - 0.01) < 1e-9 and np.allclose(refs0["b"], wh["xbar"] + 0.01)
rows_all = np.concatenate([r for _, r in pools])
assert np.allclose(refs0["z"], [xm[rows_all + m // 3 - 1].mean() for m in range(9)])
print("2. pool cells = the whitening draws of the training script (xbar reproduced); channel references: OK")

# ---------- 3. evaluation on patches = full frames ----------
def seeded(k):
    return {"kernel_initializer": tf.keras.initializers.GlorotUniform(seed=500 + k),
            "bias_initializer": tf.keras.initializers.RandomUniform(-0.3, 0.3, seed=600 + k)}


inp2 = tf.keras.layers.Input((NZ, NY, 2))
h = T.StencilPad(wall_bottom=[0.0, 0.0], wall_top=[1.0, 0.0], dirichlet=[1.0, 0.0])(inp2)
h = tf.keras.layers.Conv2D(8, 3, padding="valid", activation="relu", **seeded(0))(h)
h = tf.keras.layers.Conv2D(8, 1, activation="relu", **seeded(1))(h)
nl = tf.keras.Model(inp2, tf.keras.layers.Conv2D(1, 1, **seeded(2))(h))
X2 = rng.random((2, NZ, NY, 2)).astype(np.float32)
kern, bias = T.whitening_weights(wh)                 # the v034 layout: float64 padding and whitening layer
inp_w = tf.keras.layers.Input((NZ, NY, 1))
wl = tf.keras.layers.Conv2D(9, 3, padding="valid", dtype="float64", trainable=False)
xw = wl(T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0], dtype="float64")(inp_w))
wl.set_weights([kern, bias])
h = tf.keras.layers.Conv2D(4, 1, activation="relu", **seeded(3))(xw)
wmod = tf.keras.Model(inp_w, tf.keras.layers.Conv2D(1, 1, **seeded(4))(h))
Xw = smooth_b(2)[..., None]
for model, Xm_ in ((nl, X2), (wmod, Xw)):
    pf = xa.Patches(model, batch=5)                  # several batches per call
    C_ = Xm_.shape[3]
    for rows in ROWS:
        P = xa.patches_of(Xm_, rows, COLS).reshape(-1, C_, 9)
        assert np.allclose(pf.pred(P), xa.cells(ja.predict(model, Xm_), rows, COLS).reshape(-1), rtol=1e-5, atol=1e-6)
        lat = xa.patches_of(ja.cell_grad(model, Xm_, rows, COLS), rows, COLS).reshape(-1, C_, 9)
        assert np.allclose(pf.grad(P), lat, rtol=1e-5, atol=1e-6)
print("3. evaluation on patches (padding dropped) = full-frame prediction and lattice gradient, nonlinear model and "
      "the v034 whitening layout: OK")

# ---------- 4. path gradient on patches = ja.ig_at_cells ----------
xbar_n = 0.5 + 0.1 * rng.normal(size=9)
ig = ja.ig_at_cells(nl, X2, 0, 1.0, 1.0, xbar_n, steps=16, chunk=4)
pf = xa.Patches(nl, batch=7)
Jb = []
for rows in ROWS:
    P = xa.patches_of(X2, rows, COLS).reshape(-1, 2, 9)
    R = P.copy()
    R[:, 0] = xbar_n
    Jb.append(pf.path_grad(P, R, steps=16)[:, 0])
assert np.allclose(np.concatenate(Jb), ig["Jbar"], rtol=1e-5, atol=1e-7)
P = xa.patches_of(X2, ROWS[0], COLS).reshape(-1, 2, 9)
R = rng.random(P.shape)
G = pf.path_grad(P, R, steps=256)
dpred = pf.pred(P) - pf.pred(R)
e256 = xa.rel_rms(((P - R) * G).sum(axis=(1, 2)) - dpred, dpred)
assert e256 < 1e-2, e256
print(f"4. path gradient on patches = ja.ig_at_cells; completeness with a random reference (all channels) {e256:.1e} "
      "at 256 steps: OK")

# ---------- 5. exact operator, Q_d_b ----------
inorm, tnorm = {"b": (0.4, 0.3)}, (0.1, 2.0)
setup_b = {"name": "Q_d_b", "input_norm": inorm, "target_norm": tnorm}
eqb = xa.ExactQd(T, setup_b)
b = smooth_b(3)
Xb = ((b[..., None] - np.float32(0.4)) / np.float32(0.3)).astype(np.float32)
assert np.allclose(ja.predict(eqb, Xb), ja.predict(ja.exact_model(T, inorm, tnorm), Xb), rtol=1e-5, atol=1e-4)
lap = lap_numpy(b)
assert np.allclose(ja.predict(eqb, Xb) * 2.0 + 0.1, lap, rtol=1e-5, atol=1e-4)
N, K = 300, 4
pool_b = xa.pool_patches(smooth_b(4)[..., None], *xa.pool_draws(rng, 4, NZ, NY, N))       # (N, 1, 9) raw
bbar = pool_b[:, 0].astype(np.float64).mean(axis=0)
idx = [rng.integers(0, N, (K, len(b), len(r), len(COLS))) for r in ROWS]
run_b = {"psi_dns": dz ** 2 * lap}
ctx_b = {"inputs": ["b"], "ref_n": ((bbar - 0.4) / 0.3)[None, :], "pool_n": (pool_b - np.float32(0.4)) / np.float32(0.3),
         "ref_idx": idx, "level2": "templates", "bbar": bbar, "exact": {"linear_b": True}}
norms_b = {"input": inorm, "target": tnorm, "fac": dz ** 2}
rb = xa.analyse(eqb, Xb, norms_b, run_b, ctx_b, chunk=4)
assert rb["e0"] < 1e-5, rb["e0"]
assert rb["dev_IG"] < 1e-5 and rb["dev_G"] < 1e-5 and rb["dev_SH"] < 1e-4, (rb["dev_IG"], rb["dev_G"], rb["dev_SH"])
assert rb["comp_IG"] < 1e-5 and rb["comp_G"] < 1e-5 and rb["eff_SH"] < 1e-9 and rb["eff_ch"] < 1e-9
assert np.allclose(rb["S_IG"], rb["S_SH"], atol=1e-6) and math.isclose(float(rb["S_G"].sum()), 1.0)
A = np.concatenate([((xa.patches_of(Xb, r, COLS)[..., 0, :] * 0.3 + 0.4 - bbar) @ U).reshape(-1, 9) for r in ROWS])
dns = at_cells(run_b["psi_dns"])
e_an = np.sqrt(np.sum((L_U * A) ** 2, axis=0) / np.sum(dns ** 2))
assert np.allclose(rb["e_abl"], e_an, rtol=1e-4, atol=1e-5), (rb["e_abl"], e_an)
off = [k for k in range(9) if abs(L_U[k]) < 1e-12]
assert len(off) == 6 and np.all(rb["S_IG"][off] < 1e-12) and np.all(rb["S_SH"][off] < 1e-8)
print("5. exact operator (Q_d_b): = ja.exact_model = numpy Laplacian; IG = Shapley = (L.u) a, GradSHAP analytic, "
      "completeness, efficiency, e_-k analytic: OK")

# ---------- 6. exact operator, Q_d_a ----------
inorm_a, tnorm_a = {"b": (0.4, 0.3), "log10_Ra": (7.0, 0.6)}, (0.0, 5e-3)
setup_a = {"name": "Q_d_a", "input_norm": inorm_a, "target_norm": tnorm_a}
eqa = xa.ExactQd(T, setup_a)
rho_c, rho_ref = 6.4, 7.1
kap_c = 10 ** (-rho_c / 2)
Xa = np.concatenate([Xb, np.full(Xb.shape, np.float32((rho_c - 7.0) / 0.6))], axis=-1)
assert np.allclose(ja.predict(eqa, Xa) * 5e-3 / kap_c, lap, rtol=1e-5, atol=1e-4)
ctx_a = dict(ctx_b, inputs=["b", "log10_Ra"], ref_n=np.stack([ctx_b["ref_n"][0], np.full(9, (rho_ref - 7.0) / 0.6)]),
             pool_n=np.concatenate([ctx_b["pool_n"], np.full((N, 1, 9), np.float32((rho_c - 7.0) / 0.6))], axis=1),
             exact=xa.exact_info("Q_d_a", 10 ** rho_c, rho_ref))
assert math.isclose(ctx_a["exact"]["r"], 10 ** ((rho_c - rho_ref) / 2))
ra = xa.analyse(eqa, Xa, {"input": inorm_a, "target": tnorm_a, "fac": dz ** 2 / kap_c}, run_b, ctx_a, chunk=4)
assert ra["dev_ch"] < 1e-5 and ra["dev_IG"] < 1e-5 and ra["dev_G"] < 1e-5 and ra["e0"] < 1e-4, ra
assert np.allclose(ra["S_IG"], rb["S_IG"], atol=1e-5) and ra["S_ch"][1] > 1e-3 and np.allclose(ra["S_G"], rb["S_G"], atol=1e-5)
print(f"6. exact operator (Q_d_a): kappa(rho) Lap b; channel Shapley = (1 +- r)/2 formulas (dev {ra['dev_ch']:.1e}), "
      f"S_rho = {ra['S_ch'][1]:.2e}: OK")

# ---------- 7. exact Q_c ----------
inorm_c4, tnorm_c = {"b": (0.4, 0.3), "q_minus_qs": (-0.01, 0.02)}, (0.01, 0.05)
qc4 = xa.ExactQc(["b", "q_minus_qs"], inorm_c4, tnorm_c, 20.0, 3.0, 2.0)
s = 0.02 * rng.normal(size=(3, NZ, NY))
Xc = np.stack([Xb[..., 0], ((s + 0.01) / 0.02).astype(np.float32)], axis=-1)
s_back = -0.01 + 0.02 * Xc[..., 1].astype(np.float64)
assert np.allclose(ja.predict(qc4, Xc) * 0.05 + 0.01, 20.0 * np.maximum(s_back, 0.0), atol=1e-7)
inorm_c1 = {"b": (0.4, 0.3), "q": (0.5, 0.2), "z": (0.5, 0.29)}
qc1 = xa.ExactQc(["b", "q", "z"], inorm_c1, tnorm_c, 20.0, 3.0, 2.0)
q = np.exp(3.0 * (b - 2.0 * xm[None, :, None])) + 0.01 * rng.normal(size=b.shape)
Xc1 = np.stack([Xb[..., 0], ((q - 0.5) / 0.2).astype(np.float32),
                np.broadcast_to(((xm - 0.5) / 0.29).astype(np.float32)[None, :, None], b.shape)], axis=-1)
raw = [m + sd * Xc1[..., i].astype(np.float64) for i, (m, sd) in enumerate(inorm_c1.values())]
assert np.allclose(ja.predict(qc1, Xc1) * 0.05 + 0.01, 20.0 * np.maximum(raw[1] - np.exp(3.0 * (raw[0] - 2.0 * raw[2])), 0.0),
                   atol=1e-6)
pool_s = xa.pool_patches(np.stack([smooth_b(4), 0.02 * rng.normal(size=(4, NZ, NY)).astype(np.float32)], -1),
                         *xa.pool_draws(rng, 4, NZ, NY, N))
mean_s = pool_s[:, 1].astype(np.float64).mean(axis=0)
ctx_c = {"inputs": ["b", "q_minus_qs"], "ref_n": np.stack([(bbar - 0.4) / 0.3, (mean_s + 0.01) / 0.02]),
         "pool_n": (pool_s - np.array([0.4, -0.01], np.float32)[None, :, None]) / np.array([0.3, 0.02], np.float32)[None, :, None],
         "ref_idx": idx, "level2": "cells", "bbar": bbar, "exact": xa.exact_info("Q_c_C4", 1e6, 7.0)}
rc = xa.analyse(qc4, Xc, {"input": inorm_c4, "target": tnorm_c, "fac": 1.0},
                {"psi_dns": 20.0 * np.maximum(s_back, 0.0)}, ctx_c, chunk=4)
assert rc["dev_ch"] < 1e-12 and rc["e0"] < 1e-6, (rc["dev_ch"], rc["e0"])
for key in ("S_G", "S_IG"):
    assert rc[key].shape == (2, 9) and abs(rc[key][1, 4] - 1.0) < 1e-12, (key, rc[key])
assert xa.exact_model(T, {"name": "Q_c_C3", "input_norm": {"b": (0, 1), "q": (0, 1)}, "target_norm": (0, 1)}, 20.0) is None
setup_c1 = {"name": "Q_c_C1", "input_norm": inorm_c1, "target_norm": tnorm_c}
run_c1 = {"label": "fake", "chans": ["b", "q", "q_minus_qs"], "fields": np.stack([b, q.astype(np.float32), s.astype(np.float32)], -1),
          "xm": xm, "params": {"log10_Ra": rho_c}, "pool_rows": rng.integers(1, NZ - 1, N),
          "pool": np.concatenate([pool_s[:, :1], pool_s[:, 1:] + 0.5, pool_s[:, 1:]], axis=1)}
Xm = xa.model_inputs(T, setup_c1, run_c1)
assert np.allclose(Xm, Xc1, atol=1e-6)
cx = xa.context(T, setup_c1, run_c1, {"b": bbar, "q": np.full(9, 0.6), "z": np.full(9, 0.5)}, 7.0, idx)
assert cx["inputs"] == ["b", "q", "z"] and cx["pool_n"].shape == (N, 3, 9) and cx["level2"] == "cells"
assert np.allclose(cx["pool_n"][:, 2], (xm[run_c1["pool_rows"][:, None] + np.arange(9)[None, :] // 3 - 1] - 0.5) / 0.29, atol=1e-6)
assert np.allclose(cx["pool_n"][:, 1], (run_c1["pool"][:, 1] - 0.5) / 0.2, atol=1e-6) and np.allclose(cx["ref_n"][1], 0.5)
print("7. exact Q_c (C4, C1) = numpy; C4: no credit to b, GradSHAP and IG on the centre of q - q_s; "
      "model_inputs and context (z channel): OK")

# ---------- 8. nonlinear model ----------
ctx_n = dict(ctx_b, inputs=["b", "x2"], ref_n=np.stack([ctx_b["ref_n"][0], np.full(9, 0.5)]),
             pool_n=np.concatenate([ctx_b["pool_n"], rng.random((N, 1, 9)).astype(np.float32)], axis=1), exact=None,
             ref_idx=[rng.integers(0, N, (K, 2, len(r), len(COLS))) for r in ROWS])
rn = xa.analyse(nl, X2, {"input": inorm, "target": tnorm, "fac": 1.0}, {"psi_dns": rng.normal(size=X2.shape[:3])}, ctx_n, chunk=4)
assert rn["eff_SH"] < 1e-9 and rn["eff_ch"] < 1e-9 and rn["comp_IG"] < 0.2 and rn["comp_G"] < 0.2, rn
for r_ in (rb, ra, rc, rn):
    assert r_["patch_check"] < 1e-5 and r_["grad_check"] < 1e-5, (r_["patch_check"], r_["grad_check"])
print(f"8. nonlinear model: Shapley efficiency to round-off; completeness at {ja.M_STEPS} steps IG {rn['comp_IG']:.1e}, "
      f"GradSHAP {rn['comp_G']:.1e}: OK")

# ---------- 9. summary, data file and figures ----------
labels = T.TRAIN_RUNS + T.TEST_RUNS
runs = {l: {"ra": float(l.split("_")[1]), "frames": [811]} for l in labels}
res_d = {(v, n, 42, l): (ra if n == "Q_d_a" else rb) for v in xa.VERSIONS for n in xa.CONFIGS["qd"] for l in labels}
ctl_d = {(n, l): (ra if n == "Q_d_a" else rb) for n in xa.CONFIGS["qd"] for l in labels}


def fake_qc(n):
    C = len(T.CONFIGS[n]["inputs"])
    S = rng.random((C, 9))
    S /= S.sum()
    return dict(rc, e_ch=np.full(C, 0.5), S_ch=np.full(C, 1.0 / C), S_G=S, S_G_half=S, S_IG=S, G_half_dev=0.0)


res_c = {(v, n, 42, l): fake_qc(n) for v in xa.VERSIONS for n in xa.CONFIGS["qc"] for l in labels}
ctl_c = {(n, l): fake_qc(n) for n in ("Q_c_C1", "Q_c_C2", "Q_c_C4") for l in labels}
info = {"K": K, "pool_check": 0.0, "rho_bar": 7.0, "ig_regression": 0.0}
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    for part, res, ctl in (("qd", res_d, ctl_d), ("qc", res_c, ctl_c)):
        text = xa.summary_text(part, res, ctl, runs, info)
        assert "seed 42" in text and "Controls" in text and "=== exact" in text
        xa.save_data(res, ctl, tmp / f"{part}.npz")
    with np.load(tmp / "qd.npz") as d:
        assert np.allclose(d["v034__Q_d_a__42__Ra_1e8_Pr_1__S_ch"], ra["S_ch"]) and np.allclose(d["exact__Q_d_b__0__Ra_1e6_Pr_1__S_IG"], rb["S_IG"])
    xa.plot_template_shares(res_d, ctl_d, runs, "S_G", xa.GRADSHAP_TEX, tmp / "a.png")
    xa.plot_template_shares(res_d, ctl_d, runs, "S_SH", xa.SHAPLEY_TEX, tmp / "b.png")
    xa.plot_ablation(res_d, ctl_d, runs, tmp / "c.png")
    xa.plot_qd_a_channels(res_d, ctl_d, runs, tmp / "d.png")
    xa.plot_qc_maps(res_c, ctl_c, tmp / "e.png")
    xa.plot_qc_channels(res_c, ctl_c, runs, "shapley", tmp / "f.png")
    xa.plot_qc_channels(res_c, ctl_c, runs, "ablation", tmp / "g.png")
    assert all((tmp / f"{f}.png").stat().st_size > 10_000 for f in "abcdefg")
    xa.save_checkpoint(tmp / "ck.pkl", {"K": 4, "res": res_d, "control": ctl_d})
    ck = xa.load_checkpoint(tmp / "ck.pkl", 4)
    assert ck["res"].keys() == res_d.keys() and np.allclose(ck["control"][("Q_d_a", "Ra_1e8_Pr_1")]["S_ch"], ra["S_ch"])
    assert xa.load_checkpoint(tmp / "none.pkl", 8) == {"K": 8, "res": {}, "control": {}}
    try:
        xa.load_checkpoint(tmp / "ck.pkl", 8)
        raise AssertionError("a checkpoint made with another K was accepted")
    except SystemExit:
        pass
print("9. summary, data file, the seven figures and the checkpoint round trip: OK")
print("ALL XAI-ATTRIBUTION TESTS PASSED")
