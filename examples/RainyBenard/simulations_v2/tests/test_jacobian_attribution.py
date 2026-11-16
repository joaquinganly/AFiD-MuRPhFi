"""Test of diagnostics/jacobian_attribution.py (TensorFlow on the CPU; plain script, no pytest).
On small grids and hand-built models:
 1. jacobian_at_cells (reverse-mode AD) of a linear Laplacian model with the ghost padding gives the
    interior [0 1 0; 1 -4 1; 0 1 0] and the wall-row stencils of EXACT (NaN beyond the wall), and
    equals the central-difference probe of the evaluation (stencil_response) at the same cells;
 2. for a random nonlinear ReLU model with two input channels, the lattice gradient (one backward
    pass for all sampled cells) equals the gradient of each cell's output computed alone, which is
    zero outside the cell's 3 x 3 patch;
 3. integrated gradients: for the linear model Jbar = L at every interior cell, dB = b - bbar (computed
    independently), completeness to round-off, with a step count not divisible by the chunk size;
    for the nonlinear model completeness within 1e-2 (rms, relative) at 256 steps;
 4. Taylor basis: V orthonormal and equal (up to column signs) to the eigenvectors behind W =
    Lambda^(-1/2) V^T; x~ = W (b - bbar)/sigma up to sign; sum_k IG~_k = sum_m IG_m; the exact
    operator gives IG~ = IG~ex and r_k = 1; any linear map K gives r_k = (K.v_k)/(L.v_k); shares sum
    to 1; each Gram template is matched to itself;
 5. exact_model + scale_of in normalised units: denormalised prediction = numpy Laplacian with the
    wall ghosts; analyse() end to end gives J = L, completeness ~ 0, S = S_ex, r = 1;
 6. summary_text, save_data and the four figures run (mathtext renders).
Usage (from simulations_v2/): python tests/test_jacobian_attribution.py diagnostics/jacobian_attribution.py
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
spec = importlib.util.spec_from_file_location("jacobian_attribution", script)
ja = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ja)
T = ja.T
ev = ja.EVS["v034"]
rng = np.random.default_rng(3)
NZ, NY = 10, 18                                    # (NZ - 1) % 3 != 2 and NY - last sampled column >= 3
dz = 1.0 / NZ
ROW_SETS, COLS = ja.sample_cells(NZ, NY)
assert any(0 in r for r in ROW_SETS) and any(NZ - 1 in r for r in ROW_SETS)


def rms(v):
    return math.sqrt(float(np.mean(np.square(v))))


# ---------- 1. AD Jacobian of the exact operator, and the probe ----------
inp = tf.keras.layers.Input((NZ, NY, 1))
conv = tf.keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
lin = tf.keras.Model(inp, conv(T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0])(inp)))
conv.set_weights([(ja.L3 / dz ** 2)[:, :, None, None].astype(np.float32)])
X = rng.random((3, NZ, NY, 1)).astype(np.float32)
jac = ja.jacobian_at_cells(lin, X, 0, dz ** 2, chunk=2)
kinds = set()
for rows, J in jac:
    for i, r in enumerate(rows):
        kind = "bottom" if r == 0 else "top" if r == NZ - 1 else "interior"
        kinds.add(kind)
        ref, got = ja.EXACT[kind], J[:, i]
        assert np.array_equal(np.isnan(got), np.broadcast_to(np.isnan(ref), got.shape)), (kind, got[0, 0])
        assert np.allclose(got[..., ~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-4), (kind, got[0, 0])
assert kinds == {"bottom", "top", "interior"}
for (r1, J), (r2, D) in zip(jac, ev.stencil_response(lin, X, 0, dz ** 2)):
    assert np.array_equal(r1, r2) and np.allclose(J, D, atol=2e-3, equal_nan=True)
for ny in (17, NY, 2048):                          # sampled cells' patches are disjoint (incl. the y wrap)
    occ = np.zeros(ny, int)
    for c in ja.sample_cells(NZ, ny)[1]:
        occ[[(c + d) % ny for d in (-1, 0, 1)]] += 1
    assert occ.max() == 1, ny
Kasym = rng.normal(size=(3, 3))                    # orientation: J_pq must be the weight on b_{i+p, j+q}
conv_a = tf.keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
asym = tf.keras.Model(inp, conv_a(T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0])(inp)))
conv_a.set_weights([Kasym[:, :, None, None].astype(np.float32)])
i0, j0 = 4, 7
assert math.isclose(float(asym(X[:1])[0, i0, j0, 0]),
                    sum(Kasym[p + 1, q + 1] * X[0, i0 + p, j0 + q, 0] for p in (-1, 0, 1) for q in (-1, 0, 1)),
                    rel_tol=1e-5)
for rows, J in ja.jacobian_at_cells(asym, X, 0, 1.0):
    assert np.allclose(J[:, (rows >= 1) & (rows <= NZ - 2)], Kasym, atol=1e-5)
T.NZ, T.NY, T.WHITEN_CELLS = NZ, NY, 2000          # whitened model (v034 layout): same orientation
T.select_config("Q_d_b")
bw = (np.linspace(0, 1, NZ)[None, :, None] + 0.05 * rng.normal(size=(6, NZ, NY))).astype(np.float32)
wh = T.fit_whitening([{"fields": bw[..., None], "parts": {"train": slice(0, 6)}}])
kern, bias = T.whitening_weights(wh)
inp_w = tf.keras.layers.Input((NZ, NY, 1))
wl = tf.keras.layers.Conv2D(9, 3, padding="valid", dtype="float64", trainable=False)
xw = wl(T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0], dtype="float64")(inp_w))
wl.set_weights([kern, bias])
ro = tf.keras.layers.Conv2D(1, 1)
wmod = tf.keras.Model(inp_w, ro(xw))
ro.set_weights([np.linalg.solve(wh["W"].T, Kasym.ravel()).reshape(1, 1, 9, 1).astype(np.float32),
                np.array([Kasym.ravel() @ wh["xbar"]], np.float32)])
for rows, J in ja.jacobian_at_cells(wmod, X, 0, 1.0):
    assert np.allclose(J[:, (rows >= 1) & (rows <= NZ - 2)], Kasym, atol=1e-3)
print("1. AD Jacobian = exact stencils (interior and walls) = central-difference probe; orientation with an "
      "asymmetric kernel, also through the v034 whitening layout; disjoint patches: OK")

# ---------- 2. lattice gradient = single-cell gradients (nonlinear model) ----------
def seeded(k):                                     # explicit seeds, as the training script (tf_keras 2.17 +
    return {"kernel_initializer": tf.keras.initializers.GlorotUniform(seed=500 + k),     # Python 3.12 fail
            "bias_initializer": tf.keras.initializers.RandomUniform(-0.3, 0.3, seed=600 + k)}   # otherwise)


inp2 = tf.keras.layers.Input((NZ, NY, 2))
h = T.StencilPad(wall_bottom=[0.0, 0.0], wall_top=[1.0, 0.0], dirichlet=[1.0, 0.0])(inp2)
h = tf.keras.layers.Conv2D(8, 3, padding="valid", activation="relu", **seeded(0))(h)
h = tf.keras.layers.Conv2D(8, 1, activation="relu", **seeded(1))(h)
nl = tf.keras.Model(inp2, tf.keras.layers.Conv2D(1, 1, **seeded(2))(h))
X2 = np.concatenate([rng.random((2, NZ, NY, 1)), np.full((2, NZ, NY, 1), 0.3)], axis=-1).astype(np.float32)
for rows, J in ja.jacobian_at_cells(nl, X2, 0, 1.0, chunk=1):
    for _ in range(6):
        f, i, jc = rng.integers(2), rng.integers(len(rows)), rng.integers(len(COLS))
        r, c = rows[i], COLS[jc]
        Xt = tf.constant(X2[f:f + 1])
        with tf.GradientTape() as tape:
            tape.watch(Xt)
            y = nl(Xt)[0, r, c, 0]
        g = tape.gradient(y, Xt).numpy()[0, ..., 0].astype(np.float64)
        ref = ja.gather_patches(g[None], np.array([r]), np.array([c]))[0, 0, 0]
        assert np.allclose(J[f, i, jc], ref, rtol=1e-5, atol=1e-7, equal_nan=True)
        inside = np.zeros_like(g, bool)
        inside[np.ix_([x for x in (r - 1, r, r + 1) if 0 <= x < NZ], [(c + d) % NY for d in (-1, 0, 1)])] = True
        assert np.all(g[~inside] == 0.0)
print("2. lattice gradient = independent single-cell gradients, zero outside the patch: OK")

# ---------- 3. integrated gradients ----------
xbar_n = 0.5 + 0.1 * rng.normal(size=9)
ig = ja.ig_at_cells(lin, X, 0, dz ** 2, 0.7, xbar_n, steps=10, chunk=4)
assert np.allclose(ig["Jbar"], ja.L3.ravel()[None, :], atol=1e-4)
dB_ref = np.concatenate([ja.gather_patches(X[..., 0].astype(np.float64), r[(r >= 1) & (r <= NZ - 2)], COLS)
                         .reshape(-1, 9) for r in ROW_SETS]) - xbar_n[None, :]
assert np.allclose(ig["dB"], 0.7 * dB_ref, atol=1e-6)
assert rms((ig["dB"] * ig["Jbar"]).sum(1) - ig["dPsi"]) < 1e-5 * rms(ig["dPsi"])
ig2 = ja.ig_at_cells(nl, X2, 0, 1.0, 1.0, xbar_n, steps=256, chunk=64)
e256 = rms((ig2["dB"] * ig2["Jbar"]).sum(1) - ig2["dPsi"]) / rms(ig2["dPsi"])
assert e256 < 1e-2, e256
print(f"3. IG: exact for the linear model; nonlinear completeness rms error {e256:.1e} at 256 steps: OK")

# ---------- 4. Taylor basis ----------
A = rng.normal(size=(9, 9))
lam, V0 = np.linalg.eigh(A @ A.T + 1e-3 * np.eye(9))
lam, V0 = lam[::-1], V0[:, ::-1]
W = (V0 / np.sqrt(lam)).T
B = ja.basis_from_whitening(W, lam, np.zeros(9), 0.2, 0.5)
assert np.allclose(B["V"].T @ B["V"], np.eye(9)) and np.allclose(np.abs(B["V"]), np.abs(V0))
assert np.allclose(B["bbar"], 0.2) and B["sigma"] == 0.5
dB = rng.normal(size=(50, 9))
Jb = rng.normal(size=(50, 9))
xt, igt, igex = ja.taylor_components(dB, Jb, B)
assert np.allclose(np.abs(xt), np.abs((dB / 0.5) @ W.T))
assert np.allclose(igt.sum(1), (dB * Jb).sum(1))
assert math.isclose(float(ja.shares(igt).sum()), 1.0)
_, igL, igLex = ja.taylor_components(dB, np.tile(ja.L3.ravel(), (50, 1)), B)
assert np.allclose(igL, igLex) and np.allclose(ja.learned_fraction(igL, igLex), 1.0)
K = rng.normal(size=9)
_, igK, igKex = ja.taylor_components(dB, np.tile(K, (50, 1)), B)
assert np.allclose(ja.learned_fraction(igK, igKex), (K @ B["V"]) / (ja.L3.ravel() @ B["V"]))
for key, u in ja.TEMPLATES.items():
    a, b, d = ja.template_match(u)
    assert (a, b) == key and abs(d - 1.0) < 1e-12
print("4. Taylor basis, IG~ identities, learned fraction, template matching: OK")

# ---------- 5. control model and analyse() end to end ----------
T.NZ, T.NY = NZ, NY
inorm, tnorm = {"b": (0.4, 0.3)}, (0.1, 2.0)
cm = ja.exact_model(T, inorm, tnorm)
setup = {"name": "Q_d_b", "input_norm": inorm, "target_norm": tnorm}
sc = ja.scale_of(T, setup, {"kappa": 1e-3})
T.select_config("Q_d_a")
assert math.isclose(ja.scale_of(T, setup, {"kappa": 1e-3}), 2.0 / 0.3 * dz ** 2 / 1e-3)
T.select_config("Q_d_b")
assert T.TARGET == "Q_d_over_kappa" and math.isclose(sc, 2.0 / 0.3 * dz ** 2)
Xn = ((X - 0.4) / 0.3).astype(np.float32)
pred = ja.predict(cm, Xn) * 2.0 + 0.1
b = X[..., 0].astype(np.float64)
bp = np.concatenate([2 * 0.0 - b[:, :1], b, 2 * 1.0 - b[:, -1:]], axis=1)
lap = (bp[:, 2:] + bp[:, :-2] + np.roll(b, 1, 2) + np.roll(b, -1, 2) - 4 * b) / dz ** 2
assert np.allclose(pred, lap, rtol=1e-4, atol=1e-2), float(np.max(np.abs(pred - lap)))
Bt = ja.basis_from_whitening(W, lam, (0.5 - 0.4) / 0.3 * np.ones(9), 0.4, 0.3)
xb = (Bt["bbar"] - 0.4) / 0.3
r = ja.analyse(cm, Xn, 0, sc, 0.3, xb, Bt, ev)
assert np.allclose(r["mean"]["interior"], ja.L3, atol=1e-4)
for kind in ("bottom", "top"):
    ok = ~np.isnan(ja.EXACT[kind])
    assert np.allclose(r["mean"][kind][ok], ja.EXACT[kind][ok], atol=1e-4)
assert r["comp_rms"] < 1e-5 and r["basis_check"] < 1e-6, (r["comp_rms"], r["basis_check"])
assert np.allclose(r["S"], r["S_ex"], atol=1e-6) and np.allclose(r["r"], 1.0, atol=1e-4)
assert abs(r["pct"]["centre_J"][2] + 4.0) < 1e-4 and abs(r["pct"]["centre_D"][2] + 4.0) < 1e-3
print("5. exact model in normalised units = Laplacian with wall ghosts; analyse(): J = L, completeness, "
      "S = S_ex, r = 1: OK")

# ---------- 6. summary, data file and figures ----------
order = [(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), (2, 1), (1, 2), (2, 2)]
Bf = ja.basis_from_whitening((np.stack([ja.TEMPLATES[k] for k in order], 1) / np.sqrt(np.logspace(0, -8, 9))).T,
                             np.logspace(0, -8, 9), np.zeros(9), 0.4, 0.3)
assert [x[:2] for x in Bf["labels"]] == order
labels = T.TRAIN_RUNS + T.TEST_RUNS
runs = {l: {"ra": float(l.split("_")[1]), "frames": [811]} for l in labels}
res = {(v, n, 42, l): r for v in ja.VERSIONS for n in ja.QD for l in labels}
res[("v034", "Q_d_b", 42, "Ra_1e8_Pr_1")] = dict(r, mean={**r["mean"], "interior": r["mean"]["interior"] + 1.0})
control = {l: r for l in labels}
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    text = ja.summary_text(res, control, Bf, runs, "test")
    assert "seed 42" in text and "Control" in text
    ja.save_data(res, control, Bf, runs, tmp / "d.npz")
    with np.load(tmp / "d.npz") as d:
        assert np.allclose(d["v034__Q_d_b__42__Ra_1e8_Pr_1__mean_interior"], r["mean"]["interior"] + 1.0)
        assert np.allclose(d["v033__Q_d_b__42__Ra_1e8_Pr_1__mean_interior"], r["mean"]["interior"])
    ja.plot_stencils(res, runs, ("Ra_1e6_Pr_1", "Ra_1e8_Pr_1", "Ra_1.5e8_Pr_1"), tmp / "a.png")
    ja.plot_vs_probe(res, runs, tmp / "b.png")
    ja.plot_shares(res, runs, Bf, tmp / "c.png")
    ja.plot_learned_fraction(res, runs, Bf, tmp / "e.png")
    assert all((tmp / f).stat().st_size > 10_000 for f in ("a.png", "b.png", "c.png", "e.png"))
print("6. summary, data file and the four figures: OK")
print("ALL JACOBIAN-ATTRIBUTION TESTS PASSED")
