"""Test of diagnostics/xai_figures.py (TensorFlow on the CPU; plain script, no pytest). On small grids, synthetic
fields and hand-built models:
 1. load_results inverts xai_attribution.save_data (model and control entries, seeds as integers);
 2. shares_pct = 100 x the seed mean of S_ch (exact: the control) and sums to 100; the three figures redrawn from
    the data files run (mathtext renders);
 3. condensing_jacobian: exact C4 gives gamma/tau = 20 at the centre of q - q_s and 0 elsewhere, at exactly the
    explained cells with psi_dns > 0; exact C1 gives the analytic centre values d/dq = 20, d/db = -20 alpha q_s,
    d/dz = 20 alpha beta q_s (raw units) and 0 elsewhere; a nonlinear two-channel model gives the lattice gradient
    of jacobian_attribution.cell_grad times the raw-unit scale sigma_y / sigma_ch; patch layout: a model returning
    x(i + 1, j) (one row up, i.e. z + dz) has its only non-zero entry at m = 3 (1 + 1) + (0 + 1) = 7;
 4. describe: percentiles and mean as numpy; an empty set gives n = 0 and NaN;
 5. summary text, data file (round trip), panel_value (seed mean) and the stencil figure run; refuse_existing
    stops on an existing output and passes otherwise.
Usage (from simulations_v2/): python tests/test_xai_figures.py diagnostics/xai_figures.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("xai_figures", script)
xf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xf)
xa, ja, T = xf.xa, xf.ja, xf.T
rng = np.random.default_rng(7)
NZ, NY = 10, 18                                    # as test_xai_attribution.py
T.NZ, T.NY = NZ, NY
dz = 1.0 / NZ
xm = (np.arange(NZ) + 0.5) * dz
ROWS, COLS = xa.interior_row_sets(NZ, NY)
labels = T.TRAIN_RUNS + T.TEST_RUNS
runs = {l: {"ra": float(l.split("_")[1])} for l in labels}


def entry(n):
    C = len(T.CONFIGS[n]["inputs"])
    S = rng.random(C)
    if n == "Q_c_C4":
        S = np.array([1e-5, 1.0])                  # a share below 0.1 %: the annotation branch
    return {"e0": np.float64(rng.random()), "e_ch": rng.random(C) + 0.1, "S_ch": S / S.sum(), "S_IG": rng.random(9)}


# ---------- 1. load_results ----------
res_d = {(v, n, s, l): entry(n) for v in xa.VERSIONS for n in xa.CONFIGS["qd"] for s in (42, 43) for l in labels}
ctl_d = {(n, l): entry(n) for n in xa.CONFIGS["qd"] for l in labels}
res_c = {(v, n, s, l): entry(n) for v in xa.VERSIONS for n in xa.CONFIGS["qc"] for s in (42, 43) for l in labels}
ctl_c = {(n, l): entry(n) for n in ("Q_c_C1", "Q_c_C2", "Q_c_C4") for l in labels}
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    loaded = {}
    for part, res, ctl in (("qd", res_d, ctl_d), ("qc", res_c, ctl_c)):
        xa.save_data(res, ctl, tmp / f"{part}.npz")
        r2, c2 = loaded[part] = xf.load_results(tmp / f"{part}.npz")
        assert r2.keys() == res.keys() and c2.keys() == ctl.keys()
        for src, dst in ((res, r2), (ctl, c2)):
            for k, r in src.items():
                assert r.keys() == dst[k].keys() and all(np.allclose(dst[k][key], val) for key, val in r.items()), k
    print("1. load_results inverts save_data: OK")

    # ---------- 2. shares in per cent, figures ----------
    r2, c2 = loaded["qc"]
    P = xf.shares_pct(r2, c2, "v034", "Q_c_C2", labels)
    assert np.allclose(P, [100 * np.mean([res_c[("v034", "Q_c_C2", s, l)]["S_ch"] for s in (42, 43)], axis=0) for l in labels])
    assert np.allclose(P.sum(axis=1), 100.0)
    assert np.allclose(xf.shares_pct(r2, c2, "exact", "Q_c_C1", labels), [100 * ctl_c[("Q_c_C1", l)]["S_ch"] for l in labels])
    assert xf.shares_pct(r2, c2, "exact", "Q_c_C3", labels) is None
    xf.plot_qc_shares_pct(r2, c2, runs, tmp / "a.png")
    xf.plot_qd_a_rho(*loaded["qd"], runs, tmp / "b.png")
    xf.plot_qd_a_errors(*loaded["qd"], runs, tmp / "c.png")
    assert all((tmp / f"{f}.png").stat().st_size > 10_000 for f in "abc")
print("2. shares_pct = 100 x seed mean of S_ch, sums to 100; the three redrawn figures run: OK")

# ---------- 3. Jacobian at the condensing cells ----------
b = (np.linspace(0, 1, NZ)[None, :, None] + 0.05 * rng.normal(size=(3, NZ, NY))).astype(np.float32)
tnorm = (0.01, 0.05)
inorm_c4 = {"b": (0.4, 0.3), "q_minus_qs": (-0.01, 0.02)}
qc4 = xa.ExactQc(["b", "q_minus_qs"], inorm_c4, tnorm, 20.0, 3.0, 2.0)
s = 0.02 * rng.normal(size=(3, NZ, NY))
Xc = np.stack([((b - 0.4) / 0.3).astype(np.float32), ((s + 0.01) / 0.02).astype(np.float32)], axis=-1)
psi4 = 20.0 * np.maximum(-0.01 + 0.02 * Xc[..., 1].astype(np.float64), 0.0)
J4 = xf.condensing_jacobian(qc4, Xc, {"input": inorm_c4, "target": tnorm, "fac": 1.0}, ["b", "q_minus_qs"], psi4)
n_wet = sum(int((xa.cells(psi4, r, COLS) > 0).sum()) for r in ROWS)
expect = np.zeros((2, 9))
expect[1, 4] = 20.0
assert len(J4) == n_wet > 0 and np.allclose(J4, expect[None], rtol=0, atol=1e-4), (len(J4), n_wet)

inorm_c1 = {"b": (0.4, 0.3), "q": (0.5, 0.2), "z": (0.5, 0.29)}
qc1 = xa.ExactQc(["b", "q", "z"], inorm_c1, tnorm, 20.0, 3.0, 2.0)
q = np.exp(3.0 * (b - 2.0 * xm[None, :, None])) + 0.01 * rng.normal(size=b.shape)
Xc1 = np.stack([((b - 0.4) / 0.3).astype(np.float32), ((q - 0.5) / 0.2).astype(np.float32),
                np.broadcast_to(((xm - 0.5) / 0.29).astype(np.float32)[None, :, None], b.shape)], axis=-1)
raw = [m + sd * Xc1[..., i].astype(np.float64) for i, (m, sd) in enumerate(inorm_c1.values())]
qs = np.exp(3.0 * (raw[0] - 2.0 * raw[2]))
psi1 = 20.0 * np.maximum(raw[1] - qs, 0.0)
J1 = xf.condensing_jacobian(qc1, Xc1, {"input": inorm_c1, "target": tnorm, "fac": 1.0}, ["b", "q", "z"], psi1)
qs_c = np.concatenate([xa.cells(qs, r, COLS).reshape(-1)[xa.cells(psi1, r, COLS).reshape(-1) > 0] for r in ROWS])
assert len(J1) == len(qs_c) > 0
assert np.allclose(J1[:, 1, 4], 20.0, rtol=1e-5)
assert np.allclose(J1[:, 0, 4], -60.0 * qs_c, rtol=1e-4, atol=1e-6) and np.allclose(J1[:, 2, 4], 120.0 * qs_c, rtol=1e-4, atol=1e-6)
off = np.ones(9, bool)
off[4] = False
assert np.all(J1[:, :, off] == 0.0)


def seeded(k):
    return {"kernel_initializer": tf.keras.initializers.GlorotUniform(seed=700 + k),
            "bias_initializer": tf.keras.initializers.RandomUniform(-0.3, 0.3, seed=800 + k)}


inp = tf.keras.layers.Input((NZ, NY, 2))
h = T.StencilPad(wall_bottom=[0.0, 0.0], wall_top=[1.0, 0.0], dirichlet=[1.0, 0.0])(inp)
h = tf.keras.layers.Conv2D(8, 3, padding="valid", activation="relu", **seeded(0))(h)
nl = tf.keras.Model(inp, tf.keras.layers.Conv2D(1, 1, **seeded(1))(h))
X2 = rng.random((2, NZ, NY, 2)).astype(np.float32)
psi_r = rng.normal(size=(2, NZ, NY))
norms_n = {"input": {"u": (0.0, 2.0), "w": (1.0, 0.5)}, "target": (0.3, 3.0), "fac": 1.0}
Jn = xf.condensing_jacobian(nl, X2, norms_n, ["u", "w"], psi_r)
lat = np.concatenate([xa.patches_of(ja.cell_grad(nl, X2, r, COLS), r, COLS).reshape(-1, 2, 9)[xa.cells(psi_r, r, COLS).reshape(-1) > 0]
                      for r in ROWS])
assert len(Jn) == len(lat) > 0 and np.allclose(Jn, lat * np.array([1.5, 6.0])[None, :, None], rtol=1e-5, atol=1e-6)
inp1 = tf.keras.layers.Input((NZ, NY, 1))
conv = tf.keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
up = tf.keras.Model(inp1, conv(T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0])(inp1)))
K = np.zeros((3, 3, 1, 1), np.float32)
K[2, 1] = 1.0
conv.set_weights([K])
X1 = rng.random((1, NZ, NY, 1)).astype(np.float32)
assert np.allclose(ja.predict(up, X1)[0, 1:-1], X1[0, 2:, :, 0])          # output(i, j) = x(i + 1, j)
Ju = xf.condensing_jacobian(up, X1, {"input": {"u": (0.0, 1.0)}, "target": (0.0, 1.0), "fac": 1.0}, ["u"],
                            np.ones((1, NZ, NY)))
e7 = np.zeros(9)
e7[7] = 1.0
assert len(Ju) > 0 and np.allclose(Ju[:, 0], e7[None], rtol=0, atol=1e-7)
print(f"3. condensing_jacobian: exact C4 (20 at the centre, {n_wet} cells), exact C1 analytic, nonlinear model = "
      "lattice gradient x scale, patch layout (x(i + 1, j) at m = 7): OK")

# ---------- 4. describe ----------
A = rng.normal(size=(51, 2, 9))
d = xf.describe(A)
assert d["n"] == 51 and np.allclose(d["median"], np.median(A, axis=0)) and np.allclose(d["mean"], A.mean(axis=0))
assert np.allclose(d["p5"], np.percentile(A, 5, axis=0)) and np.allclose(d["p95"], np.percentile(A, 95, axis=0))
e = xf.describe(np.zeros((0, 2, 9)))
assert e["n"] == 0 and np.all(np.isnan(e["median"])) and e["median"].shape == (2, 9)
print("4. describe: percentiles, mean; empty set: OK")

# ---------- 5. summary, data file, stencil figure, refusal ----------
stats = {}
for v in xa.VERSIONS + ("exact",):
    for n in xa.CONFIGS["qc"]:
        if v == "exact" and n == "Q_c_C3":
            continue
        C = len(T.CONFIGS[n]["inputs"])
        for s_ in ((0,) if v == "exact" else (42, 43)):
            for l in labels + ["train"]:
                stats[(v, n, s_, l)] = xf.describe(rng.normal(size=(30, C, 9)))
text = xf.summary_jacobian(stats, labels)
assert "=== exact Q_c_C4" in text and "=== v033 Q_c_C1 seed 43" in text and "  train " in text
assert np.allclose(xf.panel_value(stats, "v034", "Q_c_C4"),
                   np.mean([stats[("v034", "Q_c_C4", s_, "train")]["median"] for s_ in (42, 43)], axis=0))
assert xf.panel_value(stats, "exact", "Q_c_C3") is None
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    xf.save_stats(stats, tmp / "j.npz")
    with np.load(tmp / "j.npz") as f:
        assert np.allclose(f["v034__Q_c_C4__42__train__median"], stats[("v034", "Q_c_C4", 42, "train")]["median"])
        assert int(f["exact__Q_c_C1__0__Ra_1e6_Pr_1__n"]) == 30
    xf.plot_stencils(stats, tmp / "s.png")
    assert (tmp / "s.png").stat().st_size > 10_000
    (tmp / "x.png").write_bytes(b"x")
    xf.refuse_existing([tmp / "none.png"])
    try:
        xf.refuse_existing([tmp / "none.png", tmp / "x.png"])
        raise AssertionError("an existing output was not refused")
    except SystemExit:
        pass
print("5. summary, data file, panel_value, stencil figure, refusal to overwrite: OK")
print("ALL XAI-FIGURES TESTS PASSED")
