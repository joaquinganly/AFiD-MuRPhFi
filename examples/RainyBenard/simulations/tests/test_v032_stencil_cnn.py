"""Tests of "v032 stencil CNN.py" on synthetic data (TensorFlow CPU).

A. Padding + a linear 3x3 Laplacian kernel reproduce laplacian() of
   "Qc and Qd only.py" exactly, wall rows included (physical units), and
   the ghost values of q, z and the constant maps are as intended.
B. Same in normalised units: output = Lap(b) / sigma_b.
C. The model's receptive field is exactly 3x3 (periodic in y).
D. End-to-end smoke test of main() on small synthetic runs (2 epochs):
   outputs written, held-out data never loaded, generator contents correct,
   model reloads with the custom layer.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

script = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("v031", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rng = np.random.default_rng(0)


def laplacian_original(field_n, dz, dy, b_wall_bottom=0.0, b_wall_top=1.0):
    d2_dz2 = np.zeros_like(field_n)
    d2_dz2[1:-1, :] = (field_n[2:, :] - 2.0 * field_n[1:-1, :] + field_n[:-2, :]) / dz**2
    d2_dz2[0, :] = (field_n[1, :] - 3.0 * field_n[0, :] + 2.0 * b_wall_bottom) / dz**2
    d2_dz2[-1, :] = (field_n[-2, :] - 3.0 * field_n[-1, :] + 2.0 * b_wall_top) / dz**2
    d2_dy2 = (np.roll(field_n, -1, axis=1) - 2.0 * field_n + np.roll(field_n, 1, axis=1)) / dy**2
    return d2_dz2 + d2_dy2


def phys_pad_args():
    bottom, top, d = [], [], []
    for c in m.INPUTS:
        wall = m.CHANNELS[c]["wall"]
        bottom.append(0.0 if wall is None else wall[0])
        top.append(0.0 if wall is None else wall[1])
        d.append(0.0 if wall is None else 1.0)
    return {"wall_bottom": bottom, "wall_top": top, "dirichlet": d}


def conv_model(kernel, pad_args):
    nz, ny, c = m.NZ, m.NY, len(m.INPUTS)
    inp = tf.keras.layers.Input((nz, ny, c))
    x = m.StencilPad(**pad_args)(inp)
    conv = tf.keras.layers.Conv2D(kernel.shape[-1], 3, padding="valid", use_bias=False)
    out = conv(x)
    model = tf.keras.Model(inp, out)
    conv.set_weights([kernel.astype(np.float32)])
    return model


# ---------- A. physical units, full grid ----------
NZ, NY = m.NZ, m.NY
dz = dy = 1.0 / 64
ib, iq, iz = m.INPUTS.index("b"), m.INPUTS.index("q"), m.INPUTS.index("z")
ira = m.INPUTS.index("log10_Ra")
C = len(m.INPUTS)
xm = (np.arange(NZ) + 0.5) * dz

b = np.linspace(0, 1, NZ)[:, None] + 0.1 * rng.normal(size=(NZ, NY))
X = np.zeros((1, NZ, NY, C))
X[0, ..., ib] = b
X[0, ..., iq] = rng.random((NZ, NY))
X[0, ..., iz] = xm[:, None]
X[0, ..., ira] = 7.3

K = np.zeros((3, 3, C, 4))
K[0, 1, ib, 0] = K[2, 1, ib, 0] = 1 / dz**2          # kernel index [z, y]
K[1, 0, ib, 0] = K[1, 2, ib, 0] = 1 / dy**2
K[1, 1, ib, 0] = -2 / dz**2 - 2 / dy**2
K[0, 1, iq, 1] = 1.0                                  # value one row below
K[0, 1, iz, 2] = 1.0
K[2, 1, ira, 3] = 1.0                                 # value one row above
out = conv_model(K, phys_pad_args())(X.astype(np.float32)).numpy()[0]

lap_ref = laplacian_original(b, dz, dy, 0.0, m.BETA - 1.0)
err = np.max(np.abs(out[..., 0] - lap_ref)) / np.max(np.abs(lap_ref))
assert err < 1e-5, err
assert np.allclose(out[0, :, 1], 2 * m.Q_WALL[0] - X[0, 0, :, iq], atol=1e-6)  # bottom ghost of q
assert np.allclose(out[0, :, 2], -dz / 2, atol=1e-7)                          # z ghost centre
assert np.allclose(out[1:, :, 2], xm[:-1, None], atol=1e-7)
assert np.allclose(out[-1, :, 3], 7.3)                                         # constant map repeated
print(f"A. Laplacian with ghost padding matches post-processing (max rel err {err:.1e})")

# ---------- B. normalised units ----------
mu, sigma = 0.37, 0.21
norm = {c: (0.0, 1.0) for c in m.INPUTS}
norm["b"] = (mu, sigma)
Xn = X.copy()
Xn[0, ..., ib] = (b - mu) / sigma
out_n = conv_model(K[..., :1], m.padding_args(norm))(Xn.astype(np.float32)).numpy()[0, ..., 0]
err_n = np.max(np.abs(out_n - lap_ref / sigma)) / np.max(np.abs(lap_ref / sigma))
assert err_n < 1e-5, err_n
print(f"B. Same in normalised units: Lap(b)/sigma (max rel err {err_n:.1e})")

# ---------- C. receptive field ----------
pad = m.padding_args({c: (0.0, 1.0) for c in m.INPUTS})
model = m.build_stencil_cnn(pad, "rf_test", seed=7)
w1 = [w.copy() for w in model.get_weights()]
w2 = m.build_stencil_cnn(pad, "rf_test2", seed=7).get_weights()
assert all(np.array_equal(a, b) for a, b in zip(w1, w2)), "same seed, different weights"
base = rng.normal(size=(1, NZ, NY, C)).astype(np.float32)
for (k, j) in [(30, 500), (0, 0), (63, 1023)]:
    pert = base.copy(); pert[0, k, j, :] += 1.0
    diff = np.abs(model(pert).numpy() - model(base).numpy())[0, ..., 0] > 0
    rows, cols = np.nonzero(diff)
    assert set(rows) <= {k - 1, k, k + 1}, rows
    assert set(cols) <= {(j - 1) % NY, j, (j + 1) % NY}, cols
print(f"C. Receptive field is 3x3 (periodic in y); parameters: {model.count_params()}")


# ---------- D. end-to-end smoke test (v032: dropped runs, frame 0 dropped, run weights) ----------
m.NZ, m.NY = 8, 16
m.EPOCHS, m.SEEDS, m.BATCH_SIZE = 2, (1, 2), 2
tmp = Path(tempfile.mkdtemp())
m.BASE_DATA_DIR, m.OUT_DIR = tmp, tmp / "v032"
train = ["Pr_05_Ra_1e5", "Ra_1e5", "Pr_2_Ra_1e6"]
val = ["Ra_1e6"]
held = ["Ra_5e6"]
dropped = {"Ra_1e8": "test: b < 0"}
m.ALL_RUNS = train + val + held + list(dropped)
m.TRAIN_RUNS, m.VAL_RUNS, m.HELD_OUT_RUNS, m.DROPPED_RUNS = train, val, held, dropped
m.USED_RUNS = train + val + held
xm_s = (np.arange(m.NZ) + 0.5) / m.NZ
raw = {}
for i, label in enumerate(m.ALL_RUNS):
    d = tmp / label
    nt_f = {"Ra_1e5": 7, "Ra_1e8": 5}.get(label, 9)   # Ra_1e5 fewest of the used runs
    (d / "outputdir").mkdir(parents=True)
    if label not in held:
        with h5py.File((d if i % 2 else d / "outputdir") / "cordin_info.h5", "w") as f:
            f["xm"] = xm_s
    theta = rng.random((m.NZ, m.NY, nt_f))
    if label in dropped:
        theta -= 0.5                                   # unphysical b
    relq = rng.normal(size=(m.NZ, m.NY, nt_f)); q = rng.normal(size=(m.NZ, m.NY, nt_f))
    Qd = rng.normal(size=(m.NZ, m.NY, 8)) * (i + 1); Qc = rng.random((m.NZ, m.NY, 8))
    np.savez_compressed(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=relq, Q=theta)
    np.savez_compressed(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, q=q, Qd=Qd, Qc=Qc)

loaded_paths = []
_orig_load = np.load
def spy_load(path, *a, **k):
    loaded_paths.append(str(path))
    return _orig_load(path, *a, **k)
m.np.load = spy_load
m.main([])
m.np.load = _orig_load

assert not any(("Ra_5e6" in p) or ("Ra_1e8" in p) for p in loaded_paths), "held-out or dropped run loaded"
summary = []
for t in ("Q_d", "Q_c", "Q"):
    cfg = json.loads((m.OUT_DIR / t / "config.json").read_text())
    assert cfg["nt_common"] == 7 and cfg["first_frame"] == 1 and cfg["target"] == t, cfg
    assert cfg["dropped_runs"] == dropped and sorted(cfg["run_weights"]) == sorted(train + val)
    assert (m.OUT_DIR / t / "summary.txt").is_file()
    summary += json.loads((m.OUT_DIR / t / "summary.json").read_text())
assert len(summary) == 6 and {s["target"] for s in summary} == {"Q_d", "Q_c", "Q"}
for s in summary:
    out = m.OUT_DIR / s["target"] / f"seed_{s['seed']}"
    for name in ("model.keras", "history.json", "history.csv", "loss_LOG.png", "loss_LinearScale.png"):
        assert (out / name).is_file(), out / name
    hist = json.loads((out / "history.json").read_text())
    assert {"loss", "val_loss", "mse_target", "val_mse_target"} <= set(hist), hist.keys()

# run weights = sigma^2 / mean(y^2) over frames 1..6, from the raw arrays, every target
def raw_target(l, t):
    qd, qc = raw[l]["Qd"][:, :, 1:7], raw[l]["Qc"][:, :, 1:7]
    return {"Q_d": qd, "Q_c": qc, "Q": qd + qc}[t]
for t in ("Q_d", "Q_c", "Q"):
    cfg = json.loads((m.OUT_DIR / t / "config.json").read_text())
    sd = cfg["target_norm"][1]
    for l in train + val:
        assert np.isclose(cfg["run_weights"][l], sd ** 2 / np.mean(raw_target(l, t) ** 2), rtol=1e-5), (t, l)
    allT = np.concatenate([raw_target(l, t).ravel() for l in train])
    assert np.isclose(cfg["target_norm"][0], allT.mean()) and np.isclose(sd, allT.std()), t

# refuses to overwrite an existing seed; rejects bad arguments; dropped data refused
try:
    m.main(["Q_c"]); raise AssertionError("expected SystemExit")
except SystemExit:
    pass
for bad in (["Qx"], ["Q_d", "99"], ["Q_d", "Q_d"], ["1", "1"]):
    try:
        m.parse_args(bad); raise AssertionError(f"expected ValueError for {bad}")
    except ValueError:
        pass
try:
    m.load_run("Ra_1e8", 5); raise AssertionError("expected ValueError for b < 0")
except ValueError:
    pass

# one seed per call: config written once, summary merged
m.OUT_DIR = tmp / "v032_single"
m.main(["Q_c", "2"])
cfg_text = (m.OUT_DIR / "Q_c" / "config.json").read_text()
m.main(["Q_c", "1"])
assert (m.OUT_DIR / "Q_c" / "config.json").read_text() == cfg_text
assert [s["seed"] for s in json.loads((m.OUT_DIR / "Q_c" / "summary.json").read_text())] == [1, 2]

# generator: sample t is DNS frame t + 1; second target channel = run weight
runs = [m.load_run(l, 7) for l in train]
inorm, tnorm = m.fit_input_norm(runs), m.fit_target_norm(runs, ("Q_d",))
for r in runs:
    m.apply_input_norm(r, inorm)
w = m.run_weights(runs, "Q_d", tnorm["Q_d"])
gen = m.RunSnapshots(runs, "Q_d", tnorm["Q_d"], 2, shuffle=False, seed=0, weights=w)
Xb, Yb = gen[3]                                   # samples 6, 7 -> run 1 (Ra_1e5), t = 0, 1
r1 = raw["Ra_1e5"]
assert np.allclose(Xb[0, ..., ib], (r1["theta"][:, :, 1] - inorm["b"][0]) / inorm["b"][1], atol=1e-5)
assert np.allclose(Yb[1, ..., 0], (r1["Qd"][:, :, 2] - tnorm["Q_d"][0]) / tnorm["Q_d"][1], atol=1e-5)
assert np.allclose(Yb[..., 1], w["Ra_1e5"])
assert len(gen.index) == 3 * 6

# weighted_mse on a hand-made batch
yt = np.stack([np.ones((2, 4, 4)), np.stack([np.full((4, 4), 2.0), np.full((4, 4), 0.5)])], -1).astype(np.float32)
yp = np.zeros((2, 4, 4, 1), np.float32)
assert np.isclose(float(np.mean(m.weighted_mse(yt, yp))), (2.0 + 0.5) / 2)
assert np.isclose(float(np.mean(m.mse_target(yt, yp))), 1.0)

reloaded = tf.keras.models.load_model(tmp / "v032" / "Q" / "seed_1" / "model.keras")
assert reloaded(Xb).shape == (2, m.NZ, m.NY, 1)
print("D. End-to-end smoke test passed (dropped and held-out runs never loaded, frame 0 dropped, run weights)")
print("ALL STENCIL CNN TESTS PASSED")
