"""Tests of "v034 stencil CNN.py" (TensorFlow on the CPU; plain script, no pytest). A-E as in the v033
test (with the v034 additions in C and E), F-G new.

A. Padding + a linear 3x3 Laplacian kernel reproduce postprocess.laplacian() on the 128 x 2048
   grid, wall rows included (physical units); ghost values of q, q - q_s, z and the constant
   log10_Ra map are as intended.
B. Same in normalised units: output = Lap(b) / sigma_b.
C. The receptive field is exactly 3x3 (periodic in y); the same seed gives the same weights.
D. Data handling on small synthetic run folders (bou.in, humid.in, setup.txt, cordin_info.h5
   and the three .npz files): frames of each block and the stride; the target of each
   configuration (Q_d / kappa with kappa from bou.in); inputs of each configuration;
   normalisation from training frames only; run weights per run and block; generator
   contents; b < 0 refused; a test-set run in TRAIN_RUNS refused; too small a gap refused;
   bad arguments refused.
E. End-to-end smoke test of main() (2 epochs): Q_c_C1 (two seeds in one call), Q_d_b, and
   Q_c_C4 (one seed per call, config written once, summary merged); outputs written; test and
   excluded runs never read; refuses to overwrite; the saved model reloads.
F. Whitening (Q_d configurations): on smooth, ill-conditioned patches the whitened patches have
   identity covariance; the model's float64 whitening layer followed by a float32 1 x 1 readout
   with the exact weights reproduces postprocess.laplacian (wall rows included) to < 1e-4; other
   channels pass their centre value; the whitening is not trainable; receptive field exactly 3x3.
G. Learning rate: DelayedReduceLROnPlateau does not reduce before start_from_epoch and reduces
   after `patience` epochs without improvement from then on; EarlyStopping(start_from_epoch)
   likewise; the learning rate is logged in every epoch.
E (v034): the whitening is stored in normalization_constants.npz and config.json, saved in the
   model, reused by later calls for the same configuration; the learning rate is in the history.
Usage (from simulations_v2/): python tests/test_v034_stencil_cnn.py "v034 stencil CNN.py"
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # exact-equality tests on the CPU
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("v034", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
pp = m.pp
rng = np.random.default_rng(0)


def phys_pad_args():
    bottom, top, d = [], [], []
    for c in m.INPUTS:
        wall = m.CHANNELS[c]["wall"]
        bottom.append(0.0 if wall is None else wall[0])
        top.append(0.0 if wall is None else wall[1])
        d.append(0.0 if wall is None else 1.0)
    return {"wall_bottom": bottom, "wall_top": top, "dirichlet": d}


def conv_model(kernel, pad_args):
    inp = tf.keras.layers.Input((m.NZ, m.NY, len(m.INPUTS)))
    x = m.StencilPad(**pad_args)(inp)
    conv = tf.keras.layers.Conv2D(kernel.shape[-1], 3, padding="valid", use_bias=False)
    out = conv(x)
    model = tf.keras.Model(inp, out)
    conv.set_weights([kernel.astype(np.float32)])
    return model


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}{args}")


# ---------- A. physical units, full v2 grid ----------
assert (m.NZ, m.NY) == (128, 2048)
m.INPUTS = ("b", "q", "q_minus_qs", "z", "log10_Ra")
NZ, NY = m.NZ, m.NY
dz = 1.0 / NZ
dy = 16.0 / NY
ib, iq, iqs, iz, ira = (m.INPUTS.index(c) for c in m.INPUTS)
C = len(m.INPUTS)
xm = (np.arange(NZ) + 0.5) * dz

b = np.linspace(0, 1, NZ)[:, None] + 0.1 * rng.normal(size=(NZ, NY))
X = np.zeros((1, NZ, NY, C))
X[0, ..., ib] = b
X[0, ..., iq] = rng.random((NZ, NY))
X[0, ..., iqs] = rng.normal(size=(NZ, NY))
X[0, ..., iz] = xm[:, None]
X[0, ..., ira] = 7.3

K = np.zeros((3, 3, C, 5))
K[0, 1, ib, 0] = K[2, 1, ib, 0] = 1 / dz ** 2          # kernel index [z, y]
K[1, 0, ib, 0] = K[1, 2, ib, 0] = 1 / dy ** 2
K[1, 1, ib, 0] = -2 / dz ** 2 - 2 / dy ** 2
K[0, 1, iq, 1] = 1.0                                  # value one row below
K[2, 1, iqs, 2] = 1.0                                 # value one row above
K[0, 1, iz, 3] = 1.0
K[2, 1, ira, 4] = 1.0
out = conv_model(K, phys_pad_args())(X.astype(np.float32)).numpy()[0]

lap_ref = pp.laplacian(b, dz, dy, 0.0, m.BETA - 1.0)
err = np.max(np.abs(out[..., 0] - lap_ref)) / np.max(np.abs(lap_ref))
assert err < 1e-5, err
assert np.allclose(out[0, :, 1], 2 * m.Q_WALL[0] - X[0, 0, :, iq], atol=1e-6)        # bottom ghost of q
qs_top = m.CHANNELS["q_minus_qs"]["wall"][1]
assert np.isclose(qs_top, -np.exp(-3.0))
assert np.allclose(out[-1, :, 2], 2 * qs_top - X[0, -1, :, iqs], atol=1e-6)          # top ghost of q - q_s
assert np.allclose(out[0, :, 3], -dz / 2, atol=1e-7)                                  # z ghost below the wall
assert np.allclose(out[1:, :, 3], xm[:-1, None], atol=1e-7)
assert np.allclose(out[-1, :, 4], 7.3)                                                # constant map repeated
print(f"A. Laplacian with ghost padding matches postprocess.laplacian (max rel err {err:.1e})")

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
m.select_config("Q_c_C1")
C = len(m.INPUTS)
pad = m.padding_args({c: (0.0, 1.0) for c in m.INPUTS})
model = m.build_stencil_cnn(pad, "rf_test", seed=7)
w2 = m.build_stencil_cnn(pad, "rf_test2", seed=7).get_weights()
assert all(np.array_equal(a, c) for a, c in zip(model.get_weights(), w2)), "same seed, different weights"
base = rng.normal(size=(1, NZ, NY, C)).astype(np.float32)
for (k, j) in [(60, 1000), (0, 0), (127, 2047)]:
    pert = base.copy(); pert[0, k, j, :] += 1.0
    diff = np.abs(model(pert).numpy() - model(base).numpy())[0, ..., 0] > 0
    rows, cols = np.nonzero(diff)
    assert set(rows) <= {k - 1, k, k + 1}, rows
    assert set(cols) <= {(j - 1) % NY, j, (j + 1) % NY}, cols
print(f"C. Receptive field is 3x3 (periodic in y); parameters: {model.count_params()}")
assert m.CONFIGS["Q_d_a"]["inputs"] == ("b", "log10_Ra") and m.CONFIGS["Q_d_b"]["inputs"] == ("b",)
assert m.CONFIGS["Q_c_C3"]["inputs"] == ("b", "q") and m.CONFIGS["Q_c_C4"]["inputs"] == ("b", "q_minus_qs")
assert all("log10_Pr" not in c["inputs"] for c in m.CONFIGS.values())
assert m.TRAIN_FRAMES == (25, 750) and m.VAL_FRAMES == (811, 1000) and m.STRIDE == 3
assert m.frame_indices(m.TRAIN_FRAMES)[-1] == 748 and m.frame_indices(m.VAL_FRAMES)[-1] == 1000
assert "Ra_2e8_Pr_1" in m.EXCLUDED_RUNS and set(m.TEST_RUNS) == {"Ra_8.7e6_Pr_1", "Ra_1.5e8_Pr_1"}
assert not set(m.TRAIN_RUNS) & (set(m.TEST_RUNS) | set(m.EXCLUDED_RUNS)) and len(m.TRAIN_RUNS) == 8
assert m.VERSION == "v034" and m.CONSTANT_LR_EPOCHS == 50 and m.WHITEN_CONFIGS == ("Q_d_a", "Q_d_b")


# ---------- F. whitening of the b patch (v034) ----------
m.select_config("Q_d_a")                                         # inputs b, log10_Ra
m.NZ, m.NY = 32, 128
nf, dzf = 6, 1.0 / 32
zf = (np.arange(m.NZ) + 0.5) * dzf
yf = (np.arange(m.NY) + 0.5) * dzf
bf = np.stack([0.9 * zf[:, None] + 0.04 * np.sin(2 * np.pi * (k + 1) * yf[None, :] / (m.NY * dzf) + k)
               * np.sin(np.pi * (k + 1) * zf[:, None]) for k in range(nf)])
bf = bf + 1e-6 * np.random.default_rng(7).normal(size=bf.shape)   # smooth + small noise: patch covariance with an
                                                                 # eigenvalue ratio ~5e11, within the real 3e9 - 5e12
mu_b, sd_b = float(bf.mean()), float(bf.std())
xf = ((bf - mu_b) / sd_b).astype(np.float32)
m.WHITEN_CELLS = 20000
wh = m.fit_whitening([{"fields": xf[..., None], "parts": {"train": slice(0, nf)}}])
lam = wh["eigenvalues"]
assert np.all(np.diff(lam) <= 0) and 1e9 < lam[0] / lam[-1] < 1e13, lam[0] / lam[-1]
rr = np.random.default_rng(m.WHITEN_SEED)
t_, i_, j_ = rr.integers(0, nf, 20000), rr.integers(1, m.NZ - 1, 20000), rr.integers(0, m.NY, 20000)
P = np.stack([xf[t_, i_ + di, (j_ + dj) % m.NY] for di in (-1, 0, 1) for dj in (-1, 0, 1)], 1).astype(np.float64)
# Cov(x~) = I up to round-off of order u lambda_1 / lambda_9 (u = 1.1e-16; as in tests/test_v033_linear_conv.py)
assert np.allclose(np.cov((P - wh["xbar"]) @ wh["W"].T, rowvar=False), np.eye(9), atol=1e-3), lam[0] / lam[-1]
pad_f = m.padding_args({"b": (mu_b, sd_b), "log10_Ra": (7.0, 0.5)})
model_w = m.build_stencil_cnn(pad_f, "whiten_test", seed=3, whiten=wh)
assert model_w.get_layer("whiten").dtype == "float64" and model_w.get_layer("stencil_pad").dtype == "float64"
assert model_w.get_layer("stencil_1x1").get_weights()[0].shape == (1, 1, 10, m.STENCIL_FILTERS)
assert len(model_w.get_layer("whiten").trainable_weights) == 0 and model_w.output.dtype == tf.float32
feat = tf.keras.Model(model_w.input, model_w.get_layer("whiten").output)
Xf = np.concatenate([xf[..., None], np.full(xf.shape + (1,), 0.6, np.float32)], -1)
xt = feat(Xf).numpy()
assert xt.dtype == np.float64 and np.allclose(xt[..., 9], 0.6, atol=1e-7)          # centre value passed through
K = np.array([0., 1., 0., 1., -4., 1., 0., 1., 0.]) / dzf ** 2                        # Lap on the normalised patch
u = np.linalg.solve(wh["W"].T, K)                                                     # W^T u = K
c0 = float(K @ wh["xbar"])
lap_ref = np.stack([pp.laplacian(xf[n].astype(np.float64), dzf, dzf, (m.B_WALL[0] - mu_b) / sd_b,
                                 (m.B_WALL[1] - mu_b) / sd_b) for n in range(nf)])
y32 = xt[..., :9].astype(np.float32) @ u.astype(np.float32) + np.float32(c0)         # as a float32 1x1 layer
err32 = float(np.max(np.abs(y32 - lap_ref)) / np.max(np.abs(lap_ref)))
assert err32 < 1e-4, err32
basew = rng.normal(size=(1, m.NZ, m.NY, 2)).astype(np.float32)
for (k, j) in [(10, 50), (0, 0), (31, 127)]:
    pert = basew.copy(); pert[0, k, j, 0] += 1.0
    diff = np.abs(model_w(pert).numpy() - model_w(basew).numpy())[0, ..., 0] > 0
    rows, cols = np.nonzero(diff)
    assert rows.size and set(rows) <= {k - 1, k, k + 1}, rows
    assert set(cols) <= {(j - 1) % m.NY, j, (j + 1) % m.NY}, cols
print(f"F. Whitening: identity covariance, float64 layer + float32 exact readout = Laplacian "
      f"(max rel err {err32:.1e}; eigenvalue ratio {lam[0] / lam[-1]:.1e}), receptive field 3x3")


# ---------- G. learning rate held, then ReduceLROnPlateau and EarlyStopping ----------
mg = tf.keras.Sequential([tf.keras.layers.Input((1,)), tf.keras.layers.Dense(1)])
mg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
rp = m.DelayedReduceLROnPlateau(start_from_epoch=3, monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7)
es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, start_from_epoch=3)
for cb in (rp, es):
    cb.set_model(mg)
    cb.on_train_begin()
lrs, stops = [], []
for epoch in range(8):
    logs = {"loss": 1.0, "val_loss": 1.0}
    rp.on_epoch_end(epoch, logs)
    es.on_epoch_end(epoch, logs)
    assert "lr" in logs or "learning_rate" in logs, logs
    lrs.append(float(tf.keras.backend.get_value(mg.optimizer.learning_rate)))
    stops.append(bool(mg.stop_training))
assert np.allclose(lrs, [1e-3] * 5 + [5e-4] * 2 + [2.5e-4]), lrs
assert stops == [False] * 6 + [True] * 2, stops
print("G. Learning rate constant before start_from_epoch, plateau reduction and early stopping after it")


# ---------- D. data handling on synthetic run folders ----------
humid_src = script.parent / "runs" / "Ra_1e6_Pr_1" / "humid.in"
m.NZ, m.NY = 8, 16
m.TRAIN_FRAMES, m.VAL_FRAMES, m.STRIDE, m.GAP_MIN = (2, 11), (17, 26), 3, 5
m.EPOCHS, m.SEEDS, m.BATCH_SIZE, m.CONSTANT_LR_EPOCHS, m.WHITEN_CELLS = 2, (1, 2), 2, 1, 2000
tmp = Path(tempfile.mkdtemp())
m.RUNS_DIR, m.OUT_DIR = tmp / "runs", tmp / "v034"
NT = 30
TR = [2, 5, 8, 11]
VA = [17, 20, 23, 26]
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
m.TRAIN_RUNS = list(train)
m.TEST_RUNS = ["Ra_8.7e6_Pr_1"]
m.EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "synthetic"}
raw = {}


def make_run(label, ra, set_name, negative_b=False):
    d = m.RUNS_DIR / label
    (d / "outputdir").mkdir(parents=True)
    (d / "bou.in").write_text(
        "NXM     NYM     NZM     NSST\n"
        f"{m.NZ}       {m.NY}      1       3\n"
        "TOUT    TFRAME  SAVE_3D\n1.0     1.0     6000.0\n"
        "ALX3   YLEN   ZLEN\n"
        f"1.0    {m.NY / m.NZ}   0.01\n"
        "RAYT    PRAT    RAYS    PRAS    FFscaleS\n"
        f"{ra:g}     1.0     1e7     1.0     0\n")
    shutil.copy(humid_src, d / "humid.in")
    (d / "setup.txt").write_text(f"set: {set_name}\n")
    (d / "postprocess.txt").write_text("synthetic\n")
    with h5py.File(d / "outputdir" / "cordin_info.h5", "w") as f:
        f["xc"] = np.arange(m.NZ + 1) / m.NZ
        f["xm"] = (np.arange(m.NZ) + 0.5) / m.NZ
        f["ym"] = (np.arange(m.NY) + 0.5) / m.NZ
    theta = rng.random((m.NZ, m.NY, NT)) + 0.01
    if negative_b:
        theta[0, 0, 20] = -0.5                       # inside the validation block
    q = rng.random((m.NZ, m.NY, NT))
    relq = rng.normal(size=(m.NZ, m.NY, NT))
    Qd = rng.normal(size=(m.NZ, m.NY, NT)) / np.sqrt(ra)
    Qc = rng.random((m.NZ, m.NY, NT))
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=relq, Q=theta)
    np.savez(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, q=q, relq=relq, Qd=Qd, Qc=Qc, ra=ra)


for label, ra in train.items():
    make_run(label, ra, "train")
make_run("Ra_8.7e6_Pr_1", 8.7e6, "test")
make_run("Ra_2e8_Pr_1", 2e8, "test")
make_run("Ra_3e6_Pr_1", 3e6, "test (held out)")
make_run("Ra_9e6_Pr_1", 9e6, "train", negative_b=True)

# frames, inputs and target of each configuration
m.select_config("Q_c_C1")
r = m.load_run("Ra_1e7_Pr_1")
assert r["frames"] == {"train": TR, "val": VA}, r["frames"]
assert r["parts"] == {"train": slice(0, 4), "val": slice(4, 8)}
assert r["fields"].shape == (8, m.NZ, m.NY, 2) and r["fields"].dtype == np.float32
assert np.allclose(r["fields"][..., 0], np.moveaxis(raw["Ra_1e7_Pr_1"]["theta"][:, :, TR + VA], -1, 0))
assert np.allclose(r["fields"][..., 1], np.moveaxis(raw["Ra_1e7_Pr_1"]["q"][:, :, TR + VA], -1, 0))
assert np.allclose(r["y"], np.moveaxis(raw["Ra_1e7_Pr_1"]["Qc"][:, :, TR + VA], -1, 0))
assert np.isclose(r["kappa"], 1e7 ** -0.5) and np.isclose(r["params"]["log10_Ra"], 7.0)
m.select_config("Q_d_b")
r = m.load_run("Ra_1e8_Pr_1")
assert r["fields"].shape[-1] == 1
assert np.allclose(r["y"], np.moveaxis(raw["Ra_1e8_Pr_1"]["Qd"][:, :, TR + VA] * np.sqrt(1e8), -1, 0), rtol=1e-6)
m.select_config("Q_c_C4")
r = m.load_run("Ra_1e6_Pr_1")
assert np.allclose(r["fields"][..., 1], np.moveaxis(raw["Ra_1e6_Pr_1"]["relq"][:, :, TR + VA], -1, 0))
m.select_config("Q_c_C1")
expect(ValueError, m.load_run, "Ra_9e6_Pr_1")                          # b < 0

# normalisation from training frames only; run weights per run and block
runs = [m.load_run(l) for l in train]
inorm, tnorm = m.fit_input_norm(runs), m.fit_target_norm(runs)
allb = np.concatenate([raw[l]["theta"][:, :, TR].ravel() for l in train])
allc = np.concatenate([raw[l]["Qc"][:, :, TR].ravel() for l in train])
assert np.isclose(inorm["b"][0], allb.mean(), rtol=1e-6) and np.isclose(inorm["b"][1], allb.std(), rtol=1e-5)
assert np.isclose(tnorm[0], allc.mean(), rtol=1e-6) and np.isclose(tnorm[1], allc.std(), rtol=1e-5)
xms = (np.arange(m.NZ) + 0.5) / m.NZ
assert np.isclose(inorm["z"][0], xms.mean()) and np.isclose(inorm["z"][1], xms.std())
w = m.run_weights(runs, tnorm)
assert sorted(w) == sorted(f"{l}/{p}" for l in train for p in ("train", "val"))
for l in train:
    for p, fr in (("train", TR), ("val", VA)):
        ref = tnorm[1] ** 2 / np.mean(raw[l]["Qc"][:, :, fr] ** 2)
        assert np.isclose(w[f"{l}/{p}"], ref, rtol=1e-5), (l, p)

# generator: block, frame and weight of every sample
for run in runs:
    m.apply_input_norm(run, inorm)
gen = m.RunSnapshots(runs, "val", tnorm, 2, shuffle=False, seed=0, weights=w)
assert len(gen.index) == 3 * len(VA)
Xb, Yb = gen[2]                                                         # samples 4, 5 -> run 1, frames 17, 20
r1 = raw["Ra_1e7_Pr_1"]
assert np.allclose(Xb[0, ..., 0], (r1["theta"][:, :, 17] - inorm["b"][0]) / inorm["b"][1], atol=1e-5)
assert np.allclose(Xb[1, ..., 1], (r1["q"][:, :, 20] - inorm["q"][0]) / inorm["q"][1], atol=1e-5)
assert np.allclose(Xb[0, ..., 2], ((xms - inorm["z"][0]) / inorm["z"][1])[:, None], atol=1e-6)
assert np.allclose(Yb[1, ..., 0], (r1["Qc"][:, :, 20] - tnorm[0]) / tnorm[1], atol=1e-5)
assert np.allclose(Yb[..., 1], w["Ra_1e7_Pr_1/val"])
gtr = m.RunSnapshots(runs, "train", tnorm, 2, shuffle=False, seed=0, weights=w)
Xt, Yt = gtr[0]                                                         # run 0, frames 2, 5
assert np.allclose(Yt[1, ..., 0], (raw["Ra_1e6_Pr_1"]["Qc"][:, :, 5] - tnorm[0]) / tnorm[1], atol=1e-5)
assert np.allclose(Yt[..., 1], w["Ra_1e6_Pr_1/train"])

yt = np.stack([np.ones((2, 4, 4)), np.stack([np.full((4, 4), 2.0), np.full((4, 4), 0.5)])], -1).astype(np.float32)
yp = np.zeros((2, 4, 4, 1), np.float32)
assert np.isclose(float(np.mean(m.weighted_mse(yt, yp))), (2.0 + 0.5) / 2)
assert np.isclose(float(np.mean(m.mse_target(yt, yp))), 1.0)

# refusals
m.TRAIN_RUNS = list(train) + ["Ra_3e6_Pr_1"]                           # its setup.txt says 'set: test'
expect(SystemExit, m.check_config, "Q_c_C1", (1,))
m.TRAIN_RUNS = list(train)
m.VAL_FRAMES = (13, 26)                                                 # gap 1 < GAP_MIN
expect(ValueError, m.check_config, "Q_c_C1", (1,))
m.VAL_FRAMES = (17, 26)
for bad in ([], ["nope"], ["Q_c_C1", "99"], ["Q_c_C1", "1", "1"], ["Q_c_C1", "x"]):
    expect(ValueError, m.parse_args, bad)
assert m.parse_args(["Q_d_a"]) == ("Q_d_a", (1, 2))
print("D. Frames, inputs, targets, normalisation, run weights, generator and refusals as intended")


# ---------- E. end-to-end smoke test of main() ----------
loaded = []
_orig_load = np.load


def spy_load(path, *a, **k):
    loaded.append(str(path))
    return _orig_load(path, *a, **k)


m.np.load = spy_load
m.main(["Q_c_C1", "1", "2"])
m.main(["Q_d_b", "1"])
m.np.load = _orig_load
for never in ("Ra_8.7e6_Pr_1", "Ra_2e8_Pr_1", "Ra_3e6_Pr_1", "Ra_9e6_Pr_1"):
    assert not any(never in p for p in loaded), f"{never} was read"

cfg = json.loads((m.OUT_DIR / "Q_c_C1" / "config.json").read_text())
assert cfg["inputs"] == ["b", "q", "z"] and cfg["target"] == "Q_c"
assert cfg["frames_used"]["Ra_1e8_Pr_1"] == {"train": TR, "val": VA}
assert cfg["blocks"] == {"train_frames": [2, 11], "val_frames": [17, 26], "stride": 3, "gap_min": 5}
assert np.isclose(cfg["run_parameters"]["Ra_1e7_Pr_1"]["kappa"], 1e7 ** -0.5)
assert np.isclose(cfg["target_norm"][1], tnorm[1])
summ = json.loads((m.OUT_DIR / "Q_c_C1" / "summary.json").read_text())
assert [s["seed"] for s in summ] == [1, 2]
for s in summ:
    out = m.OUT_DIR / "Q_c_C1" / f"seed_{s['seed']}"
    for fname in ("model.keras", "history.json", "history.csv", "loss_LOG.png", "loss_LinearScale.png"):
        assert (out / fname).is_file(), out / fname
    hist = json.loads((out / "history.json").read_text())
    assert {"loss", "val_loss", "mse_target", "val_mse_target"} <= set(hist), hist.keys()

cfg_b = json.loads((m.OUT_DIR / "Q_d_b" / "config.json").read_text())
allk = np.concatenate([(raw[l]["Qd"][:, :, TR] * np.sqrt(train[l])).ravel() for l in train])
assert cfg_b["inputs"] == ["b"] and cfg_b["target"] == "Q_d_over_kappa"
assert np.isclose(cfg_b["target_norm"][0], allk.mean(), atol=1e-5) and np.isclose(cfg_b["target_norm"][1], allk.std(), rtol=1e-5)
assert cfg["whitening"] is None and cfg["architecture"]["first_layer"] == "3x3 convolution"
assert cfg_b["whitening"] is not None and cfg_b["lr_schedule"]["constant_lr_epochs"] == 1
nb = np.load(m.OUT_DIR / "Q_d_b" / "normalization_constants.npz")
assert np.array_equal(np.array(cfg_b["whitening"]["W"]), nb["whiten_W"])
mb = tf.keras.models.load_model(m.OUT_DIR / "Q_d_b" / "seed_1" / "model.keras", compile=False)
kern, bias = mb.get_layer("whiten").get_weights()
assert kern.dtype == np.float64 and np.array_equal(kern[:, :, 0, :9], nb["whiten_W"].T.reshape(3, 3, 9))
assert np.allclose(bias[:9], -nb["whiten_W"] @ nb["whiten_xbar"], rtol=0, atol=1e-12)
hb = json.loads((m.OUT_DIR / "Q_d_b" / "seed_1" / "history.json").read_text())
assert len(hb["lr" if "lr" in hb else "learning_rate"]) == len(hb["loss"])

expect(SystemExit, m.main, ["Q_c_C1", "1"])                             # refuses to overwrite

m.OUT_DIR = tmp / "v034_single"
m.main(["Q_c_C4", "2"])
cfg_text = (m.OUT_DIR / "Q_c_C4" / "config.json").read_text()
m.main(["Q_c_C4", "1"])
cfg_new = json.loads((m.OUT_DIR / "Q_c_C4" / "config.json").read_text())
assert (m.OUT_DIR / "Q_c_C4" / "config.json").read_text() == cfg_text
assert cfg_new["inputs"] == ["b", "q_minus_qs"]
assert [s["seed"] for s in json.loads((m.OUT_DIR / "Q_c_C4" / "summary.json").read_text())] == [1, 2]
m.main(["Q_d_a", "2"])
m.main(["Q_d_a", "1"])                                                   # reuses the stored whitening
na = np.load(m.OUT_DIR / "Q_d_a" / "normalization_constants.npz")
for s_ in (1, 2):
    ka = tf.keras.models.load_model(m.OUT_DIR / "Q_d_a" / f"seed_{s_}" / "model.keras",
                                    compile=False).get_layer("whiten").get_weights()[0]
    assert np.array_equal(ka[:, :, 0, :9], na["whiten_W"].T.reshape(3, 3, 9)) and ka[1, 1, 1, 9] == 1.0

m.select_config("Q_c_C1")
reloaded = tf.keras.models.load_model(tmp / "v034" / "Q_c_C1" / "seed_1" / "model.keras")
assert reloaded(Xb).shape == (2, m.NZ, m.NY, 1)
shutil.rmtree(tmp)
print("E. End-to-end smoke test passed (test and excluded runs never read; one seed per call merges; whitening "
      "stored, saved in the model and reused)")
print("ALL v034 STENCIL CNN TESTS PASSED")
