"""Tests of "v035 stencil CNN.py" (TensorFlow on the CPU; plain script, no pytest). v035 is "v034 stencil CNN.py"
with one configuration, Q_a (b, q - q_s, log10_Ra -> Q = Q_d + Q_c), and a first layer that also passes the 3 x 3
patch of q - q_s; tests/test_v034_stencil_cnn.py covers the unchanged code (padding, loss, generator, schedule).

A. Settings: VERSION, CONFIGS, WHITEN_CONFIGS, PATCH_CHANNELS; every other setting equals v034's.
B. First layer (fixed, float64, not trainable), on smooth ill-conditioned b: outputs 0..8 = W (x - xbar) of the
   normalised b patch; 9..17 = the normalised q - q_s patch in the order m = 3 (di + 1) + (dj + 1), periodic in y,
   ghost rows 2 wall - edge at both walls; 18 = the centre value of log10_Ra. A float32 readout with the exact
   weights gives postprocess.laplacian(b) (wall rows included), so the exact
   Q = kappa(Ra) Lap b + (gamma/tau) max(q - q_s, 0) is a function of the layer's outputs.
C. Receptive field exactly 3 x 3 for each input channel (periodic in y); same seed, same weights.
D. Data on synthetic run folders: inputs b, rel_q (q - q_s), log10 Ra; target Q_d + Q_c; run weights
   sigma^2 / mean(Q^2) per run and block.
E. End-to-end smoke test of main() for Q_a, one seed per call: config.json and the whitening written once and
   reused; the saved model reloads with the stored first layer; test and excluded runs never read; refuses to
   overwrite.
Usage (from simulations_v2/): python tests/test_v035_stencil_cnn.py "v035 stencil CNN.py"
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
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


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m = load_module("v035", script)
m34 = load_module("v034", script.parent / "v034 stencil CNN.py")
pp = m.pp
rng = np.random.default_rng(0)


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}{args}")


# ---------- A. settings ----------
assert m.VERSION == "v035" and m.OUT_DIR.name == "v035"
assert m.CONFIGS == {"Q_a": {"inputs": ("b", "q_minus_qs", "log10_Ra"), "target": "Q"}}
assert m.WHITEN_CONFIGS == ("Q_a",) and m.PATCH_CHANNELS == ("q_minus_qs",)
assert m.TARGET_KEYS["Q"] == ("Q_d", "Q_c")
same = ("SEEDS", "NZ", "NY", "TRAIN_FRAMES", "VAL_FRAMES", "STRIDE", "GAP_MIN", "B_NEG_TOL", "RUN_WEIGHTING", "ALPHA",
        "BETA", "B_WALL", "Q_WALL", "QS_WALL", "Z_WALL", "CHANNELS", "STENCIL_FILTERS", "POINTWISE_FILTERS", "ACTIVATION",
        "BATCH_SIZE", "EPOCHS", "LEARNING_RATE", "CLIPNORM", "LR_PLATEAU", "EARLY_STOP_PATIENCE", "CONSTANT_LR_EPOCHS",
        "WHITEN_CELLS", "WHITEN_SEED", "USE_TF32", "TRAIN_RUNS", "TEST_RUNS", "EXCLUDED_RUNS")
for name in same:
    assert getattr(m, name) == getattr(m34, name), name
print("A. One configuration Q_a (b, q - q_s, log10 Ra -> Q); all other settings as v034")


# ---------- B. first layer ----------
m.select_config("Q_a")
NZf, NYf, nf, dzf = 32, 128, 6, 1.0 / 32
m.NZ, m.NY = NZf, NYf
zf = (np.arange(NZf) + 0.5) * dzf
yf = (np.arange(NYf) + 0.5) * dzf
bf = np.stack([0.9 * zf[:, None] + 0.04 * np.sin(2 * np.pi * (k + 1) * yf[None, :] / (NYf * dzf) + k)
               * np.sin(np.pi * (k + 1) * zf[:, None]) for k in range(nf)])
bf = bf + 1e-6 * np.random.default_rng(7).normal(size=bf.shape)     # eigenvalue ratio ~5e11 (as the v034 test)
mu_b, sd_b = float(bf.mean()), float(bf.std())
xb = ((bf - mu_b) / sd_b).astype(np.float32)
sf = rng.normal(size=bf.shape).astype(np.float32)                    # any values: the layer passes them
norm = {"b": (mu_b, sd_b), "q_minus_qs": (-0.05, 0.04), "log10_Ra": (7.1, 0.6)}
m.WHITEN_CELLS = 20000
wh = m.fit_whitening([{"fields": np.stack([xb, sf], -1), "parts": {"train": slice(0, nf)}}])
lam = wh["eigenvalues"]
assert 1e9 < lam[0] / lam[-1] < 1e13, lam[0] / lam[-1]
pad = m.padding_args(norm)
model = m.build_stencil_cnn(pad, "first_layer_test", seed=3, whiten=wh)
lw = model.get_layer("whiten")
assert lw.dtype == "float64" and len(lw.trainable_weights) == 0 and model.output.dtype == tf.float32
assert model.get_layer("stencil_1x1").get_weights()[0].shape == (1, 1, 19, m.STENCIL_FILTERS)
X = np.stack([xb, sf, np.full(xb.shape, 0.6, np.float32)], -1)
xt = tf.keras.Model(model.input, lw.output)(X).numpy()
assert xt.dtype == np.float64 and xt.shape == (nf, NZf, NYf, 19)


def ghost_pad(f, wb, wt):
    """(n, NZ, NY) -> (n, NZ + 2, NY + 2): ghost rows 2 wall - edge, periodic in y (StencilPad, independently)."""
    g = np.concatenate([2 * wb - f[:, :1], f, 2 * wt - f[:, -1:]], axis=1)
    return np.concatenate([g[:, :, -1:], g, g[:, :, :1]], axis=2)


ib, isq = 0, 1
bp = ghost_pad(xb.astype(np.float64), pad["wall_bottom"][ib], pad["wall_top"][ib])
sp = ghost_pad(sf.astype(np.float64), pad["wall_bottom"][isq], pad["wall_top"][isq])
offs = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
Pb = np.stack([bp[:, 1 + di:1 + di + NZf, 1 + dj:1 + dj + NYf] for di, dj in offs], -1)
Ps = np.stack([sp[:, 1 + di:1 + di + NZf, 1 + dj:1 + dj + NYf] for di, dj in offs], -1)
ref_w = (Pb - wh["xbar"]) @ wh["W"].T
err_w = float(np.max(np.abs(xt[..., :9] - ref_w)) / np.max(np.abs(ref_w)))
assert err_w < 1e-8, err_w
err_s = float(np.max(np.abs(xt[..., 9:18] - Ps)))                     # ghosts are computed in float64
assert err_s < 1e-12, ("q - q_s patch layout", err_s)
assert np.allclose(xt[..., 13], sf, atol=0) and np.allclose(xt[..., 18], 0.6, atol=1e-7)
for row, wall in ((0, "wall_bottom"), (NZf - 1, "wall_top")):          # the ghost row of q - q_s at both walls
    di = -1 if row == 0 else 1
    m_ = 3 * (di + 1) + 1
    assert np.allclose(xt[:, row, :, 9 + m_], 2 * pad[wall][isq] - sf[:, row, :], atol=1e-6), wall
K = np.array([0., 1., 0., 1., -4., 1., 0., 1., 0.]) / dzf ** 2           # Lap on the normalised patch
u = np.linalg.solve(wh["W"].T, K)
c0 = float(K @ wh["xbar"])
lap_ref = np.stack([pp.laplacian(xb[n].astype(np.float64), dzf, dzf, (m.B_WALL[0] - mu_b) / sd_b,
                                 (m.B_WALL[1] - mu_b) / sd_b) for n in range(nf)])
y32 = xt[..., :9].astype(np.float32) @ u.astype(np.float32) + np.float32(c0)
err32 = float(np.max(np.abs(y32 - lap_ref)) / np.max(np.abs(lap_ref)))
assert err32 < 1e-4, err32
print(f"B. First layer: whitened b (rel err {err_w:.1e}), q - q_s patch in order m = 3(di+1)+(dj+1) with ghost rows "
      f"and periodic y (exact), log10 Ra centre; exact readout = Laplacian (rel err {err32:.1e}; eigenvalue ratio "
      f"{lam[0] / lam[-1]:.1e})")


# ---------- C. receptive field ----------
w2 = m.build_stencil_cnn(pad, "first_layer_test2", seed=3, whiten=wh).get_weights()
assert all(np.array_equal(a, c) for a, c in zip(model.get_weights(), w2)), "same seed, different weights"
base = rng.normal(size=(1, NZf, NYf, 3)).astype(np.float32)
for ch in range(3):
    for (k, j) in [(10, 50), (0, 0), (31, 127)]:
        pert = base.copy(); pert[0, k, j, ch] += 1.0
        diff = np.abs(model(pert).numpy() - model(base).numpy())[0, ..., 0] > 0
        rows, cols = np.nonzero(diff)
        assert rows.size and set(rows) <= {k - 1, k, k + 1}, (ch, rows)
        assert set(cols) <= {(j - 1) % NYf, j, (j + 1) % NYf}, (ch, cols)
print("C. Receptive field exactly 3x3 for b, q - q_s and log10 Ra (periodic in y); same seed, same weights")


# ---------- D. data handling on synthetic run folders ----------
humid_src = script.parent / "runs" / "Ra_1e6_Pr_1" / "humid.in"
m.NZ, m.NY = 8, 16
m.TRAIN_FRAMES, m.VAL_FRAMES, m.STRIDE, m.GAP_MIN = (2, 11), (17, 26), 3, 5
m.EPOCHS, m.SEEDS, m.BATCH_SIZE, m.CONSTANT_LR_EPOCHS, m.WHITEN_CELLS = 2, (1, 2), 2, 1, 2000
tmp = Path(tempfile.mkdtemp())
m.RUNS_DIR, m.OUT_DIR = tmp / "runs", tmp / "v035"
NT, TR, VA = 30, [2, 5, 8, 11], [17, 20, 23, 26]
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
m.TRAIN_RUNS, m.TEST_RUNS, m.EXCLUDED_RUNS = list(train), ["Ra_8.7e6_Pr_1"], {"Ra_2e8_Pr_1": "synthetic"}
raw = {}


def make_run(label, ra, set_name):
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
    q = rng.random((m.NZ, m.NY, NT))
    relq = rng.normal(size=(m.NZ, m.NY, NT))
    Qd = rng.normal(size=(m.NZ, m.NY, NT)) / np.sqrt(ra)
    Qc = rng.random((m.NZ, m.NY, NT))
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=relq, Q=theta + 99.0)
    np.savez(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, relq=relq, Qd=Qd, Qc=Qc)


for label, ra in train.items():
    make_run(label, ra, "train")
make_run("Ra_8.7e6_Pr_1", 8.7e6, "test")
make_run("Ra_2e8_Pr_1", 2e8, "test")

m.select_config("Q_a")
assert m.field_channels() == ["b", "q_minus_qs"]
runs = [m.load_run(l) for l in train]
for r in runs:
    rw = raw[r["label"]]
    assert r["frames"] == {"train": TR, "val": VA}
    assert np.allclose(r["fields"][..., 0], np.moveaxis(rw["theta"][:, :, TR + VA], -1, 0))
    assert np.allclose(r["fields"][..., 1], np.moveaxis(rw["relq"][:, :, TR + VA], -1, 0))
    assert np.allclose(r["y"], np.moveaxis((rw["Qd"] + rw["Qc"])[:, :, TR + VA], -1, 0), rtol=1e-6, atol=1e-7)
    assert np.isclose(r["params"]["log10_Ra"], np.log10(train[r["label"]]))
tnorm = m.fit_target_norm(runs)
wts = m.run_weights(runs, tnorm)
for l in train:
    for p, fr in (("train", TR), ("val", VA)):
        ref = tnorm[1] ** 2 / np.mean((raw[l]["Qd"] + raw[l]["Qc"])[:, :, fr] ** 2)
        assert np.isclose(wts[f"{l}/{p}"], ref, rtol=1e-5), (l, p)
print("D. Inputs b, q - q_s, log10 Ra; target Q_d + Q_c (not the stored finite-difference Q); run weights per block")


# ---------- E. end-to-end smoke test of main() ----------
loaded = []
_orig_load = np.load


def spy_load(path, *a, **k):
    loaded.append(str(path))
    return _orig_load(path, *a, **k)


m.np.load = spy_load
m.main(["Q_a", "2"])
cfg_text = (m.OUT_DIR / "Q_a" / "config.json").read_text()
m.main(["Q_a", "1"])                                                  # reuses the stored whitening
m.np.load = _orig_load
for never in ("Ra_8.7e6_Pr_1", "Ra_2e8_Pr_1"):
    assert not any(never in p for p in loaded), f"{never} was read"
assert (m.OUT_DIR / "Q_a" / "config.json").read_text() == cfg_text
cfg = json.loads(cfg_text)
assert cfg["version"] == "v035" and cfg["inputs"] == ["b", "q_minus_qs", "log10_Ra"] and cfg["target"] == "Q"
assert cfg["target_definition"].startswith("Q_d + Q_c") and cfg["architecture"]["patch_channels"] == ["q_minus_qs"]
assert [s["seed"] for s in json.loads((m.OUT_DIR / "Q_a" / "summary.json").read_text())] == [1, 2]
na = np.load(m.OUT_DIR / "Q_a" / "normalization_constants.npz")
assert np.array_equal(np.array(cfg["whitening"]["W"]), na["whiten_W"])
for s_ in (1, 2):
    out = m.OUT_DIR / "Q_a" / f"seed_{s_}"
    for fname in ("model.keras", "history.json", "history.csv", "loss_LOG.png", "loss_LinearScale.png"):
        assert (out / fname).is_file(), out / fname
    mod = tf.keras.models.load_model(out / "model.keras", compile=False)
    kern, bias = mod.get_layer("whiten").get_weights()
    assert kern.shape == (3, 3, 3, 19) and kern.dtype == np.float64
    assert np.array_equal(kern[:, :, 0, :9], na["whiten_W"].T.reshape(3, 3, 9))
    assert np.allclose(bias[:9], -na["whiten_W"] @ na["whiten_xbar"], rtol=0, atol=1e-12) and not bias[9:].any()
    assert np.array_equal(kern[:, :, 1, 9:18].reshape(9, 9), np.eye(9)) and not kern[:, :, 1, :9].any()
    assert kern[1, 1, 2, 18] == 1.0 and np.count_nonzero(kern[:, :, 2, :]) == 1 and not kern[:, :, 0, 9:].any()
    assert mod(np.zeros((1, m.NZ, m.NY, 3), np.float32)).shape == (1, m.NZ, m.NY, 1)
expect(SystemExit, m.main, ["Q_a", "1"])                               # refuses to overwrite
expect(ValueError, m.parse_args, ["Q_d_a"])                            # v034 configurations are not in v035
shutil.rmtree(tmp)
print("E. End-to-end smoke test passed (test and excluded runs never read; whitening stored once, saved in the model "
      "and reused; refuses to overwrite)")
print("ALL v035 STENCIL CNN TESTS PASSED")
