"""Test of diagnostics/v033_linear_conv_test.py (CPU, float64; plain script, no pytest).

Unit checks: whitening() gives Cov(W (x - xbar)) = I on correlated synthetic patches; the whitened
model with readout weights u equals the raw model with kernel K = W^T u and bias c - u.W xbar on a
random input (both with the ghost padding), and stencil() returns that K; raw_rescaled starts from
the raw model's initial function and stencil() returns its kernel on x (S k~); the raw model with the
exact kernel (sigma_b / (dz^2 sigma_y)) L reproduces the normalised Laplacian including the wall rows.
End to end on small synthetic run folders (Q_d = kappa Lap b): Q_d_b trained for 1 epoch (for its
normalisation), then the test script for 2 epochs with the default models and again with
raw_rescaled alone: histories and stencils have the right lengths, the second run reads the first as
reference, outputs written, a second call refuses to overwrite.
Usage (from simulations_v2/): python tests/test_v033_linear_conv.py diagnostics/v033_linear_conv_test.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("v033_linear", script)
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)
DG, EV, T, pp = lc.DG, lc.EV, lc.T, lc.DG.pp
rng = np.random.default_rng(5)


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}")


# ---------- unit checks ----------
A = rng.normal(size=(9, 9)) * np.logspace(0, -4, 9)
P = rng.normal(size=(50000, 9)) @ A.T + 3.0
W, xbar, lam = lc.whitening(P)
# cond(C) ~ 1e12 here (as in the real patches), so float64 round-off leaves |Cov - I| ~ 1e-16 * 1e12
assert np.allclose(np.cov((P - xbar) @ W.T, rowvar=False), np.eye(9), atol=1e-3)
assert np.all(np.diff(lam) <= 0)

T.NZ, T.NY = 8, 16
T.select_config("Q_d_b")
pad = {"wall_bottom": [-0.7], "wall_top": [1.3], "dirichlet": [1.0]}
x = rng.normal(size=(2, T.NZ, T.NY, 1))
mw = lc.build("whitened", pad, W, xbar)
u = rng.normal(size=(1, 1, 9, 1))
c = 0.37
mw.get_layer("readout").set_weights([u, np.array([c])])
mr = lc.build("raw", pad)
K = lc.stencil(mw, "whitened", W)
mr.get_layer("stencil").set_weights([K.reshape(3, 3, 1, 1), np.array([c - u[0, 0, :, 0] @ W @ xbar])])
assert np.allclose(mw.predict(x, verbose=0), mr.predict(x, verbose=0), rtol=1e-10, atol=1e-10)
assert np.allclose(lc.stencil(mr, "raw"), K)

S = 405.3
m0 = lc.build("raw", pad)
ms = lc.build_initialised("raw_rescaled", pad, kscale=S)
assert np.allclose(ms.predict(x, verbose=0), m0.predict(x, verbose=0), rtol=1e-10, atol=1e-10)
assert np.allclose(lc.stencil(ms, "raw_rescaled", kscale=S), lc.stencil(m0, "raw"), rtol=1e-12)
assert np.abs(ms.get_layer("stencil").get_weights()[0]).max() < 1.0 / S

dz = 1.0 / T.NZ
sb, sy, mb = 0.3, 7.0, 0.2
b = rng.random((2, T.NZ, T.NY)) + 0.01
pad_b = {"wall_bottom": [(T.B_WALL[0] - mb) / sb], "wall_top": [(T.B_WALL[1] - mb) / sb], "dirichlet": [1.0]}
me = lc.build("raw", pad_b)
me.get_layer("stencil").set_weights([(EV.EXACT["interior"] * sb / (dz ** 2 * sy)).reshape(3, 3, 1, 1), np.zeros(1)])
lap = np.stack([pp.laplacian(b[n], dz, dz, T.B_WALL[0], T.B_WALL[1]) for n in range(2)]) / sy
assert np.allclose(me.predict(((b - mb) / sb)[..., None], verbose=0)[..., 0], lap, rtol=1e-10, atol=1e-8)
print("Unit checks passed (whitening, equivalence of the models, rescaled start, exact kernel incl. wall rows)")


# ---------- end to end on synthetic run folders ----------
humid_src = script.parents[1] / "runs" / "Ra_1e6_Pr_1" / "humid.in"
T.TRAIN_FRAMES, T.VAL_FRAMES, T.STRIDE, T.GAP_MIN = (2, 11), (17, 26), 3, 5
T.EPOCHS, T.SEEDS, T.BATCH_SIZE = 1, (1,), 2
lc.EPOCHS, lc.WHITEN_CELLS = 2, 3000
tmp = Path(tempfile.mkdtemp())
T.RUNS_DIR, T.OUT_DIR = tmp / "runs", tmp / "v033"
NT = 30
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
T.TRAIN_RUNS = list(train)
T.TEST_RUNS = ["Ra_8.7e6_Pr_1"]
T.EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "synthetic"}
xm = (np.arange(T.NZ) + 0.5) / T.NZ
for label, ra in train.items():
    d = T.RUNS_DIR / label
    (d / "outputdir").mkdir(parents=True)
    (d / "bou.in").write_text(
        "NXM     NYM     NZM     NSST\n"
        f"{T.NZ}       {T.NY}      1       3\n"
        "TOUT    TFRAME  SAVE_3D\n1.0     1.0     6000.0\n"
        "ALX3   YLEN   ZLEN\n"
        f"1.0    {T.NY / T.NZ}   0.01\n"
        "RAYT    PRAT    RAYS    PRAS    FFscaleS\n"
        f"{ra:g}     1.0     1e7     1.0     0\n")
    shutil.copy(humid_src, d / "humid.in")
    (d / "setup.txt").write_text("set: train\n")
    (d / "postprocess.txt").write_text("synthetic\n")
    with h5py.File(d / "outputdir" / "cordin_info.h5", "w") as f:
        f["xc"] = np.arange(T.NZ + 1) / T.NZ
        f["xm"] = xm
        f["ym"] = (np.arange(T.NY) + 0.5) / T.NZ
    theta = rng.random((T.NZ, T.NY, NT)) + 0.01
    Qd = np.stack([pp.laplacian(theta[:, :, n], dz, dz, T.B_WALL[0], T.B_WALL[1]) for n in range(NT)], -1) / math.sqrt(ra)
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=theta, Q=theta)
    np.savez(d / f"{label}_q.npz", q=theta)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=np.zeros_like(Qd))

T.main(["Q_d_b", "1"])
res = lc.main()
for kind in lc.DEFAULT_KINDS:
    h = res["hist"][kind]
    assert len(h["loss"]) == len(h["val_loss"]) == 2 and len(h["stencils"]) == 3
    assert np.array(h["stencils"]).shape == (3, 3, 3)
assert res["floor"] < 1e-10, res["floor"]
out = T.OUT_DIR / lc.OUT_NAME
for fname in ("summary.txt", "history.json", "linear_conv_raw_whitened.png"):
    assert (out / fname).is_file(), fname
assert set(json.loads((out / "history.json").read_text())["hist"]) == set(lc.DEFAULT_KINDS)
expect(SystemExit, lc.main)

res2 = lc.main(["raw_rescaled"])
assert set(res2["hist"]) == {"raw_rescaled"} and set(res2["ref"]) == set(lc.DEFAULT_KINDS)
h = res2["hist"]["raw_rescaled"]
assert len(h["val_loss"]) == 2 and np.allclose(h["stencils"][0], res["hist"]["raw"]["stencils"][0])
out2 = T.OUT_DIR / f"{lc.OUT_NAME}_raw_rescaled"
for fname in ("summary.txt", "history.json", "linear_conv_raw_rescaled.png"):
    assert (out2 / fname).is_file(), fname
expect(SystemExit, lc.main, ["raw_rescaled"])
expect(SystemExit, lc.main, ["bogus"])
shutil.rmtree(tmp)
print("ALL v033 LINEAR-CONV TESTS PASSED")
