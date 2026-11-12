"""Test of "v034 evaluate.py" (TensorFlow on the CPU; plain script, no pytest). Copy of the v033 test,
plus: the stencil probe recovers the Laplacian from a whitened linear model (fixed float64 whitening
layer of "v034 stencil CNN.py" and a 1 x 1 readout with the exact weights).

Unit checks: factorial_effects on hand-made numbers; the z-profile reference is 0 for a target
that depends on z only; stencil_response recovers [0 1 0; 1 -4 1; 0 1 0] in the interior and
the wall-row centre -5 from a linear Laplacian model with the ghost padding.
End to end, on small synthetic run folders: every configuration is trained for 2 epochs (one
seed for Q_d, two for Q_c so that the 2x2 standard errors exist) and evaluated; checks that the
recomputed validation loss equals Keras' val_loss at the best epoch for every model; two records
(a test run of Q_c_C1, a validation block of Q_d_b) recomputed independently from the raw arrays
and the saved models; the excluded run is never read; outputs written; training outputs
untouched; a second evaluation refuses to overwrite.
Usage (from simulations_v2/): python tests/test_v034_evaluate.py "v034 evaluate.py"
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # exact-equality tests on the CPU
import importlib.util
import math
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("v034_eval", script)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
T = ev.T
rng = np.random.default_rng(2)


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}{args}")


# ---------- unit checks ----------
A, B, AB = "A (q -> q - q_s)", "B (drop z)", "AB (interaction)"
L = {"Q_c_C1": np.array([0.0, 0.0]), "Q_c_C2": np.array([-2.0, -2.0]),
     "Q_c_C3": np.array([0.5, 0.5]), "Q_c_C4": np.array([-1.5, -1.5])}
e = ev.factorial_effects(L)
assert np.isclose(e[A][0], -2.0) and np.isclose(e[B][0], 0.5) and np.isclose(e[AB][0], 0.0), e
assert e[A][1] == 0.0
e2 = ev.factorial_effects({c: v + np.array([0.1, -0.1]) for c, v in L.items()})
assert np.isclose(e2[A][1], math.sqrt(4 * (np.var([0.1, -0.1], ddof=1) / 2) / 4))

T.NZ, T.NY = 8, 16
yz = np.broadcast_to(rng.normal(size=(1, 8, 1)), (5, 8, 16)).copy()
run = {"y": yz.astype(np.float32), "parts": {"train": slice(0, 3), "val": slice(3, 5)}}
prof = ev.z_profile(run, ["train", "val"])
assert np.allclose(prof, yz[0, :, 0], atol=1e-6) and ev.profile_sse(prof, yz) < 1e-9

dz = 1.0 / T.NZ
inp = tf.keras.layers.Input((T.NZ, T.NY, 1))
x = T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0])(inp)
conv = tf.keras.layers.Conv2D(1, 3, padding="valid", use_bias=False)
lin = tf.keras.Model(inp, conv(x))
K = np.zeros((3, 3, 1, 1))
K[0, 1, 0, 0] = K[2, 1, 0, 0] = K[1, 0, 0, 0] = K[1, 2, 0, 0] = 1 / dz ** 2
K[1, 1, 0, 0] = -4 / dz ** 2
conv.set_weights([K.astype(np.float32)])
Xs = rng.random((2, T.NZ, T.NY, 1)).astype(np.float32)
seen = set()
for rows, J in ev.stencil_response(lin, Xs, 0, dz ** 2):
    for i, r in enumerate(rows):
        kind = "bottom" if r == 0 else "top" if r == T.NZ - 1 else "interior"
        seen.add(kind)
        ref = ev.EXACT[kind]
        got = J[:, i]
        assert np.allclose(got[..., ~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-3), (r, got[0, 0])
        assert np.isnan(got[..., np.isnan(ref)]).all()
assert seen == {"bottom", "top", "interior"}
T.select_config("Q_d_b")                                        # input b only
bw = (np.linspace(0, 1, T.NZ)[None, :, None] + 0.05 * rng.normal(size=(6, T.NZ, T.NY))).astype(np.float32)
wh = T.fit_whitening([{"fields": bw[..., None], "parts": {"train": slice(0, 6)}}])
kern, bias = T.whitening_weights(wh)
inp = tf.keras.layers.Input((T.NZ, T.NY, 1))
xw = T.StencilPad(wall_bottom=[0.0], wall_top=[1.0], dirichlet=[1.0], dtype="float64")(inp)
wl = tf.keras.layers.Conv2D(9, 3, padding="valid", dtype="float64", trainable=False)
xw = wl(xw)
wl.set_weights([kern, bias])
Kl = np.array([0., 1., 0., 1., -4., 1., 0., 1., 0.]) / dz ** 2
ro = tf.keras.layers.Conv2D(1, 1)
linw = tf.keras.Model(inp, ro(xw))
ro.set_weights([np.linalg.solve(wh["W"].T, Kl).reshape(1, 1, 9, 1).astype(np.float32),
                np.array([Kl @ wh["xbar"]], np.float32)])
seen = set()
for rows, J in ev.stencil_response(linw, Xs, 0, dz ** 2):
    for i, r in enumerate(rows):
        kind = "bottom" if r == 0 else "top" if r == T.NZ - 1 else "interior"
        seen.add(kind)
        ref = ev.EXACT[kind]
        got = J[:, i]
        assert np.allclose(got[..., ~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-3), ("whitened", r, got[0, 0])
assert seen == {"bottom", "top", "interior"}
print("Unit checks passed (factorial effects, z-profile reference, stencil probe, also through the whitening layer)")


# ---------- end to end on synthetic run folders ----------
humid_src = script.parent / "runs" / "Ra_1e6_Pr_1" / "humid.in"
T.TRAIN_FRAMES, T.VAL_FRAMES, T.STRIDE, T.GAP_MIN = (2, 11), (17, 26), 3, 5
T.EPOCHS, T.SEEDS, T.BATCH_SIZE, T.CONSTANT_LR_EPOCHS, T.WHITEN_CELLS = 2, (1, 2), 2, 1, 2000
ev.BATCH, ev.STENCIL_FRAMES = 4, 2
tmp = Path(tempfile.mkdtemp())
T.RUNS_DIR, T.OUT_DIR = tmp / "runs", tmp / "v034"
NT = 30
TR, VA = [2, 5, 8, 11], [17, 20, 23, 26]
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
T.TRAIN_RUNS = list(train)
T.TEST_RUNS = ["Ra_8.7e6_Pr_1"]
T.EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "synthetic"}
raw = {}


def make_run(label, ra, set_name):
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
    (d / "setup.txt").write_text(f"set: {set_name}\n")
    (d / "postprocess.txt").write_text("synthetic\n")
    with h5py.File(d / "outputdir" / "cordin_info.h5", "w") as f:
        f["xc"] = np.arange(T.NZ + 1) / T.NZ
        f["xm"] = (np.arange(T.NZ) + 0.5) / T.NZ
        f["ym"] = (np.arange(T.NY) + 0.5) / T.NZ
    theta = rng.random((T.NZ, T.NY, NT)) + 0.01
    q = rng.random((T.NZ, T.NY, NT))
    relq = rng.normal(size=(T.NZ, T.NY, NT))
    Qd = rng.normal(size=(T.NZ, T.NY, NT)) / np.sqrt(ra)
    Qc = np.maximum(rng.normal(size=(T.NZ, T.NY, NT)), 0.0)          # zero in about half the cells
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=relq, Q=theta)
    np.savez(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, q=q, relq=relq, Qd=Qd, Qc=Qc, ra=ra)


for label, ra in train.items():
    make_run(label, ra, "train")
make_run("Ra_8.7e6_Pr_1", 8.7e6, "test")
make_run("Ra_2e8_Pr_1", 2e8, "test")

for c in ("Q_d_a", "Q_d_b"):
    T.main([c, "1"])
for c in ev.QC_CONFIGS:
    T.main([c, "1", "2"])
trained = {p: p.read_bytes() for p in T.OUT_DIR.rglob("*") if p.is_file()}

loaded = []
_orig_load = np.load


def spy_load(path, *a, **k):
    loaded.append(str(path))
    return _orig_load(path, *a, **k)


np.load = spy_load
res = ev.main()
np.load = _orig_load
assert not any("Ra_2e8_Pr_1" in p for p in loaded), "excluded run was read"
for p, content in trained.items():
    assert p.read_bytes() == content, f"training output changed: {p}"

for (name, s), (fixed, hist) in res["loss_check"].items():
    assert np.isclose(fixed, hist, rtol=1e-4), (name, s, fixed, hist)
assert len(res["loss_check"]) == 2 + 4 * 2

out = T.OUT_DIR / ev.EVAL_DIR_NAME
for fname in ("summary.txt", "per_run_errors.csv", "Qd_errors_by_run.png", "Qd_training_curves.png",
              "Qd_stencils.png", "Qd_stencil_sums.png", "Qc_errors_by_run.png", "Qc_training_curves.png",
              "Qc_condensing_cells.png", "Qc_2x2_effects.png"):
    assert (out / fname).is_file(), fname
assert set(res["effects"]) == {(l, "val") for l in train} | {("Ra_8.7e6_Pr_1", "test"), ("all training runs", "val")}
assert all(np.isfinite(v[1]) for e in res["effects"].values() for v in e.values())


def independent(cfg_name, seed, label, frames, target):
    d = T.OUT_DIR / cfg_name
    nc = np.load(d / "normalization_constants.npz")
    mu, sd = nc["input_mean"], nc["input_std"]
    tmu, tsd = float(nc["target_mean"]), float(nc["target_std"])
    model = tf.keras.models.load_model(d / f"seed_{seed}" / "model.keras", compile=False)
    r = raw[label]
    zmap = np.repeat(((np.arange(T.NZ) + 0.5) / T.NZ)[:, None], T.NY, 1)
    chans = {"b": r["theta"], "q": r["q"], "q_minus_qs": r["relq"]}
    sse = sy2 = 0.0
    ys = []
    for n in frames:
        X = np.stack([zmap if c == "z" else chans[c][:, :, n] for c in T.CONFIGS[cfg_name]["inputs"]], -1)
        X = ((X - mu) / sd)[None].astype(np.float32)
        p = model.predict(X, verbose=0)[0, ..., 0].astype(np.float64) * tsd + tmu
        y = target[:, :, n]
        sse += np.sum((p - y) ** 2); sy2 += np.sum(y ** 2)
        ys.append(y)
    ys = np.array(ys)
    prof = ys.mean(axis=(0, 2))
    ref = math.sqrt(np.sum((prof[None, :, None] - ys) ** 2) / sy2)
    return math.sqrt(sse / sy2), ref


def record(cfg_name, seed, label, st):
    return next(r for r in res["records"] if r["config"] == cfg_name and r["seed"] == seed
                and r["run"] == label and r["set"] == st)


got = record("Q_c_C1", 2, "Ra_8.7e6_Pr_1", "test")
rel, ref = independent("Q_c_C1", 2, "Ra_8.7e6_Pr_1", TR + VA, raw["Ra_8.7e6_Pr_1"]["Qc"])
assert np.isclose(got["rel_rmse"], rel, rtol=1e-4) and np.isclose(got["ref_rel_rmse"], ref, rtol=1e-6), (got, rel, ref)
got = record("Q_d_b", 1, "Ra_1e7_Pr_1", "val")
rel, ref = independent("Q_d_b", 1, "Ra_1e7_Pr_1", VA, raw["Ra_1e7_Pr_1"]["Qd"] * np.sqrt(1e7))
assert np.isclose(got["rel_rmse"], rel, rtol=1e-4) and np.isclose(got["ref_rel_rmse"], ref, rtol=1e-6), (got, rel, ref)
assert 0.3 < record("Q_c_C3", 1, "Ra_1e6_Pr_1", "val")["frac_cond"] < 0.7
for n in ("Q_d_a", "Q_d_b"):
    st = res["stencil"][n][1]["stats"]
    assert st["interior"]["n"] > 0 and np.isfinite(st["interior"]["mean"]).all()
    assert np.isnan(st["bottom"]["mean"][0]).all() and np.isnan(st["top"]["mean"][2]).all()

expect(SystemExit, ev.main)
shutil.rmtree(tmp)
print("End-to-end evaluation test passed (loss check, independent recomputation, excluded run never read)")
print("ALL v034 EVALUATION TESTS PASSED")
