"""Test of diagnostics/v033_diagnostics.py (TensorFlow on the CPU; plain script, no pytest).

Unit checks: the patch order (patch . Laplacian / dz^2 = postprocess.laplacian); least squares
recovers the exact interior and wall-row stencils and intercepts from random b; predict() with the
exact stencils reproduces postprocess.laplacian; the s bins hold every value, sums per bin add up,
and the shares within |s| < e are right on a hand-made example.
End to end, on small synthetic run folders in which Q_d = kappa Lap b and Q_c = (gamma/tau) max(s, 0)
are built with postprocess.py's functions: every configuration trained for 1 epoch, evaluated with
"v033 evaluate.py", then the diagnostics: fitted stencils exact, baseline and float32 floors at
round-off, binned Q_c error equal to the evaluation's relative error for every seed and run,
outputs written, training and evaluation outputs untouched, a second call refuses to overwrite.
Usage (from simulations_v2/): python tests/test_v033_diagnostics.py diagnostics/v033_diagnostics.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import math
import shutil
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("v033_diag", script)
dg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dg)
ev, T, pp = dg.EV, dg.T, dg.pp
rng = np.random.default_rng(3)


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}")


# ---------- unit checks ----------
T.NZ, T.NY = 8, 16
dz = dy = 1.0 / T.NZ
B = rng.random((5, T.NZ, T.NY)) + 0.01
Y = np.stack([pp.laplacian(B[n], dz, dy, T.B_WALL[0], T.B_WALL[1]) for n in range(5)])
t, i, j = (a.ravel() for a in np.meshgrid(np.arange(5), np.arange(1, T.NZ - 1), np.arange(T.NY), indexing="ij"))
assert np.allclose(dg.gather(B, t, i, j, dg.DIS["interior"]) @ dg.LAPLACIAN / dz ** 2, Y[t, i, j],
                   rtol=1e-12, atol=1e-9)
for kind in dg.KINDS:
    X, y = dg.sample_cells(B, Y, kind, 3000, rng)
    coef, c0, _ = dg.lstsq(X, y)
    g = dg.to_grid(coef, dg.DIS[kind]) * dz ** 2
    ref = ev.EXACT[kind]
    assert np.allclose(g[~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-8), (kind, g)
    assert np.isnan(g[np.isnan(ref)]).all()
    want = 0.0 if kind == "interior" else 2.0 * T.B_WALL[0 if kind == "bottom" else 1]
    assert math.isclose(c0 * dz ** 2, want, abs_tol=1e-8), (kind, c0 * dz ** 2)
assert np.allclose(dg.predict(B, dg.exact_fit()), Y, rtol=1e-12, atol=1e-9)

E = dg.s_edges()
s = np.concatenate([np.zeros(3), 10 ** rng.uniform(-9, 3, 500), -10 ** rng.uniform(-9, 3, 500)])
idx = dg.s_bin(s, E)
c = len(E) // 2 - 1
assert (idx[:3] == c).all() and idx.min() >= 0 and idx.max() <= len(E) - 2
inside = (s >= E[0]) & (s < E[-1])
assert ((E[idx] <= s) & (s < E[idx + 1]))[inside].all()
w = rng.random(s.size)
assert math.isclose(np.bincount(idx, weights=w, minlength=len(E) - 1).sum(), w.sum())
cen = dg.s_centres(E)
assert cen[c] == 0.0 and (cen[c + 1:] > 0).all() and (cen[:c] < 0).all()
sse = np.zeros(len(E) - 1)
sse[c], sse[c + 5], sse[c - 9] = 1.0, 1.0, 2.0
pos = E[len(E) // 2:]
assert math.isclose(dg.share_within(sse, pos[5], E), 0.5) and math.isclose(dg.share_within(sse, pos[4], E), 0.25)
assert dg.s_half(sse, E) == pos[5]
print("Unit checks passed (patch order, least-squares stencils, exact predictor, s bins and shares)")


# ---------- end to end on synthetic run folders ----------
humid_src = script.parents[1] / "runs" / "Ra_1e6_Pr_1" / "humid.in"
hum = pp.read_humid(humid_src)
assert (hum["alpha"], hum["beta"]) == (T.ALPHA, T.BETA)
T.TRAIN_FRAMES, T.VAL_FRAMES, T.STRIDE, T.GAP_MIN = (2, 11), (17, 26), 3, 5
T.EPOCHS, T.SEEDS, T.BATCH_SIZE = 1, (1, 2), 2
ev.BATCH, ev.STENCIL_FRAMES = 4, 2
dg.FIT_CELLS, dg.FIT_WALL_CELLS, dg.COND_CELLS, dg.BATCH = 4000, 1000, 4000, 4
tmp = Path(tempfile.mkdtemp())
T.RUNS_DIR, T.OUT_DIR = tmp / "runs", tmp / "v033"
NT = 30
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
T.TRAIN_RUNS = list(train)
T.TEST_RUNS = ["Ra_8.7e6_Pr_1"]
T.EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "synthetic"}
xm = (np.arange(T.NZ) + 0.5) / T.NZ


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
        f["xm"] = xm
        f["ym"] = (np.arange(T.NY) + 0.5) / T.NZ
    theta = rng.random((T.NZ, T.NY, NT)) + 0.01
    q = rng.random((T.NZ, T.NY, NT))
    qs = pp.saturation(theta, xm[:, None, None], hum["alpha"], hum["beta"])
    Qd = np.stack([pp.laplacian(theta[:, :, n], dz, dy, T.B_WALL[0], T.B_WALL[1]) for n in range(NT)], -1) / math.sqrt(ra)
    Qc = pp.condensation(q, qs, hum["gamma"], hum["tau"])
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=q - qs, Q=theta)
    np.savez(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)


for label, ra in train.items():
    make_run(label, ra, "train")
make_run("Ra_8.7e6_Pr_1", 8.7e6, "test")

for cfg in ("Q_d_a", "Q_d_b"):
    T.main([cfg, "1"])
for cfg in ev.QC_CONFIGS:
    T.main([cfg, "1", "2"])
res_ev = ev.main()
before = {p: p.read_bytes() for p in T.OUT_DIR.rglob("*") if p.is_file()}
res = dg.main()
for p, content in before.items():
    assert p.read_bytes() == content, f"changed: {p}"

fit = res["qd"]["fit"]
for kind in dg.KINDS:
    g = fit[kind]["grid"] * dz ** 2
    ref = ev.EXACT[kind]
    assert np.allclose(g[~np.isnan(ref)], ref[~np.isnan(ref)], atol=1e-4), (kind, g)
    want = 0.0 if kind == "interior" else 2.0 * T.B_WALL[0 if kind == "bottom" else 1]
    assert math.isclose(fit[kind]["intercept"] * dz ** 2, want, abs_tol=1e-4), kind
assert len(res["qd"]["errors"]) == 2 * len(train) + 1
assert max(max(e["fit"], e["exact"]) for e in res["qd"]["errors"]) < 1e-5, res["qd"]["errors"]
assert set(res["qd"]["cond"]) == set(train)
for cv in res["qd"]["cond"].values():
    assert cv["eig"].size == 9 and math.isclose(cv["share"].sum(), 1.0, rel_tol=1e-9)

n_val = len(range(T.VAL_FRAMES[0], T.VAL_FRAMES[1] + 1, T.STRIDE))
n_test = n_val + len(range(T.TRAIN_FRAMES[0], T.TRAIN_FRAMES[1] + 1, T.STRIDE))
for name in dg.QC_NAMES:
    runs = res["qc"][name]["runs"]
    assert set(runs) == set(train) | {"Ra_8.7e6_Pr_1"}
    for label, r in runs.items():
        assert r["counts"].sum() == (n_test if r["set"] == "test" else n_val) * T.NZ * T.NY
        assert r["floor_rel"] < 1e-6, (name, label, r["floor_rel"])
        assert 0 < r["s_pos"].size < r["counts"].sum()
        for seed, rel in r["rel"].items():
            rec = next(x for x in res_ev["records"] if x["config"] == name and x["seed"] == seed
                       and x["run"] == label and x["set"] == r["set"])
            assert math.isclose(rel, rec["rel_rmse"], rel_tol=1e-6), (name, label, seed, rel, rec["rel_rmse"])
assert res["qc"][dg.QC_NAMES[0]]["runs"]["Ra_1e6_Pr_1"]["s_rel"] is not None
assert res["qc"][dg.QC_NAMES[1]]["runs"]["Ra_1e6_Pr_1"]["s_rel"] is None

out = T.OUT_DIR / dg.OUT_NAME
for fname in ("summary.txt", "Qd_linear_baseline_stencil.png", "Qd_linear_baseline_errors.png",
              "Qd_input_conditioning.png", "Qc_C2_error_vs_supersaturation.png",
              "Qc_C4_error_vs_supersaturation.png", "Qc_relative_supersaturation.png"):
    assert (out / fname).is_file(), fname
expect(SystemExit, dg.main)
shutil.rmtree(tmp)
print("End-to-end diagnostics test passed (exact stencils recovered, floors at round-off, binned errors = evaluation)")
print("ALL v033 DIAGNOSTICS TESTS PASSED")
