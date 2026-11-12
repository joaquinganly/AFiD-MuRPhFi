"""Test of diagnostics/v034_dns_vs_cnn.py (TensorFlow on the CPU; plain script, no pytest); copy of the
v033 test with the v034 training settings (whitened Q_d models; CONSTANT_LR_EPOCHS 0 for 1 epoch).

Unit checks: linear and joint histograms (bins, under- and overflow); log10 histograms, their
cumulative distribution and the binned median against direct values; the zoom window (width,
wrap-around in y).
End to end, on small synthetic run folders (Q_d = kappa Lap b and Q_c = (gamma/tau) max(s, 0) built
with postprocess.py's functions): every configuration trained for 1 epoch (two seeds), evaluated with
"v034 evaluate.py", then this script. Checks: the relative RMS error over the evaluation's frames
equals per_run_errors.csv for every model; two models' every-frame statistics recomputed
independently (inputs normalised by hand, model.predict frame by frame); DNS statistics, set sizes,
t* and t_q against direct computations; outputs written (figures, GIF, data, summary); training and
evaluation outputs untouched; a second call refuses to overwrite.
Usage (from simulations_v2/): python tests/test_v034_dns_vs_cnn.py diagnostics/v034_dns_vs_cnn.py
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
import tensorflow as tf

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("v034_dns_vs_cnn", script)
dv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dv)
ev, T, pp = dv.EV, dv.T, dv.pp
rng = np.random.default_rng(5)


def expect(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} from {fn.__name__}")


# ---------- unit checks ----------
lo, hi = -1.0, 3.0
w = (hi - lo) / dv.NBIN
h = dv.lin_hist(np.array([-2.0, -1.0, 0.0, 3.0 - w / 2, 3.0, 5.0]), lo, hi)
assert h.sum() == 6 and h[0] == 1 and h[-1] == 2 and h[1] == 1 and h[dv.NBIN] == 1, h.nonzero()
assert h[1 + int(round((0.0 - lo) / w))] == 1
J = dv.joint_hist(np.array([0.0, 5.0, -3.0]), np.array([0.0, 0.0, 9.0]), lo, hi).reshape(dv.NBIN + 2, dv.NBIN + 2)
k0 = 1 + int(round((0.0 - lo) / w))
assert J.sum() == 3 and J[k0, k0] == 1 and J[dv.NBIN + 1, k0] == 1 and J[0, dv.NBIN + 1] == 1

x = 10.0 ** rng.uniform(-8.0, -2.0, 200_001)
c = dv.log_hist(x)
assert c.sum() == x.size and c[0] == 0 and c[-1] == 0
assert abs(math.log10(dv.hist_median(c)) - math.log10(np.median(x))) < 1.0 / dv.PER_DEC
e = dv.log_edges()
G = dv.hist_cdf(c)
for m in (0, 450, 700, dv.NLOG):
    assert math.isclose(G[m], np.mean(x < e[m]), abs_tol=1e-12), (m, G[m], np.mean(x < e[m]))
one = dv.log_hist(np.full(11, 3.3e-4))
assert e[np.argmax(one) - 1] <= dv.hist_median(one) <= e[np.argmax(one)]
assert math.isnan(dv.hist_median(np.zeros(dv.NLOG + 2, np.int64)))
cc = dv.log_hist(np.array([1e-20, 1e5]))
assert cc[0] == 1 and cc[-1] == 1 and math.isnan(dv.hist_median(cc))

T.NZ, T.NY = 8, 16
dv.ZOOM_W = 1.0
cols = dv.zoom_cols(1, 2.0)                       # ly = 2: 8 columns per unit length
assert list(cols) == [13, 14, 15, 0, 1, 2, 3, 4], cols
frame = np.zeros((T.NZ, T.NY))
frame[3:5, 11] = 1.0
assert dv.zoom_centre(frame) == 11 and dv.zoom_centre(np.zeros((T.NZ, T.NY))) == 0
print("Unit checks passed (linear, joint and log histograms, binned median and CDF, zoom window)")


# ---------- end to end on synthetic run folders ----------
humid_src = script.parents[1] / "runs" / "Ra_1e6_Pr_1" / "humid.in"
hum = pp.read_humid(humid_src)
T.TRAIN_FRAMES, T.VAL_FRAMES, T.STRIDE, T.GAP_MIN = (2, 11), (17, 26), 3, 5
T.EPOCHS, T.SEEDS, T.BATCH_SIZE, T.CONSTANT_LR_EPOCHS, T.WHITEN_CELLS = 1, (1, 2), 2, 0, 2000
ev.BATCH, ev.STENCIL_FRAMES = 4, 2
dv.SNAP_SEED, dv.SPIN, dv.BATCH, dv.CHUNK = 1, 3, 4, 5
tmp = Path(tempfile.mkdtemp())
T.RUNS_DIR, T.OUT_DIR = tmp / "runs", tmp / "v034"
NT = 30
dz = dy = 1.0 / T.NZ
train = {"Ra_1e6_Pr_1": 1e6, "Ra_1e7_Pr_1": 1e7, "Ra_1e8_Pr_1": 1e8}
TEST = "Ra_8.7e6_Pr_1"
T.TRAIN_RUNS = list(train)
T.TEST_RUNS = [TEST]
T.EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "synthetic"}
xm = (np.arange(T.NZ) + 0.5) / T.NZ
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
    raw[label] = dict(theta=theta, q=q, relq=q - qs, qs=qs, Qd=Qd, Qc=Qc, ra=ra)


for label, ra in train.items():
    make_run(label, ra, "train")
make_run(TEST, 8.7e6, "test")

for cfg in ev.QD_CONFIGS + ev.QC_CONFIGS:
    T.main([cfg, "1", "2"])
ev.main()
before = {p: p.read_bytes() for p in T.OUT_DIR.rglob("*") if p.is_file()}
out = dv.main()
for p, content in before.items():
    assert p.read_bytes() == content, f"changed: {p}"
res, check = out["res"], out["check"]

assert len(check) == 6 * 2
for key, (mine, ref, diff) in check.items():
    assert diff < 1e-4, (key, mine, ref)

r = res[TEST]
rw = raw[TEST]
g = hum["gamma"] / hum["tau"]
eps = g * dv.S_EPS
ncell = NT * T.NZ * T.NY
dmc = rw["Qc"].mean(axis=(0, 1))
assert r["frames"]["t_star"] == dv.SPIN + int(np.argmax(dmc[dv.SPIN:]))
assert r["frames"]["t_quiet"] == dv.SPIN + int(np.argmin(dmc[dv.SPIN:]))
for fam, key in (("Q_c", "Qc"), ("Q_d", "Qd")):
    d = r[fam]["dns"]
    q32 = rw[key].astype(np.float32).astype(np.float64)              # the script works on the float32 copies
    tol = 1e-12 * np.sqrt(np.mean(q32 ** 2))
    assert np.allclose(d["dmean"], q32.mean(axis=(0, 1)), rtol=0.0, atol=tol), fam
    assert np.allclose(d["prof"], q32.mean(axis=(1, 2)), rtol=0.0, atol=tol), fam
    assert d["hist"].sum() == ncell
    for name, o in r[fam]["cnn"].items():
        for s, st in o.items():
            assert st["hist"].sum() == ncell
            if s == dv.SNAP_SEED:
                assert st["joint"].sum() == ncell and set(st["snaps"]) == set(r["frames"].values())
d = r["Q_c"]["dns"]
Qc32 = rw["Qc"].astype(np.float32).astype(np.float64)
assert d["n_pos"] == int(np.sum(Qc32 > 0)) and d["n_eps"] == int(np.sum(Qc32 > eps))
assert d["hs_pos"].sum() == d["n_pos"] and d["hr_eps"].sum() == d["n_eps"]


def independent(cfg_name, seed):
    """Every frame of the test run predicted by hand: inputs normalised in float64, model.predict."""
    cdir = T.OUT_DIR / cfg_name
    nc = np.load(cdir / "normalization_constants.npz")
    mu, sd = nc["input_mean"], nc["input_std"]
    tmu, tsd = float(nc["target_mean"]), float(nc["target_std"])
    model = tf.keras.models.load_model(cdir / f"seed_{seed}" / "model.keras", compile=False)
    zmap = np.repeat(xm[:, None], T.NY, 1)
    chans = {"b": rw["theta"], "q": rw["q"], "q_minus_qs": rw["relq"]}
    lra = math.log10(rw["ra"])
    preds = []
    for n in range(NT):
        X = np.stack([zmap if c == "z" else np.full((T.NZ, T.NY), lra) if c == "log10_Ra"
                      else chans[c][:, :, n].astype(np.float32) for c in T.CONFIGS[cfg_name]["inputs"]], -1)
        X = ((X - mu) / sd)[None].astype(np.float32)
        preds.append(model.predict(X, verbose=0)[0, ..., 0].astype(np.float64) * tsd + tmu)
    p = np.array(preds)
    if T.CONFIGS[cfg_name]["target"] == "Q_d_over_kappa":
        p *= 1.0 / math.sqrt(rw["ra"])
    return p


for cfg_name, seed, fam, key in (("Q_c_C2", 2, "Q_c", "Qc"), ("Q_d_b", 1, "Q_d", "Qd")):
    p = independent(cfg_name, seed)
    st = r[fam]["cnn"][cfg_name][seed]
    y = np.moveaxis(rw[key], -1, 0).astype(np.float32).astype(np.float64)
    scale = np.sqrt(np.mean(y ** 2))
    assert np.allclose(st["dmean"], p.mean(axis=(1, 2)), rtol=1e-4, atol=1e-5 * scale), cfg_name
    assert np.allclose(st["prof"], p.mean(axis=(0, 2)), rtol=1e-4, atol=1e-5 * scale), cfg_name
    assert np.allclose(st["sse_t"], np.sum((p - y) ** 2, axis=(1, 2)), rtol=1e-3), cfg_name
    if fam == "Q_c":
        assert abs(st["n_set"] - int(np.sum(p > eps))) <= 2, (st["n_set"], int(np.sum(p > eps)))
        assert abs(st["n_neg"] - int(np.sum(p < 0))) <= 2
        assert abs(st["n_both"] - int(np.sum((p > eps) & (y > eps)))) <= 2
        assert st["hs"].sum() == st["n_set"]

od = out["out"]
names = sorted(x.name for x in od.iterdir())
expected = {"summary.txt", "dns_vs_cnn_data.npz", "Qc_domain_mean_vs_time.png", "Qd_domain_mean_vs_time.png",
            "Qc_mean_profiles.png", "Qd_mean_profiles.png", "Qc_joint_histograms.png", "Qd_joint_histograms.png",
            "Qc_histograms.png", "Qd_histograms.png", "Qc_condensing_cells.png", "Qc_C4_vs_DNS_Ra_8.7e6.gif"}
assert expected <= set(names), expected - set(names)
assert sum(n.startswith("Qc_snapshots_") for n in names) == 2 and sum(n.startswith("Qd_snapshots_") for n in names) == 2
assert not any("partial" in n for n in names)
with np.load(od / "dns_vs_cnn_data.npz") as f:
    assert f"{TEST}__Q_c__Q_c_C4__1__joint" in f.files and f"{TEST}__Q_d__DNS__dmean" in f.files
assert "Check against per_run_errors.csv" in (od / "summary.txt").read_text()
expect(SystemExit, dv.main)
shutil.rmtree(tmp)
print("End-to-end test passed (evaluation frames = per_run_errors.csv, independent recomputation, DNS statistics)")
print("ALL v034 DNS-VS-CNN TESTS PASSED")
