"""Tests of diagnostics/v035_q_analysis.py (CPU; plain script, no pytest). Small synthetic run folders whose DNS targets
are exact (Q_d = kappa postprocess.laplacian(b), Q_c = (gamma/tau) max(rel_q, 0)); the trained networks are replaced by
exact models (v035 Q_a: the module's exact Q; v034 Q_d_a and C4: xai_attribution.ExactQd and ExactQc), so every reported
quantity has a known value.

A. ExactQLayer: full-frame exact Q = kappa(Ra) Lap b + (gamma/tau) max(s, 0), wall rows included.
B. sum_jacobian and describe on hand-made inputs (channel mapping; medians; r_rho).
C. Part jac through main(): for Q_a (exact) and sum (ExactQd + ExactQc): Jt_b = L at every explained cell, wet J_s = 20
   at the centre and 0 elsewhere, dry J_s = 0, r_rho = 1; for C4: Jt_b = 0; Q_d_a: no J_s.
D. Part xai through main(): e_0 at round-off, the analytic Shapley credits of the exact Q (dev), efficiency, ablation of
   s leaves the dry cells unchanged.
E. Part eval through main(): e at round-off for Q_a and sum on all, wet and dry cells; mean ratio 1; the z-profile
   reference equals a direct computation; the recomputed validation loss is reported.
F. Part dns through main(): every frame of the test run; e at round-off; t* and t_q from the DNS Q_c; joint histogram
   counts the cells inside its range; refuses to overwrite.
Usage (from simulations_v2/): python tests/test_v035_q_analysis.py diagnostics/v035_q_analysis.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
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
spec = importlib.util.spec_from_file_location("v035_q_analysis", script)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
xa, ja, T34, T35, pp = m.xa, m.ja, m.T34, m.T35, m.T34.pp
rng = np.random.default_rng(0)
L = np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])

# ---------- small grid and synthetic runs ----------
for t in (T34, T35):
    t.NZ, t.NY = 16, 32
    t.TRAIN_FRAMES, t.VAL_FRAMES, t.STRIDE, t.GAP_MIN = (2, 11), (17, 26), 3, 5
    t.WHITEN_CELLS = 500
NZ, NY, NT = 16, 32, 30
DZ = 1.0 / NZ
tmp = Path(tempfile.mkdtemp())
for t in (T34, T35):
    t.RUNS_DIR = tmp / "runs"
    t.TRAIN_RUNS, t.TEST_RUNS = ["Ra_1e6_Pr_1", "Ra_1e8_Pr_1"], ["Ra_8.7e6_Pr_1"]
    t.EXCLUDED_RUNS = {}
T35.OUT_DIR = tmp / "v035"
m.OUT_DIR = tmp / "v035" / "analysis"
humid_src = script.parents[1] / "runs" / "Ra_1e6_Pr_1" / "humid.in"
h = pp.read_humid(humid_src)
GT = h["gamma"] / h["tau"]
RAS = {"Ra_1e6_Pr_1": 1e6, "Ra_1e8_Pr_1": 1e8, "Ra_8.7e6_Pr_1": 8.7e6}
raw = {}


def make_run(label, ra, set_name):
    d = T34.RUNS_DIR / label
    (d / "outputdir").mkdir(parents=True)
    (d / "bou.in").write_text("NXM     NYM     NZM     NSST\n" f"{NZ}       {NY}      1       3\n"
                              "TOUT    TFRAME  SAVE_3D\n1.0     1.0     6000.0\n" "ALX3   YLEN   ZLEN\n"
                              f"1.0    {NY / NZ}   0.01\n" "RAYT    PRAT    RAYS    PRAS    FFscaleS\n"
                              f"{ra:g}     1.0     1e7     1.0     0\n")
    shutil.copy(humid_src, d / "humid.in")
    (d / "setup.txt").write_text(f"set: {set_name}\n")
    (d / "postprocess.txt").write_text("synthetic\n")
    with h5py.File(d / "outputdir" / "cordin_info.h5", "w") as f:
        f["xc"] = np.arange(NZ + 1) / NZ
        f["xm"] = (np.arange(NZ) + 0.5) / NZ
        f["ym"] = (np.arange(NY) + 0.5) / NZ
    theta = rng.random((NZ, NY, NT)) * 0.9 + 0.05
    relq = rng.normal(scale=0.01, size=(NZ, NY, NT)) - 0.004
    kappa = 1.0 / math.sqrt(ra)
    Qd = np.stack([kappa * pp.laplacian(theta[:, :, n], DZ, DZ, 0.0, 1.0) for n in range(NT)], -1)
    Qc = GT * np.maximum(relq, 0.0)
    np.savez(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=relq, Q=theta + 99.0)
    np.savez(d / f"{label}_q.npz", q=relq)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, relq=relq, Qd=Qd, Qc=Qc, kappa=kappa)


for label, ra in RAS.items():
    make_run(label, ra, "test" if label in T34.TEST_RUNS else "train")
make_run("Ra_3e6_Pr_1", 3e6, "train")                         # dy != dz: check_spacing must refuse it
bad = T34.RUNS_DIR / "Ra_3e6_Pr_1" / "bou.in"
bad.write_text(bad.read_text().replace(f"1.0    {NY / NZ}   0.01", "1.0    1.0   0.01"))
try:
    m.check_spacing(["Ra_3e6_Pr_1"])
    raise AssertionError("check_spacing accepted dy != dz")
except SystemExit:
    pass
m.check_spacing(list(RAS))


# ---------- exact stand-ins for the trained networks ----------
def fake_setup(t, name, input_norm, target_norm, model):
    t.select_config(name)
    hist = {"loss": [0.5, 0.4, 0.3], "val_loss": [0.6, 0.45, 0.5], "lr": [1e-3, 1e-3, 5e-4]}
    return {"name": name, "t": t, "cfg": {"inputs": list(t.INPUTS), "epochs": 150}, "seeds": list(m.SEEDS),
            "models": {s: model for s in m.SEEDS}, "input_norm": input_norm, "target_norm": target_norm,
            "histories": {s: hist for s in m.SEEDS},
            "summary": {s: {"best_epoch": 2, "best_val_loss": 0.45} for s in m.SEEDS},
            "whiten": {"whiten_xbar": np.zeros(9)}}


norm = {"b": (0.5, 0.26), "q_minus_qs": (-0.004, 0.01), "log10_Ra": (7.0, 0.8)}
s_qa = fake_setup(T35, "Q_a", norm, (0.02, 0.3), None)
s_qa["models"] = {s: m.exact_q_model(s_qa, GT) for s in m.SEEDS}
s_qd = fake_setup(T34, "Q_d_a", {"b": norm["b"], "log10_Ra": norm["log10_Ra"]}, (0.0, 0.05), None)
s_qd["models"] = {s: xa.ExactQd(T34, s_qd) for s in m.SEEDS}
s_qc = fake_setup(T34, "Q_c_C4", {"b": norm["b"], "q_minus_qs": norm["q_minus_qs"]}, (0.03, 0.1), None)
s_qc["models"] = {s: xa.ExactQc(["b", "q_minus_qs"], s_qc["input_norm"], s_qc["target_norm"], GT, T34.ALPHA, T34.BETA)
                  for s in m.SEEDS}
SETUPS = {"Q_a": s_qa, "Q_d_a": s_qd, "C4": s_qc}
m.load_all_setups = lambda: SETUPS

# ---------- A. ExactQLayer on full frames ----------
lab = "Ra_1e8_Pr_1"
r0 = raw[lab]
fields = np.stack([r0["theta"][:, :, :3], r0["relq"][:, :, :3]], -1).transpose(2, 0, 1, 3).astype(np.float32)
run_stub = {"label": lab, "xm": (np.arange(NZ) + 0.5) / NZ, "params": {"log10_Ra": 8.0}}
X = m.inputs_for(s_qa, fields, run_stub)
pred = m.predict_raw(s_qa["models"][42], s_qa, X)
ref = np.moveaxis(r0["Qd"][:, :, :3] + r0["Qc"][:, :, :3], -1, 0)
errA = float(np.max(np.abs(pred - ref)) / np.max(np.abs(ref)))
assert errA < 2e-6, errA
print(f"A. Exact Q layer = kappa Lap b + (gamma/tau) max(s, 0) on full frames, wall rows included (max rel err {errA:.1e})")

# ---------- B. sum_jacobian, describe ----------
n = 5
Jd, Jc = rng.normal(size=(n, 2, 9)), rng.normal(size=(n, 2, 9))
Js = m.sum_jacobian((Jd, ["b", "log10_Ra"]), (Jc, ["b", "q_minus_qs"]))
assert np.allclose(Js[:, 0], Jd[:, 0] + Jc[:, 0]) and np.allclose(Js[:, 1], Jc[:, 1]) and np.allclose(Js[:, 2], Jd[:, 1])
q = {"Jb": np.arange(9.0)[None].repeat(3, 0), "JbT": np.ones((3, 9)), "Js": np.full((3, 9), np.nan),
     "Jr": np.array([1.0, 2.0, 3.0]), "Jr_ex": np.array([1.0, 1.0, 1.0])}
d = m.describe(q, np.array([True, True, True]))
assert np.allclose(d["Jb"][1], np.arange(9.0)) and np.isnan(d["Js"]).all() and math.isclose(d["r_rho"], 2.0) and d["n"] == 3
assert np.isnan(m.describe(q, np.zeros(3, bool))["r_rho"])
print("B. sum_jacobian maps the channels of Q_d_a (b, rho) and C4 (b, s); describe gives medians and r_rho")

# ---------- C. part jac ----------
m.main(["jac"])
st = dict(np.load(m.OUT_DIR / "jacobian.npz"))


def get(tag, s, l, c, k):
    return st[f"{tag}__{s}__{l}__{c}__{k}"]


for tag, s in (("Q_a", 42), ("sum", 43), ("exact", 0)):
    for l in RAS:
        for c in ("wet", "dry"):
            if int(get(tag, s, l, c, "n")) == 0:
                continue
            for i in range(3):                                         # 5th, 50th, 95th percentiles
                assert np.allclose(get(tag, s, l, c, "Jb")[i], L.ravel(), atol=2e-4), (tag, l, c, get(tag, s, l, c, "Jb"))
                assert np.allclose(get(tag, s, l, c, "JbT")[i], L.ravel() @ m.U, atol=3e-4)
            js = get(tag, s, l, c, "Js")
            ex = np.zeros(9); ex[4] = GT if c == "wet" else 0.0
            assert np.allclose(js, ex[None], atol=1e-4), (tag, l, c, js)
        assert abs(float(get(tag, s, l, "all", "r_rho")) - 1.0) < 1e-4, (tag, l, get(tag, s, l, "all", "r_rho"))
    for c in ("wet", "dry"):
        assert np.allclose(get(tag, s, "train", c, "Jb")[1], L.ravel(), atol=2e-4)
assert np.allclose(get("C4", 44, "Ra_1e6_Pr_1", "dry", "Jb"), 0.0) and np.isnan(get("Q_d_a", 42, "Ra_1e6_Pr_1", "wet", "Js")).all()
assert int(get("Q_a", 42, "Ra_1e6_Pr_1", "wet", "n")) > 0 and int(get("Q_a", 42, "Ra_1e6_Pr_1", "dry", "n")) > 0
for f in m.PART_OUT["jac"]:
    assert (m.OUT_DIR / f).is_file(), f
print("C. jac: exact Q and the sum of the exact Q_d and Q_c give Jt_b = L, J_s = gamma/tau at wet centres, r_rho = 1")

# ---------- D. part xai ----------
m.main(["xai"])
xr = dict(np.load(m.OUT_DIR / "xai.npz"))
for l in RAS:
    assert float(xr[f"exact__0__{l}__dev_ch"]) < 1e-5 and float(xr[f"exact__0__{l}__dev_T"]) < 1e-5, l
    assert float(xr[f"exact__0__{l}__eff_ch"]) < 1e-9 and float(xr[f"exact__0__{l}__eff_T"]) < 1e-9, l
    e0 = xr[f"Q_a__42__{l}__e0"]
    assert np.all(e0 < 2e-6), (l, e0)
    e_ch = xr[f"Q_a__42__{l}__e_ch"]                          # rows all, wet, dry; columns b, s, rho
    if float(np.max(raw[l]["relq"])) > 0:
        assert e_ch[1][1] > 1e-3                               # removing s spoils the wet cells
    assert e_ch[2][1] < 2e-6 and e_ch[2][1] >= 0               # ... but not the dry ones (s_ref < 0 there too)
for f in m.PART_OUT["xai"]:
    assert (m.OUT_DIR / f).is_file(), f
print("D. xai: analytic Shapley credits of the exact Q reproduced (channels and templates); efficiency; ablation")

# ---------- E. part eval ----------
m.main(["eval"])
with open(m.OUT_DIR / "per_run_errors.csv") as f:
    rows = list(csv.DictReader(f))
assert len(rows) == 2 * 3 * (2 * 2 + 1)                       # models x seeds x (train, val per training run; test)
for r in rows:
    for c in ("all", "wet", "dry"):
        assert float(r[f"e_{c}"]) < 2e-6, r
    assert abs(float(r["mean_ratio"]) - 1.0) < 1e-5, r
r = next(r for r in rows if r["run"] == "Ra_1e6_Pr_1" and r["set"] == "val")
fr = list(range(17, 27, 3))
Q = (raw["Ra_1e6_Pr_1"]["Qd"] + raw["Ra_1e6_Pr_1"]["Qc"])[:, :, fr]
prof = Q.mean(axis=(1, 2))
assert math.isclose(float(r["ref_all"]), math.sqrt(np.sum((prof[:, None, None] - Q) ** 2) / np.sum(Q ** 2)), rel_tol=1e-6)
assert "recomputed" in (m.OUT_DIR / "summary_eval.txt").read_text()
print("E. eval: errors at round-off for Q_a and sum on all, wet and dry cells; z-profile reference as computed directly")

# ---------- F. part dns ----------
m.main(["dns"])
dn = dict(np.load(m.OUT_DIR / "dns_vs_cnn.npz"))
l = "Ra_8.7e6_Pr_1"
qc_dm = raw[l]["Qc"].mean(axis=(0, 1))
assert list(dn[f"{l}__frames"]) == [25 + int(np.argmax(qc_dm[25:])), 25 + int(np.argmin(qc_dm[25:]))]
for tag in ("Q_a", "sum"):
    a = {k: dn[f"{l}__{tag}__42__{k}"] for k in ("sse", "sy2", "dm")}
    assert math.sqrt(a["sse"].sum() / a["sy2"].sum()) < 2e-6
    assert np.allclose(a["dm"], (raw[l]["Qd"] + raw[l]["Qc"]).mean(axis=(0, 1)), rtol=0, atol=1e-6)
    assert 0 < dn[f"{l}__{tag}__joint"].sum() <= (NT - 25) * NZ * NY
    assert np.allclose(dn[f"{l}__{tag}__42__prof"], (raw[l]["Qd"] + raw[l]["Qc"])[:, :, 25:].mean(axis=(1, 2)), rtol=0, atol=1e-6)
    assert math.sqrt(a["sse"][:25].sum() / a["sy2"][:25].sum()) < 2e-6
try:
    m.main(["dns"])
    raise AssertionError("dns overwrote its outputs")
except SystemExit:
    pass
for f in m.PART_OUT["dns"]:
    assert (m.OUT_DIR / f).is_file(), f
shutil.rmtree(tmp)
print("F. dns: every frame; errors at round-off; t* and t_q from the DNS Q_c; refuses to overwrite")
print("ALL v035 Q ANALYSIS TESTS PASSED")
