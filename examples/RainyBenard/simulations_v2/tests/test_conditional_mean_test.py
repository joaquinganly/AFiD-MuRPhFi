"""Test of diagnostics/conditional_mean_test.py (CPU; plain script, no pytest). On small synthetic fields:
 1. template_coeff / features = the patch (jacobian_attribution.gather_patches) dotted with u00, u10 at every
    interior cell, including the periodic wrap in y; a00 = 3 x patch mean;
 2. bin_index: edges[k] <= v < edges[k+1], ends clamped; fit_partition + assign: every 2D bin holds the same
    number of fitting points (+-1), and the 2D index is consistent with the 1D one;
 3. fit_estimators: a target constant on each 2D bin is reproduced exactly (also on new points); a bin
    below the minimum count takes its a00 bin's value, an a00 bin below it the overall mean; run weights:
    m = sum_r w_r s_r / sum_r w_r n_r;
 4. rel and unexplained against direct formulas; 1 - eta^2 = 0 for a function of the bin and ~ the noise
    share for function + noise;
 5. prepare_run and pool end to end with psi = the 5-point Laplacian of b: the z estimator = row means of
    the training sample; psi^ = psi gives e = 0 and d(psi^, E) = e(E); psi^ = E[psi | a00, a10] gives
    d = 0 and 1 - eta^2(psi^ | a00, a10) = 0; pooled weights as 3;
 6. psi_from_prediction (target Q_d: / kappa; target Q_d / kappa: not); summary_text, save_data and the
    three figures run (mathtext renders).
 7. (2026-11-18) load_results returns what save_data wrote; e* (the error of the bin means of psi over the same
    cells) is computed as such and bounds every bin-constant prediction from below; convergence() over two bin
    counts of the same data: a shared bin count gives identical 1 - eta^2 and e*, the summary and both figures
    are written; main's argument parsing.
Usage (from simulations_v2/): python tests/test_conditional_mean_test.py diagnostics/conditional_mean_test.py
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import importlib.util
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("conditional_mean_test", script)
cm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cm)
ja = cm.ja
rng = np.random.default_rng(5)
NZ, NY = 12, 64

# ---------- 1. template coordinates ----------
B = rng.random((3, NZ, NY))
rows, cols = np.arange(1, NZ - 1), np.arange(NY)
P = ja.gather_patches(B, rows, cols).reshape(3, NZ - 2, NY, 9)
assert not np.isnan(P).any()
for u in (cm.U00, cm.U10, ja.TEMPLATES[(2, 2)]):
    assert np.allclose(cm.template_coeff(B, u), P @ u)
a0, a1 = cm.features(B)
assert np.allclose(a0, 3.0 * P.mean(axis=-1).ravel()) and np.allclose(a1, (P @ cm.U10).ravel())
assert math.isclose(a1.reshape(3, NZ - 2, NY)[1, 4, 0],
                    (B[1, 6, [-1, 0, 1]].sum() - B[1, 4, [-1, 0, 1]].sum()) / math.sqrt(6.0))
print("1. template coordinates = patch . u (periodic in y): OK")

# ---------- 2. bins ----------
e = np.array([0.0, 1.0, 2.0, 3.0])
assert list(cm.bin_index(np.array([-5.0, 0.0, 0.5, 1.0, 2.9, 3.0, 7.0]), e)) == [0, 0, 0, 1, 2, 2, 2]
nb, m = 5, 40
x0, x1 = rng.normal(size=nb * nb * m), rng.standard_cauchy(size=nb * nb * m)
part = cm.fit_partition(x0, x1, nb)
i0, i2 = cm.assign(x0, x1, part)
c2 = np.bincount(i2, minlength=nb * nb)
assert c2.min() >= m - 1 and c2.max() <= m + 1, c2
assert np.array_equal(i2 // nb, i0)
print(f"2. bin edges and equal-count 2D partition (counts {c2.min()}..{c2.max()}): OK")

# ---------- 3. estimators ----------
f = rng.normal(size=nb * nb)
t = cm.run_table(i0, i2, f[i2], nb)
est = cm.fit_estimators([(1.0, t)], nb)
assert np.allclose(est["m2"], f) and est["ok2"].all()
y0, y1 = rng.normal(size=500), rng.standard_cauchy(size=500)
assert np.allclose(est["m2"][cm.assign(y0, y1, part)[1]], f[cm.assign(y0, y1, part)[1]])
t_small = {k: v.copy() for k, v in t.items()}
t_small["n2"][7], t_small["s2"][7] = 3.0, 99.0                       # below MIN_COUNT
t_small["n1"][2], t_small["s1"][2] = 5.0, 99.0
est_s = cm.fit_estimators([(1.0, t_small)], nb)
assert math.isclose(est_s["m2"][7], est_s["m1"][7 // nb]) and not est_s["ok2"][7]
assert math.isclose(est_s["m1"][2], t_small["s1"].sum() / t_small["n1"].sum())
yA, yB = rng.normal(size=x0.size), 10.0 * rng.normal(size=x0.size)
tA, tB = cm.run_table(i0, i2, yA, nb), cm.run_table(i0, i2[::-1].copy(), yB, nb)
wA, wB = 2.0, 0.03
est_w = cm.fit_estimators([(wA, tA), (wB, tB)], nb)
assert np.allclose(est_w["m1"], (wA * tA["s1"] + wB * tB["s1"]) / (wA * tA["n1"] + wB * tB["n1"]))
assert np.allclose(est_w["m2"], (wA * tA["s2"] + wB * tB["s2"]) / (wA * tA["n2"] + wB * tB["n2"]))
print("3. bin means exact, fallbacks, run weights: OK")

# ---------- 4. metrics ----------
g, h = rng.normal(size=1000), rng.normal(size=1000)
assert math.isclose(cm.rel(g, h, 7.0), math.sqrt(np.sum((g - h) ** 2) / 7.0))
assert cm.unexplained(f[i2], i2, nb * nb) < 1e-24
noisy = 3.0 * f[i2] + rng.normal(size=i2.size)
share = 1.0 / np.var(noisy)
assert abs(cm.unexplained(noisy, i2, nb * nb) - share) < 0.3 * share, (cm.unexplained(noisy, i2, nb * nb), share)
yy = rng.normal(size=i2.size)
mb = np.array([yy[i2 == k].mean() for k in range(nb * nb)])
assert math.isclose(cm.unexplained(yy, i2, nb * nb), np.sum((yy - mb[i2]) ** 2) / np.sum((yy - yy.mean()) ** 2))
print("4. e, d, 1 - eta^2: OK")

# ---------- 5. end to end ----------
def laplacian(b):                                  # interior rows exact, wall rows unused; dz = dy = 1
    lap = np.zeros_like(b)
    lap[:, 1:-1] = (b[:, 2:] + b[:, :-2] + np.roll(b, 1, 2)[:, 1:-1] + np.roll(b, -1, 2)[:, 1:-1] - 4 * b[:, 1:-1])
    return lap


def field(n, amp):
    zz = np.linspace(0, 1, NZ)[None, :, None]
    yy = np.linspace(0, 2 * np.pi, NY, endpoint=False)[None, None, :]
    ph = rng.uniform(0, 2 * np.pi, size=(n, 1, 1))
    return (zz + amp * np.sin(np.pi * zz) * np.cos(3 * yy + ph) + 0.02 * rng.normal(size=(n, NZ, NY))).astype(np.float32)


runs = {}
for label, amp in (("Ra_1e6_Pr_1", 0.05), ("Ra_1e8_Pr_1", 0.3)):
    Bt, Bv = field(8, amp), field(4, amp)
    Yt, Yv = laplacian(Bt.astype(np.float64)), laplacian(Bv.astype(np.float64))
    r = cm.prepare_run(label, Bt, Yt, Bv, Yv, nb=4, nb_coarse=2)
    r.update(ra=float(label.split("_")[1]), frames={"train": list(range(8)), "val": list(range(4))})
    assert np.allclose(r["c"]["z"].reshape(4, NZ - 2, NY)[0, :, 0], Yt[:, 1:-1].mean(axis=(0, 2)), atol=1e-6)
    assert math.isclose(r["w"], 1.0 / np.mean(Yt[:, 1:-1] ** 2), rel_tol=1e-6)
    runs[label] = r
tr_copy = {l: tuple(np.array(x, np.float64) for x in r["train"]) for l, r in runs.items()}
pooled = cm.pool(runs, nb=4, qsample=10 ** 9)
P4 = pooled["part"]
tabs = [(runs[l]["w"], cm.run_table(*cm.assign(a, b, P4), y, 4)) for l, (a, b, y) in tr_copy.items()]
ref = cm.fit_estimators(tabs, 4)
assert np.allclose(pooled["est"]["m2"], ref["m2"])
est_m, res = {}, {}
for l, r in runs.items():
    est_m[l] = cm.estimator_metrics(r)
    psi = r["psi"].astype(np.float64)
    q = cm.model_metrics(psi, r)
    assert q["e"] == 0.0
    assert all(math.isclose(q["d"][k], est_m[l]["e"][k], rel_tol=1e-9) for k in q["d"])
    assert all(math.isclose(q["u"][k], est_m[l]["u"][k], rel_tol=1e-9) for k in cm.BINS)
    qc = cm.model_metrics(r["c"]["ab"].astype(np.float64), r)
    assert qc["d"]["ab"] == 0.0 and qc["u"]["ab"] < 1e-20 and math.isclose(qc["e"], est_m[l]["e"]["ab"])
    for v in ja.VERSIONS:
        for n in ja.QD:
            for s in (42, 43):
                res[(v, n, s, l)] = q if v == "v034" else qc
assert est_m["Ra_1e8_Pr_1"]["e"]["ab"] < est_m["Ra_1e8_Pr_1"]["e"]["z"]
print("5. prepare_run, pool, estimator and model metrics end to end: OK")

# ---------- 6. conversion, summary, data, figures ----------
p = rng.normal(size=(2, NZ, NY))
setup = {"target_norm": (0.5, 3.0), "cfg": {"target": "Q_d"}}
assert np.allclose(cm.psi_from_prediction(p, setup, 1e-3), ((p * 3.0 + 0.5) / 1e-3)[:, 1:-1].ravel())
setup["cfg"]["target"] = "Q_d_over_kappa"
assert np.allclose(cm.psi_from_prediction(p, setup, 1e-3), (p * 3.0 + 0.5)[:, 1:-1].ravel())
text = cm.summary_text(est_m, res, runs, "test")
assert "=== v033 Q_d_a" in text and "Ra_1e8_Pr_1" in text
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    cm.save_data(est_m, res, runs, pooled, tmp / "d.npz")
    with np.load(tmp / "d.npz") as d:
        assert float(d["v034__Q_d_b__42__Ra_1e6_Pr_1__e"]) == 0.0
        assert math.isclose(float(d["Ra_1e6_Pr_1__e_w_ab"]), est_m["Ra_1e6_Pr_1"]["e"]["w_ab"])
    cm.plot_errors(est_m, res, runs, tmp / "a.png")
    cm.plot_bound(est_m, res, runs, tmp / "b.png")
    cm.plot_unexplained(est_m, {k: {**q, "u": {kk: max(x, 1e-9) for kk, x in q["u"].items()}} for k, q in res.items()},
                        runs, tmp / "c.png")
    assert all((tmp / x).stat().st_size > 10_000 for x in ("a.png", "b.png", "c.png"))
print("6. prediction conversion, summary, data file and the three figures: OK")

# ---------- 7. load_results, bound, convergence ----------
r0 = runs["Ra_1e6_Pr_1"]
m_same = np.bincount(r0["bins"]["ab"][0], weights=r0["psi"].astype(np.float64), minlength=r0["bins"]["ab"][1]) \
    / np.maximum(np.bincount(r0["bins"]["ab"][0], minlength=r0["bins"]["ab"][1]), 1)
e_m = cm.rel(m_same[r0["bins"]["ab"][0]], r0["psi"], r0["norm"])
for g in (m_same * 0.5, m_same + 1.0, rng.normal(size=m_same.size)):
    assert cm.rel(g[r0["bins"]["ab"][0]], r0["psi"], r0["norm"]) >= e_m - 1e-12
assert math.isclose(cm.estimator_metrics(r0)["estar"]["ab"], e_m, rel_tol=1e-12)
with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    dirs = []
    for nb_ in (4, 8):
        rr, ee, re_ = {}, {}, {}
        for label, amp in (("Ra_1e6_Pr_1", 0.05), ("Ra_1e8_Pr_1", 0.3)):
            rng_l = np.random.default_rng(11 if amp < 0.1 else 12)
            zz = np.linspace(0, 1, NZ)[None, :, None]
            yy = np.linspace(0, 2 * np.pi, NY, endpoint=False)[None, None, :]
            Bt = (zz + amp * np.sin(np.pi * zz) * np.cos(3 * yy + rng_l.uniform(0, 6, (8, 1, 1)))
                  + 0.02 * rng_l.normal(size=(8, NZ, NY))).astype(np.float32)
            Bv2 = (zz + amp * np.sin(np.pi * zz) * np.cos(3 * yy + rng_l.uniform(0, 6, (4, 1, 1)))
                   + 0.02 * rng_l.normal(size=(4, NZ, NY))).astype(np.float32)
            Yt, Yv = laplacian(Bt.astype(np.float64)), laplacian(Bv2.astype(np.float64))
            r = cm.prepare_run(label, Bt, Yt, Bv2, Yv, nb=nb_, nb_coarse=nb_ // 2)
            r.update(ra=float(label.split("_")[1]), frames={"train": list(range(8)), "val": list(range(4))})
            rr[label] = r
        pooled_ = cm.pool(rr, nb=nb_, qsample=10 ** 9)
        for label, r in rr.items():
            ee[label] = cm.estimator_metrics(r)
            out = r["psi"].astype(np.float64)
            for v in ja.VERSIONS:
                for n in ja.QD:
                    for s in (42, 43):
                        res_out = out * (0.5 if v == "v033" else 1.0) + (0.01 * s if v == "v033" else 0.0)
                        re_[(v, n, s, label)] = cm.model_metrics(res_out, r)
        cm.NB, cm.NB_COARSE = nb_, nb_ // 2
        d_ = tmp / f"nb{nb_}"
        d_.mkdir()
        cm.save_data(ee, re_, rr, pooled_, d_ / "conditional_mean_data.npz")
        L_ = cm.load_results(d_ / "conditional_mean_data.npz")
        assert L_["nb"] == nb_ and L_["nb_coarse"] == nb_ // 2 and set(L_["runs"]) == set(rr)
        assert all(math.isclose(L_["est"][l]["e"][k], ee[l]["e"][k]) for l in rr for k in cm.EST)
        assert all(math.isclose(L_["res"][k]["u"][j], re_[k]["u"][j]) for k in re_ for j in cm.BINS)
        assert all(math.isclose(L_["runs"][l]["ra"], rr[l]["ra"]) for l in rr)
        assert all(math.isclose(L_["est"][l]["estar"][k], ee[l]["estar"][k]) for l in rr for k in cm.BINS)
        dirs.append(d_)
    cm.convergence([str(x) for x in dirs], out_dir=tmp / "conv")
    txt = (tmp / "conv" / "summary.txt").read_text()
    diff = float(txt.split("max |difference| = ")[1].split()[0])
    assert diff < 1e-12, diff
    assert all((tmp / "conv" / f).stat().st_size > 10_000 for f in ("Qd_bin_convergence_eta.png", "Qd_bin_convergence_bound.png"))
for bad in (["--nb"], ["--nb", "abc"], ["--nb", "0"], ["--nb", "7"], ["--compare", "x"], ["--replot", "x"], ["--foo"]):
    try:
        cm.main(bad)
    except SystemExit:
        pass
    else:
        raise AssertionError(bad)
print("7. load_results round trip, the bound e*, convergence over bin counts (shared count identical): OK")
print("ALL CONDITIONAL-MEAN TESTS PASSED")
