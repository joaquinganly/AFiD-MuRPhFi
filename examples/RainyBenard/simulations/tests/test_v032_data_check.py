"""Test of "v032 data check.py" on synthetic runs: smooth runs pass, a run with
grid-scale noise and b < 0 is flagged, frame 0 = 0 passes, and the shares are
right (two runs with the same Q_d / kappa get equal kappa-scaled shares)."""
import importlib.util
import json
import math
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

spec = importlib.util.spec_from_file_location("v032_check", Path(sys.argv[1]))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
T = m.T
rng = np.random.default_rng(0)
T.NZ, T.NY = 16, 64
tmp = Path(tempfile.mkdtemp())
T.BASE_DATA_DIR, m.OUT_DIR = tmp, tmp / "check"
train, val, held = ["Ra_1e4", "Ra_1e6", "Pr_05_Ra_1e4", "Pr_2_Ra_1e8"], ["Ra_1e7"], ["Ra_5e6"]
T.ALL_RUNS, T.TRAIN_RUNS, T.VAL_RUNS, T.HELD_OUT_RUNS = train + val + held, train, val, held
m.DROP_RUNS, m.LATE_FRAMES = ["Pr_05_Ra_1e4"], 5
nt = 12
z = (np.arange(T.NZ) + 0.5) / T.NZ
y = np.arange(T.NY) / T.NY
pattern = rng.normal(size=(T.NZ, T.NY, nt))
for label in T.ALL_RUNS:
    d = tmp / label
    (d / "outputdir").mkdir(parents=True)
    with h5py.File(d / "cordin_info.h5", "w") as f:
        f["xm"] = z
    b = np.zeros((T.NZ, T.NY, nt))
    for n in range(1, nt):
        b[:, :, n] = z[:, None] * 0.9 + 0.05 + 0.03 * np.sin(2 * np.pi * (y[None, :] + 0.01 * n)) * np.sin(np.pi * z[:, None])
    if label == "Pr_05_Ra_1e4":
        b[:, :, 1:] += 0.4 * (-1.0) ** (np.arange(T.NZ)[:, None, None] + np.arange(T.NY)[None, :, None])
    ra, pr = T.extract_ra_pr(label)
    kappa = 1 / math.sqrt(ra * pr)
    Qd = kappa * (1.0 + pattern)               # same Q_d / kappa in every run
    np.savez_compressed(d / f"{label}_fields_Q.npz", u=b, v=b, theta=b, rel_q=b, Q=b)
    np.savez_compressed(d / f"{label}_q.npz", q=b)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qd)

m.main()
res = json.loads((m.OUT_DIR / "data_check.json").read_text())
flags = {l: c["flags"] for l, c in res["b_checks"].items()}
assert flags["Pr_05_Ra_1e4"] and "b < -0.01" in flags["Pr_05_Ra_1e4"], flags
assert "grid-scale noise (y)" in flags["Pr_05_Ra_1e4"] and "zigzag in z" in flags["Pr_05_Ra_1e4"]
assert all(not f for l, f in flags.items() if l != "Pr_05_Ra_1e4"), flags
assert "Ra_5e6" not in res["qd_stats"]                     # held-out targets not read
kept = [l for l in train if l != "Pr_05_Ra_1e4"]
for key in ("share_qd", "share_qd_over_kappa"):
    assert sorted(res[key]) == sorted(kept) and abs(sum(res[key].values()) - 100) < 1e-9
sk = res["share_qd_over_kappa"]
assert np.allclose(list(sk.values()), 100 / 3, rtol=1e-9), sk   # equal Q_d / kappa -> equal shares
assert res["share_qd"]["Ra_1e4"] > 80     # largest kappa dominates unscaled (the rest: mean offset)
assert abs(res["qd_stats"]["Ra_1e6"]["rms_qd_over_kappa"]
           - np.sqrt(np.mean((1 + pattern[:, :, 1:nt]) ** 2))) < 1e-9
try:
    m.main(); raise AssertionError("expected SystemExit")
except SystemExit:
    pass
print("DATA CHECK TESTS PASSED")
