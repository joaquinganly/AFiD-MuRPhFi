"""Test of "v032 evaluate Q_d.py" on small synthetic runs (TensorFlow CPU).

Trains Q_d for 2 epochs with the v032 training script (frame 0 dropped,
run-weighted loss), then evaluates, and checks:
- the fixed-weight validation loss (mean relative MSE over runs) and the
  unweighted validation MSE equal Keras' val_loss and val_mse_target at the
  best epoch;
- held-out frames' metrics match a computation from the raw arrays (sample k
  is DNS frame k + 1), with inputs built independently;
- training outputs untouched; outputs exist; a second evaluation refuses to
  overwrite.
"""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf

spec = importlib.util.spec_from_file_location("v032_eval", Path(sys.argv[1]))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)
T = ev.T
rng = np.random.default_rng(1)

T.NZ, T.NY = 8, 16
T.EPOCHS, T.SEEDS, T.BATCH_SIZE = 2, (1, 2), 2
tmp = Path(tempfile.mkdtemp())
T.BASE_DATA_DIR, T.OUT_DIR = tmp, tmp / "v032"
train, val, held = ["Pr_05_Ra_1e5", "Ra_1e5", "Pr_2_Ra_1e6"], ["Ra_1e6", "Pr_2_Ra_1e4"], ["Ra_5e6", "Ra_25e7"]
T.ALL_RUNS, T.TRAIN_RUNS, T.VAL_RUNS, T.HELD_OUT_RUNS = train + val + held, train, val, held
T.DROPPED_RUNS, T.USED_RUNS = {}, train + val + held
xm = (np.arange(T.NZ) + 0.5) / T.NZ
raw = {}
for i, label in enumerate(T.ALL_RUNS):
    d = tmp / label
    (d / "outputdir").mkdir(parents=True)
    with h5py.File((d if i % 2 else d / "outputdir") / "cordin_info.h5", "w") as f:
        f["xm"] = xm
    nt_f = 9 if label != "Ra_1e5" else 7
    theta = rng.random((T.NZ, T.NY, nt_f)); q = rng.random((T.NZ, T.NY, nt_f))
    Qd = rng.normal(size=(T.NZ, T.NY, 8)) * (i + 1); Qc = rng.random((T.NZ, T.NY, 8))
    np.savez_compressed(d / f"{label}_fields_Q.npz", u=theta, v=theta, theta=theta, rel_q=q, Q=theta)
    np.savez_compressed(d / f"{label}_q.npz", q=q)
    np.savez_compressed(d / "Q_d_c.npz", Q_d=Qd, Q_c=Qc)
    raw[label] = dict(theta=theta, q=q, Qd=Qd)

T.main(["Q_d"])
ev.BATCH_SIZE = 4                                  # 6 samples -> batches of 4, 2
trained = {p: p.read_bytes() for p in (T.OUT_DIR / "Q_d").rglob("*") if p.is_file()}
ev.main()
for p, content in trained.items():                 # training outputs untouched
    assert p.read_bytes() == content, p

out = T.OUT_DIR / "Q_d"
ed = out / "eval"
for name in ["eval_summary.txt", "metrics.npz", "eval_held_out.png", "qrmse_by_run.png",
             "power_by_frame.png", "snapshot_Ra_5e6.png", "snapshot_Ra_25e7.png"]:
    assert (ed / name).is_file(), name
met = np.load(ed / "metrics.npz")
summary = json.loads((out / "summary.json").read_text())
assert json.loads((out / "config.json").read_text())["nt_common"] == 7

# fixed weights: val loss (mean relative MSE over runs) and unweighted val MSE == Keras
for s in summary:
    hist = json.loads((out / f"seed_{s['seed']}" / "history.json").read_text())
    b = s["best_epoch"] - 1
    rel = [np.mean(met[f"{l}/seed_{s['seed']}/full_rmse"] ** 2) / np.mean(met[f"{l}/seed_{s['seed']}/ms_true"])
           for l in val]
    assert np.isclose(np.mean(rel), hist["val_loss"][b], rtol=1e-4), (np.mean(rel), hist["val_loss"][b])
    mse = np.mean([met[f"{l}/seed_{s['seed']}/mse_norm"] for l in val])
    assert np.isclose(mse, hist["val_mse_target"][b], rtol=1e-4), (mse, hist["val_mse_target"][b])

# held-out samples from the raw arrays: sample k = DNS frame k + 1
norm = np.load(out / "normalization_constants.npz")
mu, sd = norm["input_mean"], norm["input_std"]
tmu, tsd = float(norm["target_mean"]), float(norm["target_std"])
label, seed = "Ra_25e7", 2
r = raw[label]
model = tf.keras.models.load_model(out / f"seed_{seed}" / "model.keras", compile=False)
for k in (1, 5):                                   # second batch; last (partial) batch
    n = k + 1
    X = np.stack([r["theta"][:, :, n], r["q"][:, :, n], np.repeat(xm[:, None], T.NY, 1),
                  np.full((T.NZ, T.NY), np.log10(2.5e8)), np.zeros((T.NZ, T.NY))], -1)
    X = ((X - mu) / sd)[None].astype(np.float32)
    p = model.predict(X, verbose=0)[0, ..., 0].astype(np.float64) * tsd + tmu
    true = r["Qd"][:, :, n]
    err = p - true
    mask = np.abs(true) >= np.percentile(np.abs(true), 95)
    expect = {
        "full_qrmse": np.sqrt(np.mean(err ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12),
        "full_mae": np.mean(np.abs(err)),
        "active_qrmse": np.sqrt(np.mean(err[mask] ** 2)) / (np.sqrt(np.mean(true[mask] ** 2)) + 1e-12),
        "domain_mean_pred": p.mean(),
        "domain_mean_true": true.mean(),
        "ms_true": np.mean(true ** 2),
        "power_norm": np.mean(((true - tmu) / tsd) ** 2),
    }
    for key, v in expect.items():
        assert np.isclose(met[f"{label}/seed_{seed}/{key}"][k], v, rtol=1e-4, atol=1e-7), (k, key)
assert len(met[f"{label}/seed_{seed}/full_qrmse"]) == 6

text = (ed / "eval_summary.txt").read_text()
assert all(l in text for l in train + val + held)
try:
    ev.main(); raise AssertionError("expected SystemExit")
except SystemExit:
    pass
print("EVALUATION TESTS PASSED")
