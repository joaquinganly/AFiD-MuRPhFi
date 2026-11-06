# ============================================================
# v031 - evaluation of the Q_d stencil CNNs (every trained seed)
#
# Per-frame metrics as in "Qd_infer_test_new_training_method_bq_into_Qd.py"
# (v030), so that the numbers can be compared:
#   full field    : QRMSE = RMSE / RMS(Q_d true), RMSE, MAE
#   active region : the same over |Q_d true| >= 95th percentile of the frame
#   domain mean   : <Q_d>(t), predicted and true
# For comparing training and validation loss, also: the MSE in the
# normalised units of the training loss, and the mean square of the
# normalised target (its "power").
# Sets   : training and validation runs (loss check) and held-out runs
#          (inference), all from config.json.
# Inputs, normalisation and padding come from the training script, which is
# imported; they are checked against v031/Q_d/config.json and
# normalization_constants.npz.
# Output : v031/Q_d/eval/ (refuses to overwrite)
#          eval_summary.txt   loss check and per-run metrics
#          metrics.npz        per-frame metrics, per run and seed
#          eval_held_out.png  v030-style panels for the held-out runs
#          snapshot_<run>.png true / predicted / error, last frame, held-out runs
#          qrmse_by_run.png   full-field QRMSE of every run vs Ra
#          power_by_frame.png normalised target power and MSE per frame
# Usage  : python "v031 evaluate Q_d.py"
# ============================================================

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET = "Q_d"
ACTIVE_PERCENTILE = 95          # as in the v030 evaluation
BATCH_SIZE = 8
SET_COLORS = {"train": "tab:blue", "val": "tab:orange", "held_out": "tab:red"}
PR_MARKERS = {0.5: "v", 1.0: "o", 2.0: "^"}
METRICS = ("full_qrmse", "full_rmse", "full_mae", "active_qrmse", "active_rmse",
           "active_mae", "domain_mean_pred", "domain_mean_true", "mse_norm",
           "power_norm")

SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v031_train", SCRIPT_DIR / "v031 stencil CNN.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)      # registers StencilPad, needed to load the models


# ============================================================
# 1. Setup from the training outputs
# ============================================================

def load_setup(target_dir):
    cfg = json.loads((target_dir / "config.json").read_text())
    summary = json.loads((target_dir / "summary.json").read_text())
    expected = {"inputs": list(T.INPUTS), "target": TARGET, "train_runs": T.TRAIN_RUNS,
                "val_runs": T.VAL_RUNS, "held_out_runs": T.HELD_OUT_RUNS,
                "grid": [T.NZ, T.NY]}
    differ = [k for k, v in expected.items() if cfg.get(k) != v]
    if differ:
        sys.exit(f"config.json differs from the training script in {differ}")
    with np.load(target_dir / "normalization_constants.npz") as f:
        input_norm = {str(c): (float(m), float(s))
                      for c, m, s in zip(f["inputs"], f["input_mean"], f["input_std"])}
        target_norm = (float(f["target_mean"]), float(f["target_std"]))
    if not (T._same(json.loads(json.dumps(input_norm)), cfg["input_norm"])
            and T._same(list(target_norm), cfg["target_norm"])):
        sys.exit("normalization_constants.npz differs from config.json")
    return cfg, summary, input_norm, target_norm


# ============================================================
# 2. Metrics
# ============================================================

def frame_metrics(m, n, pred, true, pred_norm, true_norm):
    """v030 definitions (same 1e-12 guard), float64."""
    err = pred - true
    m["full_rmse"][n] = np.sqrt(np.mean(err ** 2))
    m["full_mae"][n] = np.mean(np.abs(err))
    m["full_qrmse"][n] = m["full_rmse"][n] / (np.sqrt(np.mean(true ** 2)) + 1e-12)
    mask = np.abs(true) >= np.percentile(np.abs(true), ACTIVE_PERCENTILE)
    m["active_rmse"][n] = np.sqrt(np.mean(err[mask] ** 2))
    m["active_mae"][n] = np.mean(np.abs(err[mask]))
    m["active_qrmse"][n] = m["active_rmse"][n] / (np.sqrt(np.mean(true[mask] ** 2)) + 1e-12)
    m["domain_mean_pred"][n] = pred.mean()
    m["domain_mean_true"][n] = true.mean()
    m["mse_norm"][n] = np.mean((pred_norm - true_norm) ** 2)
    m["power_norm"][n] = np.mean(true_norm ** 2)


def evaluate_run(label, nt, input_norm, target_norm, models, snapshot_seed):
    """Predicts every frame of one run with every model. Returns per-frame
    metrics {seed: {metric: (nt,)}} and the last frame (true, predicted by
    snapshot_seed)."""
    run = T.load_run(label, nt)
    T.apply_input_norm(run, input_norm)
    gen = T.RunSnapshots([run], TARGET, target_norm, BATCH_SIZE, shuffle=False, seed=0)
    truth = T.target_array(run, TARGET)
    mean, std = target_norm
    res = {s: {k: np.empty(nt) for k in METRICS} for s in models}
    snapshot = None
    n0 = 0
    for i in range(len(gen)):
        X, Y = gen[i]
        y_norm = Y[..., 0].astype(np.float64)
        y_true = truth[n0:n0 + len(X)].astype(np.float64)
        for seed, model in models.items():
            p_norm = np.asarray(model.predict_on_batch(X))[..., 0].astype(np.float64)
            p = p_norm * std + mean
            for k in range(len(X)):
                frame_metrics(res[seed], n0 + k, p[k], y_true[k], p_norm[k], y_norm[k])
            if seed == snapshot_seed and n0 + len(X) == nt:
                snapshot = (y_true[-1], p[-1])
        n0 += len(X)
    return res, snapshot


def run_summary(res):
    """Per-run scalars for each seed (v030 aggregation: mean over frames)."""
    out = {}
    for seed, m in res.items():
        err_ts = m["domain_mean_pred"] - m["domain_mean_true"]
        out[seed] = {
            "full_qrmse": np.mean(m["full_qrmse"]),
            "full_rmse": np.mean(m["full_rmse"]),
            "full_mae": np.mean(m["full_mae"]),
            "active_qrmse": np.mean(m["active_qrmse"]),
            "mean_ts_qrmse": np.sqrt(np.mean(err_ts ** 2))
                             / (np.sqrt(np.mean(m["domain_mean_true"] ** 2)) + 1e-12),
            "mse_norm": np.mean(m["mse_norm"]),
            "power_norm": np.mean(m["power_norm"]),
        }
    return out


# ============================================================
# 3. Figures
# ============================================================

def plot_held_out(results, sets, seeds, path):
    labels = [l for l in results if sets[l] == "held_out"]
    fig, axes = plt.subplots(3, len(labels), figsize=(7 * len(labels), 12),
                             sharex="col", squeeze=False)
    for j, label in enumerate(labels):
        res = results[label]
        frames = np.arange(len(res[seeds[0]]["full_qrmse"]))
        ax0, ax1, ax2 = axes[:, j]
        ax0.plot(frames, res[seeds[0]]["domain_mean_true"], color="black", linewidth=2.0,
                 label="Ground truth")
        for seed in seeds:
            ax0.plot(frames, res[seed]["domain_mean_pred"], linestyle="--", linewidth=1.2,
                     label=f"Predicted, seed {seed}")
            ax1.plot(frames, res[seed]["active_qrmse"], linewidth=1.2, label=f"seed {seed}")
            ax2.plot(frames, res[seed]["full_qrmse"], linewidth=1.2, label=f"seed {seed}")
        ax0.set_title(f"{T.VERSION}: Q_d, held-out run [{label}]", fontsize=11, fontweight="bold")
        ax0.set_ylabel(r"Domain mean $\langle Q_d \rangle$")
        ax1.set_ylabel(f"Active QRMSE (>= {ACTIVE_PERCENTILE}th %)")
        ax2.set_ylabel("Full-field QRMSE")
        ax2.set_xlabel("Time index (n)")
        for ax in (ax1, ax2):
            ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1)
            ax.set_yscale("log")
        for ax in (ax0, ax1, ax2):
            ax.grid(True, which="both", linestyle=":", alpha=0.6)
            ax.legend(fontsize=8)
    fig.suptitle(f"{T.VERSION} 3x3 stencil CNN, Q_d, held-out runs "
                 f"(dotted line: QRMSE = 1, the error of predicting zero)", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_snapshot(label, true, pred, seed, frame, path):
    err = pred - true
    vmax = np.percentile(np.abs(true), 99.5)
    emax = np.percentile(np.abs(err), 99.5)
    extent = (0.0, T.NY / T.NZ, 0.0, 1.0)            # dy = dz = 1/NZ, z in [0, 1]
                                                     # (continua_master.h5: ylen = 16)
    qrmse = np.sqrt(np.mean(err ** 2)) / (np.sqrt(np.mean(true ** 2)) + 1e-12)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    panels = ((true, vmax, "Ground truth Q_d"), (pred, vmax, f"Predicted (seed {seed})"),
              (err, emax, f"Predicted - true (frame QRMSE {qrmse:.3g})"))
    for ax, (field, lim, title) in zip(axes, panels):
        im = ax.imshow(field, origin="lower", extent=extent, aspect="auto",
                       cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("z")
        fig.colorbar(im, ax=ax, pad=0.01, label="(colour range: 99.5th %-ile of |.|)")
    axes[-1].set_xlabel("y")
    fig.suptitle(f"{T.VERSION}: Q_d, held-out run {label}, frame {frame}")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qrmse_by_run(per_run, sets, seeds, path):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    used = set()
    for label, summ in per_run.items():
        ra, pr = T.extract_ra_pr(label)
        vals = np.array([summ[s]["full_qrmse"] for s in seeds])
        x = np.log10(ra) + {0.5: -0.08, 1.0: 0.0, 2.0: 0.08}.get(pr, 0.0)
        key = (sets[label], pr)
        ax.errorbar(x, vals.mean(), yerr=[[vals.mean() - vals.min()], [vals.max() - vals.mean()]],
                    fmt=PR_MARKERS.get(pr, "s"), color=SET_COLORS[sets[label]], capsize=3,
                    label=f"{sets[label]}, Pr = {pr:g}" if key not in used else None)
        used.add(key)
    ax.axhline(1.0, color="0.4", linestyle=":", linewidth=1, label="QRMSE = 1 (predicting zero)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log_{10}$ Ra")
    ax.set_ylabel("Full-field QRMSE (mean over frames; bar: range over seeds)")
    ax.set_title(f"{T.VERSION}: Q_d, every run")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_power_by_frame(results, sets, seed, path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    used = set()
    for label, res in results.items():
        m = res[seed]
        frames = np.arange(len(m["power_norm"]))
        lab = sets[label] if sets[label] not in used else None
        used.add(sets[label])
        axes[0].plot(frames, m["power_norm"], color=SET_COLORS[sets[label]], linewidth=0.8,
                     alpha=0.8, label=lab)
        axes[1].plot(frames, m["mse_norm"], color=SET_COLORS[sets[label]], linewidth=0.8,
                     alpha=0.8, label=lab)
    axes[0].set_ylabel("Mean square of normalised Q_d")
    axes[1].set_ylabel(f"MSE, normalised (seed {seed})")
    axes[1].set_xlabel("Time index (n)")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.6)
        ax.legend(fontsize=8)
    axes[0].set_title(f"{T.VERSION}: Q_d per frame, one line per run (normalised units of "
                      f"the training loss)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 4. Main
# ============================================================

def main():
    t_start = time.time()
    target_dir = T.OUT_DIR / TARGET
    eval_dir = target_dir / "eval"
    if eval_dir.exists() and any(eval_dir.iterdir()):
        sys.exit(f"{eval_dir} exists and is not empty; refusing to overwrite.")
    T.configure_gpu()                                # TF32 off, as in training
    cfg, summary, input_norm, target_norm = load_setup(target_dir)
    seeds = [s["seed"] for s in summary]
    best = {s["seed"]: s for s in summary}
    snapshot_seed = min(summary, key=lambda s: s["best_val_mse_normalised"])["seed"]
    nt = cfg["nt_common"]

    models = {s: tf.keras.models.load_model(target_dir / f"seed_{s}" / "model.keras",
                                            compile=False) for s in seeds}
    sets = {**{l: "train" for l in cfg["train_runs"]}, **{l: "val" for l in cfg["val_runs"]},
            **{l: "held_out" for l in cfg["held_out_runs"]}}

    results, per_run, snapshots = {}, {}, {}
    print(f"Evaluating seeds {seeds} on {len(sets)} runs, {nt} frames each")
    for label in sets:
        results[label], snap = evaluate_run(label, nt, input_norm, target_norm, models,
                                            snapshot_seed)
        per_run[label] = run_summary(results[label])
        if sets[label] == "held_out":
            snapshots[label] = snap

    eval_dir.mkdir(parents=True, exist_ok=True)
    np.savez(eval_dir / "metrics.npz",
             **{f"{l}/seed_{s}/{k}": v for l, res in results.items()
                for s, m in res.items() for k, v in m.items()},
             runs=np.array(list(sets)), sets=np.array([sets[l] for l in sets]))

    # ---- text summary ----
    def pooled(name, which, seed):
        return float(np.mean([per_run[l][seed][name] for l in sets if sets[l] == which]))

    lines = [f"{T.VERSION} evaluation of {TARGET}: seeds {seeds}, {nt} frames per run, "
             f"target normalisation (mean, std) = {target_norm}", ""]
    lines += ["1. Training loss vs validation loss (normalised MSE, as in the Keras loss)",
              "   logged = Keras history at the best epoch (training value = mean over the",
              "   epoch's batches while the weights change); fixed = this script, saved model.",
              f"   {'seed':>4} {'best ep':>7} {'train logged':>13} {'train fixed':>12} "
              f"{'val logged':>11} {'val fixed':>11}"]
    for seed in seeds:
        hist = json.loads((target_dir / f"seed_{seed}" / "history.json").read_text())
        b = best[seed]["best_epoch"] - 1
        lines.append(f"   {seed:>4} {b + 1:>7} {hist['loss'][b]:>13.4e} "
                     f"{pooled('mse_norm', 'train', seed):>12.4e} {hist['val_loss'][b]:>11.4e} "
                     f"{pooled('mse_norm', 'val', seed):>11.4e}")
    lines.append("")

    train_power = sum(per_run[l][seeds[0]]["power_norm"] for l in sets if sets[l] == "train")
    train_sq = {s: sum(per_run[l][s]["mse_norm"] for l in sets if sets[l] == "train")
                for s in seeds}
    lines += ["2. Per run (seed mean; [min, max] over seeds for the full-field QRMSE)",
              "   power = mean square of the normalised target; shares are of the training runs' totals.",
              "   QRMSE = 1 is the error of predicting zero everywhere.",
              f"   {'set':<8} {'run':<14} {'power':>10} {'power %':>8} {'MSE norm':>10} "
              f"{'MSE %':>7} {'full QRMSE':>11} {'[min, max]':>21} {'RMSE':>10} {'MAE':>10} "
              f"{'active QRMSE':>12} {'<Q_d>(t) QRMSE':>14}"]
    for label in sets:
        r = per_run[label]
        q = np.array([r[s]["full_qrmse"] for s in seeds])
        avg = {k: np.mean([r[s][k] for s in seeds]) for k in r[seeds[0]]}
        is_train = sets[label] == "train"
        p_share = f"{100 * avg['power_norm'] / train_power:8.3f}" if is_train else f"{'-':>8}"
        e_share = (f"{100 * np.mean([r[s]['mse_norm'] / train_sq[s] for s in seeds]):7.3f}"
                   if is_train else f"{'-':>7}")
        lines.append(f"   {sets[label]:<8} {label:<14} {avg['power_norm']:>10.3e} {p_share} "
                     f"{avg['mse_norm']:>10.3e} {e_share} {avg['full_qrmse']:>11.3e} "
                     f"[{q.min():.3e}, {q.max():.3e}] {avg['full_rmse']:>10.3e} "
                     f"{avg['full_mae']:>10.3e} {avg['active_qrmse']:>12.3e} "
                     f"{avg['mean_ts_qrmse']:>14.3e}")
    text = "\n".join(lines) + "\n"
    (eval_dir / "eval_summary.txt").write_text(text)
    print("\n" + text)

    # ---- figures ----
    plot_held_out(results, sets, seeds, eval_dir / "eval_held_out.png")
    for label, (true, pred) in snapshots.items():
        plot_snapshot(label, true, pred, snapshot_seed, nt - 1, eval_dir / f"snapshot_{label}.png")
    plot_qrmse_by_run(per_run, sets, seeds, eval_dir / "qrmse_by_run.png")
    plot_power_by_frame(results, sets, snapshot_seed, eval_dir / "power_by_frame.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
