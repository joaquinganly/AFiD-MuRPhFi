"""v034 - evaluation of the round-2 stencil CNNs (every trained configuration and seed).
Copy of "v033 evaluate.py" that imports "v034 stencil CNN.py" (whitened 3 x 3 b patch for the Q_d
configurations, learning rate held for the first 50 epochs; changelog 2026-11-11); the analyses are
unchanged. For a whitened model, analysis 2 still perturbs b itself, so it measures the response of
the whole model (whitening layer and network) to b in each neighbour. The v033 description follows.


Four analyses (agreed with the user, 2026-11-10); Q_d and Q_c results go to separate figures.
 1. Per-run errors. Relative RMS error sqrt(sum (p - y)^2 / sum y^2) of every model on each
    training run's training block and validation block, and on the test runs (Ra 8.7e6, 1.5e8;
    frames of both blocks, never used for training or model selection). Reference: the best
    function of height alone, i.e. the run's own mean profile of the target over the evaluated
    frames (mean over y and time); a model that does not beat it has learned no more than the
    mean vertical profile (strong-baseline practice: McGreivy & Hakim 2024). For Q_d_b the
    target is Q_d / kappa; a run's relative error is the same in Q_d units. The validation loss
    is recomputed (mean over runs of the relative MSE) and compared with the Keras history.
 2. Q_d models: learned response to b. For cells on every 3rd row and column (so that their 3x3
    neighbourhoods do not overlap and all can be perturbed at once), the change of the prediction
    per unit change of b in each of the 9 neighbours, by central differences (EPS in normalised
    units), in physical units times dz^2 (dz = dy) and divided by kappa for target Q_d. The exact
    operator gives [0 1 0; 1 -4 1; 0 1 0] in the interior; in the wall rows the ghost value
    2 b_wall - b makes the centre -5 and there is no neighbour beyond the wall. Also the sum of
    the 9 coefficients (0 for the exact interior operator, i.e. no response to a uniform shift).
 3. Training curves: loss and validation loss per epoch, learning rate, best epoch.
 4. Q_c 2 x 2 design: factor A = q -> q - q_s, factor B = with z -> without z. Main effects and
    interaction on log10 of the validation relative MSE, per run and over all runs, with the
    seeds as replicates (Box, Hunter & Hunter 2005, ch. 5). Errors split by condensing cells
    (Q_c > 0) and the rest.
Inputs, normalisation and padding come from the training script, which is imported; each
configuration's config.json and normalization_constants.npz are checked against it.
Output: simulations_v2/v034/evaluation/ (refuses to overwrite): summary.txt, per_run_errors.csv,
        Qd_*.png and Qc_*.png.
Usage : python "v034 evaluate.py"   (inside the Docker container)
"""
import csv
import gc
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v034_train", SCRIPT_DIR / "v034 stencil CNN.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)      # registers StencilPad and the losses, needed to load the models

QD_CONFIGS = ("Q_d_a", "Q_d_b")
QC_CONFIGS = ("Q_c_C1", "Q_c_C2", "Q_c_C3", "Q_c_C4")
BATCH = 8
EPS = 1e-2                       # perturbation of normalised b for the stencil analysis
STENCIL_FRAMES = 4               # validation frames per run (evenly spaced) for the stencil analysis
SUM_SAMPLE = 5                   # keep every SUM_SAMPLE-th coefficient sum for the histograms
EVAL_DIR_NAME = "evaluation"
EXACT = {"interior": np.array([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]),
         "bottom": np.array([[np.nan] * 3, [1., -5., 1.], [0., 1., 0.]]),   # row index a = di + 1
         "top": np.array([[0., 1., 0.], [1., -5., 1.], [np.nan] * 3])}
SET_STYLE = {"train": ("o", "tab:blue", "training block"),
             "val": ("s", "tab:orange", "validation block"),
             "test": ("*", "tab:red", "test run")}


# ============================================================
# 1. Setup
# ============================================================

def load_config(name):
    """Configuration, normalisation, models and histories of one trained configuration, checked
    against the training script; None if it has not been trained."""
    d = T.OUT_DIR / name
    if not (d / "summary.json").is_file():
        return None
    T.select_config(name)
    cfg = json.loads((d / "config.json").read_text())
    expected = {"inputs": list(T.INPUTS), "target": T.TARGET, "train_runs": T.TRAIN_RUNS,
                "blocks": {"train_frames": list(T.TRAIN_FRAMES), "val_frames": list(T.VAL_FRAMES),
                           "stride": T.STRIDE, "gap_min": T.GAP_MIN},
                "grid": [T.NZ, T.NY]}
    differ = [k for k, v in expected.items() if cfg.get(k) != v]
    if differ:
        sys.exit(f"{d}/config.json differs from the training script in {differ}")
    with np.load(d / "normalization_constants.npz") as f:
        input_norm = {str(c): (float(m), float(s))
                      for c, m, s in zip(f["inputs"], f["input_mean"], f["input_std"])}
        target_norm = (float(f["target_mean"]), float(f["target_std"]))
    if not (T._same(json.loads(json.dumps(input_norm)), cfg["input_norm"])
            and T._same(list(target_norm), cfg["target_norm"])):
        sys.exit(f"{d}: normalization_constants.npz differs from config.json")
    summary = {s["seed"]: s for s in json.loads((d / "summary.json").read_text())}
    seeds = sorted(summary)
    models = {s: tf.keras.models.load_model(d / f"seed_{s}" / "model.keras", compile=False)
              for s in seeds}
    histories = {s: json.loads((d / f"seed_{s}" / "history.json").read_text()) for s in seeds}
    return {"name": name, "cfg": cfg, "summary": summary, "seeds": seeds, "models": models,
            "histories": histories, "input_norm": input_norm, "target_norm": target_norm}


def load_runs(setup):
    """Training runs and test runs (never the excluded ones), normalised as in training."""
    runs = {}
    for label in T.TRAIN_RUNS + T.TEST_RUNS:
        run = T.load_run(label)
        T.apply_input_norm(run, setup["input_norm"])
        runs[label] = run
    return runs


# ============================================================
# 2. Per-run errors
# ============================================================

def z_profile(run, parts):
    """Mean of the target over y and over the frames of `parts`, per row (NZ,)."""
    ys = [run["y"][run["parts"][p]] for p in parts]
    n = sum(len(y) for y in ys)
    return sum(y.astype(np.float64).sum(axis=(0, 2)) for y in ys) / (n * T.NY)


def profile_sse(profile, y):
    """Sum of squared errors of predicting y (n, NZ, NY) by the profile (NZ,)."""
    return float(np.sum((profile[None, :, None] - y) ** 2))


def block_sums(run, parts, setup, profile):
    """Error sums of every seed, pooled over the frames of `parts`, and of the profile."""
    mean, std = setup["target_norm"]
    out = {s: dict(sse=0.0, sy2=0.0, sse_c=0.0, sy2_c=0.0, sp2_dry=0.0, n_c=0, n=0)
           for s in setup["seeds"]}
    ref = dict(sse=0.0, sy2=0.0)
    for part in parts:
        gen = T.RunSnapshots([run], part, setup["target_norm"], BATCH, shuffle=False, seed=0)
        t0 = run["parts"][part].start
        for i in range(len(gen)):
            X, _ = gen[i]
            y = run["y"][t0 + i * BATCH: t0 + i * BATCH + len(X)].astype(np.float64)
            ref["sse"] += profile_sse(profile, y)
            ref["sy2"] += float(np.sum(y ** 2))
            cond = y > 0.0
            for s, model in setup["models"].items():
                p = np.asarray(model.predict_on_batch(X))[..., 0].astype(np.float64) * std + mean
                e2 = (p - y) ** 2
                o = out[s]
                o["sse"] += float(e2.sum()); o["sy2"] += float(np.sum(y ** 2))
                o["sse_c"] += float(e2[cond].sum()); o["sy2_c"] += float(np.sum(y[cond] ** 2))
                o["sp2_dry"] += float(np.sum(p[~cond] ** 2))
                o["n_c"] += int(cond.sum()); o["n"] += int(y.size)
    return out, ref


def run_records(setup, runs):
    recs = []
    for label, run in runs.items():
        sets = ({"train": ["train"], "val": ["val"]} if label in T.TRAIN_RUNS
                else {"test": ["train", "val"]})
        for st, parts in sets.items():
            prof = z_profile(run, parts)
            sums, ref = block_sums(run, parts, setup, prof)
            for s, o in sums.items():
                r = {"config": setup["name"], "seed": s, "run": label, "set": st,
                     "ra": run["ra"], "kappa": run["kappa"],
                     "rel_mse": o["sse"] / o["sy2"],
                     "rel_rmse": math.sqrt(o["sse"] / o["sy2"]),
                     "ref_rel_rmse": math.sqrt(ref["sse"] / ref["sy2"])}
                if T.TARGET == "Q_c":
                    r.update(frac_cond=o["n_c"] / o["n"],
                             rel_rmse_cond=math.sqrt(o["sse_c"] / o["sy2_c"]) if o["sy2_c"] > 0 else float("nan"),
                             err_share_cond=o["sse_c"] / o["sse"] if o["sse"] > 0 else float("nan"),
                             dry_rel=math.sqrt(o["sp2_dry"] / o["sy2"]))
                recs.append(r)
        print(f"  {setup['name']:<7} {label:<15} done", flush=True)
    return recs


# ============================================================
# 3. Q_d: learned response to b
# ============================================================

def sample_cells(nz, ny):
    """Two row sets (rows = 0 and 1 mod 3; together they contain both wall rows) and the columns
    (0 mod 3, without the last one if it is closer than 3 to column 0 across the periodic edge)."""
    cols = np.arange(0, ny, 3)
    if ny - cols[-1] < 3:
        cols = cols[:-1]
    if (nz - 1) % 3 == 2:
        raise ValueError(f"NZ = {nz}: row sets 0 and 1 mod 3 would miss the top wall row")
    return [np.arange(o, nz, 3) for o in (0, 1)], cols


def stencil_response(model, X, ib, scale):
    """Central-difference response of the prediction at the sampled cells to b in each of the 9
    neighbours. X (B, NZ, NY, C) normalised inputs; scale converts d(pred_norm)/d(b_norm) into
    the reported coefficient. Returns [(rows, J)], J (B, n_rows, n_cols, 3, 3) with index
    [a, c] = [di + 1, dj + 1]; NaN for neighbours beyond a wall."""
    nz, ny = X.shape[1], X.shape[2]
    row_sets, cols = sample_cells(nz, ny)
    out = []
    for rows in row_sets:
        J = np.full((X.shape[0], len(rows), len(cols), 3, 3), np.nan)
        for a, di in enumerate((-1, 0, 1)):
            r_in = rows + di
            ok = (r_in >= 0) & (r_in < nz)
            rr, cc_r = np.meshgrid(rows[ok], cols, indexing="ij")
            for c, dj in enumerate((-1, 0, 1)):
                pr, pc = np.meshgrid(r_in[ok], (cols + dj) % ny, indexing="ij")
                d = np.zeros_like(X)
                d[:, pr, pc, ib] = EPS
                fp = np.asarray(model.predict_on_batch(X + d))[..., 0].astype(np.float64)
                fm = np.asarray(model.predict_on_batch(X - d))[..., 0].astype(np.float64)
                view = J[:, :, :, a, c]
                view[:, ok, :] = (fp - fm)[:, rr, cc_r] / (2 * EPS)
        out.append((rows, J * scale))
    return out


def stencil_analysis(setup, runs):
    """Per seed: mean and std of the 9 coefficients (interior, bottom and top wall rows), a sample
    of interior coefficient sums, and per run the mean interior centre coefficient and sum."""
    mean, std = setup["target_norm"]
    ib = list(T.INPUTS).index("b")
    sb = setup["input_norm"]["b"][1]
    dz2 = (1.0 / T.NZ) ** 2
    res = {}
    for s, model in setup["models"].items():
        acc = {k: [np.zeros((3, 3)), np.zeros((3, 3)), 0] for k in ("interior", "bottom", "top")}
        sums, per_run = [], {}
        for label in T.TRAIN_RUNS:
            run = runs[label]
            n = run["parts"]["val"].stop - run["parts"]["val"].start
            k = min(STENCIL_FRAMES, n)
            sel = np.unique(np.linspace(0, n - 1, k).round().astype(int))
            gen = T.RunSnapshots([run], "val", setup["target_norm"], len(sel), shuffle=False, seed=0)
            gen.index = gen.index[sel]
            X, _ = gen[0]
            scale = std / sb * dz2 / (run["kappa"] if T.TARGET == "Q_d" else 1.0)
            run_c, run_s = [], []
            for rows, J in stencil_response(model, X, ib, scale):
                for i, r in enumerate(rows):
                    blk = J[:, i].reshape(-1, 3, 3)
                    kind = "bottom" if r == 0 else "top" if r == T.NZ - 1 else "interior"
                    a = acc[kind]
                    a[0] += np.nan_to_num(blk).sum(0); a[1] += np.nan_to_num(blk ** 2).sum(0); a[2] += len(blk)
                    if kind == "interior":
                        tot = blk.sum(axis=(1, 2))
                        sums.append(tot[::SUM_SAMPLE])
                        run_c.append(blk[:, 1, 1]); run_s.append(tot)
            per_run[label] = {"ra": run["ra"], "centre": float(np.mean(np.concatenate(run_c))),
                              "sum": float(np.mean(np.concatenate(run_s)))}
        stats = {}
        for kind, (s1, s2, nn) in acc.items():
            m = s1 / nn
            sd = np.sqrt(np.maximum(s2 / nn - m ** 2, 0.0))
            m[np.isnan(EXACT[kind])] = np.nan
            sd[np.isnan(EXACT[kind])] = np.nan
            stats[kind] = {"mean": m, "std": sd, "n": nn}
        res[s] = {"stats": stats, "sums": np.concatenate(sums), "per_run": per_run}
    return res


# ============================================================
# 4. Training curves and the Q_c factorial effects
# ============================================================

def training_summary(setup):
    out = {}
    for s in setup["seeds"]:
        h = setup["histories"][s]
        lr = h.get("lr", h.get("learning_rate", []))
        loss = h["loss"]
        n = len(loss)
        out[s] = {"epochs": n, "best_epoch": setup["summary"][s]["best_epoch"],
                  "lr_drops": [e + 1 for e in range(1, len(lr)) if lr[e] < lr[e - 1]],
                  "final_lr": lr[-1] if lr else float("nan"),
                  "stopped": "epoch limit" if n >= setup["cfg"]["epochs"] else "early stopping",
                  "train_change_last10": (loss[-1] - loss[-11]) / loss[-11] if n > 10 else float("nan"),
                  "best_val_loss": setup["summary"][s]["best_val_loss"]}
    return out


def factorial_effects(L):
    """L: {config: log10 relative MSE per seed} for the four Q_c configurations. 2 x 2 design
    (Box, Hunter & Hunter 2005, ch. 5) with A = q -> q - q_s and B = with z -> without z.
    Returns {effect: (estimate, standard error)}; the standard error uses the seed replicates:
    var(effect) = (1/4) sum_c var(L_c) / n_c."""
    m = {c: float(np.mean(v)) for c, v in L.items()}
    eff = {"A (q -> q - q_s)": 0.5 * ((m["Q_c_C2"] - m["Q_c_C1"]) + (m["Q_c_C4"] - m["Q_c_C3"])),
           "B (drop z)": 0.5 * ((m["Q_c_C3"] - m["Q_c_C1"]) + (m["Q_c_C4"] - m["Q_c_C2"])),
           "AB (interaction)": 0.5 * ((m["Q_c_C4"] - m["Q_c_C3"]) - (m["Q_c_C2"] - m["Q_c_C1"]))}
    if all(len(v) > 1 for v in L.values()):
        se = math.sqrt(sum(np.var(v, ddof=1) / len(v) for v in L.values()) / 4.0)
    else:
        se = float("nan")
    return {k: (v, se) for k, v in eff.items()}


def qc_effects(records):
    """Factorial effects per run (validation blocks and test runs) and over the training runs."""
    def L(run, st):
        return {c: np.log10([r["rel_mse"] for r in records
                             if r["config"] == c and r["run"] == run and r["set"] == st])
                for c in QC_CONFIGS}
    out = {}
    for run in T.TRAIN_RUNS:
        out[(run, "val")] = factorial_effects(L(run, "val"))
    for run in T.TEST_RUNS:
        out[(run, "test")] = factorial_effects(L(run, "test"))
    pooled = {}
    for c in QC_CONFIGS:
        seeds = sorted({r["seed"] for r in records if r["config"] == c})
        pooled[c] = np.array([np.mean([math.log10(r["rel_mse"]) for r in records
                                       if r["config"] == c and r["seed"] == s and r["set"] == "val"])
                              for s in seeds])
    out[("all training runs", "val")] = factorial_effects(pooled)
    return out


# ============================================================
# 5. Figures (Q_d and Q_c always separate)
# ============================================================

def describe(name):
    c = T.CONFIGS[name]
    return f"{name}: {', '.join(c['inputs'])} -> {c['target']}"


def plot_errors(records, configs, path, family):
    fig, axes = plt.subplots(1, len(configs), figsize=(5.0 * len(configs), 4.6), sharey=True,
                             squeeze=False)
    for ax, name in zip(axes[0], configs):
        rec = [r for r in records if r["config"] == name]
        for st, (mk, col, lab) in SET_STYLE.items():
            rs = [r for r in rec if r["set"] == st]
            if rs:
                ax.scatter(np.log10([r["ra"] for r in rs]) + {"train": -0.025, "val": 0.025, "test": 0.0}[st],
                           [r["rel_rmse"] for r in rs], marker=mk, color=col,
                           s=90 if mk == "*" else 26, alpha=0.85, label=lab)
        refs = {(r["run"], r["set"]): (r["ra"], r["ref_rel_rmse"]) for r in rec if r["set"] in ("val", "test")}
        if refs:
            ax.scatter(np.log10([v[0] for v in refs.values()]), [v[1] for v in refs.values()],
                       marker="_", color="0.25", s=260, linewidths=2,
                       label="best z-profile (validation / test frames)")
        ax.axhline(1.0, color="0.5", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log_{10}$ Ra")
        ax.set_title(describe(name), fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
    axes[0][0].set_ylabel("relative RMS error (one marker per seed)")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{T.VERSION}: {family}, relative RMS error per run (dotted: error of predicting zero)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_stencils(stencil, path):
    names = [n for n in QD_CONFIGS if n in stencil]
    seeds = sorted(next(iter(stencil.values())))
    fig, axes = plt.subplots(len(names), len(seeds) + 1, figsize=(3.1 * (len(seeds) + 1), 3.0 * len(names)),
                             squeeze=False)
    vmax = max(5.0, max(np.nanmax(np.abs(stencil[n][s]["stats"]["interior"]["mean"])) for n in names for s in seeds))
    for i, n in enumerate(names):
        panels = [("exact", EXACT["interior"], None)] + [
            (f"seed {s}", stencil[n][s]["stats"]["interior"]["mean"], stencil[n][s]["stats"]["interior"]["std"])
            for s in seeds]
        for j, (title, m, sd) in enumerate(panels):
            ax = axes[i][j]
            ax.imshow(m, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            for a in range(3):
                for c in range(3):
                    txt = f"{m[a, c]:.2f}" + ("" if sd is None else f"\n±{sd[a, c]:.2f}")
                    ax.text(c, a, txt, ha="center", va="center", fontsize=7)
            ax.set_xticks([0, 1, 2], ["y-1", "y", "y+1"]); ax.set_yticks([0, 1, 2], ["z-1", "z", "z+1"])
            ax.set_title(f"{n}, {title}", fontsize=9)
    fig.suptitle(f"{T.VERSION}: Q_d models, response to b in the 3x3 neighbourhood (interior rows; "
                 f"x dz^2, / kappa for Q_d; mean ± std over cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_stencil_sums(stencil, path):
    names = [n for n in QD_CONFIGS if n in stencil]
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(5.0 * (len(names) + 1), 4.3), squeeze=False)
    for ax, n in zip(axes[0], names):
        for s, r in sorted(stencil[n].items()):
            lo, hi = np.percentile(r["sums"], [0.5, 99.5])
            ax.hist(r["sums"], bins=120, range=(lo, hi), histtype="step", label=f"seed {s}")
        ax.axvline(0.0, color="k", linestyle=":", linewidth=1, label="exact operator")
        ax.set_xlabel("sum of the 9 coefficients (interior cells)")
        ax.set_ylabel("cells")
        ax.set_title(describe(n), fontsize=10)
        ax.legend(fontsize=7)
    ax = axes[0][-1]
    for n, mk in zip(names, ("o", "s")):
        for s, r in sorted(stencil[n].items()):
            pr = r["per_run"]
            ax.plot(np.log10([v["ra"] for v in pr.values()]), [v["centre"] for v in pr.values()],
                    marker=mk, linestyle="-", alpha=0.8, label=f"{n}, seed {s}")
    ax.axhline(-4.0, color="k", linestyle=":", linewidth=1, label="exact (-4)")
    ax.set_xlabel(r"$\log_{10}$ Ra")
    ax.set_ylabel("mean centre coefficient (interior)")
    ax.set_title("centre coefficient per run (validation frames)", fontsize=10)
    ax.legend(fontsize=6)
    fig.suptitle(f"{T.VERSION}: Q_d models, does the response look like a Laplacian?", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_training(histories, configs, path, family):
    fig, axes = plt.subplots(len(configs), 2, figsize=(12, 3.2 * len(configs)), squeeze=False)
    for i, name in enumerate(configs):
        ax, axl = axes[i]
        for s, h in sorted(histories[name]["hist"].items()):
            ep = np.arange(1, len(h["loss"]) + 1)
            line, = ax.plot(ep, h["loss"], linewidth=1.3, label=f"seed {s} training")
            ax.plot(ep, h["val_loss"], linestyle="--", color=line.get_color(), linewidth=1.3,
                    label=f"seed {s} validation")
            b = histories[name]["best"][s]
            ax.plot(b, h["val_loss"][b - 1], marker="o", color=line.get_color())
            axl.plot(ep, h.get("lr", h.get("learning_rate", [])), color=line.get_color(), label=f"seed {s}")
        ax.set_yscale("log"); axl.set_yscale("log")
        ax.set_ylabel("mean relative MSE"); axl.set_ylabel("learning rate")
        ax.set_title(f"{describe(name)} (dot: best validation epoch)", fontsize=10)
        axl.set_title("learning rate (ReduceLROnPlateau)", fontsize=10)
        for a in (ax, axl):
            a.set_xlabel("epoch"); a.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=6, ncol=3); axl.legend(fontsize=7)
    fig.suptitle(f"{T.VERSION}: {family}, training curves")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_effects(effects, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    keys = list(next(iter(effects.values())))
    for k, (col, off) in zip(keys, (("tab:green", -0.03), ("tab:purple", 0.0), ("tab:gray", 0.03))):
        for st, mk in (("val", "o"), ("test", "*")):
            pts = [(lab, e[k]) for (lab, s), e in effects.items() if s == st and lab in T.TRAIN_RUNS + T.TEST_RUNS]
            if not pts:
                continue
            ra = [T_RA[lab] for lab, _ in pts]
            ax.errorbar(np.log10(ra) + off, [v[0] for _, v in pts], yerr=[v[1] for _, v in pts],
                        fmt=mk, color=col, capsize=3, markersize=9 if mk == "*" else 5,
                        label=f"{k}" + (" (test runs)" if st == "test" else ""))
        ax.axhline(effects[("all training runs", "val")][k][0], color=col, linestyle="--", linewidth=1)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xlabel(r"$\log_{10}$ Ra")
    ax.set_ylabel(r"effect on $\log_{10}$ relative MSE")
    ax.set_title(f"{T.VERSION}: Q_c 2x2 design, effects per run (dashed: all training runs; "
                 f"bars: standard error from the seeds)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_qc_condensing(records, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for name, mk in zip(QC_CONFIGS, ("o", "s", "^", "D")):
        rec = [r for r in records if r["config"] == name and r["set"] in ("val", "test")]
        runs = sorted({r["run"] for r in rec}, key=lambda l: T_RA[l])
        x = np.log10([T_RA[l] for l in runs])

        def seed_mean(key):
            return [np.mean([r[key] for r in rec if r["run"] == l]) for l in runs]
        axes[0].plot(x, seed_mean("rel_rmse_cond"), marker=mk, label=name)
        axes[1].plot(x, seed_mean("err_share_cond"), marker=mk, label=name)
        axes[2].plot(x, seed_mean("dry_rel"), marker=mk, label=name)
    frac = {r["run"]: r["frac_cond"] for r in records if r["config"] == QC_CONFIGS[0] and r["set"] in ("val", "test")}
    runs = sorted(frac, key=lambda l: T_RA[l])
    axes[1].plot(np.log10([T_RA[l] for l in runs]), [frac[l] for l in runs], "k:", label="fraction of condensing cells")
    titles = ("relative RMS error in condensing cells (Q_c > 0)",
              "share of the squared error in condensing cells",
              "RMS of the prediction where Q_c = 0, / RMS(Q_c)")
    for ax, t in zip(axes, titles):
        ax.set_yscale("log"); ax.set_xlabel(r"$\log_{10}$ Ra"); ax.set_title(t, fontsize=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5); ax.legend(fontsize=7)
    fig.suptitle(f"{T.VERSION}: Q_c models, condensing vs non-condensing cells "
                 f"(validation blocks; test runs at Ra 8.7e6 and 1.5e8; mean over seeds)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. Main
# ============================================================

T_RA = {}


def main():
    t_start = time.time()
    out = T.OUT_DIR / EVAL_DIR_NAME
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite.")
    T.configure_gpu()
    records, stencil, histories, train_info, loss_check = [], {}, {}, {}, {}
    for name in T.CONFIGS:
        setup = load_config(name)
        if setup is None:
            print(f"skip {name}: not trained", flush=True)
            continue
        print(f"\n=== {name}: seeds {setup['seeds']}", flush=True)
        runs = load_runs(setup)
        T_RA.update({l: r["ra"] for l, r in runs.items()})
        recs = run_records(setup, runs)
        records += recs
        for s in setup["seeds"]:
            fixed = float(np.mean([r["rel_mse"] for r in recs if r["seed"] == s and r["set"] == "val"]))
            b = setup["summary"][s]["best_epoch"]
            loss_check[(name, s)] = (fixed, setup["histories"][s]["val_loss"][b - 1])
        if name in QD_CONFIGS:
            stencil[name] = stencil_analysis(setup, runs)
        train_info[name] = training_summary(setup)
        histories[name] = {"hist": setup["histories"],
                           "best": {s: setup["summary"][s]["best_epoch"] for s in setup["seeds"]}}
        del runs, setup
        tf.keras.backend.clear_session()
        gc.collect()

    out.mkdir(parents=True, exist_ok=True)
    keys = ["config", "seed", "run", "set", "ra", "kappa", "rel_mse", "rel_rmse", "ref_rel_rmse",
            "frac_cond", "rel_rmse_cond", "err_share_cond", "dry_rel"]
    with open(out / "per_run_errors.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in keys})

    effects = qc_effects(records) if all(c in train_info for c in QC_CONFIGS) else {}
    text = summary_text(records, loss_check, stencil, train_info, effects)
    (out / "summary.txt").write_text(text)
    print("\n" + text)

    qd = [n for n in QD_CONFIGS if n in train_info]
    qc = [n for n in QC_CONFIGS if n in train_info]
    if qd:
        plot_errors(records, qd, out / "Qd_errors_by_run.png", "Q_d models")
        plot_training(histories, qd, out / "Qd_training_curves.png", "Q_d models")
        plot_stencils(stencil, out / "Qd_stencils.png")
        plot_stencil_sums(stencil, out / "Qd_stencil_sums.png")
    if qc:
        plot_errors(records, qc, out / "Qc_errors_by_run.png", "Q_c models")
        plot_training(histories, qc, out / "Qc_training_curves.png", "Q_c models")
        plot_qc_condensing(records, out / "Qc_condensing_cells.png")
    if effects:
        plot_qc_effects(effects, out / "Qc_2x2_effects.png")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    return {"records": records, "loss_check": loss_check, "stencil": stencil,
            "train_info": train_info, "effects": effects}


def summary_text(records, loss_check, stencil, train_info, effects):
    L = [f"{T.VERSION} evaluation, round 1 (relative RMS error = sqrt(sum (p - y)^2 / sum y^2) per run; "
         f"1 = error of predicting zero)", ""]
    L += ["1. Validation loss recomputed from the saved models (mean over runs of the relative MSE)",
          "   vs the Keras history at the best epoch",
          f"   {'config':<7} {'seed':>4} {'recomputed':>11} {'history':>11} {'rel diff':>9}"]
    for (n, s), (a, b) in loss_check.items():
        L.append(f"   {n:<7} {s:>4} {a:>11.4e} {b:>11.4e} {abs(a - b) / b:>9.1e}")
    L += ["", "2. Per run: relative RMS error, mean over seeds [min, max]; ref = best z-profile"]
    for n in train_info:
        L += ["", f"   {describe(n)}",
              f"   {'run':<15} {'set':<5} {'rel RMSE':>9} {'[min,':>10} {'max]':>9} {'ref':>9}"]
        rec = [r for r in records if r["config"] == n]
        for run in T.TRAIN_RUNS + T.TEST_RUNS:
            for st in ("train", "val", "test"):
                v = [r["rel_rmse"] for r in rec if r["run"] == run and r["set"] == st]
                if v:
                    ref = next(r["ref_rel_rmse"] for r in rec if r["run"] == run and r["set"] == st)
                    L.append(f"   {run:<15} {st:<5} {np.mean(v):>9.3e} {min(v):>10.3e} {max(v):>9.3e} {ref:>9.3e}")
    if stencil:
        L += ["", "3. Q_d models: response to b (x dz^2, / kappa for Q_d); exact interior [0 1 0; 1 -4 1; 0 1 0],",
              "   wall rows centre -5. Rows of each 3x3: z-1, z, z+1; columns: y-1, y, y+1."]
        for n, res in stencil.items():
            for s, r in sorted(res.items()):
                st = r["stats"]
                L.append(f"   {n} seed {s}: interior mean (n = {st['interior']['n']} cells)")
                for a in range(3):
                    L.append("      " + "  ".join(f"{st['interior']['mean'][a, c]:+8.3f} ±{st['interior']['std'][a, c]:6.3f}"
                                                  for c in range(3)))
                L.append(f"      sum of coefficients: median {np.median(r['sums']):+.3f}, "
                         f"5-95% [{np.percentile(r['sums'], 5):+.3f}, {np.percentile(r['sums'], 95):+.3f}]")
                for kind in ("bottom", "top"):
                    m = st[kind]["mean"]
                    L.append(f"      {kind} wall row: centre {m[1, 1]:+.3f}, "
                             f"towards interior {m[2 if kind == 'bottom' else 0, 1]:+.3f}, "
                             f"sideways {np.nanmean([m[1, 0], m[1, 2]]):+.3f}")
                L.append("      per run (centre, sum): " + ", ".join(
                    f"{np.log10(v['ra']):.2f}: ({v['centre']:+.2f}, {v['sum']:+.2f})" for v in r["per_run"].values()))
    L += ["", "4. Training (per seed): epochs, best epoch, stop, epochs with a learning-rate drop, final lr,",
          "   relative change of the training loss over the last 10 epochs"]
    for n, info in train_info.items():
        for s, v in info.items():
            L.append(f"   {n:<7} {s:>4} {v['epochs']:>4} {v['best_epoch']:>4} {v['stopped']:<14} "
                     f"{v['lr_drops']} {v['final_lr']:.2e} {v['train_change_last10']:+.3f}")
    if effects:
        L += ["", "5. Q_c 2x2 design: effects on log10 relative MSE (-1 = ten times smaller), +- standard error",
              f"   {'run':<20} {'set':<5} " + " ".join(f"{k:>22}" for k in next(iter(effects.values())))]
        for (lab, st), e in effects.items():
            L.append(f"   {lab:<20} {st:<5} " + " ".join(f"{v:>+12.3f} ±{se:>8.3f}" for v, se in e.values()))
        L += ["", "   condensing cells (validation blocks and test runs, mean over seeds):",
              f"   {'config':<7} {'run':<15} {'frac Q_c>0':>10} {'rel RMSE cond':>13} {'err share cond':>14} {'dry rel':>9}"]
        for n in QC_CONFIGS:
            for run in T.TRAIN_RUNS + T.TEST_RUNS:
                rec = [r for r in records if r["config"] == n and r["run"] == run and r["set"] in ("val", "test")]
                if rec:
                    m = {k: np.mean([r[k] for r in rec]) for k in ("frac_cond", "rel_rmse_cond", "err_share_cond", "dry_rel")}
                    L.append(f"   {n:<7} {run:<15} {m['frac_cond']:>10.3f} {m['rel_rmse_cond']:>13.3e} "
                             f"{m['err_share_cond']:>14.3f} {m['dry_rel']:>9.3e}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
