"""v033: the CNNs' predictions against the DNS on the two test runs (agreed with the user,
2026-11-11; changelog.txt).

Every frame (t = 0 ... 1000) of each test run (Ra 8.7e6 and 1.5e8; never used for training or model
selection) is predicted by every trained v033 model (6 configurations x 3 seeds), each frame
independently from that frame's DNS inputs. The same analyses are then applied to the DNS source
terms and to the predictions. Frames 0 ... SPIN - 1 (drizzle start, no convection) were never used
in training; they are included, and marked in the time series.
  Q_c = (gamma/tau) (q - q_s) H(q - q_s)  (DNS, Q_d_c.npz)  vs  Q^_c from Q_c_C1 ... Q_c_C4;
  Q_d = kappa Lap b                       (DNS, Q_d_c.npz)  vs  Q^_d from Q_d_a (predicts Q_d) and
        Q_d_b (predicts Q_d / kappa; multiplied by kappa = (Ra Pr)^(-1/2) from bou.in).
Notation: cell c = (t, i, j), row i (height z_i), column j (y_j); <Q>(t) = (1/(NZ NY)) sum_{i,j} Q;
<Q>_{y,t}(z_i) = (1/(NY NT)) sum_{j,t} Q.
 (a) Snapshots: Q, Q^ (seed SNAP_SEED) and Q^ - Q on one colour scale, at t* = argmax_{t >= SPIN}
     <Q_c>(t) (strongest condensation) and t_q = argmin_{t >= SPIN} <Q_c>(t) (quietest frame); whole
     domain and a ZOOM_W x 1 window centred on the column with the largest sum_i Q_c in that frame.
 (b) <Q>(t) and <Q^>(t) for every frame (every seed).
 (c) <Q>_{y,t}(z) and <Q^>_{y,t}(z) (every seed).
 (d) Joint histogram of (Q, Q^) over all cells and frames (seed SNAP_SEED), with the line Q^ = Q.
 (e) Histograms of Q and Q^ (every seed); for Q_c, the fraction of cells with Q^_c < 0.
 (f) Condensing cells (Q_c only): the DNS sets {Q_c > 0} and {Q_c > eps} and the CNN set {Q^_c > eps},
     eps = (gamma/tau) S_EPS (the network never outputs exactly 0). On each set: s = Q_c tau / gamma
     (DNS) or Q^_c tau / gamma (CNN), and s / q_s with q_s = exp(alpha (b - beta z)) from the DNS b
     (which the CNN also receives). Shown: cumulative distributions, distributions and medians (the
     analysis of Qc_relative_supersaturation.png in diagnostics/v033_diagnostics.py, applied to both
     outputs), and the overlap of the two eps-sets. Medians from log10 bins of width 1 / PER_DEC
     decade, interpolating the cumulative count linearly in log10 within the bin.
Check: the relative RMS error sqrt(sum (Q^ - Q)^2 / sum Q^2) over the frames of the v033 evaluation
(every 3rd frame of 25..750 and of 811..1000) must equal evaluation/per_run_errors.csv.
GIFs (GIF_CONFIG, seed SNAP_SEED, each test run): Q_c, Q^_c and Q^_c - Q_c in every frame.
Output: simulations_v2/v033/dns_vs_cnn/ (refuses to overwrite): summary.txt, dns_vs_cnn_data.npz,
        Qc_*.png, Qd_*.png, Qc_<config>_vs_DNS_<run>.gif.
Usage : python diagnostics/v033_dns_vs_cnn.py   (from simulations_v2/, inside the Docker container)
"""
import csv
import gc
import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SIM = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("v033_evaluate", SIM / "v033 evaluate.py")
EV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EV)      # imports the training script as EV.T (loader settings, StencilPad, losses)
T = EV.T
pp = T.pp

OUT_NAME = "dns_vs_cnn"
SNAP_SEED = 42                   # seed shown in the maps, the joint histograms and the GIFs
S_EPS = 1e-5                     # CNN condensing set: Q^_c > eps = (gamma/tau) S_EPS
SPIN = 25                        # frames 0 .. SPIN - 1: drizzle start, never used in training
BATCH = 8                        # frames per prediction batch
CHUNK = 32                       # frames per chunk for the DNS statistics
ZOOM_W = 2.0                     # width in y of the zoom window (height 1)
NBIN = 240                       # linear bins per axis (histograms and joint histograms)
PCT = 0.01                       # histogram range from the [PCT, 100 - PCT] percentiles of the DNS term
LOG_LO, LOG_HI, PER_DEC = -12, 2, 100     # log10 bins of s and s / q_s
NLOG = (LOG_HI - LOG_LO) * PER_DEC
GIF_CONFIG, GIF_FPS, GIF_DPI = "Q_c_C4", 20, 72
FAMILIES = {"Q_c": EV.QC_CONFIGS, "Q_d": EV.QD_CONFIGS}
SOURCES = {"b": ("fields", "theta"), "q_minus_qs": ("fields", "rel_q"), "q": ("q", "q"),
           "Q_d": ("qdc", "Q_d"), "Q_c": ("qdc", "Q_c")}
COLOURS = {"Q_c_C1": "tab:blue", "Q_c_C2": "tab:orange", "Q_c_C3": "tab:green", "Q_c_C4": "tab:red",
           "Q_d_a": "tab:blue", "Q_d_b": "tab:orange"}
SEED_STYLE = ("-", "--", ":")
QSYM = {"Q_c": ("Q_c", r"\hat Q_c"), "Q_d": ("Q_d", r"\hat Q_d")}
DEFN = {"Q_c": r"$Q_c=(\gamma/\tau)(q-q_s)\,H(q-q_s)$", "Q_d": r"$Q_d=\kappa\nabla^2 b$"}
FILE_TAG = {"Q_c": "Qc", "Q_d": "Qd"}
WHICH = {"t_star": "strongest", "t_quiet": "quietest"}
DIV = LinearSegmentedColormap.from_list("div", ["#104281", "#3987e5", "#f0efec", "#e34948", "#8f1f1f"])


# ============================================================
# 1. Data and binning
# ============================================================

def load_test_run(label):
    """Every frame of a test run, float32 and frame-major (nt, NZ, NY) as the training loader stores
    them (T.load_run): b, q, q - q_s and the DNS terms Q_d, Q_c; parameters from bou.in and humid.in."""
    paths = T.run_paths(label)
    p = pp.read_bou(paths["bou"])
    h = pp.read_humid(paths["humid"])
    if (p["nz"], p["ny"]) != (T.NZ, T.NY):
        raise ValueError(f"{label}: bou.in grid {p['nz']} x {p['ny']}, expected {T.NZ} x {T.NY}")
    if not (math.isclose(h["alpha"], T.ALPHA) and math.isclose(h["beta"], T.BETA)):
        raise ValueError(f"{label}: humid.in alpha {h['alpha']}, beta {h['beta']} differ from the training script")
    zc, _, _ = pp.read_grid(paths["grid"], p["nz"], p["ny"], p["ly"])
    run = {"label": label, "ra": p["ra"], "pr": p["pr"], "kappa": 1.0 / math.sqrt(p["ra"] * p["pr"]),
           "ly": p["ly"], "g": h["gamma"] / h["tau"], "alpha": h["alpha"], "beta": h["beta"],
           "xm": zc[:, 0].astype(np.float64)}
    for name, (kind, key) in SOURCES.items():
        with np.load(paths[kind]) as d:
            a = d[key]
        if a.ndim != 3 or a.shape[:2] != (T.NZ, T.NY):
            raise ValueError(f"{label}: '{key}' has shape {a.shape}")
        run[name] = np.ascontiguousarray(np.moveaxis(a, -1, 0), dtype=np.float32)
        del a
    run["nt"] = run["b"].shape[0]
    if any(run[n].shape[0] != run["nt"] for n in SOURCES):
        raise ValueError(f"{label}: the arrays have different numbers of frames")
    print(f"  {label}: Ra {run['ra']:.4g}, Pr {run['pr']:g}, kappa {run['kappa']:.4e}, gamma/tau {run['g']:g}, "
          f"{run['nt']} frames", flush=True)
    return run


def saturation(run, sl):
    """q_s = exp(alpha (b - beta z)) on the frames sl, float64."""
    return np.exp(run["alpha"] * (run["b"][sl].astype(np.float64) - run["beta"] * run["xm"][None, :, None]))


def eval_frames(nt):
    """Mask of the frames of the v033 evaluation of a test run (both blocks' frames)."""
    m = np.zeros(nt, dtype=bool)
    m[np.concatenate([T.frame_indices(T.TRAIN_FRAMES), T.frame_indices(T.VAL_FRAMES)])] = True
    return m


def q_range(Q, fam):
    """Histogram range from the DNS term (all cells and frames): Q_c [-hi / 4, hi], hi its (100 - PCT)
    percentile (negative values only from the CNN); Q_d [-a, a], a the larger |PCT or (100 - PCT)
    percentile|."""
    lo_p, hi_p = (float(v) for v in np.percentile(Q, [PCT, 100.0 - PCT]))
    if fam == "Q_c":
        hi = hi_p if hi_p > 0.0 else max(float(Q.max()), 1e-30)
        return -0.25 * hi, hi
    a = max(abs(lo_p), abs(hi_p)) or max(float(np.abs(Q).max()), 1e-30)
    return -a, a


def lin_index(v, lo, hi):
    """Entry of v among NBIN equal bins of [lo, hi), shifted by 1: entry 0 below lo, NBIN + 1 at or above hi."""
    k = np.floor((v - lo) * (NBIN / (hi - lo)))
    return (np.clip(k, -1, NBIN) + 1).astype(np.int64)


def lin_hist(v, lo, hi):
    return np.bincount(lin_index(v, lo, hi).ravel(), minlength=NBIN + 2)


def joint_hist(x, y, lo, hi):
    """Counts of the pairs (x, y), entry [kx, ky] of an (NBIN + 2) x (NBIN + 2) table (flattened);
    out-of-range values in the edge entries."""
    return np.bincount((lin_index(x, lo, hi) * (NBIN + 2) + lin_index(y, lo, hi)).ravel(),
                       minlength=(NBIN + 2) ** 2)


def log_hist(v):
    """Counts of log10(v), v > 0, in NLOG bins of width 1 / PER_DEC from LOG_LO to LOG_HI; entry 0:
    below the range, entry NLOG + 1: at or above it."""
    k = np.floor((np.log10(v) - LOG_LO) * PER_DEC)
    return np.bincount((np.clip(k, -1, NLOG) + 1).astype(np.int64).ravel(), minlength=NLOG + 2)


def log_edges():
    return 10.0 ** (LOG_LO + np.arange(NLOG + 1) / PER_DEC)


def hist_median(c):
    """Median from log_hist counts: the cumulative count interpolated linearly in log10 within the
    bin; nan if empty or if the median falls outside the range."""
    n = float(c.sum())
    if n == 0.0:
        return float("nan")
    cum = np.cumsum(c)
    k = int(np.searchsorted(cum, 0.5 * n))
    if k == 0 or k == len(c) - 1:
        return float("nan")
    frac = (0.5 * n - cum[k - 1]) / c[k]
    return float(10.0 ** (LOG_LO + (k - 1 + frac) / PER_DEC))


def hist_cdf(c):
    """G at the NLOG + 1 bin edges: fraction of the values below each edge."""
    n = c.sum()
    return np.cumsum(c)[:NLOG + 1] / n if n else np.full(NLOG + 1, np.nan)


def zoom_centre(frame_qc):
    """Column with the largest sum over the rows of Q_c (column 0 if there is no condensation)."""
    return int(np.argmax(frame_qc.sum(axis=0)))


def zoom_cols(j0, ly):
    """Columns of the zoom window of width ZOOM_W around column j0 (periodic in y)."""
    n = int(round(ZOOM_W / ly * T.NY))
    return (j0 - n // 2 + np.arange(n)) % T.NY


# ============================================================
# 2. Statistics of the DNS terms and of the predictions
# ============================================================

def new_log_counts():
    return np.zeros(NLOG + 2, np.int64)


def dns_stats(run, fam, lo, hi):
    """Analyses (b)-(f) for the DNS term."""
    Q, nt, eps = run[fam], run["nt"], run["g"] * S_EPS
    d = {"dmean": np.zeros(nt), "prof": np.zeros(T.NZ), "sy2_t": np.zeros(nt),
         "hist": np.zeros(NBIN + 2, np.int64)}
    if fam == "Q_c":
        for tag in ("pos", "eps"):
            d.update({f"n_{tag}": 0, f"hs_{tag}": new_log_counts(), f"hr_{tag}": new_log_counts()})
    for t0 in range(0, nt, CHUNK):
        sl = slice(t0, min(t0 + CHUNK, nt))
        y = Q[sl].astype(np.float64)
        d["dmean"][sl] = y.mean(axis=(1, 2))
        d["prof"] += y.sum(axis=(0, 2))
        d["sy2_t"][sl] = np.sum(y ** 2, axis=(1, 2))
        d["hist"] += lin_hist(y, lo, hi)
        if fam == "Q_c":
            qs = saturation(run, sl)
            for tag, m in (("pos", y > 0.0), ("eps", y > eps)):
                s = y[m] / run["g"]
                d[f"n_{tag}"] += int(m.sum())
                d[f"hs_{tag}"] += log_hist(s)
                d[f"hr_{tag}"] += log_hist(s / qs[m])
    d["prof"] /= nt * T.NY
    return d


def cnn_stats(run, setup, fam, lo, hi, snap_frames, keep_all):
    """Every frame of `run` predicted by every seed of the configuration in `setup` (inputs built and
    normalised by the training script's own functions); analyses (b)-(f) per seed. Returns
    ({seed: statistics}, the predictions of seed SNAP_SEED for every frame (nt, NZ, NY) float32 if
    keep_all, else None)."""
    chans = T.field_channels()
    r = {"label": run["label"], "fields": np.stack([run[c] for c in chans], axis=-1), "y": run[fam],
         "xm": run["xm"], "params": {"log10_Ra": float(np.log10(run["ra"]))},
         "parts": {"all": slice(0, run["nt"])}}
    T.apply_input_norm(r, setup["input_norm"])
    gen = T.RunSnapshots([r], "all", setup["target_norm"], BATCH, shuffle=False, seed=0)
    mean, std = setup["target_norm"]
    factor = {"Q_c": 1.0, "Q_d": 1.0, "Q_d_over_kappa": run["kappa"]}[T.TARGET]
    nt, g = run["nt"], run["g"]
    eps = g * S_EPS
    out = {}
    for s in setup["seeds"]:
        o = {"sse_t": np.zeros(nt), "dmean": np.zeros(nt), "prof": np.zeros(T.NZ),
             "hist": np.zeros(NBIN + 2, np.int64)}
        if fam == "Q_c":
            o.update(n_neg=0, sum_neg=0.0, n_set=0, n_both=0, hs=new_log_counts(), hr=new_log_counts())
        if s == SNAP_SEED:
            o.update(joint=np.zeros((NBIN + 2) ** 2, np.int64), snaps={})
        out[s] = o
    allp = np.empty((nt, T.NZ, T.NY), np.float32) if keep_all else None
    for i in range(len(gen)):
        X, _ = gen[i]
        sl = slice(i * BATCH, i * BATCH + len(X))
        y = run[fam][sl].astype(np.float64)
        if fam == "Q_c":
            qs = saturation(run, sl)
            dset = y > eps
        for s, model in setup["models"].items():
            p = (np.asarray(model.predict_on_batch(X))[..., 0].astype(np.float64) * std + mean) * factor
            o = out[s]
            o["sse_t"][sl] = np.sum((p - y) ** 2, axis=(1, 2))
            o["dmean"][sl] = p.mean(axis=(1, 2))
            o["prof"] += p.sum(axis=(0, 2))
            o["hist"] += lin_hist(p, lo, hi)
            if fam == "Q_c":
                neg = p < 0.0
                o["n_neg"] += int(neg.sum())
                o["sum_neg"] += float(p[neg].sum())
                m = p > eps
                o["n_set"] += int(m.sum())
                o["n_both"] += int(np.sum(m & dset))
                sh = p[m] / g
                o["hs"] += log_hist(sh)
                o["hr"] += log_hist(sh / qs[m])
            if s == SNAP_SEED:
                o["joint"] += joint_hist(y, p, lo, hi)
                for t in snap_frames:
                    if sl.start <= t < sl.stop:
                        o["snaps"][t] = p[t - sl.start].astype(np.float32)
                if allp is not None:
                    allp[sl] = p
    for o in out.values():
        o["prof"] /= nt * T.NY
    return out, allp


def rel(a, b):
    """sqrt(sum (a - b)^2 / sum b^2)."""
    den = float(np.sum(np.square(b)))
    return math.sqrt(float(np.sum(np.square(a - b))) / den) if den > 0.0 else float("nan")


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def csv_check(res, rows):
    """{(run, config, seed): (relative RMS error over the evaluation's frames, per_run_errors.csv,
    relative difference)}."""
    out = {}
    for label, r in res.items():
        ev = r["eval"]
        for fam in FAMILIES:
            sy2 = r[fam]["dns"]["sy2_t"][ev].sum()
            for name, o in r[fam]["cnn"].items():
                for s, st in o.items():
                    mine = math.sqrt(st["sse_t"][ev].sum() / sy2)
                    ref = [float(x["rel_rmse"]) for x in rows if x["config"] == name and x["run"] == label
                           and x["set"] == "test" and int(x["seed"]) == s]
                    ref = ref[0] if ref else float("nan")
                    out[(label, name, s)] = (mine, ref, abs(mine - ref) / ref)
    return out


# ============================================================
# 3. Summary and data file
# ============================================================

def ra_text(label):
    return label.replace("Ra_", "Ra ").replace("_Pr_1", "")


def inputs_text(name):
    return ", ".join(T.CONFIGS[name]["inputs"])


def summary_text(res, check):
    L = [f"{T.VERSION}: CNN predictions against the DNS on the test runs (every frame; each frame predicted "
         f"independently from that frame's DNS inputs)",
         "Q: DNS source term (Q_d_c.npz); Q^: CNN prediction (Q_d_b: kappa x the predicted Q_d / kappa).",
         "rel RMS = sqrt(sum_c (Q^_c - Q_c)^2 / sum_c Q_c^2) over all cells c of the frames named; 'all': every frame;",
         "'eval': the frames of the v033 evaluation (every 3rd of 25..750 and 811..1000); 'csv': evaluation/",
         "per_run_errors.csv. <Q>(t): mean over the cells of frame t; mean ratio = mean_t <Q^>(t) / mean_t <Q>(t);",
         "ts rel = sqrt(sum_t (<Q^>(t) - <Q>(t))^2 / sum_t <Q>(t)^2); outside: fraction of the cells outside the",
         "histogram range (figures).",
         f"Condensing sets: DNS {{Q_c > 0}}, DNS {{Q_c > eps}}, CNN {{Q^_c > eps}}, eps = (gamma/tau) x {S_EPS:g}; "
         f"s = Q_c tau/gamma (DNS), Q^_c tau/gamma (CNN);",
         f"q_s = exp(alpha (b - beta z)) from the DNS b. Medians from log10 bins of 1/{PER_DEC} decade. "
         f"D = DNS eps-set, C = CNN eps-set.", ""]
    for label, r in res.items():
        m = r["meta"]
        ncell = m["nt"] * T.NZ * T.NY
        fr = r["frames"]
        L.append(f"=== {label}: Ra {m['ra']:.4g}, Pr {m['pr']:g}, kappa {m['kappa']:.4e}, gamma/tau {m['g']:g}, "
                 f"{m['nt']} frames ({ncell} cells); t* = {fr['t_star']} (largest <Q_c>), t_q = {fr['t_quiet']} "
                 f"(smallest <Q_c>, t >= {SPIN}); eps = {m['g'] * S_EPS:.3g}")
        for fam in FAMILIES:
            f = r[fam]
            d = f["dns"]
            dh = d["hist"]
            L.append(f"  {fam}: DNS mean_t <Q>(t) = {d['dmean'].mean():+.5e}; histogram range [{f['lo']:.4g}, "
                     f"{f['hi']:.4g}], DNS outside {(dh[0] + dh[-1]) / ncell:.1e}")
            head = (f"  {'config':<7} {'seed':>4} {'rel all':>9} {'rel eval':>9} {'csv':>9} {'diff':>8} "
                    f"{'mean ratio':>10} {'ts rel':>8} {'outside':>8}")
            if fam == "Q_c":
                head += f" {'Q^<0':>8} {'sum min(Q^,0)/sum Q':>20}"
            L.append(head)
            sy2 = d["sy2_t"].sum()
            for name, o in f["cnn"].items():
                for s in sorted(o):
                    st = o[s]
                    mine, ref, diff = check[(label, name, s)]
                    line = (f"  {name:<7} {s:>4} {math.sqrt(st['sse_t'].sum() / sy2):>9.3e} {mine:>9.3e} {ref:>9.3e} "
                            f"{diff:>8.1e} {st['dmean'].mean() / d['dmean'].mean():>10.4f} "
                            f"{rel(st['dmean'], d['dmean']):>8.3f} {(st['hist'][0] + st['hist'][-1]) / ncell:>8.1e}")
                    if fam == "Q_c":
                        line += f" {st['n_neg'] / ncell:>8.4f} {st['sum_neg'] / (d['dmean'].sum() * T.NZ * T.NY):>20.3e}"
                    L.append(line)
            if fam == "Q_c":
                L += ["  condensing sets:",
                      f"  {'set':<26} {'frac of cells':>13} {'median s':>9} {'median s/q_s':>12} "
                      f"{'|D&C|/|D|':>9} {'|D&C|/|C|':>9}"]
                for tag, lab in (("pos", "DNS Q_c > 0"), ("eps", "DNS Q_c > eps")):
                    L.append(f"  {lab:<26} {d[f'n_{tag}'] / ncell:>13.4e} {hist_median(d[f'hs_{tag}']):>9.2e} "
                             f"{hist_median(d[f'hr_{tag}']):>12.2e}")
                for name, o in f["cnn"].items():
                    for s in sorted(o):
                        st = o[s]
                        L.append(f"  {name + ' seed ' + str(s) + ' Q^ > eps':<26} {st['n_set'] / ncell:>13.4e} "
                                 f"{hist_median(st['hs']):>9.2e} {hist_median(st['hr']):>12.2e} "
                                 f"{st['n_both'] / max(d['n_eps'], 1):>9.4f} {st['n_both'] / max(st['n_set'], 1):>9.4f}")
        L.append("")
    worst = max(v[2] for v in check.values()) if check else float("nan")
    L.append(f"Check against per_run_errors.csv: largest relative difference {worst:.1e} "
             f"({'OK' if worst < 1e-4 else 'LARGER THAN 1e-4: inspect'})")
    return "\n".join(L) + "\n"


def save_data(res, path):
    """Binned statistics, time series, profiles and snapshot frames (keys <run>__<term>__<source>__...)."""
    d = {"log_edges": log_edges()}
    for label, r in res.items():
        d[f"{label}__xm"] = r["meta"]["xm"]
        d[f"{label}__frames_tstar_tquiet"] = np.array([r["frames"]["t_star"], r["frames"]["t_quiet"]])
        for fam in FAMILIES:
            f = r[fam]
            d[f"{label}__{fam}__range"] = np.array([f["lo"], f["hi"]])
            for key, v in f["dns"].items():
                if isinstance(v, np.ndarray):
                    d[f"{label}__{fam}__DNS__{key}"] = v
            for t, a in r["dns_snaps"][fam].items():
                d[f"{label}__{fam}__DNS__snap_t{t}"] = a
            for name, o in f["cnn"].items():
                for s, st in o.items():
                    for key, v in st.items():
                        if isinstance(v, np.ndarray):
                            d[f"{label}__{fam}__{name}__{s}__{key}"] = v
                    for t, a in st.get("snaps", {}).items():
                        d[f"{label}__{fam}__{name}__{s}__snap_t{t}"] = a
    np.savez_compressed(path, **d)


# ============================================================
# 4. Figures (Q_c and Q_d separate)
# ============================================================

def colour_limit(fam, dns, preds):
    """Symmetric colour limit of a snapshot: the 99.5th percentile of the positive DNS values (Q_c) or of
    |Q_d| on rows 2 .. NZ - 3 (the wall rows exceed it and saturate); for a frame with fewer than 100
    condensing cells, of |Q^_c| of the predictions."""
    if fam == "Q_c":
        pos = dns[dns > 0.0]
        if pos.size >= 100:
            return float(np.percentile(pos, 99.5))
        allp = np.abs(np.concatenate([p.ravel() for p in preds.values()]))
        return float(np.percentile(allp, 99.5)) or 1.0
    return float(np.percentile(np.abs(dns[2:-2]), 99.5)) or 1.0


def plot_snapshots(r, label, fam, which, path):
    t = r["frames"][which]
    names = list(r[fam]["cnn"])
    qs, qhs = QSYM[fam]
    dns = r["dns_snaps"][fam][t].astype(np.float64)
    preds = {n: r[fam]["cnn"][n][SNAP_SEED]["snaps"][t].astype(np.float64) for n in names}
    lim = colour_limit(fam, dns, preds)
    ly = r["meta"]["ly"]
    dy = ly / T.NY
    j0 = r["j0"][t]
    cols = zoom_cols(j0, ly)
    nw = len(cols)
    ext_zoom = ((-(nw // 2) - 0.5) * dy, (nw - nw // 2 - 0.5) * dy, 0.0, 1.0)
    strips = [(f"DNS: ${qs}$", dns)]
    for n in names:
        e = preds[n] - dns
        strips.append((f"{n} (inputs {inputs_text(n)}), seed {SNAP_SEED}: ${qhs}$", preds[n]))
        strips.append((f"{n}: ${qhs}-{qs}$   (this frame: relative RMS error "
                       rf"$\sqrt{{\sum(\hat Q-Q)^2/\sum Q^2}}$ = {rel(preds[n], dns):.3g}; "
                       rf"max $|\hat Q-Q|$ = {np.abs(e).max():.3g})", e))
    kw = dict(origin="lower", cmap=DIV, vmin=-lim, vmax=lim, interpolation="nearest")
    W, left, aw = 15.0, 0.6, 12.8
    sh, th, gap = aw / ly, 0.30, 0.10
    ncol = 1 + len(names)
    zgap = 0.45
    zw = (aw - zgap * (ncol - 1)) / ncol
    zh = zw / (nw * dy)
    H = 1.05 + len(strips) * (sh + th + gap) + 0.35 + 0.55 + 2 * (th + zh + 0.45) + 0.1
    fig = plt.figure(figsize=(W, H))

    def axes_at(x, y, w, h):
        return fig.add_axes([x / W, y / H, w / W, h / H])

    y = H - 1.05
    for k, (title, a) in enumerate(strips):
        y -= th + sh
        ax = axes_at(left, y, aw, sh)
        im = ax.imshow(a, extent=(0.0, ly, 0.0, 1.0), aspect="auto", **kw)
        ax.set_title(title, fontsize=8.5, loc="left", pad=3)
        ax.set_yticks([0, 1])
        ax.set_ylabel("z", fontsize=8, labelpad=0)
        ax.set_xticks(np.arange(0.0, ly + 1e-9, 2.0))
        ax.tick_params(labelsize=7, pad=1)
        if k < len(strips) - 1:
            ax.set_xticklabels([])
        if k == 0:
            for yb in ((j0 - nw // 2) * dy % ly, (j0 + nw - nw // 2) * dy % ly):
                ax.axvline(yb, color="k", ls="--", lw=0.9)
        y -= gap
    ax.set_xlabel("y", fontsize=8, labelpad=1)
    cax = axes_at(left + aw + 0.2, y + 0.3, 0.15, H - 1.05 - y - 0.6)
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(f"${qs}$, ${qhs}$ and ${qhs}-{qs}$ (free-fall units; one colour scale, saturating beyond "
                 rf"$\pm${lim:.3g})", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    y -= 0.35 + 0.55
    fig.text(left / W, (y + 0.2) / H,
             f"Zoom: window of width {nw * dy:g} centred on y* = {(j0 + 0.5) * dy:.3f} (dashed lines in the DNS strip), "
             r"the column with the largest $\sum_i Q_c$ in this frame. Top row: DNS and predictions; bottom row: "
             "prediction minus DNS.", fontsize=9)
    for row in range(2):
        y -= th + zh
        for c in range(ncol):
            if row == 1 and c == 0:
                continue
            if c == 0:
                title, a = f"DNS: ${qs}$", dns
            else:
                n = names[c - 1]
                title, a = (f"{n}: ${qhs}$", preds[n]) if row == 0 else (f"{n}: ${qhs}-{qs}$", preds[n] - dns)
            ax = axes_at(left + c * (zw + zgap), y, zw, zh)
            ax.imshow(a[:, cols], extent=ext_zoom, aspect="auto", **kw)
            ax.set_title(title, fontsize=8.5, pad=3)
            ax.tick_params(labelsize=7, pad=1)
            ax.set_xlabel("y - y*", fontsize=8, labelpad=1)
            ax.set_ylabel("z", fontsize=8, labelpad=0)
        y -= 0.45
    fig.text(left / W, (H - 0.45) / H,
             f"{T.VERSION}, test run {ra_text(label)}: {DEFN[fam]} from the DNS and from the CNNs at t = {t} "
             + ("(t*: the frame with the largest domain mean of $Q_c$)" if which == "t_star" else
                f"(the frame with the smallest domain mean of $Q_c$, t $\\geq$ {SPIN})"), fontsize=11)
    fig.text(left / W, (H - 0.78) / H,
             f"Each frame predicted from that frame's DNS inputs only. Rows: DNS; then for each configuration its "
             f"prediction (seed {SNAP_SEED}) and the prediction minus the DNS. 0 <= z <= 1 (vertical), "
             f"0 <= y <= {ly:g}.", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_timeseries(res, fam, path):
    labels = list(res)
    names = list(res[labels[0]][fam]["cnn"])
    qs, qhs = QSYM[fam]
    fig, axes = plt.subplots(len(names), len(labels), figsize=(8.0 * len(labels), 2.8 * len(names) + 1.3),
                             squeeze=False, layout="constrained")
    for j, label in enumerate(labels):
        r = res[label]
        d = r[fam]["dns"]
        t = np.arange(r["meta"]["nt"])
        for i, n in enumerate(names):
            ax = axes[i][j]
            o = r[fam]["cnn"][n]
            seeds = sorted(o)
            M = np.array([o[s]["dmean"] for s in seeds])
            ax.axvspan(0, SPIN, color="0.88", lw=0, label=f"t < {SPIN}: drizzle start (not used in training)")
            ax.fill_between(t, M.min(0), M.max(0), color=COLOURS[n], alpha=0.3, lw=0,
                            label=f"CNN: range over seeds {', '.join(map(str, seeds))}")
            ax.plot(t, o[SNAP_SEED]["dmean"], color=COLOURS[n], lw=0.9,
                    label=rf"CNN seed {SNAP_SEED}: $\langle {qhs}\rangle(t)$")
            ax.plot(t, d["dmean"], color="k", lw=0.9, label=rf"DNS: $\langle {qs}\rangle(t)$")
            ax.axvline(r["frames"]["t_star"], color="0.3", ls="--", lw=0.8, label="t* (snapshot)")
            ax.axvline(r["frames"]["t_quiet"], color="0.3", ls=":", lw=0.8, label="quietest frame (snapshot)")
            rr = [rel(o[s]["dmean"], d["dmean"]) for s in seeds]
            ax.set_title(f"{ra_text(label)} (test run), {n} (inputs {inputs_text(n)}): "
                         rf"$\sqrt{{\sum_t(\langle {qhs}\rangle-\langle {qs}\rangle)^2/\sum_t\langle {qs}\rangle^2}}$"
                         f" = {min(rr):.3g} to {max(rr):.3g} over the seeds", fontsize=9)
            ax.set_ylabel(rf"$\langle {qs}\rangle$, $\langle {qhs}\rangle$")
            ax.set_xlim(0, t[-1])
            ax.grid(True, ls=":", alpha=0.5)
            if i == len(names) - 1:
                ax.set_xlabel("t (frame number = time in free-fall units)")
    handles = [Line2D([], [], color="k", lw=0.9, label=rf"DNS: $\langle {qs}\rangle(t)$"),
               Line2D([], [], color="0.45", lw=0.9, label=rf"CNN seed {SNAP_SEED}: $\langle {qhs}\rangle(t)$ "
                                                           "(colour: configuration)"),
               Patch(color="0.45", alpha=0.3, lw=0, label="CNN: range over the seeds"),
               Patch(color="0.88", lw=0, label=f"t < {SPIN}: drizzle start (not used in training)"),
               Line2D([], [], color="0.3", ls="--", lw=0.8, label="t* (snapshot of the strongest condensation)"),
               Line2D([], [], color="0.3", ls=":", lw=0.8, label="quietest frame (snapshot)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8)
    fig.suptitle(f"{T.VERSION}: domain mean " + rf"$\langle Q\rangle(t)=\frac{{1}}{{N_z N_y}}\sum_{{i,j}}Q(t,z_i,y_j)$"
                 f" of {DEFN[fam]}, DNS (black) against the CNNs (colour), every frame of the test runs", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_profiles(res, fam, path):
    labels = list(res)
    qs, qhs = QSYM[fam]
    fig, axes = plt.subplots(2, len(labels), figsize=(7.5 * len(labels), 12.0), squeeze=False, layout="constrained")
    for j, label in enumerate(labels):
        r = res[label]
        z = r["meta"]["xm"]
        f = r[fam]
        curves = [f["dns"]["prof"]]
        for row in range(2):
            ax = axes[row][j]
            for n, o in f["cnn"].items():
                seeds = sorted(o)
                P = np.array([o[s]["prof"] for s in seeds])
                ax.fill_betweenx(z, P.min(0), P.max(0), color=COLOURS[n], alpha=0.3, lw=0)
                ax.plot(o[SNAP_SEED]["prof"], z, color=COLOURS[n], lw=1.2,
                        label=f"{n} (inputs {inputs_text(n)}): seed {SNAP_SEED}; band: range over seeds")
                if row == 0:
                    curves += list(P)
            ax.plot(f["dns"]["prof"], z, color="k", lw=2.0, label="DNS")
            ax.axvline(0.0, color="0.5", lw=0.6)
            ax.grid(True, ls=":", alpha=0.5)
            ax.set_ylabel("z")
            ax.set_xlabel(rf"$\langle {qs}\rangle_{{y,t}}(z)$, $\langle {qhs}\rangle_{{y,t}}(z)$ (free-fall units)")
        inner = (z >= 0.05) & (z <= 0.95)
        vals = np.concatenate([c[inner] for c in curves])
        pad = 0.05 * ((vals.max() - vals.min()) or 1.0)
        axes[1][j].set_xlim(vals.min() - pad, vals.max() + pad)
        axes[1][j].set_ylim(0.05, 0.95)
        axes[0][j].set_ylim(0.0, 1.0)
        axes[0][j].set_title(f"{ra_text(label)} (test run): all heights", fontsize=10)
        axes[1][j].set_title(f"{ra_text(label)}: 0.05 <= z <= 0.95 (horizontal axis fitted to these heights)",
                             fontsize=10)
    axes[0][0].legend(fontsize=7.5, loc="best")
    fig.suptitle(f"{T.VERSION}: mean profiles " + rf"$\langle Q\rangle_{{y,t}}(z_i)=\frac{{1}}{{N_y N_t}}\sum_{{j,t}}Q$"
                 f" over every frame of the test runs, {DEFN[fam]}: DNS (black) against the CNNs", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_joint(res, fam, path):
    labels = list(res)
    names = list(res[labels[0]][fam]["cnn"])
    qs, qhs = QSYM[fam]
    fig, axes = plt.subplots(len(labels), len(names), figsize=(4.4 * len(names) + 1.4, 4.3 * len(labels) + 1.0),
                             squeeze=False, layout="constrained")
    for i, label in enumerate(labels):
        f = res[label][fam]
        lo, hi = f["lo"], f["hi"]
        e = np.linspace(lo, hi, NBIN + 1)
        J = {n: f["cnn"][n][SNAP_SEED]["joint"].reshape(NBIN + 2, NBIN + 2) for n in names}
        vmax = max(int(a[1:-1, 1:-1].max()) for a in J.values()) or 1
        for k, n in enumerate(names):
            ax = axes[i][k]
            inner = J[n][1:-1, 1:-1]
            outside = 1.0 - inner.sum() / J[n].sum()
            pcm = ax.pcolormesh(e, e, np.ma.masked_equal(inner.T, 0), norm=LogNorm(vmin=1, vmax=vmax),
                                cmap="viridis", rasterized=True)
            ax.plot([lo, hi], [lo, hi], color="tab:red", ls="--", lw=0.9, label=rf"${qhs}={qs}$")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.set_xlabel(rf"DNS ${qs}$")
            ax.set_ylabel(rf"CNN ${qhs}$ (seed {SNAP_SEED})")
            ax.set_title(f"{ra_text(label)}, {n} (inputs {inputs_text(n)})\n{outside:.1e} of the pairs outside the axes",
                         fontsize=9)
            ax.grid(True, ls=":", alpha=0.4)
            if k == 0:
                ax.legend(fontsize=7, loc="upper left")
        cb = fig.colorbar(pcm, ax=list(axes[i]), shrink=0.9)
        cb.set_label(f"cells per bin ({ra_text(label)}: bins {e[1] - e[0]:.2g} wide; all cells, all frames)",
                     fontsize=8)
    fig.suptitle(f"{T.VERSION}: joint histogram of the DNS value and the CNN prediction in the same cell, "
                 f"{DEFN[fam]}, every cell and frame of the test runs (white: no cell)", fontsize=11)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pdfs(res, fam, path):
    labels = list(res)
    names = list(res[labels[0]][fam]["cnn"])
    qs, qhs = QSYM[fam]
    fig, axes = plt.subplots(len(labels), len(names), figsize=(4.8 * len(names) + 0.4, 4.0 * len(labels) + 1.0),
                             squeeze=False, layout="constrained")
    for i, label in enumerate(labels):
        f = res[label][fam]
        ncell = res[label]["meta"]["nt"] * T.NZ * T.NY
        e = np.linspace(f["lo"], f["hi"], NBIN + 1)
        w = e[1] - e[0]
        dh = f["dns"]["hist"]
        for k, n in enumerate(names):
            ax = axes[i][k]
            ax.stairs(dh[1:-1] / (ncell * w), e, color="k", lw=1.6,
                      label=f"DNS ${qs}$ ({(dh[0] + dh[-1]) / ncell:.1e} outside)")
            for s, ls in zip(sorted(f["cnn"][n]), SEED_STYLE):
                o = f["cnn"][n][s]
                h = o["hist"]
                extra = rf"; ${qhs}<0$: {o['n_neg'] / ncell:.1%}" if fam == "Q_c" else ""
                ax.stairs(h[1:-1] / (ncell * w), e, color=COLOURS[n], ls=ls, lw=1.1,
                          label=f"CNN ${qhs}$, seed {s} ({(h[0] + h[-1]) / ncell:.1e} outside{extra})")
            ax.set_yscale("log")
            ax.set_xlim(f["lo"], f["hi"])
            ax.set_xlabel(rf"${qs}$, ${qhs}$ (free-fall units)")
            ax.set_ylabel("probability density")
            ax.set_title(f"{ra_text(label)}, {n} (inputs {inputs_text(n)})", fontsize=9)
            ax.grid(True, ls=":", alpha=0.5)
            ax.legend(fontsize=6.5, loc="upper right")
    fig.suptitle(f"{T.VERSION}: distribution of {DEFN[fam]} over every cell and frame of the test runs, DNS (black) "
                 f"against the CNNs; density = cells per bin / (all cells x bin width), {NBIN} bins; values outside "
                 f"the axis range (fraction in the legend) not shown", fontsize=10)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_condensing(res, path):
    labels = list(res)
    fig, axes = plt.subplots(len(labels), 3, figsize=(20, 5.4 * len(labels) + 1.2), squeeze=False,
                             layout="constrained")
    xe = log_edges()
    for i, label in enumerate(labels):
        r = res[label]
        f = r["Q_c"]
        d = f["dns"]
        ncell = r["meta"]["nt"] * T.NZ * T.NY
        sets = [(r"DNS, $Q_c>0$", "k", "-", d["hs_pos"], d["hr_pos"], d["n_pos"], None),
                (r"DNS, $Q_c>\epsilon$", "k", "--", d["hs_eps"], d["hr_eps"], d["n_eps"], None)]
        for n, o in f["cnn"].items():
            st = o[SNAP_SEED]
            sets.append((rf"{n}, $\hat Q_c>\epsilon$", COLOURS[n], "-", st["hs"], st["hr"], st["n_set"], o))
        ax0, ax1, ax2 = axes[i]
        for lab, col, ls, hs, hr, nset, o in sets:
            tag = "" if o is None else f" (seed {SNAP_SEED})"
            ax0.semilogx(xe, hist_cdf(hs), color=col, ls=ls, lw=1.3, label=f"{lab}{tag}: {nset / ncell:.2%} of the cells")
            coarse = hr[1:-1].reshape(-1, PER_DEC // 10).sum(axis=1)
            ec = LOG_LO + np.arange(coarse.size + 1) / 10.0
            ax1.stairs(coarse / max(int(hr.sum()), 1) * 10.0, ec, color=col, ls=ls, lw=1.3, label=f"{lab}{tag}")
        ax0.axvline(S_EPS, color="0.4", ls=":", lw=1.0, label=rf"$s={S_EPS:g}$ (the threshold $\epsilon\,\tau/\gamma$)")
        ax0.set_xlim(1e-9, 1.0)
        ax0.set_ylim(0.0, 1.02)
        ax0.set_xlabel(r"x")
        ax0.set_ylabel(r"$G(x)$ = fraction of the set with $s<x$")
        ax0.set_title(f"{ra_text(label)}: (a) cumulative distribution of s on each set;\n"
                      r"$s=Q_c\tau/\gamma$ (DNS) or $\hat Q_c\tau/\gamma$ (CNN)", fontsize=9.5)
        ax0.legend(fontsize=7, loc="upper left")
        ax1.set_xlim(-10.0, 1.0)
        ax1.set_xlabel(r"$\log_{10}(s/q_s)$, $q_s=e^{\alpha(b-\beta z)}$ from the DNS b")
        ax1.set_ylabel(r"probability density of $\log_{10}(s/q_s)$ (per decade)")
        ax1.set_title(f"{ra_text(label)}: (b) distribution of the relative supersaturation on each set\n"
                      "(bins of 0.1 decade)", fontsize=9.5)
        ax1.legend(fontsize=7, loc="upper left")
        ticks = []
        for k, (lab, col, ls, hs, hr, nset, o) in enumerate(sets):
            ticks.append(lab.replace(", ", "\n"))
            if o is None:
                ax2.plot(k - 0.12, hist_median(hs), "o", color="tab:blue", ms=7)
                ax2.plot(k + 0.12, hist_median(hr), "s", color="tab:green", ms=7)
                continue
            seeds = sorted(o)
            for dx, key, mk, col2 in ((-0.12, "hs", "o", "tab:blue"), (0.12, "hr", "s", "tab:green")):
                v = np.array([hist_median(o[s][key]) for s in seeds])
                v0 = hist_median(o[SNAP_SEED][key])
                ax2.errorbar(k + dx, v0, yerr=[[v0 - np.nanmin(v)], [np.nanmax(v) - v0]], fmt=mk, color=col2,
                             ms=7, capsize=3)
        ax2.set_yscale("log")
        ax2.set_xticks(range(len(sets)), ticks, fontsize=8)
        ax2.grid(True, axis="y", which="both", ls=":", alpha=0.5)
        ax2.set_ylabel("median over the set")
        ax2.set_title(f"{ra_text(label)}: (c) medians over each set (CNN: seed {SNAP_SEED}; bars: range over the seeds)",
                      fontsize=9.5)
        ax2.legend(handles=[Line2D([], [], marker="o", ls="", color="tab:blue", label="median of s"),
                            Line2D([], [], marker="s", ls="", color="tab:green", label=r"median of $s/q_s$")],
                   fontsize=8, loc="best")
    fig.suptitle(f"{T.VERSION}: condensing cells in the DNS and in the CNN predictions (test runs, every frame). "
                 rf"CNN set: $\hat Q_c>\epsilon=(\gamma/\tau)\cdot${S_EPS:g}; DNS sets: $Q_c>0$ and $Q_c>\epsilon$. "
                 r"A model given q builds $q_s$ itself: a relative error $\delta$ in $q_s$ gives "
                 r"$|\hat Q_c-Q_c|/Q_c=\delta/(s/q_s)$", fontsize=10.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_gif(label, r, path):
    dns, cnn = r["gif"]
    ly, nt = r["meta"]["ly"], r["meta"]["nt"]
    pos = dns[dns > 0.0]
    lim = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    del pos
    W, left, aw = 10.0, 0.45, 8.6
    sh, th, gap = aw / ly, 0.26, 0.06
    H = 0.55 + 3 * (th + sh + gap) + 0.3
    fig = plt.figure(figsize=(W, H), facecolor="white")
    kw = dict(origin="lower", extent=(0.0, ly, 0.0, 1.0), aspect="auto", cmap=DIV, vmin=-lim, vmax=lim,
              interpolation="antialiased")
    titles = (r"DNS $Q_c$", rf"{GIF_CONFIG} (inputs {inputs_text(GIF_CONFIG)}), seed {SNAP_SEED}: $\hat Q_c$",
              r"$\hat Q_c-Q_c$")
    ims = []
    y = H - 0.55
    for k, title in enumerate(titles):
        y -= th + sh
        ax = fig.add_axes([left / W, y / H, aw / W, sh / H])
        ims.append(ax.imshow(np.zeros((T.NZ, T.NY), np.float32), **kw))
        ax.set_title(title, fontsize=8, loc="left", pad=2)
        ax.set_yticks([0, 1])
        ax.set_xticks(np.arange(0.0, ly + 1e-9, 2.0))
        ax.tick_params(labelsize=6.5, pad=1)
        if k < 2:
            ax.set_xticklabels([])
        y -= gap
    cax = fig.add_axes([(left + aw + 0.15) / W, (y + 0.1) / H, 0.1 / W, (H - 0.55 - y - 0.2) / H])
    cb = fig.colorbar(ims[0], cax=cax, extend="both")
    cb.ax.tick_params(labelsize=6.5)
    title = fig.text(left / W, (H - 0.32) / H, "", fontsize=9)

    def update(i):
        ims[0].set_data(dns[i])
        ims[1].set_data(cnn[i])
        ims[2].set_data(cnn[i] - dns[i])
        title.set_text(f"{T.VERSION}, test run {ra_text(label)}, t = {i:4d}: DNS and CNN $Q_c$ on one colour scale "
                       rf"(saturates beyond $\pm${lim:.3g})" + ("   [t < 25: drizzle start]" if i < SPIN else ""))
        return ims

    update(0)
    tmp = path.with_name(path.stem + ".partial.gif")
    FuncAnimation(fig, update, frames=nt, blit=False).save(tmp, writer=PillowWriter(fps=GIF_FPS), dpi=GIF_DPI)
    tmp.rename(path)
    plt.close(fig)
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB, {nt} frames", flush=True)


# ============================================================
# 5. Main
# ============================================================

def main():
    t_start = time.time()
    out = T.OUT_DIR / OUT_NAME
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite.")
    csv_path = T.OUT_DIR / EV.EVAL_DIR_NAME / "per_run_errors.csv"
    if not csv_path.is_file():
        sys.exit(f"{csv_path} is missing; run 'v033 evaluate.py' first")
    rows = read_rows(csv_path)
    T.configure_gpu()
    res = {}
    for label in T.TEST_RUNS:
        print(f"\n=== {label}", flush=True)
        run = load_test_run(label)
        dm = run["Q_c"].mean(axis=(1, 2), dtype=np.float64)
        frames = {"t_star": SPIN + int(np.argmax(dm[SPIN:])), "t_quiet": SPIN + int(np.argmin(dm[SPIN:]))}
        snap_t = sorted(set(frames.values()))
        r = {"meta": {k: run[k] for k in ("ra", "pr", "kappa", "ly", "g", "alpha", "beta", "nt", "xm")},
             "frames": frames, "j0": {t: zoom_centre(run["Q_c"][t]) for t in snap_t},
             "dns_snaps": {fam: {t: run[fam][t].copy() for t in snap_t} for fam in FAMILIES},
             "eval": eval_frames(run["nt"])}
        for fam, names in FAMILIES.items():
            lo, hi = q_range(run[fam], fam)
            f = {"lo": lo, "hi": hi, "dns": dns_stats(run, fam, lo, hi), "cnn": {}}
            for name in names:
                setup = EV.load_config(name)
                if setup is None:
                    print(f"  skip {name}: not trained", flush=True)
                    continue
                if SNAP_SEED not in setup["seeds"]:
                    sys.exit(f"{name}: seed {SNAP_SEED} is not trained")
                f["cnn"][name], allp = cnn_stats(run, setup, fam, lo, hi, snap_t, keep_all=(name == GIF_CONFIG))
                if allp is not None:
                    r["gif"] = (run["Q_c"], allp)
                print(f"  {name:<7} done ({time.time() - t_start:.0f} s)", flush=True)
                del setup
                tf.keras.backend.clear_session()
                gc.collect()
            r[fam] = f
        res[label] = r
        del run
        gc.collect()
    check = csv_check(res, rows)
    out.mkdir(parents=True, exist_ok=True)
    text = summary_text(res, check)
    (out / "summary.txt").write_text(text)
    print("\n" + text, flush=True)
    save_data(res, out / "dns_vs_cnn_data.npz")
    for fam in FAMILIES:
        if not any(r[fam]["cnn"] for r in res.values()):
            continue
        tag = FILE_TAG[fam]
        for label, r in res.items():
            for which in ("t_star", "t_quiet"):
                plot_snapshots(r, label, fam, which, out / f"{tag}_snapshots_{label.replace('_Pr_1', '')}_"
                                                            f"{WHICH[which]}_t{r['frames'][which]}.png")
        plot_timeseries(res, fam, out / f"{tag}_domain_mean_vs_time.png")
        plot_profiles(res, fam, out / f"{tag}_mean_profiles.png")
        plot_joint(res, fam, out / f"{tag}_joint_histograms.png")
        plot_pdfs(res, fam, out / f"{tag}_histograms.png")
    if any(r["Q_c"]["cnn"] for r in res.values()):
        plot_condensing(res, out / "Qc_condensing_cells.png")
    for label, r in res.items():
        if "gif" in r:
            make_gif(label, r, out / f"Qc_{GIF_CONFIG.split('_')[-1]}_vs_DNS_{label.replace('_Pr_1', '')}.gif")
    elapsed = time.time() - t_start
    print(f"Elapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)", flush=True)
    return {"res": res, "check": check, "out": out}


if __name__ == "__main__":
    main()
