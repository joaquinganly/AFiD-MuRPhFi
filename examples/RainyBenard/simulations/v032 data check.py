# ============================================================
# v032 - data check: is b physical in every run? (read-only)
#
# 1. b in every run and every frame of <run>_fields_Q.npz:
#    - minimum principle: Db/Dt = kappa Lap(b) + Q_c with Q_c >= 0, initial
#      b = 0 and wall values 0 and beta - 1, so b >= 0 in the solution; a
#      minimum below -B_NEG_TOL is flagged. (No upper bound: Q_c heats.)
#    - grid-scale noise: RMS of the part of b with y-wavenumber in the top
#      octave (k >= NY/4), wall rows excluded; and the fraction of interior
#      points that are local extrema along z (a zigzag in z). Flagged when
#      either is > NOISE_FACTOR times the median over runs.
#    - frame 0: max |b|, to confirm it is the initial condition b = 0.
# 2. Q_d scale per training / validation run, frames FIRST_FRAME..NT-1
#    (frame 0 and the runs in DROP_RUNS excluded), to compare target
#    normalisations: RMS(Q_d), RMS(Q_d / kappa), the share of each run in the
#    summed squared target under a global z-score of Q_d or of Q_d / kappa,
#    and the time variability RMS(Q_d - time mean) / RMS(Q_d) over the last
#    LATE_FRAMES frames.
# Inputs : the run files used by "v031 stencil CNN.py" (imported for the run
#          list, paths and the folder-name parser).
# Output : v032_data_check/data_check.txt and data_check.json (refuses to
#          overwrite).
# Usage  : python "v032 data check.py"
# ============================================================

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

B_NEG_TOL = 0.01            # 1 % of the wall-to-wall range of b
NOISE_FACTOR = 10.0
FIRST_FRAME = 1             # frame 0 is the initial condition
LATE_FRAMES = 200
DROP_RUNS = ["Pr_05_Ra_1e4"]

SCRIPT_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("v031_train", SCRIPT_DIR / "v031 stencil CNN.py")
T = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(T)
OUT_DIR = SCRIPT_DIR / "v032_data_check"


def b_checks(b):
    """b: (NZ, NY, nt) float64. Per-run b diagnostics (frames >= 1, except frame0)."""
    nz, ny, nt = b.shape
    later = b[:, :, 1:]
    i_min = np.unravel_index(np.argmin(later), later.shape)
    hi_rms, z_ext = np.empty(nt - 1), np.empty(nt - 1)
    for n in range(1, nt):
        f = b[1:-1, :, n]                                   # interior rows
        F = np.fft.rfft(f - f.mean(axis=1, keepdims=True), axis=1)
        F[:, :ny // 4] = 0.0
        hi_rms[n - 1] = np.sqrt(np.mean(np.fft.irfft(F, n=ny, axis=1) ** 2))
        d = np.diff(b[:, :, n], axis=0)
        z_ext[n - 1] = np.mean(d[:-1] * d[1:] < 0.0)
    return {
        "b_min": float(later.min()), "b_min_frame": int(i_min[2]) + 1,
        "b_max": float(later.max()),
        "frac_b_below_tol": float(np.mean(later < -B_NEG_TOL)),
        "frame0_max_abs_b": float(np.abs(b[:, :, 0]).max()),
        "gridscale_rms_median": float(np.median(hi_rms)),
        "gridscale_rms_max": float(hi_rms.max()),
        "z_extrema_frac_median": float(np.median(z_ext)),
        "nt_fields": int(nt),
    }


def qd_stats(q_d, kappa, nt):
    """q_d: (NZ, NY, >= nt). Frames FIRST_FRAME..nt-1."""
    q = q_d[:, :, FIRST_FRAME:nt].astype(np.float64)
    late = q[:, :, -LATE_FRAMES:]
    rms = math.sqrt(np.mean(q ** 2))
    return {
        "kappa": kappa,
        "rms_qd": rms,
        "rms_qd_over_kappa": rms / kappa,
        "time_var_late": float(np.sqrt(np.mean((late - late.mean(axis=2, keepdims=True)) ** 2))
                               / (math.sqrt(np.mean(late ** 2)) + 1e-300)),
        "_sum": float(q.sum()), "_sumsq": float(np.sum(q ** 2)), "_n": int(q.size),
    }


def shares(stats, train, scale):
    """Share (%) of each training run in sum((y - mean)^2) for y = Q_d / scale,
    with the global mean over the training runs (the z-score std cancels)."""
    s = sum(stats[l]["_sum"] / scale(l) for l in train)
    n = sum(stats[l]["_n"] for l in train)
    mean = s / n
    tot = {l: stats[l]["_sumsq"] / scale(l) ** 2 - 2 * mean * stats[l]["_sum"] / scale(l)
           + mean ** 2 * stats[l]["_n"] for l in train}
    total = sum(tot.values())
    return {l: 100.0 * v / total for l, v in tot.items()}


def main():
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} exists and is not empty; refusing to overwrite.")
    nt_common, _ = T.common_frame_count(T.ALL_RUNS)
    sets = {**{l: "train" for l in T.TRAIN_RUNS}, **{l: "val" for l in T.VAL_RUNS},
            **{l: "held_out" for l in T.HELD_OUT_RUNS}}

    checks, stats = {}, {}
    for label in T.ALL_RUNS:
        paths = T.run_paths(label)
        with np.load(paths["fields"]) as d:
            checks[label] = b_checks(d["theta"].astype(np.float64))
        if sets[label] != "held_out":            # held-out targets are not looked at
            ra, pr = T.extract_ra_pr(label)
            with np.load(paths["qdc"]) as d:
                stats[label] = qd_stats(d["Q_d"], 1.0 / math.sqrt(pr * ra), nt_common)
        print(f"  {label:<14s} done", flush=True)

    med_hi = np.median([c["gridscale_rms_median"] for c in checks.values()])
    med_z = np.median([c["z_extrema_frac_median"] for c in checks.values()])
    for c in checks.values():
        reasons = []
        if c["b_min"] < -B_NEG_TOL:
            reasons.append(f"b < -{B_NEG_TOL}")
        if c["gridscale_rms_median"] > NOISE_FACTOR * med_hi:
            reasons.append("grid-scale noise (y)")
        if c["z_extrema_frac_median"] > NOISE_FACTOR * med_z:
            reasons.append("zigzag in z")
        if c["frame0_max_abs_b"] > 1e-3:
            reasons.append("frame 0 is not b = 0")
        c["flags"] = reasons

    train = [l for l in T.TRAIN_RUNS if l not in DROP_RUNS]
    kappa = {l: stats[l]["kappa"] for l in stats}
    share_qd = shares(stats, train, lambda l: 1.0)
    share_k = shares(stats, train, lambda l: kappa[l])

    lines = ["1. b in every run (frames >= 1 unless stated). Minimum principle: b >= 0.",
             f"   Flags: b_min < -{B_NEG_TOL}; grid-scale RMS or z-extrema fraction > "
             f"{NOISE_FACTOR:g} x median over runs (medians {med_hi:.2e}, {med_z:.3f}).",
             f"   {'set':<8} {'run':<14} {'b_min':>8} {'@frame':>6} {'b_max':>7} "
             f"{'%b<-tol':>8} {'frame0 max|b|':>13} {'grid RMS med':>12} {'grid RMS max':>12} "
             f"{'z-extr':>7}  flags"]
    for l in T.ALL_RUNS:
        c = checks[l]
        lines.append(f"   {sets[l]:<8} {l:<14} {c['b_min']:>8.4f} {c['b_min_frame']:>6d} "
                     f"{c['b_max']:>7.4f} {100 * c['frac_b_below_tol']:>8.3f} "
                     f"{c['frame0_max_abs_b']:>13.2e} {c['gridscale_rms_median']:>12.2e} "
                     f"{c['gridscale_rms_max']:>12.2e} {c['z_extrema_frac_median']:>7.3f}  "
                     f"{', '.join(c['flags']) or '-'}")
    lines += ["",
              f"2. Q_d scale, frames {FIRST_FRAME}..{nt_common - 1}; shares over the training runs "
              f"without {DROP_RUNS}.",
              "   share = % of the summed squared normalised target (= weight in the MSE loss).",
              "   time var = RMS(Q_d - time mean) / RMS(Q_d), last "
              f"{LATE_FRAMES} frames (0 = steady).",
              f"   {'set':<6} {'run':<14} {'kappa':>9} {'RMS Q_d':>10} {'RMS Q_d/kappa':>13} "
              f"{'time var':>8} {'share Q_d %':>11} {'share Q_d/kappa %':>17}"]
    for l in stats:
        s = stats[l]
        in_train = l in share_qd
        a = f"{share_qd[l]:>11.2f}" if in_train else f"{'-':>11}"
        b = f"{share_k[l]:>17.2f}" if in_train else f"{'-':>17}"
        tag = sets[l] + ("*" if l in DROP_RUNS else "")
        lines.append(f"   {tag:<6} {l:<14} {s['kappa']:>9.2e} {s['rms_qd']:>10.3e} "
                     f"{s['rms_qd_over_kappa']:>13.3e} {s['time_var_late']:>8.3f} {a} {b}")
    lines.append(f"   * dropped from training ({', '.join(DROP_RUNS)})")
    text = "\n".join(lines) + "\n"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "data_check.txt").write_text(text)
    with open(OUT_DIR / "data_check.json", "w") as f:
        json.dump({"b_checks": checks,
                   "qd_stats": {l: {k: v for k, v in s.items() if not k.startswith("_")}
                                for l, s in stats.items()},
                   "share_qd": share_qd, "share_qd_over_kappa": share_k,
                   "settings": {"B_NEG_TOL": B_NEG_TOL, "NOISE_FACTOR": NOISE_FACTOR,
                                "FIRST_FRAME": FIRST_FRAME, "LATE_FRAMES": LATE_FRAMES,
                                "DROP_RUNS": DROP_RUNS, "nt_common": nt_common}}, f, indent=2)
    print("\n" + text)


if __name__ == "__main__":
    main()
