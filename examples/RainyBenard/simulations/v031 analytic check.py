# ============================================================
# v031 analytic check (reads the data, writes only to OUT_DIR)
#
# For every run, applies the operators of
# rainybenard_postprocessing/"Qc and Qd only.py" to the stored inputs and
# compares the result with the stored targets in <run>/Q_d_c.npz:
#
#   Q_d = kappa * Lap(b),                 kappa = 1/sqrt(Pr*Ra)
#   Q_c = (gamma/tau) (q - q_s) H(q - q_s), q_s = exp(alpha (b - beta z))
#
# b from <run>_fields_Q.npz ("theta"), q from <run>_q.npz ("q"),
# z = xm from cordin_info.h5, Pr and Ra from the folder name
# (CLAUDE.md, "Simulation run folders"). Parameters are those of humid.in,
# identical for all runs; wall values of b are those of SetTempBCs.F90 with
# moisture on (b = 0 at the bottom wall, b = beta - 1 at the top wall).
#
# Per run it reports:
#   - relative L2 and max abs error of the analytic Q_d and Q_c against the
#     stored ones (this is also the error floor of an exact model);
#   - Pr implied by the least-squares fit Q_d = c Lap(b): Pr = 1/(c^2 Ra);
#   - gamma/tau implied by the fit Q_c = c' (q - q_s)_+;
#   - max |rel_q - (q - q_s)| for the stored rel_q.
# All 22 runs are read, including the held-out ones: this is a data check,
# not a model evaluation.
# ============================================================

import csv
import json
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np


# ============================================================
# Settings
# ============================================================

NZ, NY = 64, 1024
LZ, LY = 1.0, 16.0
DZ, DY = LZ / NZ, LY / NY

ALPHA = 3.0          # humid.in
BETA = 2.0           # humid.in
GAMMA = 2.0          # humid.in
TAU = 0.1            # humid.in
B_WALL_BOTTOM = 0.0          # SetTempBCs.F90 (moist)
B_WALL_TOP = BETA - 1.0      # SetTempBCs.F90 (moist)

# Tolerances for flagging a run. The stored arrays are float64 and the
# operators below repeat the post-processing arithmetic, so agreement should
# be at round-off level.
REL_TOL = 1e-10
REL_TOL_FLOAT32 = 1e-5       # used if any array involved is stored as float32
PR_TOL = 1e-6

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DATA_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR / "v031_analytic_check"

ALL_RUNS = [
    "Pr_05_Ra_1e4", "Pr_05_Ra_1e5", "Pr_05_Ra_1e6", "Pr_05_Ra_1e7",
    "Pr_05_Ra_1e8", "Pr_05_Ra_1e9",
    "Pr_2_Ra_1e4",  "Pr_2_Ra_1e5",  "Pr_2_Ra_1e6",  "Pr_2_Ra_1e8",
    "Pr_2_Ra_1e9",
    "Ra_1e4", "Ra_1e5", "Ra_1e6", "Ra_5e6", "Ra_1e7", "Ra_5e7", "Ra_75e6",
    "Ra_1e8", "Ra_25e7", "Ra_1e9", "Ra_175e7",
]


# ============================================================
# Helpers (operators copied from "Qc and Qd only.py")
# ============================================================

def parse_parameters(folder_name):
    """Same as parse_parameters in "Qc and Qd only.py"."""
    if folder_name.startswith("Ra_"):
        Pr = 1.0
        Ra = float(folder_name.split("_")[1].replace("e", "E"))
        return Pr, Ra
    match = re.match(r"Pr_(.+)_Ra_(.+)", folder_name)
    if match is None:
        raise ValueError(f"Cannot parse folder name: {folder_name}")
    Pr = 0.5 if match.group(1) == "05" else float(match.group(1).replace("_", "."))
    Ra = float(match.group(2).replace("e", "E"))
    return Pr, Ra


def laplacian(b):
    """Laplacian of b with shape (NZ, NY, nt): same stencil, wall ghost points
    and periodic y as laplacian() in "Qc and Qd only.py", applied to all
    frames at once."""
    d2z = np.empty_like(b)
    d2z[1:-1] = (b[2:] - 2.0 * b[1:-1] + b[:-2]) / DZ**2
    d2z[0] = (b[1] - 3.0 * b[0] + 2.0 * B_WALL_BOTTOM) / DZ**2
    d2z[-1] = (b[-2] - 3.0 * b[-1] + 2.0 * B_WALL_TOP) / DZ**2
    d2y = (np.roll(b, -1, axis=1) - 2.0 * b + np.roll(b, 1, axis=1)) / DY**2
    return d2z + d2y


def run_paths(label):
    run_dir = BASE_DATA_DIR / label
    grid = run_dir / "cordin_info.h5"
    if not grid.is_file():
        grid = run_dir / "outputdir" / "cordin_info.h5"
    return {
        "fields": run_dir / f"{label}_fields_Q.npz",
        "q": run_dir / f"{label}_q.npz",
        "qdc": run_dir / "Q_d_c.npz",
        "grid": grid,
    }


def read_xm(path):
    with h5py.File(path, "r") as f:
        xm = np.asarray(f["xm"][()]).squeeze()
    if xm.size == NZ + 1:
        xm = xm[:-1]
    if xm.size != NZ:
        raise ValueError(f"{path}: xm has {xm.size} points, expected {NZ}")
    if not np.allclose(xm, (np.arange(NZ) + 0.5) / NZ, rtol=0.0, atol=1e-6):
        raise ValueError(f"{path}: xm is not the uniform cell-centre grid that "
                         f"the Laplacian assumes")
    return xm


def rel_l2(stored, analytic):
    norm = np.linalg.norm(stored)
    return float(np.linalg.norm(stored - analytic) / norm) if norm > 0 else float("nan")


def check_run(label):
    paths = run_paths(label)
    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        return {"label": label, "status": "MISSING FILES: " + ", ".join(missing)}

    Pr, Ra = parse_parameters(label)
    kappa = 1.0 / np.sqrt(Pr * Ra)

    with np.load(paths["fields"]) as d:
        b = d["theta"]
        rel_q = d["rel_q"]
    with np.load(paths["q"]) as d:
        q = d["q"]
    with np.load(paths["qdc"]) as d:
        Q_d = d["Q_d"]
        Q_c = d["Q_c"]
    xm = read_xm(paths["grid"])

    frames = {"theta": b.shape[-1], "rel_q": rel_q.shape[-1], "q": q.shape[-1],
              "Q_d": Q_d.shape[-1], "Q_c": Q_c.shape[-1]}
    dtypes = {"theta": str(b.dtype), "rel_q": str(rel_q.dtype), "q": str(q.dtype),
              "Q_d": str(Q_d.dtype), "Q_c": str(Q_c.dtype)}
    tol = REL_TOL if all(v == "float64" for v in dtypes.values()) else REL_TOL_FLOAT32
    b, rel_q, q = (np.asarray(a, dtype=np.float64) for a in (b, rel_q, q))
    Q_d, Q_c = (np.asarray(a, dtype=np.float64) for a in (Q_d, Q_c))
    for name, arr in (("theta", b), ("rel_q", rel_q), ("q", q), ("Q_d", Q_d), ("Q_c", Q_c)):
        if arr.shape[:2] != (NZ, NY):
            return {"label": label, "status": f"BAD SHAPE {name} {arr.shape}"}
    nt = min(frames.values())
    b, rel_q, q = b[..., :nt], rel_q[..., :nt], q[..., :nt]
    Q_d, Q_c = Q_d[..., :nt], Q_c[..., :nt]

    # Q_d
    lap = laplacian(b)
    Q_d_an = kappa * lap
    c_d = float(np.vdot(Q_d, lap) / np.vdot(lap, lap))
    Pr_implied = 1.0 / (c_d**2 * Ra)
    # Approximate floor for a model fed float32 b: the exact operator applied
    # to b rounded to float32.
    lap32 = laplacian(b.astype(np.float32).astype(np.float64))
    Q_d_float32_floor = rel_l2(Q_d_an, kappa * lap32)
    del lap32

    # Q_c
    supersat = q - np.exp(ALPHA * (b - BETA * xm[:, None, None]))
    Q_c_an = (GAMMA / TAU) * supersat * np.heaviside(supersat, 0.0)
    pos = np.maximum(supersat, 0.0)
    pos_sq = float(np.vdot(pos, pos))
    g_over_t = float(np.vdot(Q_c, pos) / pos_sq) if pos_sq > 0 else float("nan")

    result = {
        "label": label, "Ra": Ra, "Pr_label": Pr, "kappa": float(kappa),
        "nt": int(nt), "frames": frames, "dtypes": dtypes, "tolerance": tol,
        "Q_d_rel_l2": rel_l2(Q_d, Q_d_an),
        "Q_d_float32_floor_rel_l2": Q_d_float32_floor,
        "Q_d_max_abs": float(np.max(np.abs(Q_d - Q_d_an))),
        "Pr_implied": float(Pr_implied),
        "Q_c_rel_l2": rel_l2(Q_c, Q_c_an),
        "Q_c_max_abs": float(np.max(np.abs(Q_c - Q_c_an))),
        "gamma_over_tau_implied": g_over_t,
        "rel_q_max_abs_diff": float(np.max(np.abs(rel_q - supersat))),
        "condensing_fraction": float(np.mean(supersat > 0)),
    }

    flags = []
    if not result["Q_d_rel_l2"] <= tol:
        flags.append("Q_d")
    if abs(Pr_implied / Pr - 1.0) > PR_TOL:
        flags.append("Pr")
    if not (result["Q_c_rel_l2"] <= tol or
            (np.isnan(result["Q_c_rel_l2"]) and result["Q_c_max_abs"] == 0.0)):
        flags.append("Q_c")
    if not np.isnan(g_over_t) and abs(g_over_t / (GAMMA / TAU) - 1.0) > PR_TOL:
        flags.append("gamma/tau")
    if result["rel_q_max_abs_diff"] > 1e-12:
        flags.append("rel_q")
    result["status"] = "OK" if not flags else "MISMATCH: " + ", ".join(flags)
    return result


# ============================================================
# Main
# ============================================================

def main():
    t_start = time.time()
    if OUT_DIR.exists() and any(OUT_DIR.iterdir()):
        sys.exit(f"{OUT_DIR} exists and is not empty; refusing to overwrite.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for label in ALL_RUNS:
        print(f"Checking {label} ...", flush=True)
        res = check_run(label)
        results.append(res)
        print(f"  {res['status']}", flush=True)

    with open(OUT_DIR / "analytic_check.json", "w") as f:
        json.dump(results, f, indent=4)

    columns = ["label", "status", "Ra", "Pr_label", "Pr_implied", "kappa", "nt",
               "Q_d_rel_l2", "Q_d_max_abs", "Q_d_float32_floor_rel_l2",
               "Q_c_rel_l2", "Q_c_max_abs",
               "gamma_over_tau_implied", "rel_q_max_abs_diff", "condensing_fraction"]
    with open(OUT_DIR / "analytic_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    header = (f"{'run':<14} {'status':<22} {'Pr':>4} {'Pr_impl':>10} "
              f"{'Q_d relL2':>10} {'f32 floor':>10} {'Q_c relL2':>10} "
              f"{'g/t impl':>9} {'nt':>4}")
    lines = [header, "-" * len(header)]
    for r in results:
        if "Q_d_rel_l2" not in r:
            lines.append(f"{r['label']:<14} {r['status']}")
            continue
        lines.append(f"{r['label']:<14} {r['status'][:22]:<22} {r['Pr_label']:>4g} "
                     f"{r['Pr_implied']:>10.6g} {r['Q_d_rel_l2']:>10.2e} "
                     f"{r['Q_d_float32_floor_rel_l2']:>10.2e} "
                     f"{r['Q_c_rel_l2']:>10.2e} {r['gamma_over_tau_implied']:>9.5g} "
                     f"{r['nt']:>4d}")
    table = "\n".join(lines)
    (OUT_DIR / "analytic_check.txt").write_text(table + "\n")
    print("\n" + table)
    print(f"\nElapsed: {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()
