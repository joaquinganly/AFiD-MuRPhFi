"""Post-processing of the simulations_v2 runs: ML input fields and buoyancy source terms.

Port of the established pipeline in simulations/Staggered grid corrected/flat_layout_copies/
("Q staggered grid efficient pathing.py", "saves_q_npz.py", "Qc and Qd only.py"): same formulas,
file names, array keys, array layout (nz, ny, nt) and dtype (float64). What changed:
 - input: outputdir/flowmov/movie_zcut.h5 (the full 2D field at every output time; the v2 runs
   write no fields/*.h5 files) instead of fields/{n:05d}_*.h5;
 - parameters: Ra, Pr, grid and TFRAME from the run's bou.in (not from the folder name);
   alpha, beta, gamma, tau from its humid.in (were hard-coded); grid heights and spacings from
   outputdir/cordin_info.h5; buoyancy wall values 0 and beta - 1 (SetTempBCs.F90, moist case);
 - one script writes all three files; existing outputs are never overwritten; files are
   written under a temporary name and renamed when complete;
 - <run>_fields_Q.npz and <run>_q.npz are written uncompressed (np.savez; the old compressed
   field files were 92-96 % of their raw size), Q_d_c.npz compressed as before.

Per run folder <run>/ (arrays (nz, ny, nt), float64; frame n = movie frame n, t = n * TFRAME):
  <run>_fields_Q.npz  u      vertical velocity (AFiD vx), interpolated from z-faces to cell centres
                      v      horizontal velocity vy as stored by AFiD (cell centre in z, y-face)
                      theta  buoyancy b
                      rel_q  q - q_s, with q_s = exp(alpha (b - beta z)) at cell centres
                      Q      flux-form finite-difference estimate of Db/Dt from the frames:
                             db/dt (2nd order in time) + d(w b)/dz + d(v b)/dy
  <run>_q.npz         q      humidity
  Q_d_c.npz           Q_c    (gamma / tau) (q - q_s) H(q - q_s)        (moisture.F90, AddCondensation)
                      Q_d    kappa lap(b), kappa = (Ra Pr)^(-1/2), central differences (2nd order in
                             the interior); wall rows (f[1] - 3 f[0] + 2 b_wall) / dz^2, i.e. ghost
                             value 2 b_wall - f[0] (1st order at the wall row)
  <run>/postprocess.txt      provenance (parameters, script md5, output sizes, value ranges)
The buoyancy source term of the solver is Q_d + Q_c; the v032 training builds Q = Q_d + Q_c.

Usage: python postprocess.py [run folder names]    (default: every run in runs/ whose run.log
ends with "EXIT 0"). Memory: about 11 GB per run (runs are processed one after another).
"""
import hashlib
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


# ---------------------------------------------------------------- parameters
def read_bou(path):
    """Grid, Ra, Pr, box length and frame interval from bou.in (value line follows header line)."""
    lines = [l.split() for l in Path(path).read_text().replace("\r", "").splitlines()]
    g = {l[0]: lines[i + 1] for i, l in enumerate(lines) if l and l[0] in ("NXM", "RAYT", "ALX3", "TOUT")}
    return dict(nz=int(g["NXM"][0]), ny=int(g["NXM"][1]), ly=float(g["ALX3"][1]),
                ra=float(g["RAYT"][0]), pr=float(g["RAYT"][1]), tframe=float(g["TOUT"][1]))


def read_humid(path):
    """alpha, beta, gamma, tau from humid.in (value on the line after each header)."""
    lines = [l.strip() for l in Path(path).read_text().replace("\r", "").splitlines()]
    val = {}
    for i, l in enumerate(lines):
        for key in ("alpha", "beta", "gamma", "tau"):
            if l.split(" ")[0] == key:
                val[key] = float(lines[i + 1].split()[0])
        if l.startswith("Diffusive scale for tau"):
            val["tau_diffusive"] = lines[i + 1].split()[0]
    missing = {"alpha", "beta", "gamma", "tau", "tau_diffusive"} - set(val)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    # moisture.F90 replaces gamma < 0 by a derived value and rescales tau if the diffusive
    # scale is chosen; neither is used here, so refuse rather than reproduce them.
    if val["gamma"] < 0 or val["tau_diffusive"] != ".false.":
        raise ValueError(f"{path}: gamma < 0 or diffusive tau scaling not supported")
    return val


def read_grid(path, nz, ny, ly):
    """Cell-centre heights (nz, 1) and uniform spacings, checked against bou.in."""
    with h5py.File(path, "r") as f:
        xm, xc, ym = f["xm"][()], f["xc"][()], f["ym"][()]
    dz, dy = 1.0 / nz, ly / ny
    if xm.size != nz or xc.size != nz + 1 or ym.size != ny:
        raise ValueError(f"{path}: grid sizes {xm.size}, {xc.size}, {ym.size} vs NXM {nz}, NYM {ny}")
    if not (np.allclose(np.diff(xc), dz) and np.allclose(np.diff(ym), dy)):
        raise ValueError(f"{path}: grid is not uniform; Q_d stencil assumes a uniform grid")
    if not np.allclose(xm, 0.5 * (xc[1:] + xc[:-1])):
        raise ValueError(f"{path}: xm are not the midpoints of xc")
    return xm[:, np.newaxis], dz, dy


# ---------------------------------------------------------------- numerics (as in the old scripts)
def saturation(b, zc, alpha, beta):
    """q_s = exp(alpha (b - beta z)) at cell centres (moisture.F90)."""
    return np.exp(alpha * (b - beta * zc))


def laplacian(f, dz, dy, wall_bottom, wall_top):
    """2nd-order Laplacian of a cell-centred field f (nz, ny); Dirichlet walls via the ghost value
    2 f_wall - f[0]; periodic in y."""
    d2z = np.zeros_like(f)
    d2z[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / dz ** 2
    d2z[0] = (f[1] - 3.0 * f[0] + 2.0 * wall_bottom) / dz ** 2
    d2z[-1] = (f[-2] - 3.0 * f[-1] + 2.0 * wall_top) / dz ** 2
    d2y = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / dy ** 2
    return d2z + d2y


def condensation(q, qs, gamma, tau):
    """(gamma / tau) (q - q_s) H(q - q_s), H(0) = 0."""
    s = q - qs
    return (gamma / tau) * s * np.heaviside(s, 0.0)


def flux_divergence(b, w_edge, v, dz, dy):
    """d(w b)/dz + d(v b)/dy in flux form: w on z-faces (nz + 1 rows incl. both walls, zero there),
    v at y-faces (v[:, j] on the face between cells j - 1 and j), face values of b = mean of the
    two neighbouring cells, periodic in y."""
    nz = b.shape[0]
    bp = np.pad(b, ((1, 1), (0, 0)), mode="constant")       # pad rows only multiply w = 0 at walls
    flux_up = w_edge[1:nz + 1] * 0.5 * (b + bp[2:nz + 2])
    flux_dn = w_edge[0:nz] * 0.5 * (b + bp[0:nz])
    flux_yp = np.roll(v, -1, axis=1) * 0.5 * (b + np.roll(b, -1, axis=1))
    flux_ym = v * 0.5 * (b + np.roll(b, 1, axis=1))
    return (flux_up - flux_dn) / dz + (flux_yp - flux_ym) / dy


def time_derivative(b_all, n, dt):
    """db/dt at frame n of b_all (nz, ny, nt): central, one-sided 2nd order at the ends."""
    nt = b_all.shape[2]
    if n == 0:
        return (-3.0 * b_all[:, :, 0] + 4.0 * b_all[:, :, 1] - b_all[:, :, 2]) / (2 * dt)
    if n == nt - 1:
        return (3.0 * b_all[:, :, n] - 4.0 * b_all[:, :, n - 1] + b_all[:, :, n - 2]) / (2 * dt)
    return (b_all[:, :, n + 1] - b_all[:, :, n - 1]) / (2 * dt)


# ---------------------------------------------------------------- I/O
def save_npz(path, compressed, **arrays):
    """Write under <name>.partial and rename when complete."""
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as fh:
        (np.savez_compressed if compressed else np.savez)(fh, **arrays)
    tmp.rename(path)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg):
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}", flush=True)


def process(run):
    outs = {"fields": run / f"{run.name}_fields_Q.npz", "q": run / f"{run.name}_q.npz",
            "qdc": run / "Q_d_c.npz", "prov": run / "postprocess.txt"}
    if outs["prov"].exists():                            # written last: the run is complete
        log(f"skip {run.name}: already post-processed ({outs['prov'].name} exists)")
        return
    existing = [p.name for p in outs.values() if p.exists()]
    if existing:                                         # earlier attempt stopped part-way
        raise FileExistsError(f"incomplete earlier output {existing} without {outs['prov'].name}; "
                              f"not overwritten (move it away to redo this run)")
    t0 = time.time()
    p = read_bou(run / "bou.in")
    h = read_humid(run / "humid.in")
    zc, dz, dy = read_grid(run / "outputdir/cordin_info.h5", p["nz"], p["ny"], p["ly"])
    kappa = 1.0 / math.sqrt(p["ra"] * p["pr"])
    alpha, beta, gamma, tau = h["alpha"], h["beta"], h["gamma"], h["tau"]
    b_bot, b_top = 0.0, beta - 1.0
    nz, ny, dt = p["nz"], p["ny"], p["tframe"]
    movie = run / "outputdir/flowmov/movie_zcut.h5"
    log(f"{run.name}: Ra {p['ra']:.4g} Pr {p['pr']:g} kappa {kappa:.6e} alpha {alpha:g} beta {beta:g} "
        f"gamma {gamma:g} tau {tau:g} grid {nz} x {ny} dz {dz:.6g} dy {dy:.6g} frame dt {dt:g}")

    with h5py.File(movie, "r") as f:
        keys = sorted(f["temp"].keys())
        nt = len(keys)
        for var in ("qhum", "vx", "vy"):
            if sorted(f[var].keys()) != keys:
                raise ValueError(f"{movie}: frame keys of {var} differ from temp")
        if [int(k) for k in keys] != list(range(nt)):
            raise ValueError(f"{movie}: frame keys are not 0 .. {nt - 1}")

        b = np.empty((nz, ny, nt))
        q = np.empty((nz, ny, nt))
        for n, k in enumerate(keys):
            b[:, :, n] = f["temp"][k][()].T             # movie arrays are (ny, nz)
            q[:, :, n] = f["qhum"][k][()].T
        log(f"{run.name}: read b, q ({nt} frames, {time.time() - t0:.0f} s)")

        rel_q = np.empty_like(b)
        Q_d = np.empty_like(b)
        Q_c = np.empty_like(b)
        for n in range(nt):
            qs = saturation(b[:, :, n], zc, alpha, beta)
            rel_q[:, :, n] = q[:, :, n] - qs
            Q_d[:, :, n] = kappa * laplacian(b[:, :, n], dz, dy, b_bot, b_top)
            Q_c[:, :, n] = condensation(q[:, :, n], qs, gamma, tau)
        stats = {"b": (b.min(), b.max()), "q": (q.min(), q.max()), "rel_q": (rel_q.min(), rel_q.max()),
                 "Q_d": (Q_d.min(), Q_d.max()), "Q_c": (Q_c.min(), Q_c.max()),
                 "Q_c > 0 (fraction of points)": (float(np.mean(Q_c > 0)),)}
        save_npz(outs["qdc"], True, Q_c=Q_c, Q_d=Q_d)
        del Q_d, Q_c
        save_npz(outs["q"], False, q=q)
        del q
        log(f"{run.name}: wrote Q_d_c.npz and {outs['q'].name} ({time.time() - t0:.0f} s)")

        u = np.empty_like(b)
        v = np.empty_like(b)
        Q = np.empty_like(b)
        for n, k in enumerate(keys):
            w_edge = f["vx"][k][()].T                    # (nz + 1, ny), z-faces incl. both walls
            if w_edge.shape != (nz + 1, ny):
                raise ValueError(f"{movie}: vx frame {k} has shape {w_edge.shape}")
            w_edge[0] = 0.0                              # no penetration (as in the old script)
            w_edge[-1] = 0.0
            u[:, :, n] = 0.5 * (w_edge[:-1] + w_edge[1:])
            v[:, :, n] = f["vy"][k][()].T
            Q[:, :, n] = time_derivative(b, n, dt) + flux_divergence(b[:, :, n], w_edge, v[:, :, n], dz, dy)
    stats.update({"u": (u.min(), u.max()), "v": (v.min(), v.max()), "Q (finite difference)": (Q.min(), Q.max())})
    save_npz(outs["fields"], False, u=u, v=v, theta=b, rel_q=rel_q, Q=Q)
    del u, v, Q, b, rel_q
    log(f"{run.name}: wrote {outs['fields'].name} ({time.time() - t0:.0f} s)")

    me = Path(__file__).resolve()
    lines = [f"created  {datetime.now(timezone.utc).isoformat(timespec='seconds')} by {me.name} "
             f"(md5 {md5(me)})",
             f"input    {movie.relative_to(run)} ({movie.stat().st_size} bytes), bou.in, humid.in, "
             f"outputdir/cordin_info.h5",
             f"params   Ra {p['ra']:.6g}  Pr {p['pr']:g}  kappa = (Ra Pr)^-1/2 = {kappa:.10e}",
             f"         alpha {alpha:g}  beta {beta:g}  gamma {gamma:g}  tau {tau:g}  "
             f"b walls {b_bot:g} (bottom), {b_top:g} (top)",
             f"grid     nz {nz}  ny {ny}  dz {dz:.10g}  dy {dy:.10g}  nt {nt}  frame dt {dt:g} (TFRAME)",
             "arrays   (nz, ny, nt) float64; frame n = movie frame n"]
    for name, path in (("fields", outs["fields"]), ("q", outs["q"]), ("qdc", outs["qdc"])):
        lines.append(f"output   {path.name} ({path.stat().st_size} bytes)")
    lines += [f"range    {k}: " + "  ".join(f"{x:.6g}" for x in v) for k, v in stats.items()]
    lines.append(f"time     {time.time() - t0:.0f} s")
    tmp = outs["prov"].with_name(outs["prov"].name + ".partial")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.rename(outs["prov"])
    log(f"{run.name}: done ({time.time() - t0:.0f} s)")


def finished_runs():
    out = []
    for d in sorted(RUNS.iterdir()):
        log_file = d / "run.log"
        if (d / "bou.in").exists() and log_file.exists():
            tail = log_file.read_text(errors="replace").strip().splitlines()[-1:]
            if tail == ["EXIT 0"]:
                out.append(d)
    return sorted(out, key=lambda d: read_bou(d / "bou.in")["ra"])


if __name__ == "__main__":
    runs = [RUNS / name for name in sys.argv[1:]] if len(sys.argv) > 1 else finished_runs()
    failed = []
    for run in runs:
        try:
            process(run)
        except Exception as e:                           # report and continue with the next run
            log(f"FAILED {run.name}: {type(e).__name__}: {e}")
            failed.append(run.name)
    log(f"finished: {len(runs) - len(failed)} of {len(runs)} runs ok" + (f"; failed {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
