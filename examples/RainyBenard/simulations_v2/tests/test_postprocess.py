"""Tests for simulations_v2/postprocess.py.  Run:  python simulations_v2/tests/test_postprocess.py
1. The functions of postprocess.py reproduce the stored outputs of the old scripts
   (Q_d_c.npz, Ra_5e6_fields_Q.npz, Ra_5e6_q.npz) on the first frames of the old run Ra_5e6
   (64 x 1024; the old scripts used Ra = 5e6 parsed from the folder name, Pr = 1, alpha 3,
   beta 2, gamma 2, tau 0.1).
2. The movie file holds the same fields as the fields/*.h5 files (old run Ra_5e6), so reading
   the movie instead of fields/ does not change the data.
3. Parameters are read from a v2 run's bou.in, humid.in and cordin_info.h5."""
import math
import sys
from pathlib import Path

import tempfile

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import postprocess as pp  # noqa: E402

OLD = Path(__file__).resolve().parents[2] / "simulations" / "Ra_5e6"
V2RUN = Path(__file__).resolve().parents[1] / "runs" / "Ra_7.5e6_Pr_1"
NZ, NY, NF = 64, 1024, 6


def read_old(name, n, edge=False):
    """fields/{n:05d}_<name> as the old scripts read it: (NZ, NY) cell centres (last of NZ + 1
    rows dropped), or (NZ + 1, NY) z-faces for vx."""
    with h5py.File(OLD / "fields" / f"{n:05d}_{name}", "r") as f:
        a = np.squeeze(np.array(f["var"]))
    if a.shape[0] == NY:
        a = a.T
    return a if edge else a[:-1]


def same(new, old):
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=1e-12 * np.abs(old).max())


def test_matches_old_pipeline():
    with h5py.File(OLD / "cordin_info.h5", "r") as f:
        zc = np.asarray(f["xm"][()]).squeeze()[:NZ, np.newaxis]
    dz, dy = 1.0 / NZ, 16.0 / NY
    kappa = 1.0 / math.sqrt(5e6 * 1.0)
    b = np.stack([read_old("temp.h5", n) for n in range(NF)], axis=2)
    q = np.stack([read_old("humid.h", n) for n in range(NF)], axis=2)
    v = np.stack([read_old("vy.h5", n) for n in range(NF)], axis=2)
    old = {}
    for fname, keys in (("Q_d_c.npz", ("Q_d", "Q_c")), ("Ra_5e6_fields_Q.npz", ("u", "v", "theta", "rel_q", "Q")),
                        ("Ra_5e6_q.npz", ("q",))):
        with np.load(OLD / fname) as d:
            for k in keys:
                old[k] = d[k][:, :, :NF]
    for n in range(NF):
        qs = pp.saturation(b[:, :, n], zc, 3.0, 2.0)
        same(b[:, :, n], old["theta"][:, :, n])
        same(q[:, :, n], old["q"][:, :, n])
        same(v[:, :, n], old["v"][:, :, n])
        same(q[:, :, n] - qs, old["rel_q"][:, :, n])
        same(kappa * pp.laplacian(b[:, :, n], dz, dy, 0.0, 1.0), old["Q_d"][:, :, n])
        same(pp.condensation(q[:, :, n], qs, 2.0, 0.1), old["Q_c"][:, :, n])
        w = read_old("vx.h5", n, edge=True)
        w[0] = 0.0
        w[-1] = 0.0
        same(0.5 * (w[:-1] + w[1:]), old["u"][:, :, n])
        if n < NF - 1:          # the last loaded frame would need the backward stencil; the old file used central
            Q = pp.time_derivative(b, n, 1.0) + pp.flux_divergence(b[:, :, n], w, v[:, :, n], dz, dy)
            same(Q, old["Q"][:, :, n])


def test_movie_equals_fields():
    with h5py.File(OLD / "flowmov" / "movie_zcut.h5", "r") as f:
        for name, var, edge in (("temp.h5", "temp", False), ("humid.h", "qhum", False),
                                ("vx.h5", "vx", True), ("vy.h5", "vy", False)):
            for n in range(NF):
                np.testing.assert_array_equal(f[var][f"{n:05d}"][()].T, read_old(name, n, edge))


def test_process_on_synthetic_run(tmp_path):
    """process() end to end on a small random run folder: keys, transposes, wall forcing,
    provenance, no overwriting."""
    nz, ny, nt, ly = 8, 16, 5, 2.0
    run = tmp_path / "Ra_1e6_Pr_1"
    (run / "outputdir" / "flowmov").mkdir(parents=True)
    (run / "bou.in").write_text("NXM NYM NZM NSST\n8 16 1 3\nALX3 YLEN ZLEN\n1.0 2.0 0.1\n"
                                "RAYT PRAT\n1e6 1.0\nTOUT TFRAME SAVE_3D\n1.0 1.0 6000.0\n")
    (run / "humid.in").write_text("alpha (a)\n3.0\n\nbeta (b)\n2.0\n\ngamma (g)\n2.0\n\ntau (t)\n0.1\n\n"
                                  "Diffusive scale for tau? (.true. / .false.)\n.false.\n")
    xc = np.linspace(0.0, 1.0, nz + 1)
    with h5py.File(run / "outputdir" / "cordin_info.h5", "w") as f:
        f["xc"], f["xm"], f["ym"] = xc, 0.5 * (xc[1:] + xc[:-1]), (np.arange(ny) + 0.5) * ly / ny
    rng = np.random.default_rng(0)
    raw = {"temp": rng.uniform(0, 1.5, (nt, ny, nz)), "qhum": rng.uniform(0, 1, (nt, ny, nz)),
           "vx": rng.normal(size=(nt, ny, nz + 1)), "vy": rng.normal(size=(nt, ny, nz))}
    with h5py.File(run / "outputdir" / "flowmov" / "movie_zcut.h5", "w") as f:
        for var, a in raw.items():
            for n in range(nt):
                f[f"{var}/{n:05d}"] = a[n]
    pp.process(run)

    b = np.moveaxis(raw["temp"], 0, -1).transpose(1, 0, 2)          # (nz, ny, nt)
    q = np.moveaxis(raw["qhum"], 0, -1).transpose(1, 0, 2)
    v = np.moveaxis(raw["vy"], 0, -1).transpose(1, 0, 2)
    zc = (0.5 * (xc[1:] + xc[:-1]))[:, None]
    kappa = 1 / math.sqrt(1e6)
    with np.load(run / "Ra_1e6_Pr_1_fields_Q.npz") as d1, np.load(run / "Ra_1e6_Pr_1_q.npz") as d2, \
            np.load(run / "Q_d_c.npz") as d3:
        assert sorted(d1.files) == ["Q", "rel_q", "theta", "u", "v"] and d2.files == ["q"]
        assert sorted(d3.files) == ["Q_c", "Q_d"]
        np.testing.assert_array_equal(d1["theta"], b)
        np.testing.assert_array_equal(d2["q"], q)
        np.testing.assert_array_equal(d1["v"], v)
        for n in range(nt):
            qs = pp.saturation(b[:, :, n], zc, 3.0, 2.0)
            w = raw["vx"][n].T.copy()
            w[0] = 0.0
            w[-1] = 0.0
            np.testing.assert_array_equal(d1["rel_q"][:, :, n], q[:, :, n] - qs)
            np.testing.assert_array_equal(d3["Q_d"][:, :, n], kappa * pp.laplacian(b[:, :, n], 1 / nz, ly / ny, 0.0, 1.0))
            np.testing.assert_array_equal(d3["Q_c"][:, :, n], pp.condensation(q[:, :, n], qs, 2.0, 0.1))
            np.testing.assert_array_equal(d1["u"][:, :, n], 0.5 * (w[:-1] + w[1:]))
            np.testing.assert_array_equal(
                d1["Q"][:, :, n],
                pp.time_derivative(b, n, 1.0) + pp.flux_divergence(b[:, :, n], w, v[:, :, n], 1 / nz, ly / ny))
    assert (run / "postprocess.txt").exists() and not list(run.glob("*.partial"))
    pp.process(run)                                      # complete: skipped, nothing rewritten
    (run / "postprocess.txt").unlink()
    try:                                                 # incomplete: refused, not overwritten
        pp.process(run)
    except FileExistsError:
        pass
    else:
        raise AssertionError("incomplete output was not refused")


def test_parameters_from_run_files():
    p = pp.read_bou(V2RUN / "bou.in")
    h = pp.read_humid(V2RUN / "humid.in")
    assert (p["nz"], p["ny"], p["ly"], p["ra"], p["pr"], p["tframe"]) == (128, 2048, 16.0, 7.5e6, 1.0, 1.0)
    assert (h["alpha"], h["beta"], h["gamma"], h["tau"]) == (3.0, 2.0, 2.0, 0.1)
    zc, dz, dy = pp.read_grid(V2RUN / "outputdir" / "cordin_info.h5", 128, 2048, 16.0)
    assert zc.shape == (128, 1) and dz == 1 / 128 and dy == 16 / 2048
    assert np.isclose(zc[0, 0], 0.5 / 128) and np.isclose(zc[-1, 0], 1 - 0.5 / 128)


if __name__ == "__main__":
    tests = [(test_matches_old_pipeline, (OLD / "Q_d_c.npz").exists(), "old run Ra_5e6 not present"),
             (test_movie_equals_fields, (OLD / "flowmov" / "movie_zcut.h5").exists(), "old run Ra_5e6 not present"),
             (test_parameters_from_run_files, V2RUN.exists(), "v2 run not present"),
             (lambda: test_process_on_synthetic_run(Path(tempfile.mkdtemp())), True, "")]
    names = ["test_matches_old_pipeline", "test_movie_equals_fields", "test_parameters_from_run_files",
             "test_process_on_synthetic_run"]
    failed = 0
    for name, (fn, available, why) in zip(names, tests):
        if not available:
            print(f"SKIP {name}: {why}")
            continue
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
