# Rainy-Bénard post-processing scripts (flat-layout runs)

These are the exact scripts run on 2026-11-03 to produce the post-processed fields for the runs
`Ra_5e6`, `Ra_75e6`, `Ra_5e7`, `Ra_25e7` and `Ra_175e7` in `examples/RainyBenard/simulations/`.
The input run folders and the outputs are git-ignored, so they are not part of this repository.

| Script | Output (written into each run folder) | Array shape (z, y, t) |
|---|---|---|
| `saves_q_npz.py` | `<run>_q.npz` (`q`) | (64, 1024, 324) |
| `Q staggered grid efficient pathing.py` | `<run>_fields_Q.npz` (`u`, `v`, `theta`, `rel_q`, `Q`) | (64, 1024, 324) |
| `Qc and Qd only.py` | `Q_d_c.npz` (`Q_c`, `Q_d`) | (64, 1024, 311) |
| `Q GT efficient pathing.py` | `Q_ground_truth.npz` (`Q_gt`) | (64, 1024, 324) |

## Provenance

Each script is a copy of the file of the same name in
`examples/RainyBenard/simulations/Staggered grid corrected/` (git-ignored). The copies were made and
run from its subfolder `flat_layout_copies/`, which is what the `'../<name>'` in each script's header
comment refers to; that relative path does not resolve from this folder.

Only path and loading logic differ from the originals; each change is marked `PATH CHANGE`:

- All four: `base_dir` points to `/workspace/examples/RainyBenard/simulations` (Docker mount) instead of a Windows path.
- All four: the automatically discovered folder list is overridden by the five runs above. Only these
  five folders are processed. There is no overwrite guard, so re-running overwrites their outputs.
- All four: fields are read from `<run>/fields/` instead of `<run>/outputdir/fields/` (these runs have no `outputdir/`).
- All except `saves_q_npz.py`: the vertical grid `xm` is read directly from `<run>/cordin_info.h5`,
  because `afid.Grid` only looks in `<run>/outputdir/`. `import afidtools` is still at the top of these scripts.

Finite-difference schemes and parameters are unchanged from the originals. The parameters are hard-coded:

- All scripts: `nz = 64`, `ny = 1024`, `nt = 324` (`nt = 311` in `Qc and Qd only.py`).
- All except `saves_q_npz.py`: `Lz = 1`, `Ly = 16`, `dt = 1`; saturation humidity `qs = exp(3 (theta - 2 xm))`, with `xm` the cell-centre vertical coordinate.
- `Qc and Qd only.py` and `Q GT efficient pathing.py` only: `beta = 2`, `gamma = 2`, `tau = 0.1`; `Pr = 1` for `Ra_*` folders; `Ra` is parsed from the folder name (e.g. `Ra_175e7` gives `1.75e9`).

These five run folders contain no `bou.in` or `humid.in`, so these parameter values have not been
checked against the simulation inputs.

## How they were run

Docker image `my-tf-env:latest` (Python 3.12.3, numpy 1.26.4, h5py 3.10.0, tqdm 4.67.1, matplotlib 3.11.1),
repository mounted at `/workspace`. `base_dir` is absolute, so the working directory does not matter
for the data paths.

```bash
cd /workspace/rainybenard_postprocessing
export PYTHONPATH=/workspace/examples/RainyBenard/simulations   # for `import afidtools`
export MPLBACKEND=Agg
python "saves_q_npz.py"
python "Qc and Qd only.py"
python "Q staggered grid efficient pathing.py"   # these last two are independent;
python "Q GT efficient pathing.py"               # on 2026-11-03 they ran concurrently in separate containers
```
