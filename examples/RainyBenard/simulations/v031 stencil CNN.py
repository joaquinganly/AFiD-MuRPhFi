# ============================================================
# v031 - 3x3 stencil CNN for the buoyancy source terms (calibration stage)
#
# Targets, trained as independent models with the same architecture, seeds
# and batch order (as stored in <run>/Q_d_c.npz by
# rainybenard_postprocessing/"Qc and Qd only.py"; Vallis, Parker & Tobias
# 2019, eqs. 2.25c and 3.6):
#   Q_d = kappa Lap(b),                   kappa = 1/sqrt(Pr Ra)
#   Q_c = (gamma/tau) (q - q_s) H(q - q_s), q_s = exp(alpha (b - beta z))
#   Q   = Q_d + Q_c                        (right-hand side of the b equation)
# Inputs : configurable channels (INPUTS). Run R0 = b, q, z, log10 Ra, log10 Pr.
#          z is a channel holding the cell-centre height of each grid row.
# Model  : receptive field exactly 3x3: one 3x3 convolution, then 1x1
#          (per-point) layers. The padding reproduces the stencil used for
#          Q_d: periodic in y; in z, ghost value 2*wall - edge for channels
#          with a Dirichlet wall value (b, q, q - q_s, z), edge value for the
#          constant parameter maps.
# Loss   : plain MSE on the z-scored target, all grid points.
# Split  : by run: TRAIN_RUNS / VAL_RUNS. HELD_OUT_RUNS are never loaded
#          (only their .npz headers are read, for NT_COMMON).
# Output : OUT_DIR/<target>/ holds the config, normalisation constants and a
#          summary table; OUT_DIR/<target>/seed_<s>/ the best model, history
#          (json + csv) and loss vs epoch (log and linear scale; normalised
#          and un-normalised units). No test evaluation, XAI or ablation here.
# Usage  : python "v031 stencil CNN.py" [Q_d] [Q_c] [Q] [42] [43] [44]
#          trains the targets and seeds given (default: all TARGETS and
#          SEEDS), so that each call can be kept short (e.g. one seed per
#          call). Run calls one after another: calls for the same target share
#          its config.json (written by the first call, checked by later ones)
#          and summary.json (merged by seed).
# ============================================================

import json
import math
import os
import platform
import random
import re
import sys
import time
import zipfile
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, mixed_precision
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. Settings
# ============================================================

VERSION = "v031"

TARGETS = ("Q_d", "Q_c", "Q")
SEEDS = (42, 43, 44)
INPUTS = ("b", "q", "z", "log10_Ra", "log10_Pr")      # run R0

NZ = 64
NY = 1024

# Physical parameters, identical for all runs (humid.in). Wall values of b
# from SetTempBCs.F90 with moisture on, as used for Q_d in the
# post-processing. Humidity walls from SetHumidityBCs (moisture.F90) with
# humid.in "top 2, bottom 1": saturated bottom, q = exp(alpha * b_bottom);
# unsaturated top, q = 0.
ALPHA = 3.0
BETA = 2.0
B_WALL = (0.0, BETA - 1.0)                                   # (bottom, top)
Q_WALL = (math.exp(ALPHA * B_WALL[0]), 0.0)
QS_WALL = (math.exp(ALPHA * (B_WALL[0] - BETA * 0.0)),
           math.exp(ALPHA * (B_WALL[1] - BETA * 1.0)))
Z_WALL = (0.0, 1.0)

# Available input channels. "source": where the data come from;
# "wall": (bottom, top) physical wall value for the ghost-cell padding in z,
# or None to repeat the edge row (constant maps).
CHANNELS = {
    "b":          {"source": ("fields", "theta"), "wall": B_WALL},
    "q":          {"source": ("q", "q"),          "wall": Q_WALL},
    "q_minus_qs": {"source": ("fields", "rel_q"),
                   "wall": (Q_WALL[0] - QS_WALL[0], Q_WALL[1] - QS_WALL[1])},
    "z":          {"source": "grid",              "wall": Z_WALL},
    "log10_Ra":   {"source": "param",             "wall": None},
    "log10_Pr":   {"source": "param",             "wall": None},
}

# Architecture: receptive field exactly 3x3.
STENCIL_FILTERS = 32
POINTWISE_FILTERS = (64, 64)
ACTIVATION = "relu"

BATCH_SIZE = 2
EPOCHS = 150
LEARNING_RATE = 1e-3          # suggested default step size (Kingma & Ba 2015, Alg. 1)
CLIPNORM = 1.0
LR_PLATEAU = {"factor": 0.5, "patience": 5, "min_lr": 1e-7}   # ReduceLROnPlateau
EARLY_STOP_PATIENCE = 12
USE_MIXED_PRECISION = False   # float32 throughout
USE_TF32 = False              # TensorFlow enables TF32 by default on Ampere and newer
                              # GPUs: float32 inputs of convolutions and matmuls are
                              # rounded to 10 bits of precision (docstring of
                              # tf.config.experimental.enable_tensor_float_32_execution)

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

BASE_DATA_DIR = SCRIPT_DIR
OUT_DIR = SCRIPT_DIR / VERSION

ALL_RUNS = [
    "Pr_05_Ra_1e4", "Pr_05_Ra_1e5", "Pr_05_Ra_1e6", "Pr_05_Ra_1e7",
    "Pr_05_Ra_1e8", "Pr_05_Ra_1e9",
    "Pr_2_Ra_1e4",  "Pr_2_Ra_1e5",  "Pr_2_Ra_1e6",  "Pr_2_Ra_1e8",
    "Pr_2_Ra_1e9",
    "Ra_1e4", "Ra_1e5", "Ra_1e6", "Ra_5e6", "Ra_1e7", "Ra_5e7", "Ra_75e6",
    "Ra_1e8", "Ra_25e7", "Ra_1e9", "Ra_175e7",
]

# Same four hold-outs as the v029 evaluation (one or two per Pr value).
VAL_RUNS = ["Pr_05_Ra_1e6", "Pr_05_Ra_1e7", "Pr_2_Ra_1e5", "Ra_1e8"]
# Two runs not used by any earlier model; reserved, never loaded here.
HELD_OUT_RUNS = ["Ra_5e6", "Ra_25e7"]
TRAIN_RUNS = [r for r in ALL_RUNS if r not in VAL_RUNS + HELD_OUT_RUNS]


# ============================================================
# 2. Data loading
# ============================================================

def extract_ra_pr(label):
    """Ra and Pr from the folder name (CLAUDE.md, "Simulation run folders");
    same parsing as the v027 script."""
    pr_match = re.search(r"Pr_(\d+)", label)
    if pr_match:
        pr_str = pr_match.group(1)
        pr_val = float(pr_str) if len(pr_str) == 1 else float(pr_str) / 10.0
    else:
        pr_val = 1.0
    ra_match = re.search(r"Ra_([0-9eE\+\-]+)", label)
    if ra_match is None:
        raise ValueError(f"Cannot parse Ra from {label}")
    return float(ra_match.group(1)), pr_val


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


def field_channels():
    """Input channels read from .npz files, in INPUTS order."""
    return [c for c in INPUTS if isinstance(CHANNELS[c]["source"], tuple)]


def needed_files():
    """(file kind, npz key) pairs this configuration reads."""
    pairs = [("qdc", "Q_d"), ("qdc", "Q_c")]
    pairs += [CHANNELS[c]["source"] for c in field_channels()]
    return pairs


def npz_frame_count(path, key):
    """Length of the last (time) axis of array `key` in an .npz, read from
    the .npy header only, without loading the data."""
    with zipfile.ZipFile(path) as zf, zf.open(f"{key}.npy") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(f)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(f)
        else:
            raise ValueError(f"{path} [{key}]: unsupported .npy version {version}")
    return int(shape[-1])


def common_frame_count(labels):
    """Smallest number of frames over all runs and all arrays read."""
    counts = {}
    for label in labels:
        paths = run_paths(label)
        counts[label] = {f"{kind}:{key}": npz_frame_count(paths[kind], key)
                         for kind, key in needed_files()}
    nt_common = min(min(c.values()) for c in counts.values())
    print("Frame counts:")
    for label, c in counts.items():
        print(f"  {label:<14s} " + "  ".join(f"{k}={v}" for k, v in c.items()))
    print(f"NT_COMMON = {nt_common}")
    return nt_common, counts


def read_xm(label):
    """Cell-centre heights of the grid rows (cordin_info.h5, "xm")."""
    with h5py.File(run_paths(label)["grid"], "r") as f:
        xm = np.asarray(f["xm"][()], dtype=np.float64).squeeze()
    if xm.size == NZ + 1:
        xm = xm[:-1]
    if xm.size != NZ:
        raise ValueError(f"{label}: xm has {xm.size} points, expected {NZ}")
    if not np.allclose(xm, (np.arange(NZ) + 0.5) / NZ, rtol=0.0, atol=1e-6):
        raise ValueError(f"{label}: xm is not the uniform cell-centre grid that "
                         f"the Q_d stencil and the z padding assume")
    return xm


def load_run(label, nt):
    """
    Returns a dict with
      fields : (nt, NZ, NY, n_field_channels) float32, raw, INPUTS order
      Q_d    : (nt, NZ, NY) float32
      Q_c    : (nt, NZ, NY) float32
      xm     : (NZ,) cell-centre heights, if "z" is an input
    Frame n is DNS output file n in every .npz (all post-processing scripts
    loop n = 0 .. nt-1 over fields/{n:05d}_*.h5), so sample t pairs the
    inputs at t with Q_d(t) and Q_c(t).
    """
    paths = run_paths(label)
    chans = field_channels()
    kinds = sorted({CHANNELS[c]["source"][0] for c in chans})

    raw = {}
    for kind in kinds:
        with np.load(paths[kind]) as data:
            for c in chans:
                src_kind, key = CHANNELS[c]["source"]
                if src_kind == kind:
                    raw[c] = data[key]
    with np.load(paths["qdc"]) as data:
        q_d = data["Q_d"]
        q_c = data["Q_c"]

    for name, arr in list(raw.items()) + [("Q_d", q_d), ("Q_c", q_c)]:
        if arr.shape[:2] != (NZ, NY) or arr.shape[-1] < nt:
            raise ValueError(f"{label}: '{name}' has shape {arr.shape}, "
                             f"expected ({NZ}, {NY}, >= {nt})")

    fields = np.empty((nt, NZ, NY, len(chans)), dtype=np.float32)
    for i, c in enumerate(chans):
        fields[..., i] = np.moveaxis(raw[c][:, :, :nt], -1, 0)
    q_d = np.ascontiguousarray(np.moveaxis(q_d[:, :, :nt], -1, 0), dtype=np.float32)
    q_c = np.ascontiguousarray(np.moveaxis(q_c[:, :, :nt], -1, 0), dtype=np.float32)
    del raw

    ra, pr = extract_ra_pr(label)
    print(f"  {label:<14s} Ra = {ra:.3e}  Pr = {pr:<4g}  samples = {nt}")
    return {
        "label": label,
        "fields": fields,
        "Q_d": q_d,
        "Q_c": q_c,
        "xm": read_xm(label) if "z" in INPUTS else None,
        "params": {"log10_Ra": float(np.log10(ra)), "log10_Pr": float(np.log10(pr))},
        "n_samples": int(nt),
    }


def target_array(run, target):
    if target == "Q_d":
        return run["Q_d"]
    if target == "Q_c":
        return run["Q_c"]
    if target == "Q":
        return run["Q_d"] + run["Q_c"]
    raise ValueError(f"Unknown target {target}")


# ============================================================
# 3. Normalisation (z-score, training runs only)
# ============================================================

def _mean_std(total, total_sq, count):
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean ** 2, 0.0))
    return mean, (std if std > 0.0 else 1.0)


def fit_input_norm(train_runs):
    """Per input channel (mean, std) over all grid points and frames of the
    training runs; the parameter maps are weighted by samples per run.
    float64 accumulators."""
    norm = {}
    chans = field_channels()
    for c in INPUTS:
        s = sq = n = 0.0
        for run in train_runs:
            if c in chans:
                x = run["fields"][..., chans.index(c)].astype(np.float64).ravel()
                s += x.sum(); sq += x @ x; n += x.size
            elif c == "z":
                x = run["xm"]
                w = run["n_samples"] * NY
                s += w * x.sum(); sq += w * (x @ x); n += w * x.size
            else:
                v = run["params"][c]
                w = run["n_samples"] * NZ * NY
                s += w * v; sq += w * v * v; n += w
        norm[c] = _mean_std(s, sq, n)
    return norm


def fit_target_norm(train_runs, targets):
    norm = {}
    for target in targets:
        s = sq = n = 0.0
        for run in train_runs:
            y = target_array(run, target).astype(np.float64).ravel()
            s += y.sum(); sq += y @ y; n += y.size
        norm[target] = _mean_std(s, sq, n)
    return norm


def apply_input_norm(run, norm):
    for i, c in enumerate(field_channels()):
        mean, std = norm[c]
        x = run["fields"][..., i]
        x -= np.float32(mean)
        x /= np.float32(std)
    if "z" in INPUTS:
        mean, std = norm["z"]
        run["z_norm"] = ((run["xm"] - mean) / std).astype(np.float32)
    run["params_norm"] = {c: np.float32((run["params"][c] - norm[c][0]) / norm[c][1])
                          for c in INPUTS if CHANNELS[c]["source"] == "param"}


def padding_args(input_norm):
    """Wall values in normalised units, for StencilPad."""
    bottom, top, dirichlet = [], [], []
    for c in INPUTS:
        wall = CHANNELS[c]["wall"]
        mean, std = input_norm[c]
        if wall is None:
            bottom.append(0.0); top.append(0.0); dirichlet.append(0.0)
        else:
            bottom.append((wall[0] - mean) / std)
            top.append((wall[1] - mean) / std)
            dirichlet.append(1.0)
    return {"wall_bottom": bottom, "wall_top": top, "dirichlet": dirichlet}


# ============================================================
# 4. Batch generator
# ============================================================

class RunSnapshots(tf.keras.utils.Sequence):
    """
    Yields (X, Y): X (B, NZ, NY, len(INPUTS)) in INPUTS order,
    Y (B, NZ, NY, 1) = z-scored target. The z map, the parameter maps and
    the target Q = Q_d + Q_c are built per batch, so only the loaded fields,
    Q_d and Q_c are held in memory.
    """

    def __init__(self, runs, target, target_norm, batch_size, shuffle, seed):
        super().__init__()
        self.runs = runs
        self.target = target
        self.mean = np.float32(target_norm[0])
        self.std = np.float32(target_norm[1])
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.index = np.array(
            [(r, t) for r, run in enumerate(runs) for t in range(run["n_samples"])],
            dtype=np.int64,
        )
        chans = field_channels()
        self.plan = [(c, chans.index(c) if c in chans else None) for c in INPUTS]
        if self.shuffle:
            self.rng.shuffle(self.index)

    def __len__(self):
        return math.ceil(len(self.index) / self.batch_size)

    def __getitem__(self, i):
        rows = self.index[i * self.batch_size:(i + 1) * self.batch_size]
        X = np.empty((len(rows), NZ, NY, len(INPUTS)), dtype=np.float32)
        Y = np.empty((len(rows), NZ, NY, 1), dtype=np.float32)
        for k, (r, t) in enumerate(rows):
            run = self.runs[r]
            for ch, (c, idx) in enumerate(self.plan):
                if idx is not None:
                    X[k, ..., ch] = run["fields"][t, ..., idx]
                elif c == "z":
                    X[k, ..., ch] = run["z_norm"][:, None]
                else:
                    X[k, ..., ch] = run["params_norm"][c]
            if self.target == "Q":
                y = run["Q_d"][t] + run["Q_c"][t]
            else:
                y = run[self.target][t]
            Y[k, ..., 0] = (y - self.mean) / self.std
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.index)


# ============================================================
# 5. Model
# ============================================================

@tf.keras.utils.register_keras_serializable(package="v031")
class StencilPad(layers.Layer):
    """
    Pads (z, y) by one cell on each side for a 'valid' 3x3 convolution.
    y (axis 2): periodic. z (axis 1, row 0 = bottom wall): for channels with
    dirichlet = 1, ghost = 2 * wall - edge (cell-centred Dirichlet ghost
    point, as in the Laplacian of "Qc and Qd only.py"); otherwise the edge
    row is repeated. Wall values are in normalised units; the ghost formula
    is unchanged by the affine z-score.
    """

    def __init__(self, wall_bottom, wall_top, dirichlet, **kwargs):
        super().__init__(**kwargs)
        self.wall_bottom = [float(v) for v in wall_bottom]
        self.wall_top = [float(v) for v in wall_top]
        self.dirichlet = [float(v) for v in dirichlet]

    def call(self, x):
        wb = tf.constant(self.wall_bottom, dtype=x.dtype)
        wt = tf.constant(self.wall_top, dtype=x.dtype)
        d = tf.constant(self.dirichlet, dtype=x.dtype)
        bottom = x[:, :1]
        top = x[:, -1:]
        ghost_bottom = d * (2.0 * wb - bottom) + (1.0 - d) * bottom
        ghost_top = d * (2.0 * wt - top) + (1.0 - d) * top
        x = tf.concat([ghost_bottom, x, ghost_top], axis=1)
        return tf.concat([x[:, :, -1:], x, x[:, :, :1]], axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({"wall_bottom": self.wall_bottom,
                       "wall_top": self.wall_top,
                       "dirichlet": self.dirichlet})
        return config


def set_seeds(seed):
    """Seeds Python, NumPy and TensorFlow. tf.keras.utils.set_random_seed is
    not used: with tf_keras 2.17 (TF_USE_LEGACY_KERAS=1 in the NVIDIA image)
    on Python 3.12 it makes unseeded initialisers call randint(1, 1e9), which
    raises TypeError (tf_keras/src/backend.py)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_stencil_cnn(pad_args, name, seed):
    """Each layer gets an explicitly seeded initialiser, so the initial
    weights depend only on `seed` (identical across targets)."""
    def init(k):
        return tf.keras.initializers.GlorotUniform(seed=seed * 100 + k)

    inputs = layers.Input(shape=(NZ, NY, len(INPUTS)), name="inputs")
    x = StencilPad(**pad_args, name="stencil_pad")(inputs)
    x = layers.Conv2D(STENCIL_FILTERS, 3, padding="valid", activation=ACTIVATION,
                      kernel_initializer=init(0), name="stencil_3x3")(x)
    for i, n_filters in enumerate(POINTWISE_FILTERS, start=1):
        x = layers.Conv2D(n_filters, 1, activation=ACTIVATION,
                          kernel_initializer=init(i), name=f"pointwise_{i}")(x)
    outputs = layers.Conv2D(1, 1, activation="linear", dtype="float32",
                            kernel_initializer=init(len(POINTWISE_FILTERS) + 1),
                            name="output")(x)
    return models.Model(inputs, outputs, name=name)


# ============================================================
# 6. Helpers
# ============================================================

def parse_args(argv):
    """Targets and seeds to train in this call, from the command-line
    arguments (e.g. `Q_d 42`); all TARGETS and/or all SEEDS if none given."""
    bad = ValueError(f"Bad arguments {list(argv)}; give targets from {TARGETS} "
                     f"and/or seeds from {SEEDS}")
    targets = tuple(a for a in argv if a in TARGETS)
    try:
        seeds = tuple(int(a) for a in argv if a not in TARGETS)
    except ValueError:
        raise bad from None
    targets, seeds = targets or TARGETS, seeds or SEEDS
    if (len(set(targets)) != len(targets) or len(set(seeds)) != len(seeds)
            or not set(seeds) <= set(SEEDS)):
        raise bad
    return targets, seeds


def check_config(targets, seeds):
    unknown = [c for c in INPUTS if c not in CHANNELS]
    if unknown or len(set(INPUTS)) != len(INPUTS):
        raise ValueError(f"Bad INPUTS {INPUTS}; available: {list(CHANNELS)}")
    for target in targets:
        summary_path = OUT_DIR / target / "summary.json"
        done = ({s["seed"] for s in json.loads(summary_path.read_text())}
                if summary_path.exists() else set())
        for seed in seeds:
            out = OUT_DIR / target / f"seed_{seed}"
            if out.exists():
                sys.exit(f"{out} exists; refusing to overwrite. "
                         f"Move it aside or change VERSION.")
            if seed in done:
                sys.exit(f"{summary_path} already has seed {seed}; refusing to "
                         f"replace its entry. Change VERSION.")
    for name, runs in (("ALL_RUNS", ALL_RUNS), ("VAL_RUNS", VAL_RUNS),
                       ("HELD_OUT_RUNS", HELD_OUT_RUNS)):
        if len(set(runs)) != len(runs):
            raise ValueError(f"{name} contains duplicates")
    unknown = (set(VAL_RUNS) | set(HELD_OUT_RUNS)) - set(ALL_RUNS)
    if unknown:
        raise ValueError(f"Unknown run labels: {sorted(unknown)}")
    if set(VAL_RUNS) & set(HELD_OUT_RUNS):
        raise ValueError("VAL_RUNS and HELD_OUT_RUNS overlap")

    kinds = {kind for kind, _ in needed_files()}
    missing = [str(run_paths(label)[kind]) for label in ALL_RUNS for kind in sorted(kinds)
               if not run_paths(label)[kind].is_file()]
    if "z" in INPUTS:
        missing += [str(run_paths(label)["grid"]) for label in TRAIN_RUNS + VAL_RUNS
                    if not run_paths(label)["grid"].is_file()]
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(missing))

    print(f"Inputs: {INPUTS}   Targets: {targets}   Seeds: {seeds}")
    print(f"Runs: {len(TRAIN_RUNS)} training, {len(VAL_RUNS)} validation, "
          f"{len(HELD_OUT_RUNS)} held out (not loaded)")


def check_same_grid(runs):
    if "z" not in INPUTS:
        return
    ref = runs[0]["xm"]
    for run in runs[1:]:
        if not np.allclose(run["xm"], ref, rtol=0.0, atol=1e-12):
            raise ValueError(f"{run['label']}: xm differs from {runs[0]['label']}")


def configure_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth error: {e}")
    tf.config.experimental.enable_tensor_float_32_execution(USE_TF32)
    print(f"GPUs visible: {len(gpus)}   TF32: "
          f"{tf.config.experimental.tensor_float_32_execution_enabled()}")


def _same(a, b):
    """Equality of JSON values; floats to round-off (the normalisation sums
    may differ in the last bits between calls)."""
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def write_or_check_target_files(target_dir, config, norm_arrays):
    """The first call for a target writes config.json and
    normalization_constants.npz. Later calls (other seeds) must have the same
    configuration apart from the start time; nothing is overwritten."""
    cfg_path = target_dir / "config.json"
    npz_path = target_dir / "normalization_constants.npz"
    new = json.loads(json.dumps(config))          # tuples -> lists, as on disk
    new.pop("started")
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        old.pop("started", None)
        differ = sorted(k for k in set(old) | set(new)
                        if k not in old or k not in new or not _same(old[k], new[k]))
        if differ:
            sys.exit(f"{cfg_path} was written with different settings ({differ}); "
                     f"refusing to mix them in one folder. Change VERSION.")
        return
    if npz_path.exists():
        sys.exit(f"{npz_path} exists without config.json; refusing to overwrite.")
    np.savez(npz_path, **norm_arrays)
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=4)


def software_versions():
    versions = {"python": platform.python_version(), "tensorflow": tf.__version__,
                "numpy": np.__version__, "h5py": h5py.__version__,
                "tf.keras module": tf.keras.__name__,
                "TF_USE_LEGACY_KERAS": os.environ.get("TF_USE_LEGACY_KERAS", "")}
    for package in ("tf_keras", "keras"):
        try:
            versions[package] = __import__(package).__version__
        except Exception:
            pass
    return versions


def best_index(val_loss):
    """Epoch index of the lowest finite validation loss (NaN-safe)."""
    val = np.asarray(val_loss, dtype=float)
    return int(np.nanargmin(val)) if np.isfinite(val).any() else len(val) - 1


def plot_losses(history, target, seed, y_std, yscale, path):
    """
    Training and validation loss vs epoch. Left: the loss as optimised (MSE
    of the z-scored target). Right: the same in un-normalised units. With one
    global z-score, MSE_unnormalised = y_std**2 * MSE_normalised exactly.
    """
    train_loss = np.asarray(history["loss"])
    val_loss = np.asarray(history["val_loss"])
    epochs = np.arange(1, len(train_loss) + 1)
    best = best_index(val_loss)
    tex = {"Q_d": r"$Q_d$", "Q_c": r"$Q_c$", "Q": r"$Q$"}[target]

    panels = (
        (1.0,         f"MSE of z-scored {tex}",       "Normalised (training loss)"),
        (y_std ** 2,  f"MSE of {tex} (un-normalised)", "Un-normalised units"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (factor, ylabel, title) in zip(axes, panels):
        ax.plot(epochs, factor * train_loss, label="Training", linewidth=2)
        ax.plot(epochs, factor * val_loss, label="Validation", linewidth=2)
        ax.axvline(epochs[best], color="0.5", linestyle=":", linewidth=1,
                   label=f"Best validation (epoch {epochs[best]})")
        ax.set_yscale(yscale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
    fig.suptitle(f"{VERSION}: 3x3 stencil CNN, {tex}, seed {seed}, "
                 f"inputs {', '.join(INPUTS)} ({yscale} scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 7. Training one model
# ============================================================

def train_one(target, seed, train_runs, val_runs, target_norm, pad_args):
    out = OUT_DIR / target / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=False)

    tf.keras.backend.clear_session()
    set_seeds(seed)

    model = build_stencil_cnn(pad_args, name=f"{VERSION}_stencil_{target}_seed{seed}",
                              seed=seed)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE,
                                           clipnorm=CLIPNORM),
        loss="mse",
    )
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "model.keras"), monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", **LR_PLATEAU, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    train_gen = RunSnapshots(train_runs, target, target_norm[target], BATCH_SIZE,
                             shuffle=True, seed=seed)
    val_gen = RunSnapshots(val_runs, target, target_norm[target], BATCH_SIZE,
                           shuffle=False, seed=seed)

    print(f"\n=== {target}, seed {seed}: {len(train_gen.index)} training and "
          f"{len(val_gen.index)} validation samples ===", flush=True)
    t0 = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                        callbacks=callbacks, verbose=2)   # one log line per epoch

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out / "history.json", "w") as f:
        json.dump(hist, f, indent=4)

    y_std = float(target_norm[target][1])
    plot_losses(hist, target, seed, y_std, "log", out / "loss_LOG.png")
    plot_losses(hist, target, seed, y_std, "linear", out / "loss_LinearScale.png")

    best = best_index(hist["val_loss"])
    return {
        "target": target, "seed": seed,
        "parameters": int(model.count_params()),
        "epochs_run": len(hist["loss"]),
        "best_epoch": best + 1,
        "best_val_mse_normalised": hist["val_loss"][best],
        "best_val_mse_unnormalised": hist["val_loss"][best] * y_std ** 2,
        "train_mse_normalised_at_best": hist["loss"][best],
        "seconds": time.time() - t0,
    }


# ============================================================
# 8. Main
# ============================================================

def main(argv=None):
    t_start = time.time()
    targets, seeds = parse_args(sys.argv[1:] if argv is None else argv)

    configure_gpu()
    if USE_MIXED_PRECISION:
        mixed_precision.set_global_policy("mixed_float16")
    check_config(targets, seeds)

    nt_common, frame_counts = common_frame_count(ALL_RUNS)

    print("\nLoading training runs:")
    train_runs = [load_run(label, nt_common) for label in TRAIN_RUNS]
    print("Loading validation runs:")
    val_runs = [load_run(label, nt_common) for label in VAL_RUNS]
    check_same_grid(train_runs + val_runs)

    input_norm = fit_input_norm(train_runs)
    target_norm = fit_target_norm(train_runs, targets)
    for run in train_runs + val_runs:
        apply_input_norm(run, input_norm)
    pad_args = padding_args(input_norm)

    for target in targets:
        target_dir = OUT_DIR / target
        target_dir.mkdir(parents=True, exist_ok=True)

        norm_arrays = {
            "inputs": np.array(INPUTS),
            "input_mean": np.array([input_norm[c][0] for c in INPUTS]),
            "input_std": np.array([input_norm[c][1] for c in INPUTS]),
            "target": np.array(target),
            "target_mean": np.array(target_norm[target][0]),
            "target_std": np.array(target_norm[target][1]),
        }
        config = {
            "version": VERSION,
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inputs": list(INPUTS),
            "target": target,
            "target_definition": {"Q_d": "Q_d_c.npz Q_d", "Q_c": "Q_d_c.npz Q_c",
                                  "Q": "Q_d + Q_c (right-hand side)"}[target],
            "nt_common": nt_common,
            "frame_counts": frame_counts,
            "train_runs": TRAIN_RUNS,
            "val_runs": VAL_RUNS,
            "held_out_runs": HELD_OUT_RUNS,
            "physical_parameters": {"alpha": ALPHA, "beta": BETA, "b_wall": B_WALL,
                                    "q_wall": Q_WALL, "z_wall": Z_WALL},
            "input_norm": {c: list(input_norm[c]) for c in INPUTS},
            "target_norm": list(target_norm[target]),
            "padding_normalised": pad_args,
            "architecture": {
                "receptive_field": "3x3",
                "stencil_filters": STENCIL_FILTERS,
                "pointwise_filters": list(POINTWISE_FILTERS),
                "activation": ACTIVATION,
                "padding": "periodic in y; z: ghost 2*wall - edge (Dirichlet channels), "
                           "edge value otherwise",
            },
            "loss": "mse (all grid points, z-scored target)",
            "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE,
                          "clipnorm": CLIPNORM},
            "lr_schedule": {"ReduceLROnPlateau": LR_PLATEAU,
                            "EarlyStopping_patience": EARLY_STOP_PATIENCE,
                            "monitor": "val_loss"},
            "grid": [NZ, NY],
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "mixed_precision": USE_MIXED_PRECISION,
            "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
            "software": software_versions(),
        }
        write_or_check_target_files(target_dir, config, norm_arrays)

        # One entry per completed seed, kept across calls (check_config refuses
        # seeds that already have one).
        summary_path = target_dir / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
        for seed in seeds:
            result = train_one(target, seed, train_runs, val_runs, target_norm, pad_args)
            summary = sorted(summary + [result], key=lambda s: s["seed"])
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=4)

        header = (f"{'target':<6} {'seed':>4} {'epochs':>6} {'best':>5} "
                  f"{'val MSE (norm)':>15} {'val MSE (units)':>16} "
                  f"{'train MSE (norm)':>17}")
        lines = [header, "-" * len(header)]
        for s in summary:
            lines.append(f"{s['target']:<6} {s['seed']:>4} {s['epochs_run']:>6} "
                         f"{s['best_epoch']:>5} {s['best_val_mse_normalised']:>15.4e} "
                         f"{s['best_val_mse_unnormalised']:>16.4e} "
                         f"{s['train_mse_normalised_at_best']:>17.4e}")
        table = "\n".join(lines)
        (target_dir / "summary.txt").write_text(table + "\n")
        print("\n" + table)

    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
