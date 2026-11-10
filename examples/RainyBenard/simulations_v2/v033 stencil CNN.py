# ============================================================
# v033 - 3x3 stencil CNN on the simulations_v2 dataset (calibration stage, round 1)
#
# Same architecture, padding, loss, optimiser, learning-rate schedule and seeds as v032
# ("simulations/v032 stencil CNN.py"). Changes (decisions of 2026-11-10; changelog.txt and the
# project handover doc claude/rainy_benard_v2_handover.md):
#   - data: the 8 simulations_v2 training runs (runs/Ra_<Ra>_Pr_1), grid 128 x 2048. Ra, Pr and
#     the grid are read from each run's bou.in, alpha and beta from its humid.in (postprocess.py
#     parsers), never from folder names;
#   - split by time block within every training run: training frames 25..750, validation frames
#     811..1000 (a 60-frame gap, 3 x the longest decorrelation time of the unsteady runs;
#     Roberts et al. 2017), every STRIDE-th frame. Frames 0..24 (drizzle state, no convection,
#     kinetic energy < 1e-4) are not used;
#   - the test runs (Ra 8.7e6, 1.5e8) are never loaded; Ra 2e8 is excluded (under-resolved top
#     boundary layer);
#   - configurations (CONFIGS): inputs and target of each model. Round 1:
#       Q_d_a   b, log10_Ra       -> Q_d
#       Q_d_b   b                 -> Q_d / kappa  (= Lap b; multiply the prediction by kappa
#                                    = (Ra Pr)^(-1/2) to compare with Q_d)
#       Q_c_C1  b, q, z           -> Q_c
#       Q_c_C2  b, q - q_s, z     -> Q_c
#       Q_c_C3  b, q              -> Q_c
#       Q_c_C4  b, q - q_s        -> Q_c
#     (log10_Pr is gone: Pr = 1 in every run.) Target Q = Q_d + Q_c is supported for later;
#   - loader: each array is reduced to the frames used right after it is read (peak memory: one
#     full float64 array of 2.1 GB), float32 afterwards; only the target of the configuration
#     is kept;
#   - run weights sigma^2 / mean(y^2) per run and block (training or validation frames), so a
#     run's weighted MSE is its relative MSE on those frames (as v032; NIST/SEMATECH
#     e-Handbook 4.1.4.3). Normalisation constants from the training frames only.
# Targets as stored in <run>/Q_d_c.npz by simulations_v2/postprocess.py (Vallis, Parker &
# Tobias 2019): Q_d = kappa Lap(b), Q_c = (gamma/tau)(q - q_s) H(q - q_s), q_s = exp(alpha (b - beta z)).
# Usage : python "v033 stencil CNN.py" <config> [seed ...]   (seeds from SEEDS; default all)
#         One configuration per call; calls for the same configuration share its config.json
#         (written by the first call, checked by later ones) and summary.json (merged by seed).
# Output: OUT_DIR/<config>/ config, normalisation constants, summary; OUT_DIR/<config>/seed_<s>/
#         best model, history (json + csv), loss plots.
# ============================================================

import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()
sys.path.insert(0, str(SCRIPT_DIR))
import postprocess as pp  # noqa: E402  (read_bou, read_humid, read_grid: the parsers that built the data)


# ============================================================
# 1. Settings
# ============================================================

VERSION = "v033"
SEEDS = (42, 43, 44)

CONFIGS = {
    "Q_d_a":  {"inputs": ("b", "log10_Ra"),         "target": "Q_d"},
    "Q_d_b":  {"inputs": ("b",),                    "target": "Q_d_over_kappa"},
    "Q_c_C1": {"inputs": ("b", "q", "z"),           "target": "Q_c"},
    "Q_c_C2": {"inputs": ("b", "q_minus_qs", "z"),  "target": "Q_c"},
    "Q_c_C3": {"inputs": ("b", "q"),                "target": "Q_c"},
    "Q_c_C4": {"inputs": ("b", "q_minus_qs"),       "target": "Q_c"},
}
TARGET_DEFINITIONS = {
    "Q_d": "Q_d_c.npz Q_d",
    "Q_c": "Q_d_c.npz Q_c",
    "Q": "Q_d + Q_c (right-hand side of the b equation)",
    "Q_d_over_kappa": "Q_d / kappa, kappa = (Ra Pr)^(-1/2) from bou.in (= Lap b)",
}
TARGET_KEYS = {"Q_d": ("Q_d",), "Q_c": ("Q_c",), "Q": ("Q_d", "Q_c"), "Q_d_over_kappa": ("Q_d",)}

NZ = 128
NY = 2048
TRAIN_FRAMES = (25, 750)      # first and last frame (inclusive) of the training block
VAL_FRAMES = (811, 1000)      # first and last frame (inclusive) of the validation block
STRIDE = 3                    # every STRIDE-th frame of each block
GAP_MIN = 60                  # frames required between the two blocks
B_NEG_TOL = 0.01              # load_run refuses min(b) < -B_NEG_TOL (minimum principle: b >= 0)
RUN_WEIGHTING = True          # weight sigma^2 / mean(y^2) per run and block; False: weight 1

# Physical parameters, identical in every run (checked against each run's humid.in). Wall values
# of b from SetTempBCs.F90 with moisture on; humidity walls from SetHumidityBCs (moisture.F90)
# with humid.in "top 2, bottom 1": saturated bottom q = exp(alpha * b_bottom), dry top q = 0.
ALPHA = 3.0
BETA = 2.0
B_WALL = (0.0, BETA - 1.0)                                   # (bottom, top)
Q_WALL = (math.exp(ALPHA * B_WALL[0]), 0.0)
QS_WALL = (math.exp(ALPHA * (B_WALL[0] - BETA * 0.0)),
           math.exp(ALPHA * (B_WALL[1] - BETA * 1.0)))
Z_WALL = (0.0, 1.0)

# Available input channels. "source": where the data come from; "wall": (bottom, top) wall
# value for the ghost-cell padding in z, or None to repeat the edge row (constant maps).
CHANNELS = {
    "b":          {"source": ("fields", "theta"), "wall": B_WALL},
    "q":          {"source": ("q", "q"),          "wall": Q_WALL},
    "q_minus_qs": {"source": ("fields", "rel_q"),
                   "wall": (Q_WALL[0] - QS_WALL[0], Q_WALL[1] - QS_WALL[1])},
    "z":          {"source": "grid",              "wall": Z_WALL},
    "log10_Ra":   {"source": "param",             "wall": None},
}

# Architecture (as v032): receptive field exactly 3x3.
STENCIL_FILTERS = 32
POINTWISE_FILTERS = (64, 64)
ACTIVATION = "relu"

BATCH_SIZE = 2
EPOCHS = 150
LEARNING_RATE = 1e-3          # suggested default step size (Kingma & Ba 2015, Alg. 1)
CLIPNORM = 1.0
LR_PLATEAU = {"factor": 0.5, "patience": 5, "min_lr": 1e-7}   # ReduceLROnPlateau
EARLY_STOP_PATIENCE = 12
USE_TF32 = False              # TensorFlow enables TF32 by default on Ampere and newer GPUs
                              # (docstring of tf.config.experimental.enable_tensor_float_32_execution)

RUNS_DIR = SCRIPT_DIR / "runs"
OUT_DIR = SCRIPT_DIR / VERSION

TRAIN_RUNS = ["Ra_1e6_Pr_1", "Ra_5e6_Pr_1", "Ra_7.5e6_Pr_1", "Ra_1e7_Pr_1",
              "Ra_1.75e7_Pr_1", "Ra_2.5e7_Pr_1", "Ra_5e7_Pr_1", "Ra_1e8_Pr_1"]
TEST_RUNS = ["Ra_8.7e6_Pr_1", "Ra_1.5e8_Pr_1"]               # never loaded here
EXCLUDED_RUNS = {"Ra_2e8_Pr_1": "under-resolved top boundary layer (5.5 cells, criterion 6.2), "
                                "q < 0 in 7 frames; dropped from the test set (2026-11-10)"}

INPUTS = CONFIGS["Q_c_C1"]["inputs"]      # set by select_config()
TARGET = CONFIGS["Q_c_C1"]["target"]
PARTS = ("train", "val")


def select_config(name):
    global INPUTS, TARGET
    INPUTS = tuple(CONFIGS[name]["inputs"])
    TARGET = CONFIGS[name]["target"]


def frame_indices(block):
    first, last = block
    return np.arange(first, last + 1, STRIDE)


# ============================================================
# 2. Data loading
# ============================================================

def run_paths(label):
    d = RUNS_DIR / label
    return {
        "fields": d / f"{label}_fields_Q.npz",
        "q": d / f"{label}_q.npz",
        "qdc": d / "Q_d_c.npz",
        "grid": d / "outputdir" / "cordin_info.h5",
        "bou": d / "bou.in",
        "humid": d / "humid.in",
        "setup": d / "setup.txt",
        "postprocess": d / "postprocess.txt",
    }


def field_channels():
    """Input channels read from .npz files, in INPUTS order."""
    return [c for c in INPUTS if isinstance(CHANNELS[c]["source"], tuple)]


def needed_kinds():
    return sorted({CHANNELS[c]["source"][0] for c in field_channels()} | {"qdc"})


def _check_shape(label, name, arr, nt_needed):
    if arr.ndim != 3 or arr.shape[:2] != (NZ, NY) or arr.shape[2] < nt_needed:
        raise ValueError(f"{label}: '{name}' has shape {arr.shape}, expected ({NZ}, {NY}, >= {nt_needed})")


def load_run(label):
    """
    Training and validation frames of one run (TRAIN_FRAMES, VAL_FRAMES, STRIDE). Returns
      fields : (n, NZ, NY, n_field_channels) float32, raw, INPUTS order
      y      : (n, NZ, NY) float32, the target of the configuration
      parts  : {"train": slice, "val": slice} into the first axis
      frames : {"train": DNS frame numbers, "val": ...}
      xm     : (NZ,) cell-centre heights
    Sample i of a block is DNS frame frames[block][i] in every array (all arrays of the
    post-processing are (nz, ny, nt) with frame n = movie frame n).
    """
    paths = run_paths(label)
    p = pp.read_bou(paths["bou"])
    h = pp.read_humid(paths["humid"])
    if (p["nz"], p["ny"]) != (NZ, NY):
        raise ValueError(f"{label}: bou.in grid {p['nz']} x {p['ny']}, expected {NZ} x {NY}")
    if not (math.isclose(h["alpha"], ALPHA) and math.isclose(h["beta"], BETA)):
        raise ValueError(f"{label}: humid.in alpha {h['alpha']}, beta {h['beta']}; the wall values "
                         f"assume alpha {ALPHA}, beta {BETA}")
    zc, _, _ = pp.read_grid(paths["grid"], p["nz"], p["ny"], p["ly"])
    idx = {"train": frame_indices(TRAIN_FRAMES), "val": frame_indices(VAL_FRAMES)}
    used = np.concatenate([idx["train"], idx["val"]])
    nt_needed = int(used.max()) + 1
    n = used.size

    chans = field_channels()
    fields = np.empty((n, NZ, NY, len(chans)), dtype=np.float32)
    for i, c in enumerate(chans):
        kind, key = CHANNELS[c]["source"]
        with np.load(paths[kind]) as data:
            arr = data[key]                                   # (nz, ny, nt) float64, read in full
        _check_shape(label, key, arr, nt_needed)
        fields[..., i] = np.moveaxis(arr[:, :, used], -1, 0)
        del arr

    raw = {}
    with np.load(paths["qdc"]) as data:
        for key in TARGET_KEYS[TARGET]:
            arr = data[key]
            _check_shape(label, key, arr, nt_needed)
            raw[key] = arr[:, :, used]                         # float64, frames used only
            del arr
    kappa = 1.0 / math.sqrt(p["ra"] * p["pr"])
    if TARGET == "Q_d":
        y = raw["Q_d"]
    elif TARGET == "Q_c":
        y = raw["Q_c"]
    elif TARGET == "Q":
        y = raw["Q_d"] + raw["Q_c"]
    elif TARGET == "Q_d_over_kappa":
        y = raw["Q_d"] / kappa
    else:
        raise ValueError(f"Unknown target {TARGET}")
    y = np.ascontiguousarray(np.moveaxis(y, -1, 0), dtype=np.float32)
    del raw

    if "b" in chans:
        b_min = float(fields[..., chans.index("b")].min())
        if b_min < -B_NEG_TOL:
            raise ValueError(f"{label}: min(b) = {b_min:.4f} < -{B_NEG_TOL}; b >= 0 by the minimum "
                             f"principle (Evans 2010, 7.1.4)")
    ntr = idx["train"].size
    print(f"  {label:<15s} Ra = {p['ra']:.4g}  Pr = {p['pr']:g}  kappa = {kappa:.4e}  "
          f"training frames {ntr}  validation frames {idx['val'].size}", flush=True)
    return {
        "label": label,
        "fields": fields,
        "y": y,
        "xm": zc[:, 0].astype(np.float64),
        "params": {"log10_Ra": float(np.log10(p["ra"]))},
        "ra": p["ra"], "pr": p["pr"], "kappa": kappa,
        "parts": {"train": slice(0, ntr), "val": slice(ntr, n)},
        "frames": {k: v.tolist() for k, v in idx.items()},
    }


def part_len(run, part):
    s = run["parts"][part]
    return s.stop - s.start


# ============================================================
# 3. Normalisation (z-score, training frames only) and run weights
# ============================================================

def _mean_std(total, total_sq, count):
    mean = total / count
    std = math.sqrt(max(total_sq / count - mean ** 2, 0.0))
    return mean, (std if std > 0.0 else 1.0)


def fit_input_norm(runs):
    """Per input channel (mean, std) over all grid points of the training frames; the
    parameter maps and z are weighted by the number of training frames. float64 sums."""
    norm = {}
    chans = field_channels()
    for c in INPUTS:
        s = sq = n = 0.0
        for run in runs:
            ntr = part_len(run, "train")
            if c in chans:
                x = run["fields"][run["parts"]["train"], ..., chans.index(c)].astype(np.float64).ravel()
                s += x.sum(); sq += x @ x; n += x.size
            elif c == "z":
                x = run["xm"]
                w = ntr * NY
                s += w * x.sum(); sq += w * (x @ x); n += w * x.size
            else:
                v = run["params"][c]
                w = ntr * NZ * NY
                s += w * v; sq += w * v * v; n += w
        norm[c] = _mean_std(s, sq, n)
    return norm


def fit_target_norm(runs):
    s = sq = n = 0.0
    for run in runs:
        y = run["y"][run["parts"]["train"]].astype(np.float64).ravel()
        s += y.sum(); sq += y @ y; n += y.size
    return _mean_std(s, sq, n)


def run_weights(runs, target_norm):
    """Loss weight of each run and block: sigma^2 / mean(y^2) over the block's frames, with
    sigma the target std of the z-score. The weighted normalised MSE of a run's block is then
    its relative MSE, and every run counts equally (inverse-variance weighting, NIST/SEMATECH
    e-Handbook 4.1.4.3). Keys "<run>/<block>". 1 if not RUN_WEIGHTING."""
    std = target_norm[1]
    weights = {}
    for run in runs:
        for part in PARTS:
            key = f"{run['label']}/{part}"
            if not RUN_WEIGHTING:
                weights[key] = 1.0
                continue
            ms = float(np.mean(np.square(run["y"][run["parts"][part]], dtype=np.float64)))
            if ms == 0.0:
                raise ValueError(f"{key}: {TARGET} is zero in every frame used; relative MSE undefined")
            weights[key] = std ** 2 / ms
    return weights


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
    Samples of one block (part = "train" or "val") of every run. Yields (X, Y):
    X (B, NZ, NY, len(INPUTS)) in INPUTS order, Y (B, NZ, NY, 2) = [z-scored target, weight].
    The weight travels as a second target channel (read by weighted_mse) rather than as Keras
    sample_weight (CLAUDE.md, section 4). z and the parameter maps are built per batch.
    """

    def __init__(self, runs, part, target_norm, batch_size, shuffle, seed, weights=None):
        super().__init__()
        self.runs = runs
        self.mean = np.float32(target_norm[0])
        self.std = np.float32(target_norm[1])
        self.weights = [np.float32(1.0 if weights is None else weights[f"{run['label']}/{part}"])
                        for run in runs]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.index = np.array(
            [(r, t) for r, run in enumerate(runs)
             for t in range(run["parts"][part].start, run["parts"][part].stop)],
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
        Y = np.empty((len(rows), NZ, NY, 2), dtype=np.float32)
        for k, (r, t) in enumerate(rows):
            run = self.runs[r]
            for ch, (c, idx) in enumerate(self.plan):
                if idx is not None:
                    X[k, ..., ch] = run["fields"][t, ..., idx]
                elif c == "z":
                    X[k, ..., ch] = run["z_norm"][:, None]
                else:
                    X[k, ..., ch] = run["params_norm"][c]
            Y[k, ..., 0] = (run["y"][t] - self.mean) / self.std
            Y[k, ..., 1] = self.weights[r]
        return X, Y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.index)


# ============================================================
# 5. Model and loss (as v032)
# ============================================================

@tf.keras.utils.register_keras_serializable(package="v033")
class StencilPad(layers.Layer):
    """
    Pads (z, y) by one cell on each side for a 'valid' 3x3 convolution.
    y (axis 2): periodic. z (axis 1, row 0 = bottom wall): for channels with
    dirichlet = 1, ghost = 2 * wall - edge (cell-centred Dirichlet ghost point, as in the
    Laplacian of postprocess.py); otherwise the edge row is repeated. Wall values are in
    normalised units; the ghost formula is unchanged by the affine z-score.
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


@tf.keras.utils.register_keras_serializable(package="v033")
def weighted_mse(y_true, y_pred):
    """Per grid point: run weight * squared error. y_true = [target, weight]."""
    return y_true[..., 1] * tf.square(y_pred[..., 0] - y_true[..., 0])


@tf.keras.utils.register_keras_serializable(package="v033")
def mse_target(y_true, y_pred):
    """Unweighted squared error of the z-scored target (logged metric)."""
    return tf.square(y_pred[..., 0] - y_true[..., 0])


def set_seeds(seed):
    """Seeds Python, NumPy and TensorFlow (tf.keras.utils.set_random_seed is not used: with
    tf_keras 2.17 on Python 3.12 it raises TypeError for unseeded initialisers; see v032)."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_stencil_cnn(pad_args, name, seed):
    """Each layer gets an explicitly seeded initialiser, so the initial weights depend only on
    `seed` and the number of inputs."""
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
    """Configuration name (first argument) and seeds (the rest; all SEEDS if none)."""
    if not argv or argv[0] not in CONFIGS:
        raise ValueError(f"Bad arguments {list(argv)}; first give a configuration from "
                         f"{list(CONFIGS)}, then optional seeds from {SEEDS}")
    try:
        seeds = tuple(int(a) for a in argv[1:])
    except ValueError:
        raise ValueError(f"Bad seeds {list(argv[1:])}") from None
    seeds = seeds or SEEDS
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        raise ValueError(f"Bad seeds {list(seeds)}; choose distinct seeds from {SEEDS}")
    return argv[0], seeds


def check_config(name, seeds):
    unknown = [c for c in INPUTS if c not in CHANNELS]
    if unknown or len(set(INPUTS)) != len(INPUTS) or TARGET not in TARGET_KEYS:
        raise ValueError(f"Bad configuration {name}: inputs {INPUTS}, target {TARGET}")
    tr, va = frame_indices(TRAIN_FRAMES), frame_indices(VAL_FRAMES)
    if tr.size == 0 or va.size == 0 or VAL_FRAMES[0] - TRAIN_FRAMES[1] - 1 < GAP_MIN:
        raise ValueError(f"Blocks {TRAIN_FRAMES}, {VAL_FRAMES}: need non-empty blocks and a gap "
                         f">= {GAP_MIN} frames")
    groups = {"TRAIN_RUNS": TRAIN_RUNS, "TEST_RUNS": TEST_RUNS, "EXCLUDED_RUNS": list(EXCLUDED_RUNS)}
    for gname, runs in groups.items():
        if len(set(runs)) != len(runs):
            raise ValueError(f"{gname} contains duplicates")
    if (set(TRAIN_RUNS) & set(TEST_RUNS)) or (set(TRAIN_RUNS) & set(EXCLUDED_RUNS)) \
            or (set(TEST_RUNS) & set(EXCLUDED_RUNS)):
        raise ValueError("TRAIN_RUNS, TEST_RUNS and EXCLUDED_RUNS overlap")
    missing = []
    for label in TRAIN_RUNS:
        paths = run_paths(label)
        for kind in needed_kinds() + ["grid", "bou", "humid", "setup", "postprocess"]:
            if not paths[kind].is_file():
                missing.append(str(paths[kind]))
        if paths["setup"].is_file() and "set: test" in paths["setup"].read_text():
            sys.exit(f"{label}: setup.txt says 'set: test'; test runs must never be used for training")
    if missing:
        raise FileNotFoundError("Missing input files:\n  " + "\n  ".join(missing))
    summary_path = OUT_DIR / name / "summary.json"
    done = ({s["seed"] for s in json.loads(summary_path.read_text())}
            if summary_path.exists() else set())
    for seed in seeds:
        out = OUT_DIR / name / f"seed_{seed}"
        if out.exists():
            sys.exit(f"{out} exists; refusing to overwrite. Move it aside or change VERSION.")
        if seed in done:
            sys.exit(f"{summary_path} already has seed {seed}; refusing to replace it. Change VERSION.")
    print(f"Configuration {name}: inputs {INPUTS}, target {TARGET}, seeds {seeds}")
    print(f"Runs: {len(TRAIN_RUNS)} (training frames {TRAIN_FRAMES[0]}..{TRAIN_FRAMES[1]}, validation "
          f"frames {VAL_FRAMES[0]}..{VAL_FRAMES[1]}, stride {STRIDE}); test runs {TEST_RUNS} and "
          f"excluded {list(EXCLUDED_RUNS)} not loaded")


def check_same_grid(runs):
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
    """Equality of JSON values; floats to round-off."""
    if isinstance(a, float) or isinstance(b, float):
        return (isinstance(a, (int, float)) and isinstance(b, (int, float))
                and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def write_or_check_config_files(cfg_dir, config, norm_arrays):
    """The first call for a configuration writes config.json and normalization_constants.npz.
    Later calls (other seeds) must have the same settings apart from the start time."""
    cfg_path = cfg_dir / "config.json"
    npz_path = cfg_dir / "normalization_constants.npz"
    new = json.loads(json.dumps(config))
    new.pop("started")
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text())
        old.pop("started", None)
        differ = sorted(k for k in set(old) | set(new)
                        if k not in old or k not in new or not _same(old[k], new[k]))
        if differ:
            sys.exit(f"{cfg_path} was written with different settings ({differ}); refusing to mix "
                     f"them in one folder. Change VERSION.")
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
    val = np.asarray(val_loss, dtype=float)
    return int(np.nanargmin(val)) if np.isfinite(val).any() else len(val) - 1


TEX = {"Q_d": r"$Q_d$", "Q_c": r"$Q_c$", "Q": r"$Q$", "Q_d_over_kappa": r"$Q_d/\kappa$"}


def plot_losses(history, name, seed, y_std, yscale, path):
    """Training and validation curves vs epoch. Left: the loss as optimised (mean over runs of
    the relative MSE when RUN_WEIGHTING). Right: the unweighted MSE in un-normalised units
    (y_std**2 * MSE of the z-scored target)."""
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = best_index(history["val_loss"])
    tex = TEX[TARGET]
    left = "Mean relative MSE over runs" if RUN_WEIGHTING else f"MSE of z-scored {tex}"
    panels = (("loss", 1.0, left, "Loss as optimised"),
              ("mse_target", y_std ** 2, f"MSE of {tex} (un-normalised, unweighted)", "Un-normalised units"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (key, factor, ylabel, title) in zip(axes, panels):
        ax.plot(epochs, factor * np.asarray(history[key]), label="Training", linewidth=2)
        ax.plot(epochs, factor * np.asarray(history[f"val_{key}"]), label="Validation", linewidth=2)
        ax.axvline(epochs[best], color="0.5", linestyle=":", linewidth=1,
                   label=f"Best validation loss (epoch {epochs[best]})")
        ax.set_yscale(yscale)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend()
    fig.suptitle(f"{VERSION}: 3x3 stencil CNN, {name} ({tex}), seed {seed}, inputs "
                 f"{', '.join(INPUTS)} ({yscale} scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 7. Training one model
# ============================================================

def train_one(name, seed, runs, target_norm, pad_args, weights):
    out = OUT_DIR / name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=False)

    tf.keras.backend.clear_session()
    set_seeds(seed)

    model = build_stencil_cnn(pad_args, name=f"{VERSION}_stencil_{name}_seed{seed}", seed=seed)
    model.summary()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=CLIPNORM),
        loss=weighted_mse,
        metrics=[mse_target],
    )
    callbacks = [
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(
            str(out / "model.keras"), monitor="val_loss", mode="min", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", **LR_PLATEAU, verbose=1),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=EARLY_STOP_PATIENCE, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out / "history.csv")),
    ]
    train_gen = RunSnapshots(runs, "train", target_norm, BATCH_SIZE, shuffle=True, seed=seed,
                             weights=weights)
    val_gen = RunSnapshots(runs, "val", target_norm, BATCH_SIZE, shuffle=False, seed=seed,
                           weights=weights)
    print(f"\n=== {name}, seed {seed}: {len(train_gen.index)} training and "
          f"{len(val_gen.index)} validation samples ===", flush=True)
    t0 = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                        callbacks=callbacks, verbose=2)

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(out / "history.json", "w") as f:
        json.dump(hist, f, indent=4)
    y_std = float(target_norm[1])
    plot_losses(hist, name, seed, y_std, "log", out / "loss_LOG.png")
    plot_losses(hist, name, seed, y_std, "linear", out / "loss_LinearScale.png")
    best = best_index(hist["val_loss"])
    return {
        "config": name, "target": TARGET, "seed": seed,
        "parameters": int(model.count_params()),
        "epochs_run": len(hist["loss"]),
        "best_epoch": best + 1,
        "best_val_loss": hist["val_loss"][best],
        "train_loss_at_best": hist["loss"][best],
        "best_val_mse_unnormalised": hist["val_mse_target"][best] * y_std ** 2,
        "seconds": time.time() - t0,
    }


# ============================================================
# 8. Main
# ============================================================

def main(argv=None):
    t_start = time.time()
    name, seeds = parse_args(sys.argv[1:] if argv is None else argv)
    select_config(name)
    configure_gpu()
    check_config(name, seeds)

    print("\nLoading training runs (training and validation blocks):", flush=True)
    runs = [load_run(label) for label in TRAIN_RUNS]
    check_same_grid(runs)
    input_norm = fit_input_norm(runs)
    target_norm = fit_target_norm(runs)
    for run in runs:
        apply_input_norm(run, input_norm)
    pad_args = padding_args(input_norm)
    weights = run_weights(runs, target_norm)

    cfg_dir = OUT_DIR / name
    cfg_dir.mkdir(parents=True, exist_ok=True)
    norm_arrays = {
        "inputs": np.array(INPUTS),
        "input_mean": np.array([input_norm[c][0] for c in INPUTS]),
        "input_std": np.array([input_norm[c][1] for c in INPUTS]),
        "target": np.array(TARGET),
        "target_mean": np.array(target_norm[0]),
        "target_std": np.array(target_norm[1]),
    }
    config = {
        "version": VERSION,
        "config": name,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inputs": list(INPUTS),
        "target": TARGET,
        "target_definition": TARGET_DEFINITIONS[TARGET],
        "train_runs": TRAIN_RUNS,
        "test_runs_not_loaded": TEST_RUNS,
        "excluded_runs": EXCLUDED_RUNS,
        "blocks": {"train_frames": list(TRAIN_FRAMES), "val_frames": list(VAL_FRAMES),
                   "stride": STRIDE, "gap_min": GAP_MIN},
        "frames_used": {run["label"]: run["frames"] for run in runs},
        "run_parameters": {run["label"]: {"Ra": run["ra"], "Pr": run["pr"], "kappa": run["kappa"]}
                           for run in runs},
        "b_min_allowed": -B_NEG_TOL,
        "physical_parameters": {"alpha": ALPHA, "beta": BETA, "b_wall": B_WALL,
                                "q_wall": Q_WALL, "z_wall": Z_WALL},
        "input_norm": {c: list(input_norm[c]) for c in INPUTS},
        "target_norm": list(target_norm),
        "run_weighting": ("sigma^2 / mean(y^2) per run and block (loss = mean relative MSE)"
                          if RUN_WEIGHTING else "none"),
        "run_weights": weights,
        "padding_normalised": pad_args,
        "architecture": {
            "receptive_field": "3x3",
            "stencil_filters": STENCIL_FILTERS,
            "pointwise_filters": list(POINTWISE_FILTERS),
            "activation": ACTIVATION,
            "padding": "periodic in y; z: ghost 2*wall - edge (Dirichlet channels), edge value otherwise",
        },
        "loss": "run weight * squared error of the z-scored target, mean over grid points and "
                "samples; metric mse_target unweighted",
        "optimizer": {"name": "Adam", "learning_rate": LEARNING_RATE, "clipnorm": CLIPNORM},
        "lr_schedule": {"ReduceLROnPlateau": LR_PLATEAU,
                        "EarlyStopping_patience": EARLY_STOP_PATIENCE, "monitor": "val_loss"},
        "grid": [NZ, NY],
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "tf32": tf.config.experimental.tensor_float_32_execution_enabled(),
        "software": software_versions(),
    }
    write_or_check_config_files(cfg_dir, config, norm_arrays)

    summary_path = cfg_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else []
    for seed in seeds:
        result = train_one(name, seed, runs, target_norm, pad_args, weights)
        summary = sorted(summary + [result], key=lambda s: s["seed"])
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

    header = (f"{'config':<7} {'seed':>4} {'epochs':>6} {'best':>5} {'val loss':>11} "
              f"{'val MSE (units)':>16} {'train loss':>11}")
    lines = [header, "-" * len(header)]
    for s in summary:
        lines.append(f"{s['config']:<7} {s['seed']:>4} {s['epochs_run']:>6} {s['best_epoch']:>5} "
                     f"{s['best_val_loss']:>11.4e} {s['best_val_mse_unnormalised']:>16.4e} "
                     f"{s['train_loss_at_best']:>11.4e}")
    lines.append(f"loss = {'mean relative MSE over runs' if RUN_WEIGHTING else 'MSE'} (normalised); "
                 f"val MSE (units) unweighted, target {TARGET}")
    table = "\n".join(lines)
    (cfg_dir / "summary.txt").write_text(table + "\n")
    print("\n" + table)
    elapsed = time.time() - t_start
    print(f"\nElapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
