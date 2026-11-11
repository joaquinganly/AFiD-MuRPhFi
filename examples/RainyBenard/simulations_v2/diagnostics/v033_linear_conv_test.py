"""v033 linear test: does the conditioning of the 3 x 3 input patch stop gradient-based training
from learning the Laplacian? (agreed with the user, 2026-11-11; changelog.txt)

Data, padding, normalisation, loss and optimiser of Q_d_b (input b, target Q_d / kappa = Lap b),
purely linear models, each a function of the 3 x 3 patch of the ghost-padded normalised b:
  raw:          y_n = sum_m K_m x_m + c                              (3 x 3 kernel K and bias c, trainable)
  whitened:     x~ = W (x - xbar), W = Lambda^(-1/2) V^T (fixed),  y_n = sum_k u_k x~_k + c   (u, c trainable)
  raw_rescaled: y_n = S (sum_m k~_m x_m + c~), S = sigma_b / (dz^2 sigma_y) fixed   (k~, c~ trainable)
x: the 9 normalised b values of the patch, order [di + 1, dj + 1]; y_n = (y - mu_y) / sigma_y.
C = V Lambda V^T: covariance of x over random interior cells of the 8 training runs' training blocks
(float64). Cov(x~) = I, so for a linear model every direction has the same curvature, whereas for x
the eigenvalues of C span 9 to 13 orders of magnitude (diagnostics/v033_diagnostics.py); decorrelating
inputs is the standard remedy (LeCun, Bottou, Orr & Mueller 1998, sec. 4.3). All three models
represent the same functions (any linear 3 x 3 stencil plus a constant): the whitened model's
stencil on x is K = W^T u, the rescaled one's K = S k~. The exact operator is K* = S L,
L = [0 1 0; 1 -4 1; 0 1 0], exact in the wall rows too with the ghost padding.
The raw and whitened models differ in two ways: the conditioning (Cov(x) vs I) and the size of the
exact weights (S L, S ~ 400, vs O(1)), and Adam moves each weight by about lr per update (Kingma &
Ba 2015, sec. 2.1). raw_rescaled is the control that changes only the second: its exact weights are
k~ = L, its conditioning is that of x; it starts from the same function as raw (k~ = K_raw / S).
Training as Q_d_b round 1 (Adam, lr 1e-3, clipnorm 1, batch 2, the same frames, per-run relative-MSE
weights) except: constant learning rate, no early stopping, EPOCHS epochs, seed 42, float64, CPU.
Recorded: training and validation loss per epoch (the mean over runs of the relative MSE, as the
v033 loss) and the stencil on b in physical units x dz^2, K sigma_y dz^2 / sigma_b (exact: L), before
training and after every epoch. Reference: the exact operator applied to the float32 b on the
validation blocks (the floor that float32 inputs set).
Output: simulations_v2/v033/linear_conv_test/ for the default models (raw, whitened), else
        simulations_v2/v033/linear_conv_test_<models>/, which also shows the default run's curves if
        it exists (refuses to overwrite): summary.txt, history.json, linear_conv_<models>.png.
Usage : python diagnostics/v033_linear_conv_test.py [model ...]   (from simulations_v2/, in Docker;
        models from raw, whitened, raw_rescaled; default raw whitened)
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")         # float64 on the CPU
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("v033_diagnostics", SIM / "diagnostics" / "v033_diagnostics.py")
DG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DG)
EV, T = DG.EV, DG.T
tf.keras.backend.set_floatx("float64")   # otherwise Keras casts float64 inputs to float32 before the model

OUT_NAME = "linear_conv_test"
EPOCHS = 20
SEED = 42
LR = 1e-3
WHITEN_CELLS = 250_000           # interior cells per training run for the whitening covariance
ALL_KINDS = ("raw", "whitened", "raw_rescaled")
DEFAULT_KINDS = ("raw", "whitened")
COLOURS = {"raw": "tab:red", "whitened": "tab:blue", "raw_rescaled": "tab:green"}


class Snapshots64(T.RunSnapshots):
    """The v033 generator with the inputs in float64."""

    def __getitem__(self, i):
        X, Y = super().__getitem__(i)
        return X.astype(np.float64), Y


def wmse(y_true, y_pred):
    """Per grid point: run weight * squared error (as the v033 loss), in the prediction's dtype."""
    y_true = tf.cast(y_true, y_pred.dtype)
    return y_true[..., 1] * tf.square(y_pred[..., 0] - y_true[..., 0])


def whitening(patches):
    """W = Lambda^(-1/2) V^T and the mean of the patches (n, 9); eigenvalues descending."""
    xbar = patches.mean(axis=0)
    lam, V = np.linalg.eigh(np.cov(patches, rowvar=False))
    lam, V = lam[::-1], V[:, ::-1]
    return (V / np.sqrt(lam)).T, xbar, lam


def build(kind, pad_args, W=None, xbar=None, kscale=1.0):
    init = tf.keras.initializers.GlorotUniform(seed=SEED)
    inp = layers.Input(shape=(T.NZ, T.NY, 1), dtype="float64", name="inputs")
    x = T.StencilPad(**pad_args, dtype="float64", name="stencil_pad")(inp)
    if kind == "raw":
        out = layers.Conv2D(1, 3, padding="valid", kernel_initializer=init, dtype="float64", name="stencil")(x)
    elif kind == "raw_rescaled":
        x = layers.Conv2D(1, 3, padding="valid", kernel_initializer=init, dtype="float64", name="stencil")(x)
        sl = layers.Conv2D(1, 1, use_bias=False, trainable=False, dtype="float64", name="rescale")
        out = sl(x)                         # fixed factor S in float64 (Keras' Rescaling layer computes in float32)
        sl.set_weights([np.full((1, 1, 1, 1), kscale)])
    else:
        wl = layers.Conv2D(9, 3, padding="valid", trainable=False, dtype="float64", name="whiten")
        x = wl(x)
        wl.set_weights([W.T.reshape(3, 3, 1, 9), -W @ xbar])     # kernel[a, c, 0, k] = W[k, 3a + c]
        out = layers.Conv2D(1, 1, kernel_initializer=init, dtype="float64", name="readout")(x)
    return models.Model(inp, out, name=f"linear_{kind}")


def build_initialised(kind, pad_args, W=None, xbar=None, kscale=1.0):
    """build(); raw_rescaled starts from the raw model's initial function (k~ = K_raw / S)."""
    model = build(kind, pad_args, W, xbar, kscale)
    if kind == "raw_rescaled":
        K0, c0 = build("raw", pad_args).get_layer("stencil").get_weights()
        model.get_layer("stencil").set_weights([K0 / kscale, c0 / kscale])
    return model


def stencil(model, kind, W=None, kscale=1.0):
    """The model's 3 x 3 kernel on the normalised b, [di + 1, dj + 1]."""
    if kind == "raw":
        return model.get_layer("stencil").get_weights()[0][:, :, 0, 0].copy()
    if kind == "raw_rescaled":
        return model.get_layer("stencil").get_weights()[0][:, :, 0, 0] * kscale
    u = model.get_layer("readout").get_weights()[0][0, 0, :, 0]
    return (W.T @ u).reshape(3, 3)


class StencilRecorder(tf.keras.callbacks.Callback):
    def __init__(self, kind, W, scale, kscale):
        super().__init__()
        self.kind, self.W, self.scale, self.kscale, self.stencils = kind, W, scale, kscale, []

    def on_train_begin(self, logs=None):
        self.stencils.append((stencil(self.model, self.kind, self.W, self.kscale) * self.scale).tolist())

    def on_epoch_end(self, epoch, logs=None):
        self.stencils.append((stencil(self.model, self.kind, self.W, self.kscale) * self.scale).tolist())


def floor_rel_mse(runs):
    """Mean over runs of the relative MSE of the exact operator on the float32 b (validation blocks)."""
    vals = []
    for run in runs:
        sl = run["parts"]["val"]
        p = DG.predict(run["fields"][sl, ..., 0], DG.exact_fit())
        y = run["y"][sl].astype(np.float64)
        vals.append(float(np.sum((p - y) ** 2) / np.sum(y ** 2)))
    return float(np.mean(vals))


def neighbours(K):
    return K[:, [0, 2, 1, 1], [1, 1, 0, 2]].mean(axis=1)


def plot(hist, ref, floor, kinds, path):
    allk = list(ref) + list(kinds)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, max(2, len(kinds)))
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for kind in allk:
        h, dashed = (hist[kind], False) if kind in hist else (ref[kind], True)
        col = COLOURS[kind]
        ep = np.arange(1, len(h["val_loss"]) + 1)
        lab = kind + (" (earlier run)" if dashed else "")
        ax0.semilogy(ep, h["val_loss"], "--" if dashed else "o-", color=col, label=f"{lab}: validation")
        K = np.array(h["stencils"])                               # (epochs + 1, 3, 3), x dz^2
        e0 = np.arange(0, len(K))
        ax1.plot(e0, K[:, 1, 1], "--" if dashed else "o-", color=col, label=f"{lab}: centre")
        ax1.plot(e0, neighbours(K), ":" if dashed else "s-", color=col, label=f"{lab}: mean of the 4 neighbours")
    ax0.axhline(floor, color="k", linestyle="--", linewidth=1, label="exact operator on float32 b")
    ax0.set_xlabel("epoch (968 updates each)")
    ax0.set_ylabel(r"mean over runs of $\sum (p-y)^2/\sum y^2$")
    ax0.set_title("(a) validation loss (relative MSE, as the v033 loss)", fontsize=10)
    ax1.axhline(-4.0, color="k", linestyle="--", linewidth=1, label="exact centre -4")
    ax1.axhline(1.0, color="k", linestyle=":", linewidth=1, label="exact neighbour 1")
    ax1.set_xlabel("epoch (0 = initial weights)")
    ax1.set_ylabel(r"stencil coefficient on $b$, $\times\,dz^2$")
    ax1.set_title("(b) learned stencil coefficients", fontsize=10)
    for ax in (ax0, ax1):
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    for j, kind in enumerate(kinds):
        ax = fig.add_subplot(gs[1, j])
        K = np.array(hist[kind]["stencils"][-1])
        ax.imshow(K, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        for a in range(3):
            for c in range(3):
                ax.text(c, a, f"{K[a, c]:.4f}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1, 2], ["y-1", "y", "y+1"])
        ax.set_yticks([0, 1, 2], ["z-1", "z", "z+1"])
        ax.set_title(f"{kind}: stencil x dz^2 after {EPOCHS} epochs "
                     f"(exact [0 1 0; 1 -4 1; 0 1 0])", fontsize=10)
    fig.suptitle(f"{T.VERSION}: linear 3x3 model of Lap b, Adam lr {LR:g} constant: "
                 + ", ".join(allk), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(kinds=None):
    t_start = time.time()
    kinds = tuple(kinds) if kinds else DEFAULT_KINDS
    bad = [k for k in kinds if k not in ALL_KINDS]
    if bad or len(set(kinds)) != len(kinds):
        sys.exit(f"Bad models {list(kinds)}; choose distinct models from {ALL_KINDS}")
    name = OUT_NAME if kinds == DEFAULT_KINDS else f"{OUT_NAME}_{'_'.join(kinds)}"
    out = T.OUT_DIR / name
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite.")
    ref_path = T.OUT_DIR / OUT_NAME / "history.json"
    ref = {}
    if name != OUT_NAME and ref_path.is_file():
        ref = {k: v for k, v in json.loads(ref_path.read_text())["hist"].items() if k not in kinds}
    setup = EV.load_config("Q_d_b")                                # selects Q_d_b; its normalisation
    if setup is None:
        sys.exit("Q_d_b is not trained")
    input_norm, target_norm = setup["input_norm"], setup["target_norm"]
    del setup
    print(f"Models {kinds}; devices: {[d.name for d in tf.config.list_physical_devices()]}", flush=True)
    runs = [T.load_run(label) for label in T.TRAIN_RUNS]
    floor = floor_rel_mse(runs)                                     # before normalising b
    for run in runs:
        T.apply_input_norm(run, input_norm)
    pad_args = T.padding_args(input_norm)
    weights = T.run_weights(runs, target_norm)
    sb, sy = input_norm["b"][1], target_norm[1]
    dz2 = (1.0 / T.NZ) ** 2
    scale = sy * dz2 / sb                                           # kernel on x -> coefficient on b, x dz^2
    kscale = 1.0 / scale                                            # S: the exact kernel on x is S L

    W = xbar = lam = None
    if "whitened" in kinds:
        rng = np.random.default_rng(SEED)
        patches = []
        for run in runs:
            bt = run["fields"][run["parts"]["train"], ..., 0]
            n = bt.shape[0]
            t = rng.integers(0, n, WHITEN_CELLS)
            i = rng.integers(1, T.NZ - 1, WHITEN_CELLS)
            j = rng.integers(0, T.NY, WHITEN_CELLS)
            patches.append(DG.gather(bt, t, i, j, DG.DIS["interior"]).astype(np.float64))
        W, xbar, lam = whitening(np.concatenate(patches))
        del patches

    hist = {}
    for kind in kinds:
        tf.keras.backend.clear_session()
        T.set_seeds(SEED)
        model = build_initialised(kind, pad_args, W, xbar, kscale)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=T.CLIPNORM), loss=wmse)
        rec = StencilRecorder(kind, W, scale, kscale)
        train_gen = Snapshots64(runs, "train", target_norm, T.BATCH_SIZE, shuffle=True, seed=SEED, weights=weights)
        val_gen = Snapshots64(runs, "val", target_norm, T.BATCH_SIZE, shuffle=False, seed=SEED, weights=weights)
        print(f"\n=== {kind}: {model.count_params()} parameters "
              f"({sum(int(np.prod(v.shape)) for v in model.trainable_weights)} trainable)", flush=True)
        h = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                      callbacks=[tf.keras.callbacks.TerminateOnNaN(), rec], verbose=2)
        hist[kind] = {"loss": [float(v) for v in h.history["loss"]],
                      "val_loss": [float(v) for v in h.history["val_loss"]], "stencils": rec.stencils}

    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(
        {"hist": hist, "floor_val_rel_mse": floor,
         "whitening_eigenvalues": None if lam is None else lam.tolist(),
         "sigma_b": sb, "sigma_y": sy, "exact_kernel_scale": kscale}, indent=2))
    allk = list(ref) + list(kinds)
    every = {**ref, **hist}
    L = [f"{T.VERSION} linear 3x3 test (Q_d_b data): models {', '.join(kinds)}; Adam lr {LR:g} constant, "
         f"{EPOCHS} epochs, batch {T.BATCH_SIZE}, seed {SEED}, float64",
         f"sigma_b {sb:.6g}, sigma_y {sy:.6g}: the exact kernel on the normalised x is "
         f"S L, S = sigma_b / (dz^2 sigma_y) = {kscale:.4g}",
         f"floor (exact operator on float32 b, validation blocks): mean relative MSE {floor:.3e}"]
    if lam is not None:
        L.append("whitening covariance (pooled over the training runs) eigenvalues: " + " ".join(f"{v:.2e}" for v in lam))
    if ref:
        L.append(f"earlier run ({ref_path.parent.name}): {', '.join(ref)}")
    L += ["", "loss = mean over runs of the relative MSE sum (p - y)^2 / sum y^2 (validation blocks)",
          f"{'epoch':>5} " + " ".join(f"{k + ' val':>18}" for k in allk)]
    for e in range(EPOCHS):
        L.append(f"{e + 1:>5} " + " ".join(f"{every[k]['val_loss'][e]:>18.4e}" if e < len(every[k]['val_loss'])
                                         else f"{'':>18}" for k in allk))
    for kind in kinds:
        K = np.array(hist[kind]["stencils"])
        L += ["", f"{kind}: stencil on b x dz^2 after {EPOCHS} epochs (rows z-1, z, z+1; exact [0 1 0; 1 -4 1; 0 1 0])"]
        L += ["   " + "  ".join(f"{K[-1][a, c]:+10.5f}" for c in range(3)) for a in range(3)]
        L.append("   centre per epoch (0 = initial): " + " ".join(f"{v:+.4f}" for v in K[:, 1, 1]))
    elapsed = time.time() - t_start
    L.append(f"\nElapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    plot(hist, ref, floor, kinds, out / f"linear_conv_{'_'.join(kinds)}.png")
    return {"hist": hist, "ref": ref, "floor": floor, "W": W, "xbar": xbar, "lam": lam, "scale": scale,
            "kscale": kscale, "out": out}


if __name__ == "__main__":
    main(sys.argv[1:])
