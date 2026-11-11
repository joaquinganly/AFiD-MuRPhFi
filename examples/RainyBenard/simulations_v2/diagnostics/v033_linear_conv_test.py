"""v033 linear test: does the conditioning of the 3 x 3 input patch stop gradient-based training
from learning the Laplacian? (agreed with the user, 2026-11-11; changelog.txt)

Data, padding, normalisation, loss and optimiser of Q_d_b (input b, target Q_d / kappa = Lap b),
two purely linear models, each a function of the 3 x 3 patch of the ghost-padded normalised b:
  raw:       y_n = sum_m K_m x_m + c                               (3 x 3 kernel K and bias c, trainable)
  whitened:  x~ = W (x - xbar), W = Lambda^(-1/2) V^T (fixed),  y_n = sum_k u_k x~_k + c   (u, c trainable)
x: the 9 normalised b values of the patch, order [di + 1, dj + 1]; y_n = (y - mu_y) / sigma_y.
C = V Lambda V^T: covariance of x over random interior cells of the 8 training runs' training blocks
(float64). Cov(x~) = I, so for a linear model every direction has the same curvature, whereas for x
the eigenvalues of C span 9 to 13 orders of magnitude (diagnostics/v033_diagnostics.py); decorrelating
inputs is the standard remedy (LeCun, Bottou, Orr & Mueller 1998, sec. 4.3). Both models represent
the same functions (any linear 3 x 3 stencil plus a constant); the whitened model's stencil on x is
K = W^T u. The exact operator is K* = (sigma_b / (dz^2 sigma_y)) L, L = [0 1 0; 1 -4 1; 0 1 0],
and with the ghost padding it is exact in the wall rows too.
Training as Q_d_b round 1 (Adam, lr 1e-3, clipnorm 1, batch 2, the same frames, per-run relative-MSE
weights) except: constant learning rate, no early stopping, EPOCHS epochs, seed 42, float64, CPU.
Recorded: training and validation loss per epoch (the mean over runs of the relative MSE, as the
v033 loss) and the stencil on b in physical units x dz^2, K sigma_y dz^2 / sigma_b (exact: L), before
training and after every epoch. Reference: the exact operator applied to the float32 b on the
validation blocks (the floor that float32 inputs set).
Output: simulations_v2/v033/linear_conv_test/ (refuses to overwrite): summary.txt, history.json,
        linear_conv_raw_vs_whitened.png.
Usage : python diagnostics/v033_linear_conv_test.py   (from simulations_v2/, inside the Docker container)
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
KINDS = ("raw", "whitened")


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


def build(kind, pad_args, W=None, xbar=None):
    init = tf.keras.initializers.GlorotUniform(seed=SEED)
    inp = layers.Input(shape=(T.NZ, T.NY, 1), dtype="float64", name="inputs")
    x = T.StencilPad(**pad_args, dtype="float64", name="stencil_pad")(inp)
    if kind == "raw":
        out = layers.Conv2D(1, 3, padding="valid", kernel_initializer=init, dtype="float64", name="stencil")(x)
    else:
        wl = layers.Conv2D(9, 3, padding="valid", trainable=False, dtype="float64", name="whiten")
        x = wl(x)
        wl.set_weights([W.T.reshape(3, 3, 1, 9), -W @ xbar])     # kernel[a, c, 0, k] = W[k, 3a + c]
        out = layers.Conv2D(1, 1, kernel_initializer=init, dtype="float64", name="readout")(x)
    return models.Model(inp, out, name=f"linear_{kind}")


def stencil(model, kind, W=None):
    """The model's 3 x 3 kernel on the normalised b, [di + 1, dj + 1]."""
    if kind == "raw":
        return model.get_layer("stencil").get_weights()[0][:, :, 0, 0].copy()
    u = model.get_layer("readout").get_weights()[0][0, 0, :, 0]
    return (W.T @ u).reshape(3, 3)


class StencilRecorder(tf.keras.callbacks.Callback):
    def __init__(self, kind, W, scale):
        super().__init__()
        self.kind, self.W, self.scale, self.stencils = kind, W, scale, []

    def on_train_begin(self, logs=None):
        self.stencils.append((stencil(self.model, self.kind, self.W) * self.scale).tolist())

    def on_epoch_end(self, epoch, logs=None):
        self.stencils.append((stencil(self.model, self.kind, self.W) * self.scale).tolist())


def floor_rel_mse(runs):
    """Mean over runs of the relative MSE of the exact operator on the float32 b (validation blocks)."""
    vals = []
    for run in runs:
        sl = run["parts"]["val"]
        p = DG.predict(run["fields"][sl, ..., 0], DG.exact_fit())
        y = run["y"][sl].astype(np.float64)
        vals.append(float(np.sum((p - y) ** 2) / np.sum(y ** 2)))
    return float(np.mean(vals))


def plot(hist, floor, path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ep = np.arange(1, EPOCHS + 1)
    for kind, col in zip(KINDS, ("tab:red", "tab:blue")):
        h = hist[kind]
        axes[0, 0].semilogy(ep, h["val_loss"], "o-", color=col, label=f"{kind}: validation")
        axes[0, 0].semilogy(ep, h["loss"], ":", color=col, label=f"{kind}: training")
        K = np.array(h["stencils"])                               # (EPOCHS + 1, 3, 3), x dz^2
        e0 = np.arange(0, EPOCHS + 1)
        axes[0, 1].plot(e0, K[:, 1, 1], "o-", color=col, label=f"{kind}: centre")
        axes[0, 1].plot(e0, K[:, [0, 2, 1, 1], [1, 1, 0, 2]].mean(axis=1), "s--", color=col,
                        label=f"{kind}: mean of the 4 neighbours")
    axes[0, 0].axhline(floor, color="k", linestyle="--", label="exact operator on float32 b")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel(r"mean over runs of $\sum (p-y)^2/\sum y^2$")
    axes[0, 0].set_title("(a) loss (relative MSE, as the v033 loss)", fontsize=10)
    axes[0, 1].axhline(-4.0, color="k", linestyle="--", linewidth=1, label="exact centre -4")
    axes[0, 1].axhline(1.0, color="k", linestyle=":", linewidth=1, label="exact neighbour 1")
    axes[0, 1].set_xlabel("epoch (0 = initial weights)")
    axes[0, 1].set_ylabel(r"stencil coefficient on $b$, $\times\,dz^2$")
    axes[0, 1].set_title("(b) learned stencil coefficients", fontsize=10)
    for ax in axes[0]:
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(fontsize=7)
    for ax, kind in zip(axes[1], KINDS):
        K = np.array(hist[kind]["stencils"][-1])
        ax.imshow(K, origin="lower", cmap="RdBu_r", vmin=-5, vmax=5)
        for a in range(3):
            for c in range(3):
                ax.text(c, a, f"{K[a, c]:.4f}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1, 2], ["y-1", "y", "y+1"])
        ax.set_yticks([0, 1, 2], ["z-1", "z", "z+1"])
        ax.set_title(f"({'c' if kind == 'raw' else 'd'}) {kind}: stencil after {EPOCHS} epochs "
                     f"(x dz^2; exact [0 1 0; 1 -4 1; 0 1 0])", fontsize=10)
    fig.suptitle(f"{T.VERSION}: linear 3x3 model of Lap b, Adam lr {LR:g} constant, raw b vs whitened 3x3 patch "
                 f"(x~ = Lambda^-1/2 V^T (x - xbar))", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    t_start = time.time()
    out = T.OUT_DIR / OUT_NAME
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; refusing to overwrite.")
    setup = EV.load_config("Q_d_b")                                # selects Q_d_b; its normalisation
    if setup is None:
        sys.exit("Q_d_b is not trained")
    input_norm, target_norm = setup["input_norm"], setup["target_norm"]
    del setup
    print(f"Devices: {[d.name for d in tf.config.list_physical_devices()]}", flush=True)
    runs = [T.load_run(label) for label in T.TRAIN_RUNS]
    floor = floor_rel_mse(runs)                                     # before normalising b
    for run in runs:
        T.apply_input_norm(run, input_norm)
    pad_args = T.padding_args(input_norm)
    weights = T.run_weights(runs, target_norm)
    sb, sy = input_norm["b"][1], target_norm[1]
    dz2 = (1.0 / T.NZ) ** 2
    scale = sy * dz2 / sb                                           # kernel on x -> coefficient on b, x dz^2

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
    for kind in KINDS:
        tf.keras.backend.clear_session()
        T.set_seeds(SEED)
        model = build(kind, pad_args, W, xbar)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=T.CLIPNORM), loss=wmse)
        rec = StencilRecorder(kind, W, scale)
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
        {"hist": hist, "floor_val_rel_mse": floor, "whitening_eigenvalues": lam.tolist(),
         "sigma_b": sb, "sigma_y": sy, "exact_kernel_scale": 1.0 / scale}, indent=2))
    L = [f"{T.VERSION} linear 3x3 test (Q_d_b data): raw b vs whitened patch; Adam lr {LR:g} constant, "
         f"{EPOCHS} epochs, batch {T.BATCH_SIZE}, seed {SEED}, float64",
         f"sigma_b {sb:.6g}, sigma_y {sy:.6g}: the exact kernel on the normalised x is "
         f"(sigma_b / (dz^2 sigma_y)) L = {1.0 / scale:.4g} L",
         "whitening covariance (pooled over the training runs) eigenvalues: " + " ".join(f"{v:.2e}" for v in lam),
         f"floor (exact operator on float32 b, validation blocks): mean relative MSE {floor:.3e}",
         "", "loss = mean over runs of the relative MSE sum (p - y)^2 / sum y^2",
         f"{'epoch':>5} {'raw train':>11} {'raw val':>11} {'whit train':>11} {'whit val':>11}"]
    for e in range(EPOCHS):
        L.append(f"{e + 1:>5} {hist['raw']['loss'][e]:>11.4e} {hist['raw']['val_loss'][e]:>11.4e} "
                 f"{hist['whitened']['loss'][e]:>11.4e} {hist['whitened']['val_loss'][e]:>11.4e}")
    for kind in KINDS:
        K = np.array(hist[kind]["stencils"][-1])
        L += ["", f"{kind}: stencil on b x dz^2 after {EPOCHS} epochs (rows z-1, z, z+1; exact [0 1 0; 1 -4 1; 0 1 0])"]
        L += ["   " + "  ".join(f"{K[a, c]:+10.5f}" for c in range(3)) for a in range(3)]
    elapsed = time.time() - t_start
    L.append(f"\nElapsed: {elapsed:.1f} s ({elapsed / 60:.1f} min)")
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text)
    print("\n" + text)
    plot(hist, floor, out / "linear_conv_raw_vs_whitened.png")
    return {"hist": hist, "floor": floor, "W": W, "xbar": xbar, "lam": lam, "scale": scale}


if __name__ == "__main__":
    main()
