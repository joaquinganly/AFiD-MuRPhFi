# COPY of '../Q staggered grid efficient pathing.py' for flat-layout runs (fields/ and cordin_info.h5 directly in the run folder).
# Only path/loading logic changed (lines marked 'PATH CHANGE'); numerics untouched.
# Defino lo que voy a necesitar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import afidtools as afid
import os
from tqdm import tqdm
import re
from pathlib import Path

matplotlib.rcParams.update(matplotlib.rcParamsDefault)


# ------------------------------------------------------------
# Grid and simulation parameters
# ------------------------------------------------------------

nz = 64
ny = 1024
nt = 324

Lz = 1.0
Ly = 16.0

dz = Lz / nz
dy = Ly / ny
dt = 1

z = np.linspace(0, Lz, nz, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
t = np.arange(nt) * dt


# ------------------------------------------------------------
# Folder where your .h5 files are stored
# ------------------------------------------------------------
base_dir = Path(
    "/workspace/examples/RainyBenard/simulations"
)

simulation_folders = sorted(
    [
        f for f in base_dir.iterdir()
        if f.is_dir()
        and (f.name.startswith("Ra_")
             or f.name.startswith("Pr_"))
    ]
)
# PATH CHANGE: restrict to the new flat-layout runs only (never re-process existing folders)
simulation_folders = [base_dir / name for name in ["Ra_5e6", "Ra_75e6", "Ra_5e7", "Ra_25e7", "Ra_175e7"]]
print("\nFound simulations:")
for folder in simulation_folders:
    print("   ", folder.name)

# ------------------------------------------------------------
# Grid information
# ------------------------------------------------------------

def process_simulation(fld):

    print("\n" + "=" * 70)
    print(f"Processing {fld.name}")
    print("=" * 70)

    data_folder = fld / "fields"  # PATH CHANGE: flat layout (no outputdir/)

    # PATH CHANGE: afid.Grid hard-codes fld/outputdir/cordin_info.h5; flat layout has it in fld/.
    # Reads the same 'xm' dataset that afid.Grid reads.
    from types import SimpleNamespace
    with h5py.File(fld / "cordin_info.h5", "r") as _f:
        grid = SimpleNamespace(xm=_f["xm"][()])

    # paste EVERYTHING ELSE here
    # grid.xm is the wall-normal/vertical coordinate in AFiD (cell centers,
    # same grid theta/q live on)
    x_vertical = np.asarray(grid.xm).squeeze()

    if x_vertical.size == nz + 1:
        x_vertical = x_vertical[:-1]

    if x_vertical.size != nz:
        raise ValueError(
            f"Unexpected grid.xm size: {x_vertical.size}, expected {nz}"
        )

    x_vertical = x_vertical[:, np.newaxis]


    # ------------------------------------------------------------
    # Reading functions
    #
    # CHANGE: split into two readers instead of one, because u=vz is on a
    # *different* z-grid than theta/q/v.
    #
    #   - theta, q, v (vy) all live at cell CENTERS in z: nz points.
    #   - u (vz) is staggered: it lives at cell EDGES in z, including both
    #     walls exactly: nz+1 points. Previously this was truncated to nz
    #     points by dropping the last row, which silently discarded the
    #     real top-wall edge value and replaced it with a wrong-location
    #     interior value (later overwritten by hand to 0, compounding the
    #     error). Now it's kept at its native nz+1 resolution so the
    #     boundary is handled correctly, without needing ghost points.
    # ------------------------------------------------------------

    def read_h5_center_field(filename, nz, ny):
        """
        Reads a cell-centered AFiD field (theta, q): expects nz points in z.
        """
        with h5py.File(filename, "r") as f:
            field = np.array(f["var"])

        field = np.squeeze(field)

        if field.shape[0] == ny:
            field = field.T

        if field.shape[0] == nz + 1:
            field = field[:-1, :]

        if field.shape != (nz, ny):
            raise ValueError(
                f"Unexpected shape in {filename}: got {field.shape}, expected {(nz, ny)}"
            )

        return field


    def read_h5_edge_field_z(filename, nz, ny):
        """
        Reads vz, which is staggered on z-edges: expects nz+1 points in z,
        including both walls (edge 0 = bottom wall, edge nz = top wall).
        Not truncated -- both wall edges are kept.
        """
        with h5py.File(filename, "r") as f:
            field = np.array(f["var"])

        field = np.squeeze(field)

        if field.shape[0] == ny:
            field = field.T

        if field.shape != (nz + 1, ny):
            raise ValueError(
                f"Unexpected shape in {filename}: got {field.shape}, "
                f"expected staggered z-edge shape {(nz + 1, ny)}"
            )

        return field


    # ------------------------------------------------------------
    # Allocate arrays
    #
    # CHANGE: u_edge now has nz+1 points in z (native staggered grid).
    # A separate u_center (nz points, interpolated) is kept for anything
    # that needs to live on the same grid as theta (plots, saved .npz,
    # ML input channel).
    # ------------------------------------------------------------

    theta = np.zeros([nz, ny, nt])
    u_edge = np.zeros([nz + 1, ny, nt])  # vertical velocity, vz, on z-edges
    v = np.zeros([nz, ny, nt])           # horizontal velocity, vy, on z-centers
    q = np.zeros([nz, ny, nt])
    qs = np.zeros([nz, ny, nt])
    rel_q = np.zeros([nz, ny, nt])
    Q = np.zeros_like(theta)


    # ------------------------------------------------------------
    # Read data
    # ------------------------------------------------------------

    '''CAREFUL HERE, notice that vz reads vx, because I believe in AFid code, x==z, CHECK THIS '''

    for n in tqdm(range(nt), desc="Reading h5 data"):

        fname = f"{n:05d}"

        theta_file = os.path.join(data_folder, fname + "_temp.h5")
        vy_file    = os.path.join(data_folder, fname + "_vy.h5")
        vz_file    = os.path.join(data_folder, fname + "_vx.h5")
        q_file     = os.path.join(data_folder, fname + "_humid.h")

        theta[:, :, n]  = read_h5_center_field(theta_file, nz, ny)
        v[:, :, n]      = read_h5_center_field(vy_file, nz, ny)
        u_edge[:, :, n] = read_h5_edge_field_z(vz_file, nz, ny)
        q[:, :, n]      = read_h5_center_field(q_file, nz, ny)

        # Saturation specific humidity
        qs[:, :, n] = np.exp(
            3.0 * (theta[:, :, n] - 2.0 * x_vertical)
        )

        # Relative humidity difference for this specific time
        rel_q[:, :, n] = q[:, :, n] - qs[:, :, n]

        # No-penetration boundary conditions: these are now the TRUE wall
        # edges (index 0 and index nz, since u_edge has nz+1 points), so
        # this enforcement is exact rather than a mislocated overwrite.
        u_edge[0, :, n] = 0.0
        u_edge[-1, :, n] = 0.0

    # u interpolated onto the theta/center grid: only used for saving /
    # plotting / ML input alongside the other center-grid fields. The Q
    # calculation below uses u_edge directly, not this interpolated version.
    u_center = 0.5 * (u_edge[:-1, :, :] + u_edge[1:, :, :])


    # ------------------------------------------------------------
    # Calculate Q using the staggered-grid flux form
    #
    # Q = dtheta/dt + d(u theta)/dz + d(v theta)/dy
    #
    # CHANGE: d(u theta)/dz is now computed as a proper finite-volume flux
    # difference, using u on its native edge grid and theta interpolated
    # onto those same edges (face value = average of the two neighboring
    # cell centers):
    #
    #   d(u theta)/dz |_k = [u_edge[k+1]*0.5*(theta_k+theta_{k+1})
    #                         - u_edge[k]*0.5*(theta_k+theta_{k-1})] / dz
    #
    # At k=0, the second (missing-neighbor) term is multiplied by
    # u_edge[0]=0 (bottom wall) and vanishes exactly.
    # At k=nz-1, the first (missing-neighbor) term is multiplied by
    # u_edge[nz]=0 (top wall) and vanishes exactly.
    # No ghost point / mirrored value is needed: the true wall velocity
    # being zero does that job for us. We pad theta with dummy zero rows
    # purely so numpy doesn't index out of bounds; those dummy values are
    # always multiplied by the zero wall edge, so their value never
    # matters.
    #
    # d(v theta)/dy uses the same flux-form averaging, but with periodic
    # wraparound (np.roll) instead of a wall, since v is staggered in y
    # and the domain is periodic there.
    # ------------------------------------------------------------

    for n in tqdm(range(0, nt), desc="Calculating Q"):

        theta_n = theta[:, :, n]

        # ---- time derivative ----
        #
        # CHANGE: the interior still uses the centered O(dt^2) stencil.
        # At the two edges (n=0, n=nt-1) we no longer skip the frame and
        # leave Q=0 there -- instead we use the one-sided 3-point stencil
        # that matches the SAME O(dt^2) accuracy as the centered scheme,
        # so the initial/final frame isn't penalized with a lower-order
        # (and therefore noisier) derivative than every other frame. This
        # matters here because n=0 is the frame we most want to trust for
        # a fair comparison against Q_gt.
        if n == 0:
            # forward, O(dt^2): (-3 f0 + 4 f1 - f2) / (2 dt)
            dtheta_dt = (
                -3.0 * theta[:, :, 0] + 4.0 * theta[:, :, 1] - theta[:, :, 2]
            ) / (2 * dt)
        elif n == nt - 1:
            # backward, O(dt^2): (3 f_N - 4 f_{N-1} + f_{N-2}) / (2 dt)
            dtheta_dt = (
                3.0 * theta[:, :, n] - 4.0 * theta[:, :, n - 1] + theta[:, :, n - 2]
            ) / (2 * dt)
        else:
            dtheta_dt = (theta[:, :, n + 1] - theta[:, :, n - 1]) / (2 * dt)

        # ---- d(u theta)/dz : wall-normal flux form ----
        # Zero-padded so slicing never leaves the array; the pad values are
        # only ever multiplied by the zero wall-edge velocity, so they are
        # inert placeholders, not approximations.
        theta_pad = np.pad(theta_n, ((1, 1), (0, 0)), mode="constant")
        theta_left = theta_pad[0:nz, :]     # theta[k-1], 0 at k=0
        theta_right = theta_pad[2:nz + 2, :]  # theta[k+1], 0 at k=nz-1

        u_edge_n = u_edge[:, :, n]
        flux_high_z = u_edge_n[1:nz + 1, :] * 0.5 * (theta_n + theta_right)
        flux_low_z = u_edge_n[0:nz, :] * 0.5 * (theta_n + theta_left)

        d_u_theta_dz = (flux_high_z - flux_low_z) / dz

        # ---- d(v theta)/dy : periodic flux form ----
        v_n = v[:, :, n]
        theta_plus_y = np.roll(theta_n, -1, axis=1)
        theta_minus_y = np.roll(theta_n, 1, axis=1)
        v_plus_y = np.roll(v_n, -1, axis=1)

        flux_high_y = v_plus_y * 0.5 * (theta_n + theta_plus_y)
        flux_low_y = v_n * 0.5 * (theta_n + theta_minus_y)

        d_v_theta_dy = (flux_high_y - flux_low_y) / dy

        Q[:, :, n] = dtheta_dt + d_u_theta_dz + d_v_theta_dy

    outfile = fld / f"{fld.name}_fields_Q.npz"
    
    print('Saving data...')
    np.savez_compressed(
        outfile,
        u=u_center,
        v=v,
        theta=theta,
        rel_q=rel_q,
        Q=Q
    )
    
    print(f"Saved: {outfile}")

    # ------------------------------------------------------------
    # Plot one Q frame
    # ------------------------------------------------------------

    # nplot = nt // 2

    # qmax = np.max(np.abs(Q[:, :, nplot]))

    # plt.figure(figsize=(12, 3))
    # plt.imshow(
    #     Q[:, :, nplot],
    #     origin="lower",
    #     aspect="auto",
    #     extent=[0, Ly, 0, Lz],
    #     vmin=-qmax,
    #     vmax=qmax
    # )
    # plt.colorbar(label="Q")
    # plt.xlabel("y")
    # plt.ylabel("z")
    # plt.title(f"Q(z,y), n = {nplot}")
    # plt.tight_layout()
    # plt.show()


for fld in simulation_folders:
    process_simulation(fld)

print("\nAll simulations completed.")

# # ------------------------------------------------------------
# # Optional: plot theta, vy, vz to check inputs
# # ------------------------------------------------------------

# plt.figure(figsize=(12, 3))
# plt.imshow(
#     theta[:, :, nplot],
#     origin="lower",
#     aspect="auto",
#     extent=[0, Ly, 0, Lz]
# )
# plt.colorbar(label=r"$/theta$")
# plt.xlabel("y")
# plt.ylabel("z")
# plt.title(f"Theta, n = {nplot}")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 3))
# plt.imshow(
#     u_center[:, :, nplot],
#     origin="lower",
#     aspect="auto",
#     extent=[0, Ly, 0, Lz]
# )
# plt.colorbar(label="vz")
# plt.xlabel("y")
# plt.ylabel("z")
# plt.title(f"vz (center-grid), n = {nplot}")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 3))
# plt.imshow(
#     v[:, :, nplot],
#     origin="lower",
#     aspect="auto",
#     extent=[0, Ly, 0, Lz]
# )
# plt.colorbar(label="vy")
# plt.xlabel("y")
# plt.ylabel("z")
# plt.title(f"vy, n = {nplot}")
# plt.tight_layout()
# plt.show()


# plt.figure(figsize=(12, 3))
# plt.imshow(
#     rel_q[:, :, nplot],
#     origin="lower",
#     aspect="auto",
#     extent=[0, Ly, 0, Lz]
# )
# plt.colorbar(label="q-qs")
# plt.xlabel("y")
# plt.ylabel("z")
# plt.title(f"q-qs, n = {nplot}")
# plt.tight_layout()
# plt.show()

