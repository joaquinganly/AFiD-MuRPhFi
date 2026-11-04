# COPY of '../Qc and Qd only.py' for flat-layout runs (fields/ and cordin_info.h5 directly in the run folder).
# Only path/loading logic changed (lines marked 'PATH CHANGE'); numerics untouched.
# Defino lo que voy a necesitar
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
import afidtools as afid
import os
import re
from pathlib import Path
from tqdm import tqdm

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# ------------------------------------------------------------
# Global parameters (unchanged)
# ------------------------------------------------------------

nz = 64
ny = 1024
nt = 311

Lz = 1.0
Ly = 16.0

dz = Lz / nz
dy = Ly / ny
dt = 1

z = np.linspace(0, Lz, nz, endpoint=False)
y = np.linspace(0, Ly, ny, endpoint=False)
t = np.arange(nt) * dt

gamma = 2.0
beta = 2.0
tau = 0.1

# ------------------------------------------------------------
# Base directory containing all simulations
# ------------------------------------------------------------

base_dir = Path(
    "/workspace/examples/RainyBenard/simulations"
)

# ------------------------------------------------------------
# Automatically find simulation folders
# ------------------------------------------------------------

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
# Reading function (cell-centered fields only: b=theta, q)
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


# ------------------------------------------------------------
# Laplacian setup
# ------------------------------------------------------------

b_wall_bottom = 0.0
b_wall_top = beta - 1.0


def laplacian(field_n, dz, dy):

    d2_dz2 = np.zeros_like(field_n)

    d2_dz2[1:-1, :] = (
        field_n[2:, :]
        - 2.0 * field_n[1:-1, :]
        + field_n[:-2, :]
    ) / dz**2

    d2_dz2[0, :] = (
        field_n[1, :]
        - 3.0 * field_n[0, :]
        + 2.0 * b_wall_bottom
    ) / dz**2

    d2_dz2[-1, :] = (
        field_n[-2, :]
        - 3.0 * field_n[-1, :]
        + 2.0 * b_wall_top
    ) / dz**2

    d2_dy2 = (
        np.roll(field_n, -1, axis=1)
        - 2.0 * field_n
        + np.roll(field_n, 1, axis=1)
    ) / dy**2

    return d2_dz2 + d2_dy2


# ------------------------------------------------------------
# Parse Pr and Ra automatically from folder name
# ------------------------------------------------------------

def parse_parameters(folder_name):

    if folder_name.startswith("Ra_"):

        Pr = 1.0

        Ra_string = folder_name.split("_")[1]
        Ra = float(Ra_string.replace("e", "E"))

        return Pr, Ra

    match = re.match(r"Pr_(.+)_Ra_(.+)", folder_name)

    if match is None:
        raise ValueError(f"Cannot parse folder name: {folder_name}")

    Pr_string = match.group(1)
    Ra_string = match.group(2)

    if Pr_string == "05":
        Pr = 0.5
    else:
        Pr = float(Pr_string.replace("_", "."))

    Ra = float(Ra_string.replace("e", "E"))

    return Pr, Ra


# ------------------------------------------------------------
# Main processing routine
# ------------------------------------------------------------

def process_simulation(fld):

    print("\n" + "=" * 70)
    print(f"Processing: {fld.name}")
    print("=" * 70)

    Pr, Ra = parse_parameters(fld.name)

    print(f"Pr = {Pr}")
    print(f"Ra = {Ra:.1e}")

    diff_coeff = 1.0 / np.sqrt(Pr * Ra)

    data_folder = fld / "fields"  # PATH CHANGE: flat layout (no outputdir/)

    # --------------------------------------------------------
    # Grid information
    # --------------------------------------------------------

    # PATH CHANGE: afid.Grid hard-codes fld/outputdir/cordin_info.h5; flat layout has it in fld/.
    # Reads the same 'xm' dataset that afid.Grid reads.
    from types import SimpleNamespace
    with h5py.File(fld / "cordin_info.h5", "r") as _f:
        grid = SimpleNamespace(xm=_f["xm"][()])

    x_vertical = np.asarray(grid.xm).squeeze()

    if x_vertical.size == nz + 1:
        x_vertical = x_vertical[:-1]

    if x_vertical.size != nz:
        raise ValueError(
            f"Unexpected grid.xm size: {x_vertical.size}, expected {nz}"
        )

    x_vertical = x_vertical[:, np.newaxis]

    # --------------------------------------------------------
    # Allocate arrays
    # --------------------------------------------------------
    '''New changes made here - Qc and Qb only. Dont mind Q_gt or anything else for now'''

    b = np.zeros([nz, ny, nt])
    q = np.zeros([nz, ny, nt])
    qs = np.zeros([nz, ny, nt])
    #Q_gt = np.zeros([nz, ny, nt])
    Qd = np.zeros([nz, ny, nt])
    Qc = np.zeros([nz, ny, nt])

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    for n in tqdm(
        range(nt),
        desc=f"{fld.name}: reading",
        leave=False
    ):

        fname = f"{n:05d}"

        theta_file = data_folder / (fname + "_temp.h5")
        q_file = data_folder / (fname + "_humid.h")

        b[:, :, n] = read_h5_center_field(
            theta_file,
            nz,
            ny
        )

        q[:, :, n] = read_h5_center_field(
            q_file,
            nz,
            ny
        )

        qs[:, :, n] = np.exp(
            3.0 * (b[:, :, n] - 2.0 * x_vertical)
        )

    # --------------------------------------------------------
    # Calculate Q_gt
    # --------------------------------------------------------

    for n in tqdm(
        range(nt),
        desc=f"{fld.name}: Q_d_c",
        leave=False
    ):

        b_n = b[:, :, n]
        q_n = q[:, :, n]
        qs_n = qs[:, :, n]

        Q_d = diff_coeff * laplacian(
            b_n,
            dz,
            dy
        )

        supersat = q_n - qs_n

        Q_c = (
            gamma / tau
        ) * supersat * np.heaviside(
            supersat,
            0.0
        )
        
        Qd[:, :, n] = Q_d
        Qc[:, :, n] = Q_c
        #Q_gt[:, :, n] = Q_d + Q_c

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("Saving files...")

    np.savez_compressed(
        fld / "Q_d_c.npz",
        Q_c = Qc,
        Q_d = Qd
    )

    print("Saving complete!")


# ------------------------------------------------------------
# Run all simulations
# ------------------------------------------------------------

for fld in simulation_folders:
    process_simulation(fld)

print("\nAll simulations completed.")