# COPY of '../saves_q_npz.py' for flat-layout runs (fields/ and cordin_info.h5 directly in the run folder).
# Only path/loading logic changed (lines marked 'PATH CHANGE'); numerics untouched.
import numpy as np
import h5py
import os
from tqdm import tqdm
from pathlib import Path

# ------------------------------------------------------------
# Grid and simulation parameters
# ------------------------------------------------------------
nz = 64
ny = 1024
nt = 324

# ------------------------------------------------------------
# Folder where your .h5 files are stored
# ------------------------------------------------------------
base_dir = Path("/workspace/examples/RainyBenard/simulations")

simulation_folders = sorted(
    [
        f for f in base_dir.iterdir()
        if f.is_dir()
        and (f.name.startswith("Ra_") or f.name.startswith("Pr_"))
    ]
)
# PATH CHANGE: restrict to the new flat-layout runs only (never re-process existing folders)
simulation_folders = [base_dir / name for name in ["Ra_5e6", "Ra_75e6", "Ra_5e7", "Ra_25e7", "Ra_175e7"]]
print("\nFound simulations:")
for folder in simulation_folders:
    print("   ", folder.name)


# ------------------------------------------------------------
# Reading function
# ------------------------------------------------------------
def read_h5_center_field(filename, nz, ny):
    """
    Reads a cell-centered AFiD field (q): expects nz points in z.
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
# Processing function
# ------------------------------------------------------------
def process_simulation(fld):

    print("\n" + "=" * 70)
    print(f"Processing {fld.name}")
    print("=" * 70)

    data_folder = fld / "fields"  # PATH CHANGE: flat layout (no outputdir/)

    q = np.zeros([nz, ny, nt])

    # ------------------------------------------------------------
    # Read humidity data
    # ------------------------------------------------------------
    for n in tqdm(range(nt), desc="Reading humidity (q)"):
        fname = f"{n:05d}"
        
        # Note: If your actual output extension is .h5 instead of .h, update here.
        q_file = os.path.join(data_folder, fname + "_humid.h")

        q[:, :, n] = read_h5_center_field(q_file, nz, ny)

    # ------------------------------------------------------------
    # Save q field
    # ------------------------------------------------------------
    outfile = fld / f"{fld.name}_q.npz"

    print("Saving data...")
    np.savez_compressed(
        outfile,
        q=q
    )

    print(f"Saved: {outfile}")


# ------------------------------------------------------------
# Main execution loop
# ------------------------------------------------------------
for fld in simulation_folders:
    process_simulation(fld)

print("\nAll simulations completed.")