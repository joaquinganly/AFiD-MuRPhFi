"""Where and when q < 0 in one run (movie_zcut.h5), the frames and rows with three alternating
extrema in a row along z, and the number of nearly still frames (max |w| < 1e-3). Read-only.
Usage: python diag_qneg.py RUN_NAME   (a folder in simulations_v2/runs)"""
import sys
import numpy as np, h5py
if len(sys.argv) != 2:
    sys.exit("usage: python diag_qneg.py RUN_NAME")
R = f"/workspace/examples/RainyBenard/simulations_v2/runs/{sys.argv[1]}/outputdir/flowmov/movie_zcut.h5"
def alt_z(b):
    d = np.diff(b, axis=0); ext = (d[:-1] * d[1:] < 0) & (np.abs(d[1:] - d[:-1]) > 0.01)
    return ext[:-2] & ext[1:-1] & ext[2:]
rows = []
with h5py.File(R, "r") as f:
    keys = sorted(f["qhum"].keys())
    for k in keys:
        q = f["qhum"][k][()].T; b = f["temp"][k][()].T; w = f["vx"][k][()]
        neg = q < 0
        a = alt_z(b)
        kz, jy = np.unravel_index(np.argmin(q), q.shape)
        rows.append((int(k), q.min(), neg.sum(), (q < -1e-3).sum(), kz, a.sum(),
                     np.nonzero(a.any(axis=1))[0].min() if a.any() else -1, np.abs(w).max()))
r = np.array(rows, float)
nz = 128
print(f"frames with q<0: {int((r[:,2]>0).sum())} of {len(r)}; with q<-1e-3: {int((r[:,3]>0).sum())}")
print(f"max cells with q<0 in one frame: {int(r[:,2].max())} of {nz*2048} ({100*r[:,2].max()/(nz*2048):.3f}%); "
      f"total cell-frames q<0: {int(r[:,2].sum())} ({100*r[:,2].sum()/(nz*2048*len(r)):.4f}%)")
neg = r[r[:,2] > 0]
if len(neg):
    print(f"q<0 frames t = {int(neg[0,0])} .. {int(neg[-1,0])}; height index of min q (0=bottom,127=top): "
          f"{sorted(set(int(x) for x in neg[:,4]))[:20]}")
    print("worst 8 frames (t, q_min, n q<0, n q<-1e-3, z-index of min):")
    for x in neg[np.argsort(neg[:,1])][:8]:
        print(f"   t={int(x[0]):4d} q_min={x[1]:+.2e} n={int(x[2])} n<-1e-3={int(x[3])} k={int(x[4])}")
print(f"frames with +-+ along z: {int((r[:,5]>0).sum())}; max per frame {int(r[:,5].max())}; "
      f"lowest row index involved: {int(r[r[:,6]>=0,6].min()) if (r[:,6]>=0).any() else None}")
act = r[:,7]
print(f"max|w| per frame: median {np.median(act):.2e}; frames with max|w| < 1e-3 (quiescent): {int((act<1e-3).sum())}")
