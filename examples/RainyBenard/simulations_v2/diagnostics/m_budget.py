"""Moist static energy budget of the v2 runs (means.h5 only). m = b + 2 q.
<m>(t) = z-mean of Tbar + 2 qbar (uniform grid). Total flux F(z) = <w b> + 2<w q> - kappa d<m>/dz
(same diffusivity for b and q assumed), Nu(z) = F / kappa (Delta m / H = 1).
If the run is statistically steady, Nu(z) is constant in z and d<m>/dt = 0;
d<m>/dt = kappa (Nu_bot - Nu_top) otherwise."""
import math
from pathlib import Path
import h5py, numpy as np
V2 = Path("/workspace/examples/RainyBenard/simulations_v2/runs")
def ra_pr(run):
    l = [x.split() for x in (run / "bou.in").read_text().splitlines()]
    i = [k for k, x in enumerate(l) if x and x[0] == "RAYT"][0]
    return float(l[i + 1][0]), float(l[i + 1][1])
runs = sorted((d for d in V2.iterdir() if (d / "bou.in").exists()), key=lambda d: ra_pr(d)[0])
print(f"{'run':<15} {'<m>(0)':>7} {'(100)':>7} {'(250)':>7} {'(500)':>7} {'(750)':>7} {'(1000)':>7}  "
      f"{'d<m>/dt[500,1000]':>17} {'kap(Nub-Nut)':>12}   Nu(z) t-mean[500,1000] at z = 0.06 .25 .5 .75 .94   wall bot/top")
for run in runs:
    ra, pr = ra_pr(run); kap = 1 / math.sqrt(ra * pr)
    with h5py.File(run / "outputdir/means.h5") as me:
        ks = sorted(me["Tbar"].keys())
        g = lambda n: np.array([me[n][k][()] for k in ks], float)
        T, q, wT, wq = g("Tbar"), g("qbar"), g("vxT"), g("vxq")
    t = np.array([int(k) for k in ks], float)
    nz = T.shape[1]; dz = 1 / nz; z = (np.arange(nz) + 0.5) * dz
    m = T + 2 * q
    mm = m.mean(axis=1)
    F = wT + 2 * wq - kap * np.gradient(m, dz, axis=1)
    nub = -(m[:, 0] - 2) / (dz / 2); nut = -(1 - m[:, -1]) / (dz / 2)
    L = t >= 500
    slope = np.polyfit(t[L], mm[L], 1)[0]
    pred = kap * (nub[L].mean() - nut[L].mean())
    Nz = (F[L] / kap).mean(axis=0)
    at = [Nz[np.argmin(abs(z - zz))] for zz in (0.06, 0.25, 0.5, 0.75, 0.94)]
    pick = [mm[np.argmin(abs(t - tt))] for tt in (0, 100, 250, 500, 750, 1000)]
    print(f"{run.name:<15} " + " ".join(f"{v:7.4f}" for v in pick) +
          f"  {slope:17.2e} {pred:12.2e}   " + " ".join(f"{v:5.2f}" for v in at) +
          f"   {nub[L].mean():.2f}/{nut[L].mean():.2f}")
