"""Check the simulations_v2/runs folders: bou.in parses with afidtools (55 lines) and differs from
the template only in the intended lines; drizzle.h5 has NXM points and satisfies the drizzle
solution written by tools/drizzle.py: m = b + gamma q = gamma + (beta-1-gamma) z,
q = q_s(b,z) in the saturated layer, q < q_s above; q(top) -> 0, b(top) -> beta - 1."""
import difflib, hashlib, sys
from pathlib import Path
import h5py, numpy as np
sys.path.insert(0, "/workspace/tools")
import afidtools as afid
RB = Path("/workspace/examples/RainyBenard")
OUT = RB / "simulations_v2/runs"
tmpl = (RB / "bou.in").read_text().splitlines()
md5s = []
for d in sorted(p for p in OUT.iterdir() if (p / "bou.in").exists()):
    print(f"=== {d.name}")
    new = (d / "bou.in").read_text().splitlines()
    for l in difflib.unified_diff(tmpl, new, lineterm="", n=0):
        if l.startswith(("+", "-")) and not l.startswith(("+++", "---")):
            print("   ", l)
    ip, mp = afid.InputParams(str(d)), afid.MoistParams(str(d))
    print(f"    afidtools: nxm {ip.nxm} nym {ip.nym} nzm {ip.nzm} ylen {ip.ylen}; alpha {mp.alpha} beta {mp.beta} gamma {mp.gamma} tau {mp.tau} qfixN {mp.qfixN} qfixS {mp.qfixS}")
    with h5py.File(d / "drizzle.h5") as f:
        b, q = f["b"][()], f["q"][()]
    z = 0.5 * (np.linspace(0, 1, ip.nxm + 1)[1:] + np.linspace(0, 1, ip.nxm + 1)[:-1])
    a, be, g = mp.alpha, mp.beta, mp.gamma
    qs = np.exp(a * (b - be * z))
    m = b + g * q
    sat = np.isclose(q, qs, rtol=1e-9)
    print(f"    drizzle.h5: n = {b.size}; b [{b[0]:.4f} .. {b[-1]:.4f}], q [{q[0]:.4f} .. {q[-1]:.4f}]; "
          f"max|m - (gamma + (beta-1-gamma) z)| = {np.abs(m - (g + (be - 1 - g) * z)).max():.2e}")
    print(f"    saturated cells (q = q_s): z <= {z[sat].max():.3f} ({sat.sum()} cells); max (q - q_s) above: {(q - qs)[~sat].max():.2e}; min b {b.min():.4f}, min q {q.min():.4f}")
    md5s.append(np.concatenate([b, q]))
print("drizzle.h5 profiles identical in both folders:", all(np.array_equal(md5s[0], x) for x in md5s))
