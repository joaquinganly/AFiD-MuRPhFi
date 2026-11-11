"""Test of "Convergence test/masked_convergence.py": crossing() and split() on hand-made arrays
(plain script, no pytest).
Usage (from simulations_v2/): python tests/test_masked_convergence.py "Convergence test/masked_convergence.py"
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

script = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location("masked_convergence", script)
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

# 9 frames, 3 cells: cell 0 never saturated; cell 1 saturated from frame 5 on; cell 2 only at frame 2
pos = np.zeros((9, 1, 3), dtype=bool)
pos[5:, 0, 1] = True
pos[2, 0, 2] = True
cum = np.cumsum(pos, axis=0, dtype=np.int16)
assert mc.crossing(cum, 4, 1).tolist() == [[False, True, False]]     # frames 3..5
assert mc.crossing(cum, 2, 2).tolist() == [[False, False, True]]     # frames 0..4
assert mc.crossing(cum, 6, 1).tolist() == [[False, False, False]]    # frames 5..7: cell 1 saturated throughout
assert mc.crossing(cum, 1, 1).tolist() == [[False, False, True]]     # frames 0..2 (window starts at frame 0)
assert mc.crossing(cum, 6, 2).tolist() == [[False, True, False]]     # frames 4..8

r = np.array([[1.0, 2.0, 3.0]])
m = np.array([[False, True, True]])
c_in, c_out, within = mc.split(r, m, 2.0)
assert math.isclose(c_in ** 2 + c_out ** 2, float(np.mean(r ** 2)) / 4.0)
assert math.isclose(c_in, math.sqrt(13.0 / 3.0) / 2.0) and math.isclose(c_out, math.sqrt(1.0 / 3.0) / 2.0)
assert math.isclose(within, math.sqrt(13.0 / 2.0) / 2.0)
assert math.isnan(mc.split(r, np.zeros_like(m), 2.0)[2])
assert math.isclose(mc.order(1.0, 4.0, 1, 2), 2.0)
print("ALL MASKED-CONVERGENCE TESTS PASSED")
