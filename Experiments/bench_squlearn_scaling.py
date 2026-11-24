"""Time and peak memory of the paper's sQUlearn PQK feature evaluation vs number of data points.

Evaluating all 10,176 water training points in one call (as QSVR.fit does) needs >20 GB of RAM,
so the cost is measured for n points in a fresh process and extrapolated linearly.
Usage: python -I bench_squlearn_scaling.py <repo> <data_root> <circuit> <n> <out.json>
"""
import json
import resource
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qk_common import build_pqk, load_water_scaled  # noqa: E402

repo, data_root, circuit, n, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]), Path(sys.argv[5])
X, _ = load_water_scaled(data_root, "Water", "train")
k = build_pqk(repo, circuit, 0, 1, shots=1024, seed=0)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
t0 = time.perf_counter()
F = k.evaluate_qnn(X[:n])
dt = time.perf_counter() - t0
res = {"circuit": circuit, "n": n, "seconds": dt, "ms_per_point": 1e3 * dt / n,
       "peak_rss_GB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, "rss_before_GB": rss0}
out.write_text(json.dumps(res))
print(json.dumps(res), flush=True)
