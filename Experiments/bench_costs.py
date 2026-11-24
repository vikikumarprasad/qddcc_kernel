"""Stage-by-stage cost of the SVR pipelines on Water/STO-3G.

Quantum models are built exactly as in QKernel_Methods_Test.py (sQUlearn PQK, XYZ 1-RDM,
qasm_simulator with 1024 shots, untrained circuit parameters, outer Gaussian gamma = 1).
sQUlearn's QSVR is an SVR on a precomputed Gram matrix, so the same C/epsilon grid search is
run here on the precomputed matrix; this separates kernel construction from SVR training.

Usage:
  python -I bench_costs.py <repo> <data_root> <model> <out.json> [n_train] [n_test]
  model in {classical, MultiControl, Chebyshev, Hubregtsen}
Run with OMP_NUM_THREADS=1 etc. for single-core numbers.
"""
import json
import os
import resource
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import psutil  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.metrics.pairwise import rbf_kernel  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qk_common import build_pqk, load_water_scaled  # noqa: E402

repo, data_root, model, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
n_train = int(sys.argv[5]) if len(sys.argv) > 5 else None
n_test = int(sys.argv[6]) if len(sys.argv) > 6 else None

proc = psutil.Process()
res = {"model": model, "host_cpu": "see lscpu", "threads": os.environ.get("OMP_NUM_THREADS"), "stages": {}}


def stage(name, fn):
    t0 = time.perf_counter()
    val = fn()
    dt = time.perf_counter() - t0
    res["stages"][name] = {"seconds": dt, "rss_GB_after": proc.memory_info().rss / 1e9}
    print(f"[{model}] {name}: {dt:.2f} s, RSS {proc.memory_info().rss / 1e9:.2f} GB", flush=True)
    return val


X_tr, y_tr = load_water_scaled(data_root, "Water", "train")
X_te, y_te = load_water_scaled(data_root, "Water", "test")
if n_train:
    X_tr, y_tr = X_tr[:n_train], y_tr[:n_train]
if n_test:
    X_te, y_te = X_te[:n_test], y_te[:n_test]
res["n_train"], res["n_test"] = len(X_tr), len(X_te)
print(f"[{model}] n_train={len(X_tr)} n_test={len(X_te)}", flush=True)

if model == "classical":
    grid = {"kernel": ["rbf"], "gamma": ["auto", 1, 0.1, 0.01, 0.05, 0.001],
            "C": [0.1, 0.5, 1, 5, 10], "epsilon": [0.1, 0.05, 0.03, 0.08]}
    gs = GridSearchCV(SVR(), grid, cv=5, scoring="r2", refit=False)
    stage("hyperparameter_search_5fold", lambda: gs.fit(X_tr, y_tr))
    best = gs.best_params_
    res["n_grid_points"] = len(gs.cv_results_["params"])
    final = SVR(**best)
    stage("final_fit", lambda: final.fit(X_tr, y_tr))
    pred = stage("inference_test", lambda: final.predict(X_te))
elif "-direct" in model:
    # Same circuits/parameters, 1-RDMs from a direct statevector evaluation (fast_pqk.py);
    # "-direct1024" additionally emulates 1024-shot sampling noise, as in the paper's Aer runs.
    from fast_pqk import projected_features
    circ, mode = model.split("-direct")
    shots = int(mode) if mode else None
    F_tr = stage("quantum_features_train",
                 lambda: projected_features(repo, circ, 0, 1, X_tr, n_jobs=1, shots=shots, shot_seed=1))
    K_tr = stage("gram_matrix_train", lambda: rbf_kernel(F_tr, gamma=1.0))
    grid = {"C": [0.1, 0.25, 0.5, 1, 5, 10], "epsilon": [0.1, 0.05, 0.03, 0.08]}
    gs = GridSearchCV(SVR(kernel="precomputed"), grid, cv=5, scoring="r2", refit=False)
    stage("hyperparameter_search_5fold", lambda: gs.fit(K_tr, y_tr))
    best = gs.best_params_
    res["n_grid_points"] = len(gs.cv_results_["params"])
    final = SVR(kernel="precomputed", **best)
    stage("final_fit", lambda: final.fit(K_tr, y_tr))
    F_te = stage("quantum_features_test",
                 lambda: projected_features(repo, circ, 0, 1, X_te, n_jobs=1, shots=shots, shot_seed=2))
    K_te = stage("cross_kernel_test", lambda: rbf_kernel(F_te, F_tr, gamma=1.0))
    pred = stage("predict_test", lambda: final.predict(K_te))
    res["gram_matrix_GB"] = K_tr.nbytes / 1e9
    res["seconds_per_point_features"] = (res["stages"]["quantum_features_train"]["seconds"] / len(X_tr))
else:
    kernel = stage("build_kernel", lambda: build_pqk(repo, model, 0, 1, shots=1024, seed=0))
    F_tr = stage("quantum_features_train", lambda: kernel.evaluate_qnn(X_tr))
    K_tr = stage("gram_matrix_train", lambda: rbf_kernel(F_tr, gamma=1.0))
    grid = {"C": [0.1, 0.25, 0.5, 1, 5, 10], "epsilon": [0.1, 0.05, 0.03, 0.08]}
    gs = GridSearchCV(SVR(kernel="precomputed"), grid, cv=5, scoring="r2", refit=False)
    stage("hyperparameter_search_5fold", lambda: gs.fit(K_tr, y_tr))
    best = gs.best_params_
    res["n_grid_points"] = len(gs.cv_results_["params"])
    final = SVR(kernel="precomputed", **best)
    stage("final_fit", lambda: final.fit(K_tr, y_tr))
    F_te = stage("quantum_features_test", lambda: kernel.evaluate_qnn(X_te))
    K_te = stage("cross_kernel_test", lambda: rbf_kernel(F_te, F_tr, gamma=1.0))
    pred = stage("predict_test", lambda: final.predict(K_te))
    res["gram_matrix_GB"] = K_tr.nbytes / 1e9
    res["seconds_per_point_features"] = (res["stages"]["quantum_features_train"]["seconds"] / len(X_tr))

res["best_params"] = {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in best.items()}
res["cv_best_r2"] = float(gs.best_score_)
res["n_support_vectors"] = int(len(final.support_))
res["test_r2"] = float(r2_score(y_te, pred))
res["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # ru_maxrss is KiB on Linux
out.write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2), flush=True)
