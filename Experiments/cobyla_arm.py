"""Circuit-parameter optimization arm of the matched formaldehyde experiment.

For each circuit, start from the configuration (L, encoding) selected by the matched cross-validation
(exact features) with its untrained parameters, and optimize the circuit parameters with COBYLA
exactly as in the paper's water runs: at most 100 function evaluations, objective = training-set MSE of
an SVR with the QSVR defaults (C=1, epsilon=0.1), outer gamma=1, max_iter=50 (Training_Functions.MSE_Score),
evaluated on the same 32-conformer subset used for cross-validation. The optimized circuit then goes
through the identical C/epsilon/gamma cross-validation, refit on all training conformers, and transfer
evaluation as the other models.

Usage: python -I cobyla_arm.py <repo> <data_root> <workdir> [n_jobs]
"""
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.metrics import mean_squared_error, r2_score  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_transfer as mt  # noqa: E402
from fast_pqk import projected_features  # noqa: E402
from qk_common import build_pqk  # noqa: E402

repo, data_root, work = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MAXITER = 100
if __import__("os").environ.get("COBYLA_TEST"):  # quick functional test on the small test workdir
    mt.CIRCUITS, mt.N_CV_CONFORMERS, MAXITER = ["Hubregtsen"], 8, 4
    mt.GRID = {"C": [1], "epsilon": [0.05], "gamma": [0.2, 1]}
n_jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 20

t = np.load(work / "targets.npz", allow_pickle=True)
offs, order = t["offsets"], list(t["order"])
X_all = np.load(work / "classical_X.npy")
y_src = t["source_train"]
idx, groups = mt.cv_subset(len(y_src))
X_sub, y_sub = X_all[offs[0]:offs[1]][idx], y_src[idx]
cv = pd.read_csv(work / "cv_all.csv")

rows = []
for circuit in mt.CIRCUITS:
    q = cv[(cv["family"] == circuit) & (cv["shots"] == "exact")]
    start = q.loc[q["cv_r2"].idxmax()]
    name = start["name"]
    layers = int(name.split("_L")[1].split("_")[0])
    enc_s = name.split("_")[-1]
    enc = enc_s if enc_s == "atan" else int(enc_s)
    theta0 = np.asarray(build_pqk(repo, circuit, enc, layers, shots=None, seed=0).parameters, float)
    history = []

    def loss(theta):
        F = projected_features(repo, circuit, enc, layers, X_sub, n_jobs=n_jobs, params=theta)
        m = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma=1.0, max_iter=50).fit(F, y_sub)
        val = mean_squared_error(y_sub, m.predict(F))
        history.append(val)
        return val

    t0 = time.perf_counter()
    opt = minimize(loss, theta0, method="COBYLA", options={"maxiter": MAXITER, "rhobeg": 1.0})
    print(f"{name}: COBYLA {len(history)} evals, MSE {history[0]:.3e} -> {opt.fun:.3e} "
          f"[{time.perf_counter() - t0:.0f} s]", flush=True)
    np.save(work / f"theta_cobyla_{name}.npy", opt.x)

    F_all = projected_features(repo, circuit, enc, layers, X_all, n_jobs=n_jobs, params=opt.x)
    for shots_mode, shots in mt.SHOT_MODES.items():
        F = F_all
        if shots:
            rng = np.random.default_rng(int.from_bytes((name + "cobyla").encode(), "little") % (2 ** 32))
            F = 2 * rng.binomial(shots, np.clip((1 + F_all) / 2, 0, 1)) / shots - 1
        Fs = F[offs[0]:offs[1]][idx]
        res = Parallel(n_jobs=n_jobs)(delayed(mt._cv_one)(Fs, y_sub, groups, p) for p in mt.combos(mt.GRID))
        cvdf = pd.DataFrame([{**p, "cv_r2": r[0], "cv_r2_std": r[1]} for p, r in zip(mt.combos(mt.GRID), res)])
        cvdf.to_csv(work / f"cv_cobyla_{name}_{shots_mode}.csv", index=False)
        b = cvdf.loc[cvdf["cv_r2"].idxmax()]
        p = {"C": float(b.C), "epsilon": float(b.epsilon), "gamma": float(b.gamma)}
        m = SVR(kernel="rbf", **p).fit(F[offs[0]:offs[1]], y_src)
        out = {"family": circuit, "protocol": "matched+COBYLA", "shots": shots_mode, "config": name,
               **p, "cv_r2": float(b.cv_r2), "n_sv": int(len(m.support_)),
               "cobyla_mse_start": history[0], "cobyla_mse_end": float(opt.fun), "cobyla_evals": len(history)}
        for i, key in enumerate(order):
            if key == "source_train":
                continue
            out["Formaldehyde (test)" if key == "source_test" else key] = float(
                r2_score(t[key], m.predict(F[offs[i]:offs[i + 1]])))
        rows.append(out)
        print({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in out.items()}, flush=True)
    pd.DataFrame(rows).to_csv(work / "cobyla_results.csv", index=False)
