"""Matched-tuning transferability experiment for formaldehyde-trained SVR models.

Classical RBF-SVR (5 DDCC features) and PQK-SVR (15 projected 1-RDM features, Gaussian outer
kernel) are tuned with *identical* procedures:
  * identical grid: C x epsilon x gamma (superset of both grids used in the paper);
  * quantum models additionally select layer depth L and the feature-encoding function
    (original, degree-1, degree-2 polynomial, arctan), as was done for water in the paper;
  * 5-fold GroupKFold (groups = conformers) on a random subset of formaldehyde training
    conformers, scoring = R^2 (same criterion as the paper's GridSearchCV);
  * selected models are refit on all formaldehyde training conformers and evaluated on the
    formaldehyde test conformers and on all amplitudes of the six other molecules
    (as in Transferability_Test.py).
The grids of Table S1 are also evaluated in the same pipeline for comparison:
  * "paper" quantum: L=1, original encoding, gamma=1 fixed, C/epsilon from the paper's QSVR grid;
  * "paper" classical: the paper's classical SVR grid.
Quantum features are exact (statevector) or with finite-shot emulation (1024 shots, as in the paper).

Usage: python -I matched_transfer.py <repo> <data_root> <workdir> <stage> [n_jobs]
  stage in {features, cv, final, all}
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.svm import SVR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_pqk import projected_features  # noqa: E402
from qk_common import load_water_scaled  # noqa: E402

SOURCE = "Formaldehyde"
TARGETS = ["Water", "Ammonia", "Methane", "Ethylene", "Methanol", "Ethane"]
AMPS_PER_CONF = {"Formaldehyde": 576}
CIRCUITS = ["Hubregtsen", "MultiControl", "Chebyshev"]
LAYERS = [1, 2]
ENCODINGS = [0, 1, 2, "atan"]
SHOT_MODES = {"exact": None, "shots1024": 1024}
N_CV_CONFORMERS = 32
SEED = 0

GRID = {  # identical for classical and quantum
    "C": [0.1, 0.25, 0.5, 1, 5, 10],
    "epsilon": [0.03, 0.05, 0.08, 0.1],
    "gamma": [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5],
}
PAPER_Q_GRID = {"C": [0.1, 0.25, 0.5, 1, 5, 10], "epsilon": [0.1, 0.05, 0.03, 0.08], "gamma": [1]}
PAPER_C_GRID = {"C": [0.1, 0.5, 1, 5, 10], "epsilon": [0.1, 0.05, 0.03, 0.08],
                "gamma": [0.2, 1, 0.1, 0.01, 0.05, 0.001]}  # 'auto' = 1/n_features = 0.2


def combos(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def cfg_name(circuit, layers, enc):
    return f"{circuit}_L{layers}_{enc}"


def load_all(data_root):
    data = {"source_train": load_water_scaled(data_root, SOURCE, "train"),
            "source_test": load_water_scaled(data_root, SOURCE, "test")}
    for mol in TARGETS:
        Xa, ya = load_water_scaled(data_root, mol, "test")
        Xb, yb = load_water_scaled(data_root, mol, "train")
        data[mol] = (np.vstack([Xa, Xb]), np.concatenate([ya, yb]))  # same order as Transferability_Test.py
    return data


def blocks(data):
    order = ["source_train", "source_test"] + TARGETS
    sizes = [len(data[k][0]) for k in order]
    return order, np.cumsum([0] + sizes)


def stage_features(repo, data_root, work, n_jobs):
    data = load_all(data_root)
    order, offs = blocks(data)
    X_all = np.vstack([data[k][0] for k in order])
    np.savez(work / "targets.npz", **{k: data[k][1] for k in order}, offsets=offs, order=np.array(order))
    np.save(work / "classical_X.npy", X_all)
    for circuit, layers, enc in itertools.product(CIRCUITS, LAYERS, ENCODINGS):
        out = work / f"F_{cfg_name(circuit, layers, enc)}_exact.npy"
        if out.exists():
            continue
        t0 = time.perf_counter()
        F = projected_features(repo, circuit, enc, layers, X_all, n_jobs=n_jobs)
        np.save(out, F)
        print(f"features {out.name}: {F.shape} in {time.perf_counter() - t0:.0f} s", flush=True)


def features(work, name, shots_mode):
    """Return the full (all molecules) feature matrix for a model name."""
    if name == "classical":
        return np.load(work / "classical_X.npy")
    F = np.load(work / f"F_{name}_exact.npy")
    shots = SHOT_MODES[shots_mode]
    if shots:
        seed = int.from_bytes(name.encode(), "little") % (2 ** 32)  # deterministic per configuration
        rng = np.random.default_rng(seed)
        F = 2 * rng.binomial(shots, np.clip((1 + F) / 2, 0, 1)) / shots - 1
    return F


def cv_subset(n_train):
    per = AMPS_PER_CONF[SOURCE]
    rng = np.random.default_rng(SEED)
    confs = np.sort(rng.choice(n_train // per, N_CV_CONFORMERS, replace=False))
    idx = np.concatenate([np.arange(c * per, (c + 1) * per) for c in confs])
    return idx, np.repeat(confs, per)


def _cv_one(F, y, groups, params):
    scores = []
    for tr, va in GroupKFold(n_splits=5).split(F, y, groups):
        m = SVR(kernel="rbf", **params).fit(F[tr], y[tr])
        scores.append(r2_score(y[va], m.predict(F[va])))
    return float(np.mean(scores)), float(np.std(scores))


def stage_cv(work, n_jobs):
    t = np.load(work / "targets.npz")
    offs = t["offsets"]
    y_src = t["source_train"]
    idx, groups = cv_subset(len(y_src))
    jobs = [("classical", "none", "classical")]
    for shots_mode in SHOT_MODES:
        for circuit, layers, enc in itertools.product(CIRCUITS, LAYERS, ENCODINGS):
            jobs.append((cfg_name(circuit, layers, enc), shots_mode, circuit))
    rows = []
    for name, shots_mode, family in jobs:
        out = work / f"cv_{name}_{shots_mode}.csv"
        if out.exists():
            rows.append(pd.read_csv(out))
            continue
        F = features(work, name, shots_mode)[offs[0]:offs[1]][idx]
        y = y_src[idx]
        t0 = time.perf_counter()
        res = Parallel(n_jobs=n_jobs)(delayed(_cv_one)(F, y, groups, p) for p in combos(GRID))
        df = pd.DataFrame([{**p, "cv_r2": m, "cv_r2_std": s} for p, (m, s) in zip(combos(GRID), res)])
        df["name"], df["shots"], df["family"] = name, shots_mode, family
        df.to_csv(out, index=False)
        rows.append(df)
        best = df.loc[df["cv_r2"].idxmax()]
        print(f"cv {name:24s} {shots_mode:9s}: best R2={best.cv_r2:.4f} (C={best.C}, eps={best.epsilon}, "
              f"gamma={best.gamma}) [{time.perf_counter() - t0:.0f} s]", flush=True)
    pd.concat(rows).to_csv(work / "cv_all.csv", index=False)


def _in_grid(df, grid):
    mask = np.ones(len(df), bool)
    for k, vals in grid.items():
        mask &= df[k].isin(vals)
    return df[mask]


def select_models(cv):
    sel = []
    c = cv[cv.family == "classical"]
    sel.append(("classical", "matched", "none", c.loc[c.cv_r2.idxmax()]))
    pc = _in_grid(c, PAPER_C_GRID)
    sel.append(("classical", "paper", "none", pc.loc[pc.cv_r2.idxmax()]))
    for shots_mode in SHOT_MODES:
        for circuit in CIRCUITS:
            q = cv[(cv["family"] == circuit) & (cv["shots"] == shots_mode)]
            sel.append((circuit, "matched", shots_mode, q.loc[q["cv_r2"].idxmax()]))
            pq = _in_grid(q[q["name"] == cfg_name(circuit, 1, 0)], PAPER_Q_GRID)
            sel.append((circuit, "paper", shots_mode, pq.loc[pq.cv_r2.idxmax()]))
    return sel


def _final_one(work, family, protocol, shots_mode, row):
    t = np.load(work / "targets.npz", allow_pickle=True)
    offs, order = t["offsets"], list(t["order"])
    F = features(work, row["name"], shots_mode)
    params = {"C": float(row.C), "epsilon": float(row.epsilon), "gamma": float(row.gamma)}
    t0 = time.perf_counter()
    m = SVR(kernel="rbf", **params).fit(F[offs[0]:offs[1]], t["source_train"])
    fit_s = time.perf_counter() - t0
    out = {"family": family, "protocol": protocol, "shots": shots_mode, "config": row["name"], **params,
           "cv_r2": float(row.cv_r2), "n_sv": int(len(m.support_)), "fit_s": fit_s}
    for i, key in enumerate(order):
        if key == "source_train":
            continue
        pred = m.predict(F[offs[i]:offs[i + 1]])
        out["Formaldehyde (test)" if key == "source_test" else key] = float(r2_score(t[key], pred))
    return out


def stage_final(work, n_jobs):
    cv = pd.read_csv(work / "cv_all.csv")
    sel = select_models(cv)
    res = Parallel(n_jobs=min(n_jobs, len(sel)))(
        delayed(_final_one)(work, fam, prot, sm, row) for fam, prot, sm, row in sel)
    df = pd.DataFrame(res)
    df.to_csv(work / "final_results.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    repo, data_root, work, stage = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    n_jobs = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    work.mkdir(parents=True, exist_ok=True)
    (work / "config.json").write_text(json.dumps({
        "source": SOURCE, "targets": TARGETS, "circuits": CIRCUITS, "layers": LAYERS,
        "encodings": [str(e) for e in ENCODINGS], "shot_modes": list(SHOT_MODES), "grid": GRID,
        "paper_quantum_grid": PAPER_Q_GRID, "paper_classical_grid": PAPER_C_GRID,
        "n_cv_conformers": N_CV_CONFORMERS, "seed": SEED}, indent=2))
    if stage in ("features", "all"):
        stage_features(repo, data_root, work, n_jobs)
    if stage in ("cv", "all"):
        stage_cv(work, n_jobs)
    if stage in ("final", "all"):
        stage_final(work, n_jobs)
