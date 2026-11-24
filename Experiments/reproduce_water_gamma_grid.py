"""Search the classical numeric grid for the ZZ_map/atan/L1 quantum kernel.

Search C=[.1,.5,1,5,10], epsilon=[.1,.05,.03,.08], gamma=[.2,1,.1,.01,.05,.001]
on full Water training data with KFold(5). Classical gamma='auto' resolves to .2
on its five input features; we explicitly use .2 for the quantum kernel too.
Circuit optimization is repeated within every fold for each gamma, because gamma
enters the fixed-C/epsilon training-MSE objective. The gamma=1 folds are reused
only after their saved feature samples and CV scores pass a replay check.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reproduce_water_transfer import (MOLS, ROOT, angles, expectations, measure,
                                      optimize, sha256, write_json)
from fast_pqk import projected_features
from qk_common import build_pqk, load_water_scaled

GRID = {"C": [.1, .5, 1, 5, 10], "epsilon": [.1, .05, .03, .08]}
GAMMAS = [.2, 1., .1, .01, .05, .001]


class ZZAtanL1:
    num_parameters = 9

    def __call__(self, X, theta):
        return expectations(angles(X), theta)


def validation_rows(data):
    rows = []
    for X, _ in data.values():
        indices = np.unique(np.r_[np.linspace(0, len(X) - 1, 3, dtype=int),
                                  np.argmin(X, axis=0), np.argmax(X, axis=0)])
        rows.append(X[indices])
    return np.vstack(rows)


def replay_features(opt, simulator, X):
    theta = np.asarray(opt["theta"])
    F = measure(simulator(X, theta), 1024, opt["final_measurement_seed"])
    return theta, F, opt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "Experiments/results/water_reproduction")
    parser.add_argument("--out", type=Path, default=ROOT / "Experiments/results/water_quantum_gamma_grid")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "paper_results/experiments/water_quantum_gamma_grid")
    args = parser.parse_args()
    args.baseline = args.baseline.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    seed = 42
    prefix = "zz_atan_L1_shots1024_seed42"
    data = {(mol, split): load_water_scaled(ROOT / "data", mol, split)
            for mol in MOLS for split in ["train", "test"]}
    X, y = data["Water", "train"]
    simulator = ZZAtanL1()
    files = [Path(__file__).resolve(),
             ROOT / "Experiments/reproduce_water_transfer.py", ROOT / "Experiments/qk_common.py",
             ROOT / "Experiments/fast_pqk.py", ROOT / "Experiments/batched_pqk.py",
             ROOT / "QKernel_Methods_Test.py", ROOT / "Transferability_Test.py",
             ROOT / "Training_Functions.py", ROOT / "EncodingCircuit.py",
             ROOT / "Circuit_Classes.py", ROOT / "FunctionEncoder.py"]
    files += [args.baseline / f"{prefix}_fits.json"]
    files += [args.baseline / f"{prefix}_cv.csv", args.baseline / f"{prefix}_scores.csv",
              args.baseline / "manifest.json"]
    for mol in MOLS:
        base = ROOT / "data/training/STO-3G" / mol / "scaled"
        files.extend(base / f for f in ["full_feature_train.bin", "full_feature_test.bin",
                                       "feature_scaler.bin", "t2_scaler.bin"])
    manifest = {
        "model": "ZZ_map_atan_L1", "source": "Water", "basis": "STO-3G", "seed": seed,
        "shots": 1024, "n_train": len(y), "grid": {**GRID, "gamma": GAMMAS}, "candidates": 120,
        "gamma_auto": "Classical auto=1/5=.2 is used for both kernels.",
        "cv": "KFold(5,shuffle=False); independent fold-local optimization for each gamma",
        "optimization": "COBYLA(maxiter=100,rhobeg=1), SVR(C=1,epsilon=.1,max_iter=50,gamma=candidate_gamma) training MSE",
        "selection": "Only water CV; transfer targets evaluated once after hyperparameters selected.",
        "baseline": str(args.baseline),
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "squlearn"]},
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files}}
    write_json(args.out / "manifest.json", {k: v for k, v in manifest.items() if k != "sha256"})
    write_json(args.artifacts / "run_manifest.json", manifest)
    candidates = list(ParameterGrid(GRID))
    baseline_cv = pd.read_csv(args.baseline / f"{prefix}_cv.csv")
    baseline_fits = json.loads((args.baseline / f"{prefix}_fits.json").read_text())
    cv_rows, reuse_checks = [], []
    for gamma in GAMMAS:
        fold_scores = np.zeros((len(candidates), 5))
        for fold, (tr, va) in enumerate(KFold(5).split(X)):
            if gamma == 1:
                theta, Ftr, opt = replay_features(
                    baseline_fits[f"fold{fold}"], simulator, X[tr])
                objective = SVR(gamma=1, max_iter=50).fit(Ftr, y[tr])
                objective_error = abs(mean_squared_error(y[tr], objective.predict(Ftr)) - opt["mse_final"])
                assert objective_error < 1e-12
            else:
                theta, Ftr, opt = optimize(X[tr], y[tr], 1024, seed,
                                           f"gamma{gamma:g}_fold{fold}", args.artifacts, gamma=gamma)
            Fva = measure(simulator(X[va], theta), 1024, opt["next_measurement_seed"])
            for i, parameters in enumerate(candidates):
                model = SVR(gamma=gamma, **parameters).fit(Ftr, y[tr])
                assert model.fit_status_ == 0
                fold_scores[i, fold] = r2_score(y[va], model.predict(Fva))
                if gamma == 1:
                    old = baseline_cv[(baseline_cv.C == parameters["C"]) &
                                      (baseline_cv.epsilon == parameters["epsilon"])].iloc[0]
                    assert abs(fold_scores[i, fold] - old[f"fold{fold}"]) < 1e-12
            if gamma == 1:
                reuse_checks.append({"fold": fold, "objective_mse_difference": float(objective_error),
                                     "all_20_candidate_scores_match": True})
        means = fold_scores.mean(axis=1)
        cv_rows.extend({**parameters, "gamma": gamma, "cv_r2": means[i],
                        **{f"fold{f}": fold_scores[i, f] for f in range(5)}}
                       for i, parameters in enumerate(candidates))
        pd.DataFrame(cv_rows).to_csv(args.out / "cv.csv", index=False)
        print(f"gamma={gamma:g}: best water CV R2={means.max():.9f}", flush=True)
    write_json(args.artifacts / "baseline_replay_check.json", reuse_checks)
    cv = pd.DataFrame(cv_rows)
    winner = cv.loc[cv.cv_r2.idxmax()]
    best = {"C": float(winner.C), "epsilon": float(winner.epsilon), "gamma": float(winner.gamma)}
    print("SELECTED", best, "CV R2", winner.cv_r2, flush=True)
    if best["gamma"] == 1:
        theta, Ftr, opt = replay_features(baseline_fits["full"], simulator, X)
        write_json(args.artifacts / "full_optimization.json", opt)
    else:
        theta, Ftr, opt = optimize(X, y, 1024, seed, "full", args.artifacts, gamma=best["gamma"])
    model = SVR(**best).fit(Ftr, y)
    assert model.fit_status_ == 0
    xt, yt = data["Water", "test"]
    Ft = measure(simulator(xt, theta), 1024, opt["next_measurement_seed"])
    training_prediction = model.predict(Ft)
    training_r2 = float(r2_score(yt, training_prediction))
    # Validate the selected circuit and bandwidth independently of training.
    representative = validation_rows(data)
    reference = projected_features(ROOT, "ZZ_map", "atan", 1, representative,
                                   n_jobs=1, params=theta)
    feature_error = float(np.max(np.abs(reference - simulator(representative, theta))))
    assert feature_error < 1e-11
    small = representative[:7]
    native = build_pqk(ROOT, "ZZ_map", "atan", 1, shots=None, seed=seed, gamma=best["gamma"])
    native.assign_parameters(theta)
    native_error = float(np.max(np.abs(native.evaluate(small) - np.exp(
        -best["gamma"] * squareform(pdist(simulator(small, theta), "sqeuclidean"))))))
    assert native_error < 1e-11
    K = squareform(pdist(Ftr, "sqeuclidean"))
    K *= -best["gamma"]
    np.exp(K, out=K)
    pre = SVR(kernel="precomputed", C=best["C"], epsilon=best["epsilon"]).fit(K, y)
    cross = cdist(Ft, Ftr, "sqeuclidean")
    cross *= -best["gamma"]
    np.exp(cross, out=cross)
    prediction_error = float(np.max(np.abs(training_prediction - pre.predict(cross))))
    assert prediction_error < 1e-6
    write_json(args.artifacts / "validation.json", {"statevector_feature_max_error": feature_error,
               "native_selected_gamma_gram_max_error": native_error,
               "full_precomputed_svr_prediction_max_error": prediction_error,
               "n_train": len(y), "n_test": len(yt), "final_svr_fit_status": int(model.fit_status_)})
    del K, cross, pre
    joblib.dump({"model": model, "theta": theta, "training_features": Ftr,
                 "configuration": ["ZZ_map", "atan", 1], "shots": 1024, "seed": seed,
                 "best_params": best}, args.artifacts / "trained_model.joblib")
    np.savez_compressed(args.artifacts / "training_water_predictions.npz", y=yt, predicted=training_prediction)
    reload_seed = opt["next_measurement_seed"] + 1
    Freload = measure(simulator(X, theta), 1024, reload_seed)
    transferred = SVR(**best).fit(Freload, y)
    assert transferred.fit_status_ == 0
    joblib.dump({"model": transferred, "theta": theta, "training_features": Freload,
                 "configuration": ["ZZ_map", "atan", 1], "shots": 1024,
                 "measurement_seed": reload_seed, "best_params": best},
                args.artifacts / "transfer_refit_model.joblib")
    rows = []
    for mol in MOLS:
        xt, yt = data[mol, "test"]
        xa, ya = data[mol, "train"]
        predicted = transferred.predict(measure(simulator(xt, theta), 1024, reload_seed + 1))
        xf = xt if mol == "Water" else np.vstack([xt, xa])
        yf = yt if mol == "Water" else np.concatenate([yt, ya])
        predicted_all = transferred.predict(measure(simulator(xf, theta), 1024, reload_seed + 1))
        np.savez_compressed(args.artifacts / f"{mol}_predictions.npz", y_test=yt, predicted_test=predicted,
                            y_all=yf, predicted_all=predicted_all)
        rows.append({"molecule": mol, **best, "cv_r2": float(winner.cv_r2),
                     "test_r2": r2_score(yt, predicted),
                     "all_r2": r2_score(yf, predicted_all), "n_test": len(yt), "n_all": len(yf)})
    scores = pd.DataFrame(rows)
    scores.to_csv(args.out / "scores.csv", index=False)
    write_json(args.out / "selection.json", {"best_params": best, "cv_r2": float(winner.cv_r2),
        "training_water_test_r2": training_r2, "transfer_water_test_r2": rows[0]["test_r2"],
        "mean_transfer_test_r2": float(scores.iloc[1:].test_r2.mean()),
        "mean_transfer_all_r2": float(scores.iloc[1:].all_r2.mean()),
        "seed": seed, "shots": 1024, "full_optimization_evaluations": opt["nfev"],
        "full_optimization_converged": opt["success"], "transfer_reload_seed": reload_seed,
        "sampling": "Independent binomial draws"})
    print(scores.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
