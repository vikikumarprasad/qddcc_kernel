"""Water transfer experiment for Figure 5 and Table S8.

The quantum configuration is ZZ_map / atan / L=1, selected from Figure 5.
All 10,176 training amplitudes, KFold(5), all 24 C/epsilon candidates, gamma=1,
and the 100-evaluation COBYLA objective are used.
COBYLA is fitted inside each fold, then independently on the full training set.
Its objective does not depend on the outer C/epsilon candidate, so the identical
seeded optimization is computed once per fold and reused across those candidates.

Closed-form Pauli expectations are checked against the actual Qiskit circuit and
sQUlearn before use. 1024-shot runs use independent binomial measurements, with
new draws for each new parameter vector and evaluation matrix.
Predictions and fitted models are saved.

Usage (from the repository root, in the project environment):
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python Experiments/reproduce_water_transfer.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fast_pqk import projected_features
from qk_common import build_pqk, load_water_scaled

ROOT = Path(__file__).resolve().parents[1]
MOLS = ["Water", "Ammonia", "Methane", "Formaldehyde", "Ethylene", "Methanol", "Ethane"]
QGRID = {"C": [.1, .25, .5, 1, 5, 10], "epsilon": [.1, .05, .03, .08]}
CGRID = {"gamma": ["auto", 1, .1, .01, .05, .001],
         "C": [.1, .5, 1, 5, 10], "epsilon": [.1, .05, .03, .08]}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def angles(X):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != 5:
        raise ValueError("This evaluator is only for the five-qubit ZZ_map-atan-L1 circuit")
    return np.column_stack((np.arctan(2 * X),
                            np.arctan(2 * (np.pi - X[:, :-1]) * (np.pi - X[:, 1:]))))


def expectations(A, theta):
    """Exact [X0,Y0,Z0,...,X4,Y4,Z4] for the repository's diagonal L1 circuit.

    Following H on every qubit, Rz(a_q) and CX-P(b_edge)-CX are all diagonal.
    Tracing independent neighbouring bits multiplies the qubit coherence by
    cos(b_edge); Z vanishes. Thus X+iY = exp(i*a_q)*product(cos(b_edge)).
    """
    phase = A * np.asarray(theta)
    envelope = np.ones((len(A), 5))
    edge = np.cos(phase[:, 5:])
    envelope[:, :-1] *= edge
    envelope[:, 1:] *= edge
    result = np.zeros((len(A), 5, 3))
    result[:, :, 0] = np.cos(phase[:, :5]) * envelope
    result[:, :, 1] = np.sin(phase[:, :5]) * envelope
    return result.reshape(-1, 15)


def measure(F, shots, seed):
    if not shots:
        return F
    return 2 * np.random.default_rng(seed).binomial(
        shots, np.clip((F + 1) / 2, 0, 1)) / shots - 1


def validate(data, out):
    # Include out-of-distribution feature ranges and several parameter draws.
    X = np.vstack([pair[0][np.linspace(0, len(pair[0]) - 1, 3, dtype=int)]
                   for (mol, split), pair in data.items() if split == "test"])
    errors = []
    for seed in [0, 1, 42]:
        theta = np.random.RandomState(seed).uniform(-np.pi, np.pi, 9)
        ref = projected_features(ROOT, "ZZ_map", "atan", 1, X, n_jobs=1, params=theta)
        err = float(np.max(np.abs(ref - expectations(angles(X), theta))))
        assert err < 1e-11, err
        errors.append(err)
    kernel = build_pqk(ROOT, "ZZ_map", "atan", 1, shots=None, seed=42)
    kernel.assign_parameters(theta)
    # Representative native circuit evaluations, independent of the fast helper.
    Xsmall = X[::3]
    K_native = kernel.evaluate(Xsmall)
    Fsmall = expectations(angles(Xsmall), theta)
    K_fast = np.exp(-squareform(pdist(Fsmall, "sqeuclidean")))
    gram_error = float(np.max(np.abs(K_native - K_fast)))
    assert gram_error < 1e-11, gram_error
    # Check the truncated SVR objective and prediction, including libsvm's two
    # routes (precomputed QSVR matrix versus direct RBF on projected features).
    ysmall = np.linspace(-.8, .9, len(Fsmall))
    pre = SVR(kernel="precomputed", max_iter=50).fit(K_native, ysmall)
    direct = SVR(gamma=1, max_iter=50).fit(Fsmall, ysmall)
    prediction_error = float(np.max(np.abs(pre.predict(K_native) - direct.predict(Fsmall))))
    assert prediction_error < 1e-8, prediction_error
    # Record real Aer finite-shot measurements for the selected configuration.
    shot_kernel = build_pqk(ROOT, "ZZ_map", "atan", 1, shots=1024, seed=42)
    shot_kernel.assign_parameters(theta)
    backend = shot_kernel._executor.backend
    original_run = backend.run
    calls = []

    def record(circuits, *args, **kwargs):
        batch = circuits if isinstance(circuits, list) else [circuits]
        calls.append({"circuits": len(batch), "shots": kwargs.get("shots"),
                      "measured_bits": sorted({c.count_ops().get("measure", 0) for c in batch})})
        return original_run(circuits, *args, **kwargs)

    backend.run = record
    try:
        sampled = np.asarray(shot_kernel.evaluate_qnn(Xsmall[:2]))
    finally:
        backend.run = original_run
    assert sampled.shape == (2, 15)
    assert sum(c["circuits"] for c in calls) == 30
    assert all(c["shots"] == 1024 and c["measured_bits"] == [1] for c in calls)
    result = {"statevector_feature_max_errors": errors, "native_gram_max_error": gram_error,
              "native_vs_direct_svr_prediction_max_error": prediction_error,
              "native_shot_calls": calls, "native_shot_features": sampled.tolist(),
              "scope": "Exact circuit and SVR equivalence; separate-Pauli sampling confirmed. "
                       "Binomial random streams are not Aer random streams."}
    write_json(out / "validation.json", result)
    print("Validation:", result, flush=True)


def classical(data, out, artifacts, workers):
    X, y = data["Water", "train"]
    grid = GridSearchCV(SVR(), CGRID, cv=5, scoring="r2", n_jobs=workers,
                        error_score="raise").fit(X, y)
    pd.DataFrame(grid.cv_results_).to_csv(out / "classical_cv.csv", index=False)
    rows = []
    for name, model in [("source_cv", grid.best_estimator_)]:
        joblib.dump(model, artifacts / f"classical_{name}.joblib")
        for mol in MOLS:
            xt, yt = data[mol, "test"]
            xa, ya = data[mol, "train"]
            pred = model.predict(xt)
            full = pred if mol == "Water" else np.concatenate([pred, model.predict(xa)])
            truth = yt if mol == "Water" else np.concatenate([yt, ya])
            rows.append({"model": name, "molecule": mol, "C": model.C,
                         "epsilon": model.epsilon, "gamma": model._gamma,
                         "test_r2": r2_score(yt, pred),
                         "all_r2": r2_score(truth, full), "n_test": len(yt), "n_all": len(truth)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "classical_scores.csv", index=False)
    write_json(out / "classical_selection.json", {"best_params": grid.best_params_,
               "cv_r2": grid.best_score_, "n_train": len(y), "candidates": len(grid.cv_results_["params"])})
    print("Classical:", grid.best_params_, grid.best_score_, flush=True)
    print(df.to_string(index=False), flush=True)


def optimize(X, y, shots, seed, label, out, gamma=1):
    """One independent representation fit; no validation/target labels enter here."""
    A = angles(X)
    theta0 = np.random.RandomState(seed).uniform(-np.pi, np.pi, 9)
    history, evaluated = [], {}
    started = time.perf_counter()

    def get_features(theta):
        key = tuple(theta)
        if key not in evaluated:
            evaluated[key] = seed + len(evaluated)
        return measure(expectations(A, theta), shots, evaluated[key])

    def loss(theta):
        F = get_features(theta)
        model = SVR(C=1, epsilon=.1, gamma=gamma, max_iter=50).fit(F, y)
        value = mean_squared_error(y, model.predict(F))
        history.append({"evaluation": len(history), "mse": value, "theta": theta.tolist(),
                        "measurement_seed": evaluated[tuple(theta)]})
        return value

    result = minimize(loss, theta0, method="COBYLA", options={"maxiter": 100, "rhobeg": 1.0})
    F = get_features(result.x)
    record = {"label": label, "shots": shots, "seed": seed, "gamma": gamma, "n_train": len(y),
              "nfev": int(result.nfev), "success": bool(result.success),
              "message": str(result.message), "theta_initial": theta0.tolist(),
              "theta": result.x.tolist(), "mse_initial": history[0]["mse"],
              "mse_final": float(result.fun), "wall_s": time.perf_counter() - started,
              "next_measurement_seed": seed + len(evaluated),
              "final_measurement_seed": evaluated[tuple(result.x)], "history": history}
    write_json(out / f"{label}_optimization.json", record)
    print(f"{label}: {len(y)} rows, {result.nfev} evaluations, "
          f"MSE {history[0]['mse']:.6g} -> {result.fun:.6g} ({record['wall_s']:.1f}s)", flush=True)
    return result.x, F, record


def quantum(data, out, artifacts, shots, seed):
    label = f"zz_atan_L1_{'shots1024' if shots else 'exact'}_seed{seed}"
    X, y = data["Water", "train"]
    params = list(ParameterGrid(QGRID))
    fold_scores = np.zeros((len(params), 5))
    fits = {}
    for fold, (tr, va) in enumerate(KFold(5).split(X)):
        theta, Ftr, opt = optimize(X[tr], y[tr], shots, seed, f"{label}_fold{fold}", artifacts)
        fits[f"fold{fold}"] = {k: v for k, v in opt.items() if k != "history"}
        Fva = measure(expectations(angles(X[va]), theta), shots, opt["next_measurement_seed"])
        for i, p in enumerate(params):
            model = SVR(gamma=1, **p).fit(Ftr, y[tr])
            fold_scores[i, fold] = r2_score(y[va], model.predict(Fva))
    scores = fold_scores.mean(axis=1)
    best = params[int(np.argmax(scores))]
    pd.DataFrame([{**p, "cv_r2": scores[i], **{f"fold{f}": fold_scores[i, f] for f in range(5)}}
                  for i, p in enumerate(params)]).to_csv(out / f"{label}_cv.csv", index=False)
    theta, Ftr, opt = optimize(X, y, shots, seed, f"{label}_full", artifacts)
    fits["full"] = {k: v for k, v in opt.items() if k != "history"}
    write_json(out / f"{label}_fits.json", fits)
    model = SVR(gamma=1, **best).fit(Ftr, y)
    assert model.fit_status_ == 0, "Final SVR did not converge"
    # Validate the accelerated objective on the full training matrix, using the
    # same precomputed-kernel libsvm route as native QSVR. This is independent
    # of the small circuit/solver check in validate().
    K = squareform(pdist(Ftr, "sqeuclidean"))
    np.negative(K, out=K)
    np.exp(K, out=K)
    reference = SVR(kernel="precomputed", max_iter=50).fit(K, y).predict(K)
    direct = SVR(gamma=1, max_iter=50).fit(Ftr, y).predict(Ftr)
    difference = float(np.max(np.abs(reference - direct)))
    assert difference < 1e-6, difference
    write_json(artifacts / f"{label}_solver_check.json", {
        "n_train": len(y), "precomputed_mse": mean_squared_error(y, reference),
        "direct_mse": mean_squared_error(y, direct), "max_prediction_difference": difference,
        "final_svr_fit_status": int(model.fit_status_), "final_support_vectors": len(model.support_)})
    del K, reference, direct
    # The training script predicts water before saving, advancing the executor.
    xt, yt = data["Water", "test"]
    Ftest = measure(expectations(angles(xt), theta), shots, opt["next_measurement_seed"])
    train_script_r2 = float(r2_score(yt, model.predict(Ftest)))
    joblib.dump({"model": model, "theta": theta, "training_features": Ftr,
                 "shots": shots, "seed": seed, "best_params": best}, artifacts / f"{label}.joblib")
    # Native ProjectedQuantumKernel.__reduce__ rebuilds its QNN with empty cache.
    # Transferability_Test refits this reloaded model; KernelOptimizer keeps theta.
    reload_seed = opt["next_measurement_seed"] + 1
    Freload = measure(expectations(angles(X), theta), shots, reload_seed)
    transfer_model = SVR(gamma=1, **best).fit(Freload, y)
    joblib.dump({"model": transfer_model, "theta": theta, "training_features": Freload,
                 "shots": shots, "measurement_seed": reload_seed},
                artifacts / f"{label}_transfer_refit.joblib")
    rows = []
    for mol in MOLS:
        xt, yt = data[mol, "test"]
        xa, ya = data[mol, "train"]
        # Each target script starts by loading the same saved model. The all-data
        # diagnostic is a separate fresh evaluation, not a slice of the test run.
        Ft = measure(expectations(angles(xt), theta), shots, reload_seed + 1)
        pred = transfer_model.predict(Ft)
        xf = xt if mol == "Water" else np.vstack([xt, xa])
        yf = yt if mol == "Water" else np.concatenate([yt, ya])
        Ff = measure(expectations(angles(xf), theta), shots, reload_seed + 1)
        pred_all = transfer_model.predict(Ff)
        prediction_file = artifacts / f"{label}_{mol}_predictions.npz"
        np.savez_compressed(prediction_file,
                            y_test=yt, predicted_test=pred, y_all=yf, predicted_all=pred_all)
        with np.load(prediction_file) as saved:
            assert np.array_equal(pred, saved["predicted_test"])
            assert np.array_equal(pred_all, saved["predicted_all"])
        rows.append({"model": label, "molecule": mol, "C": best["C"], "epsilon": best["epsilon"],
                     "gamma": 1, "cv_r2": float(scores.max()),
                     "test_r2": r2_score(yt, pred), "all_r2": r2_score(yf, pred_all),
                     "n_test": len(yt), "n_all": len(yf)})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{label}_scores.csv", index=False)
    write_json(out / f"{label}_selection.json", {"best_params": best, "cv_r2": float(scores.max()),
        "train_script_water_test_r2": train_script_r2, "transfer_reload_seed": reload_seed,
        "mean_transfer_test_r2": float(df.iloc[1:].test_r2.mean()),
        "mean_transfer_all_r2": float(df.iloc[1:].all_r2.mean()),
        "sampling": "Independent binomial draws" if shots else "Exact expectations",
        "protocol": "Full training data, fold-local optimization, saved scaling, "
        "and separately recorded test-split and full-population evaluation."})
    print(df.to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "Experiments/results/water_reproduction")
    ap.add_argument("--artifacts", type=Path, default=ROOT / "paper_results/experiments/water_reproduction")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=False)
    warnings.filterwarnings("ignore", category=UserWarning)
    data = {(mol, split): load_water_scaled(ROOT / "data", mol, split)
            for mol in MOLS for split in ["train", "test"]}
    files = [Path(__file__), ROOT / "QKernel_Methods_Test.py", ROOT / "Transferability_Test.py",
             ROOT / "Training_Functions.py", ROOT / "FunctionEncoder.py", ROOT / "Circuit_Classes.py",
             ROOT / "EncodingCircuit.py", ROOT / "Experiments/qk_common.py", ROOT / "Experiments/fast_pqk.py"]
    for mol in MOLS:
        base = ROOT / "data/training/STO-3G" / mol / "scaled"
        files.extend(base / f for f in ["full_feature_train.bin", "full_feature_test.bin",
                                       "feature_scaler.bin", "t2_scaler.bin"])
    manifest = {"source": "Water", "basis": "STO-3G", "seed": args.seed,
        "quantum_configuration": "ZZ_map-atan-L1", "training_samples": len(data["Water", "train"][1]),
        "cv": "KFold(5, shuffle=False), full training data, fold-local COBYLA",
        "quantum_grid": QGRID, "classical_grid": CGRID, "quantum_gamma": 1,
        "optimizer": "COBYLA(maxiter=100, rhobeg=1); SVR(C=1,epsilon=.1,max_iter=50) training MSE",
        "versions": {p: importlib.metadata.version(p) for p in
                     ["numpy", "scipy", "scikit-learn", "qiskit", "qiskit-aer", "squlearn"]},
        "sha256": {str(p.relative_to(ROOT)): sha256(p) for p in files},
        "scope": "One retrospectively selected ZZFeatureMap/arctan/L1 configuration; "
                 "independent 1024-shot evaluations."}
    write_json(args.out / "manifest.json", {k: v for k, v in manifest.items() if k != "sha256"})
    write_json(args.artifacts / "run_manifest.json", manifest)
    validate(data, args.artifacts)
    classical(data, args.out, args.artifacts, args.workers)
    quantum(data, args.out, args.artifacts, 1024, args.seed)


if __name__ == "__main__":
    main()
