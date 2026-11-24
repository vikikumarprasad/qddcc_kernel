"""Balanced cc-pVDZ learning curves, with one complete artifact per model.

Example: OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python Experiments/run_water_learning_curve.py
The default fast search uses at most 5000 selected training amplitudes for CV,
then refits on the full selected sample. --cv-size 0 uses full-sample CV.
The experiment uses the supplied water training and test split.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

from batched_pqk import BatchedPQK
from qk_common import TOP5, untrained_parameters

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Experiments/results/water_learning_curve"
WORK = ROOT / "paper_results/experiments/water_learning_curve"
FAMILIES = ["Classical", "Chebyshev", "MultiControl", "Hubregtsen"]


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2**20), b""): h.update(b)
    return h.hexdigest()


def prepare():
    WORK.mkdir(parents=True, exist_ok=True)
    data = ROOT / "data/training/cc-pVDZ/Water/scaled"
    tr, te = (joblib.load(data / f"full_feature_{s}.bin") for s in ("train", "test"))
    fs, ts = (joblib.load(data / n) for n in ("feature_scaler.bin", "t2_scaler.bin"))
    raw = fs.inverse_transform(tr["X"])
    prob = np.abs(raw[:, 23] * (raw[:, 8] + raw[:, 12])).astype(float)
    np.save(WORK / "probabilities.npy", prob / prob.sum())
    for split, d in (("train", tr), ("test", te)):
        np.save(WORK / f"X_{split}.npy", np.asarray(d["X"][:, TOP5], float))
        np.save(WORK / f"y_{split}.npy", np.asarray(d["y"], float).ravel())
    raw_test = fs.inverse_transform(te["X"]).astype(float)
    g = raw_test[:, 22].reshape(40, 4, 4, 19, 19)
    weights = (2*g - g.swapaxes(-1, -2)).reshape(40, -1)
    np.save(WORK / "energy_weights.npy", weights)
    np.save(WORK / "mp2_amplitudes.npy", raw_test[:, 23].reshape(40, -1))
    expected = ts.inverse_transform(np.asarray(te["y"]).reshape(-1, 1)).ravel()
    np.save(WORK / "expected_raw.npy", expected)
    info = {"test_names": list(te["names"]), "n_test": len(expected), "amplitudes_per_conformer": 5776,
            "train_names": list(tr["names"]), "y_scale": float(ts.scale_[0]), "y_min": float(ts.min_[0]),
            "inputs": {str(p.relative_to(ROOT)): digest(p) for p in data.glob("*.bin")}}
    (WORK / "data_manifest.json").write_text(json.dumps(info, indent=2))
    # Check circuit evaluator against independently generated Qiskit statevectors.
    from qiskit.quantum_info import Statevector
    from fast_pqk import rdm1_expectations
    validations = []
    for family in FAMILIES[1:]:
        simulator = BatchedPQK(ROOT, family, 0, 1)
        theta = untrained_parameters(simulator.encoding)
        xx = np.asarray(tr["X"][:8, TOP5], float)
        reference = np.vstack([rdm1_expectations(
            Statevector.from_instruction(simulator.encoding.get_circuit(x, theta)).data[None, :], 5)
            for x in xx])
        error = float(np.max(np.abs(reference - simulator(xx, theta))))
        if error > 1e-12: raise ValueError(f"Circuit validation failed: {family}, {error}")
        validations.append({"family": family, "max_error": error, "parameters": theta.tolist()})
        for split, d in (("train", tr), ("test", te)):
            path = WORK / f"F_{family}_{split}.npy"
            if not path.exists():
                np.save(path, simulator(np.asarray(d["X"][:, TOP5], float), theta))
                print("features", family, split, flush=True)
    (WORK / "circuit_validation.json").write_text(json.dumps(validations, indent=2))
    return info


def energy_references(info):
    path = OUT / "energy_references.csv"
    if path.exists(): return
    import psi4
    psi4.core.be_quiet()
    psi4.set_memory("2 GB")
    psi4.set_num_threads(1)
    weights = np.load(WORK / "energy_weights.npy")
    mp2 = np.load(WORK / "mp2_amplitudes.npy")
    rows = []
    for k, name in enumerate(info["test_names"]):
        p = ROOT / f"data/geometries/training/water/water{int(name.removeprefix('Water'))+2}.xyz"
        psi4.core.clean()
        mol = psi4.geometry(p.read_text())
        mol.reset_point_group("c1")
        psi4.set_options({"basis": "cc-pVDZ", "scf_type": "pk", "mp2_type": "conv", "reference": "rhf",
                          "freeze_core": True, "e_convergence": 1e-10, "d_convergence": 1e-10})
        psi4.energy("mp2", molecule=mol)
        reference_mp2 = psi4.variable("MP2 CORRELATION ENERGY")
        archived_mp2 = float(weights[k] @ mp2[k])
        if abs(reference_mp2 - archived_mp2) > 2e-6:
            raise ValueError(f"Geometry/amplitude mismatch {name}: {reference_mp2}, {archived_mp2}")
        psi4.energy("ccsd", molecule=mol)
        ccsd = psi4.variable("CCSD CORRELATION ENERGY")
        rows.append({"name": name, "geometry": str(p.relative_to(ROOT)), "geometry_sha256": digest(p),
                     "mp2_archive": archived_mp2, "mp2_psi4": reference_mp2, "ccsd_reference": ccsd})
        print("reference", name, archived_mp2, ccsd, flush=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_one(family, fraction, rep, cv_size):
    name = f"{family}_{fraction:.2f}_{rep}"
    artifact, meta = WORK / f"{name}.npz", WORK / f"{name}.json"
    if artifact.exists() and meta.exists():
        old = json.loads(meta.read_text())
        if old["prediction_sha256"] != digest(artifact): raise ValueError(f"Changed artifact {name}")
        return old
    t0 = time.perf_counter()
    seed = 42 + rep
    y = np.load(WORK / "y_train.npy")
    yt = np.load(WORK / "y_test.npy")
    n = int(len(y) * fraction)
    indices = np.random.RandomState(seed).choice(len(y), size=n, replace=False,
                                               p=np.load(WORK / "probabilities.npy"))
    if family == "Classical":
        x = np.load(WORK / "X_train.npy", mmap_mode="r")[indices]
        xv = np.load(WORK / "X_test.npy", mmap_mode="r")
        grid = {"C": [.1, .5, 1, 5, 10], "epsilon": [.1, .05, .03, .08],
                "gamma": ["auto", 1, .1, .01, .05, .001]}
    else:
        # Independent measurements for each Pauli, as in the 1024-shot PQK.
        # Cache a fixed, label-independent realization for the CV/refit pipeline.
        rng = np.random.RandomState(1000 + seed)
        f = np.load(WORK / f"F_{family}_train.npy", mmap_mode="r")[indices]
        fv = np.load(WORK / f"F_{family}_test.npy", mmap_mode="r")
        x = 2*rng.binomial(1024, np.clip((f+1)/2, 0, 1))/1024 - 1
        xv = 2*rng.binomial(1024, np.clip((fv+1)/2, 0, 1))/1024 - 1
        grid = {"C": [.1, .25, .5, 1, 5, 10], "epsilon": [.1, .05, .03, .08], "gamma": [1.]}
    # Fixed subset shared across models for this fraction and repetition.
    ci = (np.random.RandomState(10000 + seed).choice(n, min(cv_size, n), replace=False)
          if cv_size else np.arange(n))
    search = GridSearchCV(SVR(cache_size=128), grid, cv=5, scoring="r2", n_jobs=1, refit=False, error_score="raise")
    search.fit(x[ci], y[indices][ci])
    pd.DataFrame(search.cv_results_).to_csv(WORK / f"cv_{name}.csv", index=False)
    model = SVR(**search.best_params_, cache_size=512)
    model.fit(x, y[indices])
    if model.fit_status_ != 0: raise ValueError(f"SVR did not converge: {name}")
    pred = model.predict(xv)
    if pred.shape != (231040,) or not np.isfinite(pred).all(): raise ValueError(f"Invalid predictions: {name}")
    info = json.loads((WORK / "data_manifest.json").read_text())
    raw = (pred - info["y_min"]) / info["y_scale"]
    reference = pd.read_csv(OUT / "energy_references.csv")
    energies = np.einsum("ij,ij->i", raw.reshape(40, 5776), np.load(WORK / "energy_weights.npy"))
    row = {"family": family, "fraction": fraction, "repetition": rep, "sample_seed": seed,
           "n_train": n, "n_cv": len(ci), "n_test": len(pred), "r2": r2_score(yt, pred),
           "energy_mae_mEh": float(1000*np.mean(np.abs(energies - reference.ccsd_reference))),
           "cv_r2": search.best_score_, "parameters": search.best_params_, "fit_status": int(model.fit_status_),
           "elapsed_s": time.perf_counter()-t0}
    with open(artifact.with_suffix(".tmp"), "wb") as f:
        np.savez_compressed(f, predicted=raw, expected=np.load(WORK / "expected_raw.npy"),
                            train_indices=indices, cv_indices=ci, energy=energies,
                            names=np.asarray(info["test_names"]), t2_num=np.tile(np.arange(5776), 40))
    os.replace(artifact.with_suffix(".tmp"), artifact)
    row["prediction_sha256"] = digest(artifact)
    meta.write_text(json.dumps(row, indent=2))
    joblib.dump(model, WORK / f"{name}.joblib", compress=3)
    print(row, flush=True)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cv-size", type=int, default=5000)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--stage", choices=["prepare", "run", "all"], default="all")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {"cv_size": args.cv_size, "repetitions": args.repetitions, "fractions": [i/100 for i in range(1,8)],
                "families": FAMILIES, "circuit": "5 qubits, one layer, direct encoding, fixed parameter seed 0",
                "sampling": "weighted amplitude selection; independent seeds 42+r, paired across models",
                "shot_protocol": "1024 independent binomial Pauli samples; fixed cached realization per model run",
                "gamma": "classical grid of Table S1; PQK fixed 1", "cv": "five unshuffled amplitude folds on training subset",
                "refit": "full selected training sample", "energy": "zero predicted singles; no CCSD iterations; converged CCSD reference",
                "split": "cc-pVDZ Water training and test split supplied in the dataset"}
    m = OUT / "protocol.json"
    if m.exists() and json.loads(m.read_text()) != manifest: raise ValueError("Existing output uses a different protocol")
    m.write_text(json.dumps(manifest, indent=2))
    if args.stage in ("prepare", "all"):
        info = prepare()
        energy_references(info)
    if args.stage in ("run", "all"):
        jobs = [(family, fraction, rep, args.cv_size) for fraction in manifest["fractions"]
                for rep in range(args.repetitions) for family in FAMILIES]
        rows = Parallel(n_jobs=args.jobs)(delayed(run_one)(*j) for j in jobs)
        df = pd.DataFrame(rows)
        assert len(df) == 28 * args.repetitions and (df.n_test == 231040).all()
        df.to_csv(OUT / "runs.csv", index=False)
        summary = df.groupby(["fraction", "n_train", "family"]).agg(
            repetitions=("r2", "size"), r2_mean=("r2", "mean"), r2_sem=("r2", "sem"),
            energy_mean_mEh=("energy_mae_mEh", "mean"), energy_sem_mEh=("energy_mae_mEh", "sem"))
        summary.to_csv(OUT / "summary.csv")
        print(summary.to_string(), flush=True)


if __name__ == "__main__": main()
