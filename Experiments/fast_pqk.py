"""Exact (statevector) projected-quantum-kernel features, computed without sQUlearn's executor.

The circuit is the paper's sQUlearn encoding circuit (same construction, same untrained
parameters drawn with sQUlearn's parameter_seed=0); only the evaluation of the single-qubit
<X>, <Y>, <Z> expectation values is done directly with qiskit.quantum_info.Statevector.
`verify()` checks the result against sQUlearn's own statevector evaluation.

Optional finite-shot emulation: each expectation value e is replaced by the mean of `shots`
+-1 outcomes, i.e. 2*Binomial(shots, (1+e)/2)/shots - 1 (the marginal distribution of a
`shots`-shot estimate of a single-qubit Pauli expectation).
This matches the separate single-Pauli measurements used by the installed sQUlearn
0.8.4/Qiskit 1.0.2 BackendEstimator. Draws are independent across observables;
callers control feature caching and remeasurement during cross-validation.
"""
from __future__ import annotations

import os
from functools import partial
from multiprocessing import get_context
from pathlib import Path

import numpy as np

_STATE = {}


def _standardize(qc):
    """Convert bare FunctionEncoder instructions to standard Qiskit gate classes."""
    from qiskit.circuit import Gate, QuantumCircuit
    from qiskit.circuit.library import get_standard_gate_name_mapping

    mapping = get_standard_gate_name_mapping()
    new = QuantumCircuit(qc.num_qubits)
    for ci in qc.data:
        op = ci.operation
        if op.name == "barrier":  # no effect on the statevector
            continue
        if not isinstance(op, Gate):
            if op.name not in mapping:
                raise ValueError(f"Cannot map instruction {op.name!r}")
            op = type(mapping[op.name])(*op.params)
        new.append(op, [qc.find_bit(q).index for q in ci.qubits])
    return new


def _init(repo, circuit, degree, layers, params_override=None):
    import warnings

    warnings.filterwarnings("ignore")
    from qiskit.circuit import ParameterVector
    from qk_common import build_encoding_circuit, untrained_parameters

    enc = build_encoding_circuit(Path(repo), circuit, degree, layers)
    params = np.asarray(untrained_parameters(enc) if params_override is None else params_override, dtype=float)
    xv = ParameterVector("xf", enc.num_features)
    pv = ParameterVector("pf", enc.num_parameters)
    qc = enc.get_circuit(xv, pv)
    qc = qc.assign_parameters(dict(zip(pv, params)))
    _STATE.clear()
    _STATE.update(key=(circuit, str(degree), layers, _pkey(params_override)), qc=_standardize(qc), xv=xv, n=qc.num_qubits, params=params)


def _statevectors(X):
    from qiskit.quantum_info import Statevector

    qc, xv = _STATE["qc"], _STATE["xv"]
    out = np.empty((len(X), 2 ** _STATE["n"]), dtype=complex)
    for i, x in enumerate(X):
        out[i] = Statevector(qc.assign_parameters(dict(zip(xv, x)))).data
    return out


def _pkey(params):
    return None if params is None else hash(np.asarray(params, dtype=float).tobytes())


def _worker(chunk, repo, circuit, degree, layers, params=None):
    if _STATE.get("key") != (circuit, str(degree), layers, _pkey(params)):
        _init(repo, circuit, degree, layers, params)
    return _statevectors(chunk)


def rdm1_expectations(psi: np.ndarray, n: int) -> np.ndarray:
    """psi: (B, 2**n) little-endian statevectors -> (B, 3n) [<X_q>, <Y_q>, <Z_q>] for q = 0..n-1."""
    B = psi.shape[0]
    t = psi.reshape((B,) + (2,) * n)
    cols = []
    for q in range(n):
        m = np.moveaxis(t, n - q, 1).reshape(B, 2, -1)  # qubit q is tensor axis n - q (axis 0 = batch)
        rho01 = np.einsum("bk,bk->b", m[:, 0, :], m[:, 1, :].conj())
        rho00 = np.einsum("bk,bk->b", m[:, 0, :], m[:, 0, :].conj()).real
        rho11 = np.einsum("bk,bk->b", m[:, 1, :], m[:, 1, :].conj()).real
        cols += [2 * rho01.real, -2 * rho01.imag, rho00 - rho11]
    return np.stack(cols, axis=1)


def projected_features(repo, circuit, degree, layers, X, n_jobs=None, chunk=2000, shots=None, shot_seed=0,
                       params=None):
    X = np.asarray(X, dtype=float)
    n_jobs = n_jobs or os.cpu_count()
    chunks = [X[i:i + chunk] for i in range(0, len(X), chunk)]
    fn = partial(_worker, repo=str(repo), circuit=circuit, degree=degree, layers=layers, params=params)
    if n_jobs == 1:
        psi = np.vstack([fn(c) for c in chunks])
    else:
        with get_context("fork").Pool(n_jobs) as pool:
            psi = np.vstack(pool.map(fn, chunks))
    n = int(np.log2(psi.shape[1]))
    F = rdm1_expectations(psi, n)
    if shots:
        rng = np.random.default_rng(shot_seed)
        p = np.clip((1 + F) / 2, 0, 1)
        F = 2 * rng.binomial(shots, p) / shots - 1
    return F


def verify(repo, circuit, degree, layers, X, tol=1e-8):
    """Compare pairwise squared distances (invariant to column order/sign) with sQUlearn."""
    import warnings

    warnings.filterwarnings("ignore")
    from scipy.spatial.distance import pdist
    from qk_common import build_pqk

    k = build_pqk(Path(repo), circuit, degree, layers, shots=None, seed=0)
    F_ref = np.asarray(k.evaluate_qnn(np.asarray(X)))
    F_new = projected_features(repo, circuit, degree, layers, X, n_jobs=1)
    d_ref, d_new = pdist(F_ref, "sqeuclidean"), pdist(F_new, "sqeuclidean")
    err = float(np.max(np.abs(d_ref - d_new)))
    return err, err < tol, F_ref.shape, F_new.shape
