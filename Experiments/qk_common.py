"""Shared helpers: rebuild the paper's PQK exactly as QKernel_Methods_Test.py does,
and load the Water-scaled 5-feature data exactly as the training/transfer scripts do.

The repository root (for FunctionEncoder.py / Circuit_Classes.py) and the data root are
passed in explicitly; nothing is imported from the current directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

TOP5 = np.array([20, 21, 24, 23, 22][::-1])  # same column order as the paper scripts


def add_repo_to_path(repo: Path) -> None:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def build_encoding_circuit(repo: Path, circuit: str, degree, layers: int, num_qubits: int = 5,
                           num_features: int = 5):
    """The paper's encoding circuit (QiskitEncodingCircuit), as built in QKernel_Methods_Test.py."""
    add_repo_to_path(repo)
    from qiskit.circuit import ParameterVector
    from squlearn import Executor
    from squlearn.encoding_circuit import (ChebyshevPQC, HighDimEncodingCircuit, HubregtsenEncodingCircuit,
                                           MultiControlEncodingCircuit, ParamZFeatureMap, QiskitEncodingCircuit,
                                           YZ_CX_EncodingCircuit)
    from squlearn.kernel import ProjectedQuantumKernel
    from FunctionEncoder import FunctionEncoder
    from EncodingCircuit import CloneableQiskitEncodingCircuit
    from Circuit_Classes import ZZ_featuremap

    X = ParameterVector("X", num_features)
    if circuit == "YZ_CX":
        enc = YZ_CX_EncodingCircuit(num_qubits, num_features, layers)
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    elif circuit == "Chebyshev":
        enc = ChebyshevPQC(num_qubits, num_features, layers, nonlinearity="arctan")
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    elif circuit == "ZZ_map":
        qc = ZZ_featuremap(num_features, layers, 0, "linear").get_circuit()
    elif circuit == "ParamZ":
        enc = ParamZFeatureMap(num_qubits, num_features, layers, entangling=True)
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    elif circuit == "HighDim":
        enc = HighDimEncodingCircuit(num_features, num_qubits, num_layers=layers)
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    elif circuit == "Hubregtsen":
        enc = HubregtsenEncodingCircuit(num_features, num_qubits, layers)
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    elif circuit == "MultiControl":
        enc = MultiControlEncodingCircuit(num_features, num_qubits, layers)
        qc = enc.get_circuit(X, ParameterVector("P", enc.num_parameters))
    else:
        raise ValueError(circuit)

    # Encoding functions, identical to degree_relation_function / atan_relation_function
    def degree_fn(expression, num_parameters):
        params = ParameterVector("Phi", num_parameters + degree + 1)[num_parameters:]
        return sum(params[i] * (expression ** i) for i in range(degree + 1))

    def atan_fn(expression, num_parameters):
        param = ParameterVector("Phi", num_parameters + 1)[num_parameters:]
        return param[0] * np.arctan(expression)

    if degree == "atan":
        qc = FunctionEncoder(qc, atan_fn, False)
    elif int(degree) > 0:
        qc = FunctionEncoder(qc, degree_fn, False)
    return CloneableQiskitEncodingCircuit(qc, parameter_label=["Phi", "P", "θ"], feature_label="X")


def untrained_parameters(enc_circ, parameter_seed: int = 0):
    """Circuit parameters of an untrained sQUlearn kernel: initial_parameters=None and the default
    parameter_seed=0 give enc_circ.generate_initial_parameters(0) (kernel_matrix_base.py)."""
    return enc_circ.generate_initial_parameters(parameter_seed)


def build_pqk(repo: Path, circuit: str, degree, layers: int, shots: int | None, seed: int,
              measurement: str = "XYZ", gamma: float = 1.0, num_qubits: int = 5, num_features: int = 5):
    """Mirror of quantum_kernel_method() in QKernel_Methods_Test.py (TRAINABLE=False)."""
    from squlearn import Executor
    from squlearn.kernel import ProjectedQuantumKernel

    enc_circ = build_encoding_circuit(repo, circuit, degree, layers, num_qubits, num_features)
    if shots is None:
        executor = Executor("statevector_simulator", seed=seed)
    else:
        executor = Executor("qasm_simulator", shots=shots, seed=seed)
    # initial_parameters=None + parameter_seed=0 (sQUlearn default) is what the paper scripts used
    return ProjectedQuantumKernel(enc_circ, executor=executor, measurement=measurement, gamma=gamma)


def load_water_scaled(data_root: Path, molecule: str, split: str, basis: str = "STO-3G"):
    """Return (X[:, top5], y) for `split` in {"train", "test"} rescaled to the Water scalers.

    Same transformation as QKernel_Methods_Test.py / Transferability_Test.py.
    """
    base = data_root / "training" / basis
    d = joblib.load(base / molecule / "scaled" / f"full_feature_{split}.bin")
    X, y = d["X"], d["y"]
    w_fs = joblib.load(base / "Water" / "scaled" / "feature_scaler.bin")
    w_ts = joblib.load(base / "Water" / "scaled" / "t2_scaler.bin")
    m_fs = joblib.load(base / molecule / "scaled" / "feature_scaler.bin")
    m_ts = joblib.load(base / molecule / "scaled" / "t2_scaler.bin")
    Xs = w_fs.transform(m_fs.inverse_transform(X))[:, TOP5]
    ys = w_ts.transform(m_ts.inverse_transform(np.asarray(y).reshape(-1, 1))).ravel()
    return np.asarray(Xs), np.asarray(ys)
