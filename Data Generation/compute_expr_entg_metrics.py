"""Compute expressibility and entangling capability for recoverable circuits.

The recovered notebook used 5,000 random draws without a saved seed. This
script makes that calculation rerunnable by exposing the seed and sample count.
CPMap is intentionally omitted because its source circuit is not distributed in
this repository; use the supplied CSV values for CPMap in paper-figure
reproduction.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product
from math import pi
import os
from pathlib import Path
import sys
import tempfile
from warnings import simplefilter
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))
warnings.filterwarnings("ignore", message="Observable is deprecated.*")

import numpy as np
from qiskit import transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import state_fidelity
from qiskit_aer import Aer
from scipy.special import rel_entr

from squlearn.encoding_circuit import (
    ChebyshevPQC,
    HighDimEncodingCircuit,
    HubregtsenEncodingCircuit,
    MultiControlEncodingCircuit,
    ParamZFeatureMap,
    QiskitEncodingCircuit,
    YZ_CX_EncodingCircuit,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Circuit_Classes import ZZ_featuremap
from FunctionEncoder import FunctionEncoder


NUM_QUBITS = 5
NUM_FEATURES = 5
FUNCTIONS = (0, 1, 2, "ATAN")
LAYERS = (1, 2)


def partial_trace_j(psi: np.ndarray, j: int, b: int, n: int) -> np.ndarray:
    new_psi = np.zeros(2 ** (n - 1), dtype=complex)
    for k in range(2**n):
        binary_rep = format(k, f"0{n}b")
        if int(binary_rep[j - 1]) == b:
            new_binary_rep = binary_rep[: j - 1] + binary_rep[j:]
            new_psi[int(new_binary_rep, 2)] = psi[k]
    return new_psi


def generalized_distance(u: np.ndarray, v: np.ndarray) -> float:
    total_sum = 0.0
    for i in range(len(u)):
        for j in range(len(u)):
            total_sum += np.abs(u[i] * v[j] - u[j] * v[i]) ** 2
    return 0.5 * total_sum


def meyer_wallach_entanglement_measure(psi: np.ndarray, n: int) -> float:
    if len(psi) != 2**n:
        raise ValueError("The size of the state vector does not match 2^n")

    total = 0.0
    for j in range(1, n + 1):
        psi_j_0 = partial_trace_j(psi, j, 0, n)
        psi_j_1 = partial_trace_j(psi, j, 1, n)
        total += generalized_distance(psi_j_0, psi_j_1)
    return (4 / n) * total


def p_harr(left_j: float, right_j: float, num_qubits: int) -> float:
    dimension = 2**num_qubits
    return (1 - left_j) ** (dimension - 1) - (1 - right_j) ** (dimension - 1)


def degree_relation_function(degree: int):
    def relation(expression, num_parameters):
        if degree <= 0:
            return expression

        num_params = degree + 1
        output = 0
        params = ParameterVector("Phi", num_parameters + num_params)[num_parameters:]
        for i in range(num_params):
            output += params[i] * (expression**i)
        return output

    return relation


def atan_relation_function(expression, num_parameters):
    param = ParameterVector("Phi", num_parameters + 1)[num_parameters:]
    return param[0] * expression.arctan()


def base_encoder(name: str, layer: int):
    if name == "ParamZFeatureMap":
        return ParamZFeatureMap(NUM_QUBITS, NUM_FEATURES, layer, entangling=True)
    if name == "HubregtsenEncodingCircuit":
        return HubregtsenEncodingCircuit(NUM_FEATURES, NUM_QUBITS, layer)
    if name == "HighDimEncodingCircuit":
        return HighDimEncodingCircuit(NUM_FEATURES, NUM_QUBITS, num_layers=layer)
    if name == "MultiControlEncodingCircuit":
        return MultiControlEncodingCircuit(NUM_FEATURES, NUM_QUBITS, layer)
    if name == "ChebyshevPQC":
        return ChebyshevPQC(
            NUM_QUBITS,
            NUM_FEATURES,
            layer,
            nonlinearity="arctan",
        )
    if name == "YZ_CX_EncodingCircuit":
        return YZ_CX_EncodingCircuit(NUM_QUBITS, NUM_FEATURES, layer)
    if name == "ZZ_featuremap":
        return ZZ_featuremap(NUM_FEATURES, layer)
    raise ValueError(f"Unsupported circuit: {name}")


def base_circuit(name: str, encoder):
    if name == "ZZ_featuremap":
        return encoder.get_circuit()
    return encoder.get_circuit(
        ParameterVector("X", encoder.num_features),
        ParameterVector("P", encoder.num_parameters),
    )


def function_encoder(name: str, function, layer: int) -> QiskitEncodingCircuit:
    encoder = base_encoder(name, layer)
    circuit = base_circuit(name, encoder)

    if isinstance(function, int):
        relation = degree_relation_function(function)
    else:
        relation = atan_relation_function

    function_circuit = FunctionEncoder(circuit, relation, overwrite_expression=False)
    return QiskitEncodingCircuit(
        function_circuit,
        feature_label="X",
        parameter_label=["Phi", "P", "theta", "θ"],
    )


def compute_expr_entg(
    encoder: QiskitEncodingCircuit,
    n_params: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    backend = Aer.get_backend("statevector_simulator")
    p_harr_bins = [i / 75 for i in range(76)]
    p_harr_hist = [
        p_harr(p_harr_bins[i], p_harr_bins[i + 1], encoder.num_qubits)
        for i in range(75)
    ]

    fidelity = []
    entanglement = []
    for _ in range(n_params):
        data1 = rng.uniform(-pi, pi, encoder.num_features)
        data2 = rng.uniform(-pi, pi, encoder.num_features)
        theta1 = rng.uniform(-pi, pi, encoder.num_parameters)
        theta2 = rng.uniform(-pi, pi, encoder.num_parameters)

        circuit_1 = encoder.get_circuit(data1, theta1)
        circuit_2 = encoder.get_circuit(data2, theta2)

        result1 = backend.run(transpile(circuit_1, backend)).result()
        result2 = backend.run(transpile(circuit_2, backend)).result()
        statevector_1 = result1.get_statevector()
        statevector_2 = result2.get_statevector()

        fidelity.append(state_fidelity(statevector_1, statevector_2))
        simplefilter("ignore", category=DeprecationWarning)
        entanglement.append(
            meyer_wallach_entanglement_measure(statevector_1, encoder.num_qubits)
        )

    weights = np.ones_like(fidelity) / float(len(fidelity))
    p_1_hist = np.histogram(
        fidelity,
        bins=p_harr_bins,
        weights=weights,
        range=[0, 1],
    )[0]
    expressibility = float(np.sum(rel_entr(p_1_hist, p_harr_hist)))
    entanglement_capacity = float(np.sum(entanglement) / len(entanglement))
    return expressibility, entanglement_capacity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=5000,
        help="number of random circuit pairs to sample per circuit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="NumPy random seed for deterministic recomputation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "paper_results" / "water_r2_expr_entg" / "ExprAndEntg_recomputed.csv",
        help="output CSV path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    circuit_names = (
        "HubregtsenEncodingCircuit",
        "HighDimEncodingCircuit",
        "MultiControlEncodingCircuit",
        "ChebyshevPQC",
        "ZZ_featuremap",
        "YZ_CX_EncodingCircuit",
        "ParamZFeatureMap",
    )

    with args.output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        for name, function, layer in product(circuit_names, FUNCTIONS, LAYERS):
            encoder = function_encoder(name, function, layer)
            expr, entg = compute_expr_entg(encoder, args.samples, rng)
            function_label = str(function).upper()
            writer.writerow([f"{name}-{function_label}-{layer}", expr, entg])
            print(
                f"{name} encoded by {function_label} with {layer} layers: "
                f"expr={expr}, entg={entg}"
            )

    print(
        "CPMap rows are not recomputed because CPMap source is not distributed "
        "with this repository."
    )


if __name__ == "__main__":
    main()
