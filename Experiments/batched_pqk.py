"""Batched statevectors for the small circuits used by the water reproduction.

Gate angles come from qk_common's actual Qiskit encoding circuit. Gate application
uses NumPy, preserving Qiskit's little-endian qubit order. This evaluator supports
only the gate set below and refuses unsupported operations. Independent Qiskit and
native sQUlearn checks belong to the experiment runner and precede training.
"""
from pathlib import Path

import numpy as np
import sympy as sp
from qiskit.circuit import ParameterExpression, ParameterVector

from fast_pqk import rdm1_expectations
from qk_common import build_encoding_circuit


class BatchedPQK:
    def __init__(self, repo: Path, family: str, degree, layers: int):
        self.encoding = build_encoding_circuit(repo, family, degree, layers)
        self.num_parameters = self.encoding.num_parameters
        self.num_qubits = self.encoding.num_qubits
        self.num_features = self.encoding.num_features
        xv = ParameterVector("xf", self.num_features)
        pv = ParameterVector("pf", self.num_parameters)
        qc = self.encoding.get_circuit(xv, pv)
        symbols = [sp.Symbol(str(p)) for p in [*xv, *pv]]
        indices = np.arange(2 ** self.num_qubits)
        self.gates = []
        for instruction in qc.data:
            op = instruction.operation
            if op.name == "barrier":
                continue
            qubits = [qc.find_bit(q).index for q in instruction.qubits]
            if op.name not in {"h", "x", "cx", "rx", "ry", "rz", "p", "crx", "cry", "crz"}:
                raise ValueError(f"Unsupported gate: {op.name}")
            controlled = op.name.startswith("c")
            assert len(qubits) == (2 if controlled else 1)
            target = qubits[-1]
            mask = (indices & (1 << target)) == 0
            if controlled:
                mask &= (indices & (1 << qubits[0])) != 0
            zero = indices[mask]
            one = zero | (1 << target)
            angle = None
            if op.params:
                assert len(op.params) == 1
                expression = op.params[0]
                expression = expression.sympify() if isinstance(expression, ParameterExpression) else expression
                angle = sp.lambdify(symbols, sp.sympify(expression), modules="numpy", cse=True)
            self.gates.append((op.name[1:] if controlled else op.name, zero, one, angle))

    def _evaluate(self, X, theta):
        psi = np.zeros((len(X), 2 ** self.num_qubits), dtype=complex)
        psi[:, 0] = 1
        arguments = [*X.T, *theta]
        for name, zero, one, angle_function in self.gates:
            a, b = psi[:, zero], psi[:, one]
            if name == "h":
                psi[:, zero] = (a + b) / np.sqrt(2)
                psi[:, one] = (a - b) / np.sqrt(2)
            elif name == "x":
                psi[:, zero], psi[:, one] = b, a
            else:
                angle = np.asarray(angle_function(*arguments), dtype=float).reshape(-1, 1)
                if name == "p":
                    psi[:, one] = np.exp(1j * angle) * b
                elif name == "rz":
                    psi[:, zero] = np.exp(-0.5j * angle) * a
                    psi[:, one] = np.exp(0.5j * angle) * b
                else:
                    c, s = np.cos(angle / 2), np.sin(angle / 2)
                    if name == "rx":
                        psi[:, zero] = c * a - 1j * s * b
                        psi[:, one] = -1j * s * a + c * b
                    elif name == "ry":
                        psi[:, zero] = c * a - s * b
                        psi[:, one] = s * a + c * b
                    else:
                        raise ValueError(name)
        return rdm1_expectations(psi, self.num_qubits)

    def __call__(self, X, theta, batch_size=4096):
        X, theta = np.asarray(X, dtype=float), np.asarray(theta, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError("Feature shape does not match the encoding circuit")
        if theta.shape != (self.num_parameters,):
            raise ValueError("Parameter shape does not match the encoding circuit")
        if not len(X):
            return np.empty((0, 3 * self.num_qubits))
        return np.vstack([self._evaluate(X[i:i + batch_size], theta)
                          for i in range(0, len(X), batch_size)])
