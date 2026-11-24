"""Compatibility adapter for cloning the paper's explicitly labelled circuits."""

from copy import deepcopy

from squlearn.encoding_circuit import QiskitEncodingCircuit


class CloneableQiskitEncodingCircuit(QiskitEncodingCircuit):
    """Preserve feature/parameter labels when scikit-learn clones an estimator.

    sQUlearn 0.8.x's QiskitEncodingCircuit.get_params() returns only the circuit,
    dropping the explicit labels (X, Phi, P, theta) and mode. Reconstructing that
    object with its defaults fails automatic feature/parameter identification.
    This object is a fixed circuit specification, with no fitted model state;
    copying it preserves its semantics and gives every clone its own circuit.
    Model/executor cloning and GridSearchCV's existing behavior are unchanged.
    """

    def __sklearn_clone__(self):
        return deepcopy(self)
