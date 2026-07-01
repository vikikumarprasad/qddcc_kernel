"""Generate a source-backed text montage of the recoverable encoding circuits."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
from qiskit.circuit import ParameterVector, QuantumCircuit

# sQUlearn currently imports a deprecated PennyLane compatibility alias. It
# does not affect Qiskit circuit construction and cannot be fixed here.
warnings.filterwarnings(
    "ignore",
    message="Observable is deprecated and will be removed.*",
    module=r"pennylane\.operation",
)

from squlearn.encoding_circuit import (
    ChebyshevPQC,
    HighDimEncodingCircuit,
    HubregtsenEncodingCircuit,
    MultiControlEncodingCircuit,
    ParamZFeatureMap,
    YZ_CX_EncodingCircuit,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Circuit_Classes import ZZ_featuremap


OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)
NUM_QUBITS = 5
NUM_FEATURES = 5
NUM_LAYERS = 1


def squlearn_circuit(encoding: object) -> QuantumCircuit:
    return encoding.get_circuit(
        ParameterVector("x", NUM_FEATURES),
        ParameterVector("p", encoding.num_parameters),
    )


def circuits() -> list[tuple[str, QuantumCircuit | None]]:
    return [
        (
            "A  ChebyshevPQC",
            squlearn_circuit(
                ChebyshevPQC(
                    NUM_QUBITS,
                    NUM_FEATURES,
                    NUM_LAYERS,
                    nonlinearity="arctan",
                )
            ),
        ),
        (
            "B  YZ-CX",
            squlearn_circuit(
                YZ_CX_EncodingCircuit(NUM_QUBITS, NUM_FEATURES, NUM_LAYERS)
            ),
        ),
        (
            "C  MultiControl",
            squlearn_circuit(
                MultiControlEncodingCircuit(NUM_FEATURES, NUM_QUBITS, NUM_LAYERS)
            ),
        ),
        (
            "D  HighDim",
            squlearn_circuit(
                HighDimEncodingCircuit(
                    NUM_FEATURES,
                    NUM_QUBITS,
                    num_layers=NUM_LAYERS,
                )
            ),
        ),
        (
            "E  ParamZ",
            squlearn_circuit(
                ParamZFeatureMap(
                    NUM_QUBITS,
                    NUM_FEATURES,
                    NUM_LAYERS,
                    entangling=True,
                )
            ),
        ),
        (
            "F  Hubregtsen",
            squlearn_circuit(
                HubregtsenEncodingCircuit(
                    NUM_FEATURES,
                    NUM_QUBITS,
                    NUM_LAYERS,
                )
            ),
        ),
        ("G  CPMap", None),
        (
            "H  ZZFeatureMap",
            ZZ_featuremap(NUM_FEATURES, NUM_LAYERS, 0, "linear").get_circuit(),
        ),
    ]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 2, figsize=(18, 18))

    for ax, (title, circuit) in zip(axes.flat, circuits()):
        ax.set_title(title, loc="left", fontsize=15, fontweight="bold")
        ax.axis("off")
        if circuit is None:
            text = (
                "Original CPMap implementation unavailable.\n"
                "Request the source circuit from the project owner."
            )
            ax.text(
                0.5,
                0.5,
                text,
                ha="center",
                va="center",
                fontsize=13,
                color="#8b0000",
                bbox={"boxstyle": "round", "facecolor": "#fff2f2", "edgecolor": "#8b0000"},
            )
            continue

        diagram = circuit.draw(output="text", fold=100).single_string()
        ax.text(
            0.01,
            0.94,
            diagram,
            ha="left",
            va="top",
            family="monospace",
            fontsize=7,
        )

    fig.suptitle(
        "Recovered encoding-circuit definitions (5 qubits, L=1)",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = OUTPUT_DIR / "circuit_drawings_reconstructed.png"
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
