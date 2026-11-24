"""Draw the CPMap C/P gate decompositions from the cited supplementary Fig. S7.

Source: Kautish, Sivakumar and Prasad, "Quantum kernel-based delta-learning
for correcting a semiempirical method in the prediction of reaction barrier
heights", supplementary Figure S7(b,c), page S-8. The blocks appear in
Figure S15 of the Supporting Information.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/qddcc-mplconfig")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ParameterVector

ROOT = Path(__file__).resolve().parents[1]


def blocks():
    theta = ParameterVector("θ", 6)
    circuits = []
    for name, offset in (("C", 0), ("P", 3)):
        qc = QuantumCircuit(2, name=name)
        qc.rz(-3.141592653589793/2, 1)
        qc.cx(1, 0)
        qc.rz(theta[offset], 0)
        qc.ry(theta[offset+1], 1)
        qc.cx(0, 1)
        qc.ry(theta[offset+2], 1)
        if name == "C":
            qc.cx(1, 0)
            qc.rz(3.141592653589793/2, 0)
        circuits.append(qc)
    return circuits


def main():
    out = Path(os.getenv("FIGURE_OUTPUT_DIR", ROOT / "paper_results/regenerated_figures"))
    out.mkdir(parents=True, exist_ok=True)
    circuits = blocks()
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 4.4))
    for ax, qc, title in zip(axes, circuits, ("(a)  C block", "(b)  P block")):
        available = [0, 0]
        ax.hlines([0, 1], -.65, 6.65, color="black", lw=1)
        for q in (0, 1): ax.text(-.85, 1-q, rf"$q_{q}$", ha="right", va="center", fontsize=13)
        for instruction in qc.data:
            op = instruction.operation
            qs = [qc.find_bit(q).index for q in instruction.qubits]
            x = max(available[q] for q in qs)
            for q in qs: available[q] = x + 1
            if op.name == "cx":
                yc, yt = 1-qs[0], 1-qs[1]
                ax.plot([x,x],[yc,yt], color="black", lw=1.2)
                ax.add_patch(Circle((x,yc), .055, color="black", zorder=3))
                ax.add_patch(Circle((x,yt), .12, facecolor="white", edgecolor="black", lw=1.2, zorder=3))
                ax.plot([x-.12,x+.12],[yt,yt], color="black", lw=1, zorder=4)
                ax.plot([x,x],[yt-.12,yt+.12], color="black", lw=1, zorder=4)
            else:
                y = 1-qs[0]
                value = op.params[0]
                if isinstance(value, (float,int)):
                    angle = r"-\pi/2" if value < 0 else r"\pi/2"
                else:
                    angle = rf"\theta_{{{str(value).split('[')[1].split(']')[0]}}}"
                label = rf"$R_{{{op.name[-1].upper()}}}({angle})$"
                ax.add_patch(Rectangle((x-.44,y-.23), .88,.46, facecolor="white", edgecolor="black", lw=1.1, zorder=3))
                ax.text(x,y,label,ha="center",va="center",fontsize=10,zorder=4)
        ax.set(xlim=(-1.25,6.8),ylim=(-.55,1.55),aspect="equal")
        ax.axis("off")
        ax.set_title(title, loc="left", fontsize=12, pad=3)
    fig.tight_layout(pad=.6, h_pad=1.5)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"CPMap_blocks.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    artifact = out
    artifact.mkdir(parents=True, exist_ok=True)
    with (artifact / "cpmap_blocks.qpy").open("wb") as f: qpy.dump(circuits, f)
    (artifact / "cpmap_diagram_provenance.json").write_text(json.dumps({
        "source": "Supplementary_material-1.pdf", "pdf_page": 8, "figure": "S7(b,c)",
        "source_sha256": "10ce4050602dceda3edc76ef12e086c7102ec50d8a7906b97751ec0b28a97b26",
        "source_title": "Quantum kernel-based delta-learning for correcting a semiempirical method in the prediction of reaction barrier heights",
        "qddcc_figure": "Supporting Information Figure S15",
        "block_wire_order": "upper=0, lower=1; gates applied left to right",
        "scope": "Symbolic C/P blocks transcribed from the cited figure."
    }, indent=2))


if __name__ == "__main__": main()
