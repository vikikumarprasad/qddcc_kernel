"""Regenerate the published Figure 1(b) SHAP panel from preserved values."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "qddcc-cache"),
)

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Graphs" / "figure1b_published_shap.csv"
OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)

FEATURE_LABELS = {
    "diag": r"\boldmath$\braket{ij \Vert ab}$",
    "orbdiff": (
        r"\boldmath$\varepsilon_i + \varepsilon_j - "
        r"\varepsilon_a - \varepsilon_b$"
    ),
    "doublecheck": r"\textbf{Boolean}(\boldmath$a=b$)",
    "t2start": r"\boldmath$t^{ab}_{ij}(\mathrm{MP2})$",
    "t2mag": r"\boldmath$\log_{10}|t^{ab}_{ij}(\mathrm{MP2})|$",
}


def read_values() -> list[tuple[str, float]]:
    with SOURCE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values = []
    for row in rows:
        feature = row["feature"]
        if feature not in FEATURE_LABELS:
            raise ValueError(f"Unknown Figure 1(b) feature: {feature}")
        values.append((feature, float(row["mean_absolute_shap"])))
    return values


def main() -> None:
    # The preserved values reproduce the published bars without rerunning a
    # version-sensitive SHAP calculation. Shap_Graphs.py remains available for
    # a fresh analysis and writes a separately named diagnostic figure.
    values = sorted(read_values(), key=lambda item: item[1])

    plt.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}\usepackage{braket}",
            "font.weight": "bold",
            "axes.labelweight": "bold",
        }
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.barh(
        [FEATURE_LABELS[feature] for feature, _ in values],
        [value for _, value in values],
        color="#1e88e5",
    )
    ax.set_xlabel(
        r"\boldmath\textbf{Mean $\Vert$ SHAP Value $\Vert$}",
        fontsize=12,
        fontweight="bold",
    )
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: rf"\textbf{{{value:.3f}}}")
    )
    ax.tick_params(axis="x", labelsize=11, width=1.25)
    ax.tick_params(axis="y", labelsize=10, width=1.25)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.25)
    ax.spines["bottom"].set_linewidth(1.25)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "ccsd_shap_summary_bar.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(f"Wrote published SHAP panel to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
