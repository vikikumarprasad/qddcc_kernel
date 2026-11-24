"""Regenerate Water R2 and expressibility/entanglement figures from CSV data."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)

ENCODING_NAMES = {
    "0": "Original",
    "1": "Degree 1",
    "2": "Degree 2",
    "ATAN": "arctan",
}
ENCODING_ORDER = ("Original", "Degree 1", "Degree 2", "arctan")
ENCODING_COLORS = {
    "Original": "#1f77b4",
    "Degree 1": "#ff7f0e",
    "Degree 2": "#2ca02c",
    "arctan": "#d62728",
}
CIRCUIT_NAMES = {
    "ChebyshevPQC": "ChebyshevPQC",
    "YZ_CX_EncodingCircuit": "YZ_CX",
    "MultiControlEncodingCircuit": "MultiControl",
    "HighDimEncodingCircuit": "HighDim",
    "ParamZFeatureMap": "ParamZ",
    "HubregtsenEncodingCircuit": "Hubregtsen",
    "ZZ_featuremap": "ZZFeatureMap",
    "CPMap": "CPMap",
}
CIRCUIT_ORDER = (
    "ChebyshevPQC",
    "YZ_CX",
    "MultiControl",
    "HighDim",
    "ParamZ",
    "Hubregtsen",
    "ZZFeatureMap",
    "CPMap",
)


def matplotlib_legend_order(
    row_major_labels: list[str],
    ncols: int,
) -> list[str]:
    """Convert a desired visual row order to Matplotlib's column fill order."""
    rows = [
        row_major_labels[index : index + ncols]
        for index in range(0, len(row_major_labels), ncols)
    ]
    return [
        row[column]
        for column in range(ncols)
        for row in rows
        if column < len(row)
    ]


def source_dir() -> Path:
    candidates = []
    if os.getenv("WATER_R2_EXPR_SOURCE_DIR"):
        candidates.append(Path(os.environ["WATER_R2_EXPR_SOURCE_DIR"]))
    candidates.append(ROOT / "paper_results" / "water_r2_expr_entg")

    for candidate in candidates:
        if (candidate / "5-Water-R2.csv").exists() and (
            candidate / "ExprAndEntg.csv"
        ).exists():
            return candidate

    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find 5-Water-R2.csv and ExprAndEntg.csv. Tried:\n" + tried
    )


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    src = source_dir()
    r2_data = pd.read_csv(src / "5-Water-R2.csv", header=None)
    expr_entg = pd.read_csv(src / "ExprAndEntg.csv", header=None)
    r2_data.columns = ["Model_Type", "Circuit_Name", "R2"]
    expr_entg.columns = ["Circuit_Name", "Expr", "Entg"]
    return r2_data, expr_entg


def circuit_display_name(raw_name: str, trained: bool) -> str:
    parts = raw_name.split("-")
    circuit_name = parts[0]
    if circuit_name == "Classical":
        return "Classical"

    names = {
        "CP_map": "CPMap",
        "ZZ_map": "ZZFeatureMap",
        "ParamZ": "ParamZ",
        "Hubregtsen": "Hubregtsen",
        "HighDim": "HighDim",
        "MultiControl": "MultiControl",
        "Chebyshev": "ChebyshevPQC",
        "YZ_CX": "YZ_CX",
    }
    mapped = names.get(circuit_name, circuit_name)
    if trained:
        return f"{mapped}-{parts[1].upper()}-{parts[2]}"
    return f"{mapped}-{parts[1].upper()}-{parts[2]}-{parts[3]}"


def split_circuit_parts(series: pd.Series, max_splits: int) -> pd.DataFrame:
    split = series.str.split("-", n=max_splits, expand=True)
    columns = ["Name_base", "Function", "Layers", "Job_Num"][: split.shape[1]]
    split.columns = columns
    return split


def prepare_expr_data(expr_entg: pd.DataFrame) -> pd.DataFrame:
    data = expr_entg.copy()
    parts = split_circuit_parts(data["Circuit_Name"], 2)
    data = pd.concat([data, parts], axis=1)
    data = data[data["Name_base"] != "ParamZFeatureMapNO"].copy()
    data["Function"] = data["Function"].astype(str).replace(ENCODING_NAMES)
    data["Name_base"] = data["Name_base"].replace(CIRCUIT_NAMES)

    unknown_functions = sorted(set(data["Function"]) - set(ENCODING_ORDER))
    if unknown_functions:
        raise ValueError(f"Unexpected encoding labels: {unknown_functions}")
    unknown_circuits = sorted(set(data["Name_base"]) - set(CIRCUIT_ORDER))
    if unknown_circuits:
        raise ValueError(f"Unexpected circuit labels: {unknown_circuits}")
    return data


def plot_expr_entg(expr_entg: pd.DataFrame) -> None:
    plot_data = prepare_expr_data(expr_entg)
    plt.rcParams.update({"font.size": 10, "font.weight": "heavy"})

    markers = ["o", ".", "^", "D", "*", "p", "s", "X"]
    circuits = [name for name in CIRCUIT_ORDER if name in set(plot_data["Name_base"])]
    functions = [
        name for name in ENCODING_ORDER if name in set(plot_data["Function"])
    ]
    circuit_marker_map = dict(zip(circuits, markers))

    circuit_handles_labels = {}
    function_handles_labels = {}

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    plt.subplots_adjust(wspace=0.1)

    for ax, layer in zip(axes, ["1", "2"]):
        layer_data = plot_data[plot_data["Layers"] == layer]
        for circuit in circuits:
            for function in functions:
                subset = layer_data[
                    (layer_data["Name_base"] == circuit)
                    & (layer_data["Function"] == function)
                ]
                if subset.empty:
                    continue
                ax.scatter(
                    subset["Expr"],
                    subset["Entg"],
                    s=40,
                    marker=circuit_marker_map[circuit],
                    facecolors="None",
                    edgecolors=ENCODING_COLORS[function],
                    linewidths=2,
                    alpha=0.8,
                )

                if circuit not in circuit_handles_labels:
                    handle = ax.scatter(
                        [],
                        [],
                        marker=circuit_marker_map[circuit],
                        s=50,
                        edgecolors="gray",
                        facecolors="none",
                        linewidths=1,
                    )
                    circuit_handles_labels[circuit] = (handle, circuit)

                if function not in function_handles_labels:
                    handle = ax.scatter(
                        [],
                        [],
                        marker="o",
                        s=50,
                        edgecolors=ENCODING_COLORS[function],
                        facecolors="none",
                        linewidths=1,
                    )
                    function_handles_labels[function] = (handle, function)

        ax.set_xscale("log")
        ax.set_xlim(1, 0.001)
        ax.set_ylim(0, 0.9)
        ax.set_title(f"(L={layer})")
        ax.set_xlabel(r"Expressibility, $D_\mathrm{KL}$ (higher expressibility $\rightarrow$)")
        ax.grid(True, which="both", ls="--", linewidth=0.5)

    axes[0].set_ylabel("Entangling capability")

    func_legend_order = matplotlib_legend_order(
        [name for name in ENCODING_ORDER if name in function_handles_labels],
        ncols=2,
    )
    func_handles = [
        function_handles_labels[name][0]
        for name in func_legend_order
    ]
    func_legend = fig.legend(
        func_handles,
        func_legend_order,
        loc="upper left",
        title="Encoding Function",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0, 0),
        borderaxespad=0,
    )

    circuit_legend_order = matplotlib_legend_order(circuits, ncols=3)
    circ_handles = [
        circuit_handles_labels[name][0] for name in circuit_legend_order
    ]
    circuit_legend = fig.legend(
        circ_handles,
        circuit_legend_order,
        loc="upper left",
        title="Circuit Name",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0, 0),
        borderaxespad=0,
    )

    # Treat the two keyed sections as one shared legend: measure their rendered
    # widths, place them next to one another, and center the combined block.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    func_width = func_legend.get_window_extent(renderer).width
    circuit_width = circuit_legend.get_window_extent(renderer).width
    legend_gap = 18 / 72 * fig.dpi
    group_width = func_width + legend_gap + circuit_width
    group_left = (fig.bbox.width - group_width) / (2 * fig.bbox.width)
    legend_top = -0.01
    func_legend.set_bbox_to_anchor(
        (group_left, legend_top),
        transform=fig.transFigure,
    )
    circuit_legend.set_bbox_to_anchor(
        (
            group_left + (func_width + legend_gap) / fig.bbox.width,
            legend_top,
        ),
        transform=fig.transFigure,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "Express_entg.png", dpi=450, bbox_inches="tight")
    plt.close(fig)


def prepare_r2_data(r2_data: pd.DataFrame) -> pd.DataFrame:
    data = r2_data.copy()
    trained_mask = data["Model_Type"].str.contains("True", na=False)
    data["Circuit_Name"] = [
        circuit_display_name(raw, trained)
        for raw, trained in zip(data["Circuit_Name"], trained_mask)
    ]

    trained = data[trained_mask].copy()
    untrained = data[~trained_mask].copy()

    trained = pd.concat(
        [trained, split_circuit_parts(trained["Circuit_Name"], 2)], axis=1
    )
    untrained = pd.concat(
        [untrained, split_circuit_parts(untrained["Circuit_Name"], 3)], axis=1
    )

    trained["Model_Type"] = trained["Model_Type"].str.replace(
        "True-", "", regex=False
    )
    untrained["Model_Type"] = untrained["Model_Type"].str.replace(
        "False-", "", regex=False
    )

    untrained["Circuit_Name"] = untrained["Circuit_Name"].str.replace(
        r"-\d+$", "", regex=True
    )
    untrained = untrained[untrained["Name_base"] != "Classical"].copy()

    grouped = (
        untrained.groupby(
            ["Model_Type", "Name_base", "Function", "Layers"],
            observed=True,
            sort=True,
        )
        .agg(R2_values=("R2", list))
        .reset_index()
    )
    grouped = grouped.explode("R2_values").rename(columns={"R2_values": "R2"})
    grouped["R2"] = grouped["R2"].astype(float)
    grouped["Trained"] = False

    trained["Trained"] = True
    trained["R2"] = trained["R2"].astype(float)

    combined = pd.concat(
        [
            grouped[["Model_Type", "Name_base", "Function", "Layers", "R2", "Trained"]],
            trained[
                ["Model_Type", "Name_base", "Function", "Layers", "R2", "Trained"]
            ],
        ],
        ignore_index=True,
    )
    combined["Function"] = combined["Function"].astype(str).replace({"0": "Original"})
    combined["Layers"] = combined["Layers"].astype(str)
    combined["Encoding_Scheme"] = (
        combined["Name_base"].astype(str) + "  (L=" + combined["Layers"] + ")"
    )

    category_mapping = {
        "Original": {"row_facet": "Top", "col_facet": "Left"},
        "1": {"row_facet": "Top", "col_facet": "Right"},
        "2": {"row_facet": "Bottom", "col_facet": "Left"},
        "ATAN": {"row_facet": "Bottom", "col_facet": "Right"},
    }
    missing = sorted(set(combined["Function"]) - set(category_mapping))
    if missing:
        raise ValueError(f"Unexpected function labels in R2 data: {missing}")

    combined["row_facet"] = combined["Function"].map(
        lambda value: category_mapping[value]["row_facet"]
    )
    combined["col_facet"] = combined["Function"].map(
        lambda value: category_mapping[value]["col_facet"]
    )
    return combined


def plot_water_r2(r2_data: pd.DataFrame) -> None:
    merged_df = prepare_r2_data(r2_data)
    plt.rcParams.update({"font.size": 10, "font.weight": "heavy"})
    sns.set_palette(["#1a80BB", "#ea801C"])

    grid = sns.catplot(
        data=merged_df,
        kind="box",
        x="Encoding_Scheme",
        y="R2",
        row="row_facet",
        col="col_facet",
        hue="Trained",
        height=4,
        aspect=1.5,
        sharex=True,
        sharey=False,
        legend=False,
    )
    grid.set_xlabels("")

    facet_titles = {
        "00": "Original",
        "01": "Degree 1",
        "10": "Degree 2",
        "11": "arctan",
    }
    for row_idx, row_axes in enumerate(grid.axes):
        for col_idx, ax in enumerate(row_axes):
            ax.set_title(facet_titles[f"{row_idx}{col_idx}"], fontsize=12, fontweight="bold")
            if row_idx == 1:
                ax.set_ylim(0.96, 1.0)
                plt.setp(
                    ax.get_xticklabels(),
                    rotation=45,
                    ha="right",
                    fontweight="bold",
                )
                ax.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)
            ax.set_ylabel("$R^2$")
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            ax.grid(axis="x")

            num_categories = len(ax.get_xticks())
            for index in range(0, num_categories, 2):
                if index + 1 < num_categories:
                    ax.axvspan(
                        index + 0.5,
                        index + 1.5,
                        facecolor="grey",
                        alpha=0.15,
                        zorder=0,
                    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    grid.figure.savefig(OUTPUT_DIR / "Water_R2_score.png", dpi=300, bbox_inches="tight")
    plt.close(grid.figure)


def main() -> None:
    r2_data, expr_entg = read_inputs()
    plot_expr_entg(expr_entg)
    plot_water_r2(r2_data)
    print(f"Wrote Water R2 and expressibility figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
