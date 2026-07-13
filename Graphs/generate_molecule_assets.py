"""Build the molecule overview from archived static molecule artwork."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

import joblib

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))

import matplotlib.pyplot as plt
from PIL import Image

from molecule_image_assets import display_name, molecule_image_path


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = ROOT / "data" / "geometries" / "training"
OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)

TRAINING_MOLECULES = (
    "Water",
    "Ammonia",
    "Methane",
    "Methanol",
    "Formaldehyde",
    "Ethylene",
    "Ethane",
)
BLIND_MOLECULES = (
    "Benzene",
    "Chlorine",
    "Nitrogen",
    "Ozone",
    "Pentane",
    "Pyrazine",
    "Thioformaldehyde",
)
OVERVIEW_LAYOUT = (
    ("Water", "Ammonia", "Methane"),
    ("Methanol", "Formaldehyde"),
    ("Ethylene", "Ethane"),
)


def split_row_counts(molecule: str) -> tuple[int, int]:
    scaled = ROOT / "data" / "training" / "STO-3G" / molecule / "scaled"
    train = joblib.load(scaled / "full_feature_train.bin")
    test = joblib.load(scaled / "full_feature_test.bin")
    return len(train["X"]), len(test["X"])


def conformer_count(molecule: str) -> int:
    """Return the manuscript-style count of conformer geometries.

    The recovered split metadata contains 198 calculation names for ammonia,
    methane, and formaldehyde, but the archived raw geometry directories and
    the original figure count 99 conformer geometries for those systems. Water
    has 201 archived XYZ files, while the recovered train/test split uses 199.
    Prefer the original geometry count and only use the split count to account
    for water geometries excluded from the model-ready split.
    """

    geometry_count = len(list((GEOMETRY_ROOT / molecule.lower()).glob("*.xyz")))
    scaled = ROOT / "data" / "training" / "STO-3G" / molecule / "scaled"
    train = joblib.load(scaled / "full_feature_train.bin")
    test = joblib.load(scaled / "full_feature_test.bin")
    split_count = len(set(train["names"]) | set(test["names"]))
    if molecule == "Water":
        return min(geometry_count, split_count)
    return geometry_count


def draw_static_molecule(ax: plt.Axes, molecule: str) -> None:
    image = Image.open(molecule_image_path(molecule)).convert("RGBA")
    ax.imshow(image)
    ax.axis("off")


def overview_cell(fig: plt.Figure, cell, molecule: str) -> None:
    subgrid = cell.subgridspec(2, 1, height_ratios=(1.18, 1.0), hspace=0.0)
    image_ax = fig.add_subplot(subgrid[0])
    text_ax = fig.add_subplot(subgrid[1])

    draw_static_molecule(image_ax, molecule)
    text_ax.axis("off")

    train_rows, test_rows = split_row_counts(molecule)
    text_ax.text(
        0.5,
        0.98,
        (
            f"{display_name(molecule)}\n"
            f"({conformer_count(molecule)} Conformers)\n"
            f"$\\it{{Train}}$: {train_rows}\n"
            f"$\\it{{Test}}$: {test_rows}"
        ),
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
        linespacing=1.15,
    )


def write_overview() -> None:
    fig = plt.figure(figsize=(8.8, 10.6), facecolor="white")
    outer = fig.add_gridspec(3, 6, hspace=0.08, wspace=0.16)

    for row, molecules in enumerate(OVERVIEW_LAYOUT):
        spans = ((0, 2), (2, 4), (4, 6)) if len(molecules) == 3 else ((1, 3), (3, 5))
        for molecule, (left, right) in zip(molecules, spans):
            overview_cell(fig, outer[row, left:right], molecule)

    fig.savefig(
        OUTPUT_DIR / "Molecules_Data.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for molecule in (*TRAINING_MOLECULES, *BLIND_MOLECULES):
        molecule_image_path(molecule)
    write_overview()
    print(f"Wrote dataset overview to {OUTPUT_DIR / 'Molecules_Data.png'}")
    print("Used archived static molecule artwork from images/")


if __name__ == "__main__":
    main()
