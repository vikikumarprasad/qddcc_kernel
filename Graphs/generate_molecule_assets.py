"""Generate molecule artwork and a dataset overview from preserved XYZ files."""

from __future__ import annotations

import os
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY_ROOT = ROOT / "data" / "geometries"
IMAGE_DIR = ROOT / "images"
OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)

TRAINING_MOLECULES = (
    "Water",
    "Methanol",
    "Ethylene",
    "Ethane",
    "Methane",
    "Ammonia",
    "Formaldehyde",
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

COLORS = {
    "H": "#f2f2f2",
    "C": "#3b3b3b",
    "N": "#356dcc",
    "O": "#d83232",
    "S": "#d7b51d",
    "Cl": "#43a047",
}
RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "S": 1.05,
    "Cl": 1.02,
}
DISPLAY_RADII = {
    "H": 180,
    "C": 360,
    "N": 390,
    "O": 400,
    "S": 480,
    "Cl": 500,
}


def natural_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    return path.stem, int(match.group(1)) if match else 0


def geometry_path(molecule: str) -> Path:
    blind_path = GEOMETRY_ROOT / "blind" / f"{molecule}.xyz"
    if blind_path.exists():
        return blind_path
    candidates = sorted(
        (GEOMETRY_ROOT / "training" / molecule.lower()).glob("*.xyz"),
        key=natural_key,
    )
    if not candidates:
        raise FileNotFoundError(f"No XYZ geometry found for {molecule}")
    return candidates[0]


def read_geometry(path: Path) -> tuple[list[str], np.ndarray]:
    elements: list[str] = []
    coordinates: list[list[float]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) != 4 or fields[0] not in RADII:
            continue
        try:
            xyz = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        elements.append(fields[0])
        coordinates.append(xyz)
    if not coordinates:
        raise ValueError(f"No atoms parsed from {path}")
    return elements, np.asarray(coordinates, dtype=float)


def project(coordinates: np.ndarray) -> np.ndarray:
    centered = coordinates - coordinates.mean(axis=0)
    if len(coordinates) < 2:
        return np.zeros((len(coordinates), 2))
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    if axes.shape[0] == 1:
        return np.column_stack((centered @ axes[0], np.zeros(len(centered))))
    return centered @ axes[:2].T


def bonds(elements: list[str], coordinates: np.ndarray) -> list[tuple[int, int]]:
    result = []
    for left in range(len(elements)):
        for right in range(left + 1, len(elements)):
            distance = np.linalg.norm(coordinates[left] - coordinates[right])
            cutoff = 1.25 * (RADII[elements[left]] + RADII[elements[right]])
            if 0.2 < distance <= cutoff:
                result.append((left, right))
    return result


def draw_molecule(
    ax: plt.Axes,
    elements: list[str],
    coordinates: np.ndarray,
    *,
    title: str | None = None,
) -> None:
    points = project(coordinates)
    for left, right in bonds(elements, coordinates):
        ax.plot(
            points[[left, right], 0],
            points[[left, right], 1],
            color="#7a7a7a",
            linewidth=4,
            solid_capstyle="round",
            zorder=1,
        )
    for index, element in enumerate(elements):
        ax.scatter(
            points[index, 0],
            points[index, 1],
            s=DISPLAY_RADII[element],
            color=COLORS[element],
            edgecolor="#222222",
            linewidth=1,
            zorder=2,
        )
    if title:
        ax.set_title(title, fontsize=15, fontweight="bold")
    span = np.ptp(points, axis=0)
    margin = max(float(span.max()) * 0.2, 0.6)
    ax.set_xlim(points[:, 0].min() - margin, points[:, 0].max() + margin)
    ax.set_ylim(points[:, 1].min() - margin, points[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.axis("off")


def write_individual_image(molecule: str) -> None:
    elements, coordinates = read_geometry(geometry_path(molecule))
    fig, ax = plt.subplots(figsize=(3, 3))
    draw_molecule(ax, elements, coordinates)
    fig.savefig(
        IMAGE_DIR / f"{molecule}.png",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)


def dataset_counts(molecule: str) -> tuple[int, int, int, int]:
    scaled = (
        ROOT
        / "data"
        / "training"
        / "STO-3G"
        / molecule
        / "scaled"
    )
    train = joblib.load(scaled / "full_feature_train.bin")
    test = joblib.load(scaled / "full_feature_test.bin")
    return len(train["names"]), len(test["names"]), len(train["X"]), len(test["X"])


def write_overview() -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, molecule in zip(axes.flat, TRAINING_MOLECULES):
        elements, coordinates = read_geometry(geometry_path(molecule))
        draw_molecule(ax, elements, coordinates, title=molecule)
        train_geometries, test_geometries, train_rows, test_rows = dataset_counts(
            molecule
        )
        ax.text(
            0.5,
            -0.04,
            (
                f"Conformers: {train_geometries} train / {test_geometries} test\n"
                f"$t_2$ amplitudes: {train_rows:,} train / {test_rows:,} test"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=9,
        )
    axes.flat[-1].axis("off")
    fig.tight_layout(h_pad=2.3)
    fig.savefig(
        OUTPUT_DIR / "Molecules_Data.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for molecule in (*TRAINING_MOLECULES, *BLIND_MOLECULES):
        write_individual_image(molecule)
    write_overview()
    print(f"Wrote molecule assets to {IMAGE_DIR}")
    print(f"Wrote dataset overview to {OUTPUT_DIR / 'Molecules_Data.png'}")


if __name__ == "__main__":
    main()
