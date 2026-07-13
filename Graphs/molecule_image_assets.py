"""Shared helpers for static molecule artwork used in manuscript figures."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"
IMAGE_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")

DISPLAY_NAMES = {
    "Nitrogen": "Dinitrogen",
}


def display_name(molecule: str) -> str:
    return DISPLAY_NAMES.get(molecule, molecule)


def molecule_image_path(molecule: str, *, required: bool = True) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        path = IMAGE_DIR / f"{molecule}{extension}"
        if path.exists():
            return path
    if required:
        choices = ", ".join(f"{molecule}{extension}" for extension in IMAGE_EXTENSIONS)
        raise FileNotFoundError(f"Missing static molecule image in {IMAGE_DIR}: {choices}")
    return None


def required_molecule_images(molecules: list[str] | tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for molecule in molecules:
        path = molecule_image_path(molecule)
        if path is not None:
            paths.append(path)
    return paths
