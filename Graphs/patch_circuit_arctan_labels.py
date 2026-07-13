"""Patch the historical circuit montage to use arctan instead of atan.

The original source used to draw ``latex/Figures/circuit_drawings.png`` is not
available. This script performs the minimal reproducible bitmap edit requested
for the manuscript: replacing the five ChebyshevPQC labels in panel (a) from
``atan`` to ``arctan`` while leaving the rest of the historical montage intact.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = ROOT / "latex" / "Figures" / "circuit_drawings.png"


def render_math(label: str, max_width: int, max_height: int) -> Image.Image:
    """Render a mathtext label as a transparent bitmap that fits in a gate box."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "label.png"
        fig = plt.figure(figsize=(4.0, 0.8), dpi=300)
        fig.patch.set_alpha(0)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.text(0.5, 0.5, label, ha="center", va="center", fontsize=42)
        fig.savefig(path, transparent=True, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        label_image = Image.open(path).convert("RGBA")

    alpha = np.asarray(label_image)[..., 3]
    ys, xs = np.where(alpha > 0)
    label_image = label_image.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

    scale = min(max_width / label_image.width, max_height / label_image.height, 1.0)
    if scale < 1.0:
        label_image = label_image.resize(
            (int(label_image.width * scale), int(label_image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    return label_image


def patch_image(path: Path = DEFAULT_IMAGE) -> None:
    image = Image.open(path).convert("RGBA")
    draw = ImageDraw.Draw(image)

    # First remove any previous failed patch text. These rectangles lie either
    # inside the gate boxes redrawn below or in whitespace between panel rows.
    for y in (218, 592, 968):
        draw.rectangle((780, y - 70, 1050, y + 70), fill="white")
    for y in (1342, 1717):
        draw.rectangle((780, y - 65, 1050, y + 55), fill="white")

    # Coordinates are in the original 4961 x 5906 manuscript PNG. The gate-box
    # edges were measured from the historical bitmap so the rest of the montage
    # remains untouched.
    box_left, box_right = 633, 1199
    box_tops = (190, 396, 602, 809, 1015)
    box_bottoms = (310, 517, 723, 929, 1135)
    labels = [
        r"$R_X(\theta_5 * \arctan(X_0))$",
        r"$R_X(\theta_6 * \arctan(X_1))$",
        r"$R_X(\theta_7 * \arctan(X_2))$",
        r"$R_X(\theta_8 * \arctan(X_3))$",
        r"$R_X(\theta_9 * \arctan(X_4))$",
    ]

    for top, bottom, label in zip(box_tops, box_bottoms, labels):
        draw.rectangle(
            (box_left, top, box_right, bottom),
            fill="white",
            outline="black",
            width=6,
        )
        label_image = render_math(label, max_width=520, max_height=92)
        x = box_left + (box_right - box_left - label_image.width) // 2
        y = top + (bottom - top - label_image.height) // 2
        image.alpha_composite(label_image, (x, y))

    # Repair the panel-(c) gate border overlapped by the historical-edit
    # cleanup above.
    repair_box = (853, 1561, 1115, 1678)
    draw.rectangle(repair_box, fill="white", outline="black", width=6)
    label_image = render_math(r"$R_X(\theta_0)$", max_width=220, max_height=88)
    x = repair_box[0] + (repair_box[2] - repair_box[0] - label_image.width) // 2
    y = repair_box[1] + (repair_box[3] - repair_box[1] - label_image.height) // 2
    image.alpha_composite(label_image, (x, y))

    image.save(path)
    print(f"Patched arctan labels in {path}")


if __name__ == "__main__":
    patch_image()
