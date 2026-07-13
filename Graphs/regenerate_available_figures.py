"""Regenerate every manuscript figure supported by the recovered sources."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper_results" / "regenerated_figures"


def run(script: str, **extra_env: str) -> None:
    env = os.environ.copy()
    env.update(
        {
            "MPLBACKEND": "Agg",
            "MPLCONFIGDIR": os.getenv(
                "MPLCONFIGDIR",
                str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"),
            ),
            "XDG_CACHE_HOME": os.getenv(
                "XDG_CACHE_HOME",
                str(Path(tempfile.gettempdir()) / "qddcc-cache"),
            ),
            "FIGURE_OUTPUT_DIR": str(OUTPUT_DIR),
            **extra_env,
        }
    )
    subprocess.run(
        [sys.executable, str(ROOT / "Graphs" / script)],
        cwd=ROOT,
        env=env,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-recomputed-shap",
        action="store_true",
        help=(
            "also run a fresh SHAP analysis as a separately named diagnostic; "
            "its values do not match the published figure"
        ),
    )
    parser.add_argument(
        "--skip-circuit-montage",
        action="store_true",
        help=(
            "skip the Qiskit/sQUlearn circuit montage; useful for checking the "
            "non-quantum plotting path in a partial environment"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run("generate_molecule_assets.py")

    for molecule in (
        "Water",
        "Ammonia",
        "Methane",
        "Formaldehyde",
        "Water-Formaldehyde",
    ):
        run("Heatmap_Visualization.py", MOLECULE=molecule)

    for training_set in ("Formaldehyde", "Water-Formaldehyde"):
        run("exp_pred_4.py", TRAIN_MOLECULE=training_set)

    run("create_training_loss.py")
    run("energy_deviance_graphs.py")
    run("plot_published_shap.py")
    run("plot_water_r2_energy.py")
    run("plot_water_r2_expr_entg.py")
    if not args.skip_circuit_montage:
        run("generate_circuit_montage.py")

    if args.include_recomputed_shap:
        run("Shap_Graphs.py")
        print(
            "Warning: ccsd_shap_summary_bar_recomputed.png is a fresh SHAP "
            "computation, not a reproduction of the published values."
        )

    print(f"Regenerated figures are in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
