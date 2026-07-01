"""Reconstruct the STO-3G and cc-pVDZ water R²/energy panels."""

from __future__ import annotations

import os
from pathlib import Path
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "paper_results" / "energy"
OUTPUT_DIR = Path(
    os.getenv(
        "FIGURE_OUTPUT_DIR",
        ROOT / "paper_results" / "regenerated_figures",
    )
)

DISPLAY_NAMES = {
    "Chebyshev": "Chebyshev",
    "Classical": "Classical",
    "Hubregtsen": "Hubregtsen",
    "MultiControl": "MultiControl",
}
COLORS = {
    "Chebyshev": "orange",
    "Classical": "blue",
    "Hubregtsen": "purple",
    "MultiControl": "red",
    "MP2": "green",
}
ORDER = ("Chebyshev", "Classical", "Hubregtsen", "MultiControl")
MIN_VALID_R2 = -10.0


def circuit_family(value: str) -> str | None:
    if value == "MP2":
        return "MP2"
    match = re.fullmatch(r"([A-Za-z]+)_0_1_\d+", value)
    if not match:
        return None
    family = match.group(1)
    return family if family in DISPLAY_NAMES else None


def load_summary(basis: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{basis}_5_Water_Energy_and_MAE_Data.csv"
    data = pd.read_csv(path)
    data["Family"] = data["Circuit"].map(circuit_family)
    data = data.dropna(subset=["Family"]).copy()

    train = joblib.load(
        ROOT / "data" / "training" / basis / "Water" / "scaled" / "5_feature_train.bin"
    )
    data["Training Data"] = np.rint(
        data["Percent of Train"] * len(train["X"])
    ).astype(int)
    return data


def aggregate(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (training_size, family), group in data.groupby(
        ["Training Data", "Family"], observed=True
    ):
        # The recovered tables use -50 as a failed-run sentinel and also
        # contain a few corrupted values as low as -30 million. The published
        # curves exclude these records. Values in the plotted experimental
        # range, including small negative scores, are retained.
        r2 = group.loc[group["R2"] >= MIN_VALID_R2, "R2"].to_numpy()
        energy = group["MAE Energy Devience"].to_numpy()

        def mean_and_sem(values: np.ndarray) -> tuple[float, float]:
            if len(values) == 0:
                return float("nan"), float("nan")
            mean = float(np.mean(values))
            if len(values) < 2:
                return mean, 0.0
            return mean, float(np.std(values, ddof=1) / np.sqrt(len(values)))

        r2_mean, r2_sem = mean_and_sem(r2)
        energy_mean, energy_sem = mean_and_sem(energy)
        rows.append(
            {
                "Training Data": training_size,
                "Family": family,
                "R2_mean": r2_mean,
                "R2_std": r2_sem,
                "energy_mean": energy_mean,
                "energy_std": energy_sem,
            }
        )
    return pd.DataFrame(rows)


def plot_series(ax: plt.Axes, summary: pd.DataFrame, metric: str) -> None:
    for family in ORDER:
        values = summary[summary["Family"] == family].sort_values("Training Data")
        if values.empty:
            continue
        mean = values[f"{metric}_mean"].to_numpy()
        std = values[f"{metric}_std"].to_numpy()
        x = values["Training Data"].to_numpy()
        ax.plot(
            x,
            mean,
            marker="o",
            linewidth=2,
            color=COLORS[family],
            label=DISPLAY_NAMES[family],
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            color=COLORS[family],
            alpha=0.18,
        )


def plot_mp2(ax: plt.Axes, summary: pd.DataFrame) -> None:
    values = summary[summary["Family"] == "MP2"].sort_values("Training Data")
    if values.empty:
        return
    ax.plot(
        values["Training Data"],
        values["energy_mean"],
        marker="D",
        linewidth=2,
        color=COLORS["MP2"],
        label="MP2",
    )


def plot_sto3g(summary: pd.DataFrame) -> None:
    fig, (r2_ax, energy_ax) = plt.subplots(1, 2, figsize=(15, 6))
    plot_series(r2_ax, summary, "R2")
    plot_series(energy_ax, summary, "energy")
    plot_mp2(energy_ax, summary)

    r2_ax.set_ylim(0.90, 1.0)
    r2_ax.set_ylabel(r"$R^2$")
    r2_ax.set_xlabel("Training Data")
    energy_ax.set_ylabel(r"MAE from $E_{\mathrm{CCSD}}^{\mathrm{corr}}$ ($E_h$)")
    energy_ax.set_xlabel("Training Data")
    energy_ax.set_yscale("log")
    for ax in (r2_ax, energy_ax):
        ax.grid(True, alpha=0.6)

    handles, labels = energy_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUTPUT_DIR / "STO-3G_R2_and_Energy.png", dpi=300)
    plt.close(fig)


def plot_ccpvdz(summary: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(15, 6))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(1, 1),
        height_ratios=(3, 2),
        hspace=0.08,
        wspace=0.2,
    )
    r2_top = fig.add_subplot(grid[0, 0])
    r2_bottom = fig.add_subplot(grid[1, 0], sharex=r2_top)
    energy_ax = fig.add_subplot(grid[:, 1])

    plot_series(r2_top, summary, "R2")
    plot_series(r2_bottom, summary, "R2")
    plot_series(energy_ax, summary, "energy")
    plot_mp2(energy_ax, summary)

    r2_top.set_ylim(-0.05, 1.0)
    r2_bottom.set_ylim(-1.5, -0.7)
    r2_top.spines.bottom.set_visible(False)
    r2_bottom.spines.top.set_visible(False)
    r2_top.tick_params(labelbottom=False, bottom=False)
    r2_bottom.set_xlabel("Training Data")
    fig.text(0.03, 0.53, r"$R^2$", va="center", rotation="vertical")

    marker_args = dict(
        marker=[(-1, -0.5), (1, 0.5)],
        markersize=12,
        linestyle="none",
        color="black",
        mec="black",
        mew=1,
        clip_on=False,
    )
    r2_top.plot([0, 1], [0, 0], transform=r2_top.transAxes, **marker_args)
    r2_bottom.plot([0, 1], [1, 1], transform=r2_bottom.transAxes, **marker_args)

    energy_ax.set_xlabel("Training Data")
    energy_ax.set_ylabel(r"MAE from $E_{\mathrm{CCSD}}^{\mathrm{corr}}$ ($E_h$)")
    for ax in (r2_top, r2_bottom, energy_ax):
        ax.grid(True, alpha=0.6)

    handles, labels = energy_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.12,
        top=0.86,
        wspace=0.2,
        hspace=0.08,
    )
    fig.savefig(OUTPUT_DIR / "cc-pVDZ_R2_and_Energy.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_sto3g(aggregate(load_summary("STO-3G")))
    plot_ccpvdz(aggregate(load_summary("cc-pVDZ")))
    print(f"Wrote reconstructed water panels to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
