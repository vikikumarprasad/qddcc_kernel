"""Figure: in-distribution vs transfer accuracy as a function of the kernel bandwidth gamma.

Left: cross-validated R^2 on the source molecule (what hyperparameter tuning optimizes).
Right: mean R^2 over the six other molecules (what transfer needs).
Colors, markers and direct labels distinguish the four model families.

Usage: python -I plot_bandwidth.py <scan.csv> <source label> <out.png>
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

scan, source, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
SERIES = [  # (family, label, color, marker) -- fixed order, color follows the model
    ("classical", "Classical RBF", "#2a78d6", "o"),
    ("Chebyshev", "ChebyshevPQC", "#eb6834", "s"),
    ("Hubregtsen", "Hubregtsen", "#1baf7a", "^"),
    ("MultiControl", "MultiControl", "#eda100", "D"),
]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

df = pd.read_csv(scan)
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
panels = [("cv_r2", f"Cross-validated $R^2$ on {source.lower()}", axes[0]),
          ("mean_targets", "Mean $R^2$ on the six other molecules", axes[1])]
for col, title, ax in panels:
    for fam, label, color, marker in SERIES:
        s = df[df["family"] == fam].sort_values("gamma")
        if s.empty:
            continue
        ax.plot(s["gamma"], s[col], color=color, lw=1.6, marker=marker, ms=4.5,
                mec="white", mew=0.8, label=label, zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=9, color=TEXT, loc="left")
    ax.grid(True, color=GRID, lw=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlabel(r"Kernel bandwidth $\gamma$ (larger = narrower kernel)", color=TEXT)
axes[0].set_ylabel("$R^2$")
y0, y1 = axes[0].get_ylim()
axes[0].text(1.15, y0 + 0.06 * (y1 - y0), r"PQK default $\gamma=1$", color=MUTED, fontsize=7.5,
             ha="left", va="bottom")
# direct labels at the right end of the transfer panel, nudged apart to avoid collisions
ends = []
for fam, label, color, marker in SERIES:
    s = df[df["family"] == fam].sort_values("gamma")
    if not s.empty:
        ends.append([float(s["mean_targets"].iloc[-1]), label, color])
ends.sort()
span = axes[1].get_ylim()[1] - axes[1].get_ylim()[0]
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.06 * span)
xmax = df["gamma"].max()
for y, label, color in ends:
    axes[1].annotate(label, xy=(xmax, y), xytext=(6, 0), textcoords="offset points", va="center",
                     fontsize=7.5, color=TEXT)
xmin = df["gamma"].min()
axes[0].set_xlim(xmin / 1.6, xmax * 1.6)
axes[1].set_xlim(xmin / 1.6, xmax * 8)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8,
           bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
print(f"wrote {out}")
