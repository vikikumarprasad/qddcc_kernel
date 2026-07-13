import pandas as pd
import io
import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qddcc-mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "qddcc-cache"))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from molecule_image_assets import display_name, molecule_image_path

plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 15,
    'ytick.labelsize': 12,
    'legend.fontsize': 14,
})

df = pd.read_csv("paper_results/blind/Energy_MAE_Data.csv")

# Get unique molecules and circuits
molecules = ["Pyrazine", "Benzene", "Pentane", "Nitrogen", "Chlorine", "Ozone", "Thioformaldehyde"]
circuits = ["MP2", "Classical", "Chebyshev", "Hubregtsen", "MultiControl"]
num_molecules = len(molecules)

# Create a single row of subplots
fig, axs = plt.subplots(1, num_molecules, figsize=(19, 6.8), sharey=True)
plt.subplots_adjust(top=0.7)

# Plotting. Keep the baseline colors stable while placing the classical
# baseline next to MP2 so the quantum methods are grouped together.
color_map = {
    "MP2": "tab:blue",
    "Classical": "tab:green",
    "Chebyshev": "tab:orange",
    "Hubregtsen": "tab:red",
    "MultiControl": "tab:purple",
}

# Define a target size for all images in pixels
TARGET_IMAGE_HEIGHT = 75
TARGET_IMAGE_WIDTH = 75

for i, molecule in enumerate(molecules):
    ax = axs[i]
    molecule_df = df[df['Molecule'] == molecule]
    molecule_df = molecule_df.set_index('Circuit').reindex(circuits).reset_index()

    values = 1000 * molecule_df['Energy Devience']
    bars = ax.bar(
        molecule_df['Circuit'],
        values,
        color=[color_map[c] for c in molecule_df['Circuit']],
    )
    ax.bar_label(
        bars,
        labels=[f"{value:.0f}" for value in values],
        padding=2,
        rotation=90,
        fontsize=10,
        fontweight="bold",
    )

    ax.set_title(display_name(molecule), fontsize=16)
    ax.set_xticklabels([])

    # Add image above the plot and scale
    image_path = molecule_image_path(molecule, required=False)
    try:
        image_data = plt.imread(image_path)
        
        zoom_height = TARGET_IMAGE_HEIGHT / image_data.shape[0]
        zoom_width = TARGET_IMAGE_WIDTH / image_data.shape[1]
        zoom_factor = min(zoom_height, zoom_width)
        
        img_box = OffsetImage(image_data, zoom=zoom_factor)
        ab = AnnotationBbox(img_box, (0.5, 1.05),
                            xycoords='axes fraction',
                            box_alignment=(0.5, 0),
                            frameon=False)
        ax.add_artist(ab)
    except (FileNotFoundError, OSError, TypeError):
        print(f"Warning: Image not found for {molecule} at {image_path}. Skipping.")

# Leave enough headroom for the vertical value labels above the tallest bars.
max_energy_deviance = 1000 * df["Energy Devience"].max()
axs[0].set_ylim(0, max_energy_deviance * 1.12)

# Set a shared y-axis label for the entire row
axs[0].set_ylabel(
    'Energy Deviance $mE_{h}$',
    fontsize=16,
    fontweight="bold",
)
for label in axs[0].get_yticklabels():
    label.set_fontweight("bold")
# axs[0].set_yscale('log')

# Create a single legend for the entire figure
# plt.tight_layout()
handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[c]) for c in circuits]
fig.legend(handles, circuits, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=int(len(circuits)), fontsize=15)
# fig.legend.set_in_layout(False)
# plt.suptitle('Energy Deviance for Each Molecule by Approach', fontsize=18)
fig.suptitle("", y=1.05)
plt.tight_layout()

# Save the plot
output_dir = os.getenv("FIGURE_OUTPUT_DIR", "paper_results/regenerated_figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "Water-Formaldehyde_Energy_Deviance_Legend_Only.png")
plt.savefig(output_path, dpi=450, bbox_inches="tight")
plt.close()

print(f"Graph with legend saved as {output_path}")
