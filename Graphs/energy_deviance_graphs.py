import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.rcParams['font.size'] = 13

df = pd.read_csv("Blind_Energy_MAE_Data.csv")

# Get unique molecules and circuits
molecules = ["Pyrazine", "Benzene", "Pentane", "Nitrogen", "Chlorine", "Ozone", "Thioformaldehyde"]
circuits = df['Circuit'].unique()
num_molecules = len(molecules)

# Create a single row of subplots
fig, axs = plt.subplots(1, num_molecules, figsize=(18, 6), sharey=True)
plt.subplots_adjust(top=0.7)

# Plotting
colors = plt.cm.tab10.colors[:len(circuits)]
color_map = {circuit: color for circuit, color in zip(circuits, colors)}

# Define a target size for all images in pixels
TARGET_IMAGE_HEIGHT = 75
TARGET_IMAGE_WIDTH = 75

for i, molecule in enumerate(molecules):
    ax = axs[i]
    molecule_df = df[df['Molecule'] == molecule]
    molecule_df = molecule_df.set_index('Circuit').reindex(circuits).reset_index()

    ax.bar(molecule_df['Circuit'], 1000 * molecule_df['Energy Devience'],
           color=[color_map[c] for c in molecule_df['Circuit']])

    ax.set_title(molecule, fontsize=16)
    ax.set_xticklabels([])

    # Add image above the plot and scale
    image_path = f'images/{molecule}.png'
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
    except FileNotFoundError:
        print(f"Warning: Image not found for {molecule} at {image_path}. Skipping.")

# Set a shared y-axis label for the entire row
axs[0].set_ylabel('Energy Deviance $mE_{h}$', fontsize=16)
# axs[0].set_yscale('log')

# Create a single legend for the entire figure
# plt.tight_layout()
handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[c]) for c in circuits]
fig.legend(handles, circuits, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=int(len(circuits)), fontsize=16)
# fig.legend.set_in_layout(False)
# plt.suptitle('Energy Deviance for Each Molecule by Approach', fontsize=18)
fig.suptitle("", y=1.05)
plt.tight_layout()

# Save the plot
plt.savefig('Energy_Deviance_Legend_Only.png')

print("Graph with legend only saved as Energy_Deviance_Legend_Only.png")