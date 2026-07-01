import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import re

def resize_image(image_path, size=(50, 50)):
    try:
        img = Image.open(image_path)
        img = img.resize(size, Image.Resampling.LANCZOS)
        return np.array(img)
    except (FileNotFoundError, OSError):
        return None

def extract_circuit_parts(circuit_str):
    parts = circuit_str.split('-')
    if len(parts) == 1: return "Classical SVR"
    layers = parts[2] if len(parts) >= 3 else "0"
    function = parts[1] if len(parts) >= 2 else "0"
    name_base = parts[0]
    name_mapping = {
    'CP_map': 'CPMap',
    'ZZ_map': 'ZZFeaturemap',
    'ParamZ': 'ParamZ',
    'Hubregtsen': 'Hubregtsen',
    'HighDim': 'HighDim',
    'MultiControl': 'MultiControl',
    'Chebyshev': 'ChebyshevPQC',
    'YZ_CX': 'YZ_CX'
    }
    
    if function != "0":
        if function == "atan":
            return f"{name_mapping.get(name_base, name_base)}-arctan (L={layers})"
        return f"{name_mapping.get(name_base, name_base)}-{function} (L={layers})"
    return f"{name_mapping.get(name_base, name_base)} (L={layers})"


plt.rcParams.update({'font.size': 16, 'font.weight':'heavy'})


train_molecule = os.getenv("MOLECULE")

data_folder = f'paper_results/transferability/STO-3G/{train_molecule}'

file_names = os.listdir(data_folder)
data_dict = {}

for file_name in file_names:
    clean_name = os.path.splitext(file_name)[0]
    #Change this as needed
    clean_name = clean_name.replace("TRUE_", "").replace("PROJECTED_", "").replace("5_", "").replace("FALSE_", "").replace("_XYZ", "")
    
    # Identify the base model ("Model-1" and "Model-2" both become "Model")
    model_base = re.sub(r'-\d+$', '', clean_name) if "-" in clean_name else clean_name
    file_path = os.path.join(data_folder, file_name)
    
    if not os.path.isfile(file_path): continue
    with open(file_path, 'r') as f:
        for line in f:
            try:
                molecule, value = line.strip().split(',')
                if molecule in ["Ethanol", "Dimethylether"]: continue
                if molecule not in data_dict: data_dict[molecule] = {}
                if model_base not in data_dict[molecule]: data_dict[molecule][model_base] = []
                data_dict[molecule][model_base].append(float(value))
            except: continue

mean_res = {
    mol: {model: np.mean(values) for model, values in models.items() if values}
    for mol, models in data_dict.items()
}
std_res = {
    mol: {model: np.std(values) for model, values in models.items() if values}
    for mol, models in data_dict.items()
}

order = ["Water", "Methanol", "Ethylene", "Methane", "Ammonia", "Formaldehyde", "Ethane"]
df_mean = pd.DataFrame.from_dict(mean_res, orient='index').reindex(order).fillna(-9)
df_std = pd.DataFrame.from_dict(std_res, orient='index').reindex(order).fillna(0)

base_names = [extract_circuit_parts(col) for col in df_mean.columns]

df_mean_aggr = df_mean.apply(pd.to_numeric).T.groupby(base_names).mean().T
df_std_aggr = df_std.apply(pd.to_numeric).T.groupby(base_names).mean().T

x = 8  # Change this to however many top models you want to show

# Calculate the mean performance of each model across all molecules
if "-" not in train_molecule:
    df_sort_ref = df_mean_aggr.drop(train_molecule, axis=0, errors='ignore')
else:
    df_sort_ref = df_mean_aggr
column_means = df_sort_ref.mean()

# Get all models sorted by performance
all_sorted_cols = column_means.sort_values(ascending=False).index.tolist()

final_selection = []
if "Classical SVR" in all_sorted_cols:
    final_selection.append("Classical SVR")
    all_sorted_cols.remove("Classical SVR")

top_x_models = all_sorted_cols[:x] 
final_selection.extend(top_x_models)

df_mean_aggr = df_mean_aggr.replace(-9, np.nan)

df_m_final = df_mean_aggr[final_selection].drop(train_molecule, axis=0, errors='ignore')
df_s_final = df_std_aggr[final_selection].drop(train_molecule, axis=0, errors='ignore')

gap_v = pd.Series(np.nan, index=df_m_final.index)
gap_a = pd.Series("", index=df_m_final.index)

df_heatmap = pd.concat([
    df_m_final.iloc[:, 0:1], 
    pd.DataFrame({'': gap_v}), 
    df_m_final.iloc[:, 1:]
], axis=1)

labels = df_heatmap.index 

plt.figure(figsize=(len(final_selection) * 1.5 + 5, 13))

ax = sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap='YlGnBu', 
                 linewidths=1, cbar_kws={'label': '$R^2$'}, vmin=0, vmax=1)

for text in ax.texts:
    text.set_fontweight('bold')


# Y-Axis Images
for i, label in enumerate(labels):
    image_path = f'images/{label}.png'
    resized_img = resize_image(image_path, size=(50, 50))
    if resized_img is not None:
        imagebox = OffsetImage(resized_img, zoom=1)
        ab = AnnotationBbox(imagebox, (0, i + 0.5), frameon=False, xybox=(-0.1, i + 0.2),
                            xycoords='data', boxcoords=('axes fraction', 'data'), box_alignment=(0.5, 0.5))
        ax.add_artist(ab)
    ax.text(-0.25, i + 0.5, label, ha='right', va='center', transform=ax.transData, fontsize=18, fontweight='bold')

# X-Axis Formatting
enabled_indices = [j for j in range(len(df_heatmap.columns)) if j != 1]

enabled_indices = [j for j in range(len(df_heatmap.columns)) if j != 1]

center_locations = np.array(enabled_indices) + 0.5
column_labels = df_heatmap.columns[enabled_indices]

ax.set_yticklabels([])

plt.tick_params(axis="x", labelsize=16)
ax.set_xticks(center_locations)
ax.set_xticklabels(column_labels, rotation=45, ha='right')

plt.tight_layout()
output_dir = os.getenv("FIGURE_OUTPUT_DIR", "paper_results/regenerated_figures")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"{train_molecule}_heatmap.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Heatmap generated: {output_path}")

# Produces a latex table

# latex_code = df_annot_final.T.to_latex(
#     index=True, 
#     # float_format="%.2f",
#     # caption="",
#     label="tab:Water_Transfer",
#     longtable=True,
#     escape=True,
#     column_format='l' + 'c'*len(df_annot_final.T.columns) # Left-align index, center data
# )

# latex_code = latex_code.replace(r'\begin{longtable}', 
#                                    r'\rowcolors{2}{gray!10}{white}' + '\n' + r'\begin{longtable}')

# print(latex_code)
