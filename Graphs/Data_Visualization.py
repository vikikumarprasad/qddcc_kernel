import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
# from squlearn.encoding_circuit import QiskitEncodingCircuit
from squlearn.encoding_circuit import QiskitEncodingCircuit, HighDimEncodingCircuit
import numpy as np

from Circuit_Classes import *

figure = plt.figure(figsize=(12,8))

top = os.getcwd()
folder_path = os.path.join(top, "paper_results/transferability/STO-3G/")
target_files = []
circuit_names = []
for filename in os.listdir(folder_path):
    if "Water-Scaled" in filename:
        file_path = os.path.join(folder_path, filename)
        if "_SVR_" in filename:
            target_files.insert(0, file_path)
            circuit_names.insert(0, filename.replace("_Water-Scaled.csv", "").replace("HYPERPARAMETER-", ""))
        else:
            #not ("Chebyshev" in filename) and not ("Z_param" in filename) or
            if not ("HYPERPARAMETER" in filename):
                continue
            print(filename)
            target_files.append(file_path)
            circuit_names.append(filename.replace("_Water-Scaled.csv", "").replace("HYPERPARAMETER-", ""))

#create dataframes for each file then combine
all_data_frames = []

for file in target_files:
    try:
        df = pd.read_csv(file, header=None, names=["Molecules", "R2"])
        df = df.drop_duplicates()
        print(df)
        all_data_frames += [df]
    except Exception as e:
        print(f"Execpetion occured: {e}")
desired_label_orders = ["Water", "Methanol", "Ethylene", "Ethane", "Methane", "Ammonia", "Formaldehyde", "Dimethylether", "Ethanol"]

# Concatenate all DataFrames into a single DataFrame
combined_df = pd.concat(all_data_frames, ignore_index=True)
combined_df = combined_df.groupby('Molecules')['R2'].apply(list).reset_index()

combined_df['Molecules'] = pd.Categorical(
    combined_df['Molecules'],
    categories=desired_label_orders,
    ordered=True
)
ordered_df = combined_df.sort_values('Molecules').reset_index(drop=True)
indexed_df = ordered_df.set_index("Molecules")


max_values_count = indexed_df['R2'].apply(len).max()
heatmap_data = pd.DataFrame(
    indexed_df['R2'].tolist(),
    index=indexed_df.index,
    columns=[f'R2_{i+1}' for i in range(max_values_count)]
)

print(max_values_count, len(circuit_names))
heatmap_data.columns = circuit_names

mean_threshold = 0.4

first_column_name = heatmap_data.columns[0]
print(first_column_name)

column_means = heatmap_data.mean(axis=0)
print(column_means)


sorted_column_means = column_means[1:].sort_values(ascending=False)
columns_to_keep_mask = sorted_column_means >= mean_threshold

# The index of this sorted Series gives us the new column order
new_column_order = [first_column_name] + sorted_column_means[columns_to_keep_mask].index.tolist()

print(new_column_order)
heatmap_data = heatmap_data[new_column_order]
heatmap_data.rename(columns={first_column_name: 'Classical SVR'}, inplace=True)
sns.heatmap(
    heatmap_data,
    cmap='Blues',
    annot=True,
    fmt=".3f",
    linewidths=.5,
    cbar_kws={'label': 'R2 Score'}
)
plt.title("Transferability of Quantum Kernel Methods")
plt.xlabel("Kernel Method Circuit")
plt.ylabel("Molecular Structure")
plt.tight_layout()
plt.savefig("quantum_kernel_heatmap.png")
plt.show()
