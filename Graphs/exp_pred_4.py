import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import entropy, wasserstein_distance
from scipy.special import rel_entr

plt.rcParams.update({'font.size': 15, 'font.weight':'bold'})

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def js_divergence(p, q):
    """Jensen–Shannon divergence (uses rel_entr from scipy.special)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return 0.5 * rel_entr(p, m).sum() + 0.5 * rel_entr(q, m).sum()

def plot_molecule_data(ax_scatter, ax_hist_top, ax_hist_right, ax_img, avg_df, long_df, molecule_name):
    """
    Plots a scatter plot of the data along side the distributions of that data
    """
    if avg_df.empty or long_df.empty:
        ax_scatter.text(0.5, 0.5, f"No data for {molecule_name}", ha='center')
        ax_img.axis('off')
        return

    circuit_order = ["Hubregtsen (L=1)", "Classical SVR"][::-1] #Keeps a consistent order

    #Load the required data
    avg_df['Circuit'] = pd.Categorical(avg_df['Circuit'], categories=circuit_order, ordered=True)
    long_df['Circuit'] = pd.Categorical(long_df['Circuit'], categories=circuit_order, ordered=True)


    avg_df = avg_df.sort_values(by=['Expected_value_bin'])
    long_df = long_df.sort_values(by=['Expected_value_unscaled'])

    # Scatter
    for circuit_name, circuit_group in avg_df.groupby('Circuit'):
        sns.scatterplot(x='predicted_value_mean', y='expected_value_mean',
                        data=circuit_group, ax=ax_scatter, legend=False, alpha=0.9)

    min_val = min(avg_df['expected_value_mean'].min(), avg_df['predicted_value_mean'].min())
    max_val = max(avg_df['expected_value_mean'].max(), avg_df['predicted_value_mean'].max())
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
    ax_scatter.set_xlabel('Predicted $t^{ab}_{ij(CCSD)}$', fontweight='bold', fontsize = 16)
    ax_scatter.set_ylabel('Reference $t^{ab}_{ij(CCSD)}$', fontweight='bold', fontsize = 16)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_scatter.set_aspect('equal', adjustable='box')

    # compute limits
    min_val_x = avg_df['predicted_value_mean'].min() * 1.1
    max_val_x = avg_df['predicted_value_mean'].max() * 1.1
    min_val_y = avg_df['expected_value_mean'].min() * 1.1
    max_val_y = avg_df['expected_value_mean'].max() * 1.1
    min_global = min(min_val_x, min_val_y)
    max_global = max(max_val_x, max_val_y)


    if not np.isfinite(min_global) or not np.isfinite(max_global) or min_global == max_global:
        min_global, max_global = -1, 1
    ax_scatter.set_xlim(min_global, max_global)
    ax_scatter.set_ylim(min_global, max_global)

    # Metrics

    text_base_y = 0.95
    for j, (circuit_name, _) in enumerate(avg_df.groupby('Circuit')):
        current_long_df = long_df[long_df['Circuit'] == circuit_name]
        if current_long_df.shape[0] < 2:
            continue
        r2 = r2_score(current_long_df['Expected_value_unscaled'], current_long_df['Predicted_value_unscaled'])
        large_expected = current_long_df[(current_long_df["Expected_value_unscaled"] >= 0.05) | (current_long_df["Expected_value_unscaled"] <= -0.05)]
        large_r2 = r2_score(large_expected["Expected_value_unscaled"], large_expected["Predicted_value_unscaled"]) if len(large_expected) > 1 else np.nan
        mse = root_mean_squared_error(current_long_df['Expected_value_unscaled'], current_long_df['Predicted_value_unscaled'])
        mae = mean_absolute_error(current_long_df['Expected_value_unscaled'], current_long_df['Predicted_value_unscaled'])

        # hist distances for distribution measures
        expected_hist, bin_edges = np.histogram(current_long_df['Expected_value_unscaled'], bins=2000)
        predicted_hist, _ = np.histogram(current_long_df['Predicted_value_unscaled'], bins=bin_edges)
        P_dist = expected_hist / (np.sum(expected_hist) + 1e-12)
        Q_dist = predicted_hist / (np.sum(predicted_hist) + 1e-12)

        js = js_divergence(P_dist, Q_dist)

        size = 16
        y_position = text_base_y - (j * 0.14)
        ax_scatter.text(0.05, y_position, f'{circuit_name}:\n $R^2$ = {r2:.2f}', transform=ax_scatter.transAxes, fontsize=size, verticalalignment='top', fontweight="bold")
        # ax_scatter.text(0.15, y_position, f'', transform=ax_scatter.transAxes, fontsize=size, verticalalignment='top', fontweight="bold")

    try:
        kde_expected = sns.kdeplot(data=long_df, x='Predicted_value_unscaled', ax=ax_hist_top)
        kde_expected_line = kde_expected.get_lines()[0]
        kde_expected_data = kde_expected_line.get_ydata()
        max_density_expected = kde_expected_data.max()
    except Exception:
        max_density_expected = 0.0
    ax_hist_top.cla()

    try:
        kde_predicted = sns.kdeplot(data=long_df, y='Expected_value_unscaled', ax=ax_hist_right)
        kde_predicted_line = kde_predicted.get_lines()[0]
        kde_predicted_data = kde_predicted_line.get_xdata()
        max_density_predicted = kde_predicted_data.max()
    except Exception:
        max_density_predicted = 0.0
    ax_hist_right.cla()

    max_density = max(max_density_expected, max_density_predicted, 1e-6)
    ylim_max = max_density * 1.8
    if ylim_max < 0.5:
        ylim_max = 0.5

    sns.kdeplot(data=long_df, y='Expected_value_unscaled', ax=ax_hist_right, color='skyblue', fill=True, alpha=0.6, label='Expected')

    # per-circuit distributions on top histogram
    for circuit_name, circuit_group in long_df.groupby('Circuit'):
        try:
            sns.kdeplot(data=circuit_group, x='Predicted_value_unscaled', ax=ax_hist_top,
                        clip=ax_scatter.get_xlim(), alpha=0.6, fill=True, label=circuit_name)
        except Exception:
            pass

    # overlay the overall expected distribution as a thin black line
    try:
        sns.kdeplot(data=long_df, x='Expected_value_unscaled', ax=ax_hist_top, color='black', fill=False, alpha=0.6, clip=ax_scatter.get_xlim())

    except Exception:
        pass

    ax_hist_top.set_xlabel('')
    ax_hist_top.set_ylabel('Density', fontweight='bold')
    ax_hist_top.set_yscale('log')
    ax_hist_right.set_xscale('log')
    ax_hist_right.set_xlabel('Density', fontweight='bold')
    ax_hist_right.set_ylabel('')
    ax_hist_top.set_ylim(0.5, ylim_max)
    ax_hist_right.set_xlim(0.5, ylim_max)
    #ax_hist_top.legend()

    # remove tick labels where appropriate
    plt.setp(ax_hist_top.get_xticklabels(), visible=False)
    plt.setp(ax_hist_right.get_yticklabels(), visible=False)
    ax_scatter.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax_scatter.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # molecule image
    image_path = f"images/{molecule_name}.png"
    if os.path.exists(image_path):
        img = mpimg.imread(image_path)
        ax_img.imshow(img)
    ax_img.axis('off')



def plot_4x4_molecules(csv_file_path, scaler_file_path):
    # load data & scaler
    df = pd.read_csv(csv_file_path)
    scaler = joblib.load(scaler_file_path)

    df = df.explode(['Expected', 'Predicted'])
    df['Expected_value'] = pd.to_numeric(df['Expected'])
    df['Predicted_value'] = pd.to_numeric(df['Predicted'])

    # unscale using scaler
    df['Predicted_value_unscaled'] = scaler.inverse_transform(df[['Predicted_value']])
    df['Expected_value_unscaled'] = scaler.inverse_transform(df[['Expected_value']])

    n_bins = 3000
    df['Expected_value_bin'] = pd.qcut(df['Expected_value_unscaled'], q=n_bins, labels=False, duplicates='drop')

    avg_df = df.groupby(['Molecule', 'Circuit', 'Expected_value_bin']).agg(
        expected_value_mean=('Expected_value_unscaled', 'mean'),
        predicted_value_mean=('Predicted_value_unscaled', 'mean')
    ).reset_index()

    rename_dict = {
        'MultiControl_0_1': 'MultiControl (L=1)',
        'Classical_0_1': 'Classical SVR',
        'Hubregtsen_0_1': 'Hubregtsen (L=1)'
    }
    avg_df['Circuit'] = avg_df['Circuit'].replace(rename_dict)
    df['Circuit'] = df['Circuit'].replace(rename_dict)

    molecules = ["Formaldehyde", "Water", "Ethylene", "Ammonia"]

    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.25)

    for i, molecule in enumerate(molecules):
        row, col = divmod(i, 2)
        gs_sub = gs[row, col].subgridspec(4, 4, wspace=0.15, hspace=0.15)
        ax_scatter = fig.add_subplot(gs_sub[1:, :3])
        ax_hist_top = fig.add_subplot(gs_sub[0, :3], sharex=ax_scatter)
        ax_hist_right = fig.add_subplot(gs_sub[1:, 3:], sharey=ax_scatter)
        ax_img = fig.add_subplot(gs_sub[0, 3:])
        mol_avg_df = avg_df[avg_df["Molecule"] == molecule]
        mol_long_df = df[df["Molecule"] == molecule]
        plot_molecule_data(ax_scatter, ax_hist_top, ax_hist_right, ax_img, mol_avg_df, mol_long_df, molecule)

    # plt.suptitle("Quantum Kernel Transferability – Formaldehyde, Water, Ethylene, Ammonia", fontsize=18, fontweight='bold')
    # plt.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.95, top=0.95, bottom=0.05)
    plt.savefig("graphs/Formaldehyde_4x4.png", dpi=300)
    
    molecules = ["Ammonia", "Methane", "Methanol", "Ethane"]
    for i, molecule in enumerate(molecules):
        fig = plt.figure(figsize=(16, 16))
        
        gs_sub = GridSpec(4, 4, figure=fig, wspace=0.15, hspace=0.15)
        ax_scatter = fig.add_subplot(gs_sub[1:, :3])
        ax_hist_top = fig.add_subplot(gs_sub[0, :3], sharex=ax_scatter)
        ax_hist_right = fig.add_subplot(gs_sub[1:, 3:], sharey=ax_scatter)
        ax_img = fig.add_subplot(gs_sub[0, 3:])
        mol_avg_df = avg_df[avg_df["Molecule"] == molecule]
        mol_long_df = df[df["Molecule"] == molecule]
        plot_molecule_data(ax_scatter, ax_hist_top, ax_hist_right, ax_img, mol_avg_df, mol_long_df, molecule)
        plt.savefig(f"graphs/Formaldehyde_{molecule}_exp_pred.png", dpi=300)
#"transferability/STO-3G/Water-Formaldehyde/Expected_Predicted.csv"
plot_4x4_molecules("Form_Expected_Predicted.csv", "Data/STO-3G/Water/scaled/t2_scaler.bin")
