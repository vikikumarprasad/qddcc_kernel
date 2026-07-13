import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from molecule_image_assets import molecule_image_path
from scipy.stats import entropy, wasserstein_distance
from scipy.special import rel_entr


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

    circuit_order = ["Hubregtsen RUD 1", "Classical SVR"][::-1]  # keep order consistent

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
    ax_scatter.set_xlabel(r'Predicted $t^{ab}_{ij(CCSD)}$', fontsize=12)
    ax_scatter.set_ylabel(r'Expected $t^{ab}_{ij(CCSD)}$', fontsize=12)
    ax_scatter.grid(True, linestyle='--', alpha=0.5)
    ax_scatter.set_aspect('equal', adjustable='box')

    # compute limits for graphing
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

        expected_hist, bin_edges = np.histogram(current_long_df['Expected_value_unscaled'], bins=2000)
        predicted_hist, _ = np.histogram(current_long_df['Predicted_value_unscaled'], bins=bin_edges)
        P_dist = expected_hist / (np.sum(expected_hist) + 1e-12)
        Q_dist = predicted_hist / (np.sum(predicted_hist) + 1e-12)

        js = js_divergence(P_dist, Q_dist)

        size = 13
        y_position = text_base_y - (j * 0.15)
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

    # per-circuit predicted distirbutions
    for circuit_name, circuit_group in long_df.groupby('Circuit'):
        try:
            sns.kdeplot(data=circuit_group, x='Predicted_value_unscaled', ax=ax_hist_top,
                        clip=ax_scatter.get_xlim(), alpha=0.6, fill=True, label=circuit_name)
        except Exception:
            pass

    # overlay the expected distribution as a thin black line
    try:
        sns.kdeplot(data=long_df, x='Expected_value_unscaled', ax=ax_hist_top, color='black', fill=False, alpha=0.6, clip=ax_scatter.get_xlim())

    except Exception:
        pass

    ax_hist_top.set_xlabel('')
    ax_hist_top.set_ylabel('Density')
    ax_hist_top.set_yscale('log')
    ax_hist_right.set_xscale('log')
    ax_hist_right.set_xlabel('Density')
    ax_hist_right.set_ylabel('')
    ax_hist_top.set_ylim(0.5, ylim_max)
    ax_hist_right.set_xlim(0.5, ylim_max)
    ax_hist_top.legend()

    # remove tick labels where appropriate
    plt.setp(ax_hist_top.get_xticklabels(), visible=False)
    plt.setp(ax_hist_right.get_yticklabels(), visible=False)
    ax_scatter.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_scatter.yaxis.set_major_locator(MaxNLocator(nbins=6))

    image_path = molecule_image_path(molecule_name, required=False)
    if image_path is not None:
        img = mpimg.imread(image_path)
        ax_img.imshow(img)
    ax_img.axis('off')

def plot_custom_layout(csv_file_path, scaler_file_path, selected_molecule):

    all_molecules = ["Formaldehyde", "Water", "Ammonia", "Methane", "Ethane", "Methanol", "Ethylene"]
    if selected_molecule not in all_molecules:
        raise ValueError(f"Molecule must be one of {all_molecules}")
    
    remaining_molecules = [m for m in all_molecules if m != selected_molecule]

    df = pd.read_csv(csv_file_path)
    scaler = joblib.load(scaler_file_path)
    
    # Ensure names
    df['Molecule'] = df['Molecule'].str.capitalize()

    #Rename names in data to how presented
    rename_dict = {'Hubregtsen_0_1': 'Hubregtsen (L=1)', 'Classical_0_1': 'Classical SVR'}
    df['Circuit'] = df['Circuit'].replace(rename_dict)

    # Filter after standardizing names
    df = df[df['Molecule'].isin(all_molecules)]

    df = df.explode(['Expected', 'Predicted'])
    df['Expected_value'] = pd.to_numeric(df['Expected'])
    df['Predicted_value'] = pd.to_numeric(df['Predicted'])
    
    # Unscale
    df['Predicted_value_unscaled'] = scaler.inverse_transform(df[['Predicted_value']])
    df['Expected_value_unscaled'] = scaler.inverse_transform(df[['Expected_value']])

    def bin_molecule(group):
        n_bins = min(3000, len(group))
        try:
            group['Expected_value_bin'] = pd.qcut(group['Expected_value_unscaled'], q=n_bins, labels=False, duplicates='drop')
        except ValueError:
            group['Expected_value_bin'] = range(len(group))
        return group

    df = df.groupby(['Molecule', 'Circuit'], group_keys=False).apply(bin_molecule)

    # Aggregate
    avg_df = df.groupby(['Molecule', 'Circuit', 'Expected_value_bin']).agg(
        expected_value_mean=('Expected_value_unscaled', 'mean'),
        predicted_value_mean=('Predicted_value_unscaled', 'mean')
    ).reset_index()

    fig = plt.figure(figsize=(25, 20)) 
    

    def render_to_slot(gs_slot, mol_name):
        # subgrid remains 4x4 for the internal scatter/hist/image layout
        sub_gs = gs_slot.subgridspec(4, 4, wspace=0.08, hspace=0.08)
        ax_scatter = fig.add_subplot(sub_gs[1:, :3])
        ax_hist_top = fig.add_subplot(sub_gs[0, :3], sharex=ax_scatter)
        ax_hist_right = fig.add_subplot(sub_gs[1:, 3:], sharey=ax_scatter)
        ax_img = fig.add_subplot(sub_gs[0, 3:])
        
        mol_avg = avg_df[avg_df["Molecule"] == mol_name]
        mol_long = df[df["Molecule"] == mol_name]
        
        plot_molecule_data(ax_scatter, ax_hist_top, ax_hist_right, ax_img, mol_avg, mol_long, mol_name)

    fig = plt.figure(figsize=(25, 20))

    main_gs = GridSpec(24, 32, figure=fig, wspace=0.0, hspace=0.0)

    render_to_slot(main_gs[5:19, 0:14], selected_molecule)

    right_slots = [
        main_gs[0:7,   16:23], main_gs[0:7,   25:32],
        main_gs[8:15,  16:23], main_gs[8:15,  25:32],
        main_gs[16:23, 16:23], main_gs[16:23, 25:32],
    ]


    for i, mol in enumerate(remaining_molecules):
        render_to_slot(right_slots[i], mol)

    plt.savefig(f"Transferability_{selected_molecule}_Layout.png", dpi=300, bbox_inches='tight')
    plt.show()

source_path = "paper_results/transferability/STO-3G/Formaldehyde/Expected_Predicted.csv"
plot_custom_layout(
    source_path,
    "data/training/STO-3G/Water/scaled/t2_scaler.bin",
    "Formaldehyde",
)
