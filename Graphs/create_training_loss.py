### Some of the earlier data from our results did not save correctly and so had to be recovered
### from output logs.

import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".cache"))

import pandas as pd
import matplotlib.pyplot as plt

def visualize_training_results(csv_file="paper_results/training/training_results.csv"):
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please ensure the CSV exists.")
        return

    df = pd.read_csv(csv_file)

    #Group by Name and calculate Mean and Standard Deviation
    stats = df.groupby(['Name', 'Epoch']).agg({
        'MSE': ['mean', 'std'],
        'R2': ['mean', 'std']
    }).reset_index()

    stats.columns = ['Name', 'Epoch', 'MSE_mean', 'MSE_std', 'R2_mean', 'R2_std']

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    
    # Unique models in the dataset
    models = stats['Name'].unique()

    for model in models:
        model_data = stats[stats['Name'] == model]
        
        ax1.plot(model_data['Epoch'], model_data['MSE_mean'], label=f'{model}')

        ax2.plot(model_data['Epoch'], model_data['R2_mean'], label=f'{model}')

    # Styling MAE
    ax1.set_title('Training $MSE$')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()

    # Styling R^2
    ax2.set_title('Training $R^2$')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('$R^2$ Score')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.set_ylim(0,1)
    ax2.legend()

    plt.tight_layout()
    output_dir = os.getenv("FIGURE_OUTPUT_DIR", "paper_results/regenerated_figures")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_visualization.png")
    plt.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.close()
    print(f"Visualization saved as '{output_path}'")

if __name__ == "__main__":
    visualize_training_results()
