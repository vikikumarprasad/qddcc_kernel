### Some of the earlier data from our results did not save correctly and so had to be recovered
### from output logs.

import pandas as pd
import matplotlib.pyplot as plt
import os

def visualize_training_results(csv_file="training_results.csv"):
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Unique models in the dataset
    models = stats['Name'].unique()

    for model in models:
        model_data = stats[stats['Name'] == model]
        
        # MSE (Left Subplot)
        ax1.plot(model_data['Epoch'], model_data['MSE_mean'], label=f'{model}')
        ax1.fill_between(
            model_data['Epoch'], 
            model_data['MSE_mean'] - model_data['MSE_std'], 
            model_data['MSE_mean'] + model_data['MSE_std'], 
            alpha=0.1
        )

        # R^2 (Right Subplot)
        ax2.plot(model_data['Epoch'], model_data['R2_mean'], label=f'{model}')
        ax2.fill_between(
            model_data['Epoch'], 
            model_data['R2_mean'] - model_data['R2_std'], 
            model_data['R2_mean'] + model_data['R2_std'], 
            alpha=0.1
        )

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
    plt.savefig('heatmaps/training_visualization.png', dpi=300)
    print("Visualization saved as 'training_visualization.png'")
    plt.show()

if __name__ == "__main__":
    visualize_training_results()
