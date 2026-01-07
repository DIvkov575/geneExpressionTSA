import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_hmm_walk_forward_results(results_file_path, output_dir="graphics"):
    """
    Plots the actual vs. forecasted values from the HMM walk-forward results.

    Args:
        results_file_path (str): Absolute path to the hmm_walk_forward_results.csv file.
        output_dir (str): Directory to save the plot.
    """
    try:
        df = pd.read_csv(results_file_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return

    if df.empty:
        print("Error: Results file is empty.")
        return

    plt.figure(figsize=(15, 7))
    plt.plot(df['step'], df['actual'], label='Actual', color='blue', alpha=0.7)
    plt.plot(df['step'], df['forecast'], label='Forecast', color='red', linestyle='--', alpha=0.7)
    
    plt.title('HMM Walk-Forward Forecast vs. Actuals')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'hmm_walk_forward_plot.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.close()

if __name__ == "__main__":
    # Construct the absolute path to the results file
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(current_script_dir, '..', 'run_forward', 'hmm_walk_forward_results.csv')
    
    # Construct the absolute path to the output directory
    graphics_dir = os.path.join(current_script_dir)

    plot_hmm_walk_forward_results(results_file, graphics_dir)
