"""
Script for generating individual walk plots
Saves plots to: graphics/individual_walks/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_individual_walk(csv_file, model_name, output_dir="graphics/individual_walks"):
    """
    Generate individual walk plot for a single model
    
    Args:
        csv_file: Path to CSV file with walk results
        model_name: Name of the model
        output_dir: Directory to save plots
    """
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        return None
        
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = pd.read_csv(csv_file)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Use first 200 points for clarity
    df_subset = df.head(200) if len(df) > 200 else df
    
    # Plot 1: Full time series comparison
    ax1 = axes[0, 0]
    ax1.plot(df_subset['step'], df_subset['actual'], 'b-', 
             label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(df_subset['step'], df_subset['forecast'], 'r--', 
             label='Forecast', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title(f'{model_name} - Predictions vs Actual')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error over time
    ax2 = axes[0, 1]
    ax2.plot(df_subset['step'], df_subset['absolute_error'], 'g-', 
             label='Absolute Error', linewidth=2, alpha=0.8)
    ax2.fill_between(df_subset['step'], df_subset['absolute_error'], 
                     alpha=0.3, color='green')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(f'{model_name} - Error Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(df['absolute_error'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(df['absolute_error'].mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {df["absolute_error"].mean():.4f}')
    ax3.axvline(df['absolute_error'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f'Median: {df["absolute_error"].median():.4f}')
    ax3.set_xlabel('Absolute Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title(f'{model_name} - Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot - Actual vs Forecast
    ax4 = axes[1, 1]
    ax4.scatter(df_subset['actual'], df_subset['forecast'], alpha=0.6, s=20)
    
    # Perfect prediction line
    min_val = min(df_subset['actual'].min(), df_subset['forecast'].min())
    max_val = max(df_subset['actual'].max(), df_subset['forecast'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', 
             linewidth=2, label='Perfect Prediction')
    
    ax4.set_xlabel('Actual Values')
    ax4.set_ylabel('Forecast Values')
    ax4.set_title(f'{model_name} - Actual vs Forecast Scatter')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Calculate R²
    correlation_matrix = np.corrcoef(df_subset['actual'], df_subset['forecast'])
    r_squared = correlation_matrix[0, 1] ** 2
    ax4.text(0.05, 0.95, f'R² = {r_squared:.4f}', transform=ax4.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'{model_name} - Comprehensive Walk Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Save plot
    filename = f"{model_name.lower().replace(' ', '_')}_walk_analysis.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {filepath}")
    plt.show()
    
    # Print statistics
    mae = df['absolute_error'].mean()
    median_error = df['absolute_error'].median()
    max_error = df['absolute_error'].max()
    min_error = df['absolute_error'].min()
    std_error = df['absolute_error'].std()
    
    print(f"""
📊 {model_name} Statistics:
   Mean Absolute Error: {mae:.6f}
   Median Absolute Error: {median_error:.6f}
   Standard Deviation: {std_error:.6f}
   Max Error: {max_error:.6f}
   Min Error: {min_error:.6f}
   R² Score: {r_squared:.4f}
""")
    
    return {
        'model': model_name,
        'mae': mae,
        'median_error': median_error,
        'max_error': max_error,
        'min_error': min_error,
        'std_error': std_error,
        'r_squared': r_squared,
        'filepath': filepath
    }

def generate_all_individual_walks():
    """Generate individual walk plots for all available models"""
    base_dir = "run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv',
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    print("🎯 Generating Individual Walk Plots")
    print("=" * 50)
    
    results = []
    
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        print(f"\n📈 Processing {model_name}...")
        
        result = plot_individual_walk(filepath, model_name)
        if result:
            results.append(result)
    
    # Save summary statistics
    if results:
        summary_df = pd.DataFrame(results)
        summary_path = "graphics/individual_walks/summary_statistics.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary statistics saved: {summary_path}")
        
        # Print ranking
        print(f"\n🏆 Model Ranking by MAE:")
        ranked = summary_df.sort_values('mae')
        for i, (_, row) in enumerate(ranked.iterrows(), 1):
            print(f"  {i}. {row['model']}: {row['mae']:.6f}")
    
    print(f"\n🎉 Individual walk analysis complete!")
    print(f"📁 All plots saved to: graphics/individual_walks/")
    
    return results

if __name__ == "__main__":
    generate_all_individual_walks()