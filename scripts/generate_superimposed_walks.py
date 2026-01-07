"""
Script for generating superimposed walk plots with logarithmic scale
Saves plots to: graphics/comparison_plots/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_superimposed_walks_log(output_dir="graphics/comparison_plots"):
    """
    Generate superimposed walk plots with logarithmic scale
    
    Args:
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    base_dir = "run_forward"
    
    models = {
        'Naive': 'naive_walk_forward_results.csv',
        'NBEATS': 'nbeats_walk_forward_results.csv', 
        'TFT': 'tft_walk_forward_results.csv',
        'GBM': 'gbm_walk_forward_results.csv',
        'ARIMA v3': 'arima_v3_walk_forward_results.csv',
        'ARIMA Stats': 'arima_statsmodels_walk_forward_results.csv'
    }
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # Load all available data
    model_data = {}
    for model_name, filename in models.items():
        filepath = os.path.join(base_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            model_data[model_name] = df
            print(f"✅ Loaded {model_name}: {len(df)} data points")
        else:
            print(f"❌ File not found: {filepath}")
    
    if not model_data:
        print("❌ No model data found!")
        return
    
    # Create comprehensive comparison plots
    fig = plt.figure(figsize=(20, 16))
    
    # Get a common subset for comparison (first 150 steps)
    max_steps = min(len(df) for df in model_data.values())
    display_steps = min(150, max_steps)
    
    # Plot 1: Linear Scale - Predictions vs Actual
    ax1 = plt.subplot(3, 3, 1)
    actual_plotted = False
    
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        
        if not actual_plotted:
            ax1.plot(df_subset['step'], df_subset['actual'], 'k-', 
                    linewidth=3, label='Actual', alpha=0.9)
            actual_plotted = True
        
        ax1.plot(df_subset['step'], df_subset['forecast'], '--', 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title('All Models: Linear Scale Predictions')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log Scale - Predictions vs Actual
    ax2 = plt.subplot(3, 3, 2)
    actual_plotted = False
    
    epsilon = 1e-10
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        
        # Handle log scale for negative/zero values
        actual_log = np.where(df_subset['actual'] <= 0, epsilon, np.abs(df_subset['actual']))
        forecast_log = np.where(df_subset['forecast'] <= 0, epsilon, np.abs(df_subset['forecast']))
        
        if not actual_plotted:
            ax2.plot(df_subset['step'], actual_log, 'k-', 
                    linewidth=3, label='Actual', alpha=0.9)
            actual_plotted = True
        
        ax2.plot(df_subset['step'], forecast_log, '--', 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_title('All Models: Log Scale Predictions')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Evolution - Linear Scale
    ax3 = plt.subplot(3, 3, 3)
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        ax3.plot(df_subset['step'], df_subset['absolute_error'], 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error Evolution: Linear Scale')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error Evolution - Log Scale
    ax4 = plt.subplot(3, 3, 4)
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        errors_log = np.where(df_subset['absolute_error'] <= 0, epsilon, df_subset['absolute_error'])
        ax4.plot(df_subset['step'], errors_log, 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Absolute Error (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_title('Error Evolution: Log Scale')
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: MAE Comparison Bar Chart
    ax5 = plt.subplot(3, 3, 5)
    model_names = list(model_data.keys())
    mae_values = [df['absolute_error'].mean() for df in model_data.values()]
    
    bars = ax5.bar(model_names, mae_values, 
                  color=colors[:len(model_names)], alpha=0.7)
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Mean Absolute Error')
    ax5.set_title('MAE Comparison')
    ax5.set_yscale('log')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 6: Error Distribution Box Plot
    ax6 = plt.subplot(3, 3, 6)
    error_data = []
    labels = []
    
    for model_name, df in model_data.items():
        # Use subset and handle extreme values for visualization
        errors = df.head(display_steps)['absolute_error'].values
        errors_clipped = np.clip(errors, 1e-10, np.percentile(errors, 95))  # Clip to 95th percentile
        error_data.append(errors_clipped)
        labels.append(model_name)
    
    box_plot = ax6.boxplot(error_data, tick_labels=labels, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors[:len(error_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.set_xlabel('Model')
    ax6.set_ylabel('Absolute Error (Clipped)')
    ax6.set_yscale('log')
    ax6.set_title('Error Distribution (95th Percentile)')
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 7: Cumulative Error
    ax7 = plt.subplot(3, 3, 7)
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        cumulative_error = np.cumsum(df_subset['absolute_error'])
        ax7.plot(df_subset['step'], cumulative_error, 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax7.set_xlabel('Time Step')
    ax7.set_ylabel('Cumulative Error')
    ax7.set_title('Cumulative Error Evolution')
    ax7.set_yscale('log')
    ax7.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance Metrics Table
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('tight')
    ax8.axis('off')
    
    # Create performance metrics table
    metrics_data = []
    for model_name, df in model_data.items():
        mae = df['absolute_error'].mean()
        median_error = df['absolute_error'].median()
        max_error = df['absolute_error'].max()
        std_error = df['absolute_error'].std()
        
        metrics_data.append([
            model_name,
            f"{mae:.4f}",
            f"{median_error:.4f}",
            f"{max_error:.4f}",
            f"{std_error:.4f}"
        ])
    
    table = ax8.table(cellText=metrics_data,
                     colLabels=['Model', 'MAE', 'Median', 'Max', 'Std'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax8.set_title('Performance Metrics Summary')
    
    # Plot 9: Model Rankings
    ax9 = plt.subplot(3, 3, 9)
    
    # Rank by MAE
    mae_ranking = sorted([(name, df['absolute_error'].mean()) 
                         for name, df in model_data.items()], 
                        key=lambda x: x[1])
    
    ranking_text = "Model Rankings (by MAE):\\n\\n"
    for i, (name, mae) in enumerate(mae_ranking, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
        ranking_text += f"{medal} {name}: {mae:.4f}\\n"
    
    ax9.text(0.1, 0.9, ranking_text, transform=ax9.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis('off')
    ax9.set_title('Model Performance Ranking')
    
    plt.suptitle('Comprehensive Model Comparison - Forward Walk Results (Logarithmic Scale)', 
                fontsize=18, y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    filepath = os.path.join(output_dir, "superimposed_walks_logarithmic.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive comparison saved: {filepath}")
    plt.show()
    
    # Create simplified version for cleaner viewing
    create_simplified_comparison(model_data, output_dir, colors)
    
    return filepath

def create_simplified_comparison(model_data, output_dir, colors):
    """Create a simplified 2x2 comparison plot"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    display_steps = min(100, min(len(df) for df in model_data.values()))
    epsilon = 1e-10
    
    # Plot 1: Linear Predictions
    ax1 = axes[0, 0]
    actual_plotted = False
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        
        if not actual_plotted:
            ax1.plot(df_subset['step'], df_subset['actual'], 'k-', 
                    linewidth=3, label='Actual', alpha=0.9)
            actual_plotted = True
        
        ax1.plot(df_subset['step'], df_subset['forecast'], '--', 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.set_title('All Models: Predictions (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log Predictions
    ax2 = axes[0, 1]
    actual_plotted = False
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        
        actual_log = np.where(df_subset['actual'] <= 0, epsilon, np.abs(df_subset['actual']))
        forecast_log = np.where(df_subset['forecast'] <= 0, epsilon, np.abs(df_subset['forecast']))
        
        if not actual_plotted:
            ax2.plot(df_subset['step'], actual_log, 'k-', 
                    linewidth=3, label='Actual', alpha=0.9)
            actual_plotted = True
        
        ax2.plot(df_subset['step'], forecast_log, '--', 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Value (Log Scale)')
    ax2.set_yscale('log')
    ax2.set_title('All Models: Predictions (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error Evolution Linear
    ax3 = axes[1, 0]
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        ax3.plot(df_subset['step'], df_subset['absolute_error'], 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Error Evolution (Linear Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Error Evolution Log
    ax4 = axes[1, 1]
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        errors_log = np.where(df_subset['absolute_error'] <= 0, epsilon, df_subset['absolute_error'])
        ax4.plot(df_subset['step'], errors_log, 
                color=colors[i % len(colors)], linewidth=2, 
                label=f'{model_name}', alpha=0.8)
    
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Absolute Error (Log Scale)')
    ax4.set_yscale('log')
    ax4.set_title('Error Evolution (Log Scale)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Simplified Model Comparison - Linear vs Logarithmic Scale', fontsize=16)
    plt.tight_layout()
    
    # Save simplified plot
    filepath = os.path.join(output_dir, "simplified_superimposed_walks.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Simplified comparison saved: {filepath}")
    plt.show()

if __name__ == "__main__":
    print("🎯 Generating Superimposed Walks with Logarithmic Scale")
    print("=" * 60)
    plot_superimposed_walks_log()
    print("🎉 Superimposed walk analysis complete!")
    print("📁 Plots saved to: graphics/comparison_plots/")