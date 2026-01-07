"""
Script for generating detailed error analysis plots
Saves plots to: graphics/error_analysis/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats

def plot_error_analysis(output_dir="graphics/error_analysis"):
    """
    Generate comprehensive error analysis plots
    
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
    
    # Create error analysis plots
    create_error_distribution_plots(model_data, output_dir, colors)
    create_error_statistics_plots(model_data, output_dir, colors)
    create_error_correlation_plots(model_data, output_dir, colors)
    create_performance_ranking_plots(model_data, output_dir, colors)
    
    print("🎉 Error analysis complete!")
    print(f"📁 All plots saved to: {output_dir}/")

def create_error_distribution_plots(model_data, output_dir, colors):
    """Create error distribution analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (model_name, df) in enumerate(model_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Error histogram with statistics
        errors = df['absolute_error']
        
        # Remove extreme outliers for better visualization
        q99 = np.percentile(errors, 99)
        errors_filtered = errors[errors <= q99]
        
        ax.hist(errors_filtered, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        
        # Add statistics lines
        mean_error = errors.mean()
        median_error = errors.median()
        
        ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_error:.4f}')
        ax.axvline(median_error, color='blue', linestyle='--', linewidth=2,
                  label=f'Median: {median_error:.4f}')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{model_name} - Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f"""
        Count: {len(errors)}
        Std: {errors.std():.4f}
        Skew: {stats.skew(errors):.2f}
        Kurt: {stats.kurtosis(errors):.2f}
        """
        ax.text(0.65, 0.95, stats_text.strip(), transform=ax.transAxes,
               verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(model_data), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Error Distribution Analysis by Model', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "error_distributions.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Error distributions saved: {filepath}")
    plt.show()

def create_error_statistics_plots(model_data, output_dir, colors):
    """Create error statistics comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    model_names = list(model_data.keys())
    
    # Calculate statistics
    mae_values = [df['absolute_error'].mean() for df in model_data.values()]
    median_values = [df['absolute_error'].median() for df in model_data.values()]
    std_values = [df['absolute_error'].std() for df in model_data.values()]
    max_values = [df['absolute_error'].max() for df in model_data.values()]
    
    # Plot 1: MAE Comparison
    ax1 = axes[0, 0]
    bars = ax1.bar(model_names, mae_values, color=colors[:len(model_names)], alpha=0.7)
    ax1.set_ylabel('Mean Absolute Error')
    ax1.set_title('Mean Absolute Error Comparison')
    ax1.set_yscale('log')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels
    for bar, value in zip(bars, mae_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Median Error Comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(model_names, median_values, color=colors[:len(model_names)], alpha=0.7)
    ax2.set_ylabel('Median Absolute Error')
    ax2.set_title('Median Absolute Error Comparison')
    ax2.set_yscale('log')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, value in zip(bars, median_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Standard Deviation Comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(model_names, std_values, color=colors[:len(model_names)], alpha=0.7)
    ax3.set_ylabel('Standard Deviation of Error')
    ax3.set_title('Error Variability Comparison')
    ax3.set_yscale('log')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, value in zip(bars, std_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Max Error Comparison
    ax4 = axes[1, 1]
    bars = ax4.bar(model_names, max_values, color=colors[:len(model_names)], alpha=0.7)
    ax4.set_ylabel('Maximum Absolute Error')
    ax4.set_title('Maximum Error Comparison')
    ax4.set_yscale('log')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, value in zip(bars, max_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Error Statistics Comparison', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "error_statistics.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Error statistics saved: {filepath}")
    plt.show()

def create_error_correlation_plots(model_data, output_dir, colors):
    """Create error correlation and pattern analysis plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, (model_name, df) in enumerate(model_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # Scatter plot: Actual vs Error
        subset = df.head(200)  # Use subset for clarity
        scatter = ax.scatter(subset['actual'], subset['absolute_error'], 
                           alpha=0.6, s=20, color=colors[i])
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{model_name} - Error vs Actual Values')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Calculate correlation
        correlation = np.corrcoef(subset['actual'], subset['absolute_error'])[0, 1]
        ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(model_data), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Error Pattern Analysis', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "error_patterns.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Error patterns saved: {filepath}")
    plt.show()

def create_performance_ranking_plots(model_data, output_dir, colors):
    """Create comprehensive performance ranking visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate multiple metrics
    metrics = {}
    for model_name, df in model_data.items():
        errors = df['absolute_error']
        metrics[model_name] = {
            'mae': errors.mean(),
            'median': errors.median(),
            'std': errors.std(),
            'max': errors.max(),
            'q95': np.percentile(errors, 95),
            'q75': np.percentile(errors, 75)
        }
    
    # Plot 1: Multi-metric ranking radar chart (simplified as bar chart)
    ax1 = axes[0, 0]
    
    # Normalize metrics for comparison (lower is better, so invert)
    mae_scores = [1 / metrics[name]['mae'] for name in model_data.keys()]
    std_scores = [1 / metrics[name]['std'] for name in model_data.keys()]
    max_scores = [1 / metrics[name]['max'] for name in model_data.keys()]
    
    x = np.arange(len(model_data))
    width = 0.25
    
    ax1.bar(x - width, mae_scores, width, label='1/MAE', alpha=0.7)
    ax1.bar(x, std_scores, width, label='1/STD', alpha=0.7)
    ax1.bar(x + width, max_scores, width, label='1/MAX', alpha=0.7)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Inverse Error (Higher = Better)')
    ax1.set_title('Multi-Metric Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_data.keys(), rotation=45, ha='right')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Plot 2: Error percentile comparison
    ax2 = axes[0, 1]
    
    model_names = list(model_data.keys())
    percentiles = [50, 75, 90, 95, 99]
    
    for i, model_name in enumerate(model_names):
        errors = model_data[model_name]['absolute_error']
        perc_values = [np.percentile(errors, p) for p in percentiles]
        ax2.plot(percentiles, perc_values, 'o-', 
                color=colors[i], label=model_name, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Percentile')
    ax2.set_ylabel('Error Value')
    ax2.set_title('Error Percentile Analysis')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance ranking table
    ax3 = axes[1, 0]
    ax3.axis('tight')
    ax3.axis('off')
    
    # Create ranking data
    ranking_data = []
    for model_name, metric_dict in metrics.items():
        ranking_data.append([
            model_name,
            f"{metric_dict['mae']:.4f}",
            f"{metric_dict['median']:.4f}",
            f"{metric_dict['std']:.4f}",
            f"{metric_dict['q95']:.4f}"
        ])
    
    # Sort by MAE
    ranking_data.sort(key=lambda x: float(x[1]))
    
    table = ax3.table(cellText=ranking_data,
                     colLabels=['Model', 'MAE', 'Median', 'Std Dev', '95th %'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Color code the rows
    for i in range(len(ranking_data)):
        for j in range(len(ranking_data[0])):
            if i == 0:  # Best performer
                table[(i+1, j)].set_facecolor('#90EE90')  # Light green
            elif i == 1:  # Second best
                table[(i+1, j)].set_facecolor('#FFE4B5')  # Light orange
            elif i == len(ranking_data) - 1:  # Worst performer
                table[(i+1, j)].set_facecolor('#FFB6C1')  # Light pink
    
    ax3.set_title('Performance Ranking (Sorted by MAE)')
    
    # Plot 4: Score summary
    ax4 = axes[1, 1]
    
    # Create composite score (lower MAE + lower std + lower max = better)
    composite_scores = []
    for model_name in model_data.keys():
        mae_rank = len([m for m in metrics.values() if m['mae'] < metrics[model_name]['mae']]) + 1
        std_rank = len([m for m in metrics.values() if m['std'] < metrics[model_name]['std']]) + 1
        max_rank = len([m for m in metrics.values() if m['max'] < metrics[model_name]['max']]) + 1
        
        composite_score = (mae_rank + std_rank + max_rank) / 3
        composite_scores.append(composite_score)
    
    sorted_indices = np.argsort(composite_scores)
    sorted_names = [list(model_data.keys())[i] for i in sorted_indices]
    sorted_scores = [composite_scores[i] for i in sorted_indices]
    
    bars = ax4.barh(sorted_names, sorted_scores, 
                   color=[colors[i] for i in sorted_indices], alpha=0.7)
    ax4.set_xlabel('Composite Rank Score (Lower = Better)')
    ax4.set_title('Overall Performance Ranking')
    
    # Add rank numbers
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'#{i+1} ({score:.1f})', va='center', fontsize=10)
    
    plt.suptitle('Comprehensive Performance Analysis', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "performance_ranking.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Performance ranking saved: {filepath}")
    plt.show()
    
    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(metrics).T
    metrics_path = os.path.join(output_dir, "detailed_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"✅ Detailed metrics saved: {metrics_path}")

if __name__ == "__main__":
    print("🎯 Generating Detailed Error Analysis")
    print("=" * 50)
    plot_error_analysis()
    print("🎉 Error analysis complete!")
    print("📁 All plots saved to: graphics/error_analysis/")