"""
Script for generating logarithmic scale specific analysis
Saves plots to: graphics/log_scale_plots/
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_log_scale_analysis(output_dir="graphics/log_scale_plots"):
    """
    Generate logarithmic scale specific analysis plots
    
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
    
    # Generate different log scale analyses
    create_log_scale_comparison(model_data, output_dir, colors)
    create_dynamic_range_analysis(model_data, output_dir, colors)
    create_log_error_patterns(model_data, output_dir, colors)
    create_order_of_magnitude_analysis(model_data, output_dir, colors)
    
    print("🎉 Log scale analysis complete!")
    print(f"📁 All plots saved to: {output_dir}/")

def create_log_scale_comparison(model_data, output_dir, colors):
    """Create side-by-side linear vs log scale comparison"""
    n_models = len(model_data)
    fig, axes = plt.subplots(n_models, 2, figsize=(16, 4*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    epsilon = 1e-10
    display_steps = 150
    
    for i, (model_name, df) in enumerate(model_data.items()):
        df_subset = df.head(display_steps)
        
        # Linear scale plot
        ax_linear = axes[i, 0]
        ax_linear.plot(df_subset['step'], df_subset['actual'], 'k-', 
                      linewidth=2, label='Actual', alpha=0.8)
        ax_linear.plot(df_subset['step'], df_subset['forecast'], 'r--', 
                      linewidth=2, label='Forecast', alpha=0.8, color=colors[i])
        ax_linear.set_xlabel('Time Step')
        ax_linear.set_ylabel('Value')
        ax_linear.set_title(f'{model_name} - Linear Scale')
        ax_linear.legend()
        ax_linear.grid(True, alpha=0.3)
        
        # Log scale plot
        ax_log = axes[i, 1]
        
        # Handle negative/zero values for log scale
        actual_log = np.where(df_subset['actual'] <= 0, epsilon, np.abs(df_subset['actual']))
        forecast_log = np.where(df_subset['forecast'] <= 0, epsilon, np.abs(df_subset['forecast']))
        
        ax_log.plot(df_subset['step'], actual_log, 'k-', 
                   linewidth=2, label='Actual', alpha=0.8)
        ax_log.plot(df_subset['step'], forecast_log, 'r--', 
                   linewidth=2, label='Forecast', alpha=0.8, color=colors[i])
        ax_log.set_xlabel('Time Step')
        ax_log.set_ylabel('Value (Log Scale)')
        ax_log.set_yscale('log')
        ax_log.set_title(f'{model_name} - Logarithmic Scale')
        ax_log.legend()
        ax_log.grid(True, alpha=0.3)
        
        # Add dynamic range info
        actual_range = df_subset['actual'].max() - df_subset['actual'].min()
        actual_ratio = np.abs(df_subset['actual']).max() / max(np.abs(df_subset['actual']).min(), epsilon)
        
        info_text = f"Range: {actual_range:.4f}\\nRatio: {actual_ratio:.1f}x"
        ax_log.text(0.02, 0.98, info_text, transform=ax_log.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('Linear vs Logarithmic Scale Comparison', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "linear_vs_log_comparison.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Linear vs log comparison saved: {filepath}")
    plt.show()

def create_dynamic_range_analysis(model_data, output_dir, colors):
    """Analyze dynamic range characteristics of each model"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Calculate dynamic range metrics
    range_data = {}
    for model_name, df in model_data.items():
        actual_values = np.abs(df['actual'])
        forecast_values = np.abs(df['forecast'])
        error_values = df['absolute_error']
        
        actual_ratio = actual_values.max() / max(actual_values.min(), 1e-10)
        forecast_ratio = forecast_values.max() / max(forecast_values.min(), 1e-10)
        error_ratio = error_values.max() / max(error_values.min(), 1e-10)
        
        range_data[model_name] = {
            'actual_ratio': actual_ratio,
            'forecast_ratio': forecast_ratio,
            'error_ratio': error_ratio,
            'actual_orders': np.log10(actual_ratio),
            'forecast_orders': np.log10(forecast_ratio),
            'error_orders': np.log10(error_ratio)
        }
    
    model_names = list(range_data.keys())
    
    # Plot 1: Dynamic range ratios
    ax1 = axes[0, 0]
    actual_ratios = [range_data[name]['actual_ratio'] for name in model_names]
    forecast_ratios = [range_data[name]['forecast_ratio'] for name in model_names]
    error_ratios = [range_data[name]['error_ratio'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.25
    
    ax1.bar(x - width, actual_ratios, width, label='Actual Values', alpha=0.7)
    ax1.bar(x, forecast_ratios, width, label='Forecast Values', alpha=0.7)
    ax1.bar(x + width, error_ratios, width, label='Error Values', alpha=0.7)
    
    ax1.set_ylabel('Dynamic Range Ratio (Max/Min)')
    ax1.set_title('Dynamic Range Analysis')
    ax1.set_yscale('log')
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Orders of magnitude
    ax2 = axes[0, 1]
    actual_orders = [range_data[name]['actual_orders'] for name in model_names]
    forecast_orders = [range_data[name]['forecast_orders'] for name in model_names]
    error_orders = [range_data[name]['error_orders'] for name in model_names]
    
    ax2.bar(x - width, actual_orders, width, label='Actual Values', alpha=0.7)
    ax2.bar(x, forecast_orders, width, label='Forecast Values', alpha=0.7)
    ax2.bar(x + width, error_orders, width, label='Error Values', alpha=0.7)
    
    ax2.set_ylabel('Orders of Magnitude')
    ax2.set_title('Orders of Magnitude Span')
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add recommendation line at 2 orders of magnitude
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, 
                label='Log Scale Recommended (>2 orders)')
    ax2.legend()
    
    # Plot 3: Log scale benefit assessment
    ax3 = axes[1, 0]
    
    # Create benefit score (higher = more beneficial)
    benefit_scores = []
    recommendations = []
    
    for name in model_names:
        error_orders = range_data[name]['error_orders']
        actual_orders = range_data[name]['actual_orders']
        
        # Benefit score based on orders of magnitude
        benefit_score = max(error_orders, actual_orders)
        benefit_scores.append(benefit_score)
        
        if benefit_score > 3:
            recommendations.append('Highly Recommended')
        elif benefit_score > 2:
            recommendations.append('Recommended')
        elif benefit_score > 1:
            recommendations.append('Somewhat Helpful')
        else:
            recommendations.append('Not Necessary')
    
    bars = ax3.bar(model_names, benefit_scores, color=colors[:len(model_names)], alpha=0.7)
    ax3.set_ylabel('Log Scale Benefit Score (Orders of Magnitude)')
    ax3.set_title('Logarithmic Scale Benefit Assessment')
    ax3.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Recommended Threshold')
    ax3.axhline(y=3, color='red', linestyle='--', alpha=0.7, label='Highly Recommended Threshold')
    ax3.legend()
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add recommendation labels
    for bar, rec in zip(bars, recommendations):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                rec, ha='center', va='bottom', fontsize=8, rotation=45)
    
    # Plot 4: Data characteristics summary table
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for name in model_names:
        data = range_data[name]
        table_data.append([
            name,
            f"{data['actual_orders']:.1f}",
            f"{data['forecast_orders']:.1f}",
            f"{data['error_orders']:.1f}",
            recommendations[model_names.index(name)]
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Model', 'Actual\\n(Orders)', 'Forecast\\n(Orders)', 
                               'Error\\n(Orders)', 'Log Scale\\nRecommendation'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    
    # Color code recommendations
    for i, rec in enumerate(recommendations):
        color = {'Highly Recommended': '#90EE90',  # Light green
                'Recommended': '#FFE4B5',          # Light orange
                'Somewhat Helpful': '#FFFFE0',     # Light yellow
                'Not Necessary': '#FFB6C1'}[rec]   # Light pink
        
        for j in range(5):
            table[(i+1, j)].set_facecolor(color)
    
    ax4.set_title('Dynamic Range Summary')
    
    plt.suptitle('Dynamic Range and Log Scale Benefit Analysis', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "dynamic_range_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Dynamic range analysis saved: {filepath}")
    plt.show()

def create_log_error_patterns(model_data, output_dir, colors):
    """Analyze error patterns specifically in log scale"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    epsilon = 1e-10
    display_steps = 200
    
    for i, (model_name, df) in enumerate(model_data.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        df_subset = df.head(display_steps)
        
        # Create log-scale error evolution plot with error bands
        errors_log = np.where(df_subset['absolute_error'] <= 0, epsilon, df_subset['absolute_error'])
        
        # Calculate rolling statistics in log space
        window = 20
        if len(df_subset) >= window:
            errors_smooth = df_subset['absolute_error'].rolling(window=window).mean()
            errors_std = df_subset['absolute_error'].rolling(window=window).std()
            
            # Plot error evolution
            ax.plot(df_subset['step'], errors_log, alpha=0.3, color=colors[i], linewidth=1)
            ax.plot(df_subset['step'][window-1:], errors_smooth[window-1:], 
                   color=colors[i], linewidth=3, label=f'{model_name} (smoothed)')
            
            # Add error bands
            upper_band = errors_smooth + errors_std
            lower_band = np.maximum(errors_smooth - errors_std, epsilon)
            
            ax.fill_between(df_subset['step'][window-1:], 
                          lower_band[window-1:], upper_band[window-1:],
                          alpha=0.2, color=colors[i])
        else:
            ax.plot(df_subset['step'], errors_log, color=colors[i], linewidth=2, label=model_name)
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Absolute Error (Log Scale)')
        ax.set_yscale('log')
        ax.set_title(f'{model_name} - Log Scale Error Pattern')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add error magnitude annotations
        mean_error = df_subset['absolute_error'].mean()
        median_error = df_subset['absolute_error'].median()
        
        ax.axhline(y=mean_error, color='red', linestyle='--', alpha=0.7)
        ax.axhline(y=median_error, color='blue', linestyle='--', alpha=0.7)
        
        # Add text annotations
        ax.text(0.02, 0.98, f'Mean: {mean_error:.2e}\\nMedian: {median_error:.2e}',
               transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(model_data), len(axes)):
        fig.delaxes(axes[i])
    
    plt.suptitle('Logarithmic Scale Error Pattern Analysis', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "log_error_patterns.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Log error patterns saved: {filepath}")
    plt.show()

def create_order_of_magnitude_analysis(model_data, output_dir, colors):
    """Analyze errors by order of magnitude"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define magnitude bins
    magnitude_bins = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    bin_labels = [f'10^{{{i}}}' for i in magnitude_bins[:-1]]
    
    # Plot 1: Error distribution by magnitude
    ax1 = axes[0, 0]
    
    for i, (model_name, df) in enumerate(model_data.items()):
        errors = df['absolute_error']
        log_errors = np.log10(np.maximum(errors, 1e-10))
        
        counts, _ = np.histogram(log_errors, bins=magnitude_bins)
        ax1.plot(magnitude_bins[:-1], counts, 'o-', 
                color=colors[i], label=model_name, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Log10(Error)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Error Distribution by Order of Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Cumulative error contribution by magnitude
    ax2 = axes[0, 1]
    
    for i, (model_name, df) in enumerate(model_data.items()):
        errors = df['absolute_error']
        log_errors = np.log10(np.maximum(errors, 1e-10))
        
        # Calculate cumulative contribution
        sorted_errors = np.sort(errors)[::-1]  # Descending order
        cumulative_sum = np.cumsum(sorted_errors)
        cumulative_percent = 100 * cumulative_sum / cumulative_sum[-1]
        
        # Plot first 100 largest errors
        plot_range = min(100, len(cumulative_percent))
        ax2.plot(range(1, plot_range + 1), cumulative_percent[:plot_range], 
                color=colors[i], label=model_name, linewidth=2)
    
    ax2.set_xlabel('Rank of Error (Largest to Smallest)')
    ax2.set_ylabel('Cumulative Error Contribution (%)')
    ax2.set_title('Cumulative Error Contribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=80, color='red', linestyle='--', alpha=0.7, label='80% Line')
    
    # Plot 3: Magnitude transition analysis
    ax3 = axes[1, 0]
    
    transition_data = {}
    for model_name, df in model_data.items():
        errors = df['absolute_error'].values
        log_errors = np.log10(np.maximum(errors, 1e-10))
        
        # Count transitions between magnitude levels
        transitions = []
        for j in range(1, len(log_errors)):
            mag_change = abs(log_errors[j] - log_errors[j-1])
            transitions.append(mag_change)
        
        transition_data[model_name] = np.mean(transitions)
    
    models = list(transition_data.keys())
    values = list(transition_data.values())
    
    bars = ax3.bar(models, values, color=colors[:len(models)], alpha=0.7)
    ax3.set_ylabel('Average Magnitude Change Between Steps')
    ax3.set_title('Error Magnitude Stability')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Magnitude range summary
    ax4 = axes[1, 1]
    ax4.axis('tight')
    ax4.axis('off')
    
    summary_data = []
    for model_name, df in model_data.items():
        errors = df['absolute_error']
        log_errors = np.log10(np.maximum(errors, 1e-10))
        
        min_mag = log_errors.min()
        max_mag = log_errors.max()
        range_mag = max_mag - min_mag
        
        summary_data.append([
            model_name,
            f"{min_mag:.1f}",
            f"{max_mag:.1f}",
            f"{range_mag:.1f}",
            f"{transition_data[model_name]:.2f}"
        ])
    
    table = ax4.table(cellText=summary_data,
                     colLabels=['Model', 'Min Log\\nError', 'Max Log\\nError', 
                               'Magnitude\\nRange', 'Avg Transition'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.3, 2)
    ax4.set_title('Order of Magnitude Summary')
    
    plt.suptitle('Order of Magnitude Error Analysis', fontsize=16)
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, "magnitude_analysis.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Magnitude analysis saved: {filepath}")
    plt.show()

if __name__ == "__main__":
    print("🎯 Generating Logarithmic Scale Analysis")
    print("=" * 50)
    plot_log_scale_analysis()
    print("🎉 Log scale analysis complete!")
    print("📁 All plots saved to: graphics/log_scale_plots/")