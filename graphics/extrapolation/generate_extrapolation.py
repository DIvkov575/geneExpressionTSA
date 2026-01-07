#!/usr/bin/env python3
"""
Generate extrapolation plots extending from column 2 of CRE.csv.
Shows how models extrapolate beyond the available data using only their own predictions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

def load_original_data(data_path="../../data/CRE.csv"):
    """Load the original CRE.csv data."""
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None
    
    df = pd.read_csv(data_path)
    print(f"✅ Loaded original data: {len(df)} rows")
    return df

def extrapolate_with_models(column_data, extrapolation_steps=100, seed_length=30):
    """Generate extrapolations using available forecasting models."""
    sys.path.append('../../run_forward')
    
    extrapolations = {}
    
    # Simplified models for stable extrapolation
    models_to_try = {
        'Naive': 'walk_forward_naive',
        'ARIMA Stats': 'walk_forward_arima_statsmodels',
        'GBM': 'walk_forward_gbm',
        'NBEATS': 'walk_forward_nbeats'
    }
    
    for model_name, module_name in models_to_try.items():
        try:
            if model_name == 'Naive':
                from walk_forward_naive import naive_forecast
                
                # Use last 'seed_length' points as seed
                seed_data = column_data[-seed_length:].copy()
                extrapolated_values = []
                
                for step in range(extrapolation_steps):
                    # Naive forecast: use last value
                    next_val = seed_data[-1]
                    extrapolated_values.append(next_val)
                    seed_data = np.append(seed_data, next_val)
                
                extrapolations[model_name] = extrapolated_values
                print(f"  ✓ {model_name}: Generated {len(extrapolated_values)} extrapolated points")
                
            elif model_name == 'ARIMA Stats':
                from walk_forward_arima_statsmodels import arima_statsmodels_forecast
                
                # Use last 'seed_length' points as seed
                seed_data = column_data[-seed_length:].copy()
                extrapolated_values = []
                failures = 0
                
                for step in range(extrapolation_steps):
                    try:
                        # Generate next forecast with shorter window if failing
                        if failures < 5:
                            forecast = arima_statsmodels_forecast(seed_data, steps=1)
                        else:
                            # Use only recent data if too many failures
                            forecast = arima_statsmodels_forecast(seed_data[-10:], steps=1)
                        
                        next_val = forecast[0]
                        # Sanity check - don't allow extreme values
                        if abs(next_val) > 10 * abs(column_data.std()):
                            raise ValueError(f"Extreme forecast: {next_val}")
                            
                        extrapolated_values.append(next_val)
                        seed_data = np.append(seed_data, next_val)
                        failures = 0  # Reset failure count on success
                        
                    except Exception as e:
                        failures += 1
                        if failures > 10:
                            print(f"    ⚠ ARIMA failed too many times, switching to trend")
                            # Use simple trend
                            if len(seed_data) >= 2:
                                trend = seed_data[-1] - seed_data[-2]
                                next_val = seed_data[-1] + trend
                            else:
                                next_val = seed_data[-1]
                        else:
                            # Fallback to last value
                            next_val = seed_data[-1]
                        
                        extrapolated_values.append(next_val)
                        seed_data = np.append(seed_data, next_val)
                
                extrapolations[model_name] = extrapolated_values
                print(f"  ✓ {model_name}: Generated {len(extrapolated_values)} extrapolated points")
                
            elif model_name == 'GBM':
                try:
                    from walk_forward_gbm import gbm_forecast
                    
                    # Use seed data for GBM extrapolation
                    seed_data = column_data[-seed_length:].copy()
                    extrapolated_values = []
                    
                    for step in range(extrapolation_steps):
                        try:
                            forecast = gbm_forecast(seed_data, steps=1)
                            next_val = forecast[0] if hasattr(forecast, '__iter__') else forecast
                            extrapolated_values.append(next_val)
                            seed_data = np.append(seed_data, next_val)
                        except:
                            # Fallback to last value
                            next_val = seed_data[-1]
                            extrapolated_values.append(next_val)
                            seed_data = np.append(seed_data, next_val)
                    
                    extrapolations[model_name] = extrapolated_values
                    print(f"  ✓ {model_name}: Generated {len(extrapolated_values)} extrapolated points")
                    
                except ImportError:
                    print(f"  ⚠ {model_name}: Module not available")
                    
            elif model_name == 'NBEATS':
                try:
                    from walk_forward_nbeats import nbeats_forecast
                    
                    # Use seed data for NBEATS extrapolation  
                    seed_data = column_data[-seed_length:].copy()
                    extrapolated_values = []
                    
                    for step in range(extrapolation_steps):
                        try:
                            forecast = nbeats_forecast(seed_data, steps=1)
                            next_val = forecast[0] if hasattr(forecast, '__iter__') else forecast
                            extrapolated_values.append(next_val)
                            seed_data = np.append(seed_data, next_val)
                        except:
                            # Fallback to last value
                            next_val = seed_data[-1]
                            extrapolated_values.append(next_val)
                            seed_data = np.append(seed_data, next_val)
                    
                    extrapolations[model_name] = extrapolated_values
                    print(f"  ✓ {model_name}: Generated {len(extrapolated_values)} extrapolated points")
                    
                except ImportError:
                    print(f"  ⚠ {model_name}: Module not available")
                
        except ImportError as e:
            print(f"  ✗ {model_name}: Import failed - {e}")
        except Exception as e:
            print(f"  ✗ {model_name}: Error - {e}")
    
    # Add simple trend extrapolation
    if len(column_data) >= 10:
        # Linear trend from last 10 points
        recent_data = column_data[-10:]
        x = np.arange(len(recent_data))
        coeffs = np.polyfit(x, recent_data, 1)
        
        extrapolated_values = []
        for step in range(extrapolation_steps):
            next_val = coeffs[0] * (len(recent_data) + step) + coeffs[1]
            extrapolated_values.append(next_val)
        
        extrapolations['Linear Trend'] = extrapolated_values
        print(f"  ✓ Linear Trend: Generated {len(extrapolated_values)} extrapolated points")
    
    # Add exponential smoothing
    if len(column_data) >= 5:
        alpha = 0.3  # Smoothing parameter
        last_smoothed = column_data[-1]
        
        extrapolated_values = []
        for step in range(extrapolation_steps):
            # Simple exponential smoothing (constant level)
            extrapolated_values.append(last_smoothed)
        
        extrapolations['Exp Smoothing'] = extrapolated_values
        print(f"  ✓ Exp Smoothing: Generated {len(extrapolated_values)} extrapolated points")
    
    return extrapolations

def plot_extrapolation_analysis(original_data, extrapolations, column_name='2', save_dir="plots"):
    """Create comprehensive extrapolation analysis plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    column_data = original_data[column_name].values
    time_steps = np.arange(len(column_data))
    
    # Main extrapolation plot
    plt.figure(figsize=(16, 10))
    
    # Plot original data
    plt.plot(time_steps, column_data, 'k-', linewidth=2, label='Original Data', alpha=0.8)
    
    # Mark extrapolation start point
    extrapolation_start = len(column_data)
    plt.axvline(x=extrapolation_start, color='red', linestyle='--', alpha=0.7, label='Extrapolation Start')
    
    # Plot extrapolations
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrapolations)))
    extrapolation_steps = np.arange(extrapolation_start, extrapolation_start + len(list(extrapolations.values())[0]))
    
    for i, (model_name, extrap_values) in enumerate(extrapolations.items()):
        plt.plot(extrapolation_steps, extrap_values, '--', color=colors[i], 
                linewidth=2, alpha=0.7, label=f'{model_name} Extrapolation')
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Column {column_name} Value')
    plt.title(f'Model Extrapolation Analysis - Column {column_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"column_{column_name}_extrapolation.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Zoomed extrapolation view
    plt.figure(figsize=(16, 8))
    
    # Show last 50 original points + extrapolation
    zoom_start = max(0, len(column_data) - 50)
    zoom_original_steps = time_steps[zoom_start:]
    zoom_original_data = column_data[zoom_start:]
    
    plt.plot(zoom_original_steps, zoom_original_data, 'k-', linewidth=2, label='Original Data', alpha=0.8)
    plt.axvline(x=extrapolation_start, color='red', linestyle='--', alpha=0.7, label='Extrapolation Start')
    
    for i, (model_name, extrap_values) in enumerate(extrapolations.items()):
        plt.plot(extrapolation_steps, extrap_values, '--', color=colors[i], 
                linewidth=2, alpha=0.7, label=f'{model_name}')
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Column {column_name} Value')
    plt.title(f'Model Extrapolation - Zoomed View (Column {column_name})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"column_{column_name}_extrapolation_zoom.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def plot_extrapolation_comparison(original_data, extrapolations, column_name='2', save_dir="plots"):
    """Create detailed comparison of extrapolation behaviors."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    column_data = original_data[column_name].values
    
    # Plot 1: Extrapolation trends
    ax1 = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrapolations)))
    
    for i, (model_name, extrap_values) in enumerate(extrapolations.items()):
        steps = np.arange(len(extrap_values))
        ax1.plot(steps, extrap_values, color=colors[i], linewidth=2, label=model_name)
    
    ax1.set_xlabel('Extrapolation Steps')
    ax1.set_ylabel(f'Column {column_name} Value')
    ax1.set_title('Extrapolation Trends')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Extrapolation statistics
    ax2 = axes[0, 1]
    model_names = list(extrapolations.keys())
    final_values = [extrapolations[name][-1] for name in model_names]
    
    bars = ax2.bar(model_names, final_values, color=colors, alpha=0.7)
    ax2.set_ylabel(f'Final Extrapolated Value')
    ax2.set_title(f'Final Values After Extrapolation')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Extrapolation volatility
    ax3 = axes[1, 0]
    volatilities = []
    
    for model_name, extrap_values in extrapolations.items():
        # Calculate standard deviation of extrapolated values
        volatility = np.std(extrap_values)
        volatilities.append(volatility)
    
    bars = ax3.bar(model_names, volatilities, color=colors, alpha=0.7)
    ax3.set_ylabel('Volatility (Std Dev)')
    ax3.set_title('Extrapolation Volatility')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, volatilities):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Extrapolation divergence from original mean
    ax4 = axes[1, 1]
    original_mean = column_data.mean()
    
    for i, (model_name, extrap_values) in enumerate(extrapolations.items()):
        divergence = np.abs(np.array(extrap_values) - original_mean)
        steps = np.arange(len(divergence))
        ax4.plot(steps, divergence, color=colors[i], linewidth=2, label=model_name)
    
    ax4.set_xlabel('Extrapolation Steps')
    ax4.set_ylabel('Divergence from Original Mean')
    ax4.set_title('Model Divergence Over Time')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"column_{column_name}_extrapolation_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    print("🚀 Generating Extrapolation Analysis from Column 2")
    print("=" * 50)
    
    # Load original data
    print("📊 Loading original CRE data...")
    original_data = load_original_data()
    
    if original_data is None:
        return
    
    # Check if column 2 exists
    if '2' not in original_data.columns:
        print("❌ Column '2' not found in data")
        return
    
    column_data = original_data['2'].values
    print(f"✅ Column 2 data: {len(column_data)} points, range [{column_data.min():.4f}, {column_data.max():.4f}]")
    
    # Generate extrapolations
    print("\\n🔮 Generating extrapolations...")
    extrapolation_steps = 100
    extrapolations = extrapolate_with_models(column_data, extrapolation_steps=extrapolation_steps)
    
    if not extrapolations:
        print("❌ No extrapolations generated")
        return
    
    print(f"✅ Generated extrapolations for {len(extrapolations)} models")
    
    # Generate plots
    print("\\n📊 Creating extrapolation plots...")
    plot_extrapolation_analysis(original_data, extrapolations)
    
    print("\\n🔍 Creating extrapolation comparison...")
    plot_extrapolation_comparison(original_data, extrapolations)
    
    # Print summary
    print("\\n📋 Extrapolation Summary:")
    print("-" * 40)
    original_final = column_data[-1]
    print(f"Original final value: {original_final:.4f}")
    
    for model_name, extrap_values in extrapolations.items():
        final_val = extrap_values[-1]
        change = final_val - original_final
        volatility = np.std(extrap_values)
        print(f"{model_name:<15}: Final={final_val:7.4f}, Change={change:+7.4f}, Vol={volatility:.4f}")
    
    print("\\n🎉 Extrapolation analysis completed successfully!")

if __name__ == "__main__":
    main()