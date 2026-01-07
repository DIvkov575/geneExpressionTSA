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

def load_extrapolation_data(extrapolation_path="../../forecasts/column2_extrapolation_20260107_073320.csv"):
    """Load existing extrapolation data for column 2."""
    if not os.path.exists(extrapolation_path):
        print(f"❌ Extrapolation file not found: {extrapolation_path}")
        return None
    
    df = pd.read_csv(extrapolation_path)
    extrapolations = {}
    
    # Group by model
    for model_name in df['model'].unique():
        model_data = df[df['model'] == model_name]
        extrapolations[model_name] = {
            'steps': model_data['step'].values,
            'values': model_data['value'].values,
            'is_extrapolated': model_data['is_extrapolated'].values
        }
        print(f"✓ Loaded {model_name}: {len(model_data)} data points")
    
    return extrapolations

def plot_individual_extrapolations(original_data, extrapolations, column_name='2', save_dir="plots"):
    """Create individual extrapolation plots for each model."""
    os.makedirs(save_dir, exist_ok=True)
    
    column_data = original_data[column_name].values
    time_steps = np.arange(len(column_data))
    
    for model_name, model_data in extrapolations.items():
        plt.figure(figsize=(14, 8))
        
        # Plot original column data
        plt.plot(time_steps, column_data, 'k-', linewidth=2, label='Original Data', alpha=0.8)
        
        # Plot model's complete data (historical + extrapolated)
        steps = model_data['steps']
        values = model_data['values']
        is_extrapolated = model_data['is_extrapolated']
        
        # Split into historical and extrapolated
        historical_mask = ~is_extrapolated
        extrapolated_mask = is_extrapolated
        
        if np.any(historical_mask):
            plt.plot(steps[historical_mask], values[historical_mask], 'b-', 
                    linewidth=1.5, alpha=0.7, label=f'{model_name} Historical')
        
        if np.any(extrapolated_mask):
            plt.plot(steps[extrapolated_mask], values[extrapolated_mask], 'r--', 
                    linewidth=2, alpha=0.8, label=f'{model_name} Extrapolation')
            
            # Mark extrapolation start
            extrap_start = steps[extrapolated_mask][0]
            plt.axvline(x=extrap_start, color='red', linestyle=':', alpha=0.7, label='Extrapolation Start')

        plt.xlabel('Time Step')
        plt.ylabel(f'Column {column_name} Value')
        plt.title(f'{model_name} - Extrapolation from Column {column_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save individual plot
        safe_name = model_name.replace(' ', '_').lower()
        save_path = os.path.join(save_dir, f"individual_{safe_name}_extrapolation.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {save_path}")

def plot_combined_extrapolations(original_data, extrapolations, column_name='2', save_dir="plots"):
    """Create combined extrapolation plot showing all models together."""
    os.makedirs(save_dir, exist_ok=True)
    
    column_data = original_data[column_name].values
    time_steps = np.arange(len(column_data))
    
    plt.figure(figsize=(16, 10))
    
    # Plot original data
    plt.plot(time_steps, column_data, 'k-', linewidth=3, label='Original Data', alpha=0.9, zorder=10)
    
    # Plot all model extrapolations
    colors = plt.cm.tab10(np.linspace(0, 1, len(extrapolations)))
    
    for i, (model_name, model_data) in enumerate(extrapolations.items()):
        steps = model_data['steps']
        values = model_data['values']
        is_extrapolated = model_data['is_extrapolated']
        
        # Plot only extrapolated portions for clarity
        extrapolated_mask = is_extrapolated
        if np.any(extrapolated_mask):
            plt.plot(steps[extrapolated_mask], values[extrapolated_mask], 
                    '--', color=colors[i], linewidth=2, alpha=0.8, 
                    label=f'{model_name} Extrapolation')
    
    # Mark extrapolation start point
    if len(extrapolations) > 0:
        first_model_data = list(extrapolations.values())[0]
        extrap_mask = first_model_data['is_extrapolated']
        if np.any(extrap_mask):
            extrap_start = first_model_data['steps'][extrap_mask][0]
            plt.axvline(x=extrap_start, color='red', linestyle=':', alpha=0.7, label='Extrapolation Start')
    
    plt.xlabel('Time Step')
    plt.ylabel(f'Column {column_name} Value')
    plt.title(f'All Models - Extrapolation Comparison from Column {column_name}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f"combined_extrapolation_column_{column_name}.png")
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
    
    # Load existing extrapolations
    print("\\n🔮 Loading existing extrapolation data...")
    extrapolations = load_extrapolation_data()
    
    if not extrapolations:
        print("❌ No extrapolations loaded")
        return
    
    print(f"✅ Loaded extrapolations for {len(extrapolations)} models")
    
    # Generate individual plots
    print("\\n📊 Creating individual extrapolation plots...")
    plot_individual_extrapolations(original_data, extrapolations)
    
    # Generate combined plots
    print("\\n🔄 Creating combined extrapolation plot...")
    plot_combined_extrapolations(original_data, extrapolations)
    
    print("\\n🎉 Extrapolation plots generated successfully for Column 2!")

if __name__ == "__main__":
    main()