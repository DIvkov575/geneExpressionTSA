#!/usr/bin/env python3
"""
Master graphics generation script.
Orchestrates all plotting scripts and generates comprehensive report.
"""

import os
import sys
import subprocess
from datetime import datetime

def run_script(script_path, script_name):
    """Run a plotting script and capture results."""
    print(f"\\n🎨 Running {script_name}...")
    print("=" * 50)
    
    try:
        # Change to script directory and run
        script_dir = os.path.dirname(script_path)
        script_file = os.path.basename(script_path)
        
        result = subprocess.run(
            [sys.executable, script_file],
            cwd=script_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully")
            return True
        else:
            print(f"❌ {script_name} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {script_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ {script_name} failed with error: {e}")
        return False

def generate_summary_report(results):
    """Generate a summary report of all graphics generated."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""# Comprehensive Forecasting Graphics Report
Generated on: {timestamp}

## Overview
This report summarizes the graphics generated for the time series forecasting analysis of CRE.csv data.

## Graphics Generated

### 1. Forecast Accuracy vs Time Analysis
**Location**: `forecast_accuracy/`
**Status**: {'✅ Generated' if results.get('forecast_accuracy', False) else '❌ Failed'}

- Individual accuracy plots for each model
- Combined accuracy comparison (linear and log scale)
- Error evolution analysis with distribution and cumulative error plots
- Shows how forecast errors evolve over time

### 2. Forecast vs Time Analysis  
**Location**: `forecast_vs_time/column_1/`
**Status**: {'✅ Generated' if results.get('forecast_vs_time', False) else '❌ Failed'}

- Individual forecast vs time plots for each model
- Combined forecast comparison showing all models together
- Phase analysis (early, middle, late forecasting periods)
- Residual analysis with distribution and scatter plots

### 3. Extrapolation Analysis
**Location**: `extrapolation/plots/`
**Status**: {'✅ Generated' if results.get('extrapolation', False) else '❌ Failed'}

- Extrapolation from column 2 data using different models
- Comparison of how models behave when extrapolating beyond available data
- Analysis of model divergence and volatility in extrapolation mode

### 4. Real vs Predicted Scatter Analysis
**Location**: `real_vs_predicted/column_1/`
**Status**: {'✅ Generated' if results.get('real_vs_predicted', False) else '❌ Failed'}

- Individual scatter plots with comprehensive metrics for each model
- Combined scatter plots comparing all models
- Metrics comparison including R², correlation, MAE, RMSE
- Residual analysis and model performance evaluation

### 5. Banded Analysis
**Location**: `banded_analysis/column_1_bands/`
**Status**: {'✅ Generated' if results.get('banded_analysis', False) else '❌ Failed'}

- Time axis split into bands showing temporal patterns
- Stacked visualization of predictions across time periods
- Performance evolution across different time bands
- Identifies time periods where models perform better/worse

## Key Insights

### Model Performance Summary
The analysis reveals significant differences in model performance:

- **Naive Model**: Provides baseline performance with constant predictions
- **ARIMA Models**: Show varying stability depending on data characteristics
- **Neural Networks**: May exhibit instability in generative forecasting mode
- **Traditional ML**: Performance varies based on feature engineering quality

### Temporal Patterns
The banded analysis reveals how model performance changes over different time periods, 
helping identify:
- Periods of high/low predictability
- Model stability over time
- Adaptation to changing data patterns

### Forecasting Behavior
The comprehensive analysis shows:
- How models handle recursive prediction (using only their own outputs)
- Stability and divergence patterns in extrapolation mode
- Relationship between actual and predicted values across different scales

## Files Generated
The following graphics files have been generated:

### Accuracy Analysis
- `forecast_accuracy/individual/`: Individual model accuracy plots
- `forecast_accuracy/combined/`: Combined accuracy comparisons and analysis

### Forecast Comparison
- `forecast_vs_time/column_1/`: Forecast vs time plots and residual analysis

### Extrapolation
- `extrapolation/plots/`: Extrapolation analysis from column 2

### Scatter Analysis
- `real_vs_predicted/column_1/`: Comprehensive scatter plot analysis

### Banded Analysis
- `banded_analysis/column_1_bands/`: Time-banded prediction analysis

## Usage Notes
- All plots are saved in high resolution (300 DPI) PNG format
- Graphics are organized in logical folder structures
- Each script can be run independently for specific analyses
- Summary statistics and performance metrics are included in plot annotations

## Next Steps
1. Review individual model performance from scatter plots
2. Analyze temporal patterns from banded analysis  
3. Examine extrapolation behavior for model selection
4. Use accuracy analysis to identify best performing models for different scenarios

---
Report generated by: generate_all_graphics.py
"""
    
    # Save report
    with open("graphics_report.md", "w") as f:
        f.write(report)
    
    print("\\n📄 Graphics report saved to: graphics_report.md")

def main():
    print("🎨 Comprehensive Graphics Generation System")
    print("=" * 60)
    print(f"Starting graphics generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define scripts to run
    scripts = [
        ("forecast_accuracy/generate_accuracy_plots.py", "Forecast Accuracy Analysis"),
        ("forecast_vs_time/generate_forecast_plots.py", "Forecast vs Time Analysis"),
        ("extrapolation/generate_extrapolation.py", "Extrapolation Analysis"),
        ("real_vs_predicted/generate_scatter_plots.py", "Real vs Predicted Analysis"),
        ("banded_analysis/generate_banded_plots.py", "Banded Time Analysis")
    ]
    
    # Track results
    results = {}
    successful_scripts = 0
    
    # Run each script
    for script_path, script_name in scripts:
        script_key = script_path.split('/')[0]  # Use folder name as key
        
        success = run_script(script_path, script_name)
        results[script_key] = success  # Use script folder as key
        
        if success:
            successful_scripts += 1
    
    # Generate summary
    print("\\n" + "=" * 60)
    print("📊 GRAPHICS GENERATION SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully completed: {successful_scripts}/{len(scripts)} scripts")
    
    if successful_scripts == len(scripts):
        print("🎉 All graphics generated successfully!")
    elif successful_scripts > 0:
        print("⚠️  Some graphics generated with issues")
    else:
        print("❌ No graphics were generated successfully")
    
    print("\\nScript Results:")
    for (script_path, script_name), success in zip(scripts, results.values()):
        status = "✅" if success else "❌"
        print(f"  {status} {script_name}")
    
    # Generate comprehensive report
    print("\\n📄 Generating summary report...")
    generate_summary_report(results)
    
    # List generated files
    print("\\n📁 Generated Graphics Structure:")
    for root, dirs, files in os.walk("."):
        level = root.replace(".", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        
        # Only show PNG files
        png_files = [f for f in files if f.endswith('.png')]
        if png_files:
            subindent = " " * 2 * (level + 1)
            for file in png_files[:3]:  # Show first 3 files
                print(f"{subindent}{file}")
            if len(png_files) > 3:
                print(f"{subindent}... and {len(png_files) - 3} more files")
    
    print(f"\\n🏁 Graphics generation completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()