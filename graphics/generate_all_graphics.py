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
This report summarizes the graphics generated for the time series forecasting analysis of CRE.csv data, focusing specifically on Column 2.

## Graphics Generated

### 1. Forecast Accuracy vs Time Analysis (Column 2)
**Location**: `forecast_accuracy/`
**Status**: {'✅ Generated' if results.get('forecast_accuracy', False) else '❌ Failed'}

- Individual MAPE vs time plots for each model (Column 2)
- Combined accuracy comparison (linear and log scale)
- Error evolution analysis with distribution and cumulative MAPE plots
- Shows how forecast MAPE evolves over time for Column 2

### 2. Forecast vs Time Analysis (Column 2)
**Location**: `forecast_vs_time/column_2/`
**Status**: {'✅ Generated' if results.get('forecast_vs_time', False) else '❌ Failed'}

- Individual forecast vs time plots for each model (Column 2)
- Combined forecast comparison showing all models together
- Combined forecast comparison with logarithmic scaling
- Phase analysis (early, middle, late forecasting periods)
- Residual analysis with distribution and scatter plots

### 3. Extrapolation Analysis (Column 2)
**Location**: `extrapolation/plots/`
**Status**: {'✅ Generated' if results.get('extrapolation', False) else '❌ Failed'}

- Individual extrapolation plots extending from Column 2 data
- Combined plot showing all model extrapolations superimposed
- Analysis of how models behave when extrapolating beyond available data
- Shows model divergence patterns from original Column 2 data

### 4. Real vs Predicted Scatter Analysis (Column 2)
**Location**: `real_vs_predicted/column_2/`
**Status**: {'✅ Generated' if results.get('real_vs_predicted', False) else '❌ Failed'}

- Individual scatter plots with comprehensive metrics for each model (Column 2)
- Combined scatter plots comparing all models
- Metrics comparison including R², correlation, MAE, RMSE
- Residual analysis and model performance evaluation for Column 2

## Key Insights

### Model Performance Summary (Column 2)
The analysis reveals significant differences in model performance for Column 2:

- **GBM Model**: Shows performance characteristics specific to Column 2 data
- **NBEATS Model**: Neural network behavior on Column 2 forecasting tasks  
- **Naive Model**: Baseline extrapolation performance for Column 2

### Forecasting Behavior (Column 2)
The comprehensive analysis shows:
- How models handle recursive prediction for Column 2 (using only their own outputs)
- Stability and divergence patterns in extrapolation mode from Column 2 data
- Relationship between actual and predicted Column 2 values across different scales
- MAPE evolution over time for Column 2 forecasts

## Files Generated
The following graphics files have been generated:

### Accuracy Analysis (Column 2)
- `forecast_accuracy/individual/`: Individual model MAPE vs time plots
- `forecast_accuracy/combined/`: Combined accuracy comparisons and analysis

### Forecast Comparison (Column 2)
- `forecast_vs_time/column_2/`: Forecast vs time plots and residual analysis

### Extrapolation (Column 2)
- `extrapolation/plots/`: Extrapolation analysis extending from Column 2

### Scatter Analysis (Column 2)
- `real_vs_predicted/column_2/`: Comprehensive scatter plot analysis

## Usage Notes
- All plots are saved in high resolution (300 DPI) PNG format
- Graphics are organized in logical folder structures
- Each script can be run independently for specific analyses
- Summary statistics and performance metrics are included in plot annotations

## Next Steps
1. Review individual model performance from scatter plots (Column 2)
2. Examine extrapolation behavior for model selection from Column 2 data
3. Use MAPE accuracy analysis to identify best performing models for Column 2
4. Compare forecast vs time patterns to understand model behavior on Column 2

---
Report generated by: generate_all_graphics.py
"""
    
    # Save report
    with open("graphics_report.md", "w") as f:
        f.write(report)
    
    print("\\n📄 Graphics report saved to: graphics_report.md")

def main():
    print("🎨 Comprehensive Graphics Generation System (Column 2)")
    print("=" * 60)
    print(f"Starting Column 2 graphics generation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define scripts to run
    scripts = [
        ("forecast_accuracy/generate_accuracy_plots.py", "Forecast Accuracy Analysis (Column 2)"),
        ("forecast_vs_time/generate_forecast_plots.py", "Forecast vs Time Analysis (Column 2)"),
        ("extrapolation/generate_extrapolation.py", "Extrapolation Analysis (Column 2)"),
        ("real_vs_predicted/generate_scatter_plots.py", "Real vs Predicted Analysis (Column 2)")
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