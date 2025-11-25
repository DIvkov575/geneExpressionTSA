# Synthetic Data Generation

This directory contains scripts and data for generating synthetic time series using pre-determined ARIMA models.

## Generated Files

- **`generate_arima_synthetic.py`**: Main script for generating synthetic ARIMA time series data
- **`synthetic_arima_data.csv`**: Generated synthetic dataset (1000 samples × 5 series)
- **`synthetic_data_plot.png`**: Visualization of the generated time series

## Model Configuration

The synthetic data is generated using an **ARIMA(1,1,1)** model with the following parameters:

- **AR(1) coefficient**: 0.7
- **MA(1) coefficient**: 0.3
- **Differencing order**: 1
- **Random seed**: 42 (for reproducibility)

## Dataset Details

- **Number of time series**: 5 columns (labeled '1' through '5')
- **Samples per series**: 1000 data points
- **Format**: CSV file compatible with the existing data format in `data/CRE.csv`

Each series has slightly different mean and standard deviation to add diversity while maintaining the same underlying ARIMA structure.

## Usage

### Generate New Synthetic Data

```bash
python synthetic_data/generate_arima_synthetic.py
```

### Customize Parameters

Edit the configuration section in `generate_arima_synthetic.py`:

```python
# Pre-determined ARIMA parameters
AR_PARAMS = [0.7]   # AR(1) coefficient
MA_PARAMS = [0.3]   # MA(1) coefficient
D = 1               # First-order differencing

# Dataset configuration
N_SERIES = 5        # Number of time series to generate
N_SAMPLES = 1000    # Number of data points per series
SEED = 42           # Random seed for reproducibility
```

### Use with Existing Scripts

The generated `synthetic_arima_data.csv` can be used with your existing training scripts:

```python
# Example: Use synthetic data with train_validate_standard_arima.py
FILE_PATH = 'synthetic_data/synthetic_arima_data.csv'
COLUMN = '1'  # Or any column from '1' to '5'
```

## Functions

### `generate_arima_data()`
Generates a single synthetic time series using specified ARIMA parameters.

**Parameters:**
- `n_samples`: Number of data points to generate
- `ar_params`: AR coefficients (list)
- `ma_params`: MA coefficients (list)
- `d`: Degree of differencing
- `mean`: Mean of the series
- `std`: Standard deviation of noise
- `seed`: Random seed

### `create_synthetic_dataset()`
Creates a multi-column dataset with multiple synthetic time series.

**Parameters:**
- `n_series`: Number of time series columns
- `n_samples`: Number of data points per series
- `output_file`: Output CSV filename
- `ar_params`, `ma_params`, `d`: ARIMA model parameters
- `seed`: Random seed

### `visualize_synthetic_data()`
Creates visualization plots of the generated time series.

**Parameters:**
- `df`: DataFrame with synthetic data
- `n_plots`: Number of series to plot
- `output_file`: Output filename for the plot

## Applications

This synthetic data can be used for:

1. **Model Testing**: Validate your ARIMA models on known data
2. **Algorithm Development**: Test new forecasting algorithms
3. **Data Augmentation**: Supplement real data for training
4. **Benchmarking**: Compare model performance on controlled data
5. **Educational Purposes**: Demonstrate time series concepts

## Notes

- The ARIMA parameters are chosen to match the model configuration in `train_validate_standard_arima.py`
- Each series has slightly different statistical properties (mean, std) to simulate real-world variability
- The data exhibits typical ARIMA characteristics: autocorrelation, trend, and stochastic behavior
