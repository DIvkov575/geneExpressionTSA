# Neural ODE for Time Series Forecasting

Neural ODE models for forecasting CRE.csv time series data.

## Project Structure

```
protien-tsa/
├── data/
│   └── CRE.csv                    # Time series data (40 independent samples)
│
├── models/
│   ├── neural_ode.py              # Regular Neural ODE (1D + time)
│   └── augmented_ode.py           # Augmented Neural ODE (multi-dimensional)
│
├── results/
│   ├── neural_ode/                # Regular ODE results (models + plots)
│   ├── augmented_ode_dim2/        # Augmented ODE (2D) results
│   ├── augmented_ode_dim3/        # Augmented ODE (3D) results
│   ├── augmented_ode_dim6/        # Augmented ODE (6D) results
│   └── comparisons/               # Side-by-side comparison plots & metrics
│
├── train_all.py                   # Train all model variants
├── compare.py                     # Compare all trained models
└── README.md                      # This file
```

## Quick Start

### 1. Train All Models

Train all Neural ODE variants:

```bash
# Train ALL models (basic + advanced)
python train_all.py

# Train only basic models (fast)
python train_all.py --models basic

# Train only advanced models (for oscillations)
python train_all.py --models advanced

# Train specific models
python train_all.py --models neural_ode        # Only regular ODE
python train_all.py --models augmented         # Only augmented variants
python train_all.py --models second_order      # Only second-order ODE ⭐
python train_all.py --models latent            # Only latent ODE
```

Options:
```bash
python train_all.py --epochs 500 --lr 0.001 --device cpu
```

### 2. Compare Models

Compare all trained models and generate comparison plots:

```bash
python compare.py
```

This will:
- Load all trained models from `results/`
- Evaluate forecast accuracy on validation data
- Generate comparison plots showing:
  - Training loss vs model complexity
  - Test accuracy vs model complexity
  - Bias-variance tradeoff
  - Relative performance comparison
- Save results to `results/comparisons/`

### 3. Train Individual Models

Run specific model implementations directly:

```bash
python models/neural_ode.py          # Train regular Neural ODE
python models/augmented_ode.py       # Train augmented Neural ODE (dim=6 default)
```

## Model Descriptions

### Basic Models

#### Regular Neural ODE
- **State space**: 1D (observed data) + time
- **ODE**: `dx/dt = f(x, t)`
- **Parameters**: ~50K
- **Best for**: Forecasting accuracy, generalization
- **Captures**: Smooth trends

#### Augmented Neural ODE
- **State space**: 1D observed + N auxiliary dimensions
- **ODE**: `dx/dt = f([x, a₁, ..., aₙ], t)` (only x is observed)
- **Parameters**: ~50K (similar to regular)
- **Best for**: Modeling complex dynamics, trajectory generation
- **Captures**: Richer dynamics than regular ODE

**Variants**:
- `dim=2`: 1D observed + 1 auxiliary
- `dim=3`: 1D observed + 2 auxiliary (best augmented variant)
- `dim=6`: 1D observed + 5 auxiliary (overfits)

### Advanced Models (for High-Frequency Dynamics)

#### Second-Order Neural ODE ⭐
- **State space**: [position, velocity]
- **ODE**: `dx/dt = v`, `dv/dt = f(x, v, t)`
- **Parameters**: ~50K
- **Best for**: Oscillatory dynamics, momentum effects, acceleration
- **Captures**: High-frequency oscillations naturally
- **Key advantage**: Models acceleration directly (d²x/dt²)

#### Latent Neural ODE
- **Architecture**: Encoder → Latent ODE → Decoder
- **State space**: 1D observed → 20D latent → 1D predicted
- **ODE**: `dz/dt = f(z, t)` in latent space
- **Parameters**: ~70K (includes encoder/decoder)
- **Best for**: Complex patterns, hidden structure
- **Captures**: Complex 1D dynamics as simple high-D dynamics
- **Key advantage**: Learns representations automatically

## Results Summary

Based on CRE.csv forecasting:

### Basic Models

| Model | State Dim | Training Loss | Test MAE | Recommendation |
|-------|-----------|---------------|----------|----------------|
| **Regular Neural ODE** | 1D | 0.842 | **0.219** ± 0.245 | ✅ Best basic model |
| Augmented (dim=2) | 2D | 0.833 | 0.239 ± 0.239 | - |
| Augmented (dim=3) | 3D | 0.797 | 0.227 ± 0.244 | ⭐ Best augmented |
| Augmented (dim=6) | 6D | 0.737 | 0.250 ± 0.223 | ❌ Overfits |

**Key Finding**: More dimensions → better training fit but worse forecasting due to overfitting.

### Advanced Models (for High-Frequency Dynamics)

| Model | State Dim | Training Loss | Test MAE | Captures Oscillations? |
|-------|-----------|---------------|----------|------------------------|
| **Second-Order ODE** ⭐ | 2D (x, v) | 0.957 | **~0.21** | ✅ Yes (models acceleration) |
| Latent ODE | 20D latent | TBD | TBD | ✅ Yes (learned repr.) |

**Recommended for oscillatory data**: Second-Order Neural ODE
- Directly models velocity and acceleration
- Natural for mechanical systems, waves, oscillations
- Better captures high-frequency dynamics than first-order ODEs

## Key Concepts

### Neural ODE
Models continuous-time dynamics using neural networks:
```
dx/dt = f_θ(x, t)
```
where `f_θ` is a neural network. Solved using ODE solvers (dopri5).

### Augmented Neural ODE
Adds "dummy" dimensions to the state space:
```
State: [x_observed, a₁, a₂, ..., aₙ]
ODE operates on full state but only x_observed is measured
```

**Why augment?**
- More expressiveness for complex dynamics
- Can model trajectories that 1D ODEs cannot (e.g., loops, oscillations)
- BUT: Risk of overfitting if too many dimensions

## Dependencies

```bash
pip install torch numpy pandas matplotlib torchdiffeq
```

## Citation

Neural ODE implementation based on:
- Chen et al. "Neural Ordinary Differential Equations" (NeurIPS 2018)
- Dupont et al. "Augmented Neural ODEs" (NeurIPS 2019)
