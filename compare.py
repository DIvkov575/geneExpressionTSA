"""
Compare all trained Neural ODE models
Usage: python compare.py
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
import sys

# Add models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from neural_ode import ODEFunc, NeuralODE, forecast as neural_forecast, load_data, prepare_training_data
from augmented_ode import AugmentedODEFunc, AugmentedNeuralODE, forecast as aug_forecast
from second_order_ode import SecondOrderODEFunc, SecondOrderNeuralODE, forecast as second_order_forecast, estimate_initial_velocity
from latent_ode import Encoder, Decoder, LatentODEFunc, LatentNeuralODE, forecast as latent_forecast


def load_latest_model(model_dir, model_type='neural_ode'):
    """Load the most recent model from a directory"""
    pattern = os.path.join(model_dir, '*.pt')
    model_files = glob.glob(pattern)

    if not model_files:
        print(f"⚠️  No models found in {model_dir}")
        return None

    # Get most recent
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Loading: {os.path.basename(latest_model)}")

    checkpoint = torch.load(latest_model, map_location='cpu', weights_only=False)

    if model_type == 'neural_ode':
        ode_func = ODEFunc(hidden_dim=128, num_layers=4)
        model = NeuralODE(ode_func)
        model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'model': model,
            'time_min': checkpoint['time_min'],
            'time_max': checkpoint['time_max'],
            'data_mean': checkpoint['data_mean'],
            'data_std': checkpoint['data_std'],
            'losses': checkpoint['losses'],
            'aug_dim': 1
        }
    elif model_type == 'augmented':
        aug_dim = checkpoint['aug_dim']
        ode_func = AugmentedODEFunc(aug_dim=aug_dim, hidden_dim=128, num_layers=4)
        model = AugmentedNeuralODE(ode_func, aug_dim=aug_dim)
        model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'model': model,
            'time_min': checkpoint['time_min'],
            'time_max': checkpoint['time_max'],
            'data_mean': checkpoint['data_mean'],
            'data_std': checkpoint['data_std'],
            'losses': checkpoint['losses'],
            'aug_dim': aug_dim
        }
    elif model_type == 'second_order':
        ode_func = SecondOrderODEFunc(hidden_dim=128, num_layers=4)
        model = SecondOrderNeuralODE(ode_func)
        model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'model': model,
            'time_min': checkpoint['time_min'],
            'time_max': checkpoint['time_max'],
            'data_mean': checkpoint['data_mean'],
            'data_std': checkpoint['data_std'],
            'losses': checkpoint['losses'],
            'aug_dim': 2  # [position, velocity]
        }
    elif model_type == 'latent':
        latent_dim = checkpoint['latent_dim']
        encoder = Encoder(latent_dim=latent_dim, hidden_dim=64)
        decoder = Decoder(latent_dim=latent_dim, hidden_dim=64)
        ode_func = LatentODEFunc(latent_dim=latent_dim, hidden_dim=128, num_layers=3)

        model = LatentNeuralODE(encoder, decoder, ode_func, latent_dim)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model.ode_func.load_state_dict(checkpoint['ode_func_state_dict'])

        return {
            'model': model,
            'time_min': checkpoint['time_min'],
            'time_max': checkpoint['time_max'],
            'data_mean': checkpoint['data_mean'],
            'data_std': checkpoint['data_std'],
            'losses': checkpoint['losses'],
            'aug_dim': latent_dim
        }


def evaluate_model(model_info, train_data, train_time, val_data, val_time, model_type='neural_ode'):
    """Evaluate a model and return metrics"""
    model = model_info['model']
    model.eval()

    mse_scores = []
    mae_scores = []

    # Normalize data for velocity estimation (for second-order)
    if model_type == 'second_order':
        data_mean = model_info['data_mean']
        data_std = model_info['data_std']
        train_data_norm = (train_data - data_mean) / data_std
        time_min = model_info['time_min']
        time_max = model_info['time_max']
        train_time_norm = (train_time - time_min) / (time_max - time_min)

    for i in range(train_data.shape[1]):
        y0 = train_data[-1:, i:i+1]

        if model_type == 'neural_ode':
            pred = neural_forecast(model, y0, val_time,
                                  model_info['time_min'], model_info['time_max'],
                                  model_info['data_mean'], model_info['data_std'])
        elif model_type == 'augmented':
            pred, _ = aug_forecast(model, y0, val_time,
                                  model_info['time_min'], model_info['time_max'],
                                  model_info['data_mean'], model_info['data_std'])
        elif model_type == 'second_order':
            # Estimate velocity from last two points
            dt = train_time_norm[-1] - train_time_norm[-2]
            v0 = (train_data_norm[-1, i] - train_data_norm[-2, i]) / dt
            v0 = np.array([[v0]])

            pred, _ = second_order_forecast(model, y0, v0, val_time,
                                          model_info['time_min'], model_info['time_max'],
                                          model_info['data_mean'], model_info['data_std'])
        elif model_type == 'latent':
            pred, _ = latent_forecast(model, y0, val_time,
                                    model_info['time_min'], model_info['time_max'],
                                    model_info['data_mean'], model_info['data_std'])

        mse = np.mean((pred - val_data[:, i:i+1]) ** 2)
        mae = np.mean(np.abs(pred - val_data[:, i:i+1]))

        mse_scores.append(mse)
        mae_scores.append(mae)

    return {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'train_loss': min(model_info['losses']),
        'aug_dim': model_info['aug_dim']
    }


def main():
    print("="*80)
    print("COMPARING ALL NEURAL ODE MODELS")
    print("="*80)

    # Load data
    time_axis, data, _ = load_data('data/CRE.csv')
    train_time, train_data, val_time, val_data = prepare_training_data(time_axis, data)

    # Load all models
    models = {}

    print("\n📦 Loading models...")

    # Regular Neural ODE
    print("\n1. Regular Neural ODE:")
    reg_model = load_latest_model('results/neural_ode', 'neural_ode')
    if reg_model:
        models['Regular Neural ODE'] = (reg_model, 'neural_ode')

    # Augmented models
    for dim in [2, 3, 6]:
        print(f"\n2. Augmented Neural ODE (dim={dim}):")
        aug_model = load_latest_model(f'results/augmented_ode_dim{dim}', 'augmented')
        if aug_model:
            models[f'Augmented ODE (dim={dim})'] = (aug_model, 'augmented')

    # Second-Order Neural ODE
    print(f"\n3. Second-Order Neural ODE:")
    second_order_model = load_latest_model('results/second_order_ode', 'second_order')
    if second_order_model:
        models['Second-Order Neural ODE'] = (second_order_model, 'second_order')

    # Latent Neural ODE
    print(f"\n4. Latent Neural ODE:")
    latent_model = load_latest_model('results/latent_ode', 'latent')
    if latent_model:
        models['Latent Neural ODE'] = (latent_model, 'latent')

    if not models:
        print("\n❌ No models found! Run 'python train_all.py' first.")
        return

    # Evaluate all models
    print("\n" + "="*80)
    print("📊 EVALUATING MODELS...")
    print("="*80)

    results = []
    for name, (model_info, model_type) in models.items():
        print(f"\nEvaluating {name}...")
        metrics = evaluate_model(model_info, train_data, train_time, val_data, val_time, model_type)
        results.append({
            'Model': name,
            'Aug_Dim': metrics['aug_dim'],
            'Train_Loss': metrics['train_loss'],
            'Test_MAE': metrics['mae_mean'],
            'MAE_Std': metrics['mae_std'],
            'Test_MSE': metrics['mse_mean'],
            'MSE_Std': metrics['mse_std']
        })

    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Aug_Dim')

    # Save results
    os.makedirs('results/comparisons', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f'results/comparisons/model_comparison_{timestamp}.csv'
    df.to_csv(csv_path, index=False)

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # Find best model
    best_idx = df['Test_MAE'].idxmin()
    best_model = df.loc[best_idx]

    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Test MAE: {best_model['Test_MAE']:.6f} ± {best_model['MAE_Std']:.6f}")
    print(f"   Test MSE: {best_model['Test_MSE']:.6f} ± {best_model['MSE_Std']:.6f}")

    # Create comparison plots
    print(f"\n📈 Generating comparison plots...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Training Loss vs Dimensions
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    ax.plot(df['Aug_Dim'], df['Train_Loss'], 'o-', linewidth=2.5, markersize=10, color='darkblue')
    for i, (x, y) in enumerate(zip(df['Aug_Dim'], df['Train_Loss'])):
        ax.scatter(x, y, s=200, c=[colors[i]], edgecolors='black', linewidth=2, zorder=5)
    ax.set_xlabel('State Space Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Best Training Loss', fontsize=13, fontweight='bold')
    ax.set_title('Training Loss vs Complexity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['Aug_Dim'].values)

    # 2. Test MAE vs Dimensions
    ax = axes[0, 1]
    ax.errorbar(df['Aug_Dim'], df['Test_MAE'], yerr=df['MAE_Std'],
                fmt='o-', linewidth=2.5, markersize=10, capsize=8, capthick=2, color='darkgreen')
    for i, (x, y) in enumerate(zip(df['Aug_Dim'], df['Test_MAE'])):
        ax.scatter(x, y, s=200, c=[colors[i]], edgecolors='black', linewidth=2, zorder=5)
    ax.set_xlabel('State Space Dimension', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test MAE', fontsize=13, fontweight='bold')
    ax.set_title('Forecast Accuracy vs Complexity', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(df['Aug_Dim'].values)
    ax.axhline(y=df['Test_MAE'].min(), color='green', linestyle='--', alpha=0.5, linewidth=2)

    # 3. Bias-Variance Tradeoff
    ax = axes[1, 0]
    for i, row in df.iterrows():
        ax.scatter(row['Train_Loss'], row['Test_MAE'], s=300, c=[colors[i]],
                  edgecolors='black', linewidth=2,
                  label=f"dim={int(row['Aug_Dim'])}", alpha=0.8)
    ax.set_xlabel('Training Loss (Lower = Better Fit)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test MAE (Lower = Better)', fontsize=13, fontweight='bold')
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # 4. Summary Bar Chart
    ax = axes[1, 1]
    x_pos = np.arange(len(df))
    width = 0.35

    # Normalize
    train_norm = (df['Train_Loss'] - df['Train_Loss'].min()) / (df['Train_Loss'].max() - df['Train_Loss'].min())
    test_norm = (df['Test_MAE'] - df['Test_MAE'].min()) / (df['Test_MAE'].max() - df['Test_MAE'].min())

    ax.bar(x_pos - width/2, train_norm, width, label='Train Loss (norm)',
           color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.bar(x_pos + width/2, test_norm, width, label='Test MAE (norm)',
           color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Normalized Performance\n(Lower = Better)', fontsize=12, fontweight='bold')
    ax.set_title('Relative Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"dim={int(d)}" for d in df['Aug_Dim']], fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plot_path = f'results/comparisons/model_comparison_{timestamp}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {plot_path}")

    print(f"\n✅ Comparison saved to: results/comparisons/")
    print("="*80)


if __name__ == "__main__":
    main()
