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
import argparse

# Add models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from neural_ode import ODEFunc, NeuralODE, forecast as neural_forecast, load_data, prepare_training_data
from augmented_ode import AugmentedODEFunc, AugmentedNeuralODE, forecast as aug_forecast
from second_order_ode import SecondOrderODEFunc, SecondOrderNeuralODE, forecast as second_order_forecast
from latent_ode import Encoder, Decoder, LatentODEFunc, LatentNeuralODE, forecast as latent_forecast


def get_model_handler(model_type):
    """Returns a handler object for a given model type."""
    if model_type == 'neural_ode':
        return {
            'loader': load_neural_ode,
            'forecaster': run_neural_ode_forecast,
            'display_name': 'Neural ODE'
        }
    elif model_type == 'augmented':
        return {
            'loader': load_augmented_ode,
            'forecaster': run_augmented_ode_forecast,
            'display_name': 'Augmented ODE'
        }
    elif model_type == 'second_order':
        return {
            'loader': load_second_order_ode,
            'forecaster': run_second_order_ode_forecast,
            'display_name': '2nd-Order ODE'
        }
    elif model_type == 'latent':
        return {
            'loader': load_latent_ode,
            'forecaster': run_latent_ode_forecast,
            'display_name': 'Latent ODE'
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_neural_ode(checkpoint):
    ode_func = ODEFunc(hidden_dim=128, num_layers=4)
    model = NeuralODE(ode_func)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, {'aug_dim': 1}


def load_augmented_ode(checkpoint):
    aug_dim = checkpoint['aug_dim']
    ode_func = AugmentedODEFunc(aug_dim=aug_dim, hidden_dim=128, num_layers=4)
    model = AugmentedNeuralODE(ode_func, aug_dim=aug_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, {'aug_dim': aug_dim}


def load_second_order_ode(checkpoint):
    ode_func = SecondOrderODEFunc(hidden_dim=128, num_layers=4)
    model = SecondOrderNeuralODE(ode_func)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, {'aug_dim': 2}


def load_latent_ode(checkpoint):
    latent_dim = checkpoint['latent_dim']
    encoder = Encoder(latent_dim=latent_dim, hidden_dim=64)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=64)
    ode_func = LatentODEFunc(latent_dim=latent_dim, hidden_dim=128, num_layers=3)
    model = LatentNeuralODE(encoder, decoder, ode_func, latent_dim)
    model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    model.ode_func.load_state_dict(checkpoint['ode_func_state_dict'])
    return model, {'aug_dim': latent_dim}


def run_neural_ode_forecast(model_info, y0, val_time, *args, **kwargs):
    pred = neural_forecast(model_info['model'], y0, val_time,
                           model_info['time_min'], model_info['time_max'],
                           model_info['data_mean'], model_info['data_std'])
    return pred, None


def run_augmented_ode_forecast(model_info, y0, val_time, *args, **kwargs):
    pred, _ = aug_forecast(model_info['model'], y0, val_time,
                           model_info['time_min'], model_info['time_max'],
                           model_info['data_mean'], model_info['data_std'])
    return pred, None


def run_second_order_ode_forecast(model_info, y0, val_time, train_data_norm=None, train_time_norm=None, i=None):
    dt = train_time_norm[-1] - train_time_norm[-2]
    v0 = (train_data_norm[-1, i] - train_data_norm[-2, i]) / dt
    v0 = np.array([[v0]])
    pred, _ = second_order_forecast(model_info['model'], y0, v0, val_time,
                                    model_info['time_min'], model_info['time_max'],
                                    model_info['data_mean'], model_info['data_std'])
    return pred, None


def run_latent_ode_forecast(model_info, y0, val_time, *args, **kwargs):
    pred, _ = latent_forecast(model_info['model'], y0, val_time,
                              model_info['time_min'], model_info['time_max'],
                              model_info['data_mean'], model_info['data_std'])
    return pred, None


def load_latest_model(model_dir, model_type):
    """Load the most recent model from a directory"""
    pattern = os.path.join(model_dir, '*.pt')
    model_files = glob.glob(pattern)

    if not model_files:
        print(f"⚠️  No models found in {model_dir}")
        return None

    latest_model_path = max(model_files, key=os.path.getctime)
    print(f"Loading: {os.path.basename(latest_model_path)}")
    checkpoint = torch.load(latest_model_path, map_location='cpu', weights_only=False)

    handler = get_model_handler(model_type)
    model, extra_info = handler['loader'](checkpoint)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_info = {
        'model': model,
        'time_min': checkpoint['time_min'],
        'time_max': checkpoint['time_max'],
        'data_mean': checkpoint['data_mean'],
        'data_std': checkpoint['data_std'],
        'losses': checkpoint['losses'],
        'num_params': num_params,
        'display_name': handler['display_name']
    }
    model_info.update(extra_info)
    return model_info


def evaluate_model(model_info, train_data, train_time, val_data, val_time, model_type):
    """Evaluate a model and return metrics and predictions"""
    model = model_info['model']
    model.eval()

    mse_scores, mae_scores, all_preds = [], [], []

    # Normalize data for velocity estimation (for second-order)
    train_data_norm, train_time_norm = None, None
    if model_type == 'second_order':
        train_data_norm = (train_data - model_info['data_mean']) / model_info['data_std']
        train_time_norm = (train_time - model_info['time_min']) / (model_info['time_max'] - model_info['time_min'])

    handler = get_model_handler(model_type)
    for i in range(train_data.shape[1]):
        y0 = train_data[-1:, i:i+1]
        pred, _ = handler['forecaster'](model_info, y0, val_time, train_data_norm, train_time_norm, i)

        mse = np.mean((pred - val_data[:, i:i+1]) ** 2)
        mae = np.mean(np.abs(pred - val_data[:, i:i+1]))
        mse_scores.append(mse)
        mae_scores.append(mae)
        all_preds.append(pred)

    # Average predictions across all validation series
    avg_pred = np.mean(np.array(all_preds), axis=0)

    return {
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'train_loss': min(model_info['losses']),
        'aug_dim': model_info['aug_dim'],
        'num_params': model_info['num_params'],
        'predictions': avg_pred
    }


def plot_results(df, val_data, val_time, timestamp, results_dir):
    """Generate and save all comparison plots."""
    print(f"\n📈 Generating comparison plots...")
    os.makedirs(results_dir, exist_ok=True)

    # --- Plot Setup ---
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    model_names = df['Model'].values

    # --- Figure 1: Core Metrics ---
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Model Performance Analysis', fontsize=20, fontweight='bold')

    # 1. Test MAE vs Parameters
    ax = axes1[0, 0]
    ax.scatter(df['Params'] / 1e3, df['Test_MAE'], s=200, c=colors, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Number of Parameters (Thousands)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Accuracy vs. Model Size', fontsize=14, fontweight='bold')
    for i, txt in enumerate(model_names):
        ax.annotate(txt, (df['Params'][i] / 1e3, df['Test_MAE'][i]), xytext=(5, 5), textcoords='offset points')

    # 2. Bias-Variance Tradeoff
    ax = axes1[0, 1]
    ax.scatter(df['Train_Loss'], df['Test_MAE'], s=200, c=colors, edgecolors='black', linewidth=1.5)
    ax.set_xlabel('Training Loss (Bias)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Test MAE (Variance Proxy)', fontsize=12, fontweight='bold')
    ax.set_title('Bias-Variance Tradeoff', fontsize=14, fontweight='bold')
    for i, txt in enumerate(model_names):
        ax.annotate(txt, (df['Train_Loss'][i], df['Test_MAE'][i]), xytext=(5, 5), textcoords='offset points')

    # 3. Summary Bar Chart
    ax = axes1[1, 0]
    x_pos = np.arange(len(df))
    width = 0.35
    ax.bar(x_pos - width/2, df['Test_MAE'], width, label='Test MAE', color='skyblue', edgecolor='black')
    ax.bar(x_pos + width/2, df['Train_Loss'], width, label='Train Loss', color='salmon', edgecolor='black')
    ax.set_ylabel('Error Metric', fontsize=12, fontweight='bold')
    ax.set_title('Overall Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax.legend()

    # 4. Parameter Count Bar Chart
    ax = axes1[1, 1]
    ax.bar(x_pos, df['Params'] / 1e3, color=colors, edgecolor='black')
    ax.set_ylabel('Parameters (Thousands)', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity (Number of Parameters)', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path1 = os.path.join(results_dir, f'model_comparison_metrics_{timestamp}.png')
    plt.savefig(plot_path1, dpi=150)
    print(f"   Saved metrics plot: {plot_path1}")

    # --- Figure 2: Forecast & Residuals ---
    fig2, axes2 = plt.subplots(2, 1, figsize=(16, 12), sharex=True)
    fig2.suptitle('Qualitative Forecast Analysis', fontsize=20, fontweight='bold')
    avg_val_data = val_data.mean(axis=1)

    # 1. Forecast Plot
    ax = axes2[0]
    ax.plot(val_time, avg_val_data, 'k--', linewidth=2.5, label='Ground Truth')
    for i, row in df.iterrows():
        ax.plot(val_time, row['Predictions'], label=row['Model'], color=colors[i], linewidth=2)
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Model Forecasts vs. Ground Truth', fontsize=14, fontweight='bold')
    ax.legend()

    # 2. Residuals Plot
    ax = axes2[1]
    for i, row in df.iterrows():
        residuals = row['Predictions'].flatten() - avg_val_data.flatten()
        ax.plot(val_time, residuals, label=f"{row['Model']} Residuals", color=colors[i], alpha=0.8)
    ax.axhline(0, color='black', linestyle='--', linewidth=1.5)
    ax.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residual (Prediction - Truth)', fontsize=12, fontweight='bold')
    ax.set_title('Forecast Residuals', fontsize=14, fontweight='bold')
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path2 = os.path.join(results_dir, f'model_comparison_forecasts_{timestamp}.png')
    plt.savefig(plot_path2, dpi=150)
    print(f"   Saved forecast plot: {plot_path2}")


def main(dataset_path):
    print("="*80)
    print("COMPARING ALL NEURAL ODE MODELS")
    print("="*80)

    # Load data
    print(f"\n💾 Loading data from {dataset_path}...")
    time_axis, data, _ = load_data(dataset_path)
    train_time, train_data, val_time, val_data = prepare_training_data(time_axis, data)

    # Define models to load
    models_to_load = {
        'Regular Neural ODE': ('results/neural_ode', 'neural_ode'),
        'Augmented ODE (dim=2)': ('results/augmented_ode_dim2', 'augmented'),
        'Augmented ODE (dim=3)': ('results/augmented_ode_dim3', 'augmented'),
        'Augmented ODE (dim=6)': ('results/augmented_ode_dim6', 'augmented'),
        'Second-Order Neural ODE': ('results/second_order_ode', 'second_order'),
        'Latent Neural ODE': ('results/latent_ode', 'latent'),
    }

    # Load all models
    print("\n📦 Loading models...")
    loaded_models = {}
    for name, (path, type) in models_to_load.items():
        print(f"\n-> Loading {name} from {path}")
        model_info = load_latest_model(path, type)
        if model_info:
            loaded_models[name] = (model_info, type)

    if not loaded_models:
        print("\n❌ No models found! Run 'python train_all.py' first.")
        return

    # Evaluate all models
    print("\n" + "="*80)
    print("📊 EVALUATING MODELS...")
    print("="*80)

    results = []
    for name, (model_info, model_type) in loaded_models.items():
        print(f"Evaluating {name}...")
        metrics = evaluate_model(model_info, train_data, train_time, val_data, val_time, model_type)
        results.append({
            'Model': name,
            'Aug_Dim': metrics['aug_dim'],
            'Params': metrics['num_params'],
            'Train_Loss': metrics['train_loss'],
            'Test_MAE': metrics['mae_mean'],
            'MAE_Std': metrics['mae_std'],
            'Test_MSE': metrics['mse_mean'],
            'MSE_Std': metrics['mse_std'],
            'Predictions': metrics['predictions']
        })

    # Create results DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('Test_MAE').reset_index(drop=True)

    # Save results
    results_dir = 'results/comparisons'
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f'model_comparison_{timestamp}.csv')
    # Drop predictions before saving to CSV
    df.drop(columns=['Predictions']).to_csv(csv_path, index=False)

    # Print results
    print("\n" + "="*80)
    print("RESULTS (sorted by Test MAE)")
    print("="*80)
    print(df.drop(columns=['Predictions']).to_string(index=False))
    print("="*80)

    # Find best model
    best_model = df.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   Test MAE: {best_model['Test_MAE']:.6f} ± {best_model['MAE_Std']:.6f}")
    print(f"   Num Params: {best_model['Params'] / 1e3:.1f}k")

    # Create comparison plots
    plot_results(df, val_data, val_time, timestamp, results_dir)

    print(f"\n✅ Comparison artifacts saved to: {results_dir}/")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained Neural ODE models.")
    parser.add_argument('--dataset', type=str, default='data/CRE.csv',
                        help='Path to the dataset CSV file.')
    args = parser.parse_args()
    main(args.dataset)
