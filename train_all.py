"""
Train all Neural ODE variants
Usage: python train_all.py [--epochs 500] [--lr 0.001] [--device cpu]
"""

import argparse
import sys
import os

# Add models directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from neural_ode import train_neural_ode_model
from augmented_ode import train_augmented_ode_model
from second_order_ode import train_second_order_ode_model
from latent_ode import train_latent_ode_model


def main():
    parser = argparse.ArgumentParser(description='Train all Neural ODE variants')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--models', type=str, default='all',
                       help='Which models to train: all, basic, advanced, neural_ode, augmented, second_order, latent (comma-separated)')
    args = parser.parse_args()

    models_to_train = args.models.lower().split(',')

    print("="*80)
    print("TRAINING NEURAL ODE MODELS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"  Models: {models_to_train}")
    print("="*80)

    # Train Regular Neural ODE
    if 'all' in models_to_train or 'basic' in models_to_train or 'neural_ode' in models_to_train:
        print("\n" + "="*80)
        print("1. TRAINING REGULAR NEURAL ODE")
        print("="*80)
        train_neural_ode_model(
            data_path='data/CRE.csv',
            output_dir='results/neural_ode',
            n_epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    # Train Augmented Neural ODE variants
    if 'all' in models_to_train or 'basic' in models_to_train or 'augmented' in models_to_train:
        for aug_dim in [2, 3, 6]:
            print("\n" + "="*80)
            print(f"2. TRAINING AUGMENTED NEURAL ODE (dim={aug_dim})")
            print("="*80)
            train_augmented_ode_model(
                data_path='data/CRE.csv',
                output_dir=f'results/augmented_ode_dim{aug_dim}',
                aug_dim=aug_dim,
                n_epochs=args.epochs,
                lr=args.lr,
                device=args.device
            )

    # Train Second-Order Neural ODE
    if 'all' in models_to_train or 'advanced' in models_to_train or 'second_order' in models_to_train:
        print("\n" + "="*80)
        print("3. TRAINING SECOND-ORDER NEURAL ODE")
        print("="*80)
        train_second_order_ode_model(
            data_path='data/CRE.csv',
            output_dir='results/second_order_ode',
            n_epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    # Train Latent Neural ODE
    if 'all' in models_to_train or 'advanced' in models_to_train or 'latent' in models_to_train:
        print("\n" + "="*80)
        print("4. TRAINING LATENT NEURAL ODE")
        print("="*80)
        train_latent_ode_model(
            data_path='data/CRE.csv',
            output_dir='results/latent_ode',
            latent_dim=20,
            n_epochs=args.epochs,
            lr=args.lr,
            device=args.device
        )

    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  Basic models:")
    print("    - results/neural_ode/")
    print("    - results/augmented_ode_dim{2,3,6}/")
    print("  Advanced models:")
    print("    - results/second_order_ode/")
    print("    - results/latent_ode/")
    print("\nTo compare models, run: python compare.py")


if __name__ == "__main__":
    main()
