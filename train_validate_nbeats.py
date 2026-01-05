import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import warnings
import os
warnings.filterwarnings("ignore")

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("PyTorch not available. Please install: pip install torch")

class NBeatsBlock(nn.Module):
    """Single N-BEATS block with backcast and forecast branches."""
    def __init__(self, input_size, theta_size, basis_function, hidden_size=256):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Backcast and forecast parameter layers
        self.theta_b = nn.Linear(hidden_size, theta_size)
        self.theta_f = nn.Linear(hidden_size, theta_size)
        
    def forward(self, x):
        # x shape: (batch, input_size)
        # Shared layers
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        h = F.relu(self.fc4(h))
        
        # Get theta parameters
        theta_b = self.theta_b(h)  # backcast coefficients
        theta_f = self.theta_f(h)  # forecast coefficients
        
        # Apply basis functions
        backcast = self.basis_function(theta_b, self.input_size)
        forecast = self.basis_function(theta_f, 1)  # Predict 1 step
        
        return backcast, forecast

class GenericBasis(nn.Module):
    """Generic basis function for N-BEATS."""
    def __init__(self, backcast_size, forecast_size):
        super(GenericBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        
    def forward(self, theta, target_size):
        # theta shape: (batch, theta_size)
        # For generic basis, theta directly produces the output
        if target_size == self.backcast_size:
            return theta[:, :self.backcast_size]
        else:
            return theta[:, :self.forecast_size]

class NBeatsNet(nn.Module):
    """N-BEATS network with stacks of blocks."""
    def __init__(self, input_size, num_stacks=2, num_blocks_per_stack=3, hidden_size=128):
        super(NBeatsNet, self).__init__()
        self.input_size = input_size
        self.num_stacks = num_stacks
        
        # Create stacks
        self.stacks = nn.ModuleList()
        for _ in range(num_stacks):
            stack = nn.ModuleList()
            for _ in range(num_blocks_per_stack):
                # Generic basis function
                basis_function = lambda theta, size: theta[:, :size]
                block = NBeatsBlock(
                    input_size=input_size,
                    theta_size=max(input_size, 1),
                    basis_function=basis_function,
                    hidden_size=hidden_size
                )
                stack.append(block)
            self.stacks.append(stack)
    
    def forward(self, x):
        # x shape: (batch, input_size)
        residuals = x
        forecast = torch.zeros(x.size(0), 1).to(x.device)
        
        for stack in self.stacks:
            for block in stack:
                # Each block produces backcast and forecast
                backcast, block_forecast = block(residuals)
                
                # Ensure shapes match
                if backcast.size(1) != residuals.size(1):
                    backcast = backcast[:, :residuals.size(1)]
                
                # Update residuals and forecast
                residuals = residuals - backcast
                forecast = forecast + block_forecast
        
        return forecast

class TimeSeriesDataset(Dataset):
    """Dataset for time series windows."""
    def __init__(self, windows, context_length, prediction_length):
        self.windows = windows
        self.context_length = context_length
        self.prediction_length = prediction_length
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        if len(window) < self.context_length + self.prediction_length:
            window = np.pad(window, (0, self.context_length + self.prediction_length - len(window)))
        
        x = window[:self.context_length].astype(np.float32)
        y = window[self.context_length].astype(np.float32)
        
        return torch.FloatTensor(x), torch.FloatTensor([y])

def load_data(file_path, window_size=25):
    """Load data and create sliding windows."""
    df = pd.read_csv(file_path)
    series_cols = [col for col in df.columns if col != 'time-axis']
    all_windows = []
    
    for col in series_cols:
        series = df[col].values
        if len(series) < window_size:
            continue
        for i in range(len(series) - window_size + 1):
            all_windows.append(series[i : i + window_size])
            
    return np.array(all_windows)

def load_naive_baseline(filepath):
    """Load naive baseline MAE values for MASE calculation."""
    df = pd.read_csv(filepath)
    naive_maes = {}
    for _, row in df.iterrows():
        naive_maes[int(row['horizon'])] = row['naive_mae']
    return naive_maes

def train_model(model, train_loader, epochs=30, lr=0.001, device='cpu'):
    """Train the N-BEATS model."""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

def forecast_recursive(model, history, steps, context_length, device='cpu'):
    """Recursively forecast multiple steps ahead."""
    model.eval()
    history = list(history[-context_length:])
    forecasts = []
    
    with torch.no_grad():
        for _ in range(steps):
            x = np.array(history[-context_length:]).astype(np.float32)
            x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)
            
            pred = model(x_tensor).item()
            forecasts.append(pred)
            history.append(pred)
    
    return np.array(forecasts)

def evaluate_horizon(model, test_windows, horizon, naive_mae, context_length=15, device='cpu'):
    """Evaluate N-BEATS model for a specific forecast horizon using MASE."""
    actuals, predictions = [], []
    
    for window in test_windows:
        history_size = len(window) - horizon
        if history_size < context_length:
            continue
            
        initial_history = window[:history_size]
        actual_future = window[history_size:]
        
        try:
            pred = forecast_recursive(model, initial_history, horizon, context_length, device)
            actuals.extend(actual_future)
            predictions.extend(pred)
        except Exception:
            continue
    
    if len(actuals) == 0:
        return {'MASE': np.nan, 'MSE': np.nan, 'MAE': np.nan}
    
    y_true = np.array(actuals)
    y_pred = np.array(predictions)
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    mase = mae / naive_mae
    
    return {'MASE': mase, 'MSE': mse, 'MAE': mae}

if __name__ == "__main__":
    if not PYTORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed!")
        print("Please install with: pip install torch")
        exit(1)
    
    # Configuration
    FILE_PATH = 'data/CRE.csv'
    WINDOW_SIZE = 25
    HORIZONS = [1, 2, 3, 5, 7, 10]
    CONTEXT_LENGTH = 15
    PREDICTION_LENGTH = 1
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print(f"\nLoading data from {FILE_PATH}...")
    windows = load_data(FILE_PATH, WINDOW_SIZE)
    
    TRAIN_SIZE = int(0.8 * len(windows))
    
    np.random.seed(42)
    torch.manual_seed(42)
    np.random.shuffle(windows)
    
    train_windows = windows[:TRAIN_SIZE]
    test_windows = windows[TRAIN_SIZE:]
    
    MAX_TRAIN = 2000
    if len(train_windows) > MAX_TRAIN:
        print(f"Downsampling training data from {len(train_windows)} to {MAX_TRAIN} windows...")
        train_windows = train_windows[:MAX_TRAIN]
    
    print(f"Train windows: {len(train_windows)}")
    
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    print("\nLoading naive baseline MAE values...")
    naive_baseline_path = os.path.join(output_dir, 'naive_results.csv')
    naive_maes = load_naive_baseline(naive_baseline_path)
    
    print("\nPreparing N-BEATS training data...")
    train_dataset = TimeSeriesDataset(train_windows, CONTEXT_LENGTH, PREDICTION_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"\nTraining N-BEATS model...")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Stacks: 2")
    print(f"  Blocks per stack: 3")
    print(f"  Hidden size: 128")
    print(f"  Training samples: {len(train_windows)}")
    
    model = NBeatsNet(
        input_size=CONTEXT_LENGTH,
        num_stacks=2,
        num_blocks_per_stack=3,
        hidden_size=128
    ).to(device)
    
    train_model(model, train_loader, epochs=30, lr=0.001, device=device)
    print("Training complete!")
    
    print("\n" + "="*50)
    print("    N-BEATS MULTI-HORIZON EVALUATION (MASE)")
    print("="*50)
    print(f"{'Horizon':<10} | {'MASE':<10} | {'MSE':<12} | {'MAE':<12}")
    print("-" * 50)
    
    results = []
    for h in HORIZONS:
        metrics = evaluate_horizon(model, test_windows, h, naive_maes[h], CONTEXT_LENGTH, device)
        print(f"{h:<10} | {metrics['MASE']:>9.4f} | {metrics['MSE']:>12.6f} | {metrics['MAE']:>12.6f}")
        
        results.append({
            'horizon': h,
            'mase': metrics['MASE'],
            'mse': metrics['MSE'],
            'mae': metrics['MAE']
        })
    
    print("="*50)
    
    output_path = os.path.join(output_dir, 'nbeats_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")