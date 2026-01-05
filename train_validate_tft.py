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

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network (GRN) - key component of TFT."""
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(GatedResidualNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(hidden_size, output_size)
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Skip connection
        if input_size != output_size:
            self.skip = nn.Linear(input_size, output_size)
        else:
            self.skip = None
    
    def forward(self, x):
        # Main path
        eta1 = self.dropout(F.elu(self.layer1(x)))
        eta2 = self.layer2(eta1)
        
        # Gating
        gate = torch.sigmoid(self.gate(eta1))
        
        # Skip connection
        if self.skip is not None:
            x = self.skip(x)
        
        # Gated residual
        out = self.layer_norm(x + gate * eta2)
        return out

class TemporalFusionTransformer(nn.Module):
    """Simplified Temporal Fusion Transformer for time series forecasting."""
    def __init__(self, input_size=1, hidden_size=64, num_heads=4, dropout=0.1):
        super(TemporalFusionTransformer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Input encoding
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Gated Residual Networks
        self.grn1 = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.grn2 = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # Output layers
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.output_layer = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        batch_size, seq_len, _ = x.shape
        
        # Input embedding
        embedded = self.input_embedding(x)  # (batch, seq_len, hidden)
        
        # Variable selection via GRN
        selected = self.grn1(embedded)
        
        # LSTM encoding
        lstm_out, _ = self.lstm(selected)  # (batch, seq_len, hidden)
        
        # Self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.dropout(attn_out)
        
        # Add & Norm (residual connection)
        attn_out = attn_out + lstm_out
        
        # Temporal fusion via GRN
        fused = self.grn2(attn_out)
        
        # Take last timestep
        last_step = fused[:, -1, :]  # (batch, hidden)
        
        # Output processing
        output_processed = self.output_grn(last_step)
        prediction = self.output_layer(output_processed)
        
        return prediction

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
        
        x = window[:self.context_length].reshape(-1, 1).astype(np.float32)
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
    """Train the TFT model."""
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
            x = np.array(history[-context_length:]).reshape(1, -1, 1).astype(np.float32)
            x_tensor = torch.FloatTensor(x).to(device)
            
            pred = model(x_tensor).item()
            forecasts.append(pred)
            history.append(pred)
    
    return np.array(forecasts)

def evaluate_horizon(model, test_windows, horizon, naive_mae, context_length=15, device='cpu'):
    """Evaluate TFT model for a specific forecast horizon using MASE."""
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
    
    print("\nPreparing TFT training data...")
    train_dataset = TimeSeriesDataset(train_windows, CONTEXT_LENGTH, PREDICTION_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    print(f"\nTraining Temporal Fusion Transformer (TFT)...")
    print(f"  Context length: {CONTEXT_LENGTH}")
    print(f"  Hidden size: 64")
    print(f"  Attention heads: 4")
    print(f"  Training samples: {len(train_windows)}")
    
    model = TemporalFusionTransformer(
        input_size=1,
        hidden_size=64,
        num_heads=4,
        dropout=0.1
    ).to(device)
    
    train_model(model, train_loader, epochs=30, lr=0.001, device=device)
    print("Training complete!")
    
    print("\n" + "="*50)
    print("    TFT MULTI-HORIZON EVALUATION (MASE)")
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
    
    output_path = os.path.join(output_dir, 'tft_results.csv')
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")