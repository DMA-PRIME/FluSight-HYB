import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(0.2)
        self.norm = nn.GroupNorm(8, out_channels) # Better for small batches than BatchNorm

    def forward(self, x):
        # x: (batch, channels, seq_len)
        x = self.conv(x)
        x = x[:, :, :-self.padding] # Remove future padding
        return F.relu(self.norm(self.dropout(x)))

class HybridCNNForecaster(nn.Module):
    """TCN-based Forecaster for Lag Reduction
    Replaces LSTM with Causal Convolutions to eliminate sequential persistence bias.
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        # TCN Backbone: Captures features at different temporal scales
        self.tcn = nn.Sequential(
            CausalConvBlock(input_size, hidden_size, kernel_size=3, dilation=1),
            CausalConvBlock(hidden_size, hidden_size, kernel_size=3, dilation=2),
            CausalConvBlock(hidden_size, hidden_size, kernel_size=3, dilation=4)
        )
        
        # Multi-scale aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Independent Predictors for each week-ahead (Direct Multi-Horizon)
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_quantiles)
            ) for _ in range(output_steps)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2) # (batch, channels, seq_len)
        
        # TCN features
        features = self.tcn(x) # (batch, hidden_size, seq_len)
        
        # Aggregate across time (Focus on the 'whole window context')
        # We use both the 'current' time step features and the window-wide average
        last_step = features[:, :, -1] # (batch, hidden_size)
        avg_pool = self.global_pool(features).squeeze(2) # (batch, hidden_size)
        
        context = (last_step + avg_pool) / 2.0
        
        # Generate predictions for each horizon independently
        preds = [predictor(context) for predictor in self.predictors]
        
        return torch.stack(preds, dim=1) # (batch, output_steps, num_quantiles)
