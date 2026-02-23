import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadTemporalAttention(nn.Module):
    def __init__(self, hidden_size, num_horizons):
        super().__init__()
        # Separate attention for each forecast week to capture different lead times
        self.horizons = num_horizons
        self.attn_heads = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_horizons)
        ])

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        contexts = []
        for head in self.attn_heads:
            weights = F.softmax(head(x), dim=1) # (batch, seq_len, 1)
            context = torch.sum(weights * x, dim=1) # (batch, hidden_size)
            contexts.append(context)
        return torch.stack(contexts, dim=1) # (batch, num_horizons, hidden_size)

class HybridCNNForecaster(nn.Module):
    """Adaptive Lag-Compensating Forecaster
    Uses horizon-specific attention heads to independently align each week's lead time.
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        self.hidden_size = hidden_size
        
        # Dilated Encoder
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Adaptive Lag Compensation: Separate attention heads for each week ahead
        self.temporal_heads = MultiHeadTemporalAttention(hidden_size, output_steps)
        
        # Horizon-specific predictors
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_quantiles)
            ) for _ in range(output_steps)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_conv = x.transpose(1, 2)
        x1 = F.relu(self.conv1(x_conv))
        x2 = F.relu(self.conv2(x1))
        conv_out = x2.transpose(1, 2)
        
        lstm_out, _ = self.lstm(conv_out)
        
        # Get horizon-specific contexts (Adaptive Alignment)
        horizon_contexts = self.temporal_heads(lstm_out) # (batch, output_steps, hidden_size)
        
        # Predict each week independently
        preds = []
        for i in range(self.output_steps):
            preds.append(self.heads[i](horizon_contexts[:, i, :]))
            
        prediction = torch.stack(preds, dim=1) # (batch, output_steps, num_quantiles)
        return prediction
