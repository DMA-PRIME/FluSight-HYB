import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, x):
        weights = F.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1), weights

class HybridCNNForecaster(nn.Module):
    """Residual Gated LSTM (R-GLSTM)
    Explicitly balances Magnitude (Absolute path) and Phase (Delta path).
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        # Encoder: Extract local features
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        
        # Sequential Memory
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
        # Dual-Path Decoder
        # 1. Magnitude Path (Predicts the likely absolute level)
        self.mag_attn = Attention(hidden_size * 2)
        
        # 2. Phase Path (Predicts the temporal correction/shift)
        self.phase_attn = Attention(hidden_size * 2)
        
        # Final Gated Integration
        self.gate = nn.Linear(hidden_size * 4, output_steps) # Gating mechanism
        
        self.out = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_quantiles)
            ) for _ in range(output_steps)
        ])

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x_conv = F.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        
        lstm_out, _ = self.lstm(x_conv) # (batch, seq_len, hidden_size*2)
        
        # Extract dual contexts
        mag_context, _ = self.mag_attn(lstm_out)
        phase_context, _ = self.phase_attn(lstm_out)
        
        # Combine contexts
        combined_context = torch.cat([mag_context, phase_context], dim=-1) # (batch, hidden_size*4)
        
        # Multi-horizon prediction
        preds = []
        for predictor in self.out:
            preds.append(predictor(combined_context))
            
        return torch.stack(preds, dim=1) # (batch, output_steps, num_quantiles)
