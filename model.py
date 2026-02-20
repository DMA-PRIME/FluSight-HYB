import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attn(lstm_output), dim=1) # (batch, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1) # (batch, hidden_size)
        return context, attn_weights

class FluForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        
        # Predicting the DELTA from the last known value for each step
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps * num_quantiles)
        )
        
    def forward(self, x):
        # x: (batch, input_steps, input_size)
        
        # LSTM output: (batch, input_steps, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Apply Attention to all time steps
        context, _ = self.attention(lstm_out) # (batch, hidden_size)
        
        # Predict all quantiles for all steps
        prediction = self.fc(context) # (batch, output_steps * num_quantiles)
        prediction = prediction.view(-1, self.output_steps, self.num_quantiles)
        
        return prediction
