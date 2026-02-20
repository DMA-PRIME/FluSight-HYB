import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attn(x), dim=1) # (batch, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1) # (batch, hidden_size)
        return context, attn_weights

class FluForecaster(nn.Module):
    """LSTM + Attention Forecaster"""
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps * num_quantiles)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        prediction = self.fc(context)
        return prediction.view(-1, self.output_steps, self.num_quantiles)

class HybridCNNForecaster(nn.Module):
    """CNN + LSTM + Attention Forecaster
    CNN layers extract local 'peaky' features while LSTM handles long-term dependencies.
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        # 1D Convolutional layer to extract local patterns (magnitude of peaks)
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.attention = Attention(hidden_size)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_steps * num_quantiles)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        # Conv1d expects (batch, channels, seq_len)
        x_conv = x.transpose(1, 2)
        conv_out = self.conv(x_conv)
        conv_out = conv_out.transpose(1, 2) # (batch, seq_len, hidden_size)
        
        lstm_out, _ = self.lstm(conv_out)
        context, _ = self.attention(lstm_out)
        
        prediction = self.fc(context)
        return prediction.view(-1, self.output_steps, self.num_quantiles)
