import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        attn_weights = F.softmax(self.attn(x), dim=1)
        context = torch.sum(attn_weights * x, dim=1)
        return context, attn_weights

class HybridCNNForecaster(nn.Module):
    """Refined Dilated CNN + LSTM + Attention Forecaster
    Uses Dilated Convolutions to expand the receptive field and capture early warning signals.
    """
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.3):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        # Dilated Convolutional layers to capture multi-scale temporal dependencies
        # Dilation=2 allows the model to "see" further back with the same kernel size
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2, dilation=2)
        
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
        x_conv = x.transpose(1, 2)
        
        x1 = F.relu(self.conv1(x_conv))
        x2 = F.relu(self.conv2(x1))
        
        conv_out = x2.transpose(1, 2)
        
        lstm_out, _ = self.lstm(conv_out)
        context, _ = self.attention(lstm_out)
        
        prediction = self.fc(context)
        return prediction.view(-1, self.output_steps, self.num_quantiles)
