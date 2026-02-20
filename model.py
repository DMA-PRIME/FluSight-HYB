import torch
import torch.nn as nn

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        """
        preds: (batch_size, output_steps, num_quantiles)
        target: (batch_size, output_steps)
        """
        assert preds.shape[2] == len(self.quantiles)
        
        loss = 0
        # Target needs to be expanded to match preds for broadcasting or iterated
        # Let's iterate over quantiles for clarity and safety
        target = target.unsqueeze(2) # (batch, steps, 1)
        
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, :, i:i+1]
            loss += torch.max((q - 1) * errors, q * errors).mean()
            
        return loss

class FluForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps, num_quantiles, num_layers=2, dropout=0.2):
        super().__init__()
        self.output_steps = output_steps
        self.num_quantiles = num_quantiles
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # We want to output 'output_steps' predictions for each of 'num_quantiles'
        # One approach: Use the last hidden state to predict all 4 steps at once.
        self.fc = nn.Linear(hidden_size, output_steps * num_quantiles)
        
    def forward(self, x):
        # x: (batch, input_steps, input_size)
        
        # LSTM output: (batch, input_steps, hidden_size)
        # We only care about the last time step's output for forecasting the sequence
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :] # (batch, hidden_size)
        
        # Predict all quantiles for all steps
        prediction = self.fc(last_hidden) # (batch, output_steps * num_quantiles)
        
        # Reshape to (batch, output_steps, num_quantiles)
        prediction = prediction.view(-1, self.output_steps, self.num_quantiles)
        
        return prediction
