import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

from data_loader import load_and_preprocess_data, create_dataloaders
from model import HybridCNNForecaster

# Configuration
PRISMA_PATH = 'dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv'
TARGET_PATH = 'dataset/target-ed-visits-prop.csv'
INPUT_WINDOW = 10
OUTPUT_WINDOW = 4
QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def quantile_loss(preds, target, quantiles):
    loss = 0
    target_unsq = target.unsqueeze(2)
    
    # 1. Standard Pinball Loss
    for i, q in enumerate(quantiles):
        errors = target_unsq - preds[:, :, i:i+1]
        weight = 1.0 + (target_unsq.abs() * 10.0) # Even heavier weight on changes
        loss += (torch.max((q - 1) * errors, q * errors) * weight).mean()
        
    # 2. Advanced Gradient Alignment (Phase Penalty)
    # Penalize not just slope mismatch, but direction mismatch specifically
    if target.shape[1] > 1:
        target_slope = target[:, 1:] - target[:, :-1]
        pred_median = preds[:, :, 11]
        pred_slope = pred_median[:, 1:] - pred_median[:, :-1]
        
        # Phase Penalty: MSE of slopes
        slope_loss = F.mse_loss(pred_slope, target_slope)
        
        # Sign Penalty: Encourage predicting the correct direction of change
        sign_loss = torch.mean(F.relu(-pred_slope * target_slope)) 
        
        loss += slope_loss * 5.0 + sign_loss * 2.0
        
    return loss

def train_model(model, train_loader, val_loader, optimizer, epochs=1000):
    model.to(DEVICE)
    best_val_loss = float('inf')
    patience = 80
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = quantile_loss(preds, y_batch, QUANTILES)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                preds = model(X_batch)
                loss = quantile_loss(preds, y_batch, QUANTILES)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
    print("Training complete. Best Val Loss:", best_val_loss)
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def predict_and_postprocess(model, X_input, anchor_val, scaler_target, quantiles):
    """
    anchor_val: The last known absolute scaled value (from 'anchors' list)
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
            
        preds_deltas = model(X_tensor) 
        preds_deltas = preds_deltas.cpu().numpy().squeeze(0) # (4, 23)
        
        # Adaptive Compensation: Restore using the explicit anchor
        preds_scaled = anchor_val + preds_deltas
        
    preds_original_scale = np.zeros_like(preds_scaled)
    for i in range(len(quantiles)):
        col_preds = preds_scaled[:, i].reshape(-1, 1)
        inv_scaled = scaler_target.inverse_transform(col_preds).flatten()
        preds_original_scale[:, i] = np.expm1(inv_scaled)
        
    preds_original_scale = np.clip(preds_original_scale, 0, 1)
    preds_sorted = np.sort(preds_original_scale, axis=1)
    return preds_sorted

def main():
    print("Loading data with Adaptive Lag Compensation...")
    X, y, dates, scaler_target, df_merged, last_input, last_date, anchors, last_anchor = load_and_preprocess_data(PRISMA_PATH, TARGET_PATH, INPUT_WINDOW, OUTPUT_WINDOW)
    
    train_loader, val_loader = create_dataloaders(X, y, BATCH_SIZE)
    
    model = HybridCNNForecaster(
        input_size=X.shape[2], 
        hidden_size=HIDDEN_SIZE, 
        output_steps=OUTPUT_WINDOW, 
        num_quantiles=len(QUANTILES), 
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    print("Starting training (Adaptive Lag Compensation mode)...")
    model = train_model(model, train_loader, val_loader, optimizer, epochs=EPOCHS)
    
    # --- Generation & Backtesting ---
    all_forecast_rows = []
    for i in range(len(X)):
        x_in = X[i]
        anchor = anchors[i]
        forecast_start_date = dates[i]
        reference_date = forecast_start_date - timedelta(weeks=1)
        
        preds = predict_and_postprocess(model, x_in, anchor, scaler_target, QUANTILES)
        
        for step in range(OUTPUT_WINDOW):
            target_end_date = forecast_start_date + timedelta(weeks=step)
            horizon = (target_end_date - reference_date).days // 7
            for q_idx, q_val in enumerate(QUANTILES):
                all_forecast_rows.append({
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'target': 'wk inc flu prop ed visits',
                    'horizon': horizon,
                    'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                    'location': 'South Carolina',
                    'output_type': 'quantile',
                    'output_type_id': q_val,
                    'value': preds[step, q_idx],
                    'type': 'backtest'
                })
            
    # --- Future Forecast ---
    future_preds = predict_and_postprocess(model, last_input, last_anchor, scaler_target, QUANTILES)
    future_start_date = last_date + timedelta(weeks=1)
    for step in range(OUTPUT_WINDOW):
        target_end_date = future_start_date + timedelta(weeks=step)
        horizon = (target_end_date - last_date).days // 7
        for q_idx, q_val in enumerate(QUANTILES):
            all_forecast_rows.append({
                'reference_date': last_date.strftime('%Y-%m-%d'),
                'target': 'wk inc flu prop ed visits',
                'horizon': horizon,
                'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                'location': 'South Carolina',
                'output_type': 'quantile',
                'output_type_id': q_val,
                'value': future_preds[step, q_idx],
                'type': 'future'
            })
            
    results_df = pd.DataFrame(all_forecast_rows)
    results_df.drop(columns=['type']).to_csv('forecast_results.csv', index=False)
    
    # Validation Metric
    backtest_medians = results_df[(results_df['type']=='backtest') & (results_df['output_type_id'] == 0.5)].copy()
    backtest_medians['target_end_date'] = pd.to_datetime(backtest_medians['target_end_date'])
    results_with_truth = pd.merge(backtest_medians, df_merged[['date', 'value']], left_on='target_end_date', right_on='date', how='left')
    valid_results = results_with_truth.dropna(subset=['value_y'])
    if not valid_results.empty:
        mae = np.mean(np.abs(valid_results['value_x'] - valid_results['value_y']))
        print(f"Mean Absolute Error (Backtest Median): {mae:.4f}")

if __name__ == "__main__":
    main()
