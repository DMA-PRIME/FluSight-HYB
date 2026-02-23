import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
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
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def balanced_quantile_loss(preds, target, quantiles):
    loss = 0
    target_unsq = target.unsqueeze(2)
    
    # 1. Primary Magnitude Loss (Pinball)
    for i, q in enumerate(quantiles):
        errors = target_unsq - preds[:, :, i:i+1]
        # Emphasis on absolute value accuracy
        loss += torch.max((q - 1) * errors, q * errors).mean()
        
    # 2. Smooth L1 for the Median (Magnitude Stability)
    median_pred = preds[:, :, 11]
    loss += F.smooth_l1_loss(median_pred, target) * 5.0
    
    # 3. Temporal Alignment (Direction Penalty)
    if target.shape[1] > 1:
        target_diff = target[:, 1:] - target[:, :-1]
        pred_diff = median_pred[:, 1:] - median_pred[:, :-1]
        # Penalize if prediction moves in opposite direction of ground truth
        direction_penalty = F.relu(-pred_diff * target_diff).mean()
        loss += direction_penalty * 10.0
        
    return loss

def train_model(model, train_loader, val_loader, optimizer, epochs=1000):
    model.to(DEVICE)
    best_val_loss = float('inf')
    patience = 60
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = balanced_quantile_loss(preds, y_batch, QUANTILES)
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
                loss = balanced_quantile_loss(preds, y_batch, QUANTILES)
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
            
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def predict_and_postprocess(model, X_input, scaler_target, quantiles):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
        preds_scaled = model(X_tensor).cpu().numpy().squeeze(0) # (4, 23)
        
    # Inverse scaling and log transform
    preds_original = np.zeros_like(preds_scaled)
    for i in range(len(quantiles)):
        col = preds_scaled[:, i].reshape(-1, 1)
        inv_scaled = scaler_target.inverse_transform(col).flatten()
        preds_original[:, i] = np.expm1(inv_scaled)
        
    preds_original = np.clip(preds_original, 0, 1)
    return np.sort(preds_original, axis=1)

def main():
    print("Loading data (Balanced Strategy)...")
    X, y, dates, scaler_target, df_merged, last_input, last_date, anchors, last_anchor = load_and_preprocess_data(PRISMA_PATH, TARGET_PATH, INPUT_WINDOW, OUTPUT_WINDOW)
    train_loader, val_loader = create_dataloaders(X, y, BATCH_SIZE)
    
    model = HybridCNNForecaster(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, output_steps=OUTPUT_WINDOW, num_quantiles=len(QUANTILES))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    print("Training Balanced R-GLSTM Model...")
    model = train_model(model, train_loader, val_loader, optimizer, epochs=EPOCHS)
    
    all_forecast_rows = []
    for i in range(len(X)):
        preds = predict_and_postprocess(model, X[i], scaler_target, QUANTILES)
        start_date = dates[i]
        ref_date = start_date - timedelta(weeks=1)
        for step in range(OUTPUT_WINDOW):
            target_end = start_date + timedelta(weeks=step)
            for q_idx, q_val in enumerate(QUANTILES):
                all_forecast_rows.append({
                    'reference_date': ref_date.strftime('%Y-%m-%d'),
                    'target': 'wk inc flu prop ed visits',
                    'horizon': step + 1,
                    'target_end_date': target_end.strftime('%Y-%m-%d'),
                    'location': 'South Carolina',
                    'output_type': 'quantile',
                    'output_type_id': q_val,
                    'value': preds[step, q_idx]
                })
            
    # Future
    future_preds = predict_and_postprocess(model, last_input, scaler_target, QUANTILES)
    for step in range(OUTPUT_WINDOW):
        target_end = last_date + timedelta(weeks=step+1)
        for q_idx, q_val in enumerate(QUANTILES):
            all_forecast_rows.append({
                'reference_date': last_date.strftime('%Y-%m-%d'),
                'target': 'wk inc flu prop ed visits',
                'horizon': step + 1,
                'target_end_date': target_end.strftime('%Y-%m-%d'),
                'location': 'South Carolina',
                'output_type': 'quantile',
                'output_type_id': q_val,
                'value': future_preds[step, q_idx]
            })
            
    pd.DataFrame(all_forecast_rows).to_csv('forecast_results.csv', index=False)
    print("Done.")

if __name__ == "__main__":
    main()
