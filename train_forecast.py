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

# Configuration - "Optimal Stability" Tuning
PRISMA_PATH = 'dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv'
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

TARGETS_CONFIG = [
    {
        'path': 'dataset/target-ed-visits-prop.csv',
        'name': 'wk inc flu prop ed visits'
    },
    {
        'path': 'dataset/target-hospital-admissions.csv',
        'name': 'wk inc flu hosp'
    }
]

INPUT_WINDOW = 10
OUTPUT_WINDOW = 4
QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
HIDDEN_SIZE = 96 
NUM_LAYERS = 2
DROPOUT = 0.4 
BATCH_SIZE = 16 
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-3 
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def balanced_quantile_loss(preds, target, quantiles):
    loss = 0
    target_unsq = target.unsqueeze(2)
    for i, q in enumerate(quantiles):
        errors = target_unsq - preds[:, :, i:i+1]
        loss += torch.max((q - 1) * errors, q * errors).mean()
    
    median_pred = preds[:, :, 11]
    loss += F.smooth_l1_loss(median_pred, target) * 5.0
    
    if target.shape[1] > 1:
        target_diff = target[:, 1:] - target[:, :-1]
        pred_diff = median_pred[:, 1:] - median_pred[:, :-1]
        loss += F.relu(-pred_diff * target_diff).mean() * 10.0
        
    return loss

def train_model(model, train_loader, val_loader, optimizer, scheduler, model_path, epochs=1000):
    model.to(DEVICE)
    best_val_loss = float('inf')
    patience = 100
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if model.training:
                noise = torch.randn_like(X_batch) * 0.01
                X_batch = X_batch + noise
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
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
            
    model.load_state_dict(torch.load(model_path))
    return model

def predict_and_postprocess(model, X_input, scaler_target, quantiles):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
        preds_scaled = model(X_tensor).cpu().numpy().squeeze(0)
        
    preds_original = np.zeros_like(preds_scaled)
    for i in range(len(quantiles)):
        col = preds_scaled[:, i].reshape(-1, 1)
        inv_scaled = scaler_target.inverse_transform(col).flatten()
        preds_original[:, i] = np.expm1(inv_scaled)
        
    preds_original = np.clip(preds_original, 0, None) # Allow values > 1 for admissions
    return np.sort(preds_original, axis=1)

def main():
    for config in TARGETS_CONFIG:
        all_combined_rows = []
        target_path = config['path']
        target_name = config['name']
        print(f"\nProcessing Target: {target_name} ({target_path})")
        
        X, y, dates, scaler_target, df_merged, last_input, last_date, anchors, last_anchor = load_and_preprocess_data(
            PRISMA_PATH, target_path, INPUT_WINDOW, OUTPUT_WINDOW
        )
        train_loader, val_loader = create_dataloaders(X, y, BATCH_SIZE, train_split=0.9)
        
        model = HybridCNNForecaster(input_size=X.shape[2], hidden_size=HIDDEN_SIZE, output_steps=OUTPUT_WINDOW, num_quantiles=len(QUANTILES))
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        
        model_filename = os.path.join(RESULTS_DIR, f"best_model_{target_name.replace(' ', '_')}.pth")
        model = train_model(model, train_loader, val_loader, optimizer, scheduler, model_filename, epochs=EPOCHS)
        
        # Generation
        for i in range(len(X)):
            preds = predict_and_postprocess(model, X[i], scaler_target, QUANTILES)
            start_date = dates[i]
            ref_date = start_date - timedelta(weeks=1)
            for step in range(OUTPUT_WINDOW):
                target_end = start_date + timedelta(weeks=step)
                for q_idx, q_val in enumerate(QUANTILES):
                    all_combined_rows.append({
                        'reference_date': ref_date.strftime('%Y-%m-%d'),
                        'target': target_name,
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
                all_combined_rows.append({
                    'reference_date': last_date.strftime('%Y-%m-%d'),
                    'target': target_name,
                    'horizon': step + 1,
                    'target_end_date': target_end.strftime('%Y-%m-%d'),
                    'location': 'South Carolina',
                    'output_type': 'quantile',
                    'output_type_id': q_val,
                    'value': future_preds[step, q_idx]
                })
            
        output_filename = os.path.join(RESULTS_DIR, f"forecast_results_{target_name.replace(' ', '_')}.csv")
        results_df = pd.DataFrame(all_combined_rows)
        results_df.to_csv(output_filename, index=False)
        print(f"Results for {target_name} saved to {output_filename}")
        
        # New Requirement: Generate latest-only file
        last_ref_date = results_df['reference_date'].max()
        latest_only_df = results_df[results_df['reference_date'] == last_ref_date].copy()
        
        # Clean target name for filename
        clean_target = target_name.replace(' ', '-')
        latest_filename = os.path.join(RESULTS_DIR, f"{last_ref_date}-team-model-{clean_target}.csv")
        latest_only_df.to_csv(latest_filename, index=False)
        print(f"Latest-only results for {target_name} saved to {latest_filename}")

if __name__ == "__main__":
    main()
