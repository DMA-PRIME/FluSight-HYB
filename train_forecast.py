import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
import os

from data_loader import load_and_preprocess_data, create_dataloaders
from model import FluForecaster

# Configuration
PRISMA_PATH = 'dataset/Prisma_Health_Weekly_Influenza_State_dx_cond_lab_Severity.csv'
TARGET_PATH = 'dataset/target-ed-visits-prop.csv'
INPUT_WINDOW = 10
OUTPUT_WINDOW = 4
QUANTILES = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.50, 
             0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    model.to(DEVICE)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(X_batch)
            
            # y_batch is (Batch, Output_Steps) -> needs to be compared against preds (Batch, Output_Steps, Num_Quantiles)
            # The custom loss handles the shapes.
            loss = 0
            # Target needs to be expanded or handled in loss
            target_expanded = y_batch.unsqueeze(2).expand(-1, -1, len(QUANTILES))
            
            # Simplified manual loss loop in model.py handled this, but let's correct the usage here if needed.
            # model.py's QuantileLoss expects (preds, target) where target is (Batch, Output_Steps)
            loss = criterion(preds, y_batch)
            
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
                loss = criterion(preds, y_batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            
    print("Training complete. Best Val Loss:", best_val_loss)
    model.load_state_dict(torch.load('best_model.pth'))
    return model

def quantile_loss(preds, target, quantiles):
    # preds: (batch, steps, quantiles)
    # target: (batch, steps)
    loss = 0
    target = target.unsqueeze(2)
    for i, q in enumerate(quantiles):
        errors = target - preds[:, :, i:i+1]
        loss += torch.max((q - 1) * errors, q * errors).mean()
    return loss

def predict_and_postprocess(model, X_input, scaler_target, quantiles):
    """
    X_input: (1, input_window, features) numpy array
    Returns: (output_window, num_quantiles) dataframe with columns as quantiles
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_input).to(DEVICE)
        # Add batch dim if missing
        if len(X_tensor.shape) == 2:
            X_tensor = X_tensor.unsqueeze(0)
            
        preds = model(X_tensor) # (1, output_window, num_quantiles)
        preds = preds.cpu().numpy().squeeze(0) # (output_window, num_quantiles)
        
    # Inverse Transform
    # Scaler expects (n_samples, n_features). Here n_features=1 (target).
    # We have (output_window, num_quantiles). We need to inverse transform each quantile column.
    # Since it's a linear transformation, we can flatten, transform, and reshape, 
    # OR just transform each column.
    
    preds_original_scale = np.zeros_like(preds)
    for i in range(len(quantiles)):
        col_preds = preds[:, i].reshape(-1, 1)
        preds_original_scale[:, i] = scaler_target.inverse_transform(col_preds).flatten()
        
    # Constraints: [0, 1]
    preds_original_scale = np.clip(preds_original_scale, 0, 1)
    
    # Monotonicity: Sort along the quantile axis (axis 1)
    preds_sorted = np.sort(preds_original_scale, axis=1)
    
    return preds_sorted

def generate_forecast_for_date(target_date, df_merged, model, scaler_target, input_window=10):
    """
    Requirement 6: Reusable function.
    Finds the 10-week window ending *before* target_date to predict target_date and following 3 weeks.
    Or, if target_date is the *start* of the forecast, we look back 10 weeks from there.
    Let's assume target_date is the first date of the *forecast*.
    """
    target_date = pd.to_datetime(target_date)
    
    # Find index of target_date
    if target_date not in df_merged['date'].values:
        # If date is in the future beyond our dataset, we might need the *latest* available data
        # For this specific project, let's assume we are forecasting from the end of the data 
        # OR backtesting on existing data.
        
        # If target_date is later than last data date + 1 week, we can't do it with this logic 
        # unless we have the inputs. 
        # Let's assume standard backtesting where we have the data up to the forecast point.
        return None
        
    idx = df_merged[df_merged['date'] == target_date].index[0]
    
    if idx < input_window:
        return None # Not enough history
        
    # Extract input window (10 weeks prior)
    # Rows: idx-input_window to idx-1
    # But wait, we need the *features* for these weeks.
    # We need to re-scale this slice exactly as we did globally.
    # To ensure consistency, we should use the globally scaled data (passed in or re-derived).
    # For simplicity here, let's assume we pass the raw df and scaler, but ideally we pass scaled data.
    pass

    # Actually, simpler approach for this script:
    # The 'generate_forecast_for_date' needs the *scaled* input.
    # We'll handle this in the main workflow.

def main():
    print("Loading data...")
    X, y, dates, scaler_target, df_merged, last_input, last_date = load_and_preprocess_data(PRISMA_PATH, TARGET_PATH, INPUT_WINDOW, OUTPUT_WINDOW)
    
    # Split
    dataset_size = len(X)
    train_size = int(dataset_size * 0.8)
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_loader, val_loader = create_dataloaders(X, y, BATCH_SIZE)
    
    print(f"Data loaded. Train size: {len(X_train)}, Val size: {len(X_val)}")
    
    model = FluForecaster(
        input_size=X.shape[2], 
        hidden_size=HIDDEN_SIZE, 
        output_steps=OUTPUT_WINDOW, 
        num_quantiles=len(QUANTILES), 
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Custom loss function wrapper
    def criterion(preds, target):
        return quantile_loss(preds, target, QUANTILES)
    
    print("Starting training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, epochs=EPOCHS)
    
    # --- Generation & Backtesting ---
    print("Generating historical backtest forecasts...")
    
    all_forecast_rows = []
    
    for i in range(len(X)):
        x_in = X[i]
        forecast_start_date = dates[i] # This is the target_end_date of the first predicted week
        
        # Calculate reference_date (Saturday before the first forecast start date)
        # Assuming forecast_start_date is a Saturday, the reference date for a 1-4 week ahead 
        # forecast is often the Saturday of the week the forecast is made.
        # However, to match the example: horizon = (target_end_date - reference_date) / 7
        # Let's define reference_date as the date 1 week before the first prediction.
        reference_date = forecast_start_date - timedelta(weeks=1)
        
        preds = predict_and_postprocess(model, x_in, scaler_target, QUANTILES)
        
        for step in range(OUTPUT_WINDOW):
            target_end_date = forecast_start_date + timedelta(weeks=step)
            horizon = (target_end_date - reference_date).days // 7
            
            for q_idx, q_val in enumerate(QUANTILES):
                row = {
                    'reference_date': reference_date.strftime('%Y-%m-%d'),
                    'target': 'wk inc flu prop ed visits',
                    'horizon': horizon,
                    'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                    'location': 'South Carolina',
                    'output_type': 'quantile',
                    'output_type_id': q_val,
                    'value': preds[step, q_idx],
                    'type': 'backtest' # Internal use for metrics
                }
                all_forecast_rows.append(row)
            
    # --- Future Forecast ---
    print("Generating future 4-week forecast...")
    future_preds = predict_and_postprocess(model, last_input, scaler_target, QUANTILES)
    
    future_start_date = last_date + timedelta(weeks=1)
    future_reference_date = last_date # Using the last known date as reference
    
    for step in range(OUTPUT_WINDOW):
        target_end_date = future_start_date + timedelta(weeks=step)
        horizon = (target_end_date - future_reference_date).days // 7
        
        for q_idx, q_val in enumerate(QUANTILES):
            row = {
                'reference_date': future_reference_date.strftime('%Y-%m-%d'),
                'target': 'wk inc flu prop ed visits',
                'horizon': horizon,
                'target_end_date': target_end_date.strftime('%Y-%m-%d'),
                'location': 'South Carolina',
                'output_type': 'quantile',
                'output_type_id': q_val,
                'value': future_preds[step, q_idx],
                'type': 'future'
            }
            all_forecast_rows.append(row)
            
    results_df = pd.DataFrame(all_forecast_rows)
    
    # Save
    output_path = 'forecast_results.csv'
    # Drop the 'type' column before saving to match exact requested format
    results_df.drop(columns=['type']).to_csv(output_path, index=False)
    print(f"Consolidated results saved to {output_path}")
    
    # Simple validation metrics for backtest portion
    # Need to handle the long format for MAE calculation
    backtest_medians = results_df[(results_df['type']=='backtest') & (results_df['output_type_id'] == 0.5)].copy()
    backtest_medians['target_end_date'] = pd.to_datetime(backtest_medians['target_end_date'])
    
    results_with_truth = pd.merge(backtest_medians, 
                                  df_merged[['date', 'value']], 
                                  left_on='target_end_date', right_on='date', how='left')
    
    valid_results = results_with_truth.dropna(subset=['value_y'])
    if not valid_results.empty:
        mae = np.mean(np.abs(valid_results['value_x'] - valid_results['value_y']))
        print(f"Mean Absolute Error (Backtest Median): {mae:.4f}")

if __name__ == "__main__":
    main()
