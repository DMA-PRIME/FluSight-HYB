import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

class FluDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def load_and_preprocess_data(prisma_path, target_path, input_window=10, output_window=4):
    df_prisma = pd.read_csv(prisma_path)
    df_target = pd.read_csv(target_path)

    df_prisma['Week'] = pd.to_datetime(df_prisma['Week'])
    df_target['date'] = pd.to_datetime(df_target['date'])

    if 'SC' in df_prisma['State'].unique():
        df_prisma = df_prisma[df_prisma['State'] == 'SC'].copy()
    if 'South Carolina' in df_target['location_name'].unique():
        df_target = df_target[df_target['location_name'] == 'South Carolina'].copy()

    df_prisma = df_prisma.rename(columns={'Week': 'date'})
    merged_df = pd.merge(df_prisma, df_target, on='date', how='inner')
    merged_df = merged_df.sort_values('date').reset_index(drop=True)

    # --- Feature Orchestration for Lag Reduction ---
    # Velocity and Acceleration
    merged_df['Tests_delta'] = merged_df['Weekly_Tests'].diff()
    merged_df['PosTests_delta'] = merged_df['Weekly_Positive_Tests'].diff()
    merged_df['PosTests_accel'] = merged_df['PosTests_delta'].diff()
    
    # Seasonality
    week_of_year = merged_df['date'].dt.isocalendar().week
    merged_df['sin_week'] = np.sin(2 * np.pi * week_of_year / 52.18)
    merged_df['cos_week'] = np.cos(2 * np.pi * week_of_year / 52.18)

    merged_df = merged_df.dropna().reset_index(drop=True)

    # Log-transform target
    merged_df['value_log'] = np.log1p(merged_df['value'])
    merged_df['target_delta'] = merged_df['value_log'].diff().fillna(0)
    
    # IMPORTANT: We REMOVE absolute value_log as a feature to prevent persistence lag.
    # The model must infer the current level from the input history, 
    # but primarily focus on the DELTAS to predict future movement.
    feature_cols = [
        'Weekly_Inpatient_Hospitalizations', 
        'Weekly_Tests', 'Tests_delta',
        'Weekly_Positive_Tests', 'PosTests_delta', 'PosTests_accel',
        'Weekly_Encounters',
        'target_delta', # Predicting future delta from past deltas
        'sin_week', 'cos_week'
    ]
    
    # We keep value_log ONLY for the residual calculation anchor
    data = merged_df[feature_cols].values
    target_data = merged_df[['value_log']].values

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    data_scaled = scaler_features.fit_transform(data)
    target_scaled = scaler_target.fit_transform(target_data)
    
    X, y = [], []
    dates = []
    
    # Store the absolute level separately to use as anchor during prediction
    anchors = [] 
    
    for i in range(len(data_scaled) - input_window - output_window + 1):
        X.append(data_scaled[i : i + input_window])
        
        # Residual Target: Future absolute level - current absolute level
        current_val = target_scaled[i + input_window - 1]
        future_vals = target_scaled[i + input_window : i + input_window + output_window]
        
        deltas = future_vals - current_val
        y.append(deltas.flatten())
        
        dates.append(merged_df['date'].iloc[i + input_window])
        # The 'last seen' absolute value for this window
        anchors.append(target_scaled[i + input_window - 1])

    X = np.array(X)
    y = np.array(y)
    
    last_input_sequence = data_scaled[-input_window:]
    last_date = merged_df['date'].iloc[-1]
    last_anchor = target_scaled[-1]
    
    # Return everything needed for training and reconstruction
    return X, y, dates, scaler_target, merged_df, last_input_sequence, last_date, anchors, last_anchor

def create_dataloaders(X, y, batch_size=32, train_split=0.8):
    dataset_size = len(X)
    train_size = int(dataset_size * train_split)
    
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    train_dataset = FluDataset(X_train, y_train)
    val_dataset = FluDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
