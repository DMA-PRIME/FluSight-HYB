import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    # --- Dual-Stream Feature Engineering ---
    # Stream 1: Magnitude Indicators
    merged_df['positivity_rate'] = merged_df['Weekly_Positive_Tests'] / (merged_df['Weekly_Tests'] + 1e-5)
    merged_df['value_log'] = np.log1p(merged_df['value'])
    
    # Stream 2: Temporal Lead Indicators
    merged_df['pos_rate_delta'] = merged_df['positivity_rate'].diff()
    merged_df['pos_rate_accel'] = merged_df['pos_rate_delta'].diff()
    merged_df['target_delta'] = merged_df['value_log'].diff()
    
    # Seasonality
    week_of_year = merged_df['date'].dt.isocalendar().week
    merged_df['sin_week'] = np.sin(2 * np.pi * week_of_year / 52.18)
    merged_df['cos_week'] = np.cos(2 * np.pi * week_of_year / 52.18)

    merged_df = merged_df.dropna().reset_index(drop=True)

    feature_cols = [
        'value_log', 'positivity_rate', # Level
        'target_delta', 'pos_rate_delta', 'pos_rate_accel', # Change
        'sin_week', 'cos_week', # Time
        'Weekly_Encounters', 'Weekly_Inpatient_Hospitalizations' # Activity
    ]
    
    # Scaling: Use StandardScaler for features to normalize variance
    scaler_features = StandardScaler()
    scaler_target = MinMaxScaler() # Keep MinMaxScaler for the residual target range
    
    data_scaled = scaler_features.fit_transform(merged_df[feature_cols])
    target_scaled = scaler_target.fit_transform(merged_df[['value_log']])
    
    X, y, dates, anchors = [], [], [], []
    for i in range(len(data_scaled) - input_window - output_window + 1):
        X.append(data_scaled[i : i + input_window])
        # We predict ABSOLUTE levels now to restore magnitude accuracy, 
        # but the model architecture will handle the residual logic internally.
        y.append(target_scaled[i + input_window : i + input_window + output_window].flatten())
        dates.append(merged_df['date'].iloc[i + input_window])
        # Anchor is the last seen absolute log-value (unscaled by target_scaler for calculation)
        anchors.append(merged_df['value_log'].iloc[i + input_window - 1])

    return np.array(X), np.array(y), dates, scaler_target, merged_df, data_scaled[-input_window:], merged_df['date'].iloc[-1], anchors, merged_df['value_log'].iloc[-1]

def create_dataloaders(X, y, batch_size=32, train_split=0.8):
    dataset_size = len(X)
    train_size = int(dataset_size * train_split)
    return DataLoader(FluDataset(X[:train_size], y[:train_size]), batch_size=batch_size, shuffle=True), \
           DataLoader(FluDataset(X[train_size:], y[train_size:]), batch_size=batch_size, shuffle=False)
