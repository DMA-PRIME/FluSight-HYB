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

    # --- Refined Feature Extraction (Lag Reduction) ---
    # 1. Positivity Rate (Cleaner signal)
    merged_df['positivity_rate'] = merged_df['Weekly_Positive_Tests'] / (merged_df['Weekly_Tests'] + 1e-5)
    
    # 2. Momentum (Moving Averages of lead signals)
    merged_df['pos_rate_ma3'] = merged_df['positivity_rate'].rolling(window=3).mean()
    merged_df['pos_rate_ma5'] = merged_df['positivity_rate'].rolling(window=5).mean()
    
    # 3. Explicit Long-Lead Lags (4 and 6 weeks)
    # These help the 3-week and 4-week ahead forecasts align with earlier signals
    merged_df['pos_rate_lag4'] = merged_df['positivity_rate'].shift(4)
    merged_df['pos_rate_lag6'] = merged_df['positivity_rate'].shift(6)
    
    # 4. Acceleration of Positivity
    merged_df['pos_rate_delta'] = merged_df['positivity_rate'].diff()
    merged_df['pos_rate_accel'] = merged_df['pos_rate_delta'].diff()
    
    # 5. Target Deltas and Seasonality
    merged_df['value_log'] = np.log1p(merged_df['value'])
    merged_df['target_delta'] = merged_df['value_log'].diff().fillna(0)
    
    week_of_year = merged_df['date'].dt.isocalendar().week
    merged_df['sin_week'] = np.sin(2 * np.pi * week_of_year / 52.18)
    merged_df['cos_week'] = np.cos(2 * np.pi * week_of_year / 52.18)

    merged_df = merged_df.dropna().reset_index(drop=True)

    feature_cols = [
        'Weekly_Inpatient_Hospitalizations', 
        'positivity_rate', 'pos_rate_ma3', 'pos_rate_ma5',
        'pos_rate_lag4', 'pos_rate_lag6',
        'pos_rate_delta', 'pos_rate_accel',
        'target_delta',
        'sin_week', 'cos_week'
    ]
    
    data = merged_df[feature_cols].values
    target_data = merged_df[['value_log']].values

    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    data_scaled = scaler_features.fit_transform(data)
    target_scaled = scaler_target.fit_transform(target_data)
    
    X, y, dates, anchors = [], [], [], []
    
    for i in range(len(data_scaled) - input_window - output_window + 1):
        X.append(data_scaled[i : i + input_window])
        current_val = target_scaled[i + input_window - 1]
        future_vals = target_scaled[i + input_window : i + input_window + output_window]
        y.append((future_vals - current_val).flatten())
        dates.append(merged_df['date'].iloc[i + input_window])
        anchors.append(target_scaled[i + input_window - 1])

    return np.array(X), np.array(y), dates, scaler_target, merged_df, data_scaled[-input_window:], merged_df['date'].iloc[-1], anchors, target_scaled[-1]

def create_dataloaders(X, y, batch_size=32, train_split=0.8):
    dataset_size = len(X)
    train_size = int(dataset_size * train_split)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    return DataLoader(FluDataset(X_train, y_train), batch_size=batch_size, shuffle=True), \
           DataLoader(FluDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
