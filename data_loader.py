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
    # Load datasets
    df_prisma = pd.read_csv(prisma_path)
    df_target = pd.read_csv(target_path)

    # Convert date columns to datetime
    df_prisma['Week'] = pd.to_datetime(df_prisma['Week'])
    df_target['date'] = pd.to_datetime(df_target['date'])

    # Filter for South Carolina
    # Check if 'SC' exists in State column
    if 'SC' in df_prisma['State'].unique():
        df_prisma = df_prisma[df_prisma['State'] == 'SC'].copy()
    
    # Check if 'South Carolina' exists in location_name
    if 'South Carolina' in df_target['location_name'].unique():
        df_target = df_target[df_target['location_name'] == 'South Carolina'].copy()

    # Rename Week to date for merging
    df_prisma = df_prisma.rename(columns={'Week': 'date'})

    # Merge datasets on date
    merged_df = pd.merge(df_prisma, df_target, on='date', how='inner')
    
    # Sort by date
    merged_df = merged_df.sort_values('date').reset_index(drop=True)

    # Features and Target
    feature_cols = [
        'Weekly_Inpatient_Hospitalizations', 
        'Weekly_Tests', 
        'Weekly_Positive_Tests', 
        'Weekly_Encounters',
        'value' # Target is also an input feature per requirements/standard time series practice
    ]
    target_col = 'value'
    
    # Check for missing values and fill/drop
    merged_df = merged_df.dropna(subset=feature_cols)

    data = merged_df[feature_cols].values
    target = merged_df[[target_col]].values

    # Scaling
    # Requirement: Scale target using MinMaxScaler. 
    # We should scale all features to help the model learn better, but definitely the target.
    # For simplicity and requirements, let's scale the target specifically for inverse transform later.
    # We'll use a separate scaler for the target to easily inverse transform.
    
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()
    
    # Scale all features (including 'value' which is in feature_cols)
    data_scaled = scaler_features.fit_transform(data)
    
    # Fit target scaler separately for inverse transform usage
    target_scaled = scaler_target.fit_transform(target)
    
    # Create sequences
    X, y = [], []
    dates = [] # Store dates corresponding to the forecast start
    
    for i in range(len(data_scaled) - input_window - output_window + 1):
        # Input: features for 'input_window' weeks
        X.append(data_scaled[i : i + input_window])
        # Output: target 'value' for 'output_window' weeks
        # Note: We are predicting 'value', so we take it from the target array (which is just 'value')
        # However, the model output should be scaled, so we use the scaled target.
        # But wait, 'data_scaled' already includes 'value' as the last column.
        # The target for the loss function should be the future values of 'value'.
        
        # We need the future 'value's. 
        # Let's use the 'target_scaled' which is just the 'value' column scaled.
        y.append(target_scaled[i + input_window : i + input_window + output_window].flatten())
        
        # Store the date of the first prediction (i + input_window)
        dates.append(merged_df['date'].iloc[i + input_window])

    X = np.array(X)
    y = np.array(y)
    
    # Extract the very last available input window for future forecasting
    # This is the last 'input_window' rows of data_scaled
    last_input_sequence = data_scaled[-input_window:]
    last_date = merged_df['date'].iloc[-1]
    
    return X, y, dates, scaler_target, merged_df, last_input_sequence, last_date

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
