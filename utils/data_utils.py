import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(path, sequence_length=24):
    df = pd.read_csv(path, parse_dates=['Datetime'])
    df = df.sort_values('Datetime')
    df = df.dropna(subset=['PM2.5'])

    # Feature engineering: cyclical datetime features
    df['hour_sin'] = np.sin(2 * np.pi * df['Datetime'].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['Datetime'].dt.hour / 24)
    # Optionally, add more (day of week, month, etc.)

    scaler = MinMaxScaler()
    df['PM2.5'] = scaler.fit_transform(df[['PM2.5']])

    features = ['PM2.5', 'hour_sin', 'hour_cos']
    X, y = [], []
    for i in range(len(df) - sequence_length):
        X.append(df[features].iloc[i:i+sequence_length].values)
        y.append(df['PM2.5'].iloc[i+sequence_length])
    return np.array(X), np.array(y)
