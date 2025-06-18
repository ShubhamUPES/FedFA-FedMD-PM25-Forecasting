import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(filepath, sequence_length=24):
    df = pd.read_csv(filepath, parse_dates=['Datetime'])
    df = df.sort_values(by='Datetime')
    df = df.dropna(subset=['PM2.5'])

    scaler = MinMaxScaler()
    df['PM2.5'] = scaler.fit_transform(df[['PM2.5']])

    X, y = [], []
    for i in range(len(df) - sequence_length):
        seq_x = df['PM2.5'].values[i:i+sequence_length]
        seq_y = df['PM2.5'].values[i+sequence_length]
        X.append(seq_x)
        y.append(seq_y)

    return np.array(X), np.array(y), scaler