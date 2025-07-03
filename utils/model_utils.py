from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, LSTM, Conv1D, Dense, Flatten, Dropout, Add,
    LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Lambda
)
import tensorflow as tf
import numpy as np

def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def build_tcn_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=2, padding="causal", dilation_rate=1, activation="relu")(inputs)
    x1 = Conv1D(64, kernel_size=2, padding="causal", dilation_rate=2, activation="relu")(x)
    x2 = Conv1D(64, kernel_size=2, padding="causal", dilation_rate=4, activation="relu")(x1)
    res = Add()([x, x2])
    res = Dropout(0.2)(res)
    flat = Flatten()(res)
    dense = Dense(32, activation="relu")(flat)
    output = Dense(1)(dense)
    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model

def positional_encoding(length, d_model):
    pos = np.arange(length)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    angle_rads = pos * angle_rates
    pos_encoding = np.zeros((length, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.constant(pos_encoding, dtype=tf.float32)

def build_transformer_model(input_shape):
    seq_len, features = input_shape
    d_model = 16

    inputs = Input(shape=input_shape)
    x = Dense(d_model)(inputs)
    pos_enc = positional_encoding(seq_len, d_model)
    x = Lambda(lambda z: z + pos_enc)(x)
    attn_out = MultiHeadAttention(num_heads=2, key_dim=d_model, dropout=0.3)(x, x)  # Increased dropout
    x = Add()([x, attn_out])
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = LayerNormalization()(x)  # Additional normalization after FFN
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling1D()(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer="adam", loss="mse")
    return model
