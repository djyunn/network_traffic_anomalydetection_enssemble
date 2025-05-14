import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

def build_autoencoder(input_dim, encoding_dim=32):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    decoded = Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    return autoencoder

def train_autoencoder(X_normal, epochs=50, batch_size=256):
    input_dim = X_normal.shape[1]
    autoencoder = build_autoencoder(input_dim)
    autoencoder.fit(X_normal, X_normal, epochs=epochs, batch_size=batch_size, shuffle=True)
    return autoencoder

def predict_anomalies(autoencoder, X, threshold):
    reconstructions = autoencoder.predict(X)
    mse = np.mean(np.square(X - reconstructions), axis=1)
    return np.where(mse > threshold, 1, 0)