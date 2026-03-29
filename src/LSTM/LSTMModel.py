import tensorflow as tf
from keras import layers, models, regularizers, callbacks
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def load_cmaps_data(file_path, window_size=40, rul_clip=125):
    data = np.loadtxt(file_path)

    # Columns
    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)
    settings = data[:, 2:5]
    sensors = data[:, 5:]

    # ---- Remove useless sensors (standard CMAPSS practice)
    useful_idx = [1,2,3,6,7,8,10,11,12,13,14,16,19,20]  # 0-based from sensors
    sensors = sensors[:, useful_idx]

    # Combine settings + sensors
    features = np.hstack([settings, sensors])

    # ---- Normalize per sensor (using whole dataset here; ideally only train)
    mu = features.mean(axis=0)
    sigma = features.std(axis=0) + 1e-8
    features = (features - mu) / sigma
    np.save("data/mu.npy", mu)
    np.save("data/sigma.npy", sigma)


    # ---- Build sequences
    X, Y = [], []

    unique_units = np.unique(unit_ids)

    for unit in unique_units:
        idx = unit_ids == unit

        unit_feat = features[idx]
        unit_cycles = cycles[idx]

        max_cycle = unit_cycles.max()
        rul = max_cycle - unit_cycles

        # clip RUL
        rul = np.clip(rul, 0, rul_clip)

        # sliding window
        for i in range(len(unit_feat) - window_size + 1):
            X.append(unit_feat[i:i+window_size])
            Y.append(rul[i+window_size-1])

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

# Carrega e prepara dados
X_train, Y_train = load_cmaps_data(os.path.join(os.path.dirname(__file__), '../data/train_FD001.txt'))

if __name__ == "__main__":

    # Modelo CNN
    l2_strength = 1e-4

    model = models.Sequential([
        # (window_size, num_sensors)
        layers.Input(shape=(X_train.shape[1], X_train.shape[2])),

        # --- CNN feature extraction ---
        layers.Conv1D(64, 3, padding='same',
                    kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 3, padding='same',
                    kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),

        # --- LSTM for temporal modeling ---
        layers.LSTM(100, return_sequences=False),

        # --- Dense head ---
        layers.Dense(64, activation='relu',
                    kernel_regularizer=regularizers.l2(l2_strength)),
        layers.Dropout(0.3),

        layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Treina o modelo
    history = model.fit(
        X_train, Y_train,
        epochs=50,
        batch_size=64,
        validation_split=0.2,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )

    # Salva o modelo
    model.save(os.path.join(os.path.dirname(__file__), '../models/LSTM_model.keras'))
