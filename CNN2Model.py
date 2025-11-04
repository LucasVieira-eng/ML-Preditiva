import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_cmaps_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.strip().split()])
    data = np.array(data)
    
    units = np.unique(data[:, 0]).astype(int)
    X_list, Y_list = [], []
    
    for unit in units:
        unit_data = data[data[:, 0] == unit]
        cycles = unit_data[:, 1].astype(int)
        measurements = unit_data[:, 2:]
        max_cycle = cycles.max()
        X_list.append(measurements)
        Y_list.append(max_cycle - cycles)
    
    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)
    return X, Y

# Carrega e prepara dados
X_train, Y_train = load_cmaps_data('train_FD001.txt')

# Normaliza e adiciona dimensão de canal
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train = np.expand_dims(X_train, axis=-1)



# Modelo CNN
l2_strength = 1e-3
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool1D(2),

    layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.BatchNormalization(),
    layers.ReLU(),
    layers.MaxPool1D(2),

    layers.Flatten(),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Treina o modelo
history = model.fit(
    X_train, Y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
)

# Salva o modelo
model.save('CNN_model.keras')
