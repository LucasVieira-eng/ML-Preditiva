import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras import regularizers, layers, models
def load_cmaps_data(file_path):
    # Carrega os dados
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append([float(x) for x in line.strip().split()])
    data = np.array(data)
    
    # Colunas: 0 = unidade, 1 = ciclo, 2: = medições
    units = np.unique(data[:, 0]).astype(int)
    X_list = []
    Y_list = []
    
    for unit in units:
        unit_data = data[data[:, 0] == unit]
        cycles = unit_data[:, 1].astype(int)
        measurements = unit_data[:, 2:]
        max_cycle = cycles.max()
        # X: cada linha = ciclo, cada coluna = medição
        X_list.append(measurements)
        # Y: RUL para cada ciclo
        Y_list.append(max_cycle - cycles)
    
    # Concatena todos os ciclos de todas as unidades
    X = np.vstack(X_list)
    Y = np.concatenate(Y_list)
    return X, Y



X_train, Y_train = load_cmaps_data('train_FD001.txt')

print(X_train.shape)


l2_strength = 1e-3
model = models.Sequential([
layers.Input(shape=(X_train.shape[1], 1)),
layers.Conv1D(filters=64, kernel_size=3, padding='same',
kernel_regularizer=regularizers.l2(l2_strength)),
layers.BatchNormalization(),
layers.Activation('relu'),
layers.MaxPool1D(pool_size=2),

layers.Conv1D(filters=128, kernel_size=3, padding='same',
kernel_regularizer=regularizers.l2(l2_strength)),
layers.BatchNormalization(),
layers.Activation('relu'),
layers.MaxPool1D(pool_size=2),

layers.Flatten(),
layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
layers.Dense(1) # RUL output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, Y_train, epochs=10, batch_size=32)
model.save('CNN_model_lucasv.keras')