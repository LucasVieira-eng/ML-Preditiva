import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
from tensorflow.keras import regularizers
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


model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], 1)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train, Y_train, epochs=10, batch_size=32)
model.save('mlp_model_lucasv.keras')