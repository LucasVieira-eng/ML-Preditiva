import numpy as np
from tensorflow.keras import models



def load_cmaps_data(file_path, window_size=40, rul_clip=125):
    data = np.loadtxt(file_path)

    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)
    settings = data[:, 2:5]
    sensors = data[:, 5:]

    useful_idx = [1,2,3,6,7,8,10,11,12,13,14,16,19,20]
    sensors = sensors[:, useful_idx]

    features = np.hstack([settings, sensors])

    mu = np.load("data/mu.npy")
    sigma = np.load("data/sigma.npy")
    features = (features - mu) / sigma

    X, Y, unit_window_ids = [], [], []

    unique_units = np.unique(unit_ids)

    for unit in unique_units:
        idx = unit_ids == unit

        unit_feat = features[idx]
        unit_cycles = cycles[idx]

        max_cycle = unit_cycles.max()
        rul = max_cycle - unit_cycles
        rul = np.clip(rul, 0, rul_clip)

        if len(unit_feat) < window_size:
            # pad with first value (common practice)
            pad = np.repeat(unit_feat[0:1], window_size - len(unit_feat), axis=0)
            window = np.vstack([pad, unit_feat])

            X.append(window)
            Y.append(rul[-1])
            unit_window_ids.append(unit)

        else:
            for i in range(len(unit_feat) - window_size + 1):
                X.append(unit_feat[i:i+window_size])
                Y.append(rul[i+window_size-1])
                unit_window_ids.append(unit)

    return np.array(X), np.array(Y), np.array(unit_window_ids)

X_test, Y_test, unit_ids_test = load_cmaps_data("data/test_FD001.txt")
model = models.load_model("models/LSTM_model.keras")
Y_pred = model.predict(X_test).flatten()

engine_rul_pred = {}

for unit in np.unique(unit_ids_test):
    idx = np.where(unit_ids_test == unit)[0]
    last_idx = idx[-1]  # last window of that engine
    engine_rul_pred[unit] = Y_pred[last_idx]

true_rul = np.loadtxt("data/RUL_FD001.txt")

pred_list = []
for i, unit in enumerate(sorted(engine_rul_pred.keys())):
    pred_list.append(engine_rul_pred[unit])

pred_list = np.array(pred_list)

rmse = np.sqrt(np.mean((pred_list - true_rul) ** 2))


