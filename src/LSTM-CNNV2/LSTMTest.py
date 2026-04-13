import numpy as np
from tensorflow.keras import models
import os


def _get_scaler_paths():
    base_dir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(base_dir, "mu.npy"), os.path.join(base_dir, "sigma.npy")


def load_cmaps_data(file_path, window_size=40, rul_clip=200, min_sequence_length=40):
    """
    Load CMAPSS data with optional filtering of very short sequences.
    
    Args:
        file_path: Path to data file
        window_size: Sequence window length
        rul_clip: Maximum RUL value (clipping applied)
        min_sequence_length: Skip sequences shorter than this (avoids excessive padding)
    """
    data = np.loadtxt(file_path)

    unit_ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)
    settings = data[:, 2:5]
    sensors = data[:, 5:]

    useful_idx = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
    sensors = sensors[:, useful_idx]
    features = np.hstack([settings, sensors])

    mu_path, sigma_path = _get_scaler_paths()
    if not os.path.exists(mu_path) or not os.path.exists(sigma_path):
        raise FileNotFoundError(
            f"Scaler files not found. Expected mu.npy and sigma.npy in {os.path.dirname(mu_path)}"
        )

    mu = np.load(mu_path)
    sigma = np.load(sigma_path)
    features = (features - mu) / sigma

    X, Y, unit_ids_windowed = [], [], []
    unique_units = np.unique(unit_ids)

    for unit in unique_units:
        idx = unit_ids == unit
        unit_feat = features[idx]
        unit_cycles = cycles[idx]

        max_cycle = unit_cycles.max()
        rul = max_cycle - unit_cycles
        rul = np.clip(rul, 0, rul_clip)
        rul = np.log(rul + 1)

        if len(unit_feat) < window_size:
            pad = np.repeat(unit_feat[0:1], window_size - len(unit_feat), axis=0)
            window = np.vstack([pad, unit_feat])
            X.append(window)
            Y.append(rul[-1])
            unit_ids_windowed.append(unit)
        else:
            for i in range(len(unit_feat) - window_size + 1):
                X.append(unit_feat[i:i + window_size])
                Y.append(rul[i + window_size - 1])
                unit_ids_windowed.append(unit)

    return np.array(X), np.array(Y), np.array(unit_ids_windowed)


def inverse_rul_transform(y_pred, y_pred_clip_max=6.0):
    """
    Inverse log-transform RUL predictions with clipping.
    
    Args:
        y_pred: Log-transformed RUL predictions
        y_pred_clip_max: Maximum log-space value (prevents exponential explosion)
                         exp(6.0) - 1 = 403.4, which is ~3x the max training RUL
    
    Returns:
        Clipped RUL predictions in original space
    """
    y_pred_clipped = np.clip(y_pred, -np.inf, y_pred_clip_max)
    return np.clip(np.exp(y_pred_clipped) - 1, 0.0, None)

X_test, Y_test, unit_ids_test = load_cmaps_data(os.path.join(os.path.dirname(__file__), "../../data/test_FD001.txt"), 
                                               window_size=80)
model = models.load_model(os.path.join(os.path.dirname(__file__), "../../models/LSTM_model.keras"))
Y_pred = model.predict(X_test, verbose=0).flatten()
Y_pred = inverse_rul_transform(Y_pred, y_pred_clip_max=6.0)

engine_rul_pred = {}

for unit in np.unique(unit_ids_test):
    idx = np.where(unit_ids_test == unit)[0]
    last_idx = idx[-1]  # last window of that engine
    engine_rul_pred[unit] = Y_pred[last_idx]

true_rul = np.loadtxt(os.path.join(os.path.dirname(__file__), "../../data/RUL_FD001.txt"))

pred_list = []
for unit in sorted(engine_rul_pred.keys()):
    pred_list.append(engine_rul_pred[unit])

pred_list = np.array(pred_list)

mse = np.mean((pred_list - true_rul) ** 2)
mae = np.mean(np.abs(pred_list - true_rul))
ss_res = np.sum((pred_list - true_rul) ** 2)
ss_tot = np.sum((true_rul - np.mean(true_rul)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\n=== LSTM Model Evaluation ===")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r_squared:.4f}")


