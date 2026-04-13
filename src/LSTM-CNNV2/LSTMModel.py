import os
import numpy as np
import tensorflow as tf
from keras import layers, models, regularizers, callbacks


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

        if __name__ == "__main__":
            if len(unit_feat) < min_sequence_length:
                continue  # Skip this engine entirely
        
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


def compute_sample_weights(Y_train, high_rul_scale=1.0):
    """
    Compute sample weights based on RUL values.
    Higher RULs get higher weights by the formula: 1 + high_rul_scale * (RUL / max(RUL))
    
    Args:
        Y_train: Array of RUL values (log-transformed)
        high_rul_scale: Scaling factor for high RUL weight emphasis (default=1.0)
    
    Returns:
        sample_weights: Array of weights for each sample
    """
    max_rul = np.max(Y_train)
    normalized_rul = Y_train / max_rul
    sample_weights = 1.0 + high_rul_scale * normalized_rul
    return sample_weights


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


def build_lstm_model(input_shape, l2_strength=1e-4):
    """Build LSTM-CNN hybrid model for RUL prediction."""
    return models.Sequential([
        layers.Input(shape=input_shape),
        
        # CNN feature extraction
        layers.Conv1D(64, 3, padding='same', kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(128, 3, padding='same', kernel_regularizer=regularizers.l2(l2_strength)),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.MaxPooling1D(2),
        
        # LSTM for temporal modeling
        layers.LSTM(100, return_sequences=False),
        
        # Dense head
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(l2_strength)),
        layers.Dropout(0.3),
        
        layers.Dense(1)
    ])


def train_lstm_model(model, X_train, Y_train, model_path, epochs=50, batch_size=64, 
                     validation_split=0.2, high_rul_scale=1.0):
    """
    Train LSTM model with optional sample weighting for high RUL values.
    
    Args:
        model: Keras model to train
        X_train: Training features
        Y_train: Training targets (RUL values, log-transformed)
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        high_rul_scale: Scale factor for penalizing high RUL predictions (0 = uniform weights)
    """
    
    model.compile(optimizer='adam', loss='huber', metrics=['mae'])
    
    sample_weights = compute_sample_weights(Y_train, high_rul_scale=high_rul_scale) if high_rul_scale > 0 else None
    
    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        sample_weight=sample_weights,
        callbacks=[callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    model.save(model_path)
    return model

if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), '../../data/train_FD001.txt')
    model_path = os.path.join(os.path.dirname(__file__), '../../models/LSTM_model.keras')
    window_size = 80
    min_sequence_length = 40
    high_rul_scale = 3.5
    y_pred_clip_max = 6.0

    X_train, Y_train, _ = load_cmaps_data(train_path, window_size=window_size, 
                                          min_sequence_length=min_sequence_length)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    train_lstm_model(model, X_train, Y_train, model_path, high_rul_scale=high_rul_scale)
    print(f"Model saved to {model_path}")
    print(f"Training completed with:")
    print(f"  - window_size={window_size}")
    print(f"  - min_sequence_length={min_sequence_length}")
    print(f"  - high_rul_scale={high_rul_scale}")
    print(f"  - y_pred_clip_max={y_pred_clip_max} (for inverse transform)")
