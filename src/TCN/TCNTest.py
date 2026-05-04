import os
import numpy as np
from keras import models
from TCNModel import load_cmaps_data, inverse_rul_transform, ResidualTCNBlock  # noqa: F401


if __name__ == "__main__":
    test_path  = os.path.join(os.path.dirname(__file__), "../../data/test_FD001.txt")
    rul_path   = os.path.join(os.path.dirname(__file__), "../../data/RUL_FD001.txt")
    model_path = os.path.join(os.path.dirname(__file__), "../../models/TCN_model.keras")

    model = models.load_model(model_path)

    X_test, _, unit_ids = load_cmaps_data(test_path, min_sequence_length=1)

    # Última janela de cada motor
    last_indices = []
    for unit in np.unique(unit_ids):
        idx = np.where(unit_ids == unit)[0]
        last_indices.append(idx[-1])
    last_indices = np.array(last_indices)
    X_test_last  = X_test[last_indices]

    Y_true = np.clip(np.loadtxt(rul_path), 0, 200)

    print(f"Motores no teste: {len(X_test_last)} | RUL esperado: {len(Y_true)}")

    y_pred_log = model.predict(X_test_last).flatten()
    y_pred     = inverse_rul_transform(y_pred_log)

    mae  = np.mean(np.abs(y_pred - Y_true))
    rmse = np.sqrt(np.mean((y_pred - Y_true) ** 2))

    print(f"MAE  (ciclos): {mae:.2f}")
    print(f"RMSE (ciclos): {rmse:.2f}")