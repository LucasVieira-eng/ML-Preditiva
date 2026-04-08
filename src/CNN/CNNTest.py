import os
import numpy as np
from keras import models
from CNNModel import load_cmaps_data, inverse_rul_transform


def predict_engine_rul(model_path, data_path, window_size=80, y_pred_clip_max=6.0):
    X_test, _, unit_ids_test = load_cmaps_data(data_path, window_size=window_size)
    model = models.load_model(model_path)
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = inverse_rul_transform(y_pred, y_pred_clip_max=y_pred_clip_max)

    engine_rul_pred = {}
    for unit in np.unique(unit_ids_test):
        idx = np.where(unit_ids_test == unit)[0]
        engine_rul_pred[unit] = float(y_pred[idx[-1]])

    return engine_rul_pred


def evaluate_model(model_path, data_path, true_rul_path, window_size=80, y_pred_clip_max=6.0):
    engine_rul_pred = predict_engine_rul(model_path, data_path, window_size=window_size, 
                                         y_pred_clip_max=y_pred_clip_max)
    true_rul = np.loadtxt(true_rul_path).flatten()
    pred_list = np.array([engine_rul_pred[unit] for unit in sorted(engine_rul_pred.keys())])

    mse = np.mean((pred_list - true_rul) ** 2)
    mae = np.mean(np.abs(pred_list - true_rul))
    ss_res = np.sum((pred_list - true_rul) ** 2)
    ss_tot = np.sum((true_rul - np.mean(true_rul)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r_squared': float(r_squared),
        'predictions': pred_list,
        'true_rul': true_rul,
        'engine_rul_pred': engine_rul_pred,
    }
    return metrics


if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '../../models/CNN_modelV2.keras')
    data_path = os.path.join(base_dir, '../../data/test_FD001.txt')
    true_rul_path = os.path.join(base_dir, '../../data/RUL_FD001.txt')

    metrics = evaluate_model(model_path, data_path, true_rul_path)
    print("\n=== CNN Model Evaluation ===")
    print(f"MSE: {metrics['mse']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"R-squared: {metrics['r_squared']:.4f}")
