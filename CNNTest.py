import CNN2Model
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
X_test, Y_test = CNN2Model.load_cmaps_data('test_FD001.txt')


def predict_engine_rul(model, X, unit_ids, cycles):
    """
    Predict RUL per engine using a trained model and adjust for offsets.

    Parameters:
        model : keras.Model
            Trained model that predicts RUL per cycle.
        X : np.ndarray
            Feature array, shape (num_cycles, num_features) or (num_cycles, num_features, 1) for Conv1D.
        unit_ids : np.ndarray
            Array of engine/unit IDs, same length as X.
        cycles : np.ndarray
            Array of cycle numbers corresponding to each row in X.

    Returns:
        engine_rul : dict
            Dictionary mapping unit_id -> predicted RUL for that engine (mean of predictions adjusted).
        cycle_rul : dict
            Dictionary mapping (unit_id, cycle) -> predicted RUL at that cycle.
    """
    # Ensure Conv1D input shape
    if len(X.shape) == 2:
        X_input = np.expand_dims(X, axis=-1)
    else:
        X_input = X

    engine_rul = {}
    cycle_rul = {}

    units = np.unique(unit_ids)
    for unit in units:
        idx = np.where(unit_ids == unit)[0]
        X_unit = X_input[idx]
        cycles_unit = cycles[idx]

        # Predict per cycle
        Y_pred = model.predict(X_unit, verbose=0).flatten()

        # Adjust each cycle prediction by the number of cycles since start
        Y_adjusted = Y_pred - (cycles_unit - cycles_unit[0])

        # Store per-cycle adjusted RUL
        for c, y in zip(cycles_unit, Y_adjusted):
            cycle_rul[(unit, c)] = y

        # Engine-level RUL: mean of all adjusted predictions
        engine_rul[unit] = Y_adjusted.mean()

    return engine_rul, cycle_rul

model = models.load_model('CNN_model.keras')
unit_ids = X_test[:, 0].astype(int)
cycles = X_test[:, 1].astype(int)

engine_rul, cycle_rul = predict_engine_rul(model, X_test, unit_ids, cycles)

print("Engine RUL keys:", engine_rul.keys())
print(unit_ids)
print(X_test.shape)
print(np.unique(unit_ids))