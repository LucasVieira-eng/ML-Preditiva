import CNNModel
import numpy as np
import tensorflow as tf
from keras import models
import os
    
X_test, Y_test, unit_ids_test = CNNModel.load_cmaps_data(os.path.join(os.path.dirname(__file__), '../../data/test_FD001.txt'))

model = models.load_model(os.path.join(os.path.dirname(__file__), '../../models/CNN_model.keras'))

# Make predictions on all test windows
Y_pred = model.predict(X_test, verbose=0).flatten()

# Get RUL prediction per engine (last window of each engine)
engine_rul_pred = {}

for unit in np.unique(unit_ids_test):
    idx = np.where(unit_ids_test == unit)[0]
    last_idx = idx[-1]  # last window of that engine
    engine_rul_pred[unit] = Y_pred[last_idx]

# Load true RUL values for evaluation
true_rul = np.loadtxt(os.path.join(os.path.dirname(__file__), '../../data/RUL_FD001.txt')).flatten()

# Get predictions in the same order as true RUL
pred_list = []
for i, unit in enumerate(sorted(engine_rul_pred.keys())):
    pred_list.append(engine_rul_pred[unit])

pred_list = np.array(pred_list)

# Calculate metrics
mse = np.mean((pred_list - true_rul) ** 2)
mae = np.mean(np.abs(pred_list - true_rul))
ss_res = np.sum((pred_list - true_rul) ** 2)
ss_tot = np.sum((true_rul - np.mean(true_rul)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"\n=== CNN Model Evaluation ===")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r_squared:.4f}")
