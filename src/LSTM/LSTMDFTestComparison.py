from LSTMTest import engine_rul_pred
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

true_rul = np.loadtxt("data/RUL_FD001.txt")

engine_rul_values = np.array(list(engine_rul_pred.values()))

# Create an x-axis for the indices
x = np.arange(len(engine_rul_values))
'''
# Plot both arrays on the same graph
plt.plot(x, engine_rul_values, label='Array 1', color='b', marker='o')  # Blue line with circle markers
plt.plot(x, true_rul, label='Array 2', color='r', marker='x')  # Red line with x markers

# Adding labels and title
plt.xlabel('Comparison')
plt.savefig("plots/LSTM_comparison_plot.png")
print("Plot saved as LSTM_comparison_plot.png")
'''

# Plot error between predictions and true RUL


error = engine_rul_values - true_rul

plt.figure()
plt.plot(error)
plt.xlabel("Engine Index")
plt.ylabel("Prediction Error (Pred - True)")
plt.title("RUL Prediction Error per Engine")
plt.savefig("plots/LSTM_error.png")
print("Plot saved as LSTM_error.png")