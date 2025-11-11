from CNNTest import engine_rul
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

Y_test = []
with open('RUL_FD001.txt', 'r') as f:
    for line in f:
        if line.strip():
            Y_test.append([float(x) for x in line.strip().split()])
Y_test = np.array(Y_test)

engine_rul_values = np.array(list(engine_rul.values()))

# Create an x-axis for the indices
x = np.arange(len(engine_rul_values))
'''
# Plot both arrays on the same graph
plt.plot(x, engine_rul_values, label='Array 1', color='b', marker='o')  # Blue line with circle markers
plt.plot(x, Y_test, label='Array 2', color='r', marker='x')  # Red line with x markers

# Adding labels and title
plt.xlabel('Comparison')
plt.savefig("comparison_plot.png")
print("Plot saved as comparison_plot.png")
'''


v_T = engine_rul_values.reshape(-1, 1)  # vira vetor coluna
plt.plot(x,(Y_test - v_T), label='Array 1', color='b')
plt.xlabel('Error')
plt.savefig("error_plot.png")
print("Plot saved as error_plot.png")