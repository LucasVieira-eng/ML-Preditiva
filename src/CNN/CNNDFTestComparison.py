from CNNTest import engine_rul_pred
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

true_rul = np.loadtxt(os.path.join(os.path.dirname(__file__), '../../data/RUL_FD001.txt'))

# Get predictions in the same order as true RUL
engine_rul_values = np.array([engine_rul_pred[unit] for unit in sorted(engine_rul_pred.keys())])

# Create an x-axis for the indices
x = np.arange(len(engine_rul_values))

plt.rcParams.update({'font.size': 18})
# Figure 1: Comparison plot with vertical distance lines
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, engine_rul_values, linestyle='None', marker='o', label='Previsto')
ax.plot(x, true_rul, linestyle='None', marker='x', label='Real')

# vertical distances
ax.vlines(x, engine_rul_values, true_rul, colors='gray', linestyles='dashed')

ax.legend()
ax.set_xlabel('Índice do Motor')
ax.set_ylabel('RUL')
ax.set_title('CNN: RUL Previsto vs Real')
plt.savefig(os.path.join(os.path.dirname(__file__), "../../plots/CNN_comparison_plot.png"), dpi=100, bbox_inches='tight')
print("Gráfico salvo como CNN_comparison_plot.png")
plt.close()

# Figure 2: Error plot
error = engine_rul_values - true_rul

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, error, marker='o', color='red', linestyle='-', linewidth=1.5)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_xlabel("Índice do Motor")
ax.set_ylabel("Erro de Predição (Prev - Real)")
ax.set_title("CNN: Erro de Predição RUL por Motor")
plt.savefig(os.path.join(os.path.dirname(__file__), "../../plots/CNN_error.png"), dpi=100, bbox_inches='tight')
print("Gráfico salvo como CNN_error.png")
plt.close()