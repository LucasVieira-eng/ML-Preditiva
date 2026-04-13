import os
import numpy as np
import matplotlib.pyplot as plt
from CNNTest import evaluate_model


def plot_cnn_predictions(metrics, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    true_rul = metrics['true_rul']
    engine_rul_pred = metrics['engine_rul_pred']
    engine_rul_values = np.array([engine_rul_pred[unit] for unit in sorted(engine_rul_pred.keys())])
    x = np.arange(len(engine_rul_values))

    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, engine_rul_values, linestyle='None', marker='o', label='Previsto')
    ax.plot(x, true_rul, linestyle='None', marker='x', label='Real')
    ax.vlines(x, engine_rul_values, true_rul, colors='gray', linestyles='dashed')
    ax.legend()
    ax.set_xlabel('Índice do Motor')
    ax.set_ylabel('RUL')
    ax.set_title('CNN: RUL Previsto vs Real')

    comparison_path = os.path.join(output_dir, 'CNNV2_comparison_plot.png')
    plt.savefig(comparison_path, dpi=100, bbox_inches='tight')
    print(f"Gráfico salvo como {comparison_path}")
    plt.close()

    error = engine_rul_values - true_rul
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x, error, marker='o', color='red', linestyle='-', linewidth=1.5)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Índice do Motor')
    ax.set_ylabel('Erro de Predição (Prev - Real)')
    ax.set_title('CNN: Erro de Predição RUL por Motor')

    error_path = os.path.join(output_dir, 'CNNV2_error.png')
    plt.savefig(error_path, dpi=100, bbox_inches='tight')
    print(f"Gráfico salvo como {error_path}")
    plt.close()


if __name__ == '__main__':
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, '../../models/CNN_modelV2.keras')
    data_path = os.path.join(base_dir, '../../data/test_FD001.txt')
    true_rul_path = os.path.join(base_dir, '../../data/RUL_FD001.txt')
    output_dir = os.path.join(base_dir, '../../plots')

    metrics = evaluate_model(model_path, data_path, true_rul_path)
    plot_cnn_predictions(metrics, output_dir)
