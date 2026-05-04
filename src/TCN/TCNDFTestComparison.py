import os
import numpy as np
import matplotlib.pyplot as plt
from keras import models
from TCNModel import load_cmaps_data, inverse_rul_transform, ResidualTCNBlock  # noqa: F401


def plot_comparison(y_true, y_pred, plot_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="RUL Real",     color="blue",  linewidth=1.5)
    plt.plot(y_pred, label="RUL Previsto", color="green", linewidth=1.5, linestyle="--")
    plt.xlabel("Motor (índice)")
    plt.ylabel("RUL (ciclos)")
    plt.title("TCN — RUL Real vs Previsto")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico salvo em {plot_path}")


def plot_error(y_true, y_pred, plot_path):
    errors = y_pred - y_true
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(errors)), errors,
            color=["red" if e > 0 else "blue" for e in errors], alpha=0.7)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("Motor (índice)")
    plt.ylabel("Erro (ciclos)")
    plt.title("TCN — Distribuição de Erro")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de erro salvo em {plot_path}")


def plot_receptive_field(num_blocks, kernel_size, plot_path):
    """
    Visualiza como as convoluções dilatadas expandem o campo receptivo
    a cada bloco — exclusivo do TCN, ótimo para apresentação.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    y_positions = list(range(num_blocks + 1))

    for block in range(num_blocks):
        dilation  = 2 ** block
        receptive = 1 + (kernel_size - 1) * dilation
        color     = colors[block % len(colors)]

        for pos in range(30):
            alpha = 0.8 if pos % dilation == 0 else 0.15
            ax.barh(block + 1, 1, left=pos, height=0.6,
                    color=color, alpha=alpha, edgecolor="white", linewidth=0.3)

        ax.text(31, block + 1,
                f"Bloco {block+1} | dilation={dilation} | campo={receptive}",
                va="center", fontsize=9)

    ax.set_yticks(range(1, num_blocks + 1))
    ax.set_yticklabels([f"Bloco {i+1}" for i in range(num_blocks)])
    ax.set_xlabel("Timestep (ciclo)")
    ax.set_title("TCN — Expansão do Campo Receptivo por Dilatação")
    ax.set_xlim(0, 45)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de campo receptivo salvo em {plot_path}")


if __name__ == "__main__":
    test_path  = os.path.join(os.path.dirname(__file__), "../../data/test_FD001.txt")
    rul_path   = os.path.join(os.path.dirname(__file__), "../../data/RUL_FD001.txt")
    model_path = os.path.join(os.path.dirname(__file__), "../../models/TCN_model.keras")
    plot_comp  = os.path.join(os.path.dirname(__file__), "../../plots/TCN_comparison_plot.png")
    plot_err   = os.path.join(os.path.dirname(__file__), "../../plots/TCN_error.png")
    plot_rf    = os.path.join(os.path.dirname(__file__), "../../plots/TCN_receptive_field.png")

    model = models.load_model(model_path)

    X_test, _, unit_ids = load_cmaps_data(test_path, min_sequence_length=1)

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

    plot_comparison(Y_true, y_pred, plot_comp)
    plot_error(Y_true, y_pred, plot_err)
    plot_receptive_field(num_blocks=4, kernel_size=3, plot_path=plot_rf)