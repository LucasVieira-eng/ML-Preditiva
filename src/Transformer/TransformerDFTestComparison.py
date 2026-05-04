import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models
from TransformerModel import (
    load_cmaps_data, inverse_rul_transform,
    TransformerBlock  # noqa: F401
)


def plot_comparison(y_true, y_pred, plot_path):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="RUL Real",     color="blue", linewidth=1.5)
    plt.plot(y_pred, label="RUL Previsto", color="red",  linewidth=1.5, linestyle="--")
    plt.xlabel("Motor (índice)")
    plt.ylabel("RUL (ciclos)")
    plt.title("Transformer — RUL Real vs Previsto")
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
    plt.title("Transformer — Distribuição de Erro")
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico de erro salvo em {plot_path}")


def plot_attention(attention_weights, sample_idx=0, plot_path=None):
    attn     = attention_weights[sample_idx].numpy()  # (heads, seq, seq)
    avg_attn = attn.mean(axis=0)                      # (seq, seq)

    plt.figure(figsize=(8, 6))
    plt.imshow(avg_attn, cmap="viridis", aspect="auto")
    plt.colorbar(label="Peso de Atenção")
    plt.xlabel("Timestep (Key)")
    plt.ylabel("Timestep (Query)")
    plt.title("Mapa de Atenção — Transformer (média das heads)")
    plt.tight_layout()
    if plot_path:
        plt.savefig(plot_path)
        print(f"Mapa de atenção salvo em {plot_path}")
    plt.close()


if __name__ == "__main__":
    test_path  = os.path.join(os.path.dirname(__file__), "../../data/test_FD001.txt")
    rul_path   = os.path.join(os.path.dirname(__file__), "../../data/RUL_FD001.txt")
    model_path = os.path.join(os.path.dirname(__file__), "../../models/Transformer_model.keras")
    plot_comp  = os.path.join(os.path.dirname(__file__), "../../plots/Transformer_comparison_plot.png")
    plot_err   = os.path.join(os.path.dirname(__file__), "../../plots/Transformer_error.png")
    plot_attn  = os.path.join(os.path.dirname(__file__), "../../plots/Transformer_attention.png")

    model = models.load_model(model_path)

    # min_sequence_length=1 garante todos os 100 motores
    X_test, _, unit_ids = load_cmaps_data(test_path, min_sequence_length=1)

    # Última janela de cada motor
    last_indices = []
    for unit in np.unique(unit_ids):
        idx = np.where(unit_ids == unit)[0]
        last_indices.append(idx[-1])
    last_indices = np.array(last_indices)
    X_test_last  = X_test[last_indices]

    # RUL verdadeiro
    Y_true = np.clip(np.loadtxt(rul_path), 0, 200)

    print(f"Motores no teste: {len(X_test_last)} | RUL esperado: {len(Y_true)}")

    # Predições
    y_pred_log = model.predict(X_test_last).flatten()
    y_pred     = inverse_rul_transform(y_pred_log)

    mae  = np.mean(np.abs(y_pred - Y_true))
    rmse = np.sqrt(np.mean((y_pred - Y_true) ** 2))
    print(f"MAE  (ciclos): {mae:.2f}")
    print(f"RMSE (ciclos): {rmse:.2f}")

    plot_comparison(Y_true, y_pred, plot_comp)
    plot_error(Y_true, y_pred, plot_err)

    # Mapa de atenção
    sample = tf.expand_dims(X_test_last[0], axis=0)
    _ = model(sample, training=False)

    for layer in model.layers:
        if isinstance(layer, TransformerBlock):
            attn_weights = layer.last_attn_weights
            break

    plot_attention(attn_weights, sample_idx=0, plot_path=plot_attn)