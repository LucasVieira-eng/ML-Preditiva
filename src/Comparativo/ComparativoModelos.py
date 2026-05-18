"""
ComparativoModelos.py
Gera dois gráficos comparativos dos 4 modelos:
  1. Erro Absoluto por motor (MAE visual)
  2. Erro Normalizado por motor (erro / RUL_real)

Coloque este arquivo em: src/Comparativo/ComparativoModelos.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras import models as km

# ── paths ─────────────────────────────────────────────────────────────────────
BASE  = os.path.join(os.path.dirname(__file__), "../..")
DATA  = os.path.join(BASE, "data")
MDLS  = os.path.join(BASE, "models")
PLOTS = os.path.join(BASE, "plots")

sys.path.insert(0, os.path.join(BASE, "src/CNNV2"))
sys.path.insert(0, os.path.join(BASE, "src/LSTM-CNNV2"))
sys.path.insert(0, os.path.join(BASE, "src/TCN"))
sys.path.insert(0, os.path.join(BASE, "src/Transformer"))

# ── imports dos modelos (registra os decorators customizados) ─────────────────
import CNNModel
import LSTMModel
import TCNModel
from TCNModel import ResidualTCNBlock          # noqa — registra decorator
import TransformerModel
from TransformerModel import TransformerBlock  # noqa — registra decorator

# ── helpers ───────────────────────────────────────────────────────────────────
def get_last_windows(X, unit_ids):
    """Retorna índice da última janela de cada motor."""
    return np.array([
        np.where(unit_ids == u)[0][-1]
        for u in np.unique(unit_ids)
    ])

def inverse(y, clip=6.0):
    """Desfaz transformação log(RUL+1) → RUL em ciclos."""
    return np.clip(np.exp(np.clip(y, -np.inf, clip)) - 1, 0, None)

# ── RUL verdadeiro ────────────────────────────────────────────────────────────
Y_true = np.clip(np.loadtxt(os.path.join(DATA, "RUL_FD001.txt")), 0, 200)
n_motors = len(Y_true)
print(f"Motores no teste: {n_motors}")

# ── CNN ───────────────────────────────────────────────────────────────────────
print("Carregando CNN...")
cnn_model       = km.load_model(os.path.join(MDLS, "CNN_modelV2.keras"))
X_cnn, _, uid   = CNNModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), window_size=80)
y_cnn           = inverse(cnn_model.predict(X_cnn[get_last_windows(X_cnn, uid)], verbose=0).flatten())

# ── LSTM ──────────────────────────────────────────────────────────────────────
print("Carregando LSTM...")
lstm_model      = km.load_model(os.path.join(MDLS, "LSTM_modelV2.keras"))
X_lstm, _, uid  = LSTMModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), window_size=40)
y_lstm          = inverse(lstm_model.predict(X_lstm[get_last_windows(X_lstm, uid)], verbose=0).flatten())

# ── TCN ───────────────────────────────────────────────────────────────────────
print("Carregando TCN...")
tcn_model       = km.load_model(os.path.join(MDLS, "TCN_model.keras"))
X_tcn, _, uid   = TCNModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), min_sequence_length=1)
y_tcn           = inverse(tcn_model.predict(X_tcn[get_last_windows(X_tcn, uid)], verbose=0).flatten())

# ── Transformer ───────────────────────────────────────────────────────────────
print("Carregando Transformer...")
tr_model        = km.load_model(os.path.join(MDLS, "Transformer_model.keras"))
X_tr, _, uid    = TransformerModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), min_sequence_length=1)
y_tr            = inverse(tr_model.predict(X_tr[get_last_windows(X_tr, uid)], verbose=0).flatten())

# ── métricas ──────────────────────────────────────────────────────────────────
modelos = {
    "CNN":         y_cnn,
    "CNN+LSTM":    y_lstm,
    "TCN":         y_tcn,
    "Transformer": y_tr,
}
cores = {
    "CNN":         "#1f77b4",   # azul
    "CNN+LSTM":    "#ff7f0e",   # laranja
    "TCN":         "#2ca02c",   # verde
    "Transformer": "#d62728",   # vermelho
}

print("\n=== Resultados ===")
for nome, pred in modelos.items():
    mae  = np.mean(np.abs(pred - Y_true))
    rmse = np.sqrt(np.mean((pred - Y_true) ** 2))
    print(f"{nome:12s} — MAE: {mae:.2f}  RMSE: {rmse:.2f}")

motores = np.arange(n_motors)

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 1 — Erro Absoluto por motor
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

largura = 0.2
offsets = [-1.5, -0.5, 0.5, 1.5]

for (nome, pred), offset in zip(modelos.items(), offsets):
    erro_abs = np.abs(pred - Y_true)
    ax.bar(
        motores + offset * largura,
        erro_abs,
        width=largura,
        color=cores[nome],
        alpha=0.85,
        label=f"{nome} (MAE={np.mean(erro_abs):.2f})"
    )

ax.set_xlabel("Motor (índice)", fontsize=12)
ax.set_ylabel("Erro Absoluto (ciclos)", fontsize=12)
ax.set_title("Comparativo — Erro Absoluto por Motor (4 Modelos)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(-1, n_motors)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

path1 = os.path.join(PLOTS, "Comparativo_ErroAbsoluto.png")
plt.savefig(path1, dpi=150)
plt.close()
print(f"\nGráfico 1 salvo em {path1}")

# ─────────────────────────────────────────────────────────────────────────────
# GRÁFICO 2 — Erro Normalizado por motor  (erro / RUL_real)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))

# evita divisão por zero em motores com RUL=0
denom = np.where(Y_true == 0, 1, Y_true)

for (nome, pred), offset in zip(modelos.items(), offsets):
    erro_norm = np.abs(pred - Y_true) / denom
    ax.bar(
        motores + offset * largura,
        erro_norm,
        width=largura,
        color=cores[nome],
        alpha=0.85,
        label=f"{nome} (Média={np.mean(erro_norm):.3f})"
    )

ax.set_xlabel("Motor (índice)", fontsize=12)
ax.set_ylabel("Erro Normalizado (|erro| / RUL_real)", fontsize=12)
ax.set_title("Comparativo — Erro Normalizado por Motor (4 Modelos)", fontsize=14, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.set_xlim(-1, n_motors)
ax.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

path2 = os.path.join(PLOTS, "Comparativo_ErroNormalizado.png")
plt.savefig(path2, dpi=150)
plt.close()
print(f"Gráfico 2 salvo em {path2}")