"""
DashboardComparativo.py
Dashboard visual completo dos 4 modelos de Manutenção Preditiva.

Coloque em: src/Comparativo/DashboardComparativo.py
Rode com:   python DashboardComparativo.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
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

import CNNModel
import LSTMModel
import TCNModel
from TCNModel import ResidualTCNBlock          # noqa — registra decorator
import TransformerModel
from TransformerModel import TransformerBlock  # noqa — registra decorator

# ── helpers ───────────────────────────────────────────────────────────────────
def get_last_windows(X, unit_ids):
    return np.array([np.where(unit_ids == u)[0][-1] for u in np.unique(unit_ids)])

def inverse(y, clip=6.0):
    return np.clip(np.exp(np.clip(y, -np.inf, clip)) - 1, 0, None)

# ── predições ─────────────────────────────────────────────────────────────────
Y_true = np.clip(np.loadtxt(os.path.join(DATA, "RUL_FD001.txt")), 0, 200)
n = len(Y_true)

print("Carregando CNN...")
cnn_model     = km.load_model(os.path.join(MDLS, "CNN_modelV2.keras"))
X_cnn, _, uid = CNNModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), window_size=80)
y_cnn         = inverse(cnn_model.predict(X_cnn[get_last_windows(X_cnn, uid)], verbose=0).flatten())

print("Carregando LSTM...")
lstm_model     = km.load_model(os.path.join(MDLS, "LSTM_modelV2.keras"))
X_ls, _, uid   = LSTMModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), window_size=40)
y_lstm         = inverse(lstm_model.predict(X_ls[get_last_windows(X_ls, uid)], verbose=0).flatten())

print("Carregando TCN...")
tcn_model      = km.load_model(os.path.join(MDLS, "TCN_model.keras"))
X_tcn, _, uid  = TCNModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), min_sequence_length=1)
y_tcn          = inverse(tcn_model.predict(X_tcn[get_last_windows(X_tcn, uid)], verbose=0).flatten())

print("Carregando Transformer...")
tr_model       = km.load_model(os.path.join(MDLS, "Transformer_model.keras"))
X_tr, _, uid   = TransformerModel.load_cmaps_data(os.path.join(DATA, "test_FD001.txt"), min_sequence_length=1)
y_tr           = inverse(tr_model.predict(X_tr[get_last_windows(X_tr, uid)], verbose=0).flatten())

modelos = {"CNN": y_cnn, "CNN+LSTM": y_lstm, "TCN": y_tcn, "Transformer": y_tr}
cores   = {"CNN": "#1f77b4", "CNN+LSTM": "#ff7f0e", "TCN": "#2ca02c", "Transformer": "#d62728"}
motores = np.arange(n)

print("\n=== Métricas ===")
for nome, pred in modelos.items():
    mae  = np.mean(np.abs(pred - Y_true))
    rmse = np.sqrt(np.mean((pred - Y_true) ** 2))
    print(f"{nome:12s} — MAE: {mae:.2f}  RMSE: {rmse:.2f}")

# ── layout ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor("#0f1117")
gs  = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.32,
                         top=0.93, bottom=0.05, left=0.07, right=0.97)

TITLE_COLOR = "#ffffff"
LABEL_COLOR = "#cccccc"
GRID_COLOR  = "#2a2a3a"
BG_AX       = "#1a1a2e"

def style_ax(ax, title, xlabel, ylabel):
    ax.set_facecolor(BG_AX)
    ax.tick_params(colors=LABEL_COLOR, labelsize=9)
    ax.set_title(title, color=TITLE_COLOR, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=LABEL_COLOR, fontsize=9)
    ax.set_ylabel(ylabel, color=LABEL_COLOR, fontsize=9)
    ax.grid(axis="y", color=GRID_COLOR, linestyle="--", linewidth=0.6)
    ax.spines[["top", "right", "left", "bottom"]].set_color("#333355")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(LABEL_COLOR)

fig.text(0.5, 0.965, "Dashboard Comparativo — Manutenção Preditiva (C-MAPSS FD001)",
         ha="center", va="center", fontsize=16, fontweight="bold", color=TITLE_COLOR)
fig.text(0.5, 0.948, "NASA Turbofan Jet Engine · 4 Modelos · 100 Motores de Teste",
         ha="center", va="center", fontsize=11, color="#8888aa")

larg = 0.2
offs = [-1.5, -0.5, 0.5, 1.5]

# ── ① Erro Absoluto por motor ─────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
style_ax(ax1, "① Erro Absoluto por Motor", "Motor (índice)", "Erro Absoluto (ciclos)")
for (nome, pred), off in zip(modelos.items(), offs):
    ea = np.abs(pred - Y_true)
    ax1.bar(motores + off * larg, ea, width=larg, color=cores[nome],
            alpha=0.85, label=f"{nome}  MAE={np.mean(ea):.2f}")
ax1.set_xlim(-1, n)
ax1.axhline(30, color="#ff4444", linewidth=1, linestyle=":", alpha=0.7)
ax1.text(n - 1, 31, "limite 30 ciclos", color="#ff4444", fontsize=8, ha="right")
ax1.legend(fontsize=9, loc="upper left", framealpha=0.2,
           labelcolor=LABEL_COLOR, facecolor=BG_AX, edgecolor="#333355")

# ── ② Viés — superestimação vs subestimação ───────────────────────────────────
ax2 = fig.add_subplot(gs[1, :])
style_ax(ax2, "② Viés dos Modelos — Superestimação (+) vs Subestimação (−)",
         "Motor (índice)", "Erro com Sinal (ciclos)")
ax2.axhline(0, color="#ffffff", linewidth=1.2, alpha=0.5)
ax2.set_xlim(-1, n)
for (nome, pred), off in zip(modelos.items(), offs):
    erro = pred - Y_true
    ax2.bar(motores + off * larg, np.where(erro >= 0, erro, 0),
            width=larg, color=cores[nome], alpha=0.85)
    ax2.bar(motores + off * larg, np.where(erro < 0, erro, 0),
            width=larg, color=cores[nome], alpha=0.85)
handles = [Patch(color=c, label=nm) for nm, c in cores.items()]
ax2.legend(handles=handles, fontsize=9, loc="upper left", framealpha=0.2,
           labelcolor=LABEL_COLOR, facecolor=BG_AX, edgecolor="#333355")
bias_txt = "  ".join([f"{nm}: bias={np.mean(p - Y_true):+.1f}" for nm, p in modelos.items()])
ax2.text(0.5, -0.13, bias_txt, transform=ax2.transAxes,
         ha="center", fontsize=9, color="#aaaacc")

# ── ③ Boxplot ─────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2, 0])
style_ax(ax3, "③ Distribuição do Erro Absoluto (Boxplot)", "Modelo", "Erro Absoluto (ciclos)")
data_box = [np.abs(pred - Y_true) for pred in modelos.values()]
bp = ax3.boxplot(data_box, patch_artist=True, widths=0.5,
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(color=LABEL_COLOR),
                  capprops=dict(color=LABEL_COLOR),
                  flierprops=dict(marker="o", markerfacecolor="#ff4444",
                                  markersize=4, alpha=0.6))
for patch, (nome, _) in zip(bp["boxes"], modelos.items()):
    patch.set_facecolor(cores[nome])
    patch.set_alpha(0.8)
ax3.set_xticks(range(1, 5))
ax3.set_xticklabels(modelos.keys(), color=LABEL_COLOR, fontsize=9)
for i, data in enumerate(data_box, 1):
    ax3.text(i, np.max(data) + 1, f"σ={np.std(data):.1f}",
             ha="center", fontsize=8, color=LABEL_COLOR)

# ── ④ Violino ─────────────────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 1])
style_ax(ax4, "④ Distribuição do Erro com Sinal (Violino)", "Modelo", "Erro (ciclos)")
ax4.axhline(0, color="#ffffff", linewidth=1, alpha=0.4, linestyle="--")
data_vio = [pred - Y_true for pred in modelos.values()]
vp = ax4.violinplot(data_vio, positions=range(1, 5),
                     showmedians=True, showextrema=True)
vp["cmedians"].set_color("white")
vp["cmedians"].set_linewidth(2)
for body, (nome, _) in zip(vp["bodies"], modelos.items()):
    body.set_facecolor(cores[nome])
    body.set_alpha(0.75)
for part in ["cbars", "cmins", "cmaxes"]:
    vp[part].set_color(LABEL_COLOR)
    vp[part].set_linewidth(1)
ax4.set_xticks(range(1, 5))
ax4.set_xticklabels(modelos.keys(), color=LABEL_COLOR, fontsize=9)

# ── ⑤ Tabela de métricas ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[3, :])
ax5.set_facecolor(BG_AX)
ax5.axis("off")
ax5.set_title("⑤ Tabela de Métricas Comparativas", color=TITLE_COLOR,
               fontsize=11, fontweight="bold", pad=8)

col_labels = ["Modelo", "MAE (ciclos)", "RMSE (ciclos)", "Bias Médio",
              "Desvio Padrão do Erro", "% Motores erro > 30 ciclos", "Destaque"]
best_mae = min(np.mean(np.abs(p - Y_true)) for p in modelos.values())
rows = []
for nome, pred in modelos.items():
    ea   = np.abs(pred - Y_true)
    erro = pred - Y_true
    mae  = np.mean(ea)
    rmse = np.sqrt(np.mean((pred - Y_true) ** 2))
    bias = np.mean(erro)
    std  = np.std(erro)
    pct  = np.mean(ea > 30) * 100
    destaque = "✓ Melhor MAE" if mae == best_mae else ""
    rows.append([nome, f"{mae:.2f}", f"{rmse:.2f}", f"{bias:+.2f}",
                 f"{std:.2f}", f"{pct:.1f}%", destaque])

table = ax5.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.2)

for (r, c), cell in table.get_celld().items():
    cell.set_edgecolor("#333355")
    if r == 0:
        cell.set_facecolor("#16213e")
        cell.set_text_props(color=TITLE_COLOR, fontweight="bold")
    else:
        nome_linha = rows[r - 1][0]
        cell.set_facecolor(cores[nome_linha] + "33")
        cell.set_text_props(color=LABEL_COLOR)
        if rows[r - 1][6] == "✓ Melhor MAE" and c == 6:
            cell.set_facecolor("#2ca02c55")
            cell.set_text_props(color="#00ff88", fontweight="bold")

# ── salva ─────────────────────────────────────────────────────────────────────
out_path = os.path.join(PLOTS, "Dashboard_Comparativo.png")
plt.savefig(out_path, dpi=150, facecolor=fig.get_facecolor())
plt.close()
print(f"\nDashboard salvo em {out_path}")