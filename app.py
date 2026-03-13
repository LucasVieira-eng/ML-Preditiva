import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard - Machine Learning", layout="wide")

st.title("📊 Dashboard do Projeto de Machine Learning - Indústria 4.0")

st.write("""
Projeto desenvolvido para o Projeto Integrador de Indústria 4.0.

Autor:
- Kauã Luiz Ramos

Objetivo:
Explorar e analisar os dados utilizados no treinamento dos modelos
de Machine Learning (CNN e MLP) aplicados à previsão de falhas
em motores aeronáuticos.
""")

@st.cache_data
def load_data():
    train = pd.read_csv("data/train_FD001.txt", sep="\s+", header=None)
    test = pd.read_csv("data/test_FD001.txt", sep="\s+", header=None)
    rul = pd.read_csv("data/RUL_FD001.txt", header=None)
    return train, test, rul

train, test, rul = load_data()

cols = ["engine_id","cycle","setting1","setting2","setting3"]
sensor_cols = [f"sensor{i}" for i in range(1,22)]

train.columns = cols + sensor_cols
test.columns = cols + sensor_cols
rul.columns = ["RUL"]

st.header("📂 Conjunto de Dados")

st.write("""
O projeto utiliza dados de sensores de motores aeronáuticos.
Cada linha representa um ciclo de funcionamento de um motor
e registra medições de diversos sensores utilizados para análise.
""")

st.subheader("Amostra do Dataset de Treinamento")

st.dataframe(train.head())

st.write("Número de motores analisados:", train["engine_id"].nunique())
st.write("Total de registros no dataset:", train.shape[0])

st.header("⚙️ Ciclos de Funcionamento dos Motores")

cycle_per_engine = train.groupby("engine_id")["cycle"].max()

fig, ax = plt.subplots()

ax.hist(cycle_per_engine, bins=20)
ax.set_xlabel("Ciclos até falha")
ax.set_ylabel("Quantidade de motores")

st.pyplot(fig)

st.write("""
Este gráfico mostra quantos ciclos cada motor operou até ocorrer
a falha no conjunto de treinamento.
""")

st.header("📈 Distribuição dos Sensores")

sensor = st.selectbox(
"Selecione um sensor para visualizar",
sensor_cols
)

fig2, ax2 = plt.subplots()

ax2.hist(train[sensor], bins=30)
ax2.set_xlabel(sensor)
ax2.set_ylabel("Frequência")

st.pyplot(fig2)

st.write("""
Os sensores registram diferentes condições de operação do motor.
Essas informações são utilizadas pelos modelos de Machine Learning
para identificar padrões relacionados à falha dos equipamentos.
""")

st.header("🔧 Remaining Useful Life (RUL)")

st.write("""
O RUL (Remaining Useful Life) representa o número de ciclos restantes
até que o motor apresente falha.
""")

fig3, ax3 = plt.subplots()

ax3.hist(rul["RUL"], bins=30)
ax3.set_xlabel("Ciclos restantes")
ax3.set_ylabel("Quantidade")

st.pyplot(fig3)

st.header("📌 Conclusão")

st.write("""
A análise exploratória dos dados permite compreender o comportamento
dos sensores e o tempo de operação dos motores.

Essas informações são utilizadas no treinamento de modelos de Machine Learning,
como redes neurais CNN e MLP, capazes de prever o tempo restante de vida útil
dos motores com base nos dados coletados pelos sensores.
""")