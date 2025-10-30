import pandas as pd
import os
import matplotlib.pyplot as plt

'''function para carregar o arquivo e gerar um data frame para utilizarmos 
na hora de entregar os dados para o código do machine learning. Além disso, usando
a biblioteca matplotlib.pyplot será gerado os gráficos para conferência das  informações coletando
as informações do sensor, o código identificador da máquina e o ciclo dela.'''

def carregar_dados():
    # Define o caminho do arquivo que utilizamos (mesma pasta do script)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train_FD001.txt')

    # Lê o arquivo — separando por espaços, sem cabeçalho
    df_sensors = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python')

    # Mostra as últimas colunas
    print("Últimas 3 colunas do arquivo original:")
    print(df_sensors.iloc[:, -3:].head())

    # Remove as 3 últimas colunas que são vazias no dataset original
    df_sensors = df_sensors.iloc[:, :-3]

    # Define nomes das colunas
    col_names = ['engine_id', 'cycle'] + [f'sensor{i}' for i in range(1, 22)]
    df_sensors.columns = col_names

    # Converte todos os valores para numéricos 
    df_sensors = df_sensors.apply(pd.to_numeric)

    # Mostra as primeiras linhas para conferência de que o df está correto
    print("\nPrévia dos dados formatados:")
    print(df_sensors.head())

    return df_sensors
def plot_sensors_vs_cycle(df_sensors, save_dir='plots'):
    """
    Gera um gráfico com subplots (um por sensor) mostrando todos os motores (engine_id)
    ao longo do ciclo (cycle). Salva uma imagem com todos os subplots e imagens individuais por sensor.
    """
    os.makedirs(save_dir, exist_ok=True)

    sensors = [f'sensor{i}' for i in range(1, 22)]

    # Figura com todos os sensores (7x3 = 21)
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 20), sharex=True)
    axes = axes.flatten()

    for i, sensor in enumerate(sensors):
        ax = axes[i]
        # plota cada engine como uma linha com transparência
        for engine_id, g in df_sensors.groupby('engine_id'):
            ax.plot(g['cycle'], g[sensor], alpha=0.25, linewidth=0.8)
        ax.set_title(sensor)
        ax.set_ylabel('Valor')
        if i >= 18:  # últimas 3 linhas mostram o xlabel
            ax.set_xlabel('cycle')
    # remove qualquer eixo extra (não usado)
    for j in range(len(sensors), len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Leitura dos Sensores x Cycle (todos os engines)', fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    combined_path = os.path.join(save_dir, 'sensors_all_engines.png')
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)

    # Salva imagens individuais por sensor (útil para inspeção)
    for sensor in sensors:
        fig_s = plt.figure(figsize=(8,4))
        for engine_id, g in df_sensors.groupby('engine_id'):
            plt.plot(g['cycle'], g[sensor], alpha=0.25, linewidth=0.8)
        plt.title(f'{sensor} vs cycle (todos os engines)')
        plt.xlabel('cycle')
        plt.ylabel('Valor')
        path = os.path.join(save_dir, f'{sensor}.png')
        fig_s.tight_layout()
        fig_s.savefig(path, dpi=150)
        plt.close(fig_s)

    print(f'Gráficos salvos em: {os.path.abspath(save_dir)}')

if __name__ == '__main__':
    df_sensors = carregar_dados()
    print(f"\nShape do DataFrame final: {df_sensors.shape}")


plot_sensors_vs_cycle(df_sensors)
