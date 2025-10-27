import pandas as pd
import os

'''function para carregar o arquivo e gerar um data frame para utiizarmos 
na hora de entregar os dados para o código do machine learning. '''

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

if __name__ == '__main__':
    df_sensors = carregar_dados()
    print(f"\nShape do DataFrame final: {df_sensors.shape}")
