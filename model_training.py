import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def main():
    # Caminho base e arquivo de treino
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train_FD001.txt')

    # Ler o dataset (ignora múltiplos espaços)
    df = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python')

    # O dataset original tem 26 colunas úteis + 1 vazia no final → total de 27
    # Remove colunas totalmente vazias
    df = df.dropna(axis=1, how='all')

    print(f"Número de colunas lidas: {df.shape[1]}")

    # Corrigir número de colunas (caso tenha sobrado colunas extras)
    expected_cols = 26
    if df.shape[1] > expected_cols:
        df = df.iloc[:, :expected_cols]

    # Nomear colunas corretamente (2 primeiras + 3 op_settings + 21 sensores)
    col_names = (
        ['engine_id', 'cycle'] +
        [f'op_setting{i}' for i in range(1, 4)] +
        [f'sensor{i}' for i in range(1, 22)]
    )
    df.columns = col_names

    # Converter para tipo numérico (garante que nada quebre)
    df = df.apply(pd.to_numeric, errors='coerce')

    # Criar coluna RUL (Remaining Useful Life)
    df['RUL'] = df.groupby('engine_id')['cycle'].transform('max') - df['cycle']

    # Features = sensores, Target = RUL
    feature_cols = [c for c in df.columns if c.startswith('sensor')]
    X = df[feature_cols]
    y = df['RUL']

    # Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Criar e treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Fazer previsões
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'\nRMSE no conjunto de teste: {rmse:.2f}')

    # Exemplo de previsão
    print("\nExemplo de predição de RUL para 5 amostras:")
    print(pd.DataFrame({
        "Real": y_test.iloc[:5].values,
        "Previsto": np.round(y_pred[:5], 1)
    }))

if __name__ == '__main__':
    main()
