import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import os

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, 'train_FD001.txt')

    df = pd.read_csv(train_path, sep=r'\s+', header=None, engine='python')

    print("Mostrando as 3 últimas colunas para entender:")
    print(df.iloc[:, -3:].head(10))  # DEBUG

    df = df.iloc[:, :-3]
    print(f"Número de colunas após remover as 3 últimas colunas: {df.shape[1]}")

    # Renomear colunas corretamente
    col_names = ['engine_id', 'cycle'] + [f'sensor{i}' for i in range(1, 22)]
    df.columns = col_names

    df = df.apply(pd.to_numeric)

    # Criar RUL
    df['RUL'] = df.groupby('engine_id')['cycle'].transform(max) - df['cycle']

    feature_cols = [c for c in df.columns if c.startswith('sensor')]
    X = df[feature_cols]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE no conjunto de teste: {rmse:.2f}')

if __name__ == '__main__':
    main()
