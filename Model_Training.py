import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def main():
    # 1. Carregar dados
    train_path = 'data/train_FD001.txt'
    df = pd.read_csv(train_path, sep='\s+', header=None)
    df.columns = ['engine_id', 'cycle'] + [f'sensor{i}' for i in range(1, df.shape[1]-1)]

    # 2. Criar coluna RUL
    df['RUL'] = df.groupby('engine_id')['cycle'].transform(max) - df['cycle']

    # 3. Separar features e alvo
    feature_cols = [col for col in df.columns if col.startswith('sensor')]
    X = df[feature_cols]
    y = df['RUL']

    # 4. Dividir treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # 5. Treinar modelo
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 6. Avaliar modelo
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE no conjunto de teste: {rmse:.2f}')

if __name__ == "__main__":
    main()
