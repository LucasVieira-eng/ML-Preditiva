import os
import numpy as np
import keras
from keras import layers, models, callbacks, regularizers


def _get_scaler_paths():
    base_dir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(base_dir, "mu.npy"), os.path.join(base_dir, "sigma.npy")


def load_cmaps_data(file_path, window_size=30, rul_clip=200, min_sequence_length=1):
    data = np.loadtxt(file_path)

    unit_ids = data[:, 0].astype(int)
    cycles   = data[:, 1].astype(int)
    settings = data[:, 2:5]
    sensors  = data[:, 5:]

    useful_idx = [1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 16, 19, 20]
    sensors  = sensors[:, useful_idx]
    features = np.hstack([settings, sensors])

    mu_path, sigma_path = _get_scaler_paths()
    mu    = np.load(mu_path)
    sigma = np.load(sigma_path)
    features = (features - mu) / sigma

    X, Y, unit_ids_windowed = [], [], []
    for unit in np.unique(unit_ids):
        idx         = unit_ids == unit
        unit_feat   = features[idx]
        unit_cycles = cycles[idx]

        if len(unit_feat) < min_sequence_length:
            continue

        max_cycle = unit_cycles.max()
        rul = np.clip(max_cycle - unit_cycles, 0, rul_clip)
        rul = np.log(rul + 1)

        if len(unit_feat) < window_size:
            pad    = np.repeat(unit_feat[0:1], window_size - len(unit_feat), axis=0)
            window = np.vstack([pad, unit_feat])
            X.append(window)
            Y.append(rul[-1])
            unit_ids_windowed.append(unit)
        else:
            for i in range(len(unit_feat) - window_size + 1):
                X.append(unit_feat[i:i + window_size])
                Y.append(rul[i + window_size - 1])
                unit_ids_windowed.append(unit)

    return np.array(X), np.array(Y), np.array(unit_ids_windowed)


@keras.saving.register_keras_serializable(package="TCN")
class ResidualTCNBlock(layers.Layer):
    """
    Bloco TCN com convoluções causais dilatadas + conexão residual.
    A dilatação dobra a cada bloco: 1, 2, 4, 8...
    Isso expande o campo receptivo exponencialmente sem aumentar parâmetros.
    """
    def __init__(self, filters, kernel_size, dilation_rate,
                 dropout=0.1, l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout       = dropout
        self.l2            = l2

        # Duas convoluções causais dilatadas empilhadas
        self.conv1 = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2)
        )
        self.conv2 = layers.Conv1D(
            filters, kernel_size,
            padding="causal",
            dilation_rate=dilation_rate,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2)
        )
        self.bn1   = layers.BatchNormalization()
        self.bn2   = layers.BatchNormalization()
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

        # Projeção residual — ajusta canais se necessário
        self.downsample = layers.Conv1D(filters, 1, padding="same")

    def call(self, x, training=False):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.drop1(out, training=training)

        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.drop2(out, training=training)

        return layers.ReLU()(out + residual)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters":       self.filters,
            "kernel_size":   self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout":       self.dropout,
            "l2":            self.l2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_tcn_model(input_shape, filters=64, kernel_size=3,
                    num_blocks=4, dropout=0.1, l2=1e-4):
    """
    Arquitetura TCN:
      Input → [ResidualTCNBlock(dilation=1) → dilation=2 → 4 → 8]
      → GlobalAveragePooling → Dense(64) → Dropout → Dense(1)

    Campo receptivo total = 1 + (kernel_size-1) * 2 * (2^num_blocks - 1)
    Com kernel=3 e 4 blocos: campo receptivo = 1 + 2*2*15 = 61 ciclos
    """
    inputs = layers.Input(shape=input_shape)
    x = inputs

    for i in range(num_blocks):
        dilation = 2 ** i   # 1, 2, 4, 8
        x = ResidualTCNBlock(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout=dropout,
            l2=l2,
            name=f"tcn_block_{i}"
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    return models.Model(inputs, outputs)


def inverse_rul_transform(y_pred, y_pred_clip_max=6.0):
    y_pred_clipped = np.clip(y_pred, -np.inf, y_pred_clip_max)
    return np.clip(np.exp(y_pred_clipped) - 1, 0.0, None)


def train_tcn_model(model, X_train, Y_train, model_path,
                    epochs=60, batch_size=64, validation_split=0.2):
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])

    model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[
            callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5)
        ]
    )
    model.save(model_path)
    print(f"Modelo salvo em {model_path}")
    return model


if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(__file__), "../../data/train_FD001.txt")
    model_path = os.path.join(os.path.dirname(__file__), "../../models/TCN_model.keras")

    window_size = 30

    X_train, Y_train, _ = load_cmaps_data(train_path, window_size=window_size)
    print(f"Shape treino: {X_train.shape}")

    model = build_tcn_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        filters=64,
        kernel_size=3,
        num_blocks=4,
        dropout=0.1
    )
    model.summary()

    train_tcn_model(model, X_train, Y_train, model_path)