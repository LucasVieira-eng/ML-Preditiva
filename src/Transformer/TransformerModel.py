import os
import numpy as np
import keras
import tensorflow as tf
from keras import layers, models, callbacks, regularizers


def _get_scaler_paths():
    base_dir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(base_dir, "mu.npy"), os.path.join(base_dir, "sigma.npy")


def load_cmaps_data(file_path, window_size=30, rul_clip=200, min_sequence_length=40):
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


def positional_encoding(length, depth):
    positions = np.arange(length)[:, np.newaxis]
    dims      = np.arange(depth)[np.newaxis, :]
    angles    = positions / np.power(10000, (2 * (dims // 2)) / depth)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.cast(angles[np.newaxis, :, :], dtype=tf.float32)


@keras.saving.register_keras_serializable(package="Transformer")
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, l2=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim    = ff_dim
        self.dropout   = dropout
        self.l2        = l2

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            kernel_regularizer=regularizers.l2(l2)
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation="relu",
                         kernel_regularizer=regularizers.l2(l2)),
            layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2)),
        ])
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.last_attn_weights = None

    def call(self, x, training=False):
        attn_out, attn_weights = self.attention(
            x, x, return_attention_scores=True, training=training
        )
        self.last_attn_weights = attn_weights
        x = self.norm1(x + self.drop1(attn_out, training=training))
        x = self.norm2(x + self.drop2(self.ffn(x), training=training))
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim":    self.ff_dim,
            "dropout":   self.dropout,
            "l2":        self.l2,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_transformer_model(input_shape, embed_dim=64, num_heads=4,
                             ff_dim=128, num_blocks=2, dropout=0.1, l2=1e-4):
    inputs = layers.Input(shape=input_shape)

    x = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2))(inputs)
    x = x + positional_encoding(input_shape[0], embed_dim)

    transformer_blocks = []
    for _ in range(num_blocks):
        block = TransformerBlock(embed_dim, num_heads, ff_dim, dropout, l2)
        transformer_blocks.append(block)
        x = block(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(l2))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    return model, transformer_blocks


def inverse_rul_transform(y_pred, y_pred_clip_max=6.0):
    y_pred_clipped = np.clip(y_pred, -np.inf, y_pred_clip_max)
    return np.clip(np.exp(y_pred_clipped) - 1, 0.0, None)


def train_transformer_model(model, X_train, Y_train, model_path,
                             epochs=60, batch_size=64,
                             validation_split=0.2):
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
    model_path = os.path.join(os.path.dirname(__file__), "../../models/Transformer_model.keras")

    window_size = 30

    X_train, Y_train, _ = load_cmaps_data(train_path, window_size=window_size)
    print(f"Shape treino: {X_train.shape}")

    model, _ = build_transformer_model(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        embed_dim=64,
        num_heads=4,
        ff_dim=128,
        num_blocks=2,
        dropout=0.1
    )
    model.summary()

    train_transformer_model(model, X_train, Y_train, model_path)