"""Sliding-window construction and LSTM training on FD001.

Usage: python -m src.train_lstm --data-dir data
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_prep import FEATURES, add_rul, engine_split, fit_scaler, load_fd001, scale
from src.scoring import evaluate

WINDOW = 30


def make_windows(df: pd.DataFrame, window: int = WINDOW, features=FEATURES):
    """All sliding windows of `window` cycles per engine, labelled with the RUL at the window end."""
    X, y = [], []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values, rul = g[features].to_numpy(), g["RUL"].to_numpy()
        for end in range(window, len(g) + 1):
            X.append(values[end - window:end])
            y.append(rul[end - 1])
    return np.asarray(X, dtype="float32"), np.asarray(y, dtype="float32")


def last_windows(df: pd.DataFrame, window: int = WINDOW, features=FEATURES):
    """Last window of each engine; engines shorter than `window` are padded with their first row."""
    X = []
    for _, g in df.sort_values(["unit", "cycle"]).groupby("unit"):
        values = g[features].to_numpy()
        if len(values) < window:
            pad = np.repeat(values[:1], window - len(values), axis=0)
            values = np.vstack([pad, values])
        X.append(values[-window:])
    return np.asarray(X, dtype="float32")


def build_model(window: int = WINDOW, n_features: int = len(FEATURES)):
    from tensorflow import keras

    model = keras.Sequential([
        keras.layers.Input(shape=(window, n_features)),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def run(data_dir="data", out_dir="figures", epochs=60, batch_size=256, seed=42):
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    train, test, rul_true = load_fd001(data_dir)
    train = add_rul(train)

    tr, val = engine_split(train, val_frac=0.2, seed=seed)
    scaler = fit_scaler(tr)
    X_tr, y_tr = make_windows(scale(tr, scaler))
    X_val, y_val = make_windows(scale(val, scaler))
    X_test = last_windows(scale(test, scaler))

    model = build_model()
    t0 = time.perf_counter()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
    )
    train_time = time.perf_counter() - t0

    pred = model.predict(X_test, verbose=0).ravel().clip(min=0)
    metrics = evaluate(rul_true, pred)
    print(f"LSTM  RMSE={metrics['rmse']:.2f}  NASA={metrics['nasa_score']:.1f}  "
          f"RMSE(<50)={metrics['rmse_critical']:.2f}  time={train_time:.2f}s")

    Path(out_dir).mkdir(exist_ok=True)
    pd.DataFrame([{"model": "LSTM", **metrics, "train_time_s": train_time}]).to_csv(
        Path(out_dir) / "results_lstm.csv", index=False)
    return model, history, pred, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="figures")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()
    run(args.data_dir, args.out_dir, epochs=args.epochs)
