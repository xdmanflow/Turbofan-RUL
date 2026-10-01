"""Train and evaluate the tabular baselines on FD001.

Usage: python -m src.train_baselines --data-dir data
"""
import argparse
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from src.data_prep import FEATURES, add_rul, fit_scaler, last_cycle, load_fd001, scale
from src.scoring import evaluate

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=42),
}


def run(data_dir="data", out_dir="figures"):
    train, test, rul_true = load_fd001(data_dir)
    train = add_rul(train)

    scaler = fit_scaler(train)
    train_s = scale(train, scaler)
    test_last = scale(last_cycle(test), scaler)

    rows, fitted = [], {}
    for name, make in MODELS.items():
        model = make()
        t0 = time.perf_counter()
        model.fit(train_s[FEATURES], train_s["RUL"])
        train_time = time.perf_counter() - t0

        pred = model.predict(test_last[FEATURES]).clip(min=0)
        metrics = evaluate(rul_true, pred)
        rows.append({"model": name, **metrics, "train_time_s": train_time})
        fitted[name] = model
        print(f"{name:<18} RMSE={metrics['rmse']:.2f}  NASA={metrics['nasa_score']:.1f}  "
              f"RMSE(<50)={metrics['rmse_critical']:.2f}  time={train_time:.2f}s")

    results = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True)
    results.to_csv(Path(out_dir) / "results_baselines.csv", index=False)

    importances = pd.Series(fitted["Random Forest"].feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nRandom Forest feature importance (top 5):")
    print(importances.head().round(3).to_string())
    return results, fitted, importances


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="figures")
    args = parser.parse_args()
    run(args.data_dir, args.out_dir)
